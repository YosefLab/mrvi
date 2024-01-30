import logging
import os
import warnings
from collections.abc import Sequence
from copy import deepcopy
from functools import partial
from typing import Literal, Optional, Union

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro.distributions as dist
import pandas as pd
import seaborn as sns
import xarray as xr
from anndata import AnnData
from scvi import REGISTRY_KEYS
from scvi.data import AnnDataManager
from scvi.data.fields import (
    CategoricalObsField,
    LayerField,
    NumericalJointObsField,
    NumericalObsField,
)
from scvi.model.base import BaseModelClass, JaxTrainingMixin
from sklearn.mixture import GaussianMixture
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

from ._constants import MRVI_REGISTRY_KEYS
from ._module import MrVAE
from ._tree_utils import (
    compute_dendrogram_from_distance_matrix,
    convert_pandas_to_colors,
)
from ._types import MrVIReduction
from ._utils import (
    _parse_local_statistics_requirements,
    compute_statistic,
    permutation_test,
    rowwise_max_excluding_diagonal,
)

logger = logging.getLogger(__name__)

DEFAULT_TRAIN_KWARGS = {
    "max_epochs": 100,
    "early_stopping": True,
    "early_stopping_patience": 15,
    "check_val_every_n_epoch": 1,
    "batch_size": 256,
    "train_size": 0.9,
    "plan_kwargs": {
        "lr": 2e-3,
        "n_epochs_kl_warmup": 20,
        "max_norm": 40,
        "eps": 1e-8,
        "weight_decay": 1e-8,
    },
}


class MrVI(JaxTrainingMixin, BaseModelClass):
    """
    Multi-resolution Variational Inference (MrVI).

    Parameters
    ----------
    adata
        AnnData object that has been registered via ``setup_anndata``.
    n_latent
        Dimensionality of the latent space.
    n_latent_donor
        Dimensionality of the latent space for sample embeddings.
    encoder_n_hidden
        Number of nodes per hidden layer in the encoder.
    px_kwargs
        Keyword args for :class:`~mrvi.DecoderZX`.
    qz_kwargs
        Keyword args for :class:`~mrvi.EncoderUZ`.
    qu_kwargs
        Keyword args for :class:`~mrvi.EncoderXU`.
    """

    def __init__(
        self,
        adata,
        **model_kwargs,
    ):
        super().__init__(adata)

        n_sample = self.summary_stats.n_sample
        n_batch = self.summary_stats.n_batch
        n_labels = self.summary_stats.n_labels
        n_continuous_cov = self.summary_stats.get("n_extra_continuous_covs", 0)

        obs_df = adata.obs.copy()
        obs_df = obs_df.loc[~obs_df._scvi_sample.duplicated("first")]
        self.donor_info = obs_df.set_index("_scvi_sample").sort_index()
        self.sample_key = self.adata_manager.get_state_registry("sample").original_key
        self.sample_order = self.adata_manager.get_state_registry(
            MRVI_REGISTRY_KEYS.SAMPLE_KEY
        ).categorical_mapping

        self.n_obs_per_sample = jnp.array(
            adata.obs._scvi_sample.value_counts().sort_index().values
        )

        self.data_splitter = None
        self.can_compute_normalized_dists = (
            model_kwargs.get("qz_nn_flavor", "linear") == "linear"
        )
        self.module = MrVAE(
            n_input=self.summary_stats.n_vars,
            n_sample=n_sample,
            n_batch=n_batch,
            n_labels=n_labels,
            n_continuous_cov=n_continuous_cov,
            n_obs_per_sample=self.n_obs_per_sample,
            **model_kwargs,
        )
        self.can_compute_normalized_dists = (
            model_kwargs.get("qz_nn_flavor", "linear") == "linear"
        ) and (
            (model_kwargs.get("n_latent_u", None) is None)
            or (model_kwargs.get("n_latent", 10) == model_kwargs.get("n_latent_u", None))
        )
        self.init_params_ = self._get_init_params(locals())

    def to_device(self, device):
        # TODO(jhong): remove this once we have a better way to handle device.
        pass

    def _generate_stacked_rngs(
        self, n_sets: Union[int, tuple]
    ) -> dict[str, jax.random.PRNGKey]:
        return_1d = isinstance(n_sets, int)
        if return_1d:
            n_sets_1d = n_sets
        else:
            n_sets_1d = np.prod(n_sets)
        rngs_list = [self.module.rngs for _ in range(n_sets_1d)]
        # Combine list of RNG dicts into a single list. This is necessary for vmap/map.
        rngs = {
            required_rng: jnp.concatenate(
                [rngs_dict[required_rng][None] for rngs_dict in rngs_list], axis=0
            )
            for required_rng in self.module.required_rngs
        }
        if not return_1d:
            # Reshaping the random keys to the desired shape in
            # the case of multiple sets.
            rngs = {
                key: random_key.reshape(n_sets + random_key.shape[1:])
                for (key, random_key) in rngs.items()
            }
        return rngs

    @classmethod
    def setup_anndata(
        cls,
        adata: AnnData,
        layer: Optional[str] = None,
        sample_key: Optional[str] = None,
        batch_key: Optional[str] = None,
        labels_key: Optional[str] = None,
        continuous_covariate_keys: Optional[list[str]] = None,
        **kwargs,
    ):
        setup_method_args = cls._get_setup_method_args(**locals())
        # Add index for batched computation of local statistics.
        adata.obs["_indices"] = np.arange(adata.n_obs).astype(int)
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
            CategoricalObsField(MRVI_REGISTRY_KEYS.SAMPLE_KEY, sample_key),
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            CategoricalObsField(REGISTRY_KEYS.LABELS_KEY, labels_key),
            NumericalJointObsField(
                REGISTRY_KEYS.CONT_COVS_KEY, continuous_covariate_keys
            ),
            NumericalObsField(REGISTRY_KEYS.INDICES_KEY, "_indices"),
        ]
        if labels_key is None:
            # Hack to load old models pre labels field.
            # TODO: remove. not necessary for official release
            sr = kwargs.get("source_registry", None)
            if sr is not None:
                sr["field_registries"][REGISTRY_KEYS.LABELS_KEY] = {
                    "state_registry": {"categorical_mapping": np.array([0])}
                }
        adata_manager = AnnDataManager(
            fields=anndata_fields, setup_method_args=setup_method_args
        )
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

    def train(
        self,
        max_epochs: Optional[int] = None,
        accelerator: Optional[str] = "auto",
        devices: int | list[int] | str = "auto",
        train_size: float = 0.9,
        validation_size: Optional[float] = None,
        batch_size: int = 128,
        early_stopping: bool = False,
        plan_kwargs: Optional[dict] = None,
        **trainer_kwargs,
    ):
        train_kwargs = dict(
            max_epochs=max_epochs,
            accelerator=accelerator,
            devices=devices,
            train_size=train_size,
            validation_size=validation_size,
            batch_size=batch_size,
            early_stopping=early_stopping,
            **trainer_kwargs,
        )
        train_kwargs = dict(deepcopy(DEFAULT_TRAIN_KWARGS), **train_kwargs)
        plan_kwargs = plan_kwargs or {}
        train_kwargs["plan_kwargs"] = dict(
            deepcopy(DEFAULT_TRAIN_KWARGS["plan_kwargs"]), **plan_kwargs
        )
        super().train(**train_kwargs)

    def get_latent_representation(
        self,
        adata: Optional[AnnData] = None,
        indices=None,
        batch_size: Optional[int] = None,
        use_mean: bool = True,
        give_z: bool = False,
    ) -> np.ndarray:
        """
        Computes the latent representation of the data.

        Parameters
        ----------
        adata
            AnnData object to use. Defaults to the AnnData object used to initialize the model.
        indices
            Indices of cells to use.
        batch_size
            Batch size to use for computing the latent representation.
        use_mean
            Whether to use the mean of the distribution as the latent representation.
        give_z
            Whether to return the z latent representation or the u latent representation.

        Returns
        -------
        The latent representation of the data.
        """
        self._check_if_trained(warn=False)
        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size, iter_ndarray=True
        )

        us = []
        zs = []
        jit_inference_fn = self.module.get_jit_inference_fn(
            inference_kwargs={"use_mean": use_mean}
        )
        for array_dict in tqdm(scdl):
            outputs = jit_inference_fn(self.module.rngs, array_dict)

            us.append(outputs["u"])
            zs.append(outputs["z"])

        u = np.array(jax.device_get(jnp.concatenate(us, axis=0)))
        z = np.array(jax.device_get(jnp.concatenate(zs, axis=0)))
        return z if give_z else u

    def compute_local_statistics(
        self,
        reductions: list[MrVIReduction],
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        batch_size: Optional[int] = None,
        use_vmap: bool = True,
        norm: str = "l2",
        mc_samples: int = 10,
    ) -> xr.Dataset:
        """
        Compute local statistics from counterfactual sample representations.

        Local statistics are reductions over either the local counterfactual latent representations
        or the resulting local sample distance matrices. For a large number of cells and/or samples,
        this method can avoid scalability issues by grouping over cell-level covariates.

        Parameters
        ----------
        reductions
            List of reductions to compute over local counterfactual sample representations.
        adata
            AnnData object to use.
        indices
            Indices of cells to use.
        batch_size
            Batch size to use for computing the local statistics.
        use_vmap
            Whether to use vmap to compute the local statistics.
        norm
            Norm to use for computing the distances.
        mc_samples
            Number of Monte Carlo samples to use for computing the local statistics. Only applies if using
            sampled representations.
        """
        if not reductions or len(reductions) == 0:
            raise ValueError("At least one reduction must be provided.")

        adata = self.adata if adata is None else adata
        self._check_if_trained(warn=False)
        # Hack to ensure new AnnDatas have indices.
        adata.obs["_indices"] = np.arange(adata.n_obs).astype(int)

        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size, iter_ndarray=True
        )
        n_sample = self.summary_stats.n_sample

        reqs = _parse_local_statistics_requirements(reductions)

        vars_in = {"params": self.module.params, **self.module.state}

        @partial(jax.jit, static_argnames=["use_mean", "mc_samples"])
        def mapped_inference_fn(
            stacked_rngs,
            x,
            sample_index,
            cf_sample,
            use_mean,
            mc_samples=None,
        ):
            # TODO: use `self.module.get_jit_inference_fn` when it supports traced values.
            def inference_fn(
                rngs,
                cf_sample,
            ):
                return self.module.apply(
                    vars_in,
                    rngs=rngs,
                    method=self.module.inference,
                    x=x,
                    sample_index=sample_index,
                    cf_sample=cf_sample,
                    use_mean=use_mean,
                    mc_samples=mc_samples,
                )["z"]

            if use_vmap:
                return jax.vmap(inference_fn, in_axes=(0, 0), out_axes=-2)(
                    stacked_rngs,
                    cf_sample,
                )
            else:

                def per_sample_inference_fn(pair):
                    rngs, cf_sample = pair
                    return inference_fn(rngs, cf_sample)

                return jax.lax.transpose(
                    jax.lax.map(per_sample_inference_fn, (stacked_rngs, cf_sample)),
                    (1, 0, 2),
                )

        ungrouped_data_arrs = {}
        grouped_data_arrs = {}
        for ur in reqs.ungrouped_reductions:
            ungrouped_data_arrs[ur.name] = []
        for gr in reqs.grouped_reductions:
            grouped_data_arrs[
                gr.name
            ] = {}  # Will map group category to running group sum.

        for array_dict in tqdm(scdl):
            indices = array_dict[REGISTRY_KEYS.INDICES_KEY].astype(int).flatten()
            n_cells = array_dict[REGISTRY_KEYS.X_KEY].shape[0]
            cf_sample = np.broadcast_to(
                np.arange(n_sample)[:, None, None], (n_sample, n_cells, 1)
            )
            inf_inputs = self.module._get_inference_input(
                array_dict,
            )
            stacked_rngs = self._generate_stacked_rngs(cf_sample.shape[0])

            # Compute necessary inputs.
            if reqs.needs_mean_representations:
                mean_zs_ = mapped_inference_fn(
                    stacked_rngs=stacked_rngs,  # OK to use stacked rngs here since there is no stochasticity for mean rep.
                    x=jnp.array(inf_inputs["x"]),
                    sample_index=jnp.array(inf_inputs["sample_index"]),
                    cf_sample=jnp.array(cf_sample),
                    use_mean=True,
                )
                mean_zs = xr.DataArray(
                    mean_zs_,
                    dims=["cell_name", "sample", "latent_dim"],
                    coords={
                        "cell_name": self.adata.obs_names[indices],
                        "sample": self.sample_order,
                    },
                    name="sample_representations",
                )
            if reqs.needs_sampled_representations:
                sampled_zs_ = mapped_inference_fn(
                    stacked_rngs=stacked_rngs,
                    x=jnp.array(inf_inputs["x"]),
                    sample_index=jnp.array(inf_inputs["sample_index"]),
                    cf_sample=jnp.array(cf_sample),
                    use_mean=False,
                    mc_samples=mc_samples,
                )  # (n_mc_samples, n_cells, n_samples, n_latent)
                sampled_zs_ = sampled_zs_.transpose((1, 0, 2, 3))
                sampled_zs = xr.DataArray(
                    sampled_zs_,
                    dims=["cell_name", "mc_sample", "sample", "latent_dim"],
                    coords={
                        "cell_name": self.adata.obs_names[indices],
                        "sample": self.sample_order,
                    },
                    name="sample_representations",
                )

            if reqs.needs_mean_distances:
                mean_dists = self._compute_distances_from_representations(
                    mean_zs_, indices, norm=norm
                )

            if reqs.needs_sampled_distances or reqs.needs_normalized_distances:
                sampled_dists = self._compute_distances_from_representations(
                    sampled_zs_, indices, norm=norm
                )

                if reqs.needs_normalized_distances:
                    if norm != "l2":
                        raise ValueError(
                            f"Norm must be 'l2' when using normalized distances. Got {norm}."
                        )
                    (
                        normalization_means,
                        normalization_vars,
                    ) = self._compute_local_baseline_dists(
                        array_dict, mc_samples=mc_samples
                    )  # both are shape (n_cells,)
                    normalization_means = normalization_means.reshape(-1, 1, 1, 1)
                    normalization_vars = normalization_vars.reshape(-1, 1, 1, 1)
                    normalized_dists = (
                        np.clip(sampled_dists - normalization_means, a_min=0, a_max=None)
                        / (normalization_vars**0.5)
                    ).mean(dim="mc_sample")  # (n_cells, n_samples, n_samples)

            # Compute each reduction
            for r in reductions:
                if r.input == "mean_representations":
                    inputs = mean_zs
                elif r.input == "sampled_representations":
                    inputs = sampled_zs
                elif r.input == "mean_distances":
                    inputs = mean_dists
                elif r.input == "sampled_distances":
                    inputs = sampled_dists
                elif r.input == "normalized_distances":
                    inputs = normalized_dists
                else:
                    raise ValueError(f"Unknown reduction input: {r.input}")

                outputs = r.fn(inputs)

                if r.group_by is not None:
                    group_by = self.adata.obs[r.group_by][indices]
                    group_by_cats = group_by.unique()
                    for cat in group_by_cats:
                        cat_summed_outputs = outputs.sel(
                            cell_name=self.adata.obs_names[indices][
                                group_by == cat
                            ].values
                        ).sum(dim="cell_name")
                        cat_summed_outputs = cat_summed_outputs.assign_coords(
                            {f"{r.group_by}_name": cat}
                        )
                        if cat not in grouped_data_arrs[r.name]:
                            grouped_data_arrs[r.name][cat] = cat_summed_outputs
                        else:
                            grouped_data_arrs[r.name][cat] += cat_summed_outputs
                else:
                    ungrouped_data_arrs[r.name].append(outputs)

        # Combine all outputs.
        final_data_arrs = {}
        for ur_name, ur_outputs in ungrouped_data_arrs.items():
            final_data_arrs[ur_name] = xr.concat(ur_outputs, dim="cell_name")

        for gr in reqs.grouped_reductions:
            group_by = adata.obs[gr.group_by]
            group_by_counts = group_by.value_counts()
            averaged_grouped_data_arrs = []
            for cat, count in group_by_counts.items():
                averaged_grouped_data_arrs.append(
                    grouped_data_arrs[gr.name][cat] / count
                )
            final_data_arr = xr.concat(
                averaged_grouped_data_arrs, dim=f"{gr.group_by}_name"
            )
            final_data_arrs[gr.name] = final_data_arr

        return xr.Dataset(data_vars=final_data_arrs)

    def _compute_local_baseline_dists(
        self, batch: dict, mc_samples: int = 250
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Approximate the distributions used as baselines for normalizing the local sample distances.

        Approximates the means and variances of the Euclidean distance between two samples of
        the z latent representation for the original sample for each cell in ``adata``.

        Reference: https://www.overleaf.com/read/mhdxcrknzxpm.

        Parameters
        ----------
        batch
            Batch of data to compute the local sample representation for.
        mc_samples
            Number of Monte Carlo samples to use for computing the local baseline distributions.
        """

        def get_A_s(module, u, sample_covariate):
            sample_covariate = sample_covariate.astype(int).flatten()
            if not module.qz.use_nonlinear:
                A_s = module.qz.A_s_enc(sample_covariate)
            else:
                # A_s output by a non-linear function without an explicit intercept
                sample_one_hot = jax.nn.one_hot(sample_covariate, module.qz.n_sample)
                A_s_dec_inputs = jnp.concatenate([u, sample_one_hot], axis=-1)
                A_s = module.qz.A_s_enc(A_s_dec_inputs, training=False)
            # cells by n_latent by n_latent
            return A_s.reshape(sample_covariate.shape[0], module.qz.n_latent, -1)

        def apply_get_A_s(u, sample_covariate):
            vars_in = {"params": self.module.params, **self.module.state}
            rngs = self.module.rngs
            A_s = self.module.apply(
                vars_in,
                rngs=rngs,
                method=get_A_s,
                u=u,
                sample_covariate=sample_covariate,
            )
            return A_s

        if self.can_compute_normalized_dists:
            jit_inference_fn = self.module.get_jit_inference_fn()
            qu = jit_inference_fn(self.module.rngs, batch)["qu"]
            qu_vars_diag = jax.vmap(jnp.diag)(qu.variance)

            sample_index = self.module._get_inference_input(batch)["sample_index"]
            A_s = apply_get_A_s(
                qu.mean, sample_index
            )  # use mean of latent representation to compute the baseline
            B = jnp.expand_dims(jnp.eye(A_s.shape[1]), 0) + A_s
            u_diff_sigma = 2 * jnp.einsum(
                "cij, cjk, clk -> cil", B, qu_vars_diag, B
            )  # 2 * (I + A_s) @ qu_vars_diag @ (I + A_s).T
            eigvals = jax.vmap(jnp.linalg.eigh)(u_diff_sigma)[0].astype(float)
            normal_rng = self.module.rngs[
                "params"
            ]  # Hack to get new rng for normal samples.
            normal_samples = jax.random.normal(
                normal_rng, shape=(eigvals.shape[0], mc_samples, eigvals.shape[1])
            )  # n_cells by mc_samples by n_latent
            squared_l2_dists = jnp.sum(
                jnp.einsum("cij, cj -> cij", (normal_samples**2), eigvals), axis=2
            )
            l2_dists = squared_l2_dists**0.5
        else:
            mc_samples_per_cell = (
                mc_samples * 2
            )  # need double for pairs of samples to compute distance between
            jit_inference_fn = self.module.get_jit_inference_fn(
                inference_kwargs={"use_mean": False, "mc_samples": mc_samples_per_cell}
            )

            outputs = jit_inference_fn(self.module.rngs, batch)

            # figure out how to compute dists here
            z = outputs["z"]
            first_half_z, second_half_z = z[:mc_samples], z[mc_samples:]
            l2_dists = jnp.sqrt(jnp.sum((first_half_z - second_half_z) ** 2, axis=2)).T

        return np.array(jnp.mean(l2_dists, axis=1)), np.array(jnp.var(l2_dists, axis=1))

    def _compute_distances_from_representations(
        self, reps, indices, norm="l2"
    ) -> xr.DataArray:
        @jax.jit
        def _compute_distance(rep):
            delta_mat = jnp.expand_dims(rep, 0) - jnp.expand_dims(rep, 1)
            if norm == "l2":
                res = delta_mat**2
                res = jnp.sqrt(res.sum(-1))
            elif norm == "l1":
                res = jnp.abs(delta_mat).sum(-1)
            elif norm == "linf":
                res = jnp.abs(delta_mat).max(-1)
            else:
                raise ValueError(f"norm {norm} not supported")
            return res

        if reps.ndim == 3:
            dists = jax.vmap(_compute_distance)(reps)
            return xr.DataArray(
                dists,
                dims=["cell_name", "sample_x", "sample_y"],
                coords={
                    "cell_name": self.adata.obs_names[indices],
                    "sample_x": self.sample_order,
                    "sample_y": self.sample_order,
                },
                name="sample_distances",
            )
        else:
            # Case with sampled representations
            dists = jax.vmap(jax.vmap(_compute_distance))(reps)
            return xr.DataArray(
                dists,
                dims=["cell_name", "mc_sample", "sample_x", "sample_y"],
                coords={
                    "cell_name": self.adata.obs_names[indices],
                    "mc_sample": np.arange(reps.shape[1]),
                    "sample_x": self.sample_order,
                    "sample_y": self.sample_order,
                },
                name="sample_distances",
            )

    def get_local_sample_representation(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[list[str]] = None,
        batch_size: int = 256,
        use_mean: bool = True,
        use_vmap: bool = True,
    ) -> xr.DataArray:
        """
        Computes the local sample representation of the cells in the adata object.

        For each cell, it returns a matrix of size (n_sample, n_features).

        Parameters
        ----------
        adata
            AnnData object to use for computing the local sample representation.
        batch_size
            Batch size to use for computing the local sample representation.
        use_mean
            Whether to use the mean of the latent representation as the local sample representation.
        use_vmap
            Whether to use vmap for computing the local sample representation.
            Disabling vmap can be useful if running out of memory on a GPU.
        """
        reductions = [
            MrVIReduction(
                name="sample_representations",
                input="mean_representations" if use_mean else "sampled_representations",
                fn=lambda x: x,
                group_by=None,
            )
        ]
        return self.compute_local_statistics(
            reductions,
            adata=adata,
            indices=indices,
            batch_size=batch_size,
            use_vmap=use_vmap,
        ).sample_representations

    def get_local_sample_distances(
        self,
        adata: Optional[AnnData] = None,
        batch_size: int = 256,
        use_mean: bool = True,
        normalize_distances: bool = False,
        use_vmap: bool = True,
        groupby: Optional[Union[list[str], str]] = None,
        keep_cell: bool = True,
        norm: str = "l2",
        mc_samples: int = 10,
    ) -> xr.Dataset:
        """
        Computes local sample distances as `xr.Dataset`.

        Computes cell-specific distances between samples, of size (n_sample, n_sample),
        stored as a Dataset, with variable name `cell`, of size (n_cell, n_sample, n_sample).
        If in addition, groupby is provided, distances are also aggregated by group.
        In this case, the group-specific distances via group name key.

        Parameters
        ----------
        adata
            AnnData object to use for computing the local sample representation.
        batch_size
            Batch size to use for computing the local sample representation.
        use_mean
            Whether to use the mean of the latent representation as the local sample representation.
        normalize_distances
            Whether to normalize the local sample distances. Normalizes by
            the standard deviation of the original intra-sample distances.
            Only works with ``use_mean=False``.
        use_vmap
            Whether to use vmap for computing the local sample representation.
            Disabling vmap can be useful if running out of memory on a GPU.
        groupby
            List of categorical keys or single key of the anndata that is
            used to group the cells.
        keep_cell
            Whether to keep the original cell sample-sample distance matrices.
        norm
            Norm to use for computing the local sample distances.
        mc_samples
            Number of Monte Carlo samples to use for computing the local sample distances.
            Only relevants if ``use_mean=False``.
        """
        input = "mean_distances" if use_mean else "sampled_distances"
        if normalize_distances:
            if use_mean:
                warnings.warn(
                    "Normalizing distances uses sampled distances. Ignoring ``use_mean``.",
                    UserWarning,
                    stacklevel=2,
                )
            input = "normalized_distances"
        if groupby and not isinstance(groupby, list):
            groupby = [groupby]

        reductions = []
        if not keep_cell and not groupby:
            raise ValueError(
                "Undefined computation because not keep_cell and no groupby."
            )
        if keep_cell:
            reductions.append(
                MrVIReduction(
                    name="cell",
                    input=input,
                    fn=lambda x: x,
                )
            )
        if groupby:
            for groupby_key in groupby:
                reductions.append(
                    MrVIReduction(
                        name=groupby_key,
                        input=input,
                        group_by=groupby_key,
                    )
                )
        return self.compute_local_statistics(
            reductions,
            adata=adata,
            batch_size=batch_size,
            use_vmap=use_vmap,
            norm=norm,
            mc_samples=mc_samples,
        )

    def get_aggregated_posterior(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[list[str]] = None,
        batch_size: int = 256,
    ) -> dist.Distribution:
        """
        Computes the aggregated posterior over the ``u`` latent representations.

        Parameters
        ----------
        adata
            AnnData object to use. Defaults to the AnnData object used to initialize the model.
        indices
            Indices of cells to use.
        batch_size
            Batch size to use for computing the latent representation.

        Returns
        -------
        A mixture distribution of the aggregated posterior.
        """
        self._check_if_trained(warn=False)
        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size, iter_ndarray=True
        )

        qu_locs = []
        qu_scales = []
        jit_inference_fn = self.module.get_jit_inference_fn(
            inference_kwargs={"use_mean": True}
        )
        for array_dict in scdl:
            outputs = jit_inference_fn(self.module.rngs, array_dict)

            qu_locs.append(outputs["qu"].loc)
            qu_scales.append(outputs["qu"].scale)

        qu_loc = jnp.concatenate(qu_locs, axis=0).T
        qu_scale = jnp.concatenate(qu_scales, axis=0).T
        return dist.MixtureSameFamily(
            dist.Categorical(probs=jnp.ones(qu_loc.shape[1]) / qu_loc.shape[1]),
            dist.Normal(qu_loc, qu_scale),
        )

    def get_outlier_cell_sample_pairs(
        self,
        adata=None,
        flavor: Literal["ball", "ap", "MoG"] = "ball",
        subsample_size: int = 5000,
        quantile_threshold: float = 0.05,
        admissibility_threshold: float = 0.0,
        minibatch_size: int = 256,
    ) -> xr.Dataset:
        """Utils function to get outlier cell-sample pairs.

        This function fits a GMM for each sample based on the latent representation
        of the cells in the sample or computes an approximate aggregated posterior for each sample.
        Then, for every cell, it computes the log-probability of the cell under the approximated posterior
        of each sample as a measure of admissibility.

        Parameters
        ----------
        adata
            AnnData object containing the cells for which to compute the outlier cell-sample pairs.
        flavor
            Method of approximating posterior on latent representation.
        subsample_size
            Number of cells to use from each sample to approximate the posterior. If None, uses all of the available cells.
        quantile_threshold
            Quantile of the within-sample log probabilities to use as a baseline for admissibility.
        admissibility_threshold
            Threshold for admissibility. Cell-sample pairs with admissibility below this threshold are considered outliers.
        """
        adata = self.adata if adata is None else adata
        adata = self._validate_anndata(adata)

        # Compute u reps
        us = self.get_latent_representation(adata, use_mean=True, give_z=False)
        adata.obsm["U"] = us

        log_probs = []
        threshs = []
        unique_samples = adata.obs[self.sample_key].unique()
        for sample_name in tqdm(unique_samples):
            sample_idxs = np.where(adata.obs[self.sample_key] == sample_name)[0]
            if subsample_size is not None and sample_idxs.shape[0] > subsample_size:
                sample_idxs = np.random.choice(
                    sample_idxs, size=subsample_size, replace=False
                )
            adata_s = adata[sample_idxs]
            if flavor == "MoG":
                n_components = min(adata_s.n_obs // 4, 20)
                gmm_ = GaussianMixture(n_components=n_components).fit(adata_s.obsm["U"])
                log_probs_s = jnp.quantile(
                    gmm_.score_samples(adata_s.obsm["U"]), q=quantile_threshold
                )
                log_probs_ = gmm_.score_samples(adata.obsm["U"])[:, None]
            elif flavor == "ap":
                ap = self.get_aggregated_posterior(adata=adata, indices=sample_idxs)
                log_probs_s = jnp.quantile(
                    ap.log_prob(adata_s.obsm["U"]).sum(axis=1), q=quantile_threshold
                )
                n_splits = adata.n_obs // minibatch_size
                log_probs_ = []
                for u_rep in np.array_split(adata.obsm["U"], n_splits):
                    log_probs_.append(
                        jax.device_get(ap.log_prob(u_rep).sum(-1, keepdims=True))
                    )

                log_probs_ = np.concatenate(log_probs_, axis=0)  # (n_cells, 1)
            elif flavor == "ball":
                ap = self.get_aggregated_posterior(adata=adata, indices=sample_idxs)
                in_max_comp_log_probs = ap.component_distribution.log_prob(
                    np.expand_dims(adata_s.obsm["U"], ap.mixture_dim)
                ).sum(axis=1)
                log_probs_s = rowwise_max_excluding_diagonal(in_max_comp_log_probs)

                log_probs_ = []
                n_splits = adata.n_obs // minibatch_size
                for u_rep in np.array_split(adata.obsm["U"], n_splits):
                    log_probs_.append(
                        jax.device_get(
                            ap.component_distribution.log_prob(
                                np.expand_dims(u_rep, ap.mixture_dim)
                            )
                            .sum(axis=1)
                            .max(axis=1, keepdims=True)
                        )
                    )

                log_probs_ = np.concatenate(log_probs_, axis=0)  # (n_cells, 1)
            else:
                raise ValueError(f"Unknown flavor {flavor}")

            threshs.append(np.array(log_probs_s))
            log_probs.append(np.array(log_probs_))

        if flavor == "ball":
            # Compute a threshold across all samples
            threshs_all = np.concatenate(threshs)
            global_thresh = np.quantile(threshs_all, q=quantile_threshold)
            threshs = len(log_probs) * [global_thresh]

        log_probs = np.concatenate(log_probs, 1)
        threshs = np.array(threshs)
        log_ratios = log_probs - threshs

        coords = {
            "cell_name": adata.obs_names.to_numpy(),
            "sample": unique_samples,
        }
        data_vars = {
            "log_probs": (["cell_name", "sample"], log_probs),
            "log_ratios": (
                ["cell_name", "sample"],
                log_ratios,
            ),
            "is_admissible": (
                ["cell_name", "sample"],
                log_ratios > admissibility_threshold,
            ),
        }
        return xr.Dataset(data_vars, coords=coords)

    def perform_multivariate_analysis(
        self,
        adata: Optional[AnnData] = None,
        donor_keys: list[tuple] = None,
        donor_subset: Optional[list[str]] = None,
        batch_size: int = 256,
        use_vmap: bool = True,
        normalize_design_matrix: bool = True,
        add_batch_specific_offsets: bool = False,
        mc_samples: int = 100,
        store_lfc: bool = False,
        store_lfc_metadata_subset: Optional[list[str]] = None,
        store_baseline: bool = False,
        eps_lfc: float = 1e-3,
        filter_donors: bool = False,
        lambd: float = 0.0,
        delta: Optional[float] = 0.3,
        **filter_donors_kwargs,
    ) -> xr.Dataset:
        """Utility function to perform cell-specific multivariate analysis.

        For every cell, we first compute all counterfactual cell-state shifts, defined as
        e_d = z_d - u, where z_d is the latent representation of the cell for donor d and u is the donor-unaware latent representation.
        Then, we fit a linear model in each cell of the form
        e_d = X_d * beta_d + iid gaussian noise.

        Parameters
        ----------
        adata
            AnnData object to use for computing the local sample representation.
            If None, the analysis is performed on all cells in the dataset.
        donor_keys
            List of donor metadata to consider for the multivariate analysis.
            These keys should be present in `adata.obs`.
        donor_subset
            Optional list of donors to consider for the multivariate analysis.
            If None, all donors are considered.
        batch_size
            Batch size to use for computing the local sample representation.
        use_vmap
            Whether to use vmap for computing the local sample representation.
        normalize_design_matrix
            Whether to normalize the design matrix.
        add_batch_specific_offsets
            Whether to offset the design matrix by adding batch-specific offsets to the design matrix.
            Setting this option to True is recommended when considering multi-site datasets.
        mc_samples
            How many MC samples should be taken for computing betas.
        store_lfc
            Whether to store the log-fold changes in the module.
            Storing log-fold changes is memory-intensive and may require to specify
            a smaller set of cells to analyze, e.g., by specifying `adata`.
        store_lfc_metadata_subset
            Specifies a subset of metadata for which log-fold changes are computed.
            These keys must be a subset of `donor_keys`.
            Only applies when `store_lfc=True`.
        store_baseline
            Whether to store the expression in the module if logfoldchanges are computed.
        eps_lfc
            Epsilon to add to the log-fold changes to avoid detecting genes with low expression.
        filter_donors
            Whether to filter out-of-distribution donors prior to performing the analysis.
        filter_donors_kwargs
            Keyword arguments to pass to `get_outlier_cell_sample_pairs`.
        lambd
            Regularization parameter for the linear model.
        delta
            LFC threshold used to compute posterior DE probabilities.
            If None does not compute them to save memory consumption.
        """
        adata = self.adata if adata is None else adata
        self._check_if_trained(warn=False)
        # Hack to ensure new AnnDatas have indices and indices have correct dimensions.
        if adata is not None:
            adata.obs["_indices"] = np.arange(adata.n_obs).astype(int)

        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(
            adata=adata, indices=None, batch_size=batch_size, iter_ndarray=True
        )
        n_sample = self.summary_stats.n_sample
        vars_in = {"params": self.module.params, **self.module.state}

        donor_mask = (
            np.isin(self.sample_order, donor_subset)
            if donor_subset is not None
            else np.ones(n_sample, dtype=bool)
        )
        donor_mask = np.array(donor_mask)
        donor_order = self.sample_order[donor_mask]
        n_samples_kept = donor_mask.sum()

        if filter_donors:
            admissible_donors = self.get_outlier_cell_sample_pairs(
                adata=adata, **filter_donors_kwargs
            )["is_admissible"].loc[{"sample": donor_order}]
            assert (admissible_donors.sample == donor_order).all()
            admissible_donors = admissible_donors.values
        else:
            admissible_donors = np.ones((adata.n_obs, n_samples_kept), dtype=bool)
        n_admissible_donors = admissible_donors.sum(1)

        (
            Xmat,
            Xmat_names,
            covariates_require_lfc,
            offset_indices,
        ) = self._construct_design_matrix(
            donor_keys=donor_keys,
            donor_mask=donor_mask,
            normalize_design_matrix=normalize_design_matrix,
            add_batch_specific_offsets=add_batch_specific_offsets,
            store_lfc=store_lfc,
            store_lfc_metadata_subset=store_lfc_metadata_subset,
        )
        add_batch_specific_offsets = offset_indices is not None
        n_covariates = Xmat.shape[1]

        @partial(jax.jit, backend="cpu")
        def process_design_matrix(
            admissible_donors_dmat,
            Xmat,
        ):
            xtmx = jnp.einsum("ak,nkl,lm->nam", Xmat.T, admissible_donors_dmat, Xmat)
            xtmx += lambd * jnp.eye(n_covariates)

            prefactor = jnp.real(jax.vmap(jax.scipy.linalg.sqrtm)(xtmx))
            inv_ = jax.vmap(jnp.linalg.pinv)(xtmx)
            Amat = jnp.einsum("nab,bc,ncd->nad", inv_, Xmat.T, admissible_donors_dmat)
            return Amat, prefactor

        # @partial(jax.jit, static_argnames=["use_mean", "mc_samples"])
        def mapped_inference_fn(
            stacked_rngs,
            x,
            sample_index,
            continuous_covs,
            cf_sample,
            Amat,
            prefactor,
            n_donors_per_cell,
            admissible_donors_mat,
            use_mean,
            mc_samples,
            rngs_de=None,
        ):
            def inference_fn(
                rngs,
                cf_sample,
            ):
                return self.module.apply(
                    vars_in,
                    rngs=rngs,
                    method=self.module.inference,
                    x=x,
                    sample_index=sample_index,
                    cf_sample=cf_sample,
                    use_mean=use_mean,
                    mc_samples=mc_samples,
                )["eps"]

            if use_vmap:
                eps_ = jax.vmap(inference_fn, in_axes=(0, 0), out_axes=-2)(
                    stacked_rngs,
                    cf_sample,
                )
            else:

                def per_sample_inference_fn(pair):
                    rngs, cf_sample = pair
                    return inference_fn(rngs, cf_sample)

                # eps_ has shape (mc_samples, n_cells, n_donors, n_latent)
                eps_ = jax.lax.transpose(
                    jax.lax.map(per_sample_inference_fn, (stacked_rngs, cf_sample)),
                    (1, 2, 0, 3),
                )
            eps_std = eps_.std(axis=2, keepdims=True)
            eps_mean = eps_.mean(axis=2, keepdims=True)

            eps = (eps_ - eps_mean) / (1e-6 + eps_std)  # over donors
            # MLE for betas
            betas = jnp.einsum("nks,ansd->ankd", Amat, eps)

            # Statistical tests
            betas_norm = jnp.einsum("ankd,nkl->anld", betas, prefactor)
            ts = (betas_norm**2).mean(axis=0).sum(axis=-1)
            pvals = 1 - jax.scipy.stats.chi2.cdf(ts, df=n_donors_per_cell[:, None])

            betas = betas * eps_std

            if store_lfc:
                betas_ = betas.transpose((0, 2, 1, 3))
                eps_mean_ = eps_mean.transpose((0, 2, 1, 3))
                betas_covariates = betas_[:, covariates_require_lfc, :, :]

                def h_inference_fn(extra_eps, batch_index_cf, batch_offset_eps):
                    extra_eps += batch_offset_eps

                    return self.module.apply(
                        vars_in,
                        rngs=rngs_de,
                        method=self.module.compute_h_from_x_eps,
                        x=x,
                        extra_eps=extra_eps,
                        sample_index=sample_index,
                        batch_index=batch_index_cf,
                        cf_sample=None,
                        continuous_covs=continuous_covs,
                        mc_samples=None,  # mc_samples also taken for eps. vmap over mc_samples
                    )

                batch_index_ = jnp.arange(self.summary_stats.n_batch)[:, None]
                batch_index_ = jnp.repeat(batch_index_, repeats=n_cells, axis=1)[
                    ..., None
                ]  # (n_batch, n_cells, 1)
                betas_null = jnp.zeros_like(betas_covariates)

                if add_batch_specific_offsets:
                    batch_weights = jnp.einsum(
                        "nd,db->nb", admissible_donors_mat, Xmat[:, offset_indices]
                    ).mean(0)
                    betas_offset_ = betas_[:, offset_indices, :, :] + eps_mean_
                else:
                    batch_weights = (1.0 / self.summary_stats.n_batch) * jnp.ones(
                        self.summary_stats.n_batch
                    )
                    mc_samples, _, n_cells_, n_latent = betas_covariates.shape
                    betas_offset_ = (
                        jnp.zeros(
                            (mc_samples, self.summary_stats.n_batch, n_cells_, n_latent)
                        )
                        + eps_mean_
                    )
                # batch_offset shape (mc_samples, n_batch, n_cells, n_latent)

                f_ = jax.vmap(
                    h_inference_fn, in_axes=(0, None, 0), out_axes=0
                )  # fn over MC samples
                f_ = jax.vmap(
                    f_, in_axes=(1, None, None), out_axes=1
                )  # fn over covariates
                f_ = jax.vmap(f_, in_axes=(None, 0, 1), out_axes=0)  # fn over batches
                h_fn = jax.jit(
                    f_
                )  # final expected shapes for x_1, x_0 (n_batch, mc_samples, n_covariates, n_cells, n_genes)

                x_1 = h_fn(betas_covariates, batch_index_, betas_offset_)
                x_0 = h_fn(betas_null, batch_index_, betas_offset_)

                lfcs = jnp.log2(x_1 + eps_lfc) - jnp.log2(x_0 + eps_lfc)
                lfc_mean = jnp.average(lfcs.mean(1), weights=batch_weights, axis=0)
                if delta is not None:
                    lfc_std = jnp.sqrt(
                        jnp.average(lfcs.var(1), weights=batch_weights, axis=0)
                    )
                    pde = (jnp.abs(lfcs) >= delta).mean(1).mean(0)
                else:
                    lfc_std = None
                    pde = None
            else:
                lfc_mean = None
                lfc_std = None
                pde = None
            if store_baseline:
                baseline_expression = x_1.mean(1)
            else:
                baseline_expression = None
            return {
                "beta": betas.mean(0),
                "effect_size": ts,
                "pvalue": pvals,
                "lfc_mean": lfc_mean,
                "lfc_std": lfc_std,
                "pde": pde,
                "baseline_expression": baseline_expression,
            }

        beta = []
        effect_size = []
        pvalue = []
        lfc = []
        lfc_std = []
        pde = []
        baseline_expression = []
        for array_dict in tqdm(scdl):
            indices = array_dict[REGISTRY_KEYS.INDICES_KEY].astype(int).flatten()
            n_cells = array_dict[REGISTRY_KEYS.X_KEY].shape[0]
            cf_sample = np.broadcast_to(
                (np.where(donor_mask)[0])[:, None, None], (n_samples_kept, n_cells, 1)
            )
            inf_inputs = self.module._get_inference_input(
                array_dict,
            )
            continuous_covs = inf_inputs.get(REGISTRY_KEYS.CONT_COVS_KEY, None)
            if continuous_covs is not None:
                continuous_covs = jnp.array(continuous_covs)
            stacked_rngs = self._generate_stacked_rngs(cf_sample.shape[0])

            rngs_de = self.module.rngs if store_lfc else None
            admissible_donors_mat = jnp.array(
                admissible_donors[indices]
            )  # (n_cells, n_donors)
            n_donors_per_cell = admissible_donors_mat.sum(axis=1)
            admissible_donors_dmat = jax.vmap(jnp.diag)(admissible_donors_mat).astype(
                float
            )  # (n_cells, n_donors, n_donors)
            # element nij is 1 if donor i is admissible and i=j for cell n
            Amat, prefactor = process_design_matrix(admissible_donors_dmat, Xmat)
            Amat = jax.device_put(Amat, self.device)
            prefactor = jax.device_put(prefactor, self.device)

            res = mapped_inference_fn(
                stacked_rngs=stacked_rngs,
                x=jnp.array(inf_inputs["x"]),
                sample_index=jnp.array(inf_inputs["sample_index"]),
                continuous_covs=continuous_covs,
                cf_sample=jnp.array(cf_sample),
                Amat=Amat,
                prefactor=prefactor,
                n_donors_per_cell=n_donors_per_cell,
                admissible_donors_mat=admissible_donors_mat,
                use_mean=False,
                rngs_de=rngs_de,
                mc_samples=mc_samples,
            )
            beta.append(np.array(res["beta"]))
            effect_size.append(np.array(res["effect_size"]))
            pvalue.append(np.array(res["pvalue"]))
            if store_lfc:
                lfc.append(np.array(res["lfc_mean"]))
                if delta is not None:
                    lfc_std.append(np.array(res["lfc_std"]))
                    pde.append(np.array(res["pde"]))
            if store_baseline:
                baseline_expression.append(np.array(res["baseline_expression"]))
        beta = np.concatenate(beta, axis=0)
        effect_size = np.concatenate(effect_size, axis=0)
        pvalue = np.concatenate(pvalue, axis=0)
        pvalue_shape = pvalue.shape
        padj = multipletests(pvalue.flatten(), method="fdr_bh")[1].reshape(pvalue_shape)

        coords = {
            "cell_name": (("cell_name"), adata.obs_names),
            "covariate": (("covariate"), Xmat_names),
            "latent_dim": (("latent_dim"), np.arange(beta.shape[2])),
            "gene": (("gene"), adata.var_names),
        }
        data_vars = {
            "beta": (
                ["cell_name", "covariate", "latent_dim"],
                beta,
            ),
            "effect_size": (
                ["cell_name", "covariate"],
                effect_size,
            ),
            "pvalue": (
                ["cell_name", "covariate"],
                pvalue,
            ),
            "padj": (
                ["cell_name", "covariate"],
                padj,
            ),
        }
        if filter_donors:
            data_vars["n_donor"] = (
                ["cell_name"],
                n_admissible_donors,
            )
        if store_lfc:
            if store_lfc_metadata_subset is None and not add_batch_specific_offsets:
                coords_lfc = ["covariate", "cell_name", "gene"]
            else:
                coords_lfc = ["covariate_sub", "cell_name", "gene"]
                coords["covariate_sub"] = (
                    ("covariate_sub"),
                    Xmat_names[covariates_require_lfc],
                )
            lfc = np.concatenate(lfc, axis=1)
            data_vars["lfc"] = (coords_lfc, lfc)
            if delta is not None:
                lfc_std = np.concatenate(lfc_std, axis=1)
                pde = np.concatenate(pde, axis=1)
                data_vars["lfc_std"] = (coords_lfc, lfc_std)
                data_vars["pde"] = (coords_lfc, pde)

        if store_baseline:
            baseline_expression = np.concatenate(baseline_expression, axis=1)
            data_vars["baseline_expression"] = (
                ["covariate", "cell_name", "gene"],
                baseline_expression,
            )
        return xr.Dataset(data_vars, coords=coords)

    def _construct_design_matrix(
        self,
        donor_keys: list[str],
        donor_mask: np.ndarray,
        normalize_design_matrix: bool,
        add_batch_specific_offsets: bool,
        store_lfc: bool,
        store_lfc_metadata_subset: Optional[list[str]] = None,
    ):
        """
        Starting from a list of donor keys, construct a design matrix of donors and covariates.

        Categorical covariates are one-hot encoded.
        This method returns a tuple of:
        1. The design matrix
        2. A name for each column in the design matrix
        3. The original donor key for each column in the design matrix

        Parameters
        ----------
        donor_keys:
            List of donor metadata to use as covariates.
        donor_mask:
            Mask of admissible donors. Must have the same length as the number of donors in the dataset.
        normalize_design_matrix:
            Whether the design matrix should be 0-1 normalized. This is useful to ensure that the
            beta coefficients are comparable across covariates.
        add_batch_specific_offsets:
            Whether the design matrix should be offset. If True, the matrix includes batch-specific
            offsets. This ensures that we can learn perturbation effects that do not depend on batch effects.

        Returns
        -------
        A tuple consisting of:
        1. The design matrix
        2. Names for each column in the design matrix
        3. A mask precising which coefficients from the design matrix require to compute LFCs.
        4. A mask precising which coefficients from the design matrix correspond to offsets.
        """
        Xmat = []
        Xmat_names = []
        Xmat_dim_to_key = []
        donor_info = self.donor_info.iloc[donor_mask]
        for donor_key in tqdm(donor_keys):
            cov = donor_info[donor_key]
            if (cov.dtype == str) or (cov.dtype == "category"):
                cov = cov.cat.remove_unused_categories()
                cov = pd.get_dummies(cov, drop_first=True)
                cov_names = donor_key + np.array(cov.columns)
                cov = cov.values
            else:
                cov_names = np.array([donor_key])
                cov = cov.values[:, None]
            n_covs = cov.shape[1]
            Xmat.append(cov)
            Xmat_names.append(cov_names)
            Xmat_dim_to_key.append([donor_key] * n_covs)
        Xmat_names = np.concatenate(Xmat_names)
        Xmat = np.concatenate(Xmat, axis=1).astype(np.float32)
        Xmat_dim_to_key = np.concatenate(Xmat_dim_to_key)

        if normalize_design_matrix:
            Xmat = (Xmat - Xmat.min(axis=0)) / (
                1e-6 + Xmat.max(axis=0) - Xmat.min(axis=0)
            )
        if add_batch_specific_offsets:
            cov = donor_info["_scvi_batch"]
            if cov.nunique() == self.summary_stats.n_batch:
                cov = np.eye(self.summary_stats.n_batch)[
                    donor_info["_scvi_batch"].values
                ]
                cov_names = [
                    "offset_batch_" + str(i) for i in range(self.summary_stats.n_batch)
                ]
                Xmat = np.concatenate([cov, Xmat], axis=1)
                Xmat_names = np.concatenate([np.array(cov_names), Xmat_names])
                Xmat_dim_to_key = np.concatenate([np.array(cov_names), Xmat_dim_to_key])

                # Retrieve indices of offset covariates in the right order
                offset_indices = (
                    pd.Series(np.arange(len(Xmat_names)), index=Xmat_names)
                    .loc[cov_names]
                    .values
                )
                offset_indices = jnp.array(offset_indices)
            else:
                warnings.warn(
                    """
                        Number of batches in donor_info does not match number of batches in summary_stats.
                        `add_batch_specific_offsets=True` assumes that samples are not shared across batches.
                        Setting `add_batch_specific_offsets=False`...
                    """,
                    stacklevel=2,
                )
                offset_indices = None
        else:
            offset_indices = None

        Xmat = jnp.array(Xmat)
        if store_lfc:
            covariates_require_lfc = (
                np.isin(Xmat_dim_to_key, store_lfc_metadata_subset)
                if store_lfc_metadata_subset is not None
                else np.isin(Xmat_dim_to_key, donor_keys)
            )
        else:
            covariates_require_lfc = np.zeros(len(Xmat_names), dtype=bool)
        covariates_require_lfc = jnp.array(covariates_require_lfc)

        return Xmat, Xmat_names, covariates_require_lfc, offset_indices

    def compute_cell_scores(
        self,
        donor_keys: list[tuple],
        adata=None,
        batch_size=256,
        use_vmap: bool = True,
        n_mc_samples: int = 200,
        compute_pval: bool = True,
    ) -> xr.Dataset:
        """Computes for each cell a statistic (effect size or p-value)

        Parameters
        ----------
        donor_keys
            List of tuples, where the first element is the sample key and
            the second element is the statistic to be computed
        adata
            AnnData object to use for computing the local sample representation.
        batch_size
            Batch size to use for computing the local sample representation.
        use_vmap
            Whether to use vmap for computing the local sample representation.
            Disabling vmap can be useful if running out of memory on a GPU.
        n_mc_samples
            Number of Monte Carlo trials to use for computing the p-values (if `compute_pval` is True).
        compute_pval
            Whether to compute p-values or effect sizes.
        """
        test_out = "pval" if compute_pval else "effect_size"

        # not jitted because the statistic arg is causing troubles
        def _get_scores(w, x, statistic):
            if compute_pval:
                fn = lambda w, x: permutation_test(
                    w, x, statistic=statistic, n_mc_samples=n_mc_samples
                )
            else:
                fn = lambda w, x: compute_statistic(w, x, statistic=statistic)
            return jax.vmap(fn, in_axes=(0, None))(w, x)

        def get_scores_data_arr_fn(cov, sample_covariate_test):
            return lambda x: xr.DataArray(
                _get_scores(x.data, cov, sample_covariate_test),
                dims=["cell_name"],
                coords={"cell_name": x.coords["cell_name"]},
            )

        reductions = []
        for sample_covariate_key, sample_covariate_test in donor_keys:
            cov = self.donor_info[sample_covariate_key].values
            if sample_covariate_test == "nn":
                cov = pd.Categorical(cov).codes
            cov = jnp.array(cov)
            fn = get_scores_data_arr_fn(cov, sample_covariate_test)
            reductions.append(
                MrVIReduction(
                    name=f"{sample_covariate_key}_{sample_covariate_test}_{test_out}",
                    input="mean_distances",
                    fn=fn,
                )
            )

        return self.compute_local_statistics(
            reductions, adata=adata, batch_size=batch_size, use_vmap=use_vmap
        )

    @property
    def original_donor_key(self):
        """Original donor key used for training the model."""
        return self.adata_manager.registry["setup_args"]["sample_key"]

    def explore_stratifications(
        self,
        distances: xr.Dataset,
        cell_type_keys: Optional[Union[str, list[str]]] = None,
        linkage_method: str = "complete",
        figure_dir: Optional[str] = None,
        show_figures: bool = False,
        sample_metadata: Optional[Union[str, list[str]]] = None,
        cmap_name: str = "tab10",
        cmap_requires_int: bool = True,
        **sns_kwargs,
    ):
        """Analysis of distance matrix stratifications.

        Parameters
        ----------
        distances :
            Cell-type specific distance matrices.
        cell_type_keys :
            Subset of cell types to analyze, by default None
        linkage_method :
            Linkage method to use to cluster distance matrices, by default "complete"
        figure_dir :
            Optional directory in which to save figures, by default None
        show_figures :
            Whether to show figures, by default False
        sample_metadata :
            Metadata keys to plot, by default None
        """
        if figure_dir is not None:
            os.makedirs(figure_dir, exist_ok=True)

        # Convert metadata to hex colors
        colors = None
        if sample_metadata is not None:
            sample_metadata = (
                [sample_metadata]
                if isinstance(sample_metadata, str)
                else sample_metadata
            )
            donor_info_ = self.donor_info.set_index(
                self.registry_["setup_args"]["sample_key"]
            )
            colors = convert_pandas_to_colors(
                donor_info_.loc[:, sample_metadata],
                cmap_name=cmap_name,
                cmap_requires_int=cmap_requires_int,
            )

        # Subsample distances if necessary
        distances_ = distances
        celltype_dimname = distances.dims[0]
        if cell_type_keys is not None:
            cell_type_keys = (
                [cell_type_keys] if isinstance(cell_type_keys, str) else cell_type_keys
            )
            dimname_to_vals = {celltype_dimname: cell_type_keys}
            distances_ = distances.sel(dimname_to_vals)

        figs = []
        for dist_ in distances_:
            celltype_name = dist_.coords[celltype_dimname].item()
            dendrogram = compute_dendrogram_from_distance_matrix(
                dist_,
                linkage_method=linkage_method,
            )
            assert dist_.ndim == 2

            fig = sns.clustermap(
                dist_.to_pandas(),
                row_linkage=dendrogram,
                col_linkage=dendrogram,
                row_colors=colors,
                **sns_kwargs,
            )
            fig.fig.suptitle(celltype_name)
            if figure_dir is not None:
                fig.savefig(os.path.join(figure_dir, f"{celltype_name}.png"))
            if show_figures:
                plt.show()
                plt.clf()
            figs.append(fig)
        return figs
