import logging
import os
from copy import deepcopy
from functools import partial
from typing import Dict, List, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
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
)

logger = logging.getLogger(__name__)

DEFAULT_TRAIN_KWARGS = {
    "early_stopping": True,
    "early_stopping_patience": 15,
    "check_val_every_n_epoch": 1,
    "batch_size": 128,
    "train_size": 0.9,
    "plan_kwargs": {"lr": 1e-2, "n_epochs_kl_warmup": 400, "max_norm": 40, "eps": 1e-8, "weight_decay": 1e-8},
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
        n_continuous_cov = self.summary_stats.get("n_extra_continuous_covs", 0)

        obs_df = adata.obs.copy()
        obs_df = obs_df.loc[~obs_df._scvi_sample.duplicated("first")]
        self.donor_info = obs_df.set_index("_scvi_sample").sort_index()
        self.sample_order = self.adata_manager.get_state_registry(MRVI_REGISTRY_KEYS.SAMPLE_KEY).categorical_mapping

        self.n_obs_per_sample = jnp.array(adata.obs._scvi_sample.value_counts().sort_index().values)

        self.data_splitter = None
        self.module = MrVAE(
            n_input=self.summary_stats.n_vars,
            n_sample=n_sample,
            n_batch=n_batch,
            n_continuous_cov=n_continuous_cov,
            n_obs_per_sample=self.n_obs_per_sample,
            **model_kwargs,
        )
        self.can_compute_normalized_dists = (model_kwargs.get("qz_nn_flavor", "linear") == "linear") and (
            (model_kwargs.get("n_latent_u", None) is None)
            or (model_kwargs.get("n_latent", 10) == model_kwargs.get("n_latent_u", None))
        )
        self.init_params_ = self._get_init_params(locals())

    def to_device(self, device):  # noqa: #D102
        # TODO(jhong): remove this once we have a better way to handle device.
        pass

    def _generate_stacked_rngs(self, n_sets: int) -> Dict[str, jax.random.PRNGKey]:
        rngs_list = [self.module.rngs for _ in range(n_sets)]
        # Combine list of RNG dicts into a single list. This is necessary for vmap/map.
        return {
            required_rng: jnp.concatenate([rngs_dict[required_rng][None] for rngs_dict in rngs_list], axis=0)
            for required_rng in self.module.required_rngs
        }

    @classmethod
    def setup_anndata(  # noqa: #D102
        cls,
        adata: AnnData,
        layer: Optional[str] = None,
        sample_key: Optional[str] = None,
        batch_key: Optional[str] = None,
        continuous_covariate_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        setup_method_args = cls._get_setup_method_args(**locals())
        # Add index for batched computation of local statistics.
        adata.obs["_indices"] = np.arange(adata.n_obs).astype(int)
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
            CategoricalObsField(MRVI_REGISTRY_KEYS.SAMPLE_KEY, sample_key),
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            NumericalJointObsField(REGISTRY_KEYS.CONT_COVS_KEY, continuous_covariate_keys),
            NumericalObsField(REGISTRY_KEYS.INDICES_KEY, "_indices"),
        ]
        adata_manager = AnnDataManager(fields=anndata_fields, setup_method_args=setup_method_args)
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

    def train(  # noqa: #D102
        self,
        max_epochs: Optional[int] = None,
        use_gpu: Optional[Union[str, int, bool]] = None,
        train_size: float = 0.9,
        validation_size: Optional[float] = None,
        batch_size: int = 128,
        early_stopping: bool = False,
        plan_kwargs: Optional[dict] = None,
        **trainer_kwargs,
    ):
        train_kwargs = dict(
            max_epochs=max_epochs,
            use_gpu=use_gpu,
            train_size=train_size,
            validation_size=validation_size,
            batch_size=batch_size,
            early_stopping=early_stopping,
            **trainer_kwargs,
        )
        train_kwargs = dict(deepcopy(DEFAULT_TRAIN_KWARGS), **train_kwargs)
        plan_kwargs = plan_kwargs or {}
        train_kwargs["plan_kwargs"] = dict(deepcopy(DEFAULT_TRAIN_KWARGS["plan_kwargs"]), **plan_kwargs)
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
        scdl = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size, iter_ndarray=True)

        us = []
        zs = []
        jit_inference_fn = self.module.get_jit_inference_fn(inference_kwargs={"use_mean": use_mean})
        for array_dict in tqdm(scdl):
            outputs = jit_inference_fn(self.module.rngs, array_dict)

            us.append(outputs["u"])
            zs.append(outputs["z"])

        u = np.array(jax.device_get(jnp.concatenate(us, axis=0)))
        z = np.array(jax.device_get(jnp.concatenate(zs, axis=0)))
        return z if give_z else u

    def compute_local_statistics(
        self,
        reductions: List[MrVIReduction],
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        batch_size: Optional[int] = None,
        use_vmap: bool = True,
        norm: str = "l2",
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
        """
        if not reductions or len(reductions) == 0:
            raise ValueError("At least one reduction must be provided.")

        adata = self.adata if adata is None else adata
        self._check_if_trained(warn=False)
        # Hack to ensure new AnnDatas have indices.
        if "_indices" not in adata.obs:
            adata.obs["_indices"] = np.arange(adata.n_obs).astype(int)

        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size, iter_ndarray=True)
        n_sample = self.summary_stats.n_sample

        reqs = _parse_local_statistics_requirements(reductions)

        vars_in = {"params": self.module.params, **self.module.state}

        @partial(jax.jit, static_argnames=["use_mean"])
        def mapped_inference_fn(
            stacked_rngs,
            x,
            sample_index,
            cf_sample,
            use_mean,
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

                return jax.lax.transpose(jax.lax.map(per_sample_inference_fn, (stacked_rngs, cf_sample)), (1, 0, 2))

        ungrouped_data_arrs = {}
        grouped_data_arrs = {}
        for ur in reqs.ungrouped_reductions:
            ungrouped_data_arrs[ur.name] = []
        for gr in reqs.grouped_reductions:
            grouped_data_arrs[gr.name] = {}  # Will map group category to running group sum.

        for array_dict in tqdm(scdl):
            indices = array_dict[REGISTRY_KEYS.INDICES_KEY].astype(int).flatten()
            n_cells = array_dict[REGISTRY_KEYS.X_KEY].shape[0]
            cf_sample = np.broadcast_to(np.arange(n_sample)[:, None, None], (n_sample, n_cells, 1))
            inf_inputs = self.module._get_inference_input(
                array_dict,
            )
            stacked_rngs = self._generate_stacked_rngs(cf_sample.shape[0])

            # Compute necessary inputs.
            if reqs.needs_mean_representations:
                mean_zs_ = mapped_inference_fn(
                    stacked_rngs=stacked_rngs,
                    x=jnp.array(inf_inputs["x"]),
                    sample_index=jnp.array(inf_inputs["sample_index"]),
                    cf_sample=jnp.array(cf_sample),
                    use_mean=True,
                )
                mean_zs = xr.DataArray(
                    mean_zs_,
                    dims=["cell_name", "sample", "latent_dim"],
                    coords={"cell_name": self.adata.obs_names[indices], "sample": self.sample_order},
                    name="sample_representations",
                )
            if reqs.needs_sampled_representations:
                sampled_zs_ = mapped_inference_fn(
                    stacked_rngs=stacked_rngs,  # OK to use stacked rngs here since there is no stochasticity for mean rep.
                    x=jnp.array(inf_inputs["x"]),
                    sample_index=jnp.array(inf_inputs["sample_index"]),
                    cf_sample=jnp.array(cf_sample),
                    use_mean=False,
                )
                sampled_zs = xr.DataArray(
                    sampled_zs_,
                    dims=["cell_name", "sample", "latent_dim"],
                    coords={"cell_name": self.adata.obs_names[indices], "sample": self.sample_order},
                    name="sample_representations",
                )

            if reqs.needs_mean_distances or reqs.needs_normalized_distances:
                mean_dists = self._compute_distances_from_representations(mean_zs_, indices, norm=norm)

                if reqs.needs_normalized_distances:
                    if norm != "l2":
                        raise ValueError(f"Norm must be 'l2' when using normalized distances. Got {norm}.")
                    normalization_means, normalization_vars = self._compute_local_baseline_dists(array_dict)
                    normalization_means = normalization_means.reshape(-1, 1, 1)
                    normalization_vars = normalization_vars.reshape(-1, 1, 1)
                    normalized_dists = np.clip(mean_dists - normalization_means, a_min=0, a_max=None) / (
                        normalization_vars**0.5
                    )

            if reqs.needs_sampled_distances:
                sampled_dists = self._compute_distances_from_representations(sampled_zs_, indices, norm=norm)

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
                            cell_name=self.adata.obs_names[indices][group_by == cat].values
                        ).sum(dim="cell_name")
                        cat_summed_outputs = cat_summed_outputs.assign_coords({f"{r.group_by}_name": cat})
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
                averaged_grouped_data_arrs.append(grouped_data_arrs[gr.name][cat] / count)
            final_data_arr = xr.concat(averaged_grouped_data_arrs, dim=f"{gr.group_by}_name")
            final_data_arrs[gr.name] = final_data_arr

        return xr.Dataset(data_vars=final_data_arrs)

    def _compute_local_baseline_dists(self, batch: dict, mc_samples: int = 250) -> Tuple[np.ndarray, np.ndarray]:
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
            A_s = self.module.apply(vars_in, rngs=rngs, method=get_A_s, u=u, sample_covariate=sample_covariate)
            return A_s

        if self.can_compute_normalized_dists:
            jit_inference_fn = self.module.get_jit_inference_fn()
            qu = jit_inference_fn(self.module.rngs, batch)["qu"]
            qu_vars_diag = jax.vmap(jnp.diag)(qu.variance)

            sample_index = self.module._get_inference_input(batch)["sample_index"]
            A_s = apply_get_A_s(qu.mean, sample_index)  # use mean of latent representation to compute the baseline
            B = jnp.expand_dims(jnp.eye(A_s.shape[1]), 0) + A_s
            u_diff_sigma = 2 * jnp.einsum(
                "cij, cjk, clk -> cil", B, qu_vars_diag, B
            )  # 2 * (I + A_s) @ qu_vars_diag @ (I + A_s).T
            eigvals = jax.vmap(jnp.linalg.eigh)(u_diff_sigma)[0].astype(float)
            normal_rng = self.module.rngs["params"]  # Hack to get new rng for normal samples.
            normal_samples = jax.random.normal(
                normal_rng, shape=(eigvals.shape[0], mc_samples, eigvals.shape[1])
            )  # n_cells by mc_samples by n_latent
            squared_l2_dists = jnp.sum(jnp.einsum("cij, cj -> cij", (normal_samples**2), eigvals), axis=2)
            l2_dists = squared_l2_dists**0.5
        else:
            mc_samples_per_cell = mc_samples * 2  # need double for pairs of samples to compute distance between
            jit_inference_fn = self.module.get_jit_inference_fn(
                inference_kwargs={"use_mean": False, "mc_samples": mc_samples_per_cell}
            )

            outputs = jit_inference_fn(self.module.rngs, batch)

            # figure out how to compute dists here
            z = outputs["z"]
            first_half_z, second_half_z = z[:mc_samples], z[mc_samples:]
            l2_dists = jnp.sqrt(jnp.sum((first_half_z - second_half_z) ** 2, axis=2)).T

        return np.array(jnp.mean(l2_dists, axis=1)), np.array(jnp.var(l2_dists, axis=1))

    def _compute_distances_from_representations(self, reps, indices, norm="l2"):
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

    def get_local_sample_representation(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[List[str]] = None,
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
            reductions, adata=adata, indices=indices, batch_size=batch_size, use_vmap=use_vmap
        ).sample_representations

    def get_local_sample_distances(
        self,
        adata: Optional[AnnData] = None,
        batch_size: int = 256,
        use_mean: bool = True,
        normalize_distances: bool = False,
        use_vmap: bool = True,
        groupby: Optional[Union[List[str], str]] = None,
        keep_cell: bool = True,
        norm: str = "l2",
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
        """
        input = "mean_distances" if use_mean else "sampled_distances"
        if normalize_distances:
            if not use_mean:
                raise ValueError("Normalizing distances requires `use_mean=True`.")
            input = "normalized_distances"
        if groupby and not isinstance(groupby, list):
            groupby = [groupby]

        reductions = []
        if not keep_cell and not groupby:
            raise ValueError("Undefined computation because not keep_cell and no groupby.")
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
        )

    def compute_cell_scores(
        self,
        donor_keys: List[Tuple],
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
                fn = lambda w, x: permutation_test(w, x, statistic=statistic, n_mc_samples=n_mc_samples)
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

        return self.compute_local_statistics(reductions, adata=adata, batch_size=batch_size, use_vmap=use_vmap)

    @property
    def original_donor_key(self):
        """Original donor key used for training the model."""
        return self.adata_manager.registry["setup_args"]["sample_key"]

    def differential_expression(
        self,
        adata: AnnData,
        samples_a: List[str],
        samples_b: List[str],
        delta: float = 0.5,
        return_dist: bool = False,
        batch_size: int = 128,
        mc_samples_total: int = 10000,
        max_mc_samples_per_pass: int = 50,
        mc_samples_for_permutation: int = 30000,
        use_vmap: bool = False,
        eps: float = 1e-4,
    ):
        """Computes differential expression between two sets of samples.

        Background on the computation can be found here:
        https://www.overleaf.com/project/63c08ee8d7475a4c8478b1a3.
        To correct for batch effects, the current approach consists in
        computing batch-specific LFCs, and then averaging them.
        Right now, other covariates are not considered.

        Parameters
        ----------
        adata
            AnnData object to use.
        samples_a
            Set of samples to characterize the first sample subpopulation.
            Should be a subset of values associated to the sample key.
        samples_b
            Set of samples to characterize the second sample subpopulation.
            Should be a subset of values associated to the sample key.
        delta
            LFC threshold to use for differential expression.
        return_dist
            Whether to return the distribution of LFCs.
            If False, returns a summarized result, contained in a DataFrame.
        batch_size
            Batch size to use for inference.
        mc_samples_total
            Number of Monte Carlo samples to sample normalized gene expressions, per group of samples.
        max_mc_samples_per_pass
            Maximum number of Monte Carlo samples to sample normalized gene expressions at once.
            Lowering this value can help with memory issues.
        mc_samples_for_permutation
            Number of Monte Carlo samples to use to compute the LFC distribution from normalized expression levels.
        use_vmap
            Determines which parallelization strategy to use. If True, uses `jax.vmap`, otherwise uses `jax.lax.map`.
            The former is faster, but requires more memory.
        eps
            Pseudo-count to add to the normalized expression levels.
        """
        mc_samples_per_obs = np.maximum((mc_samples_total // adata.n_obs), 1)
        if mc_samples_per_obs >= max_mc_samples_per_pass:
            n_passes_per_obs = np.maximum((mc_samples_per_obs // max_mc_samples_per_pass), 1)
            mc_samples_per_obs = max_mc_samples_per_pass
        else:
            n_passes_per_obs = 1

        self._check_if_trained(warn=False)
        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(adata=adata, batch_size=batch_size, iter_ndarray=True)

        donor_mapper = self.donor_info.reset_index().set_index(self.original_donor_key)
        try:
            samples_idx_a_ = np.array(donor_mapper.loc[samples_a, "_scvi_sample"])
            samples_idx_b_ = np.array(donor_mapper.loc[samples_b, "_scvi_sample"])
        except KeyError:
            raise KeyError(
                "Some samples cannot be found."
                "Please make sure that the provided samples can be found in {}.".format(self.original_donor_key)
            )

        vars_in = {"params": self.module.params, **self.module.state}
        rngs = self.module.rngs

        def _get_all_inputs(
            inputs,
        ):
            x = jnp.array(inputs[REGISTRY_KEYS.X_KEY])
            sample_index = jnp.array(inputs[MRVI_REGISTRY_KEYS.SAMPLE_KEY])
            batch_index = jnp.array(inputs[REGISTRY_KEYS.BATCH_KEY])
            continuous_covs = inputs.get(REGISTRY_KEYS.CONT_COVS_KEY, None)
            if continuous_covs is not None:
                continuous_covs = jnp.array(continuous_covs)
            return {
                "x": x,
                "sample_index": sample_index,
                "batch_index": batch_index,
                "continuous_covs": continuous_covs,
            }

        def h_inference_fn(x, sample_index, batch_index, cf_sample, continuous_covs):
            return self.module.apply(
                vars_in,
                rngs=rngs,
                method=self.module.compute_h_from_x,
                x=x,
                sample_index=sample_index,
                batch_index=batch_index,
                cf_sample=cf_sample,
                continuous_covs=continuous_covs,
                mc_samples=mc_samples_per_obs,
            )

        @jax.jit
        def get_hs(inputs, cf_sample):
            _h_inference_fn = lambda cf_sample: h_inference_fn(
                inputs["x"],
                inputs["sample_index"],
                inputs["batch_index"],
                cf_sample,
                inputs["continuous_covs"],
            )
            if use_vmap:
                hs = jax.vmap(h_inference_fn, in_axes=(None, None, None, 0, None), out_axes=0)(
                    inputs["x"],
                    inputs["sample_index"],
                    inputs["batch_index"],
                    cf_sample,
                    inputs["continuous_covs"],
                )
            else:
                hs = jax.lax.map(_h_inference_fn, cf_sample)

            # convert to pseudocounts
            hs = jnp.log(hs + eps)
            return hs

        ha_samples = []
        hb_samples = []

        for array_dict in tqdm(scdl):
            n_cells = array_dict[REGISTRY_KEYS.X_KEY].shape[0]
            inputs = _get_all_inputs(
                array_dict,
            )
            cf_sample_a = np.broadcast_to(samples_idx_a_[:, None, None], (samples_idx_a_.shape[0], n_cells, 1))
            cf_sample_b = np.broadcast_to(samples_idx_b_[:, None, None], (samples_idx_b_.shape[0], n_cells, 1))
            for _ in range(n_passes_per_obs):
                _ha = get_hs(inputs, cf_sample_a)
                _hb = get_hs(inputs, cf_sample_b)

                # shapes (n_samples_x, max_mc_samples_per_pass, n_cells, n_vars)
                ha_samples.append(
                    np.asarray(jax.device_put(_ha.reshape((samples_idx_a_.shape[0], -1, self.summary_stats.n_vars))))
                )
                hb_samples.append(
                    np.asarray(jax.device_put(_hb.reshape((samples_idx_b_.shape[0], -1, self.summary_stats.n_vars))))
                )
        ha_samples = np.concatenate(ha_samples, axis=1)
        hb_samples = np.concatenate(hb_samples, axis=1)

        # Giving equal weight to each sample ==> exchangeability assumption
        ha_samples = ha_samples.reshape((-1, self.summary_stats.n_vars))
        hb_samples = hb_samples.reshape((-1, self.summary_stats.n_vars))

        # compute LFCs
        rdm_idx_a = np.random.choice(ha_samples.shape[0], size=mc_samples_for_permutation, replace=True)
        rdm_idx_b = np.random.choice(hb_samples.shape[0], size=mc_samples_for_permutation, replace=True)
        lfc_dist = ha_samples[rdm_idx_a, :] - hb_samples[rdm_idx_b, :]
        if return_dist:
            return lfc_dist
        else:
            de_probs = np.mean(np.abs(lfc_dist) > delta, axis=0)
            lfc_means = np.mean(lfc_dist, axis=0)
            lfc_medians = np.median(lfc_dist, axis=0)
            results = (
                pd.DataFrame(
                    {
                        "de_prob": de_probs,
                        "lfc_mean": lfc_means,
                        "lfc_median": lfc_medians,
                        "gene_name": self.adata.var_names,
                    }
                )
                .set_index("gene_name")
                .sort_values("de_prob", ascending=False)
            )
            return results

    def explore_stratifications(
        self,
        distances: xr.Dataset,
        cell_type_keys: Optional[Union[str, List[str]]] = None,
        linkage_method: str = "complete",
        figure_dir: Optional[str] = None,
        show_figures: bool = False,
        sample_metadata: Optional[Union[str, List[str]]] = None,
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
            sample_metadata = [sample_metadata] if isinstance(sample_metadata, str) else sample_metadata
            donor_info_ = self.donor_info.set_index(self.registry_["setup_args"]["sample_key"])
            colors = convert_pandas_to_colors(
                donor_info_.loc[:, sample_metadata], cmap_name=cmap_name, cmap_requires_int=cmap_requires_int
            )

        # Subsample distances if necessary
        distances_ = distances
        celltype_dimname = distances.dims[0]
        if cell_type_keys is not None:
            cell_type_keys = [cell_type_keys] if isinstance(cell_type_keys, str) else cell_type_keys
            dimname_to_vals = {celltype_dimname: cell_type_keys}
            distances_ = distances.sel(dimname_to_vals)

        figs = []
        for dist in distances_:
            celltype_name = dist.coords[celltype_dimname].item()
            dendrogram = compute_dendrogram_from_distance_matrix(
                dist,
                linkage_method=linkage_method,
            )
            assert dist.ndim == 2

            fig = sns.clustermap(
                dist.to_pandas(), row_linkage=dendrogram, col_linkage=dendrogram, row_colors=colors, **sns_kwargs
            )
            fig.fig.suptitle(celltype_name)
            if figure_dir is not None:
                fig.savefig(os.path.join(figure_dir, f"{celltype_name}.png"))
            if show_figures:
                plt.show()
                plt.clf()
            figs.append(fig)
        return figs
