import logging
import os
from copy import deepcopy
from typing import Dict, List, Optional, Tuple, Union

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
from scvi.data.fields import CategoricalObsField, LayerField, NumericalJointObsField
from scvi.model.base import BaseModelClass, JaxTrainingMixin
from sklearn.metrics import pairwise_distances
from tqdm import tqdm

from ._constants import MRVI_REGISTRY_KEYS
from ._module import MrVAE
from ._tree_utils import (
    compute_dendrogram_from_distance_matrix,
    convert_pandas_to_colors,
)
from ._utils import compute_statistic, permutation_test

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
    pz_kwargs
        Keyword args for :class:`~mrvi.DecoderUZ`.
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

        self.data_splitter = None
        self.module = MrVAE(
            n_input=self.summary_stats.n_vars,
            n_sample=n_sample,
            n_batch=n_batch,
            n_continuous_cov=n_continuous_cov,
            **model_kwargs,
        )
        self.init_params_ = self._get_init_params(locals())

        obs_df = adata.obs.copy()
        obs_df = obs_df.loc[~obs_df._scvi_sample.duplicated("first")]
        self.donor_info = obs_df.set_index("_scvi_sample").sort_index()

    def to_device(self, device):  # noqa: #D102
        # TODO(jhong): remove this once we have a better way to handle device.
        pass

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
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
            CategoricalObsField(MRVI_REGISTRY_KEYS.SAMPLE_KEY, sample_key),
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            NumericalJointObsField(REGISTRY_KEYS.CONT_COVS_KEY, continuous_covariate_keys),
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

    @staticmethod
    def compute_distance_matrix_from_representations(
        representations: xr.DataArray,
        metric: str = "euclidean",
        groups: Optional[Dict[str, List[np.ndarray]]] = None,
        keep_cell: bool = True,
    ) -> xr.Dataset:
        """
        Compute distance matrices from counterfactual sample representations.

        Parameters
        ----------
        representations
            Counterfactual sample representations of shape (n_cells, n_sample, n_features).
        metric
            Metric to use for computing distance matrix.
        groups
            ``n_cells``-length vectors indicating the groups for each cell.
        keep_cell
            Whether to compute and keep per-cell distance matrices.
            Requires that ``groups`` is not ``None``.
        """
        if (not keep_cell) and (groups is None):
            raise ValueError("`keep_cell` must be `True` if `groups` is `None`.")

        n_cells, n_donors, _ = representations.shape
        data_arrays = {}
        if keep_cell:
            pairwise_dists = np.zeros((n_cells, n_donors, n_donors))
            for i, cell_rep in enumerate(representations):
                d_ = pairwise_distances(cell_rep, metric=metric)
                pairwise_dists[i, :, :] = d_
            data_arrays["cell"] = xr.DataArray(
                pairwise_dists,
                dims=["cell_name", "sample_x", "sample_y"],
                coords={
                    "cell_name": representations.cell_name.values,
                    "sample_x": representations.sample.values,
                    "sample_y": representations.sample.values,
                },
            )
        if groups is not None:
            if "cell" in list(groups.keys()):
                raise ValueError("`cell` is an ambiguous dimension name. Please rename the dimension name.")

            for groupby_key, group in groups.items():
                group_cats = group.unique()
                group_dists = []
                new_dimension_key = (
                    f"{groupby_key}_name"  # needs to be different from groupby_key name to construct a valid dataset
                )

                # Computing the mean distance matrix for each group
                for group_cat in group_cats:
                    group_mask = (group == group_cat).values
                    group_reps = representations[group_mask, :, :]
                    group_dist = np.zeros((1, n_donors, n_donors))
                    for cell_rep in group_reps:
                        d_ = pairwise_distances(cell_rep, metric=metric)
                        group_dist[0, :, :] += d_
                    group_dist /= len(group_reps)
                    group_dist_data_array = xr.DataArray(
                        group_dist,
                        dims=[new_dimension_key, "sample_x", "sample_y"],
                        coords={
                            new_dimension_key: [group_cat],
                            "sample_x": representations.sample.values,
                            "sample_y": representations.sample.values,
                        },
                    )
                    group_dists.append(group_dist_data_array)
                group_dist_data = xr.concat(group_dists, dim=new_dimension_key)
                data_arrays[groupby_key] = group_dist_data
        return xr.Dataset(
            {
                **data_arrays,
            }
        )

    def get_local_sample_representation(
        self,
        adata: Optional[AnnData] = None,
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
        adata = self.adata if adata is None else adata
        self._check_if_trained(warn=False)
        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(adata=adata, indices=None, batch_size=batch_size, iter_ndarray=True)
        n_sample = self.summary_stats.n_sample

        vars_in = {"params": self.module.params, **self.module.state}

        # TODO: use `self.module.get_jit_inference_fn` when it supports traced values.
        def inference_fn(
            rngs,
            x,
            sample_index,
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

        @jax.jit
        def mapped_inference_fn(
            stacked_rngs,
            x,
            sample_index,
            cf_sample,
        ):
            if use_vmap:
                return jax.vmap(inference_fn, in_axes=(0, None, None, 0), out_axes=-2)(
                    stacked_rngs,
                    x,
                    sample_index,
                    cf_sample,
                )
            else:

                def per_sample_inference_fn(pair):
                    rngs, cf_sample = pair
                    return inference_fn(rngs, x, sample_index, cf_sample)

                return jax.lax.transpose(jax.lax.map(per_sample_inference_fn, (stacked_rngs, cf_sample)), (1, 0, 2))

        reps = []
        for array_dict in tqdm(scdl):
            n_cells = array_dict[REGISTRY_KEYS.X_KEY].shape[0]
            cf_sample = np.broadcast_to(np.arange(n_sample)[:, None, None], (n_sample, n_cells, 1))
            inputs = self.module._get_inference_input(
                array_dict,
            )
            # Generate a set of RNGs for every cf_sample.
            rngs_list = [self.module.rngs for _ in range(cf_sample.shape[0])]
            # Combine list of RNG dicts into a single list. This is necessary for vmap/map.
            stacked_rngs = {
                required_rng: jnp.concatenate([rngs_dict[required_rng][None] for rngs_dict in rngs_list], axis=0)
                for required_rng in self.module.required_rngs
            }
            zs = mapped_inference_fn(
                stacked_rngs=stacked_rngs,
                x=jnp.array(inputs["x"]),
                sample_index=jnp.array(inputs["sample_index"]),
                cf_sample=jnp.array(cf_sample),
            )
            reps.append(zs)

        reps = np.array(jax.device_get(jnp.concatenate(reps, axis=0)))
        sample_order = self.adata_manager.get_state_registry(MRVI_REGISTRY_KEYS.SAMPLE_KEY).categorical_mapping
        reps_data_array = xr.DataArray(
            reps,
            dims=["cell_name", "sample", "latent_dim"],
            coords={"cell_name": adata.obs_names, "sample": sample_order},
            name="sample_representation",
        )

        return reps_data_array

    def get_local_sample_distances(
        self,
        adata: Optional[AnnData] = None,
        batch_size: int = 256,
        use_mean: bool = True,
        normalize_distances: bool = False,
        use_vmap: bool = True,
        groupby: Optional[Union[List[str], str]] = None,
        keep_cell: bool = True,
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
        adata = self.adata if adata is None else adata
        self._check_if_trained(warn=False)
        adata = self._validate_anndata(adata)
        reps_data = self.get_local_sample_representation(
            adata=adata, batch_size=batch_size, use_mean=use_mean, use_vmap=use_vmap
        )
        cell_groups = None
        if groupby is not None:
            if not isinstance(groupby, list):
                groupby = [groupby]
            cell_groups = {groupby_key: adata.obs[groupby_key] for groupby_key in groupby}
        dists_data = self.compute_distance_matrix_from_representations(
            reps_data, keep_cell=keep_cell, groups=cell_groups
        )
        if normalize_distances:
            if use_mean:
                raise ValueError("normalize_distances can only be used with use_mean=False")
            local_baseline_means, local_baseline_vars = self._compute_local_baseline_dists(adata)
            local_baseline_means = local_baseline_means.reshape(-1, 1, 1)
            local_baseline_vars = local_baseline_vars.reshape(-1, 1, 1)

            for varname, data_arr in dists_data.data_vars.items():
                dists_data[varname] = np.clip(data_arr - local_baseline_means, a_min=0, a_max=None) / (
                    local_baseline_vars**0.5
                )
        return dists_data

    def _compute_local_baseline_dists(
        self, adata: Optional[AnnData] = None, mc_samples: int = 1000, batch_size: int = 256
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Approximate the distributions used as baselines for normalizing the local sample distances.

        Approximates the means and variances of the Euclidean distance between two samples of
        the z latent representation for the original sample for each cell in ``adata``.

        Reference: https://www.overleaf.com/read/mhdxcrknzxpm.

        Parameters
        ----------
        adata
            AnnData object to use for computing the local baseline distributions.
        mc_samples
            Number of Monte Carlo samples to use for computing the local baseline distributions.
        batch_size
            Batch size to use for computing the local baseline distributions.
        """
        adata = self.adata if adata is None else adata
        self._check_if_trained(warn=False)
        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(adata=adata, indices=None, batch_size=batch_size, iter_ndarray=True)

        jit_inference_fn = self.module.get_jit_inference_fn()

        def get_A_s(module, u, sample_covariate):
            sample_covariate = sample_covariate.astype(int).flatten()
            if not module.pz.use_nonlinear:
                A_s = module.pz.A_s_enc(sample_covariate)
            else:
                # A_s output by a non-linear function without an explicit intercept
                sample_one_hot = jax.nn.one_hot(sample_covariate, module.pz.n_sample)
                A_s_dec_inputs = jnp.concatenate([u, sample_one_hot], axis=-1)
                A_s = module.pz.A_s_enc(A_s_dec_inputs, training=False)
            # cells by n_latent by n_latent
            return A_s.reshape(sample_covariate.shape[0], module.pz.n_latent, module.pz.n_latent)

        def apply_get_A_s(u, sample_covariate):
            vars_in = {"params": self.module.params, **self.module.state}
            rngs = self.module.rngs
            A_s = self.module.apply(vars_in, rngs=rngs, method=get_A_s, u=u, sample_covariate=sample_covariate)
            return A_s

        baseline_means = []
        baseline_vars = []
        for array_dict in scdl:
            qu = jit_inference_fn(self.module.rngs, array_dict)["qu"]
            qu_vars_diag = jax.vmap(jnp.diag)(qu.variance)

            sample_index = self.module._get_inference_input(array_dict)["sample_index"]
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
            baseline_means.append(jnp.mean(l2_dists, axis=1))
            baseline_vars.append(jnp.var(l2_dists, axis=1))

        return np.array(jnp.concatenate(baseline_means, axis=0)), np.array(jnp.concatenate(baseline_vars, axis=0))

    def compute_cell_scores(
        self,
        donor_keys: List[Tuple],
        adata=None,
        batch_size=256,
        use_vmap: bool = True,
        n_mc_samples: int = 200,
        compute_pval: bool = True,
    ):
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
        adata = self.adata if adata is None else adata
        self._check_if_trained(warn=False)
        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(adata=adata, indices=None, batch_size=batch_size, iter_ndarray=True)
        n_sample = self.summary_stats.n_sample

        vars_in = {"params": self.module.params, **self.module.state}
        rngs = self.module.rngs

        sample_covariates_values = []
        for sample_covariate_key, sample_covariate_test in donor_keys:
            x = self.donor_info[sample_covariate_key].values
            if sample_covariate_test == "nn":
                x = pd.Categorical(x).codes
            x = jnp.array(x)
            sample_covariates_values.append(x)
        sample_covariate_tests = [key[1] for key in donor_keys]

        def inference_fn(
            x,
            sample_index,
            cf_sample,
        ):
            return self.module.apply(
                vars_in,
                rngs=rngs,
                method=self.module.inference,
                x=x,
                sample_index=sample_index,
                cf_sample=cf_sample,
                use_mean=True,
            )["z"]

        @jax.jit
        def mapped_inference_fn(
            x,
            sample_index,
            cf_sample,
        ):
            if use_vmap:
                reps = jax.vmap(inference_fn, in_axes=(None, None, 0), out_axes=-2)(
                    x,
                    sample_index,
                    cf_sample,
                )
            else:
                per_sample_inference_fn = lambda cf_sample: inference_fn(x, sample_index, cf_sample)
                reps = jax.lax.transpose(jax.lax.map(per_sample_inference_fn, cf_sample), (1, 0, 2))

            euclid_d = lambda x: jnp.sqrt(((x[:, None] - x[None, :]) ** 2).sum(-1))
            dists = jax.vmap(euclid_d, in_axes=0)(reps)
            return dists

        # not jitted because the statistic arg is causing troubles
        def _get_scores(w, x, statistic):
            if compute_pval:
                fn = lambda w, x: permutation_test(w, x, statistic=statistic, n_mc_samples=n_mc_samples)
            else:
                fn = lambda w, x: compute_statistic(w, x, statistic=statistic)
            return jax.vmap(fn, in_axes=(0, None))(w, x)

        sigs = []
        for array_dict in tqdm(scdl):
            n_cells = array_dict[REGISTRY_KEYS.X_KEY].shape[0]
            cf_sample = np.broadcast_to(np.arange(n_sample)[:, None, None], (n_sample, n_cells, 1))
            inputs = self.module._get_inference_input(
                array_dict,
            )
            dists = mapped_inference_fn(
                x=jnp.array(inputs["x"]),
                sample_index=jnp.array(inputs["sample_index"]),
                cf_sample=jnp.array(cf_sample),
            )

            all_pvals = []
            for x, sample_covariate_test in zip(sample_covariates_values, sample_covariate_tests):
                pvals = _get_scores(dists, x, sample_covariate_test)
                all_pvals.append(pvals)
            all_pvals = jnp.stack(all_pvals, axis=-1)
            sigs.append(np.array(jax.device_get(all_pvals)))
        sigs = np.concatenate(sigs, axis=0)
        prepand = "pval_" if compute_pval else "effect_size_"
        columns = [prepand + key[0] for key in donor_keys]
        sigs = pd.DataFrame(sigs, columns=columns)
        return sigs

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
            colors = convert_pandas_to_colors(self.donor_info.loc[:, sample_metadata])

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

            fig = sns.clustermap(dist.to_pandas(), row_linkage=dendrogram, col_linkage=dendrogram, row_colors=colors)
            fig.fig.suptitle(celltype_name)
            if figure_dir is not None:
                fig.savefig(os.path.join(figure_dir, f"{celltype_name}.png"))
            if show_figures:
                plt.show()
                plt.clf()
            figs.append(fig)
        return figs
