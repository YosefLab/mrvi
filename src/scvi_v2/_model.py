import logging
from copy import deepcopy
from typing import List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
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
from ._tree_utils import compute_dendrogram_from_distance_matrix
from ._utils import compute_statistic, permutation_test

logger = logging.getLogger(__name__)

DEFAULT_TRAIN_KWARGS = {
    "early_stopping": True,
    "early_stopping_patience": 15,
    "check_val_every_n_epoch": 1,
    "batch_size": 128,
    "train_size": 0.9,
    "plan_kwargs": {"lr": 1e-2, "n_epochs_kl_warmup": 400, "max_norm": 20},
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
        jit_inference_fn = self.module.get_jit_inference_fn(inference_kwargs={"use_mean": True})
        for array_dict in tqdm(scdl):
            outputs = jit_inference_fn(self.module.rngs, array_dict)

            us.append(outputs["u"])
            zs.append(outputs["z"])

        u = np.array(jax.device_get(jnp.concatenate(us, axis=0)))
        z = np.array(jax.device_get(jnp.concatenate(zs, axis=0)))
        return z if give_z else u

    @staticmethod
    def compute_distance_matrix_from_representations(
        representations: xr.DataArray, metric: str = "euclidean"
    ) -> xr.DataArray:
        """
        Compute distance matrices from counterfactual sample representations.

        Parameters
        ----------
        representations
            Counterfactual sample representations of shape (n_cells, n_sample, n_features).
        metric
            Metric to use for computing distance matrix.
        """
        n_cells, n_donors, _ = representations.shape
        pairwise_dists = np.zeros((n_cells, n_donors, n_donors))
        for i, cell_rep in enumerate(representations):
            d_ = pairwise_distances(cell_rep, metric=metric)
            pairwise_dists[i, :, :] = d_
        dists_data_array = xr.DataArray(
            pairwise_dists,
            dims=["obs_name", "sample", "sample"],
            coords=representations.coords,
            name="sample_distances",
        )
        return dists_data_array

    def get_local_sample_representation(
        self,
        adata: Optional[AnnData] = None,
        batch_size: int = 256,
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
        rngs = self.module.rngs

        # TODO: use `self.module.get_jit_inference_fn` when it supports traced values.
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
                return jax.vmap(inference_fn, in_axes=(None, None, 0), out_axes=-2)(
                    x,
                    sample_index,
                    cf_sample,
                )
            else:
                per_sample_inference_fn = lambda cf_sample: inference_fn(x, sample_index, cf_sample)
                return jax.lax.transpose(jax.lax.map(per_sample_inference_fn, cf_sample), (1, 0, 2))

        reps = []
        for array_dict in tqdm(scdl):
            n_cells = array_dict[REGISTRY_KEYS.X_KEY].shape[0]
            cf_sample = np.broadcast_to(np.arange(n_sample)[:, None, None], (n_sample, n_cells, 1))
            inputs = self.module._get_inference_input(
                array_dict,
            )
            zs = mapped_inference_fn(
                x=jnp.array(inputs["x"]),
                sample_index=jnp.array(inputs["sample_index"]),
                cf_sample=jnp.array(cf_sample),
            )
            reps.append(zs)

        reps = np.array(jax.device_get(jnp.concatenate(reps, axis=0)))
        sample_order = self.adata_manager.get_state_registry(MRVI_REGISTRY_KEYS.SAMPLE_KEY).categorical_mapping
        reps_data_array = xr.DataArray(
            reps,
            dims=["obs_name", "sample", "latent_dim"],
            coords={"obs_name": adata.obs_names, "sample": sample_order},
            name="sample_representation",
        )

        return reps_data_array

    def get_local_sample_distances(
        self,
        adata: Optional[AnnData] = None,
        batch_size: int = 256,
        normalize_distances: bool = False,
        use_vmap: bool = True,
    ) -> xr.DataArray:
        """
        Computes the local sample distances of the cells in the adata object.

        For each cell, it returns a matrix of size (n_sample, n_sample).

        Parameters
        ----------
        adata
            AnnData object to use for computing the local sample representation.
        batch_size
            Batch size to use for computing the local sample representation.
        normalize_distances
            Whether to normalize the local sample distances. Normalizes by
            the standard deviation of the original intra-sample distances.
        use_vmap
            Whether to use vmap for computing the local sample representation.
            Disabling vmap can be useful if running out of memory on a GPU.
        """
        reps_data_array = self.get_local_sample_representation(adata=adata, batch_size=batch_size, use_vmap=use_vmap)
        dists_data_array = self.compute_distance_matrix_from_representations(reps_data_array)

        if normalize_distances:
            local_baseline_means, local_baseline_vars = self._compute_local_baseline_dists(adata)
            local_baseline_means = local_baseline_means.reshape(-1, 1, 1)
            local_baseline_vars = local_baseline_vars.reshape(-1, 1, 1)
            dists_data_array = np.clip(dists_data_array - local_baseline_means, a_min=0, a_max=None) / (
                local_baseline_vars**0.5
            )

        return dists_data_array

    def _compute_local_baseline_dists(
        self, adata: Optional[AnnData], mc_samples: int = 1000, batch_size: int = 256
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

        def get_A_s(module, sample_index):
            A_s_embed = module.pz.get_variable("params", "A_s")["embedding"]
            sample_index = sample_index.astype(int).flatten()
            return jnp.take(A_s_embed, sample_index, axis=0)

        def apply_get_A_s(sample_index):
            vars_in = {"params": self.module.params, **self.module.state}
            rngs = self.module.rngs
            A_s = self.module.apply(vars_in, rngs=rngs, method=get_A_s, sample_index=sample_index).reshape(
                sample_index.shape[0], self.module.n_latent, self.module.n_latent
            )
            return A_s

        baseline_means = []
        baseline_vars = []
        for array_dict in tqdm(scdl):
            qu = jit_inference_fn(self.module.rngs, array_dict)["qu"]
            qu_vars_diag = jax.vmap(jnp.diag)(qu.variance)

            sample_index = self.module._get_inference_input(array_dict)["sample_index"]
            A_s = apply_get_A_s(sample_index)
            B = jnp.expand_dims(jnp.eye(A_s.shape[1]), 0) + A_s
            squared_l2_sigma = 2 * jnp.einsum(
                "cij, cjk, clk -> cil", B, qu_vars_diag, B
            )  # (I + A_s) @ qu_vars_diag @ (I + A_s).T
            eigvals = jax.vmap(jnp.linalg.eigh)(squared_l2_sigma)[0].astype(float)
            _ = self.module.rngs  # Regenerate seed_rng
            normal_samples = jax.random.normal(
                self.module.seed_rng, shape=(eigvals.shape[0], mc_samples, eigvals.shape[1])
            )
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

    def compute_sample_stratification(
        adata: AnnData,
        distance_matrices: xr.DataArray,
        linkage_method: str = "ward",
        symmmetry_inconsistency_thresh: float = 0.1,
    ):
        """Computes sample stratification based on distance matrices.

        Performs the following steps for each distance matrix:
        1. Performs hierarchical clustering using `scipy`.
        2. Optionally, computes top differentially expressed genes for the first tree branchings.
        3. Optionally, plot the distance matrices and the top DE genes.
        """
        if distance_matrices.ndim == 2:
            distance_matrices = distance_matrices[None, :, :]
        for dmat in distance_matrices:
            if dmat.shape[0] != dmat.shape[1]:
                raise ValueError("Distance matrices must be square.")
            sym_inconsistency = np.abs(dmat - dmat.T).max()
            if sym_inconsistency > symmmetry_inconsistency_thresh:
                raise ValueError(
                    f"Distance matrix is not symmetric. Maximum symmetry inconsistency is {sym_inconsistency}."
                )
            compute_dendrogram_from_distance_matrix(
                distance_matrix=dmat,
                linkage_method=linkage_method,
                symmetrize=True,
            )
