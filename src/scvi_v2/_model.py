import logging
from copy import deepcopy
from typing import Optional, Union

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from anndata import AnnData
from scvi import REGISTRY_KEYS
from scvi.data import AnnDataManager
from scvi.data.fields import CategoricalObsField, LayerField
from scvi.model.base import BaseModelClass, JaxTrainingMixin
from sklearn.metrics import pairwise_distances
from tqdm import tqdm

from ._constants import MRVI_REGISTRY_KEYS
from ._module import MrVAE

logger = logging.getLogger(__name__)

DEFAULT_TRAIN_KWARGS = {
    "early_stopping": True,
    "early_stopping_patience": 15,
    "check_val_every_n_epoch": 1,
    "batch_size": 256,
    "train_size": 0.9,
    "plan_kwargs": {"lr": 1e-2, "n_epochs_kl_warmup": 20},
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
        self.data_splitter = None
        self.module = MrVAE(
            n_input=self.summary_stats.n_vars,
            n_sample=n_sample,
            n_batch=n_batch,
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
        **kwargs,
    ):
        setup_method_args = cls._get_setup_method_args(**locals())
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
            CategoricalObsField(MRVI_REGISTRY_KEYS.SAMPLE_KEY, sample_key),
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
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
        representations: np.ndarray, metric: str = "euclidean"
    ) -> np.ndarray:
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
        return pairwise_dists

    def compute_lfcs_from_representations(
        self,
        representations: np.ndarray,
        batch_size: Optional[int] = 16,
    ):
        """Computes n_samples x n_samples x n_genes mean and variance LFC matrices

        Element ijg of the matrix computes the average LFC of gene g between sample i and sample j.
        This tensor can then be used to compute the average LFC between two groups of
        samples.
        """
        n_cells, n_samples, _ = representations.shape
        n_batches = self.summary_stats.n_batch
        n_passes = int(np.ceil(n_cells / batch_size))
        vars_in = {"params": self.module.params, **self.module.state}
        rngs = self.module.rngs
        # Storing the genes by genes matrices on the GPU may be too expensive
        cpu_device = jax.devices("cpu")[0]

        @jax.jit
        def decode(z, library, batch_index):
            return self.module.apply(
                vars_in,
                rngs=rngs,
                method=self.module.generative,
                z=z,
                library=library,
                batch_index=batch_index,
            )["h"]

        # this function decodes all counterfactual latents z at once
        parallel_decode = jax.jit(jax.vmap(decode, in_axes=(1, None, None)))

        n_genes = self.summary_stats.n_vars
        # Initialize first and second LFC moments to 0
        lfcs_m1 = jax.device_put(jnp.zeros((n_samples, n_samples, n_genes)), device=cpu_device)
        lfcs_m2 = jax.device_put(jnp.zeros((n_samples, n_samples, n_genes)), device=cpu_device)
        for cell_reps in np.array_split(representations, n_passes):
            cell_reps_ = jnp.array(cell_reps)
            # the libraries are not used to compute hs but we need placeholders.
            libraries = 5 * jnp.ones((cell_reps_.shape[0], 1))

            hs_all = []
            for batch in range(n_batches):
                nuisance_factors = jnp.ones((cell_reps_.shape[0], 1))
                nuisance_factors *= batch
                hs = parallel_decode(cell_reps_, libraries, nuisance_factors)  # shape (n_bio_samples, n_cells, n_genes)
                hs_all.append(hs[None])
            # shape (n_batches, n_bio_samples, n_cells, n_genes)
            hs_all = jnp.concatenate(hs_all, axis=0)
            hs_all = jnp.log(hs_all)
            lfcs = hs_all[:, None] - hs_all[:, :, None]
            lfcs = jnp.mean(lfcs, axis=0)
            # shape (n_samples, n_samples, n_cells, n_genes)
            lfcs_m1 += jnp.sum(lfcs, axis=2) / n_cells
            lfcs_m2 += jnp.sum(lfcs**2, axis=2) / (n_cells - 1)
        lfcs_std = jnp.sqrt(lfcs_m2 - (lfcs_m1**2))
        return lfcs_m1, lfcs_std

    def get_local_sample_representation(
        self,
        adata=None,
        batch_size=256,
        return_distances=False,
        return_lfcs=False,
        use_vmap=True,
    ):
        """
        Computes the local sample representation of the cells in the adata object.

        For each cell, it returns a matrix of size (n_sample, n_features).

        Parameters
        ----------
        adata
            AnnData object to use for computing the local sample representation.
        batch_size
            Batch size to use for computing the local sample representation.
        return_distances
            If ``return_distances`` is ``True``, returns a distance matrix of
            size (n_sample, n_sample) for each cell.
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

        # reps have shape (n_cells, n_sample, n_features)
        if return_distances:
            return self.compute_distance_matrix_from_representations(reps)
        if return_lfcs:
            lfcs_mean, _ = self.compute_lfcs_from_representations(reps)
            return lfcs_mean
        return reps

    def compute_degs_from_lfcs(self, lfcs, binary_donor_key):
        """Rank genes by LFCs from pairwise LFCs.

        This LFC is defined as
        LFC_g
        = Average(Expression_g for donors in group 2) - Average(Expression_g for donors in group 1)
        = Average(LFC_g^ij, i being a sample from group 2 and j being a sample from group 1)
        """
        n_sample = self.summary_stats.n_sample
        n_genes = self.summary_stats.n_vars
        assert lfcs.shape == (n_sample, n_sample, n_genes)

        sample_cats = self.donor_info[binary_donor_key]
        if sample_cats.nunique() != 2:
            raise ValueError("Binary donor key must have exactly two categories.")
        sample_cats = pd.Categorical(sample_cats)
        sample_cats = sample_cats.codes

        groups_mat = sample_cats[:, None] - sample_cats[None, :]
        # groups_mat[i, j] = 1 iff sample i is in group 1 and sample j is in group 0

        selected_pairs = (groups_mat == 1).reshape(-1)
        lfcs_1d = lfcs.reshape((n_sample * n_sample, n_genes))
        lfcs_1d = lfcs_1d[selected_pairs, :].mean(0)
        # selected_lfcs =

        res = (
            pd.DataFrame(
                dict(
                    gene_name=self.adata.var_names,
                    lfc=np.array(lfcs_1d),
                )
            )
            .assign(abs_lfc=lambda x: x.lfc.abs())
            .sort_values("abs_lfc", ascending=False)
        )
        return res
