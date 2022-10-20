import logging
from copy import deepcopy
from typing import List, Optional, Union

import jax
import jax.numpy as jnp
import numpy as np
from anndata import AnnData
from scvi import REGISTRY_KEYS
from scvi.data import AnnDataManager
from scvi.data.fields import CategoricalJointObsField, CategoricalObsField, LayerField
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
    "plan_kwargs": {"lr": 1e-3, "n_epochs_kl_warmup": 20},
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
    uz_scaler
        Whether to incorporate a learned scaler term in the decoder from u to z.
    uz_scaler_n_hidden
        The number of hidden units in the neural network used to produce the scaler term
        in decoder from u to z.
    px_kwargs
        Keyword args for :class:`~mrvi.components.DecoderZX`.
    pz_kwargs
        Keyword args for :class:`~mrvi.components.DecoderUZ`.
    """

    def __init__(
        self,
        adata,
        **model_kwargs,
    ):
        super().__init__(adata)
        n_cats_per_nuisance_keys = (
            self.adata_manager.get_state_registry(MRVI_REGISTRY_KEYS.CATEGORICAL_NUISANCE_KEYS).n_cats_per_key
            if MRVI_REGISTRY_KEYS.CATEGORICAL_NUISANCE_KEYS in self.adata_manager.data_registry
            else []
        )

        n_sample = self.summary_stats.n_sample
        n_obs_per_sample = (
            adata.obs.groupby(self.adata_manager.get_state_registry(MRVI_REGISTRY_KEYS.SAMPLE_KEY)["original_key"])
            .size()
            .loc[self.adata_manager.get_state_registry(MRVI_REGISTRY_KEYS.SAMPLE_KEY)["categorical_mapping"]]
            .values
        )
        self.data_splitter = None
        self.module = MrVAE(
            n_input=self.summary_stats.n_vars,
            n_sample=n_sample,
            n_obs_per_sample=n_obs_per_sample,
            n_cats_per_nuisance_keys=n_cats_per_nuisance_keys,
            **model_kwargs,
        )
        self.init_params_ = self._get_init_params(locals())

    def to_device(self, device):  # noqa: #D102
        # TODO(jhong): remove this once we have a better way to handle device.
        pass

    @classmethod
    def setup_anndata(
        cls,
        adata: AnnData,
        layer: Optional[str] = None,
        sample_key: Optional[str] = None,
        categorical_nuisance_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        setup_method_args = cls._get_setup_method_args(**locals())
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
            CategoricalObsField(MRVI_REGISTRY_KEYS.SAMPLE_KEY, sample_key),
            CategoricalObsField(REGISTRY_KEYS.LABELS_KEY, None),
            CategoricalJointObsField(MRVI_REGISTRY_KEYS.CATEGORICAL_NUISANCE_KEYS, categorical_nuisance_keys),
        ]
        adata_manager = AnnDataManager(fields=anndata_fields, setup_method_args=setup_method_args)
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

    def train(
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
        mc_samples: int = 5000,
        batch_size: Optional[int] = None,
        give_z: bool = False,
    ) -> np.ndarray:
        self._check_if_trained(warn=False)
        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size, iter_ndarray=True)

        u = []
        z = []
        jit_inference_fn = self.module.get_jit_inference_fn(inference_kwargs={"mc_samples": mc_samples})
        for array_dict in tqdm(scdl):
            outputs = jit_inference_fn(self.module.rngs, array_dict)
            u.append(outputs["u"].mean(0))
            z.append(outputs["z"].mean(0))

        u = np.array(jax.device_get(jnp.concatenate(u, axis=0)))
        z = np.array(jax.device_get(jnp.concatenate(z, axis=0)))
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

    def get_local_sample_representation(
        self,
        adata=None,
        batch_size=256,
        mc_samples: int = 10,
        return_distances=False,
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
        mc_samples
            Number of Monte Carlo samples to use for computing the local sample representation.
        return_distances
            If ``return_distances`` is ``True``, returns a distance matrix of
            size (n_sample, n_sample) for each cell.
        """
        adata = self.adata if adata is None else adata
        self._check_if_trained(warn=False)
        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(adata=adata, indices=None, batch_size=batch_size, iter_ndarray=True)
        n_sample = self.summary_stats.n_sample

        reps = []
        for array_dict in tqdm(scdl):
            n_cells = array_dict[REGISTRY_KEYS.X_KEY].shape[0]
            cf_sample = np.broadcast_to(np.arange(n_sample)[:, None], (n_sample, n_cells)).reshape(-1, 1)
            jit_inference_fn = self.module.get_jit_inference_fn(
                inference_kwargs={"mc_samples": mc_samples, "cf_sample": cf_sample}
            )
            tiled_array_dict = {k: np.tile(v, (n_sample, 1)) for k, v in array_dict.items()}
            inference_outputs = jit_inference_fn(self.module.rngs, tiled_array_dict)
            zs = inference_outputs["z"].reshape(mc_samples, n_cells, n_sample, -1).mean(0)
            reps.append(zs)
        reps = np.array(jax.device_get(jnp.concatenate(reps, axis=0)))

        if return_distances:
            return self.compute_distance_matrix_from_representations(reps)

        return reps
