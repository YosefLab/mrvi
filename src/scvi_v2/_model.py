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
import numpyro.distributions as dist
import pandas as pd

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
    def setup_anndata(  # noqa: #D102
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
        use_mean: bool = True,
        mc_samples: int = 5000,
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
        use_mean
            Whether to use the mean of the posterior in the computation of the latent. If False,
            `mc_samples` samples from the posterior are used.
        mc_samples
            Number of Monte Carlo samples to use for computing the latent representation.
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
        jit_inference_fn = self.module.get_jit_inference_fn(
            inference_kwargs={"use_mean": use_mean, "mc_samples": mc_samples if not use_mean else 1}
        )
        for array_dict in tqdm(scdl):
            outputs = jit_inference_fn(self.module.rngs, array_dict)

            u = outputs["u"]
            z = outputs["z"]
            if use_mean is False and mc_samples > 1:
                u = u.mean(0)
                z = z.mean(0)
            us.append(u)
            zs.append(z)

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

    def get_local_sample_representation(
        self,
        adata=None,
        indices=None,
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
        scdl = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size, iter_ndarray=True)
        n_sample = self.summary_stats.n_sample

        vars_in = {"params": self.module.params, **self.module.state}
        rngs = self.module.rngs

        # TODO: use `self.module.get_jit_inference_fn` when it supports traced values.
        def inference_fn(
            x,
            sample_index,
            categorical_nuisance_keys,
            cf_sample,
        ):
            return self.module.apply(
                vars_in,
                rngs=rngs,
                method=self.module.inference,
                x=x,
                sample_index=sample_index,
                categorical_nuisance_keys=categorical_nuisance_keys,
                cf_sample=cf_sample,
                mc_samples=mc_samples,
            )["z"].mean(0)

        @jax.jit
        def vmapped_inference_fn(
            x,
            sample_index,
            categorical_nuisance_keys,
            cf_sample,
        ):

            return jax.vmap(inference_fn, in_axes=(None, None, None, 0), out_axes=-2)(
                x,
                sample_index,
                categorical_nuisance_keys,
                cf_sample,
            )

        reps = []
        for array_dict in tqdm(scdl):
            n_cells = array_dict[REGISTRY_KEYS.X_KEY].shape[0]
            cf_sample = np.broadcast_to(np.arange(n_sample)[:, None, None], (n_sample, n_cells, 1))
            inputs = self.module._get_inference_input(
                array_dict,
            )
            zs = vmapped_inference_fn(
                x=jnp.array(inputs["x"]),
                sample_index=jnp.array(inputs["sample_index"]),
                categorical_nuisance_keys=jnp.array(inputs["categorical_nuisance_keys"]),
                cf_sample=jnp.array(cf_sample),
            )
            reps.append(zs)
        reps = np.array(jax.device_get(jnp.concatenate(reps, axis=0)))

        if return_distances:
            return self.compute_distance_matrix_from_representations(reps)

        return reps

    def get_cell_scores(
        self,
        adata=None,
        batch_size=256,
    ):
        """Computes cell overlap probability score square matrix.

        For each pair of cells, the score corresponds to the minimum value alpha s.t.
        n belongs in the alpha-confidence ellipse of np (symmetrized).
        """

        adata = self.adata if adata is None else adata
        self._check_if_trained(warn=False)
        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(adata=adata, batch_size=batch_size, iter_ndarray=True, shuffle=False)
        cpus = jax.devices("cpu")

        vars_in = {"params": self.module.params, **self.module.state}
        rngs = self.module.rngs

        @jax.jit
        def inference_fn(
            x,
            sample_index,
            categorical_nuisance_keys,
        ):
            outs = self.module.apply(
                vars_in,
                rngs=rngs,
                method=self.module.inference,
                x=x,
                sample_index=sample_index,
                categorical_nuisance_keys=categorical_nuisance_keys,
            )
            return outs["qu"].loc, outs["qu"].scale

        qu_m = []
        qu_std = []
        for array_dict in tqdm(scdl):
            inputs = self.module._get_inference_input(
                array_dict,
            )
            _qu_m, _qu_std = inference_fn(
                x=jnp.array(inputs["x"]),
                sample_index=jnp.array(inputs["sample_index"]),
                categorical_nuisance_keys=jnp.array(inputs["categorical_nuisance_keys"]),
            )
            qu_m.append(_qu_m)
            qu_std.append(_qu_std)
        qu_m = jax.device_put(jnp.concatenate(qu_m, axis=0), cpus[0])
        qu_std = jax.device_put(jnp.concatenate(qu_std, axis=0), cpus[0])

        n_latent = qu_m.shape[-1]
        base_dist = dist.Chi2(df=n_latent)

        def compute_ccdf(qu_m, qu_mb, qu_mb_std):
            delta = qu_m - qu_mb
            prec = qu_mb_std**-2
            t = ((delta**2) * prec).sum(-1)
            tail_prob = 1.0 - base_dist.cdf(t)
            return tail_prob

        qu_m1 = qu_m.copy()
        qu_m2 = qu_m.copy()
        tail_probs = jax.vmap(compute_ccdf, (None, 0, 0), -1)(qu_m1, qu_m2, qu_std)

        scores = jnp.maximum(tail_probs, tail_probs.T)
        return scores

    def get_aggregated_distance_mat(self, adata, batch_size=256, mc_samples: int = 10, proba_threshold=0.01):
        assert adata is not None
        # get uncertainty masks
        cell_scores = self.get_cell_scores(
            adata=adata,
            batch_size=batch_size,
        )

        sampleid_assignments = adata.obs["_scvi_sample"]
        sample_keys = adata.obs.loc[
            :, ["_scvi_sample", self.adata_manager.get_state_registry(MRVI_REGISTRY_KEYS.SAMPLE_KEY)["original_key"]]
        ]
        sample_keys = sample_keys[lambda x: ~x.duplicated(keep="first")].set_index("_scvi_sample").squeeze()
        cell_sample_probs = []
        cpus = jax.devices("cpu")
        observed_sampleids = sampleid_assignments.unique()
        observed_samples = sample_keys.loc[observed_sampleids].values
        for i in observed_sampleids:
            cell_in_s = sampleid_assignments.values == i
            cell_sample_probs.append(cell_scores[:, cell_in_s].max(-1)[:, None])
        cell_sample_probs = jax.device_put(jnp.concatenate(cell_sample_probs, axis=1), cpus[0])

        # local reps
        local_reps = self.get_local_sample_representation(
            adata=adata, batch_size=batch_size, mc_samples=mc_samples, return_distances=False
        )
        local_reps = jax.device_put(local_reps, cpus[0])
        local_reps = local_reps[:, observed_sampleids]

        # weightings
        def compute_local_distance(rep, sample_probs):
            rep_norm = (rep**2).sum(-1, keepdims=True)
            rep_norm = jnp.sqrt(rep_norm)
            reps_ = rep / rep_norm
            cos_sim = (reps_[None, :] * reps_[:, None]).sum(-1)
            cos_dist = 1 - cos_sim

            ood_sample = (sample_probs < proba_threshold).astype(float)
            ood_mask = ood_sample[:, None] * ood_sample[None, :]

            adj_dist = cos_dist * (1.0 - ood_mask) + (2.0 * ood_mask)
            return adj_dist

        # adj_dists = jax.vmap(compute_local_distance, (0, 0), 0)(local_reps, cell_sample_probs)
        adj_dists = [compute_local_distance(local_reps[i], cell_sample_probs[i]) for i in range(local_reps.shape[0])]
        adj_dists = jnp.stack(adj_dists, axis=0).mean(0)
        dists = np.array(adj_dists)
        dists = (dists + dists.T) / 2.0
        return dict(
            distance_matrix=pd.DataFrame(
                dists, index=observed_samples.astype(str), columns=observed_samples.astype(str)
            ),
            cell_sample_probs=np.array(cell_sample_probs),
            cell_cell_probs=np.array(cell_scores),
            observed_samples=observed_samples,
        )
