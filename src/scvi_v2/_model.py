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
import numpyro.distributions as db
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

    def get_latent_distributions(
        self,
        adata=None,
        batch_size=256,
    ):
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
            qloc = outs["qu"].loc
            qstd = outs["qu"].scale
            if qloc.ndim == 3:
                qloc = qloc.mean(0)
                qstd = qstd.mean(0)
            return qloc, qstd

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
        return qu_m, qu_std

    def get_propensity_scores(
        self,
        adata=None,
        batch_size=256,
    ):
        """Computes cell overlap probability score square matrix.

        For each pair of cells, the score corresponds to the minimum value alpha s.t.
        n belongs in the alpha-confidence ellipse of np (symmetrized).
        """

        qu_m, qu_std = self.get_latent_distributions(adata=adata, batch_size=batch_size)
        observed_sampleids = adata.obs["_scvi_sample"].unique()

        log_ps = []
        for sample in tqdm(observed_sampleids):
            where_sample = (adata.obs["_scvi_sample"] == sample).values
            n_samples = where_sample.sum()
            uz_m_sample, uz_std_sample = qu_m[where_sample], qu_std[where_sample]
            sections = 1 + (qu_m.shape[0] // 512)
            log_p_sample = []
            for _uz_m in jnp.array_split(qu_m, sections):
                log_p = db.Normal(uz_m_sample[:, None], uz_std_sample[:, None]).log_prob(_uz_m).sum(-1)
                log_p = jax.scipy.special.logsumexp(log_p, axis=0)
                log_p = log_p - jnp.log(n_samples)
                log_p_sample.append(log_p)
            log_p_sample = jnp.concatenate(log_p_sample)[:, None]
            log_ps.append(log_p_sample)
        log_ps = jnp.concatenate(log_ps, 1)
        log_probs = log_ps - jnp.log(1.0 / log_ps.shape[1])
        map_ = pd.Series(observed_sampleids).to_frame("_scvi_sample").reset_index().set_index("_scvi_sample")
        sample_ids = map_.loc[adata.obs["_scvi_sample"].values]["index"].values

        log_probs_orig = log_probs[jnp.arange(log_probs.shape[0]), sample_ids]
        log_denom = jnp.logaddexp(log_probs, log_probs_orig[:, None])

        e_scores = jnp.exp(log_probs - log_denom)
        return e_scores, observed_sampleids, sample_ids

    def get_aggregated_distance_mat(self, adata, batch_size=256, mc_samples: int = 10, proba_threshold=0.0):
        assert adata is not None
        # get uncertainty masks
        e_scores, observed_sampleids, sample_ids = self.get_propensity_scores(
            adata=adata,
            batch_size=batch_size,
        )
        sample_keys = adata.obs.loc[
            :, ["_scvi_sample", self.adata_manager.get_state_registry(MRVI_REGISTRY_KEYS.SAMPLE_KEY)["original_key"]]
        ]
        sample_keys = sample_keys[lambda x: ~x.duplicated(keep="first")].set_index("_scvi_sample").squeeze()
        cpus = jax.devices("cpu")
        observed_samples = sample_keys.loc[observed_sampleids].values

        # local reps
        local_reps = self.get_local_sample_representation(
            adata=adata, batch_size=batch_size, mc_samples=mc_samples, return_distances=False
        )
        local_reps = jax.device_put(local_reps, cpus[0])
        local_reps = local_reps[:, observed_sampleids]

        # weightings
        n_samples = local_reps.shape[1]
        distance_matrix = jnp.zeros((n_samples, n_samples))
        count_matrix = jnp.zeros((n_samples, n_samples))
        for i in tqdm(range(local_reps.shape[0])):
            local_rep_ = local_reps[i]
            e_scores_ = e_scores[i]
            sample_ids_ = sample_ids[i]

            good_d = (e_scores_ <= 1.0 - proba_threshold) & (e_scores_ >= proba_threshold)
            dists = (local_rep_ - local_rep_[sample_ids_]) ** 2
            dists = jnp.sqrt(dists.sum(-1))
            dists = dists * good_d

            new_counts = (jnp.ones_like(dists) * good_d) + count_matrix[sample_ids_]
            new_dists = dists + distance_matrix[sample_ids_]

            distance_matrix = distance_matrix.at[sample_ids_].set(new_dists)
            distance_matrix = distance_matrix.at[:, sample_ids_].set(new_dists)
            count_matrix = count_matrix.at[sample_ids_].set(new_counts)
            count_matrix = count_matrix.at[:, sample_ids_].set(new_counts)

        distance_matrix = distance_matrix / jnp.maximum(1.0, count_matrix)
        distance_matrix = distance_matrix + distance_matrix.T
        distance_matrix = distance_matrix / 2.0
        distance_matrix = np.array(distance_matrix)
        e_scores = np.array(e_scores)
        return dict(
            distance_matrix=pd.DataFrame(
                distance_matrix, index=observed_samples.astype(str), columns=observed_samples.astype(str)
            ),
            e_scores=pd.DataFrame(e_scores, index=observed_samples.astype(str), columns=observed_samples.astype(str)),
            observed_samples=observed_samples,
        )
