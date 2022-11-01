from typing import Any, Dict, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from scvi import REGISTRY_KEYS
from scvi.distributions import JaxNegativeBinomialMeanDisp as NegativeBinomial
from scvi.module.base import JaxBaseModuleClass, LossOutput, flax_configure

from ._components import ConditionalBatchNorm1d, Dense, NormalNN
from ._constants import MRVI_REGISTRY_KEYS
from ._types import NdArray

DEFAULT_PX_KWARGS = {"n_hidden": 32}
DEFAULT_PZ_KWARGS = {"scaler_n_hidden": 32}


class _DecoderZX(nn.Module):

    n_in: int
    n_out: int
    n_nuisance: int
    n_hidden: int = 128
    activation: str = "softmax"
    training: Optional[bool] = None

    def setup(self):
        if self.activation == "softmax":
            self.activation_fn = nn.softmax
        elif self.activation == "softplus":
            self.activation_fn = nn.softplus
        elif self.activation == "exp":
            self.activation_fn = jnp.exp
        else:
            raise ValueError(f"Invalid activation {self.activation}")

        self.n_latent = self.n_in - self.n_nuisance
        self.amat = Dense(self.n_out, use_bias=False)
        self.amat_site = self.param("amat_site", jax.random.normal, (self.n_nuisance, self.n_latent, self.n_out))
        self.offsets = self.param("zx_offsets", jax.random.normal, (self.n_nuisance, self.n_out))
        self.dropout_ = nn.Dropout(0.1)
        self.px_r = self.param("px_r", jax.random.normal, (self.n_out,))

    def __call__(self, z: NdArray, size_factor: NdArray, training: Optional[bool] = None) -> NegativeBinomial:
        training = nn.merge_param("training", self.training, training)
        nuisance_oh = z[..., -self.n_nuisance :]
        z0 = z[..., : -self.n_nuisance]
        x1 = self.amat(z0)

        nuisance_ids = jnp.argmax(nuisance_oh, axis=-1)
        As = self.amat_site[nuisance_ids]
        z0_stop_grad = self.dropout_(jax.lax.stop_gradient(z0), deterministic=not training)[..., None]
        x2 = (As * z0_stop_grad).sum(-2)
        offsets = self.offsets[nuisance_ids]
        mu = x1 + x2 + offsets
        mu = self.activation_fn(mu)
        mu = mu * size_factor
        return NegativeBinomial(mean=mu, inverse_dispersion=jnp.exp(self.px_r))


class _DecoderUZ(nn.Module):

    n_latent: int
    n_sample: int
    n_out: int
    use_scaler: bool = False
    scaler_n_hidden: int = 32
    training: Optional[bool] = None

    def setup(self):
        self.amat_sample = self.param("amat_sample", jax.random.normal, (self.n_sample, self.n_latent, self.n_out))
        self.offsets = self.param("uz_offsets", jax.random.normal, (self.n_sample, self.n_out))

        self.scaler = None
        if self.use_scaler:
            self.scaler = nn.Sequential(
                [
                    Dense(self.scaler_n_hidden),
                    nn.LayerNorm(),
                    nn.relu,
                    Dense(1),
                    nn.sigmoid,
                ]
            )

    def __call__(self, u: NdArray, sample_index: NdArray, training: Optional[bool] = None) -> jnp.ndarray:
        training = nn.merge_param("training", self.training, training)
        sample_index_ = sample_index.squeeze().astype(int)
        As = self.amat_sample[sample_index_]

        u_stop_grad = jax.lax.stop_gradient(u)[..., None]
        z2 = (As * u_stop_grad).sum(-2)
        offsets = self.offsets[sample_index_]
        delta = z2 + offsets
        if self.scaler is not None:
            sample_oh = jax.nn.one_hot(sample_index, self.n_sample)
            if u.ndim != sample_oh.ndim:
                sample_oh = jax.lax.broadcast(sample_oh, (u.shape[0], ))
            inputs = jnp.concatenate([jax.lax.stop_gradient(u), sample_oh], axis=-1)
            delta = delta * self.scaler(inputs)
        return u + delta


@flax_configure
class MrVAE(JaxBaseModuleClass):
    """Flax module for the Multi-resolution Variational Inference (MrVI) model."""

    n_input: int
    n_sample: int
    n_obs_per_sample: NdArray
    n_cats_per_nuisance_keys: NdArray
    n_latent: int = 10
    n_latent_sample: int = 2
    uz_scaler: bool = False
    encoder_n_hidden: int = 128
    px_kwargs: Optional[dict] = None
    pz_kwargs: Optional[dict] = None
    training: bool = True

    def setup(self): # noqa: D102
        px_kwargs = DEFAULT_PX_KWARGS.copy()
        if self.px_kwargs is not None:
            px_kwargs.update(self.px_kwargs)
        pz_kwargs = DEFAULT_PZ_KWARGS.copy()
        if self.pz_kwargs is not None:
            pz_kwargs.update(self.pz_kwargs)

        assert self.n_latent_sample != 0
        self.sample_embeddings = nn.Embed(self.n_sample, self.n_latent_sample)
        n_nuisance = sum(self.n_cats_per_nuisance_keys)

        # Generative model
        self.px = _DecoderZX(
            self.n_latent + n_nuisance,
            self.n_input,
            n_nuisance=n_nuisance,
            **px_kwargs,
        )
        self.pz = _DecoderUZ(
            self.n_latent,
            self.n_sample,
            self.n_latent,
            use_scaler=self.uz_scaler,
            **pz_kwargs,
        )

        # Inference model
        self.x_featurizer = nn.Sequential([Dense(self.encoder_n_hidden), nn.relu])
        self.bnn = ConditionalBatchNorm1d(self.encoder_n_hidden, self.n_sample)
        self.x_featurizer2 = nn.Sequential([Dense(self.encoder_n_hidden), nn.relu])
        self.bnn2 = ConditionalBatchNorm1d(self.encoder_n_hidden, self.n_sample)
        self.qu = NormalNN(self.encoder_n_hidden + self.n_latent_sample, self.n_latent, n_categories=1)

    @property
    def required_rngs(self):  # noqa: D102
        return ("params", "u", "dropout")

    def _get_inference_input(self, tensors: Dict[str, NdArray]) -> Dict[str, Any]:
        x = tensors[REGISTRY_KEYS.X_KEY]
        sample_index = tensors[MRVI_REGISTRY_KEYS.SAMPLE_KEY]
        categorical_nuisance_keys = tensors[MRVI_REGISTRY_KEYS.CATEGORICAL_NUISANCE_KEYS]
        return {"x": x, "sample_index": sample_index, "categorical_nuisance_keys": categorical_nuisance_keys}

    def inference(self, x, sample_index, categorical_nuisance_keys, mc_samples=1, cf_sample=None, use_mean=False):
        x_ = jnp.log1p(x)

        sample_index_cf = sample_index if cf_sample is None else cf_sample
        zsample = self.sample_embeddings(sample_index_cf.squeeze(-1).astype(int))
        zsample_ = zsample
        if mc_samples >= 2:
            zsample_ = jax.lax.broadcast(zsample, (mc_samples,))

        nuisance_oh = []
        for dim in range(categorical_nuisance_keys.shape[-1]):
            nuisance_oh.append(
                jax.nn.one_hot(
                    categorical_nuisance_keys[:, dim],
                    self.n_cats_per_nuisance_keys[dim],
                )
            )
        nuisance_oh = jnp.concatenate(nuisance_oh, axis=-1)

        x_feat = self.x_featurizer(x_)
        x_feat = self.bnn(x_feat, sample_index, training=self.training)
        x_feat = self.x_featurizer2(x_feat)
        x_feat = self.bnn2(x_feat, sample_index, training=self.training)
        if x_.ndim != zsample_.ndim:
            x_feat_ = jax.lax.broadcast(x_feat, (mc_samples, ))
            nuisance_oh = jax.lax.broadcast(nuisance_oh, (mc_samples, ))
        else:
            x_feat_ = x_feat

        inputs = jnp.concatenate([x_feat_, zsample_], axis=-1)
        qu = self.qu(inputs, training=self.training)
        if use_mean:
            u = qu.loc
        else:
            u_rng = self.make_rng("u")
            u = qu.rsample(u_rng)

        z = self.pz(u, sample_index_cf, training=self.training)
        library = jnp.expand_dims(jnp.log(x.sum(1)), 1)

        return {
            "qu": qu,
            "u": u,
            "z": z,
            "zsample": zsample,
            "library": library,
            "nuisance_oh": nuisance_oh,
        }

    def _get_generative_input(self, tensors: Dict[str, NdArray], inference_outputs: Dict[str, Any]) -> Dict[str, Any]:
        z = inference_outputs["z"]
        library = inference_outputs["library"]
        nuisance_oh = inference_outputs["nuisance_oh"]
        return {"z": z, "library": library, "nuisance_oh": nuisance_oh}

    def generative(self, z, library, nuisance_oh):
        inputs = jnp.concatenate([z, nuisance_oh], axis=-1)
        px = self.px(inputs, size_factor=jnp.exp(library), training=self.training)
        h = px.mean / jnp.exp(library)

        pu = dist.Normal(0, 1)
        return {"px": px, "pu": pu, "h": h}

    def loss(
        self,
        tensors: Dict[str, NdArray],
        inference_outputs: Dict[str, Any],
        generative_outputs: Dict[str, Any],
        kl_weight: float = 1.0,
    ) -> jnp.ndarray:

        reconstruction_loss = -generative_outputs["px"].log_prob(tensors[REGISTRY_KEYS.X_KEY]).sum(-1)
        kl_u = dist.kl_divergence(inference_outputs["qu"], generative_outputs["pu"]).sum(-1)

        weighted_kl_local = kl_weight * kl_u
        loss = jnp.mean(reconstruction_loss + weighted_kl_local)

        return LossOutput(
            loss=loss,
            reconstruction_loss=reconstruction_loss,
            kl_local=kl_u,
        )
