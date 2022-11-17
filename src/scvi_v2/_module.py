from typing import Any, Callable, Dict, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from scvi import REGISTRY_KEYS
from scvi.distributions import JaxNegativeBinomialMeanDisp as NegativeBinomial
from scvi.module.base import JaxBaseModuleClass, LossOutput, flax_configure

from ._components import ConditionalBatchNorm1d, Dense, NormalDistOutputNN
from ._constants import MRVI_REGISTRY_KEYS
from ._types import NdArray

DEFAULT_PX_KWARGS = {"n_hidden": 32}
DEFAULT_PZ_KWARGS = {}
DEFAULT_QU_KWARGS = {}


class _DecoderZX(nn.Module):

    n_in: int
    n_out: int
    n_batch: int
    n_hidden: int = 128
    activation: Callable = nn.softplus
    dropout_rate: float = 0.1
    training: Optional[bool] = None

    @nn.compact
    def __call__(
        self, z: NdArray, batch_covariate: NdArray, size_factor: NdArray, training: Optional[bool] = None
    ) -> NegativeBinomial:
        h1 = Dense(self.n_out, use_bias=False, name="amat")(z)
        z_drop = nn.Dropout(self.dropout_rate)(jax.lax.stop_gradient(z), deterministic=not training)
        batch_covariate = batch_covariate.astype(int).flatten()
        # cells by n_out by n_latent (n_in)
        A_b = nn.Embed(self.n_batch, self.n_out * self.n_in)(batch_covariate).reshape(
            batch_covariate.shape[0], self.n_out, self.n_in
        )
        if z_drop.ndim == 3:
            h2 = jnp.einsum("cgl,bcl->bcg", A_b, z_drop)
        else:
            h2 = jnp.einsum("cgl,cl->cg", A_b, z_drop)
        h3 = nn.Embed(self.n_batch, self.n_out)(batch_covariate)
        mu = self.activation(h1 + h2 + h3)
        return NegativeBinomial(
            mean=mu * size_factor, inverse_dispersion=jnp.exp(self.param("px_r", jax.random.normal, (self.n_out,)))
        )


class _DecoderUZ(nn.Module):

    n_latent: int
    n_sample: int
    dropout_rate: float = 0.0
    training: Optional[bool] = None

    @nn.compact
    def __call__(self, u: NdArray, sample_covariate: NdArray, training: Optional[bool] = None) -> jnp.ndarray:
        training = nn.merge_param("training", self.training, training)
        u_drop = nn.Dropout(self.dropout_rate)(jax.lax.stop_gradient(u), deterministic=not training)
        sample_covariate = sample_covariate.astype(int).flatten()
        # cells by n_latent by n_latent
        A_s = nn.Embed(self.n_sample, self.n_latent * self.n_latent)(sample_covariate).reshape(
            sample_covariate.shape[0], self.n_latent, self.n_latent
        )
        if u_drop.ndim == 3:
            h2 = jnp.einsum("cgl,bcl->bcg", A_s, u_drop)
        else:
            h2 = jnp.einsum("cgl,cl->cg", A_s, u_drop)
        h3 = nn.Embed(self.n_sample, self.n_latent)(sample_covariate)
        delta = h2 + h3
        return u + delta


class _EncoderXU(nn.Module):

    n_latent: int
    n_sample: int
    n_latent_sample: int
    n_hidden: int
    training: Optional[bool] = None

    @nn.compact
    def __call__(self, x: NdArray, sample_covariate: NdArray, training: Optional[bool] = None) -> dist.Normal:
        training = nn.merge_param("training", self.training, training)
        x_ = jnp.log1p(x)
        zsample = nn.Embed(self.n_sample, self.n_latent_sample, embedding_init=jax.nn.initializers.normal())(
            sample_covariate.squeeze(-1).astype(int)
        )
        zsample_ = zsample

        x_feat = nn.Sequential([Dense(self.n_hidden), nn.relu])(x_)
        x_feat = ConditionalBatchNorm1d(self.n_hidden, self.n_sample)(x_feat, sample_covariate, training=training)
        x_feat = nn.Sequential([Dense(self.n_hidden), nn.relu])(x_feat)
        x_feat = ConditionalBatchNorm1d(self.n_hidden, self.n_sample)(x_feat, sample_covariate, training=training)

        inputs = jnp.concatenate([x_feat, zsample_], axis=-1)
        return NormalDistOutputNN(self.n_hidden + self.n_latent_sample, self.n_latent)(inputs, training=training)


@flax_configure
class MrVAE(JaxBaseModuleClass):
    """Flax module for the Multi-resolution Variational Inference (MrVI) model."""

    n_input: int
    n_sample: int
    n_batch: int
    n_latent: int = 10
    n_latent_sample: int = 2
    encoder_n_hidden: int = 128
    px_kwargs: Optional[dict] = None
    pz_kwargs: Optional[dict] = None
    qu_kwargs: Optional[dict] = None
    training: bool = True

    def setup(self):  # noqa: D102
        px_kwargs = DEFAULT_PX_KWARGS.copy()
        if self.px_kwargs is not None:
            px_kwargs.update(self.px_kwargs)
        pz_kwargs = DEFAULT_PZ_KWARGS.copy()
        if self.pz_kwargs is not None:
            pz_kwargs.update(self.pz_kwargs)
        qu_kwargs = DEFAULT_QU_KWARGS.copy()
        if self.qu_kwargs is not None:
            qu_kwargs.update(self.qu_kwargs)

        assert self.n_latent_sample != 0

        # Generative model
        self.px = _DecoderZX(
            self.n_latent,
            self.n_input,
            self.n_batch,
            **px_kwargs,
        )
        self.pz = _DecoderUZ(
            self.n_latent,
            self.n_sample,
            **pz_kwargs,
        )

        # Inference model
        self.qu = _EncoderXU(
            self.n_latent,
            self.n_sample,
            self.n_latent_sample,
            self.encoder_n_hidden,
            **qu_kwargs,
        )

    @property
    def required_rngs(self):  # noqa: D102
        return ("params", "u", "dropout")

    def _get_inference_input(self, tensors: Dict[str, NdArray]) -> Dict[str, Any]:
        x = tensors[REGISTRY_KEYS.X_KEY]
        sample_index = tensors[MRVI_REGISTRY_KEYS.SAMPLE_KEY]
        return {"x": x, "sample_index": sample_index}

    def inference(self, x, sample_index, mc_samples=1, cf_sample=None, use_mean=False):
        """Latent variable inference."""
        qu = self.qu(x, sample_index, training=self.training)
        if use_mean:
            u = qu.mean
        else:
            u_rng = self.make_rng("u")
            u = qu.rsample(u_rng, sample_shape=(mc_samples,))

        sample_index_cf = sample_index if cf_sample is None else cf_sample
        z = self.pz(u, sample_index_cf, training=self.training)
        library = jnp.expand_dims(jnp.log(x.sum(1)), 1)

        return {
            "qu": qu,
            "u": u,
            "z": z,
            "library": library,
        }

    def _get_generative_input(self, tensors: Dict[str, NdArray], inference_outputs: Dict[str, Any]) -> Dict[str, Any]:
        z = inference_outputs["z"]
        library = inference_outputs["library"]
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
        return {"z": z, "library": library, "batch_index": batch_index}

    def generative(self, z, library, batch_index):
        """Generative model."""
        px = self.px(z, batch_index, size_factor=jnp.exp(library), training=self.training)
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
        """Compute the loss function value."""
        reconstruction_loss = -generative_outputs["px"].log_prob(tensors[REGISTRY_KEYS.X_KEY]).sum(-1)
        kl_u = dist.kl_divergence(inference_outputs["qu"], generative_outputs["pu"]).sum(-1)

        weighted_kl_local = kl_weight * kl_u
        loss = jnp.mean(reconstruction_loss + weighted_kl_local)

        return LossOutput(
            loss=loss,
            reconstruction_loss=reconstruction_loss,
            kl_local=kl_u,
        )
