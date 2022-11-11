from typing import Any, Dict, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from flax.linen.initializers import variance_scaling
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
    n_hidden: int = 128
    activation: str = "softplus"
    dropout_rate: float = 0.1
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

        self.amat = Dense(self.n_out, use_bias=False, name="amat")
        self.amat_site = nn.DenseGeneral(
            (self.n_out, self.n_in),
            use_bias=False,
            kernel_init=variance_scaling(1 / 3, "fan_in", "uniform"),
            name="amat_site",
        )
        self.offset = Dense(self.n_out)
        self.z_dropout = nn.Dropout(self.dropout_rate)
        self.px_r = self.param("px_r", jax.random.normal, (self.n_out,))

    def __call__(
        self, z: NdArray, nuisance_oh: NdArray, size_factor: NdArray, training: Optional[bool] = None
    ) -> NegativeBinomial:
        training = nn.merge_param("training", self.training, training)
        h1 = self.amat(z)
        z_drop = self.z_dropout(jax.lax.stop_gradient(z), deterministic=not training)
        # cells by n_out by n_latent (n_in)
        A_b = self.amat_site(nuisance_oh)
        h2 = jnp.einsum("cgl,cl->cg", A_b, z_drop)
        h3 = self.offset(nuisance_oh)
        mu = self.activation_fn(h1 + h2 + h3)
        return NegativeBinomial(mean=mu * size_factor, inverse_dispersion=jnp.exp(self.px_r))


class _DecoderUZ(nn.Module):

    n_latent: int
    dropout_rate: float = 0.0
    training: Optional[bool] = None

    def setup(self):
        self.amat_sample = nn.DenseGeneral(
            (self.n_latent, self.n_latent),
            use_bias=False,
            kernel_init=variance_scaling(1 / 3, "fan_in", "uniform"),
            name="amat_sample",
        )
        self.dropout_u = nn.Dropout(self.dropout_rate)
        self.offset = Dense(self.n_latent)

    def __call__(self, u: NdArray, sample_covariate: NdArray, training: Optional[bool] = None) -> jnp.ndarray:
        training = nn.merge_param("training", self.training, training)
        u_drop = self.dropout_u(jax.lax.stop_gradient(u), deterministic=not training)
        # cells by n_out by n_latent
        A_s = self.amat_sample(sample_covariate)
        if u_drop.ndim == 3:
            h2 = jnp.einsum("cgl,bcl->bcg", A_s, u_drop)
        else:
            h2 = jnp.einsum("cgl,cl->cg", A_s, u_drop)
        h3 = self.offset(sample_covariate)
        delta = h2 + h3
        return u + delta


class _EncoderXU(nn.Module):

    n_latent: int
    n_sample: int
    n_latent_sample: int
    n_hidden: int
    training: Optional[bool] = None

    def setup(self):
        self.sample_embeddings = nn.Embed(
            self.n_sample, self.n_latent_sample, embedding_init=jax.nn.initializers.normal()
        )
        self.x_featurizer = nn.Sequential([Dense(self.n_hidden), nn.relu])
        self.bnn = ConditionalBatchNorm1d(self.n_hidden, self.n_sample)
        self.x_featurizer2 = nn.Sequential([Dense(self.n_hidden), nn.relu])
        self.bnn2 = ConditionalBatchNorm1d(self.n_hidden, self.n_sample)
        self.qu = NormalDistOutputNN(self.n_hidden + self.n_latent_sample, self.n_latent)

    def __call__(
        self, x: NdArray, sample_covariate: NdArray, mc_samples: int = 1, training: Optional[bool] = None
    ) -> dist.Normal:
        training = nn.merge_param("training", self.training, training)
        x_ = jnp.log1p(x)
        zsample = self.sample_embeddings(sample_covariate.squeeze(-1).astype(int))
        zsample_ = zsample
        if mc_samples >= 2:
            zsample_ = jax.lax.broadcast(zsample, (mc_samples,))

        x_feat = self.x_featurizer(x_)
        x_feat = self.bnn(x_feat, sample_covariate, training=training)
        x_feat = self.x_featurizer2(x_feat)
        x_feat = self.bnn2(x_feat, sample_covariate, training=training)
        if mc_samples >= 2:
            x_feat = jax.lax.broadcast(x_feat, (mc_samples,))

        inputs = jnp.concatenate([x_feat, zsample_], axis=-1)
        return self.qu(inputs, training=training)


@flax_configure
class MrVAE(JaxBaseModuleClass):
    """Flax module for the Multi-resolution Variational Inference (MrVI) model."""

    n_input: int
    n_sample: int
    n_obs_per_sample: NdArray
    n_cats_per_nuisance_keys: NdArray
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
            **px_kwargs,
        )
        self.pz = _DecoderUZ(
            self.n_latent,
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
        qu = self.qu(x, sample_index, mc_samples=mc_samples, training=self.training)
        if use_mean:
            u = qu.mean
        else:
            u_rng = self.make_rng("u")
            u = qu.rsample(u_rng)

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
        categorical_nuisance_keys = tensors[MRVI_REGISTRY_KEYS.CATEGORICAL_NUISANCE_KEYS]
        return {"z": z, "library": library, "categorical_nuisance_keys": categorical_nuisance_keys}

    def generative(self, z, library, categorical_nuisance_keys):
        """Generative model."""
        nuisance_oh = []
        for dim in range(categorical_nuisance_keys.shape[-1]):
            nuisance_oh.append(
                jax.nn.one_hot(
                    categorical_nuisance_keys[:, dim],
                    self.n_cats_per_nuisance_keys[dim],
                )
            )
        nuisance_oh = jnp.concatenate(nuisance_oh, axis=-1)
        if z.ndim != nuisance_oh.ndim:
            nuisance_oh = jax.lax.broadcast(nuisance_oh, (z.shape[0],))

        px = self.px(z, nuisance_oh, size_factor=jnp.exp(library), training=self.training)
        h = px.mean / jnp.exp(library)

        pu = dist.Normal(0, 1)
        return {"px": px, "pu": pu, "h": h, "nuisance_oh": nuisance_oh}

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
