from typing import Any, Callable, Dict, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from scvi import REGISTRY_KEYS
from scvi.distributions import JaxNegativeBinomialMeanDisp as NegativeBinomial
from scvi.module.base import JaxBaseModuleClass, LossOutput, flax_configure

from ._components import (
    MLP,
    AttentionBlock,
    ConditionalNormalization,
    Dense,
    FactorizedEmbedding,
    NormalDistOutputNN,
)
from ._constants import MRVI_REGISTRY_KEYS
from ._types import NdArray

DEFAULT_PX_KWARGS = {"n_hidden": 32}
DEFAULT_QZ_KWARGS = {}
DEFAULT_QU_KWARGS = {}

# Lower stddev leads to better initial loss values
_normal_initializer = jax.nn.initializers.normal(stddev=0.1)


class _DecoderZX(nn.Module):
    n_in: int
    n_out: int
    n_batch: int
    n_hidden: int = 128
    activation: Callable = nn.softmax
    dropout_rate: float = 0.1
    training: Optional[bool] = None

    @nn.compact
    def __call__(
        self,
        z: NdArray,
        batch_covariate: NdArray,
        size_factor: NdArray,
        continuous_covariates: Optional[NdArray],
        training: Optional[bool] = None,
    ) -> NegativeBinomial:
        h1 = Dense(self.n_out, use_bias=False, name="amat")(z)
        z_drop = nn.Dropout(self.dropout_rate)(jax.lax.stop_gradient(z), deterministic=not training)
        batch_covariate = batch_covariate.astype(int).flatten()
        # cells by n_out by n_latent (n_in)
        A_b = nn.Embed(self.n_batch, self.n_out * self.n_in, embedding_init=_normal_initializer, name="A_b")(
            batch_covariate
        ).reshape(batch_covariate.shape[0], self.n_out, self.n_in)
        if z_drop.ndim == 3:
            h2 = jnp.einsum("cgl,bcl->bcg", A_b, z_drop)
        else:
            h2 = jnp.einsum("cgl,cl->cg", A_b, z_drop)
        h3 = nn.Embed(self.n_batch, self.n_out, embedding_init=_normal_initializer)(batch_covariate)
        h = h1 + h2 + h3
        if continuous_covariates is not None:
            h4 = Dense(self.n_out, use_bias=False, name="cont_covs_term")(continuous_covariates)
            h += h4
        mu = self.activation(h)
        return NegativeBinomial(
            mean=mu * size_factor, inverse_dispersion=jnp.exp(self.param("px_r", jax.random.normal, (self.n_out,)))
        )


class _DecoderZXAttention(nn.Module):
    n_in: int
    n_out: int
    n_batch: int
    n_latent_sample: int = 16
    h_activation: Callable = nn.softmax
    n_channels: int = 4
    n_heads: int = 2
    dropout_rate: float = 0.1
    stop_gradients: bool = False
    training: Optional[bool] = None
    n_hidden: int = 32
    n_layers: int = 1
    training: Optional[bool] = None
    activation: Callable = nn.gelu

    @nn.compact
    def __call__(
        self,
        z: NdArray,
        batch_covariate: NdArray,
        size_factor: NdArray,
        continuous_covariates: Optional[NdArray],
        training: Optional[bool] = None,
    ) -> NegativeBinomial:

        has_mc_samples = z.ndim == 3
        z_stop = z if not self.stop_gradients else jax.lax.stop_gradient(z)
        z_ = nn.LayerNorm(name="u_ln")(z_stop)

        batch_covariate = batch_covariate.astype(int).flatten()
        batch_embed = nn.Embed(self.n_in, self.n_latent_sample, embedding_init=_normal_initializer)(
            batch_covariate
        )  # (batch, n_latent_sample)
        batch_embed = nn.LayerNorm(name="batch_embed_ln")(batch_embed)
        if has_mc_samples:
            batch_embed = jnp.tile(batch_embed, (z_.shape[0], 1, 1))

        residual = AttentionBlock(
            query_dim=self.n_in,
            out_dim=self.n_out,
            outerprod_dim=self.n_latent_sample,
            n_channels=self.n_channels,
            n_heads=self.n_heads,
            dropout_rate=self.dropout_rate,
            n_hidden_mlp=self.n_hidden,
            n_layers_mlp=self.n_layers,
            training=training,
            activation=self.activation,
        )(query_embed=z_, kv_embed=batch_embed)

        mu = nn.Dense(self.n_out)(z) + residual
        mu = self.h_activation(mu)
        return NegativeBinomial(
            mean=mu * size_factor, inverse_dispersion=jnp.exp(self.param("px_r", jax.random.normal, (self.n_out,)))
        )


class _EncoderUZ(nn.Module):
    n_latent: int
    n_sample: int
    n_latent_u: Optional[int] = None
    use_nonlinear: bool = False
    n_factorized_embed_dims: Optional[int] = None
    dropout_rate: float = 0.0
    training: Optional[bool] = None

    def setup(self):
        self.dropout = nn.Dropout(self.dropout_rate)
        n_latent_u = self.n_latent_u if self.n_latent_u is not None else self.n_latent
        if self.n_latent_u is not None:
            self.A_z = self.param("A_z", jax.random.normal, (self.n_latent, n_latent_u))
        else:
            self.A_z = None
        if not self.use_nonlinear:
            if self.n_factorized_embed_dims is None:
                self.A_s_enc = nn.Embed(
                    self.n_sample, self.n_latent * n_latent_u, embedding_init=_normal_initializer, name="A_s_enc"
                )
            else:
                self.A_s_enc = FactorizedEmbedding(
                    self.n_sample,
                    self.n_latent * n_latent_u,
                    self.n_factorized_embed_dims,
                    embedding_init=_normal_initializer,
                    name="A_s_enc",
                )
        else:
            self.A_s_enc = MLP(self.n_latent * n_latent_u, name="A_s_enc", activation=nn.gelu)
        self.h3_embed = nn.Embed(self.n_sample, self.n_latent, embedding_init=_normal_initializer)

    def __call__(self, u: NdArray, sample_covariate: NdArray, training: Optional[bool] = None) -> jnp.ndarray:
        training = nn.merge_param("training", self.training, training)
        sample_covariate = sample_covariate.astype(int).flatten()
        n_latent_u = self.n_latent_u if self.n_latent_u is not None else self.n_latent
        if not self.use_nonlinear:
            u_drop = self.dropout(jax.lax.stop_gradient(u), deterministic=not training)
            A_s = self.A_s_enc(sample_covariate)
        else:
            # A_s output by a non-linear function without an explicit intercept.
            u_drop = self.dropout(u, deterministic=not training)  # No stop gradient for nonlinear.
            sample_one_hot = jax.nn.one_hot(sample_covariate, self.n_sample)
            A_s_enc_inputs = jnp.concatenate([u_drop, sample_one_hot], axis=-1)
            A_s = self.A_s_enc(A_s_enc_inputs, training=training)
        # cells by n_latent by n_latent
        A_s = A_s.reshape(sample_covariate.shape[0], self.n_latent, n_latent_u)
        if u_drop.ndim == 3:
            h2 = jnp.einsum("cgl,bcl->bcg", A_s, u_drop)
        else:
            h2 = jnp.einsum("cgl,cl->cg", A_s, u_drop)
        h3 = self.h3_embed(sample_covariate)
        delta = h2 + h3
        if self.n_latent_u is not None:
            if u_drop.ndim == 3:
                z_base = jnp.einsum("lu,bcu->bcl", self.A_z, u_drop)
            else:
                z_base = jnp.einsum("lu,cu->cl", self.A_z, u_drop)
            return z_base, delta, A_s
        else:
            return u, delta, A_s


class _EncoderUZ2(nn.Module):
    n_latent: int
    n_sample: int
    n_latent_u: Optional[int] = None
    use_map: bool = False
    n_hidden: int = 32
    n_layers: int = 1
    stop_gradients: bool = False
    training: Optional[bool] = None
    activation: Callable = nn.gelu

    @nn.compact
    def __call__(self, u: NdArray, sample_covariate: NdArray, training: Optional[bool] = None):
        training = nn.merge_param("training", self.training, training)
        sample_covariate = sample_covariate.astype(int).flatten()
        n_latent_u = self.n_latent_u if self.n_latent_u is not None else self.n_latent
        u_stop = u if not self.stop_gradients else jax.lax.stop_gradient(u)

        n_outs = 1 if self.use_map else 2
        sample_oh = jax.nn.one_hot(sample_covariate, self.n_sample)
        if u_stop.ndim == 3:
            sample_oh = jnp.tile(sample_oh, (u_stop.shape[0], 1, 1))
        inputs = jnp.concatenate(
            [u_stop, sample_oh],
            axis=-1,
        )
        eps_ = MLP(
            n_out=n_outs * self.n_latent,
            n_hidden=self.n_hidden,
            n_layers=self.n_layers,
            training=training,
            activation=self.activation,
        )(inputs=inputs)
        A_z = None
        if self.n_latent_u is not None:
            A_z = self.param("A_z", jax.random.normal, (self.n_latent, n_latent_u))
            if u_stop.ndim == 3:
                z_base = jnp.einsum("lu,bcu->bcl", A_z, u_stop)
            else:
                z_base = jnp.einsum("lu,cu->cl", A_z, u_stop)
            return z_base, eps_
        else:
            return u, eps_


class _EncoderUZ2Attention(nn.Module):
    n_latent: int
    n_sample: int
    n_latent_u: Optional[int] = None
    n_latent_sample: int = 16
    n_channels: int = 4
    n_heads: int = 2
    dropout_rate: float = 0.0
    stop_gradients: bool = False
    use_map: bool = True
    n_hidden: int = 32
    n_layers: int = 1
    training: Optional[bool] = None
    activation: Callable = nn.gelu

    @nn.compact
    def __call__(self, u: NdArray, sample_covariate: NdArray, training: Optional[bool] = None):
        training = nn.merge_param("training", self.training, training)
        sample_covariate = sample_covariate.astype(int).flatten()
        n_latent_u = self.n_latent_u if self.n_latent_u is not None else self.n_latent
        has_mc_samples = u.ndim == 3
        u_stop = u if not self.stop_gradients else jax.lax.stop_gradient(u)
        u_ = nn.LayerNorm(name="u_ln")(u_stop)

        sample_embed = nn.Embed(self.n_sample, self.n_latent_sample, embedding_init=_normal_initializer)(
            sample_covariate
        )  # (batch, n_latent_sample)
        sample_embed = nn.LayerNorm(name="sample_embed_ln")(sample_embed)
        if has_mc_samples:
            sample_embed = jnp.tile(sample_embed, (u_.shape[0], 1, 1))

        n_outs = 1 if self.use_map else 2
        residual = AttentionBlock(
            query_dim=self.n_latent,
            out_dim=n_outs * self.n_latent,
            outerprod_dim=self.n_latent_sample,
            n_channels=self.n_channels,
            n_heads=self.n_heads,
            dropout_rate=self.dropout_rate,
            n_hidden_mlp=self.n_hidden,
            n_layers_mlp=self.n_layers,
            training=training,
            activation=self.activation,
        )(query_embed=u_, kv_embed=sample_embed)

        A_z = None
        if self.n_latent_u is not None:
            A_z = self.param("A_z", jax.random.normal, (self.n_latent, n_latent_u))
            if u.ndim == 3:
                z_base = jnp.einsum("lu,bcu->bcl", A_z, u_stop)
            else:
                z_base = jnp.einsum("lu,cu->cl", A_z, u_stop)
            return z_base, residual
        else:
            return u, residual


class _EncoderXU(nn.Module):
    n_latent: int
    n_sample: int
    n_hidden: int
    n_layers: int = 1
    activation: Callable = nn.gelu
    training: Optional[bool] = None

    @nn.compact
    def __call__(self, x: NdArray, sample_covariate: NdArray, training: Optional[bool] = None) -> dist.Normal:
        training = nn.merge_param("training", self.training, training)
        x_feat = jnp.log1p(x)
        for _ in range(2):
            x_feat = Dense(self.n_hidden)(x_feat)
            x_feat = ConditionalNormalization(self.n_hidden, self.n_sample)(x_feat, sample_covariate, training=training)
            x_feat = self.activation(x_feat)
        sample_effect = nn.Embed(self.n_sample, self.n_hidden, embedding_init=_normal_initializer)(
            sample_covariate.squeeze(-1).astype(int)
        )
        inputs = x_feat + sample_effect
        return NormalDistOutputNN(self.n_latent, self.n_hidden, self.n_layers)(inputs, training=training)


@flax_configure
class MrVAE(JaxBaseModuleClass):
    """Flax module for the Multi-resolution Variational Inference (MrVI) model."""

    n_input: int
    n_sample: int
    n_batch: int
    n_continuous_cov: int
    n_latent: int = 20
    n_latent_u: Optional[int] = None
    encoder_n_hidden: int = 128
    encoder_n_layers: int = 2
    z_u_prior: bool = True
    z_u_prior_scale: float = 1.0
    laplace_scale: float = None
    scale_observations: bool = False
    px_nn_flavor: str = "linear"
    qz_nn_flavor: str = "linear"
    px_kwargs: Optional[dict] = None
    qz_kwargs: Optional[dict] = None
    qu_kwargs: Optional[dict] = None
    training: bool = True
    n_obs_per_sample: Optional[jnp.ndarray] = None

    def setup(self):  # noqa: D102
        px_kwargs = DEFAULT_PX_KWARGS.copy()
        if self.px_kwargs is not None:
            px_kwargs.update(self.px_kwargs)
        qz_kwargs = DEFAULT_QZ_KWARGS.copy()
        if self.qz_kwargs is not None:
            qz_kwargs.update(self.qz_kwargs)
        qu_kwargs = DEFAULT_QU_KWARGS.copy()
        if self.qu_kwargs is not None:
            qu_kwargs.update(self.qu_kwargs)

        is_isomorphic_uz = self.n_latent_u is None or self.n_latent == self.n_latent_u
        n_latent_u = None if is_isomorphic_uz else self.n_latent_u

        # Generative model
        if self.px_nn_flavor == "linear":
            px_cls = _DecoderZX
        else:
            px_cls = _DecoderZXAttention
        self.px = px_cls(
            self.n_latent,
            self.n_input,
            self.n_batch,
            **px_kwargs,
        )

        if self.qz_nn_flavor == "attention":
            qz_cls = _EncoderUZ2Attention
        elif self.qz_nn_flavor == "mlp":
            qz_cls = _EncoderUZ2
        elif self.qz_nn_flavor == "linear":
            qz_cls = _EncoderUZ
        else:
            raise ValueError(f"Unknown qz_nn_flavor: {self.qz_nn_flavor}")
        self.qz = qz_cls(
            self.n_latent,
            self.n_sample,
            n_latent_u=n_latent_u,
            **qz_kwargs,
        )

        # Inference model
        self.qu = _EncoderXU(
            n_latent=self.n_latent if is_isomorphic_uz else n_latent_u,
            n_sample=self.n_sample,
            n_hidden=self.encoder_n_hidden,
            n_layers=self.encoder_n_layers,
            **qu_kwargs,
        )

    @property
    def required_rngs(self):  # noqa: D102
        return ("params", "u", "dropout", "eps")

    def _get_inference_input(self, tensors: Dict[str, NdArray]) -> Dict[str, Any]:
        x = tensors[REGISTRY_KEYS.X_KEY]
        sample_index = tensors[MRVI_REGISTRY_KEYS.SAMPLE_KEY]
        return {"x": x, "sample_index": sample_index}

    def inference(self, x, sample_index, mc_samples=None, cf_sample=None, use_mean=False):
        """Latent variable inference."""
        qu = self.qu(x, sample_index, training=self.training)
        if use_mean:
            u = qu.mean
        else:
            u_rng = self.make_rng("u")
            sample_shape = (mc_samples,) if mc_samples is not None else ()
            u = qu.rsample(u_rng, sample_shape=sample_shape)

        sample_index_cf = sample_index if cf_sample is None else cf_sample
        if self.qz_nn_flavor != "linear":
            z_base, eps = self.qz(u, sample_index_cf, training=self.training)
            qeps_ = eps

            qeps = None
            if qeps_.shape[-1] == 2 * self.n_latent:
                loc_, scale_ = qeps_[..., : self.n_latent], qeps_[..., self.n_latent :]
                qeps = dist.Normal(loc_, scale_)
                eps = qeps.mean if use_mean else qeps.rsample(self.make_rng("eps"))
            As = None
            z = z_base + eps
        else:
            z_base, eps, As = self.qz(u, sample_index_cf, training=self.training)
            qeps = None
            z = z_base + eps
        library = jnp.log(x.sum(1, keepdims=True))

        return {
            "qu": qu,
            "qeps": qeps,
            "As": As,
            "u": u,
            "z": z,
            "z_base": z_base,
            "library": library,
        }

    def _get_generative_input(self, tensors: Dict[str, NdArray], inference_outputs: Dict[str, Any]) -> Dict[str, Any]:
        z = inference_outputs["z"]
        library = inference_outputs["library"]
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
        continuous_covs = tensors.get(REGISTRY_KEYS.CONT_COVS_KEY, None)
        return {"z": z, "library": library, "batch_index": batch_index, "continuous_covs": continuous_covs}

    def generative(self, z, library, batch_index, continuous_covs):
        """Generative model."""
        library_exp = jnp.exp(library)
        px = self.px(
            z, batch_index, size_factor=library_exp, continuous_covariates=continuous_covs, training=self.training
        )
        h = px.mean / library_exp

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
        if self.qz_nn_flavor != "linear":
            qeps = inference_outputs["qeps"]
            eps = inference_outputs["z"] - inference_outputs["z_base"]
            peps = dist.Normal(0, self.z_u_prior_scale)
            kl_z = dist.kl_divergence(qeps, peps).sum(-1) if qeps is not None else -peps.log_prob(eps).sum(-1)
        else:
            kl_z = (
                -dist.Normal(inference_outputs["z_base"], self.z_u_prior_scale).log_prob(inference_outputs["z"]).sum(-1)
                if self.z_u_prior_scale is not None
                else 0
            )

        weighted_kl_local = kl_weight * (kl_u + kl_z)
        loss = reconstruction_loss + weighted_kl_local

        if self.laplace_scale is not None:
            As = inference_outputs["As"]
            n_obs = As.shape[0]
            As = As.reshape((n_obs, -1))
            p_As = dist.Laplace(0, self.laplace_scale).log_prob(As).sum(-1)

            n_obs_total = self.n_obs_per_sample.sum()
            As_pen = -p_As / n_obs_total
            As_pen = As_pen.sum()
            loss = loss + (kl_weight * As_pen)

        if self.scale_observations:
            sample_index = tensors[MRVI_REGISTRY_KEYS.SAMPLE_KEY].flatten().astype(int)
            prefactors = self.n_obs_per_sample[sample_index]
            loss = loss / prefactors

        loss = jnp.mean(loss)

        return LossOutput(
            loss=loss,
            reconstruction_loss=reconstruction_loss,
            kl_local=(kl_u + kl_z),
        )

    def compute_h_from_x(self, x, sample_index, batch_index, cf_sample=None, continuous_covs=None, mc_samples=10):
        """Compute normalized gene expression from observations"""
        library = 7.0 * jnp.ones_like(sample_index)  # placeholder, has no effect on the value of h.
        inference_outputs = self.inference(x, sample_index, mc_samples=mc_samples, cf_sample=cf_sample, use_mean=False)
        generative_inputs = {
            "z": inference_outputs["z"],
            "library": library,
            "batch_index": batch_index,
            "continuous_covs": continuous_covs,
        }
        generative_outputs = self.generative(**generative_inputs)
        return generative_outputs["h"]
