from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from scvi import REGISTRY_KEYS
from scvi.distributions import JaxNegativeBinomialMeanDisp as NegativeBinomial
from scvi.module.base import JaxBaseModuleClass, LossRecorder
from typings import Dict, Optional

from ._components import ConditionalBatchNorm1d, Dense, NormalNN
from ._constants import MRVI_REGISTRY_KEYS
from ._types import NdArray

DEFAULT_PX_KWARGS = {"n_hidden": 32}
DEFAULT_PZ_KWARGS = {}


class DecoderZX(nn.Module):

    n_int: int
    n_out: int
    n_nuisance: int
    n_hidden: int = 128
    activation: str = "softmax"

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
        self.amat = Dense(self.n_latent, self.n_out, bias=False)
        self.amat_site = self.param("amat_site", jax.random.normal, (self.n_nuisance, self.n_latent, self.n_out))
        self.offsets = self.param("zx_offsets", jax.random.normal, (self.n_nuisance, self.n_out))
        self.dropout_ = nn.Dropout(0.1)
        self.px_r = self.param("px_r", jax.random.normal, self.n_out)

    def __call__(self, z: NdArray, size_factor: NdArray, training: bool = False) -> NegativeBinomial:
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
        return NegativeBinomial(mu=mu, theta=self.px_r.exp())


class DecoderUZ(nn.Module):

    n_latent: int
    n_sample: int
    n_out: int
    use_scaler: bool = False
    scaler_n_hidden: int = 32

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
                    nn.Dense(1),
                    nn.sigmoid,
                ]
            )

    def __call__(self, u: NdArray, sample_index: NdArray, training: bool = False) -> jnp.ndarray:
        sample_index_ = sample_index.squeeze()
        As = self.amat_sample[sample_index_]

        u_detach = u.detach()[..., None]
        z2 = (As * u_detach).sum(-2)
        offsets = self.offsets[sample_index_]
        delta = z2 + offsets
        if self.scaler is not None:
            sample_oh = jax.nn.one_hot(sample_index, self.n_sample)
            if u.ndim != sample_oh.ndim:
                sample_oh = jnp.broadcast_to(sample_oh[None], (u.shape[0], *sample_oh.shape))
            inputs = jnp.concatenate([u.detach(), sample_oh], axis=-1)
            delta = delta * self.scaler(inputs)
        return u + delta


class MrVAE(JaxBaseModuleClass):
    """Flax module for the Multi-resolution Variational Inference (MrVI) model."""

    n_input: int
    n_sample: int
    n_obs_per_sample: NdArray
    n_cats_per_nuisance_keys: NdArray
    n_latent: int = 10
    n_latent_sample: int = 10
    uz_scaler: bool = False
    uz_scaler_n_hidden: int = 32
    encoder_n_hidden: int = 128
    px_kwargs: Optional[dict] = None
    pz_kwargs: Optional[dict] = None

    def setup(self):
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
        self.px = DecoderZX(
            self.n_latent + n_nuisance,
            self.n_input,
            n_nuisance=n_nuisance,
            **px_kwargs,
        )
        self.pz = DecoderUZ(
            self.n_latent,
            self.n_sample,
            self.n_latent,
            scaler=self.uz_scaler,
            scaler_n_hidden=self.uz_scaler_n_hidden,
            **pz_kwargs,
        )

        # Inference model
        self.x_featurizer = nn.Sequential([Dense(self.encoder_n_hidden), nn.relu()])
        self.bnn = ConditionalBatchNorm1d(self.encoder_n_hidden, self.n_sample)
        self.x_featurizer2 = nn.Sequential([Dense(self.encoder_n_hidden), nn.relu()])
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
        zsample = self.sample_embeddings(sample_index_cf.long().squeeze(-1))
        zsample_ = zsample
        if mc_samples >= 2:
            zsample_ = zsample[None].expand(mc_samples, *zsample.shape)

        nuisance_oh = []
        for dim in range(categorical_nuisance_keys.shape[-1]):
            nuisance_oh.append(
                jax.nn.one_hot(
                    categorical_nuisance_keys[:, [dim]],
                    self.n_cats_per_nuisance_keys[dim],
                )
            )
        nuisance_oh = jnp.concatenate(nuisance_oh, axis=-1)

        x_feat = self.x_featurizer(x_)
        x_feat = self.bnn(x_feat, sample_index)
        x_feat = self.x_featurizer2(x_feat)
        x_feat = self.bnn2(x_feat, sample_index)
        if x_.ndim != zsample_.ndim:
            x_feat_ = jnp.broadcast_to(x_feat[None], (mc_samples, *x_feat.shape))
            nuisance_oh = jnp.broadcast_to(nuisance_oh[None], (mc_samples, *nuisance_oh.shape))
        else:
            x_feat_ = x_feat

        inputs = jnp.concatenate([x_feat_, zsample_], axis=-1)
        # inputs = x_feat_
        qu = self.qu(inputs)
        if use_mean:
            u = qu.loc
        else:
            u_rng = self.make_rng("u")
            u = qu.rsample(u_rng)

        if self.linear_decoder_uz:
            z = self.pz(u, sample_index_cf)
        else:
            inputs = jnp.concatenate([u, zsample_], axis=-1)
            z = self.pz(inputs)
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
        px = self.px(inputs, size_factor=library.exp())
        h = px.loc / library.exp()

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
        kl_u = inference_outputs["qu"].kl_divergence(generative_outputs["pu"]).sum(-1)
        kl_local_for_warmup = kl_u

        weighted_kl_local = kl_weight * kl_local_for_warmup
        loss = jnp.mean(reconstruction_loss + weighted_kl_local)

        kl_local = jnp.array(0.0)
        kl_global = jnp.array(0.0)
        return LossRecorder(
            loss,
            reconstruction_loss,
            kl_local,
            kl_global,
        )
