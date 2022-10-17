from typing import Any, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from flax.linen.initializers import variance_scaling

from ._types import NdArray


class Dense(nn.Dense):
    """Jax dense layer."""

    def __init__(self, *args, **kwargs):
        # scale set to reimplement pytorch init
        scale = 1 / 3
        kernel_init = variance_scaling(scale, "fan_in", "uniform")
        # bias init can't see input shape so don't include here
        kwargs.update({"kernel_init": kernel_init})
        super().__init__(*args, **kwargs)


class ResnetFC(nn.Module):

    n_in: int
    n_out: int
    n_hidden: int = 128
    activation: str = "softmax"

    def setup(self):
        self.module1 = nn.Sequential(
            [
                nn.Dense(self.n_hidden),
                nn.BatchNorm(),
                nn.relu,
            ]
        )
        self.module2 = nn.Sequential(
            [
                nn.Dense(self.n_out),
                nn.BatchNorm(),
            ]
        )
        if self.n_in != self.n_hidden:
            self.id_map1 = nn.Dense(self.n_hidden)
        else:
            self.id_map1 = None

        self.activation_fn = None
        if self.activation == "softmax":
            self.activation_fn = nn.softmax
        elif self.activation == "relu":
            self.activation_fn = nn.relu
        else:
            raise ValueError(f"Invalid activation {self.activation}")

    def __call__(self, inputs: NdArray, training: bool = False) -> NdArray:
        need_reshaping = False
        if inputs.ndim == 3:
            n_d1, nd2 = inputs.shape[:2]
            inputs = inputs.reshape(n_d1 * nd2, -1)
            need_reshaping = True
        h = self.module1(inputs)
        if self.id_map1 is not None:
            h = h + self.id_map1(inputs)
        h = self.module2(h)
        if need_reshaping:
            h = h.view(n_d1, nd2, -1)
        if self.activation_fn is not None:
            return self.activation_fn(h)
        return h


class _NormalNN(nn.Module):

    n_in: int
    n_out: int
    n_hidden: int = 128
    n_layers: int = 1

    def setup(self):
        self.hidden = ResnetFC(n_in=self.n_in, n_out=self.n_hidden, activation="relu")
        self._mean = nn.Dense(self.n_out)
        self._var = nn.Sequential([nn.Dense(self.n_out), nn.softplus])

    def __call__(self, inputs: NdArray, training: bool = False) -> dist.Normal:
        if self.n_layers >= 1:
            h = self.hidden(inputs)
            mean = self._mean(h)
            var = self._var(h)
        else:
            mean = self._mean(inputs)
            k = mean.shape[0]
            var = self._var[None].expand(k, -1)
        return dist.Normal(mean, var)


class NormalNN(nn.Module):
    n_in: int
    n_out: int
    n_categories: int
    n_hidden: int = 128
    n_layers: int = 1
    eps: float = 1e-5

    def setup(self):
        nn_kwargs = {
            "n_in": self.n_in,
            "n_out": self.n_out,
            "n_hidden": self.n_hidden,
            "n_layers": self.n_layers,
        }
        self.cat_modules = [_NormalNN(**nn_kwargs) for _ in range(self.n_categories)]

    def __call__(self, inputs: NdArray, categories: Optional[NdArray] = None, training: bool = False) -> dist.Normal:
        means = []
        vars = []
        for module in self.cat_modules:
            _means, _vars = module(inputs)
            means.append(_means[..., None])
            vars.append(_vars[..., None])
        means = jnp.concatenate(means, axis=-1)
        vars = jnp.concatenate(vars, axis=-1)
        if categories is not None:
            # categories (minibatch, 1)
            n_cats = categories.shape[0]
            cat_ = categories.unsqueeze(-1).expand(n_cats, self.n_out, 1)
            if means.ndim == 4:
                d1, n_cats, _, _ = means.shape
                cat_ = categories[None, :, None].expand(d1, n_cats, self.n_out, 1)
            means = jnp.take_along_axis(means, cat_, -1)
            vars = jnp.take_along_axis(vars, cat_, -1)
        means = means.squeeze(-1)
        vars = vars.squeeze(-1)
        return dist.Normal(means, vars + self.eps)


class ConditionalBatchNorm1d(nn.Module):
    num_features: int
    num_classes: int

    @staticmethod
    def _get_embedding_initializer(num_features: int) -> nn.initializers.Initializer:
        def init(key: jax.random.KeyArray, shape: tuple, dtype: Any = jnp.float_) -> jnp.ndarray:
            weights = jnp.zeros(shape)
            weights[:, :num_features] = jax.random.normal(key, shape, dtype) * 0.02 + 1
            return weights

        return init

    def setup(self):
        self.bn = nn.BatchNorm(use_bias=False, use_scale=False)
        self.embed = nn.Embed(
            self.num_classes, self.num_features * 2, embedding_init=self._get_embedding_initializer(self.num_features)
        )

    def __call__(self, x: NdArray, y: NdArray, training: bool = False) -> jnp.ndarray:
        if x.ndim == 3:
            n_d1, nd2 = x.shape[:2]
            x = x.reshape(n_d1 * nd2, -1)
            need_reshaping = True

            y = jnp.broadcast_to(y[None], (n_d1, nd2, -1))
            y = y.reshape(n_d1 * nd2, -1)

        out = self.bn(x)
        gamma, beta = self.embed(y.squeeze(-1).long()).chunk(2, 1)
        out = gamma * out + beta

        if need_reshaping:
            out = out.reshape(n_d1, nd2, -1)

        return out
