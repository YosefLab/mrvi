from typing import Any, Callable, Optional

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


class ResnetBlock(nn.Module):
    """Resnet block."""

    n_in: int
    n_out: int
    n_hidden: int = 128
    output_activation: Callable = nn.relu
    training: Optional[bool] = None

    @nn.compact
    def __call__(self, inputs: NdArray, training: Optional[bool] = None) -> NdArray:  # noqa: D102
        training = nn.merge_param("training", self.training, training)
        h = Dense(self.n_hidden)(inputs)
        h = nn.BatchNorm(momentum=0.9)(h, use_running_average=not training)
        h = nn.relu(h)
        if self.n_in != self.n_hidden:
            h = h + Dense(self.n_hidden)(inputs)
        h = Dense(self.n_out)(h)
        h = nn.BatchNorm(momentum=0.9)(h, use_running_average=not training)
        return self.output_activation(h)


class NormalDistOutputNN(nn.Module):
    """Fully-connected neural net parameterizing a normal distribution."""

    n_in: int
    n_out: int
    n_hidden: int = 128
    n_layers: int = 1
    var_eps: float = 1e-5
    training: Optional[bool] = None

    @nn.compact
    def __call__(self, inputs: NdArray, training: Optional[bool] = None) -> dist.Normal:  # noqa: D102
        training = nn.merge_param("training", self.training, training)
        h = inputs
        for _ in range(self.n_layers):
            h = ResnetBlock(n_in=self.n_in, n_out=self.n_hidden)(h, training=training)
        mean = Dense(self.n_out)(h)
        var = nn.Sequential([Dense(self.n_out), nn.softplus])(h)
        return dist.Normal(mean, var + self.var_eps)


class ConditionalBatchNorm1d(nn.Module):
    """Batch norm with condition-specific gamma and beta."""

    n_features: int
    n_conditions: int
    training: Optional[bool] = None

    @staticmethod
    def _gamma_initializer() -> jax.nn.initializers.Initializer:
        def init(key: jax.random.KeyArray, shape: tuple, dtype: Any = jnp.float_) -> jnp.ndarray:
            weights = jax.random.normal(key, shape, dtype) * 0.02 + 1
            return weights

        return init

    @staticmethod
    def _beta_initializer() -> jax.nn.initializers.Initializer:
        def init(key: jax.random.KeyArray, shape: tuple, dtype: Any = jnp.float_) -> jnp.ndarray:
            del key
            weights = jnp.zeros(shape, dtype=dtype)
            return weights

        return init

    @nn.compact
    def __call__(self, x: NdArray, condition: NdArray, training: Optional[bool] = None) -> jnp.ndarray:  # noqa: D102
        training = nn.merge_param("training", self.training, training)

        out = nn.BatchNorm(momentum=0.9, use_bias=False, use_scale=False)(x, use_running_average=not training)
        cond_int = condition.squeeze(-1).astype(int)
        gamma = nn.Embed(
            self.n_conditions, self.n_features, embedding_init=self._gamma_initializer(), name="gamma_conditional"
        )(cond_int)
        beta = nn.Embed(
            self.n_conditions, self.n_features, embedding_init=self._beta_initializer(), name="beta_conditional"
        )(cond_int)
        out = gamma * out + beta

        return out
