import jax
import jax.numpy as jnp

from scvi_v2._components import (
    ConditionalBatchNorm1d,
    Dense,
    NormalDistOutputNN,
    ResnetBlock,
)


def test_dense():
    key = jax.random.PRNGKey(0)
    x = jnp.ones((20, 10))
    dense = Dense(10)
    params = dense.init(key, x)
    dense.apply(params, x)


def test_resnetblock():
    key = jax.random.PRNGKey(0)
    x = jnp.ones((20, 10))
    block = ResnetBlock(10, 30, training=True)
    params = block.init(key, x)
    block.apply(params, x, mutable=["batch_stats"])


def test_normalnn():
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    x = jnp.ones((20, 10))
    nn = NormalDistOutputNN(10, 30, 3, training=True)
    params = nn.init(key, x)
    nn.apply(params, x, mutable=["batch_stats"])


def test_conditionalbatchnorm1d():
    key = jax.random.PRNGKey(0)
    x = jnp.ones((20, 10))
    y = jnp.ones((20, 1))
    conditionalbatchnorm1d = ConditionalBatchNorm1d(10, 3, training=True)
    params = conditionalbatchnorm1d.init(key, x, y)
    conditionalbatchnorm1d.apply(params, x, y, mutable=["batch_stats"])
