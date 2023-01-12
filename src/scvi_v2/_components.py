import dataclasses
from typing import Any, Callable, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from flax.linen.dtypes import promote_dtype
from flax.linen.initializers import variance_scaling

from ._types import Array, Dtype, NdArray, PRNGKey, Shape

_normal_initializer = jax.nn.initializers.normal(stddev=0.1)


class Dense(nn.DenseGeneral):
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
    scale_eps: float = 1e-5
    training: Optional[bool] = None

    @nn.compact
    def __call__(self, inputs: NdArray, training: Optional[bool] = None) -> dist.Normal:  # noqa: D102
        training = nn.merge_param("training", self.training, training)
        h = inputs
        for _ in range(self.n_layers):
            h = ResnetBlock(n_in=self.n_in, n_out=self.n_hidden)(h, training=training)
        mean = Dense(self.n_out)(h)
        scale = nn.Sequential([Dense(self.n_out), nn.softplus])(h)
        return dist.Normal(mean, scale + self.scale_eps)


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


class FactorizedEmbedding(nn.Module):
    """
    Factorized Embedding Module.

    A parameterized function from integers [0, n) to d-dimensional vectors;
    however, the d-dimensional vectors are parameterized with a matrix factorization.

    Attributes
    ----------
    num_embeddings
        number of embeddings.
    features
        number of feature dimensions for each embedding.
    dtype
        the dtype of the embedding vectors (default: same as embedding).
    param_dtype
        the dtype passed to parameter initializers (default: float32).
    embedding_init
        embedding initializer.
    """

    num_embeddings: int
    features: int
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32
    embedding_init: Callable[[PRNGKey, Shape, Dtype], Array] = _normal_initializer

    embedding: NdArray = dataclasses.field(init=False)

    def setup(self):
        """Initialize the embedding matrix."""
        self.embedding = self.param(
            "embedding", self.embedding_init, (self.num_embeddings, self.features), self.param_dtype
        )

    def __call__(self, inputs: NdArray) -> NdArray:
        """
        Embeds the inputs along the last dimension.

        Parameters
        ----------
        inputs
            input data, all dimensions are considered batch dimensions.

        Returns
        -------
        Output which is embedded input data.  The output shape follows the input,
        with an additional `features` dimension appended.
        """
        if not jnp.issubdtype(inputs.dtype, jnp.integer):
            raise ValueError("Input type must be an integer or unsigned integer.")
        # Use take because fancy indexing numpy arrays with JAX indices does not
        # work correctly.
        (embedding,) = promote_dtype(self.embedding, dtype=self.dtype, inexact=False)
        return jnp.take(embedding, inputs, axis=0)

    def attend(self, query: NdArray) -> NdArray:
        """
        Attend over the embedding using a query array.

        Parameters
        ----------
        query
            array with last dimension equal the feature depth `features` of the
            embedding.

        Returns
        -------
        An array with final dim `num_embeddings` corresponding to the batched
        inner-product of the array of query vectors against each embedding.
        Commonly used for weight-sharing between embeddings and logit transform
        in NLP models.
        """
        query, embedding = promote_dtype(query, self.embedding, dtype=self.dtype)
        return jnp.dot(query, embedding.T)
