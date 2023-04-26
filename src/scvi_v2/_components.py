import dataclasses
from typing import Any, Callable, Literal, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from flax.linen.dtypes import promote_dtype
from flax.linen.initializers import variance_scaling

from ._types import Dtype, NdArray, PRNGKey, Shape

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

    n_out: int
    n_hidden: int = 128
    internal_activation: Callable = nn.relu
    output_activation: Callable = nn.relu
    training: Optional[bool] = None

    @nn.compact
    def __call__(self, inputs: NdArray, training: Optional[bool] = None) -> NdArray:  # noqa: D102
        training = nn.merge_param("training", self.training, training)
        h = Dense(self.n_hidden)(inputs)
        h = nn.LayerNorm()(h)
        h = self.internal_activation(h)
        n_in = inputs.shape[-1]
        if n_in != self.n_hidden:
            h = h + Dense(self.n_hidden)(inputs)
        else:
            h = h + inputs
        h = Dense(self.n_out)(h)
        h = nn.LayerNorm()(h)
        return self.output_activation(h)


class MLP(nn.Module):
    """Multi-layer perceptron with resnet blocks."""

    n_out: int
    n_hidden: int = 128
    n_layers: int = 1
    activation: Callable = nn.relu
    training: Optional[bool] = None

    @nn.compact
    def __call__(self, inputs: NdArray, training: Optional[bool] = None) -> dist.Normal:  # noqa: D102
        training = nn.merge_param("training", self.training, training)
        h = inputs
        for _ in range(self.n_layers):
            h = ResnetBlock(
                n_out=self.n_hidden, internal_activation=self.activation, output_activation=self.activation
            )(h, training=training)
        return Dense(self.n_out)(h)


class NormalDistOutputNN(nn.Module):
    """Fully-connected neural net parameterizing a normal distribution."""

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
            h = ResnetBlock(n_out=self.n_hidden)(h, training=training)
        mean = Dense(self.n_out)(h)
        scale = nn.Sequential([Dense(self.n_out), nn.softplus])(h)
        return dist.Normal(mean, scale + self.scale_eps)


class ConditionalNormalization(nn.Module):
    """Condition-specific normalization."""

    n_features: int
    n_conditions: int
    training: Optional[bool] = None
    normalization_type: Literal["batch", "layer"] = "layer"

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
        if self.normalization_type == "batch":
            x = nn.BatchNorm(use_bias=False, use_scale=False)(x, use_running_average=not training)
        elif self.normalization_type == "layer":
            x = nn.LayerNorm(use_bias=False, use_scale=False)(x)
        else:
            raise ValueError(f"normalization_type must be one of ['batch', 'layer'], not {self.normalization_type}")
        cond_int = condition.squeeze(-1).astype(int)
        gamma = nn.Embed(
            self.n_conditions, self.n_features, embedding_init=self._gamma_initializer(), name="gamma_conditional"
        )(cond_int)
        beta = nn.Embed(
            self.n_conditions, self.n_features, embedding_init=self._beta_initializer(), name="beta_conditional"
        )(cond_int)
        out = gamma * x + beta

        return out


class AttentionBlock(nn.Module):
    """Attention block consisting of multi-head self-attention and MLP."""

    query_dim: int
    out_dim: int
    outerprod_dim: int = 16
    n_channels: int = 4
    n_heads: int = 2
    dropout_rate: float = 0.0
    n_hidden_mlp: int = 32
    n_layers_mlp: int = 1
    training: Optional[bool] = None
    activation: Callable = nn.gelu

    @nn.compact
    def __call__(self, query_embed: NdArray, kv_embed: NdArray, training: Optional[bool] = None):
        has_mc_samples = query_embed.ndim == 3

        query_for_att = nn.DenseGeneral((self.outerprod_dim, 1), use_bias=False)(query_embed)
        kv_for_att = nn.DenseGeneral((self.outerprod_dim, 1), use_bias=False)(kv_embed)
        eps = nn.MultiHeadDotProductAttention(
            num_heads=self.n_heads,
            qkv_features=self.n_channels * self.n_heads,
            out_features=self.n_channels,
            dropout_rate=self.dropout_rate,
            use_bias=True,
        )(inputs_q=query_for_att, inputs_kv=kv_for_att, deterministic=not training)

        # now remove that extra dimension
        # (batch, n_latent_sample * n_channels)
        if not has_mc_samples:
            eps = jnp.reshape(eps, (eps.shape[0], eps.shape[1] * eps.shape[2]))
        else:
            eps = jnp.reshape(eps, (eps.shape[0], eps.shape[1], eps.shape[2] * eps.shape[3]))

        eps_ = MLP(
            n_out=self.outerprod_dim,
            n_hidden=self.n_hidden_mlp,
            training=training,
            activation=self.activation,
        )(inputs=eps)
        inputs = jnp.concatenate([query_embed, eps_], axis=-1)
        residual = MLP(
            n_out=self.out_dim,
            n_hidden=self.n_hidden_mlp,
            n_layers=self.n_layers_mlp,
            training=training,
            activation=self.activation,
        )(inputs=inputs)
        return residual


class FactorizedEmbedding(nn.Module):
    """
    Factorized Embedding Module.

    A parameterized function from integers [0, n) to d-dimensional vectors;
    however, the d-dimensional vectors are parameterized with a matrix factorization.
    This allows sharing of information between embeddings

    Attributes
    ----------
    num_embeddings
        number of embeddings.
    features
        number of feature dimensions for each embedding.
    factorized_features
        Number of latent dimensions (features) to use for internal factorization
    dtype
        the dtype of the embedding vectors (default: same as embedding).
    param_dtype
        the dtype passed to parameter initializers (default: float32).
    embedding_init
        embedding initializer.
    """

    num_embeddings: int
    features: int
    factorized_features: int
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32
    embedding_init: Callable[[PRNGKey, Shape, Dtype], NdArray] = _normal_initializer

    embedding: NdArray = dataclasses.field(init=False)

    def setup(self) -> None:
        """Initialize the embedding matrix."""
        self.embedding = self.param(
            "embedding", self.embedding_init, (self.num_embeddings, self.factorized_features), self.param_dtype
        )
        self.factor_tensor = self.param(
            "factor_tensor",
            self.embedding_init,
            (self.factorized_features, self.features),
            self.param_dtype,
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
        (factor_tensor,) = promote_dtype(self.factor_tensor, dtype=self.dtype, inexact=False)
        final_embedding = jnp.dot(embedding, factor_tensor)
        return jnp.take(final_embedding, inputs, axis=0)
