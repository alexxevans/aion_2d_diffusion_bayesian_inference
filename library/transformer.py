import jax
import jax.numpy as jnp

import haiku as hk

from jaxtyping import Array, PyTree
from typing import Callable, Optional

from .attention import MultiHeadAttention


# B -> batch size
# T -> sequence length
# D -> embedding dimension


class Transformer(hk.Module):
    """A transformer stack."""

    num_heads: int  # Number of attention heads.
    num_layers: int  # Number of transformer (attention + MLP) layers to stack.
    attn_size: int  # Size of the attention (key, query, value) vectors.
    dropout_rate: float  # Probability with which to apply dropout.
    widening_factor: int = 4  # Factor by which the MLP hidden layer widens.
    name: str | None = None  # Optional identifier for the module.

    def __init__(
            self,
            num_heads: int,
            num_layers: int,
            attn_size: int,
            dropout_rate: Optional[float] = None,
            widening_factor: int = 4,
            num_hidden_layers: int = 1,
            act: Callable = jax.nn.gelu,
            skip_connection_attn: bool = True,
            skip_connection_mlp: bool = True,
            initializer: Optional[hk.initializers.Initializer] = None,
            save_attention_weights: bool = False,
            attention_method: str = "dense",
            name: str | None = "transformer",
    ):
        super().__init__(name=name)
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.attn_size = attn_size
        self.dropout_rate = dropout_rate
        self.widening_factor = widening_factor
        self.num_hidden_layers = num_hidden_layers
        if initializer is None:
            initializer = hk.initializers.VarianceScaling(2 / self.num_layers)
        self.initializer = initializer
        self.act = act
        self.save_attention_weights = save_attention_weights
        self.attention_method = attention_method
        self.skip_connection_attn = skip_connection_attn
        self.skip_connection_mlp = skip_connection_mlp

    def __call__(
            self,
            inputs: Array,  # [B, T, D]
            context: Optional[Array] = None,  # [B, D_context]
            mask: Array | None = None,  # [T, T] or [B, T, T]
    ) -> jax.Array:  # [B, T, D]
        """Transforms input embedding sequences to output embedding sequences."""

        if mask is not None:
            if mask.ndim == 2:
                mask = mask[None, None, :, :]
            elif mask.ndim == 3:
                mask = mask[:, None, :, :]
            else:
                raise ValueError(f"Mask must have ndim 2 or 3, got {mask.ndim}.")

        h = inputs

        for _ in range(self.num_layers):
            # First the attention block.

            h = self.layer_norm(h)
            h_attn = self.attention_block(h, mask=mask)

            if self.skip_connection_attn:
                h = h + h_attn
            else:
                h = h_attn

            # Then the dense block.
            h = self.layer_norm(h)
            h_dense = self.dense_block(h, context)

            if self.skip_connection_mlp:
                h = h + h_dense
            else:
                h = h_dense

        out = self.layer_norm(h)

        return out

    @hk.transparent
    def layer_norm(self, x: Array) -> Array:
        """Applies a unique LayerNorm to `x` with default settings."""
        ln = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        return ln(x)

    @hk.transparent
    def attention_block(self, x: Array, mask: Array | None = None) -> Array:
        """Applies a multi-head attention block to `x` with default settings."""
        attn_block = MultiHeadAttention(
            num_heads=self.num_heads,
            key_size=self.attn_size,
            model_size=x.shape[-1],
            w_init=self.initializer,
            save_attention_weights=self.save_attention_weights,
            attention_method=self.attention_method,
        )
        attn = attn_block(x, x, x, mask=mask)

        if self.dropout_rate is not None:
            attn = hk.dropout(hk.next_rng_key(), self.dropout_rate, attn)

        return attn

    @hk.transparent
    def dense_block(self, x: Array, context: Optional[Array] = None) -> Array:

        model_size = x.shape[-1]
        hidden_block = []
        for _ in range(self.num_hidden_layers):
            hidden_block.append(hk.Linear(self.widening_factor * model_size, w_init=self.initializer))
            hidden_block.append(self.act)
        dense_block = hk.Sequential(
            hidden_block
            +
            [
                hk.Linear(model_size, w_init=self.initializer),
            ]
        )

        x = dense_block(x)
        if self.dropout_rate is not None:
            x = hk.dropout(hk.next_rng_key(), self.dropout_rate, x)

        if context is not None:
            context_emb = hk.Linear(model_size, w_init=self.initializer)(context)
            context_emb = self.act(context_emb)
            while context_emb.ndim < x.ndim:
                context_emb = context_emb[..., None, :]

            x = x + context_emb

        return x

import haiku as hk
import jax
import jax.numpy as jnp

from typing import Callable, Any, List, Optional
from functools import partial
from jaxtyping import Array, PyTree

import math

from probjax.core.custom_primitives.custom_inverse import custom_inverse


class Flip(hk.Module):
    def __init__(self, axis: int = -1, name: str = "flip"):
        """Flip the array along an axis.

        Args:
            axis (int, optional): Axis to flip. Defaults to -1.
            name (str, optional): Name of the module. Defaults to "flip".
        """
        super().__init__(name=name)
        self.axis = axis

    def __call__(self, x: Array, *args) -> Array:
        return jnp.flip(x, axis=self.axis)


class Permute(hk.Module):
    def __init__(self, permutation: Array, axis: int = -1, name: str = "permute"):
        """Permutes the array along an axis.

        Args:
            permutation (Array): An array of indices to permute.
            axis (int, optional): Axis to permute. Defaults to -1.
            name (str, optional): _description_. Defaults to "permute".
        """
        super().__init__(name=name)
        self.permutation = permutation
        self.axis = axis

    def __call__(self, x: Array, *args) -> Array:
        return jnp.take(x, self.permutation, axis=self.axis)


@partial(custom_inverse, inv_argnum=1)
def rotate(R, x):
    return jnp.matmul(R, x.T).T


rotate.definv_and_logdet(lambda R, x: (jnp.matmul(R.T, x.T).T, 0.0))


class Rotate(hk.Module):
    def __init__(self, key: Array, output_dim: int, name: str = "rotate"):
        """Rotate the array.

        Args:
            rotation_matrix (Array): Rotation matrix.
            name (str, optional): Name of the module. Defaults to "rotate".
        """
        super().__init__(name=name)
        self.rotation_matrix = jax.random.orthogonal(key, output_dim)

    def __call__(self, x: Array, *args) -> Array:
        return rotate(self.rotation_matrix, x)


class SinusoidalEmbedding(hk.Module):
    def __init__(self, output_dim: int = 128, name: str = "sinusoidal_embedding"):
        """Sinusoidal embedding module. Mostly used to embed time.

        Args:
            output_dim (int, optional): Output dimesion. Defaults to 128.
            name (str, optional): Name of the module. Defaults to "sinusoidal_embedding".
        """
        super().__init__(name=name)
        self.output_dim = output_dim

    def __call__(self, inputs):
        half_dim = self.output_dim // 2 + 1
        emb = jnp.log(10000) / (half_dim - 1)
        emb = jnp.exp(jnp.arange(half_dim) * -emb)
        emb = inputs[..., None] * emb[None, ...]
        emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], -1)
        out = jnp.squeeze(emb, axis=-2)
        return out[..., : self.output_dim]


class GaussianFourierEmbedding(hk.Module):
    def __init__(
        self,
        output_dim: int = 128,
        learnable=False,
        name: str = "gaussian_fourier_embedding",
    ):
        """Gaussian Fourier embedding module. Mostly used to embed time.

        Args:
            output_dim (int, optional): Output dimesion. Defaults to 128.
            name (str, optional): Name of the module. Defaults to "gaussian_fourier_embedding".
        """
        super().__init__(name=name)
        self.output_dim = output_dim
        self.learnable = learnable

    def __call__(self, inputs):
        half_dim = self.output_dim // 2 + 1
        B = hk.get_parameter(
            "B", [half_dim, inputs.shape[-1]], init=hk.initializers.RandomNormal()
        )
        if not self.learnable:
            B = jax.lax.stop_gradient(B)
        term1 = jnp.cos(2 * jnp.pi * jnp.dot(inputs, B.T))
        term2 = jnp.sin(2 * jnp.pi * jnp.dot(inputs, B.T))
        out = jnp.concatenate([term1, term2], axis=-1)
        return out[..., : self.output_dim]


class OneHot(hk.Module):
    """One hot encoding module."""

    num_tokens: int  # Size of the vocabulary.
    name: str | None = None  # Optional identifier for the module.

    def __init__(self, num_tokens: int, name: str | None = "one_hot_embed"):
        """_summary_

        Args:
            num_tokens (int): Number of distinct tokens.
            name (str | None, optional): Name of the module. Defaults to "one_hot_embed".
        """
        super().__init__(name=name)
        self.num_tokens = num_tokens

    def __call__(self, x: Array, rng=None) -> Array:
        """One hot encodes the input.

        Args:
            x (jax.Array): Input array of shape [B, T]
        """
        return jax.nn.one_hot(x, self.num_tokens)


class PosEmbed(hk.Module):
    def __init__(self, token_dim: int, max_seq_len: int = 500):
        """Positional embedding module.

        Args:
            token_dim (int): Dimension of the token embedding.
            max_seq_len (int, optional): Maximal length of the sequence. Defaults to 500.
        """
        super().__init__()
        position = jnp.arange(max_seq_len).reshape(-1, 1)
        div_term = jnp.exp(
            jnp.arange(0, token_dim, 2) * (-jnp.log(10000.0) / token_dim)
        )
        pe = jnp.zeros((1, max_seq_len, token_dim))
        pe = pe.at[..., 0::2].set(jnp.sin(position * div_term))
        pe = pe.at[..., 1::2].set(jnp.cos(position * div_term))
        self.pe = pe

    def __call__(self, x: Array, rng=None) -> Array:
        """
        Arguments:
            x: jnp.ndarray, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:, : x.shape[1]]
        return x


class LearnedPosEmbed(hk.Module):
    def __init__(self, max_seq_len: int, name: str = "learned_pos_embed"):
        super().__init__(name=name)
        self.max_seq_len = max_seq_len
        self.embed_init = hk.initializers.TruncatedNormal(stddev=0.02)

    def __call__(self, x: Array, rng=None) -> Array:
        """Embeds the input with learned positional embeddings.

        Args:
            x (Array): Input array of shape [B, T, D]
            max_len (int, optional): Maximum length of the sequence. Defaults to 512.

        Returns:
            Array: Output array of shape [B, T, D]
        """
        _, seq_len, embed_dim = x.shape
        assert (
            seq_len <= self.max_seq_len
        ), "Sequence length cannot be greater than max_len"
        positional_embeddings = hk.get_parameter(
            "positional_embeddings", [self.max_seq_len, embed_dim], init=self.embed_init
        )
        positional_embeddings = positional_embeddings[:seq_len, :]
        return x + positional_embeddings[None, :, :]

