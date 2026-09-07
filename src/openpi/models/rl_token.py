"""AR token layers adapted from pravsels/openpi@9cb105b2, pi0_rl.py (Apache-2.0).

Changes: standalone frozen-feature module, explicit config, masked-value sanitation,
no VLA/flow training path. Original numerical architecture is retained.
"""

from collections.abc import Callable
from dataclasses import dataclass

import flax.nnx as nnx
import jax
import jax.numpy as jnp


class RLTokenTransformerBlock(nnx.Module):
    """Pre-norm transformer block with SwiGLU FFN."""

    def __init__(self, dim: int, num_heads: int, mlp_dim: int, rngs: nnx.Rngs):
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.attn_norm = nnx.RMSNorm(dim, rngs=rngs)
        self.q_proj = nnx.Linear(dim, dim, rngs=rngs)
        self.k_proj = nnx.Linear(dim, dim, rngs=rngs)
        self.v_proj = nnx.Linear(dim, dim, rngs=rngs)
        self.o_proj = nnx.Linear(dim, dim, rngs=rngs)

        self.ffn_norm = nnx.RMSNorm(dim, rngs=rngs)
        self.ffn_gate = nnx.Linear(dim, mlp_dim, rngs=rngs)
        self.ffn_up = nnx.Linear(dim, mlp_dim, rngs=rngs)
        self.ffn_down = nnx.Linear(mlp_dim, dim, rngs=rngs)

    def __call__(self, x: jax.Array, mask: jax.Array | None = None) -> jax.Array:
        b, s, d = x.shape

        # --- self-attention with pre-norm ---
        h = self.attn_norm(x)
        q = self.q_proj(h).reshape(b, s, self.num_heads, self.head_dim)
        k = self.k_proj(h).reshape(b, s, self.num_heads, self.head_dim)
        v = self.v_proj(h).reshape(b, s, self.num_heads, self.head_dim)

        scale = jnp.float32(self.head_dim) ** -0.5
        logits = jnp.einsum("bsnh,btnh->bnst", q, k) * scale
        if mask is not None:
            # mask: (b, s, s) → (b, 1, s, s) for head broadcast
            logits = jnp.where(mask[:, None, :, :], logits, jnp.finfo(logits.dtype).min)
        attn_weights = jax.nn.softmax(logits.astype(jnp.float32), axis=-1).astype(x.dtype)
        attn_out = jnp.einsum("bnst,btnh->bsnh", attn_weights, v).reshape(b, s, d)
        x = x + self.o_proj(attn_out)

        # --- SwiGLU FFN with pre-norm ---
        h = self.ffn_norm(x)
        x = x + self.ffn_down(nnx.silu(self.ffn_gate(h)) * self.ffn_up(h))

        return x


class RLTokenEncoder(nnx.Module):
    """Compresses VLA prefix embeddings into a single RL token via a learned query."""

    def __init__(self, dim: int, num_heads: int, mlp_dim: int, num_layers: int, rngs: nnx.Rngs):
        self.rl_query = nnx.Param(jax.random.normal(rngs.params(), (1, 1, dim)) * 0.02)
        self.layers = {f"layer_{i}": RLTokenTransformerBlock(dim, num_heads, mlp_dim, rngs) for i in range(num_layers)}

    def __call__(self, vla_embeddings: jax.Array, mask: jax.Array | None = None) -> jax.Array:
        """
        Args:
            vla_embeddings: (b, M, dim) stop-gradiented VLA final-layer embeddings.
            mask: (b, M) True for valid tokens.

        Returns:
            rl_token: (b, dim).
        """
        vla_embeddings = jax.lax.stop_gradient(vla_embeddings)
        if mask is not None:
            vla_embeddings = jnp.where(mask[..., None], vla_embeddings, 0)
        b = vla_embeddings.shape[0]
        query = jnp.broadcast_to(self.rl_query.value, (b, 1, vla_embeddings.shape[-1]))
        x = jnp.concatenate([vla_embeddings, query], axis=1)  # (b, M+1, dim)

        if mask is not None:
            ext = jnp.concatenate([mask, jnp.ones((b, 1), dtype=jnp.bool_)], axis=1)
            attn_mask = ext[:, None, :] & ext[:, :, None]  # (b, M+1, M+1) bidirectional
        else:
            attn_mask = None

        for key in sorted(self.layers):
            x = self.layers[key](x, attn_mask)

        return x[:, -1, :]  # RL token at query position


class RLTokenDecoder(nnx.Module):
    """Autoregressively reconstructs VLA embeddings from the RL token."""

    def __init__(self, dim: int, num_heads: int, mlp_dim: int, num_layers: int, rngs: nnx.Rngs):
        self.layers = {f"layer_{i}": RLTokenTransformerBlock(dim, num_heads, mlp_dim, rngs) for i in range(num_layers)}
        self.output_proj = nnx.Linear(dim, dim, rngs=rngs)

    def __call__(
        self,
        rl_token: jax.Array,
        target_embeddings: jax.Array,
        mask: jax.Array | None = None,
    ) -> jax.Array:
        """Teacher-forced autoregressive reconstruction.

        Decoder input:  [z_rl, z̄_1, z̄_2, ..., z̄_{M-1}]
        Target output:  [z̄_1, z̄_2, z̄_3, ..., z̄_M       ]

        Causal masking ensures position i only attends to positions ≤ i.

        Args:
            rl_token: (b, dim).
            target_embeddings: (b, M, dim) stop-gradiented targets.
            mask: (b, M) True for valid target tokens.

        Returns:
            predictions: (b, M, dim).
        """
        target_embeddings = jax.lax.stop_gradient(target_embeddings)
        if mask is not None:
            target_embeddings = jnp.where(mask[..., None], target_embeddings, 0)
        b, seq_len, _ = target_embeddings.shape

        decoder_input = jnp.concatenate([rl_token[:, None, :], target_embeddings[:, :-1, :]], axis=1)  # (b, M, dim)

        causal = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))[None]  # (1, M, M)
        if mask is not None:
            # Key validity: pos 0 is rl_token (always valid), pos 1..M-1 map to targets 0..M-2
            key_valid = jnp.concatenate([jnp.ones((b, 1), dtype=jnp.bool_), mask[:, :-1]], axis=1)
            attn_mask = causal & key_valid[:, None, :]  # (b, M, M)
        else:
            attn_mask = jnp.broadcast_to(causal, (b, seq_len, seq_len))

        x = decoder_input
        for key in sorted(self.layers):
            x = self.layers[key](x, attn_mask)

        return self.output_proj(x)


# ---------------------------------------------------------------------------
# Reconstruction diagnostics
# ---------------------------------------------------------------------------


def _reconstruction_loss(
    predictions: jax.Array,
    target_embeddings: jax.Array,
    mask: jax.Array | None = None,
) -> jax.Array:
    target_embeddings = jax.lax.stop_gradient(target_embeddings)
    residual = predictions - target_embeddings
    if mask is not None:
        residual = jnp.where(mask[..., None], residual, 0)
    recon_sq = jnp.square(residual)
    per_token_l2 = recon_sq.sum(axis=-1)

    if mask is not None:
        per_token_l2 = per_token_l2 * mask
        num_valid = jnp.clip(mask.sum(axis=1), 1)
        per_example = per_token_l2.sum(axis=1) / num_valid
    else:
        per_example = jnp.mean(per_token_l2, axis=1)

    return jnp.mean(per_example)


def compute_reconstruction_ablation_metrics(
    decoder_fn: Callable[[jax.Array, jax.Array, jax.Array | None], jax.Array],
    rl_token: jax.Array,
    target_embeddings: jax.Array,
    mask: jax.Array | None = None,
    *,
    shuffle_perm: jax.Array | None = None,
) -> dict[str, float]:
    """Compare real, zeroed, and shuffled RL-token reconstruction losses."""
    real_predictions = decoder_fn(rl_token, target_embeddings, mask)
    real_loss = _reconstruction_loss(real_predictions, target_embeddings, mask)

    zero_token = jnp.zeros_like(rl_token)
    zero_predictions = decoder_fn(zero_token, target_embeddings, mask)
    zero_loss = _reconstruction_loss(zero_predictions, target_embeddings, mask)

    if shuffle_perm is None:
        shuffle_perm = jnp.roll(jnp.arange(rl_token.shape[0]), 1)
    shuffled_token = rl_token[shuffle_perm]
    shuffled_predictions = decoder_fn(shuffled_token, target_embeddings, mask)
    shuffled_loss = _reconstruction_loss(shuffled_predictions, target_embeddings, mask)

    # Position zero has no preceding teacher-forced target. Do not substitute
    # the first unmasked later position, which would have a different meaning.
    first_valid = jnp.ones(target_embeddings.shape[0], dtype=bool) if mask is None else mask[:, 0]

    def first_loss(predictions):
        residual = jnp.where(first_valid[:, None], predictions[:, 0] - target_embeddings[:, 0], 0)
        total = jnp.square(residual).sum()
        return jnp.where(first_valid.any(), total / jnp.maximum(first_valid.sum(), 1), jnp.nan)

    real_first, zero_first, shuffled_first = map(first_loss, (real_predictions, zero_predictions, shuffled_predictions))

    return {
        "real_recon_loss": float(real_loss),
        "zero_recon_loss": float(zero_loss),
        "shuffled_recon_loss": float(shuffled_loss),
        "zero_recon_gap": float(zero_loss - real_loss),
        "shuffled_recon_gap": float(shuffled_loss - real_loss),
        "first_token_valid_count": int(first_valid.sum()),
        "real_first_token_l2": float(real_first),
        "zero_first_token_l2": float(zero_first),
        "shuffled_first_token_l2": float(shuffled_first),
        "zero_first_token_gap": float(zero_first - real_first),
        "shuffled_first_token_gap": float(shuffled_first - real_first),
    }


@dataclass(frozen=True)
class ARTokenConfig:
    dim: int = 2048
    num_heads: int = 8
    mlp_dim: int = 8192
    num_layers: int = 2

    def __post_init__(self):
        if min(self.dim, self.num_heads, self.mlp_dim, self.num_layers) <= 0 or self.dim % self.num_heads:
            raise ValueError("positive dimensions and head divisibility required")


class ARToken(nnx.Module):
    def __init__(self, config: ARTokenConfig, rngs: nnx.Rngs):
        self.config = config
        args = (config.dim, config.num_heads, config.mlp_dim, config.num_layers, rngs)
        self.encoder = RLTokenEncoder(*args)
        self.decoder = RLTokenDecoder(*args)

    def encode(self, prefix, mask):
        return self.encoder(jax.lax.stop_gradient(prefix), mask)

    def loss(self, prefix, mask):
        prefix = jax.lax.stop_gradient(prefix)
        return _reconstruction_loss(self.decoder(self.encode(prefix, mask), prefix, mask), prefix, mask)


def validate_features(prefix, mask):
    """Host-side batch boundary check; called before JIT, not inside the model."""
    import numpy as np

    prefix, mask = np.asarray(prefix), np.asarray(mask)
    if prefix.ndim != 3 or mask.shape != prefix.shape[:2] or mask.dtype != np.bool_:
        raise ValueError("expected [B,M,D] features and boolean [B,M] mask")
    if not prefix.size or not mask.any(axis=1).all() or not np.isfinite(prefix[mask]).all():
        raise ValueError("each sample needs finite, nonempty valid features")
