"""Incremental key/value caches for generation (moe-modernization
Requirement 8).

Without a cache, sampling token `n` re-runs a full forward pass over all
`n` tokens generated so far -- the attention cost per new token is
`O(n)` for the score matrix, `O(n^2)` cumulative to produce `n` tokens. A
cache keeps each layer's already-computed keys/values around so a new
token only computes its *own* k/v and attends against the stored prefix:
`O(1)` new work per layer per token (`O(n)` cumulative).

Two shapes, one per attention variant:

- `LayerKVCache` (GQA): stores the post-RoPE `k` and `v`, each
  `(batch, n_kv_heads, length, head_dim)`, growing along the length axis.
- `LatentKVCache` (MLA): stores only the shared low-rank latent `c_kv`
  `(batch, length, latent_dim)` and the *one* shared decoupled-RoPE key
  `(batch, 1, length, rope_head_dim)` -- the `n_heads`-wide decompressed
  K/V are re-derived from the latent each step and never stored. This is
  the whole point of MLA: `latent_dim + rope_head_dim` cached elements per
  token per layer instead of `2 * n_kv_heads * head_dim`.

`nbytes` on each is what the memory-comparison test (Requirement 8.3)
measures.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from qwenmoe.model.attention import MLAAttention


def _nbytes(*tensors: torch.Tensor | None) -> int:
    return sum(t.element_size() * t.nelement() for t in tensors if t is not None)


class LayerKVCache:
    """Growing `(k, v)` buffers for one GQA attention layer."""

    def __init__(self) -> None:
        self.k: torch.Tensor | None = None
        self.v: torch.Tensor | None = None

    def append(
        self, k_new: torch.Tensor, v_new: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Concatenate `(k_new, v_new)` -- shape `(batch, n_kv_heads,
        new_len, head_dim)` -- along the length axis and return the full
        running `(k, v)`."""
        self.k = k_new if self.k is None else torch.cat([self.k, k_new], dim=-2)
        self.v = v_new if self.v is None else torch.cat([self.v, v_new], dim=-2)
        return self.k, self.v

    @property
    def length(self) -> int:
        return 0 if self.k is None else self.k.shape[-2]

    @property
    def nbytes(self) -> int:
        return _nbytes(self.k, self.v)


class LatentKVCache:
    """Growing shared-latent + shared-decoupled-key buffers for one MLA
    attention layer."""

    def __init__(self) -> None:
        self.c_kv: torch.Tensor | None = None
        self.k_rope: torch.Tensor | None = None

    def append(
        self, c_kv_new: torch.Tensor, k_rope_new: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`c_kv_new`: `(batch, new_len, latent_dim)` grows along axis 1.
        `k_rope_new`: `(batch, 1, new_len, rope_head_dim)` grows along axis
        2. Returns the running `(c_kv, k_rope)`."""
        self.c_kv = (
            c_kv_new if self.c_kv is None else torch.cat([self.c_kv, c_kv_new], dim=1)
        )
        self.k_rope = (
            k_rope_new
            if self.k_rope is None
            else torch.cat([self.k_rope, k_rope_new], dim=2)
        )
        return self.c_kv, self.k_rope

    @property
    def length(self) -> int:
        return 0 if self.c_kv is None else self.c_kv.shape[1]

    @property
    def nbytes(self) -> int:
        return _nbytes(self.c_kv, self.k_rope)


def build_kv_cache(model: nn.Module) -> list[LayerKVCache | LatentKVCache]:
    """One cache per transformer block, matching that block's attention
    variant (Requirement 8.1). Pass the returned list to
    `QwenMoEModel.forward(..., kv_caches=...)`."""
    return [
        LatentKVCache() if isinstance(block.attn, MLAAttention) else LayerKVCache()
        for block in model.blocks
    ]
