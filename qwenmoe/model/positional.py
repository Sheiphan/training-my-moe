"""Positional-encoding strategies: RoPE (default), NoPE, and ALiBi.

Each strategy exposes the same two hooks so `Attention` (task 4) can treat
them uniformly:
  - `apply(q, k, positions) -> (q, k)`: a q/k rotation (RoPE) or a no-op
    (NoPE, ALiBi).
  - `attention_bias(seq_len) -> Tensor | None`: an additive bias for the
    raw attention scores (ALiBi only; `None` for RoPE/NoPE).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    """RoPE (Su et al. 2021 / RoFormer) via the "rotate-half" formulation
    used by Llama/Qwen-style models (Requirement 1.3)."""

    def __init__(self, head_dim: int, theta: float = 10000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _angles(self, positions: torch.Tensor) -> torch.Tensor:
        """(seq_len, head_dim) rotation angles -- shared by `apply_to_qk`
        and `positional_trace` so there's one computation to keep in sync."""
        positions = positions.to(device=self.inv_freq.device, dtype=torch.float32)
        freqs = torch.outer(positions, self.inv_freq)  # (seq_len, head_dim/2)
        return torch.cat((freqs, freqs), dim=-1)       # (seq_len, head_dim)

    def apply_to_qk(
        self, q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """q, k: (..., seq_len, head_dim); positions: (seq_len,) token indices."""
        emb = self._angles(positions).to(q.device)
        cos = emb.cos().to(q.dtype)
        sin = emb.sin().to(q.dtype)
        q_rot = q * cos + _rotate_half(q) * sin
        k_rot = k * cos + _rotate_half(k) * sin
        return q_rot, k_rot

    def attention_bias(self, seq_len: int) -> torch.Tensor | None:
        return None

    def positional_trace(self, positions: torch.Tensor) -> torch.Tensor:
        """The raw rotation angles `apply_to_qk` derives cos/sin from, for
        the embedding/positional-encoding scene to animate the RoPE
        rotation step (Requirement 7.2)."""
        return self._angles(positions)


class NoPositional(nn.Module):
    """No positional embedding: relies purely on causal masking (Requirement 12.1)."""

    def apply_to_qk(
        self, q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return q, k

    def attention_bias(self, seq_len: int) -> torch.Tensor | None:
        return None

    def positional_trace(self, positions: torch.Tensor) -> torch.Tensor | None:
        return None


class AlibiBias(nn.Module):
    """ALiBi (Press et al. 2021): a fixed linear-slope bias added to raw
    attention scores instead of rotating q/k (Requirement 12.1)."""

    def __init__(self, n_heads: int):
        super().__init__()
        slopes = torch.tensor(_alibi_slopes(n_heads), dtype=torch.float32)
        self.register_buffer("slopes", slopes, persistent=False)

    def apply_to_qk(
        self, q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return q, k

    def attention_bias(self, seq_len: int) -> torch.Tensor:
        """(n_heads, seq_len, seq_len) bias to add to attention scores
        before the softmax: more negative the further back a key is."""
        positions = torch.arange(seq_len, device=self.slopes.device, dtype=torch.float32)
        distance = positions[None, :] - positions[:, None]  # [i, j] = j - i
        return distance[None, :, :] * self.slopes[:, None, None]

    def positional_trace(self, positions: torch.Tensor) -> torch.Tensor:
        """Same bias `attention_bias` adds to scores -- captured here too
        so the embedding/positional-encoding scene has a uniform field to
        read regardless of `positional_scheme` (Requirement 7.2)."""
        return self.attention_bias(positions.shape[0])


def _alibi_slopes(n_heads: int) -> list[float]:
    """Standard ALiBi geometric-sequence slopes (Press et al. 2021)."""

    def _slopes_for_power_of_2(n: int) -> list[float]:
        start = 2.0 ** (-8.0 / n)
        return [start ** (i + 1) for i in range(n)]

    if math.log2(n_heads).is_integer():
        return _slopes_for_power_of_2(n_heads)
    closest_pow2 = 2 ** math.floor(math.log2(n_heads))
    base = _slopes_for_power_of_2(closest_pow2)
    extra = _slopes_for_power_of_2(2 * closest_pow2)[0::2][: n_heads - closest_pow2]
    return base + extra
