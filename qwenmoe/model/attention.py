"""Scaled dot-product self-attention with Grouped-Query Attention (GQA)
and a swappable positional scheme (Requirements 1.5, 1.6, 12.1).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from qwenmoe.config import ModelConfig
from qwenmoe.model.positional import AlibiBias, NoPositional, RotaryEmbedding


def build_positional_scheme(config: ModelConfig, head_dim: int) -> nn.Module:
    """Constructs the positional-encoding strategy named by
    `config.positional_scheme` (Requirement 12.1)."""
    if config.positional_scheme == "rope":
        return RotaryEmbedding(head_dim, theta=config.rope_theta)
    if config.positional_scheme == "nope":
        return NoPositional()
    if config.positional_scheme == "alibi":
        return AlibiBias(config.n_heads)
    raise ValueError(f"Unknown positional_scheme: {config.positional_scheme!r}")


class Attention(nn.Module):
    """Causal self-attention with Grouped-Query Attention: `n_kv_heads`
    key/value heads are shared across `n_heads // n_kv_heads` query heads
    each (Requirement 1.5). `n_kv_heads == n_heads` is standard MHA.
    `ModelConfig.__post_init__` already guarantees `d_model % n_heads == 0`
    and `n_heads % n_kv_heads == 0`, so this trusts that invariant rather
    than re-checking it.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.n_rep = config.n_heads // config.n_kv_heads
        self.head_dim = config.d_model // config.n_heads

        self.w_q = nn.Linear(config.d_model, config.n_heads * self.head_dim, bias=False)
        self.w_k = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.w_v = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.w_o = nn.Linear(config.n_heads * self.head_dim, config.d_model, bias=False)

        self.positional = build_positional_scheme(config, self.head_dim)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        q = self.w_q(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.w_k(x).view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.w_v(x).view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q, k = self.positional.apply_to_qk(q, k, positions)

        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        bias = self.positional.attention_bias(seq_len)
        if bias is not None:
            scores = scores + bias.to(scores.dtype)

        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=x.device), diagonal=1
        )
        scores = scores + causal_mask

        probs = F.softmax(scores, dim=-1)
        out = probs @ v  # (batch, n_heads, seq_len, head_dim)

        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.n_heads * self.head_dim)
        return self.w_o(out)

    def forward_with_trace(
        self, x: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        """Same computation as `forward`, additionally capturing every
        intermediate tensor `LayerTrace` needs (Requirement 7.1) -- not a
        separate visualization math path, just extra capture points
        spliced into the same steps `forward` already performs. Only
        `TransformerBlock.forward_with_trace` calls this.
        """
        batch, seq_len, _ = x.shape

        q = self.w_q(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.w_k(x).view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.w_v(x).view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        trace_q, trace_k, trace_v = q.squeeze(0), k.squeeze(0), v.squeeze(0)

        q, k = self.positional.apply_to_qk(q, k, positions)
        rope_angles_or_alibi_bias = self.positional.positional_trace(positions)

        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        bias = self.positional.attention_bias(seq_len)
        if bias is not None:
            scores = scores + bias.to(scores.dtype)
        trace_scores_pre_mask = scores.squeeze(0)

        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=x.device), diagonal=1
        )
        scores = scores + causal_mask

        probs = F.softmax(scores, dim=-1)
        trace_scores_post_softmax = probs.squeeze(0)
        out = probs @ v  # (batch, n_heads, seq_len, head_dim)

        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.n_heads * self.head_dim)
        output = self.w_o(out)

        trace = {
            "rope_angles_or_alibi_bias": rope_angles_or_alibi_bias,
            "q": trace_q,
            "k": trace_k,
            "v": trace_v,
            "attn_scores_pre_mask": trace_scores_pre_mask,
            "attn_scores_post_softmax": trace_scores_post_softmax,
        }
        return output, trace


class MLAAttention(nn.Module):
    """Simplified Multi-Head Latent Attention (DeepSeek-V2/V3, Requirement
    12.4). Instead of caching a full per-head K and V, compresses both into
    one shared low-rank latent `c_kv` (dim `mla_latent_dim`) that every head
    up-projects back to full size on the fly -- only the latent (plus a
    small shared positional key, below) ever needs to be cached for
    generation. A compressed-then-rotated K can't be decompressed with a
    fixed matrix (RoPE rotation depends on position), so positional
    information instead rides on a small *extra* "decoupled" RoPE
    sub-dimension appended to every head's q/k rather than rotating the
    content dimensions themselves -- see the journal entry for the full
    derivation. This means `MLAAttention` always applies its own decoupled
    RoPE regardless of `ModelConfig.positional_scheme` -- a deliberate
    exception documented in design.md, not an oversight. Exposes the same
    `forward(x, positions) -> Tensor` signature as `Attention`, so
    `TransformerBlock` doesn't need to know which one it holds. Skips the
    paper's optional query-side low-rank compression (it only saves
    training activation memory, not KV cache), matching Requirement 12.4's
    own "simplified" wording.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        if config.mla_latent_dim is None:
            raise ValueError("MLAAttention requires ModelConfig.mla_latent_dim to be set")
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        if self.head_dim % 4 != 0:
            raise ValueError(
                "MLAAttention needs head_dim (d_model // n_heads) divisible by 4 for its "
                f"decoupled-RoPE split, got head_dim={self.head_dim} (d_model={config.d_model}, "
                f"n_heads={config.n_heads})"
            )
        self.latent_dim = config.mla_latent_dim
        self.rope_head_dim = self.head_dim // 2

        # Low-rank K/V joint compression (DeepSeek-V2 Eq. 9-11): every
        # token's K and V, for every head, are up-projected from one shared
        # latent rather than from `x` directly.
        self.w_dkv = nn.Linear(config.d_model, self.latent_dim, bias=False)
        self.w_uk = nn.Linear(self.latent_dim, config.n_heads * self.head_dim, bias=False)
        self.w_uv = nn.Linear(self.latent_dim, config.n_heads * self.head_dim, bias=False)

        # Plain (uncompressed) query content path.
        self.w_q = nn.Linear(config.d_model, config.n_heads * self.head_dim, bias=False)

        # Decoupled RoPE (Eq. 14-15): a small per-head query slice, plus a
        # small key slice SHARED across every head -- only this one shared
        # slice (not n_heads copies) needs caching during generation.
        self.w_qr = nn.Linear(config.d_model, config.n_heads * self.rope_head_dim, bias=False)
        self.w_kr = nn.Linear(config.d_model, self.rope_head_dim, bias=False)
        self.rope = RotaryEmbedding(self.rope_head_dim, theta=config.rope_theta)

        self.w_o = nn.Linear(config.n_heads * self.head_dim, config.d_model, bias=False)

    @property
    def kv_cache_elements_per_token(self) -> int:
        """What generation would actually need to cache per token per layer
        -- the shared latent plus the shared decoupled-RoPE key -- NOT the
        full `head_dim`-sized K/V every head uses once decompressed
        (Requirement 12.4's KV-cache-reduction claim)."""
        return self.latent_dim + self.rope_head_dim

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        c_kv = self.w_dkv(x)  # (batch, seq_len, latent_dim) -- this is what generation caches
        k_content = (
            self.w_uk(c_kv).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        )
        v = self.w_uv(c_kv).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        q_content = self.w_q(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        q_rope = self.w_qr(x).view(batch, seq_len, self.n_heads, self.rope_head_dim).transpose(1, 2)
        k_rope = self.w_kr(x).view(batch, seq_len, 1, self.rope_head_dim).transpose(1, 2)
        q_rope, k_rope = self.rope.apply_to_qk(q_rope, k_rope, positions)
        k_rope = k_rope.expand(-1, self.n_heads, -1, -1)  # one shared key, broadcast to all heads

        q = torch.cat([q_content, q_rope], dim=-1)
        k = torch.cat([k_content, k_rope], dim=-1)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim + self.rope_head_dim)

        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=x.device), diagonal=1
        )
        scores = scores + causal_mask

        probs = F.softmax(scores, dim=-1)
        out = probs @ v  # (batch, n_heads, seq_len, head_dim) -- content V only, no rope part

        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.n_heads * self.head_dim)
        return self.w_o(out)

    def forward_with_trace(
        self, x: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        """Same computation as `forward`, additionally capturing the
        intermediate tensors `LayerTrace` needs (Requirement 7.1). MLA has
        no single "pre-RoPE q/k" the way GQA does -- only the small
        decoupled-RoPE slice ever rotates -- so unlike `Attention.
        forward_with_trace`, the captured `q`/`k` here are the FULL
        concat(content, rope) tensors actually used in the dot product
        (i.e. already-rotated on their rope slice), and
        `rope_angles_or_alibi_bias` covers only that `rope_head_dim`-sized
        slice, not the full `head_dim`. Only `TransformerBlock.
        forward_with_trace` calls this.
        """
        batch, seq_len, _ = x.shape

        c_kv = self.w_dkv(x)
        k_content = self.w_uk(c_kv).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.w_uv(c_kv).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        q_content = self.w_q(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        q_rope = self.w_qr(x).view(batch, seq_len, self.n_heads, self.rope_head_dim).transpose(1, 2)
        k_rope = self.w_kr(x).view(batch, seq_len, 1, self.rope_head_dim).transpose(1, 2)
        q_rope, k_rope = self.rope.apply_to_qk(q_rope, k_rope, positions)
        rope_angles_or_alibi_bias = self.rope.positional_trace(positions)
        k_rope = k_rope.expand(-1, self.n_heads, -1, -1)

        q = torch.cat([q_content, q_rope], dim=-1)
        k = torch.cat([k_content, k_rope], dim=-1)
        trace_q, trace_k, trace_v = q.squeeze(0), k.squeeze(0), v.squeeze(0)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim + self.rope_head_dim)

        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=x.device), diagonal=1
        )
        trace_scores_pre_mask = scores.squeeze(0)
        scores = scores + causal_mask

        probs = F.softmax(scores, dim=-1)
        trace_scores_post_softmax = probs.squeeze(0)
        out = probs @ v

        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.n_heads * self.head_dim)
        output = self.w_o(out)

        trace = {
            "rope_angles_or_alibi_bias": rope_angles_or_alibi_bias,
            "q": trace_q,
            "k": trace_k,
            "v": trace_v,
            "attn_scores_pre_mask": trace_scores_pre_mask,
            "attn_scores_post_softmax": trace_scores_post_softmax,
        }
        return output, trace


def build_attention(config: ModelConfig) -> nn.Module:
    """Constructs the self-attention variant named by
    `config.attention_variant` (Requirement 12.4). Both variants expose the
    same `forward(x, positions) -> Tensor` signature, so `TransformerBlock`
    doesn't need to know which one it holds.
    """
    if config.attention_variant == "gqa":
        return Attention(config)
    if config.attention_variant == "mla":
        return MLAAttention(config)
    raise ValueError(f"Unknown attention_variant: {config.attention_variant!r}")
