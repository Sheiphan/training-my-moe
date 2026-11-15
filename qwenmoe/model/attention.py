"""Scaled dot-product self-attention with Grouped-Query Attention (GQA)
and a swappable positional scheme (Requirements 1.5, 1.6, 12.1).

`forward` (the path every real training/inference call uses) runs fused
`F.scaled_dot_product_attention` (moe-modernization Requirement 7.1-7.2,
task 5); `forward_with_trace` (only ever called by
`viz/scenes_*.py`/`TransformerBlock.forward_with_trace`) keeps the
original explicit matmul/mask/softmax/matmul steps so every intermediate
tensor stays inspectable for the Manim scenes -- the two are numerically
equivalent within `1e-5`, not bit-identical, since a fused kernel is free
to sum in a different order (`tests/test_trace.py`).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from qwenmoe.config import ModelConfig
from qwenmoe.model.positional import AlibiBias, NoPositional, RotaryEmbedding
from qwenmoe.model.rmsnorm import RMSNorm


def _causal_position_mask(
    q_positions: torch.Tensor, kv_len: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """`(1, len(q_positions), kv_len)` additive attention mask: `0` where a
    key (at absolute position `0 … kv_len-1`) is at or before its query's
    absolute position, `-inf` otherwise.

    Only built on the paths `F.scaled_dot_product_attention(is_causal=True)`
    can't serve directly -- an ALiBi bias to fold in, a document-boundary
    mask to fold in (moe-modernization Requirement 10), or a KV-cache
    incremental step whose query positions don't line up `1:1` with the key
    positions (moe-modernization Requirement 8). For the ordinary
    `q_positions == arange(seq_len)`, `kv_len == seq_len` case it is exactly
    `torch.triu(full(-inf), diagonal=1)`.
    """
    key_pos = torch.arange(kv_len, device=device)
    allowed = key_pos[None, :] <= q_positions.to(device)[:, None]  # (q_len, kv_len)
    mask = torch.zeros(q_positions.shape[0], kv_len, device=device, dtype=dtype)
    mask.masked_fill_(~allowed, float("-inf"))
    return mask.unsqueeze(0)


def build_document_mask(segment_ids: torch.Tensor) -> torch.Tensor:
    """`(batch, seq_len)` integer document ids -> a `(batch, 1, seq_len,
    seq_len)` additive attention mask (moe-modernization Requirement 10.2):
    `0.0` where query position `i` and key position `j` fall in the *same*
    document, `-inf` where they fall in different documents.

    Added onto the causal mask inside `Attention` / `MLAAttention` when
    `TrainConfig.document_masking` is on, it stops a token attending back
    across a `<eos>`/separator boundary into a previous packed document --
    so a training window that concatenates several short stories is
    attended over as if each story were its own sequence (Llama-3's
    intra-document masking). The head axis is size 1 and broadcasts across
    all query heads inside SDPA. Never built when `document_masking` is off,
    so a pre-spec run carries no extra work (Requirement 10.3).
    """
    if segment_ids.dim() != 2:
        raise ValueError(
            f"segment_ids must be (batch, seq_len), got shape {tuple(segment_ids.shape)}"
        )
    same_document = segment_ids[:, :, None] == segment_ids[:, None, :]  # (batch, S, S)
    mask = torch.zeros(same_document.shape, dtype=torch.float32, device=segment_ids.device)
    mask.masked_fill_(~same_document, float("-inf"))
    return mask.unsqueeze(1)


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

        # QK-norm (moe-modernization Requirement 7.3, default off): a
        # learnable per-head RMSNorm applied to q and k, independently,
        # *before* the positional transform -- only constructed when
        # requested, so a `qk_norm=False` model (Requirement 7.5's
        # invariant) carries no extra parameters at all, not just unused
        # ones.
        self.qk_norm = config.qk_norm
        if self.qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

    def forward(
        self, x: torch.Tensor, positions: torch.Tensor, kv_cache=None, document_mask=None
    ) -> torch.Tensor:
        """moe-modernization Requirement 7.1-7.2: the non-trace path calls
        fused `F.scaled_dot_product_attention` instead of materializing an
        explicit `(seq_len, seq_len)` score matrix -- the same causal-GQA
        math as `forward_with_trace` below, just computed by one fused
        kernel instead of separate matmul/mask/softmax/matmul steps.
        `enable_gqa=True` lets SDPA itself broadcast each of the
        `n_kv_heads` key/value heads across its `n_rep` query heads,
        replacing the explicit `repeat_interleave` this method used to do.

        `kv_cache` (moe-modernization Requirement 8, `None` for every
        training call): a `LayerKVCache`. The freshly-projected, RoPE'd
        `k`/`v` for *this* call's tokens are appended to it and the full
        running `k`/`v` come back, so an incremental decode step attends
        over the whole prefix without re-encoding it.

        `document_mask` (moe-modernization Requirement 10.2, `None` unless
        `TrainConfig.document_masking` is on): a `(batch, 1, seq_len,
        seq_len)` additive `0`/`-inf` mask from `build_document_mask` that
        blocks attention across packed-document boundaries.

        When the query positions no longer line up `1:1` with the keys (an
        incremental step: one query, `N` keys), an ALiBi bias must be
        folded in, or a document mask is supplied, an explicit additive
        `attn_mask` replaces `is_causal=True`; addition is commutative and
        associative so the causal-plus-ALiBi-plus-document total is
        order-independent and, with `document_mask=None`, byte-identical to
        the pre-spec path (Requirement 10.3).
        """
        batch, seq_len, _ = x.shape

        q = self.w_q(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.w_k(x).view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.w_v(x).view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q, k = self.positional.apply_to_qk(q, k, positions)

        if kv_cache is not None:
            k, v = kv_cache.append(k, v)
        kv_len = k.shape[-2]

        key_positions = torch.arange(kv_len, device=x.device)
        bias = self.positional.attention_bias_positions(positions, key_positions)
        if bias is None and document_mask is None and kv_len == seq_len:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        else:
            attn_mask = _causal_position_mask(positions, kv_len, x.device, q.dtype)
            if bias is not None:
                attn_mask = attn_mask + bias.to(q.dtype).unsqueeze(0)
            if document_mask is not None:
                attn_mask = attn_mask + document_mask.to(q.dtype)
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, enable_gqa=True
            )

        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.n_heads * self.head_dim)
        return self.w_o(out)

    def forward_with_trace(
        self, x: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        """Same computation as `forward`, additionally capturing every
        intermediate tensor `LayerTrace` needs (Requirement 7.1) -- not a
        separate visualization math path, just extra capture points
        spliced into the same steps `forward` already performs. Only
        `TransformerBlock.forward_with_trace` calls this. `trace_q`/
        `trace_k` are captured *after* QK-norm (Requirement 7.3) when it's
        active, so the trace always reflects the actual q/k that reach the
        positional transform and the dot product -- not a pre-norm value
        the rest of the computation never uses.
        """
        batch, seq_len, _ = x.shape

        q = self.w_q(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.w_k(x).view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.w_v(x).view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Captured only when QK-norm is active: with it off, `q`/`k` below
        # are already the un-normalized projections, so a second copy would
        # be pure duplication (and `LayerTrace` documents the `None`).
        trace_q_pre, trace_k_pre = None, None
        if self.qk_norm:
            trace_q_pre, trace_k_pre = q.squeeze(0), k.squeeze(0)
            q = self.q_norm(q)
            k = self.k_norm(k)
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
            "q_pre_norm": trace_q_pre,
            "k_pre_norm": trace_k_pre,
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

    def forward(
        self, x: torch.Tensor, positions: torch.Tensor, kv_cache=None, document_mask=None
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        c_kv_new = self.w_dkv(x)  # (batch, seq_len, latent_dim) -- the compressed cache entry

        q_content = self.w_q(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        q_rope = self.w_qr(x).view(batch, seq_len, self.n_heads, self.rope_head_dim).transpose(1, 2)
        k_rope_new = self.w_kr(x).view(batch, seq_len, 1, self.rope_head_dim).transpose(1, 2)
        q_rope, k_rope_new = self.rope.apply_to_qk(q_rope, k_rope_new, positions)

        # moe-modernization Requirement 8: MLA caches only the shared latent
        # `c_kv` and the *one* shared decoupled-RoPE key per token -- never
        # the n_heads-wide decompressed K/V (those are re-derived each step
        # from the cached latent). `LatentKVCache.append` grows both and
        # returns the running tensors.
        if kv_cache is not None:
            c_kv, k_rope = kv_cache.append(c_kv_new, k_rope_new)
        else:
            c_kv, k_rope = c_kv_new, k_rope_new
        kv_len = c_kv.shape[1]

        k_content = self.w_uk(c_kv).view(batch, kv_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.w_uv(c_kv).view(batch, kv_len, self.n_heads, self.head_dim).transpose(1, 2)
        k_rope = k_rope.expand(-1, self.n_heads, -1, -1)  # one shared key, broadcast to all heads

        q = torch.cat([q_content, q_rope], dim=-1)
        k = torch.cat([k_content, k_rope], dim=-1)

        # Fused SDPA (Requirement 7.1). No ALiBi branch -- MLA always uses
        # its own decoupled RoPE. SDPA's default scale (1/sqrt(q's last dim))
        # already equals 1/sqrt(head_dim + rope_head_dim) since q is exactly
        # that wide. An incremental cache step (query positions not lined up
        # 1:1 with the keys) or a document mask (moe-modernization
        # Requirement 10.2) needs an explicit additive mask instead of
        # `is_causal`; with `document_mask=None` and no cache this is
        # byte-identical to the pre-spec path.
        if document_mask is None and kv_len == seq_len:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            attn_mask = _causal_position_mask(positions, kv_len, x.device, q.dtype)
            if document_mask is not None:
                attn_mask = attn_mask + document_mask.to(q.dtype)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

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

if __name__ == "__main__":
    # Minimal demo for GQA: construct a lightweight config-like object
    from types import SimpleNamespace

    cfg = SimpleNamespace(
        attention_variant="gqa",
        n_heads=8,
        n_kv_heads=2,
        d_model=512,
        positional_scheme="rope",
        qk_norm=False,
        rope_theta=10000.0,
        mla_latent_dim=None,
    )

    attn = build_attention(cfg)
    x = torch.randn(1, 4, cfg.d_model)
    positions = torch.arange(4)
    out = attn(x, positions)
    print("GQA demo output shape:", out.shape)
