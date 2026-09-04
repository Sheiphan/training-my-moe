"""Trace dataclasses populated by `QwenMoEModel.forward_with_trace`
(Requirement 7.1): every intermediate tensor from one real forward pass
over a single example prompt, in the shape a Manim scene in `qwenmoe/viz/`
needs. No batch dimension anywhere -- tracing is always one prompt at a
time (see `forward_with_trace`'s batch-size-1 requirement).
"""
from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


@dataclass
class LayerTrace:
    """One `TransformerBlock`'s intermediate tensors for a single traced
    sequence. `rope_angles_or_alibi_bias` is `None` for
    `positional_scheme="nope"`, the RoPE angle tensor for `"rope"`, or the
    additive bias matrix for `"alibi"`. `expert_outputs` only has an entry
    for routed experts that received at least one token this call,
    mirroring `MoELayer`'s own zero-token skip (Requirement 2.5).
    """

    rope_angles_or_alibi_bias: Tensor | None
    q: Tensor                         # (n_heads, seq_len, head_dim), pre-RoPE projection
    k: Tensor                         # (n_kv_heads, seq_len, head_dim), pre-RoPE projection
    v: Tensor                         # (n_kv_heads, seq_len, head_dim)
    attn_scores_pre_mask: Tensor      # (n_heads, seq_len, seq_len), before the causal mask
    attn_scores_post_softmax: Tensor  # (n_heads, seq_len, seq_len)
    attn_output: Tensor               # (seq_len, d_model)
    router_logits: Tensor             # (seq_len, n_routed_experts)
    router_probs: Tensor              # (seq_len, n_routed_experts)
    topk_indices: Tensor              # (seq_len, top_k)
    topk_weights: Tensor              # (seq_len, top_k)
    expert_outputs: dict[int, Tensor]  # expert_id -> (n_tokens_dispatched_to_it, d_model)
    moe_output: Tensor                 # (seq_len, d_model)

    # QK-norm (moe-modernization Requirement 7.3). `q`/`k` above are always
    # the vectors that actually reach the positional transform and the dot
    # product -- i.e. already normalized when `qk_norm=True`. These two
    # capture the *un*-normalized projections that QK-norm consumed, so a
    # visualization can show both sides of the operation rather than only
    # its result. `None` whenever QK-norm is off (nothing was normalized, so
    # `q`/`k` already are the pre-norm values) and for MLA, which has no
    # QK-norm path at all.
    q_pre_norm: Tensor | None = None   # (n_heads, seq_len, head_dim)
    k_pre_norm: Tensor | None = None   # (n_kv_heads, seq_len, head_dim)


@dataclass
class TraceBundle:
    """Everything the Manim scenes (tasks 13-14) need from one real forward
    pass over a single prompt, sourced only from real computation
    (Requirement 7.1) -- never illustrative/fabricated numbers.
    """

    token_ids: list[int]
    embeddings: Tensor         # (seq_len, d_model)
    layers: list[LayerTrace]
    final_norm_output: Tensor  # (seq_len, d_model)
    logits: Tensor             # (seq_len, vocab_size)
    probs: Tensor              # softmax(logits[-1]), (vocab_size,)
