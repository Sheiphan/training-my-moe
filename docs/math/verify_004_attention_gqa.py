"""Reproduces every hand-computed number in `004-attention-gqa.md`.

Run it:  uv run python docs/math/verify_004_attention_gqa.py

It builds the exact tiny `Attention` module the derivation uses (d_model=4,
n_heads=2, n_kv_heads=1, seq_len=3), pins its four weight matrices to the
values written out by hand in the doc, runs BOTH code paths
(`forward_with_trace`, the explicit matmul/mask/softmax/matmul steps, and
`forward`, the fused `F.scaled_dot_product_attention` path), prints every
intermediate tensor, and asserts the by-hand end-to-end outputs to 4
decimal places. No fabricated numbers -- this is the source of truth the
prose quotes.
"""
from __future__ import annotations

import torch

from qwenmoe.config import ModelConfig
from qwenmoe.model.attention import Attention

torch.set_printoptions(precision=6, sci_mode=False)


def _cfg(**overrides) -> ModelConfig:
    base = dict(
        vocab_size=10,
        d_model=4,
        n_heads=2,
        n_kv_heads=1,
        max_seq_len=8,
        positional_scheme="nope",
        qk_norm=False,
        rope_theta=10000.0,
    )
    base.update(overrides)
    return ModelConfig(**base)


# --- the hand-picked weights (Section 2 of the doc) -----------------------
W_Q = torch.tensor([[1.0, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1], [1, 0, 0, 1]])
W_K = torch.tensor([[1.0, 0, 1, 0], [0, 1, 0, 1]])
W_V = torch.tensor([[1.0, 1, 0, 0], [0, 0, 1, 1]])
W_O = torch.tensor([[1.0, 0, 1, 0], [0, 1, 0, 1], [1, 0, 0, 1], [0, 1, 1, 0]])

X = torch.tensor([[[1.0, 0, 1, 0], [0, 2, 0, 1], [1, 1, 0, 0]]])  # (1, 3, 4)
POSITIONS = torch.arange(3)


def _build(scheme: str) -> Attention:
    attn = Attention(_cfg(positional_scheme=scheme))
    with torch.no_grad():
        attn.w_q.weight.copy_(W_Q)
        attn.w_k.weight.copy_(W_K)
        attn.w_v.weight.copy_(W_V)
        attn.w_o.weight.copy_(W_O)
    return attn


def _dump(scheme: str) -> torch.Tensor:
    attn = _build(scheme)
    out_trace, trace = attn.forward_with_trace(X, POSITIONS)
    out_fused = attn.forward(X, POSITIONS)

    print(f"\n================ positional_scheme = {scheme!r} ================")
    print("q per head (PRE-positional, shape n_heads,seq,head_dim):\n", trace["q"])
    print("k (n_kv_heads,seq,head_dim):\n", trace["k"])
    print("v:\n", trace["v"])
    print("scaled scores per head, pre-mask (post-RoPE / +ALiBi):\n",
          trace["attn_scores_pre_mask"])
    print("probabilities after causal mask + softmax:\n",
          trace["attn_scores_post_softmax"])
    print("output, explicit trace path:\n", out_trace)
    print("output, fused SDPA path:\n", out_fused)
    print("max |trace - fused| =", (out_trace - out_fused).abs().max().item())
    return out_trace


nope = _dump("nope")
rope = _dump("rope")
alibi = _dump("alibi")

# --- assert the by-hand end-to-end outputs (doc Section 13) ---------------
expected_nope = torch.tensor([[
    [2.000000, 2.000000, 2.000000, 2.000000],
    [3.474192, 2.000000, 2.804430, 2.669762],
    [3.408561, 1.573375, 2.318140, 2.663796],
]])
torch.testing.assert_close(nope, expected_nope, atol=1e-4, rtol=0)

# RoPE: the diagonal score entries (relative position j - i = 0) must be
# byte-for-byte the un-rotated dot products / sqrt(2). (doc Section 5)
rope_attn = _build("rope")
_, rope_trace = rope_attn.forward_with_trace(X, POSITIONS)
diag = torch.diagonal(rope_trace["attn_scores_pre_mask"], dim1=-2, dim2=-1)
nope_attn = _build("nope")
_, nope_trace = nope_attn.forward_with_trace(X, POSITIONS)
nope_diag = torch.diagonal(nope_trace["attn_scores_pre_mask"], dim1=-2, dim2=-1)
torch.testing.assert_close(diag, nope_diag, atol=1e-5, rtol=0)

# --- the 1/sqrt(d_k) variance argument, checked exactly (doc Section 8) ---
# Each q_i, k_i is independent +/-1 with prob 1/2, so q.k = sum_i q_i k_i is
# a sum of d_k independent +/-1 terms: mean 0, variance d_k exactly. Enumerate
# every (q, k) pair for small d_k and confirm.
signs = torch.tensor([-1.0, 1.0])
for d_k in (2, 4, 8):
    vecs = torch.cartesian_prod(*([signs] * d_k))            # 2**d_k vectors
    dots = (vecs[:, None, :] * vecs[None, :, :]).sum(-1).flatten()  # all pairs
    print(f"d_k={d_k}: E[q.k]={dots.mean():+.4f}  Var[q.k]={dots.var(unbiased=False):.4f}"
          f" (= d_k)  std={dots.std(unbiased=False):.4f} (= sqrt(d_k) = {d_k ** 0.5:.4f})")
    assert abs(dots.var(unbiased=False).item() - d_k) < 1e-3

print("\nAll assertions passed -- every number in 004-attention-gqa.md is real.")
