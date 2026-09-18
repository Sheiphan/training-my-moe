"""Reproduces every hand-computed number in `022-qk-norm.md`.

Run it:  uv run python docs/math/verify_022_qk_norm.py

ONE running example, carried end to end: the smallest `Attention`
configuration that still exercises every behaviour of QK-norm --

    d_model=8, n_heads=2, n_kv_heads=1, head_dim=4, seq_len=3, batch=1,
    positional_scheme="nope" (spine) / "rope" (one section), qk_norm on.

Two query heads but ONE kv head (so `k_norm` normalizes 3 vectors while
`q_norm` normalizes 6, and the GQA replication happens *after* the norm);
head_dim=4 (so the root-mean-square is an average over four numbers, not a
degenerate pair); three tokens whose per-head q/k vectors have mean-square
1, 2, 3 and 6.25 (so the normalization factor is genuinely different for
each) -- including the journal's own `[3, 0, 4, 0] -> [1.2, 0, 1.6, 0]`
callback.

The token embeddings are the first three standard basis vectors, so each
projected `q_i` / `k_i` / `v_i` is exactly the i-th COLUMN of the pinned
weight matrix -- the projection step is trivial on purpose, so the
arithmetic that matters (the RMSNorm, the bounded dot product, the
softmax) is the arithmetic on display.

It pins the hand-picked weights into a REAL `qwenmoe.model.attention.
Attention`, runs BOTH code paths (`forward_with_trace`, the explicit
matmul/mask/softmax/matmul steps, and `forward`, the fused
`F.scaled_dot_product_attention` path), prints every intermediate, and
asserts:

  * RMSNorm forces mean-square 1 and L2 norm sqrt(head_dim)=2 on every
    per-head q and k vector, regardless of its pre-norm scale;
  * scale-invariance: multiplying the layer input by 10x and 50x leaves
    every post-norm score unchanged (to the epsilon-guard's c-dependent
    slack), while with qk_norm OFF the score std grows ~quadratically;
  * the Cauchy-Schwarz ceiling |q'.k'| <= head_dim = 4 holds for every
    (i, j) pair after the norm, and the scaled score is bounded by
    sqrt(head_dim) = 2;
  * with RoPE, the norm is preserved through the rotation (still 2), and
    norm-then-rotate == rotate-then-norm exactly (w = 1 at init);
  * qk_norm=True changes the output vs qk_norm=False on identical w_q/w_k/
    w_v/w_o weights, and the OFF softmax rows collapse toward one-hot
    (near-zero entropy) as the input scale grows while the ON rows do not;
  * the by-hand end-to-end output tensor, to 5 decimal places;
  * `forward` (fused) and `forward_with_trace` (explicit) agree, and
    prints the measured max |difference|.

No fabricated numbers -- this file is the source of truth the prose quotes.
"""
from __future__ import annotations

import math

import torch

from qwenmoe.config import ModelConfig
from qwenmoe.model.attention import Attention
from qwenmoe.model.rmsnorm import RMSNorm

torch.set_printoptions(precision=8, sci_mode=False)


def _section(title: str) -> None:
    print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)


# ======================================================================
# The running example
# ======================================================================
D_MODEL, N_HEADS, N_KV_HEADS, HEAD_DIM, SEQ_LEN = 8, 2, 1, 4, 3

# Each token embedding is a standard basis vector, so q_i / k_i / v_i is
# the i-th column of the corresponding weight matrix.
X = torch.zeros(1, SEQ_LEN, D_MODEL)
X[0, 0, 0] = 1.0
X[0, 1, 1] = 1.0
X[0, 2, 2] = 1.0
POSITIONS = torch.arange(SEQ_LEN)

# w_q: (out=n_heads*head_dim=8, in=d_model=8). Column i = q_i = [head0 | head1].
W_Q = torch.zeros(N_HEADS * HEAD_DIM, D_MODEL)
W_Q[:, 0] = torch.tensor([1.0, 1, 1, 1, 2, 0, 2, 0])   # q_0
W_Q[:, 1] = torch.tensor([3.0, 0, 4, 0, 1, 1, 1, 1])   # q_1  (head 0 -> journal's [3,0,4,0])
W_Q[:, 2] = torch.tensor([2.0, 2, 0, 0, 0, 3, 0, 4])   # q_2

# w_k / w_v: (out=n_kv_heads*head_dim=4, in=8). One kv head.
W_K = torch.zeros(N_KV_HEADS * HEAD_DIM, D_MODEL)
W_K[:, 0] = torch.tensor([2.0, 0, 0, 0])   # k_0  (mean-square 1 -> RMSNorm is identity here)
W_K[:, 1] = torch.tensor([0.0, 2, 2, 2])   # k_1  (mean-square 3)
W_K[:, 2] = torch.tensor([1.0, 1, 1, 1])   # k_2  (mean-square 1)

W_V = torch.zeros(N_KV_HEADS * HEAD_DIM, D_MODEL)
W_V[:, 0] = torch.tensor([1.0, 0, 0, 0])   # v_0
W_V[:, 1] = torch.tensor([0.0, 1, 0, 0])   # v_1
W_V[:, 2] = torch.tensor([1.0, 1, 1, 0])   # v_2

W_O = torch.eye(N_HEADS * HEAD_DIM)         # identity: read the per-head outputs directly


def _cfg(**overrides) -> ModelConfig:
    base = dict(
        vocab_size=10,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_kv_heads=N_KV_HEADS,
        max_seq_len=8,
        positional_scheme="nope",
        qk_norm=True,
        rope_theta=10000.0,
    )
    base.update(overrides)
    return ModelConfig(**base)


def _build(scheme: str = "nope", qk_norm: bool = True) -> Attention:
    attn = Attention(_cfg(positional_scheme=scheme, qk_norm=qk_norm))
    with torch.no_grad():
        attn.w_q.weight.copy_(W_Q)
        attn.w_k.weight.copy_(W_K)
        attn.w_v.weight.copy_(W_V)
        attn.w_o.weight.copy_(W_O)
    return attn


# ======================================================================
# 1. RMSNorm from scratch, on one head vector
# ======================================================================
_section("1. RMSNorm(head_dim=4) on the journal's q = [3, 0, 4, 0]")
rms = RMSNorm(HEAD_DIM)
q_demo = torch.tensor([3.0, 0.0, 4.0, 0.0])
mean_sq = q_demo.pow(2).mean()
print(f"  q                     = {q_demo.tolist()}")
print(f"  mean of squares       = (9 + 0 + 16 + 0) / 4 = {mean_sq.item()}")
print(f"  rsqrt(meansq + 1e-6)  = {torch.rsqrt(mean_sq + 1e-6).item():.10f}")
out_demo = rms(q_demo)
print(f"  RMSNorm(q)            = {out_demo.tolist()}   (hand: [1.2, 0, 1.6, 0])")
print(f"  ||RMSNorm(q)||_2      = {out_demo.norm().item():.10f}   (hand: sqrt(4) = 2)")
print(f"  mean-square of result = {out_demo.pow(2).mean().item():.10f}   (hand: 1)")
# eps makes it 1.19999992..., not exactly 1.2:
assert abs(out_demo[0].item() - 1.2) < 1e-6
assert abs(out_demo.norm().item() - 2.0) < 1e-6
# scale-invariance: 10x the input, same normalized output (to eps slack)
out_demo_10x = rms(10.0 * q_demo)
print(f"  RMSNorm(10 * q)       = {out_demo_10x.tolist()}")
print(f"  max|RMSNorm(10q) - RMSNorm(q)| = {(out_demo_10x - out_demo).abs().max().item():.2e}")


# ======================================================================
# 2. The full nope spine: every intermediate, both code paths
# ======================================================================
_section("2. nope spine -- forward_with_trace, every intermediate")
attn = _build("nope", qk_norm=True)
out_trace, trace = attn.forward_with_trace(X, POSITIONS)
out_fused = attn.forward(X, POSITIONS)

print("q  PRE-norm  (trace['q_pre_norm'], shape n_heads,seq,head_dim):\n", trace["q_pre_norm"])
print("q  POST-norm (trace['q'] -- captured after q_norm):\n", trace["q"])
print("k  PRE-norm  (trace['k_pre_norm'], shape n_kv_heads,seq,head_dim):\n", trace["k_pre_norm"])
print("k  POST-norm (trace['k']):\n", trace["k"])
print("v (trace['v']):\n", trace["v"])
print("scaled scores per head, pre-mask (trace['attn_scores_pre_mask']):\n",
      trace["attn_scores_pre_mask"])
print("probs after causal mask + softmax (trace['attn_scores_post_softmax']):\n",
      trace["attn_scores_post_softmax"])
print("output, explicit forward_with_trace:\n", out_trace)
print("output, fused forward (SDPA):\n", out_fused)
max_path_diff = (out_trace - out_fused).abs().max().item()
print(f"max |forward_with_trace - forward| = {max_path_diff:.3e}")
torch.testing.assert_close(out_trace, out_fused, atol=1e-5, rtol=0)

# The by-hand end-to-end output (doc Part 6). Filled from this script's own
# run -- transcribed to 8 dp, asserted to 5 dp.
EXPECTED = torch.tensor([[
    [1.00000000, 0.00000000, 0.00000000, 0.00000000, 1.00000000, 0.00000000, 0.00000000, 0.00000000],
    [0.56862396, 0.43137604, 0.00000000, 0.00000000, 0.32474494, 0.67525506, 0.00000000, 0.00000000],
    [0.78429329, 0.60785335, 0.39214665, 0.00000000, 0.50095916, 0.90090215, 0.40186128, 0.00000000],
]])
print("\nexpected (hand-computed, doc Part 6):\n", EXPECTED)
torch.testing.assert_close(out_trace, EXPECTED, atol=1e-5, rtol=0)
torch.testing.assert_close(out_fused, EXPECTED, atol=1e-5, rtol=0)
print("both paths match the by-hand output to 5 dp.  OK")


# ======================================================================
# 3. RMSNorm forces mean-square 1 and norm sqrt(head_dim) on every vector
# ======================================================================
_section("3. every post-norm q and k vector: mean-square 1, L2 norm 2")
q_post = trace["q"]          # (2, 3, 4)
k_post = trace["k"]          # (1, 3, 4)
for name, t in (("q", q_post), ("k", k_post)):
    ms = t.pow(2).mean(dim=-1)
    nrm = t.norm(dim=-1)
    print(f"  {name}: mean-square per vector =\n{ms}")
    print(f"  {name}: L2 norm per vector =\n{nrm}")
    torch.testing.assert_close(ms, torch.ones_like(ms), atol=1e-5, rtol=0)
    torch.testing.assert_close(nrm, torch.full_like(nrm, 2.0), atol=1e-5, rtol=0)
print("  every vector: mean-square 1, norm 2 = sqrt(head_dim).  OK")


# ======================================================================
# 4. Scale-invariance: 1x / 10x / 50x the layer input
# ======================================================================
_section("4. scale-invariance of the post-norm scores (qk_norm ON vs OFF)")
attn_on = _build("nope", qk_norm=True)
attn_off = _build("nope", qk_norm=False)
rows = []
base_scores_on = None
for scale in (1.0, 10.0, 50.0, 200.0):
    xs = X * scale
    _, tr_on = attn_on.forward_with_trace(xs, POSITIONS)
    _, tr_off = attn_off.forward_with_trace(xs, POSITIONS)
    s_on = tr_on["attn_scores_pre_mask"]
    s_off = tr_off["attn_scores_pre_mask"]
    if base_scores_on is None:
        base_scores_on = s_on.clone()
    drift = (s_on - base_scores_on).abs().max().item()
    rows.append((scale, s_off.std().item(), s_on.std().item(), drift))
print(f"  {'input scale':>12} | {'score std OFF':>14} | {'score std ON':>13} | {'|ON - ON@1x|':>13}")
for scale, off, on, drift in rows:
    print(f"  {scale:>12.0f} | {off:>14.4f} | {on:>13.6f} | {drift:>13.2e}")
# ON: essentially flat across a 200x input scale-up.
assert rows[-1][2] / rows[0][2] < 1.05
# OFF: grows ~ scale**2 (Cauchy-Schwarz quadratic ceiling).
ratio_off = rows[2][1] / rows[0][1]           # 50x row / 1x row
print(f"  OFF score-std ratio 50x / 1x = {ratio_off:.1f}   (~50^2 = 2500)")
assert ratio_off > 1000


# ======================================================================
# 5. The Cauchy-Schwarz ceiling on the post-norm dot product
# ======================================================================
_section("5. |q'.k'| <= head_dim = 4  for every (i, j)  (scaled: <= 2)")
q0 = q_post[0]                # head 0, (3, 4)
kk = k_post[0]                # the single kv head, (3, 4)
print(f"  {'i':>3} {'j':>3} | {'q_i.k_j':>10} | {'||q_i|| ||k_j||':>15} | {'/sqrt(4)':>10}")
for i in range(SEQ_LEN):
    for j in range(SEQ_LEN):
        dot = torch.dot(q0[i], kk[j]).item()
        bound = q0[i].norm().item() * kk[j].norm().item()
        print(f"  {i:>3} {j:>3} | {dot:>10.6f} | {bound:>15.6f} | {dot / 2.0:>10.6f}")
        assert abs(dot) <= bound + 1e-4
        assert abs(dot) <= HEAD_DIM + 1e-4
        assert abs(dot / math.sqrt(HEAD_DIM)) <= math.sqrt(HEAD_DIM) + 1e-4
print("  every scaled score is within +/- sqrt(head_dim) = 2.  OK")


# ======================================================================
# 6. RoPE: norm preserved through rotation; norm and rotation commute
# ======================================================================
_section("6. qk_norm + RoPE: rotation preserves norm; the two orders commute")
from qwenmoe.model.positional import RotaryEmbedding

rope = RotaryEmbedding(HEAD_DIM, theta=10000.0)
attn_rope = _build("rope", qk_norm=True)
_, tr_rope = attn_rope.forward_with_trace(X, POSITIONS)

# trace['q'] is captured AFTER q_norm but BEFORE RoPE (attention.py:210).
q_normed = tr_rope["q"].unsqueeze(0)          # (1, 2, 3, 4)
k_normed = tr_rope["k"].unsqueeze(0)          # (1, 1, 3, 4)
q_nr, k_nr = rope.apply_to_qk(q_normed, k_normed, POSITIONS)   # norm -> rotate
print("  ||q||_2 per vector, norm THEN rotate:\n", q_nr.norm(dim=-1))
torch.testing.assert_close(q_nr.norm(dim=-1), torch.full_like(q_nr.norm(dim=-1), 2.0),
                           atol=1e-5, rtol=0)

# rotate -> then norm (build the pre-norm q, rotate it, then RMSNorm)
q_pre = tr_rope["q_pre_norm"].unsqueeze(0)
k_pre = tr_rope["k_pre_norm"].unsqueeze(0)
q_rot_pre, k_rot_pre = rope.apply_to_qk(q_pre, k_pre, POSITIONS)
q_rn = attn_rope.q_norm(q_rot_pre)            # rotate -> norm
k_rn = attn_rope.k_norm(k_rot_pre)
print("  max |norm-then-rotate  -  rotate-then-norm|  (q):",
      f"{(q_nr - q_rn).abs().max().item():.2e}")
print("  max |norm-then-rotate  -  rotate-then-norm|  (k):",
      f"{(k_nr - k_rn).abs().max().item():.2e}")
torch.testing.assert_close(q_nr, q_rn, atol=1e-5, rtol=0)
torch.testing.assert_close(k_nr, k_rn, atol=1e-5, rtol=0)

# same-position (diagonal) scores: rotation of q_i and k_i by the SAME angle
# leaves q_i . k_i unchanged, so the rope diagonal == the nope diagonal.
diag_rope = torch.diagonal(tr_rope["attn_scores_pre_mask"], dim1=-2, dim2=-1)
diag_nope = torch.diagonal(trace["attn_scores_pre_mask"], dim1=-2, dim2=-1)
print("  rope diagonal scores:\n", diag_rope)
print("  nope diagonal scores:\n", diag_nope)
torch.testing.assert_close(diag_rope, diag_nope, atol=1e-5, rtol=0)
print("  norm survives the rotation, and the two orders agree at init (w = 1).  OK")


# ======================================================================
# 7. qk_norm=True changes the output vs qk_norm=False
# ======================================================================
_section("7. qk_norm ON vs OFF -- same w_q/w_k/w_v/w_o, different output")
out_on = _build("nope", qk_norm=True).forward(X, POSITIONS)
out_off = _build("nope", qk_norm=False).forward(X, POSITIONS)
print("  output qk_norm=OFF:\n", out_off)
print("  output qk_norm=ON :\n", out_on)
print(f"  max |on - off| = {(out_on - out_off).abs().max().item():.4f}")
assert not torch.allclose(out_on, out_off, atol=1e-3)
# and OFF constructs no q_norm / k_norm attributes at all (Requirement 7.5)
attn_off_obj = _build("nope", qk_norm=False)
assert not hasattr(attn_off_obj, "q_norm") and not hasattr(attn_off_obj, "k_norm")
print("  qk_norm=False builds no q_norm/k_norm submodules at all.  OK")


# ======================================================================
# 8. Softmax entropy collapse (the failure mode QK-norm targets)
# ======================================================================
_section("8. attention-row entropy H(p) = -sum p log p, ON vs OFF, by input scale")


def _row_entropies(attn_mod, scale):
    _, tr = attn_mod.forward_with_trace(X * scale, POSITIONS)
    probs = tr["attn_scores_post_softmax"]          # (2, 3, 3), rows already causal-masked
    ent = -(probs.clamp_min(1e-12).log() * probs).sum(dim=-1)
    return probs, ent


for scale in (1.0, 50.0):
    p_off, e_off = _row_entropies(attn_off, scale)
    p_on, e_on = _row_entropies(attn_on, scale)
    print(f"\n  --- input scale {scale:.0f}x ---")
    print(f"  OFF probs, head 1, query row 2 (keys 0,1,2): "
          f"{[round(x, 5) for x in p_off[1, 2].tolist()]}   H = {e_off[1, 2].item():.5f}")
    print(f"  ON  probs, head 1, query row 2 (keys 0,1,2): "
          f"{[round(x, 5) for x in p_on[1, 2].tolist()]}   H = {e_on[1, 2].item():.5f}")
    print(f"  mean entropy over all non-trivial rows -- OFF {e_off[:, 1:].mean().item():.5f}"
          f"   ON {e_on[:, 1:].mean().item():.5f}   (max possible on 3 keys = ln 3 = {math.log(3):.5f})")

_, e_off_1x = _row_entropies(attn_off, 1.0)
_, e_off_50x = _row_entropies(attn_off, 50.0)
_, e_on_1x = _row_entropies(attn_on, 1.0)
_, e_on_50x = _row_entropies(attn_on, 50.0)
# OFF: last query row's entropy collapses toward 0 as scale grows.
assert e_off_50x[1, 2].item() < 0.05
assert e_off_1x[1, 2].item() > 5 * e_off_50x[1, 2].item()
# ON: essentially unchanged across the 50x scale-up.
assert abs(e_on_1x[1, 2].item() - e_on_50x[1, 2].item()) < 1e-3
print("\n  OFF row entropy collapses ~50x-scale -> near 0; ON row entropy is flat.  OK")


print("\nAll assertions passed -- every number in 022-qk-norm.md is real.")
