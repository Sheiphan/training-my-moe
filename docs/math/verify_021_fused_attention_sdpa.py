"""Reproduces every hand-computed number in `021-fused-attention-sdpa.md`.

Run it:  uv run python docs/math/verify_021_fused_attention_sdpa.py

ONE running example, carried end to end -- the SAME tiny `Attention` module
as `docs/math/004-attention-gqa.md` (d_model=4, n_heads=2, n_kv_heads=1,
seq_len=3), pinned to the SAME four weight matrices and the SAME 3-token
input. Doc 004 derives the explicit `forward_with_trace` path digit by
digit; this file/doc picks up where `forward` (the fused
`F.scaled_dot_product_attention` path, moe-modernization task 5) diverges
from it and shows -- by hand and against the real module -- that the fused
path computes the identical mathematics.

It asserts:

  * `forward` (fused) and `forward_with_trace` (explicit) agree for all
    three positional schemes, and prints the measured max |difference|
    (float32-noise-sized, orders of magnitude under the 1e-5 the tests
    enforce);
  * the by-hand end-to-end output tensor for `nope` (doc 004's final box);
  * the ONLINE (streaming / FlashAttention) softmax recurrence -- running
    max `m`, running denominator `l`, running weighted-value accumulator
    `o`, with the `alpha = exp(m_old - m_new)` rescale -- reproduces
    one-shot `softmax(scores) @ v` EXACTLY, row by row, for every query row
    of both heads;
  * `enable_gqa=True` (SDPA broadcasting the 1 kv head to 2 query heads
    internally) equals an explicit `repeat_interleave` then plain SDPA;
  * the ALiBi branch's `attn_mask = causal(-inf) + alibi_bias` equals the
    explicit path's `scores + alibi_bias + causal(-inf)` -- addition is
    commutative/associative and `-inf + finite = -inf` (no NaN);
  * floating-point addition is not associative (why the two paths agree
    only to ~1e-7, not bit-for-bit);
  * a direct wall-clock speed comparison on this repo's BASELINE-sized
    attention shapes (fused SDPA vs. a hand copy of the pre-task-5 manual
    softmax code).

No fabricated numbers -- this file is the source of truth the prose quotes.
"""
from __future__ import annotations

import math
import time

import torch
import torch.nn.functional as F

from qwenmoe.config import ModelConfig
from qwenmoe.model.attention import Attention, _causal_position_mask
from qwenmoe.model.positional import _alibi_slopes

torch.manual_seed(0)
torch.set_printoptions(precision=6, sci_mode=False)


def _section(title: str) -> None:
    print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)


# ======================================================================
# The running example -- identical to docs/math/004-attention-gqa.md
# ======================================================================
W_Q = torch.tensor([[1.0, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1], [1, 0, 0, 1]])
W_K = torch.tensor([[1.0, 0, 1, 0], [0, 1, 0, 1]])
W_V = torch.tensor([[1.0, 1, 0, 0], [0, 0, 1, 1]])
W_O = torch.tensor([[1.0, 0, 1, 0], [0, 1, 0, 1], [1, 0, 0, 1], [0, 1, 1, 0]])

X = torch.tensor([[[1.0, 0, 1, 0], [0, 2, 0, 1], [1, 1, 0, 0]]])  # (1, 3, 4)
POSITIONS = torch.arange(3)


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


def _build(scheme: str) -> Attention:
    attn = Attention(_cfg(positional_scheme=scheme))
    with torch.no_grad():
        attn.w_q.weight.copy_(W_Q)
        attn.w_k.weight.copy_(W_K)
        attn.w_v.weight.copy_(W_V)
        attn.w_o.weight.copy_(W_O)
    return attn


# ======================================================================
# 1. Fused `forward` vs explicit `forward_with_trace`, all three schemes
# ======================================================================
_section("1. fused forward  vs  explicit forward_with_trace")

diffs = {}
outs = {}
for scheme in ("nope", "rope", "alibi"):
    attn = _build(scheme)
    out_trace, trace = attn.forward_with_trace(X, POSITIONS)
    out_fused = attn.forward(X, POSITIONS)
    d = (out_trace - out_fused).abs().max().item()
    diffs[scheme] = d
    outs[scheme] = out_trace
    print(f"\n--- positional_scheme = {scheme!r} ---")
    print("scaled scores per head, pre-mask (trace):\n", trace["attn_scores_pre_mask"])
    print("probs after causal mask + softmax (trace):\n", trace["attn_scores_post_softmax"])
    print("forward_with_trace output:\n", out_trace)
    print("fused forward output:\n", out_fused)
    print(f"max |trace - fused| = {d:.3e}")
    torch.testing.assert_close(out_trace, out_fused, atol=1e-5, rtol=0)

print("\nmeasured max |forward - forward_with_trace| (this module):")
for scheme, d in diffs.items():
    print(f"  {scheme:5s}  {d:.2e}   ({d / 1e-5:.3f} x the 1e-5 test tolerance)")

# ======================================================================
# 2. The by-hand end-to-end output for `nope` (doc 004's final box)
# ======================================================================
_section("2. ASSERT the by-hand end-to-end output (nope spine)")
EXPECTED_NOPE = torch.tensor([[
    [2.000000, 2.000000, 2.000000, 2.000000],
    [3.474192, 2.000000, 2.804430, 2.669762],
    [3.408561, 1.573375, 2.318140, 2.663796],
]])
print("expected (hand-computed, doc 004 + doc 021):\n", EXPECTED_NOPE)
torch.testing.assert_close(outs["nope"], EXPECTED_NOPE, atol=1e-4, rtol=0)
attn_nope = _build("nope")
torch.testing.assert_close(attn_nope.forward(X, POSITIONS), EXPECTED_NOPE, atol=1e-4, rtol=0)
print("both forward and forward_with_trace match to 4 dp.  OK")

# ======================================================================
# 3. Online (streaming) softmax == one-shot softmax, exactly, every row
# ======================================================================
_section("3. ONLINE softmax recurrence  ==  one-shot softmax @ v")


def online_softmax_attention(scores_row: torch.Tensor, values: torch.Tensor,
                             block_size: int = 1, verbose: bool = False):
    """The FlashAttention recurrence, by hand, for one query row.

    scores_row : (n_keys,)      -- already scaled (QK^T / sqrt(d_k)), causal
                                   entries dropped (only keys 0..i passed in)
    values     : (n_keys, d_k)
    Maintains running max m, running denominator l, running weighted-value
    accumulator o; rescales the accumulators by alpha = exp(m_old - m_new)
    whenever a new block raises the max.  Returns o / l.
    """
    n_keys, d_k = values.shape
    m = torch.tensor(float("-inf"))
    l = torch.tensor(0.0)
    o = torch.zeros(d_k)
    for start in range(0, n_keys, block_size):
        blk = slice(start, start + block_size)
        s_blk = scores_row[blk]                     # (b,)
        v_blk = values[blk]                         # (b, d_k)
        m_local = s_blk.max()
        m_new = torch.maximum(m, m_local)
        alpha = torch.exp(m - m_new) if torch.isfinite(m) else torch.tensor(0.0)
        p_blk = torch.exp(s_blk - m_new)            # (b,)
        l = alpha * l + p_blk.sum()
        o = alpha * o + (p_blk[:, None] * v_blk).sum(dim=0)
        if verbose:
            s_disp = [round(x, 6) for x in s_blk.tolist()]
            o_disp = [round(x, 6) for x in o.tolist()]
            print(f"    block keys [{start}:{start + len(s_blk)})  s={s_disp}"
                  f"  m_new={m_new.item():.6f}  alpha={alpha.item():.6f}"
                  f"  l={l.item():.6f}  o={o_disp}")
        m = m_new
    return o / l


# Rebuild the nope scaled scores and values by hand-matching the trace.
attn = _build("nope")
_, trace = attn.forward_with_trace(X, POSITIONS)
scaled_scores = trace["attn_scores_pre_mask"]        # (2, 3, 3), pre-mask, scaled
# values per head: with n_kv_heads=1 both heads share v^(0)
v_head = torch.tensor([[1.0, 1.0], [2.0, 1.0], [2.0, 0.0]])  # doc 004 step 1c
probs = trace["attn_scores_post_softmax"]             # (2, 3, 3) one-shot softmax
one_shot_out = probs @ torch.stack([v_head, v_head])  # (2, 3, 2)

print("featured walkthrough -- head 1, query row 2, block_size = 1:")
row = scaled_scores[1, 2, :3]   # keys 0,1,2 all visible for row 2
print(f"  scaled scores  = {[round(x, 6) for x in row.tolist()]}")
print(f"  values         = {[[round(x, 1) for x in r] for r in v_head.tolist()]}")
o_stream = online_softmax_attention(row, v_head, block_size=1, verbose=True)
print(f"  final: o / l = {[round(x, 6) for x in o_stream.tolist()]}   (last accumulator / last denominator)")
print(f"  one-shot softmax(scores) @ v = {[round(x, 6) for x in one_shot_out[1, 2].tolist()]}")
torch.testing.assert_close(o_stream, one_shot_out[1, 2], atol=1e-6, rtol=0)

print("\nevery query row, both heads, block sizes 1 and 2:")
for h in range(2):
    for i in range(3):
        visible = scaled_scores[h, i, : i + 1]
        vv = v_head[: i + 1]
        for bs in (1, 2):
            got = online_softmax_attention(visible, vv, block_size=bs)
            torch.testing.assert_close(got, one_shot_out[h, i], atol=1e-6, rtol=0)
        print(f"  head {h} row {i}: streamed == one-shot  "
              f"({[round(x, 6) for x in one_shot_out[h, i].tolist()]})  OK")

# ======================================================================
# 4. enable_gqa=True  ==  explicit repeat_interleave + plain SDPA
# ======================================================================
_section("4. enable_gqa=True  ==  repeat_interleave then plain SDPA")
attn = _build("nope")
q = attn.w_q(X).view(1, 3, 2, 2).transpose(1, 2)   # (1, 2, 3, 2)
k = attn.w_k(X).view(1, 3, 1, 2).transpose(1, 2)   # (1, 1, 3, 2)
vv = attn.w_v(X).view(1, 3, 1, 2).transpose(1, 2)  # (1, 1, 3, 2)
out_gqa = F.scaled_dot_product_attention(q, k, vv, is_causal=True, enable_gqa=True)
k_rep = k.repeat_interleave(2, dim=1)              # (1, 2, 3, 2)  -- what forward_with_trace does
v_rep = vv.repeat_interleave(2, dim=1)
out_rep = F.scaled_dot_product_attention(q, k_rep, v_rep, is_causal=True)
print("k shape into enable_gqa path :", tuple(k.shape), " (never expanded)")
print("k shape after repeat_interleave:", tuple(k_rep.shape))
print("max |enable_gqa - repeat_interleave| =",
      (out_gqa - out_rep).abs().max().item())
torch.testing.assert_close(out_gqa, out_rep, atol=1e-6, rtol=0)
# and the index map repeat_interleave uses: query head h -> kv head h // n_rep
n_rep = 2
qh_to_kv = [h // n_rep for h in range(2)]
print(f"query-head -> kv-head map (h // n_rep, n_rep={n_rep}): {qh_to_kv}")
assert qh_to_kv == [0, 0]

# ======================================================================
# 5. ALiBi branch: attn_mask = causal(-inf) + alibi_bias   (fold-in)
# ======================================================================
_section("5. ALiBi fold-in: (scores + bias) + causal  ==  scores + (causal + bias)")
slopes = torch.tensor(_alibi_slopes(2))
print("alibi slopes (n_heads=2):", slopes.tolist(), " = [2^-4, (2^-4)^2]")
i_idx = torch.arange(3)[:, None]
j_idx = torch.arange(3)[None, :]
bias = (j_idx - i_idx).float()[None] * slopes[:, None, None]   # (2, 3, 3), (j-i)*slope[h]
causal_triu = torch.triu(torch.full((3, 3), float("-inf")), diagonal=1)  # forward_with_trace
causal_posn = _causal_position_mask(POSITIONS, 3, torch.device("cpu"), torch.float32)  # forward
print("\n_causal_position_mask(arange(3), 3) == triu(-inf, diag=1)? ",
      torch.equal(causal_posn[0], causal_triu))

attn = _build("alibi")
_, tr = attn.forward_with_trace(X, POSITIONS)
raw_scaled = (tr["attn_scores_pre_mask"] - bias)   # undo the +bias the trace already did

# explicit-path order:  (scores + bias) + causal
A = (raw_scaled + bias) + causal_triu
# fused-path order:      scores + (causal + bias)
B = raw_scaled + (causal_triu + bias)
finite = torch.isfinite(A)
print("finite entries equal   :", torch.allclose(A[finite], B[finite], atol=0, rtol=0))
print("masked entries -inf both:", bool((~finite == ~torch.isfinite(B)).all()))
print("any NaN (would mean -inf + +inf somewhere):", bool(A.isnan().any() or B.isnan().any()))
torch.testing.assert_close(torch.softmax(A, dim=-1), torch.softmax(B, dim=-1), atol=0, rtol=0)
print("softmax(A) == softmax(B) bit-for-bit:  True")

# ======================================================================
# 6. Floating-point addition is not associative
# ======================================================================
_section("6. float32:  (a + b) + c  !=  a + (b + c)")
a = torch.tensor(1.0, dtype=torch.float32)
b = torch.tensor(-1.0, dtype=torch.float32)
c = torch.tensor(1e-8, dtype=torch.float32)
left = (a + b) + c
right = (a + c) + b
print(f"  (1.0 + -1.0) + 1e-8 = {left.item():.3e}   <- the 1e-8 survives")
print(f"  (1.0 + 1e-8) + -1.0 = {right.item():.3e}   <- 1.0 + 1e-8 rounds back to 1.0, 1e-8 lost")
assert left.item() != right.item()
# a longer reduction in two orders -> different last bits
g = torch.randn(4096, dtype=torch.float32)
fwd = torch.tensor(0.0)
for x in g:
    fwd = fwd + x
rev = torch.tensor(0.0)
for x in reversed(g):
    rev = rev + x
print(f"  sum(4096 randn) left-to-right = {fwd.item():.8f}")
print(f"  sum(4096 randn) right-to-left = {rev.item():.8f}")
print(f"  difference (pure summation order) = {abs(fwd.item() - rev.item()):.2e}")

# ======================================================================
# 7. Wall-clock: fused SDPA vs the pre-task-5 manual softmax code
# ======================================================================
_section("7. speed -- fused SDPA vs a hand copy of the pre-task-5 manual softmax")


def _manual_attention(q, k, v, n_rep):
    """Verbatim shape of the old (pre-task-5) explicit code path."""
    k = k.repeat_interleave(n_rep, dim=1)
    v = v.repeat_interleave(n_rep, dim=1)
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    s = q.shape[-2]
    mask = torch.triu(torch.full((s, s), float("-inf")), diagonal=1)
    scores = scores + mask
    probs = torch.softmax(scores, dim=-1)
    return probs @ v


bcfg = ModelConfig(vocab_size=10, d_model=256, n_heads=8, n_kv_heads=4,
                   max_seq_len=256, positional_scheme="nope")
battn = Attention(bcfg)
bx = torch.randn(4, 256, 256)
bpos = torch.arange(256)


def _time(fn, n=50, warmup=5):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n * 1e3  # ms/call


with torch.no_grad():
    q = battn.w_q(bx).view(4, 256, 8, 32).transpose(1, 2)
    k = battn.w_k(bx).view(4, 256, 4, 32).transpose(1, 2)
    v = battn.w_v(bx).view(4, 256, 4, 32).transpose(1, 2)
    t_fused = _time(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True))
    t_manual = _time(lambda: _manual_attention(q, k, v, 2))
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
    man = _manual_attention(q, k, v, 2)
print(f"  BASELINE shape: d_model=256, n_heads=8, n_kv_heads=4, batch=4, seq_len=256, CPU")
print(f"  fused SDPA        : {t_fused:.3f} ms/call")
print(f"  manual softmax    : {t_manual:.3f} ms/call")
print(f"  speedup           : {t_manual / t_fused:.2f}x")
print(f"  max |fused - manual| on the output: {(ref - man).abs().max().item():.2e}")

print("\nAll assertions passed -- every number in 021-fused-attention-sdpa.md is real.")
