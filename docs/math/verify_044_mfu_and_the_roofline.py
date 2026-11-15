"""Reproduces every derived number in `044-mfu-and-the-roofline.md`.

Run it:  uv run python docs/math/verify_044_mfu_and_the_roofline.py

Journals 044 (first real H100 session) and 045 (what to optimize next) contain
two kinds of number, and this script keeps them apart:

  MEASURED on the H100 (throughputs in tok/s, profiler milliseconds, the
  2.48x compile speed-up). This script CANNOT re-run them; it takes them as
  inputs, labelled MEASURED, and derives everything that follows from them.

  DERIVED (FLOPs per token, MFU, hours, dollars, profile shares, the roofline
  traffic, arithmetic intensity, the capacity-factor tail, the dense-expert
  trade). These are recomputed here from the REAL model (`count_params` on the
  real SCALE_300M preset) and the real `dispatch_batched`, and asserted
  against the journal's figures, with every discrepancy printed rather than
  hidden.

It also checks two assumptions the journal makes silently (the profiler-vs-wall
scaling, and per-expert counts under top-2 of distinct experts).
"""
from __future__ import annotations

import dataclasses
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from configs.presets import SCALE_300M  # noqa: E402
from qwenmoe.model.moe import dispatch_batched  # noqa: E402
from qwenmoe.model.transformer import QwenMoEModel  # noqa: E402
from qwenmoe.train.scaling import count_params  # noqa: E402


def hr(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


PEAK_FLOPS = 989e12        # H100 SXM dense bf16 peak (vendor figure, as the journal uses)
HBM_BW = 3.35e12           # H100 SXM HBM3 bandwidth, bytes/s (vendor figure)
PRICE = 3.49               # $/hr, RunPod SECURE tier (journal 044)
BUDGET = 2.69e9            # training tokens the journal prices
B, S = 16, 1024            # the measured micro-batch
T = B * S                  # 16,384 tokens

cfg = dataclasses.replace(SCALE_300M, vocab_size=4097)
pb = count_params(QwenMoEModel(cfg))

# ----------------------------------------------------------------------
# Part 1 -- FLOPs per token
# ----------------------------------------------------------------------
hr("Part 1 -- FLOPs per token")

active = pb.active
embed_table = cfg.vocab_size * cfg.d_model           # tied: counted once as the lm_head, not as "active non-embedding"
non_embed = active - embed_table
head = cfg.vocab_size * cfg.d_model
attn_scores = 6 * cfg.n_layers * S * cfg.d_model
flops_tok = 6 * non_embed + 6 * head + attn_scores
print(f"active params               = {active:,}")
print(f"- embedding table (V x d)   = {embed_table:,}")
print(f"active non-embedding        = {non_embed:,}   (journal: 121,975,808)")
print(f"6 x non-embedding           = {6 * non_embed:,}")
print(f"6 x tied lm_head            = {6 * head:,}   (journal: 6 x 3,146,496)")
print(f"6 x L x S x d (attn scores) = {attn_scores:,}   (6 x {cfg.n_layers} x {S} x {cfg.d_model})")
print(f"FLOPs per token             = {flops_tok:,}   (journal: 797,919,744)")
assert non_embed == 121_975_808 and flops_tok == 797_919_744
print("PASS  matches journal 044 to the digit, from the real preset and real count_params")

# ----------------------------------------------------------------------
# Part 2 -- MFU, hours, dollars from the MEASURED throughputs
# ----------------------------------------------------------------------
hr("Part 2 -- MFU table (inputs: MEASURED tok/s)")

def flops_per_token(seq: int) -> int:
    """6N over non-embedding + 6 x tied head + 6 x L x S x d: only the attention-score term depends on S."""
    return 6 * non_embed + 6 * head + 6 * cfg.n_layers * seq * cfg.d_model


MEASURED = [  # label, seq_len, tok/s measured on the H100 (journal 044)
    ("B=8  x 1024 bf16", 1024, 73_313),
    ("B=16 x 1024 bf16", 1024, 82_323),
    ("B=32 x  512 bf16", 512, 91_625),
    ("B=64 x  256 bf16", 256, 100_504),
    ("B=16 x 1024 fp32", 1024, 57_721),
]
JOURNAL = {  # tflops, mfu %, hours, dollars as printed in journal 044
    "B=8  x 1024 bf16": (58.5, 5.9, 10.20, 35.60),
    "B=16 x 1024 bf16": (65.7, 6.6, 9.08, 31.69),
    "B=32 x  512 bf16": (70.9, 7.2, 8.16, 28.48),
    "B=64 x  256 bf16": (80.2, 8.1, 7.44, 25.96),
    "B=16 x 1024 fp32": (46.1, 4.7, 12.95, 45.20),
}
print(f"FLOPs/token by sequence length: " + ", ".join(f"S={q}: {flops_per_token(q) / 1e6:.1f}M" for q in (1024, 512, 256)))
print(f"{'config':18} {'tok/s':>9} {'MFLOP/tok':>10} {'TFLOP/s':>8} {'MFU':>6} {'hours':>6} {'cost':>7}")
mismatch = []
for label, seq, tps in MEASURED:
    ft = flops_per_token(seq)
    tflops = tps * ft / 1e12
    mfu = 100 * tflops * 1e12 / PEAK_FLOPS
    hours = BUDGET / tps / 3600
    cost = hours * PRICE
    print(f"{label:18} {tps:>9,} {ft / 1e6:>10.1f} {tflops:>8.1f} {mfu:>5.1f}% {hours:>6.2f} ${cost:>6.2f}")
    j = JOURNAL[label]
    if (round(tflops, 1), round(mfu, 1)) != (j[0], j[1]):
        mismatch.append((label, tflops, mfu, j))
    assert abs(hours / j[2] - 1) < 0.003 and abs(cost / j[3] - 1) < 0.003, (label, hours, cost, j)  # 0.3%
print("hours and dollars: all five rows within 0.3% of the journal")
implied = [JOURNAL[lab][2] * 3600 * tps for lab, _seq, tps in MEASURED]
print(f"      (the journal's hours imply a budget of {min(implied) / 1e9:.4f}-{max(implied) / 1e9:.4f}e9 tokens per row; the text "
      f"rounds it to '2.69B', used here -- hence the 0.01-hour differences)")

assert [m[0] for m in mismatch] == ["B=64 x  256 bf16"], mismatch
lab, tf_c, mfu_c, j = mismatch[0]
print(f"\nDISCREPANCY in journal 044's table, row '{lab}':")
print(f"  journal prints {j[0]} TFLOP/s and {j[1]}% MFU; recomputed from S=256 it is {tf_c:.1f} TFLOP/s and {mfu_c:.1f}% MFU")
print(f"  the journal's figure equals tok/s x the S=1024 FLOPs/token: {100_504 * flops_per_token(1024) / 1e12:.1f} TFLOP/s "
      f"-> {100 * 100_504 * flops_per_token(1024) / PEAK_FLOPS:.1f}%  (the 512 row, by contrast, correctly uses S=512)")
assert round(100_504 * flops_per_token(1024) / 1e12, 1) == j[0] and round(100 * 100_504 * flops_per_token(1024) / PEAK_FLOPS, 1) == j[1]
assert round(tf_c, 1) == 76.6 and round(mfu_c, 1) == 7.7
print("  the headline (B=16 x 1024 at 6.6%) and every hours/dollars figure are unaffected; only this row's TFLOP/s and MFU shift")
print("PASS  4 of 5 rows reproduce exactly; the 5th is explained and corrected above")

ratio_bf16_fp32 = 82_323 / 57_721
print(f"bf16 / fp32 at B=16 x 1024 = {ratio_bf16_fp32:.3f}   (journal: 1.43x)")
assert round(ratio_bf16_fp32, 2) == 1.43

EAGER_TPS, COMPILED_TPS = 77_770, 192_832   # MEASURED, the compile A/B run
eager_ms, comp_ms = T / EAGER_TPS * 1e3, T / COMPILED_TPS * 1e3
comp_mfu = 100 * COMPILED_TPS * flops_tok / PEAK_FLOPS
comp_hours = BUDGET / COMPILED_TPS / 3600
print(f"\ncompile A/B: eager {eager_ms:.1f} ms/step -> compiled {comp_ms:.1f} ms/step; "
      f"speed-up {COMPILED_TPS / EAGER_TPS:.2f}x; MFU {100 * EAGER_TPS * flops_tok / PEAK_FLOPS:.1f}% -> {comp_mfu:.1f}%")
print(f"compiled run: {comp_hours:.2f} h, ${comp_hours * PRICE:.2f}   (journal: 3.88 h, $13.53)")
assert round(eager_ms, 1) == 210.7 and round(comp_ms, 1) == 85.0
assert round(COMPILED_TPS / EAGER_TPS, 2) == 2.48 and round(comp_mfu, 1) == 15.6 and abs(comp_hours - 3.88) < 0.011
eager_mfu = 100 * EAGER_TPS * flops_tok / PEAK_FLOPS
print(f"  note: journal 044's A/B table prints the eager MFU as 6.5%; from its own 77,770 tok/s it is {eager_mfu:.2f}%, "
      f"which is the 6.3% journal 045 uses")
assert round(eager_mfu, 1) == 6.3
cost_c = comp_hours * PRICE
print(f"  note: {comp_hours:.3f} h / ${cost_c:.2f} here vs the journal's 3.88 h / $13.53 -- the same 2.69B-vs-2.6917B rounding as above")
assert abs(cost_c - 13.53) < 0.02

at30 = BUDGET / (0.30 * PEAK_FLOPS / flops_tok) / 3600
print(f"\nwhat the 30% assumption would give on these inputs: {at30:.2f} h, ${at30 * PRICE:.2f}")
print("  journal 042 quoted 347,345 tok/s and 2.15 h at 30%: that implies "
      f"{0.30 * PEAK_FLOPS / 347_345 / 1e6:.1f} MFLOP/token, vs {flops_tok / 1e6:.1f} here, so 042 used a different FLOPs-per-token")
print("  convention (not reproduced); the 4-5x shortfall in 044 is measured against 042's number, not this one")

# ----------------------------------------------------------------------
# Part 3 -- the profile (inputs: MEASURED profiler milliseconds)
# ----------------------------------------------------------------------
hr("Part 3 -- profile shares (inputs: MEASURED profiler ms, 3 steps, B=16 x 1024)")

TOTAL = 1115.2
PROFILE = [
    ("aten::_index_put_impl_", 241.46, "scatter"),
    ("indexing_backward_kernel", 212.62, "scatter"),
    ("aten::mul", 76.19, "other"),
    ("aten::copy_", 53.69, "other"),
    ("indexing_backward_kernel (2)", 26.31, "scatter"),
    ("aten::bmm", 36.30, "matmul"),
    ("aten::mm", 33.77, "matmul"),
    ("nvjet_sm90 (tensor-core GEMM)", 11.62, "matmul"),
]
for name, ms, _ in PROFILE:
    print(f"  {name:32} {ms:>8.2f} ms  {100 * ms / TOTAL:>5.1f}%")
scatter_ms = sum(ms for _, ms, k in PROFILE if k == "scatter")
matmul_ms = sum(ms for _, ms, k in PROFILE if k == "matmul")
scatter_share, matmul_share = scatter_ms / TOTAL, matmul_ms / TOTAL
print(f"scatter/gather = {scatter_ms:.2f} ms = {100 * scatter_share:.1f}% of GPU time   (journal: ~43%)")
print(f"matmul         = {matmul_ms:.2f} ms = {100 * matmul_share:.1f}%   (journal: ~7%)")
assert round(100 * scatter_share) == 43 and round(100 * matmul_share) == 7
print(f"tensor cores idle ~ {100 * (1 - matmul_share):.0f}% of GPU time")

per_step_profiled = TOTAL / 3
print(f"\nASSUMPTION CHECK: profiled GPU time per step = {TOTAL}/3 = {per_step_profiled:.1f} ms, "
      f"but the UNprofiled eager step is {eager_ms:.1f} ms")
print(f"  -> the profiler run is {per_step_profiled / eager_ms:.2f}x slower than the real step, so the profile's "
      f"SHARES are being applied to the unprofiled wall time (journal 045 does exactly this).")
assert per_step_profiled > eager_ms

# ----------------------------------------------------------------------
# Part 4 -- the roofline
# ----------------------------------------------------------------------
hr("Part 4 -- the roofline")

k, d, L, E, bytes_el = cfg.top_k, cfg.d_model, cfg.n_layers, cfg.n_routed_experts, 2
P = T * k
fwd_elems = (3 * P + T) * d
print(f"T = {T:,} tokens, k = {k}, d = {d}, L = {L}, E = {E}, b = {bytes_el} (bf16)")
print(f"P = T*k = {P:,} dispatch pairs;  3P + T = {3 * P + T:,};  x d = {fwd_elems:,} elements")
per_layer_fwd = fwd_elems * bytes_el
print(f"bytes / layer / forward = {per_layer_fwd:,} ({per_layer_fwd / 1e6:.1f} MB)")
assert per_layer_fwd == 176_160_768


def traffic(mult: int) -> float:
    return mult * L * per_layer_fwd


t_mem = traffic(3) / HBM_BW * 1e3
print(f"x L x 3 = {traffic(3) / 1e9:.3f} GB/step;  at {HBM_BW / 1e12:.2f} TB/s -> {t_mem:.3f} ms")
assert round(traffic(3) / 1e9, 3) == 5.285 and round(t_mem, 3) == 1.578

measured_share_journal = 0.432
for share, tag in ((measured_share_journal, "journal's rounded 43.2%"), (scatter_share, f"exact {100 * scatter_share:.2f}%")):
    perm_ms = share * eager_ms
    print(f"permutation time ({tag}) = {perm_ms:.1f} ms  ->  {perm_ms / t_mem:.1f}x off the bandwidth roof")
assert round(measured_share_journal * eager_ms / t_mem, 1) == 57.7
print("\nsensitivity to the (empirical) backward-traffic multiplier:")
for mult in (2, 3, 4):
    r = measured_share_journal * eager_ms / (traffic(mult) / HBM_BW * 1e3)
    print(f"  multiplier {mult}: {traffic(mult) / 1e9:.2f} GB/step, {traffic(mult) / HBM_BW * 1e3:.2f} ms, {r:.1f}x off")
assert round(measured_share_journal * eager_ms / (traffic(2) / HBM_BW * 1e3)) == 87
assert round(measured_share_journal * eager_ms / (traffic(4) / HBM_BW * 1e3)) == 43
print("PASS  57.7x at 3x, spanning 43x..87x over 4x..2x -- the verdict survives the range")

# padding check on the REAL dispatch_batched: does the (K, G) grid gather P*d elements, or more?
torch.manual_seed(0)
n_tok, dd, ff = 4096, 16, 32
experts = torch.nn.ModuleList(
    [torch.nn.Module() for _ in range(E)]
)
for ex in experts:
    ex.w_gate = torch.nn.Linear(dd, ff, bias=False)
    ex.w_up = torch.nn.Linear(dd, ff, bias=False)
    ex.w_down = torch.nn.Linear(ff, dd, bias=False)
scores = torch.rand(n_tok, E)
topk_w, topk_i = scores.topk(k, dim=-1)
counts = torch.bincount(topk_i.reshape(-1), minlength=E)
K, G = int((counts > 0).sum()), int(counts.max())
x = torch.randn(n_tok, dd)
out = dispatch_batched(x, topk_i, torch.softmax(topk_w, -1), experts, "swiglu")
assert out.shape == x.shape
pad = K * G / (n_tok * k)
print(f"real dispatch_batched, uniform top-{k} of {E}, T={n_tok}: K*G = {K}*{G} = {K * G} slots for P = {n_tok * k} pairs "
      f"-> padded grid is {pad:.3f}x the unpadded P*d the formula assumes")
assert 1.0 <= pad < 1.05

# intensity
I = 3 * cfg.d_ff / bytes_el
flops_pair = 3 * 2 * d * cfg.d_ff
bytes_pair = 2 * d * bytes_el
ridge = PEAK_FLOPS / HBM_BW
print(f"\nper dispatch pair: FLOPs = 3 x 2 x d x d_ff = {flops_pair:,}; bytes = 2 x d x b = {bytes_pair:,}")
print(f"intensity I = 3*d_ff/b = {I:,.0f} FLOP/byte;  ridge = {PEAK_FLOPS / 1e12:.0f}/{HBM_BW / 1e12:.2f} = {ridge:.1f};  "
      f"I / ridge = {I / ridge:.1f}x")
assert I == 2304 and flops_pair / bytes_pair == I and round(ridge) == 295 and round(I / ridge, 1) == 7.8
print("PASS  2,304 FLOP/byte, 7.8x past the ridge (weights traffic ignored: reused across the pairs of an expert)")

# ----------------------------------------------------------------------
# Part 5 -- capacity factor and the binomial
# ----------------------------------------------------------------------
hr("Part 5 -- fixed-capacity dispatch: how far out is the capacity?")


def log_binom_pmf(n: int, p: float, x: int) -> float:
    return (math.lgamma(n + 1) - math.lgamma(x + 1) - math.lgamma(n - x + 1)
            + x * math.log(p) + (n - x) * math.log1p(-p))


def tail_prob(n: int, p: float, c: int) -> float:
    """exact P[X >= c], summed in log space."""
    logs = [log_binom_pmf(n, p, x) for x in range(c, n + 1)]
    mx = max(logs)
    return math.exp(mx) * sum(math.exp(v - mx) for v in logs)


# journal 045's model: P independent pair draws, each to one of E experts
mu, sigma = P / E, math.sqrt(P * (1 / E) * (1 - 1 / E))
print(f"journal model  X ~ Binomial(P={P:,}, 1/{E}): mu = {mu:,.0f}, sigma = {sigma:.1f}")
for cf in (1.25, 1.05):
    cap = math.ceil(cf * mu)
    z = (cap - mu) / sigma
    print(f"  cf {cf}: capacity {cap:,} = {z:.1f} sigma;  exact tail P[X >= cap] = {tail_prob(P, 1 / E, cap):.2e}")
assert round(sigma, 1) == 59.9 and math.ceil(1.25 * mu) == 5120 and math.ceil(1.05 * mu) == 4301
assert round((5120 - mu) / sigma, 1) == 17.1 and round((4301 - mu) / sigma, 1) == 3.4

# refinement: top-k picks k DISTINCT experts per token, so per-expert count ~ Binomial(T, k/E)
mu2, sigma2 = T * k / E, math.sqrt(T * (k / E) * (1 - k / E))
print(f"\nrefined model  X ~ Binomial(T={T:,}, k/E={k / E}): mu = {mu2:,.0f}, sigma = {sigma2:.1f}  "
      f"(each token includes a given expert with probability k/E)")
for cf in (1.25, 1.05):
    cap = math.ceil(cf * mu2)
    print(f"  cf {cf}: capacity {cap:,} = {(cap - mu2) / sigma2:.1f} sigma;  exact tail = {tail_prob(T, k / E, cap):.2e}")
trials = []
for _ in range(2000):
    scr = torch.rand(T, E)
    _, ti = scr.topk(k, dim=-1)
    trials.append(torch.bincount(ti.reshape(-1), minlength=E)[0].item())
tr = torch.tensor(trials, dtype=torch.float)
print(f"simulation (2000 random uniform top-{k} batches, expert 0): mean {tr.mean():.1f}, sd {tr.std():.1f}")
assert abs(tr.mean() - mu2) < 5 and abs(tr.std() - sigma2) < 3
print("PASS  simulated sd matches the refined model, which is SMALLER than the journal's 59.9")
print("      -> the journal's 17.1 sigma is conservative; the same capacity is further out under top-k-of-distinct")
print("      (both assume balanced, independent routing; a collapsing router breaks that, which is the point of the metric)")

# ----------------------------------------------------------------------
# Part 6 -- the dense-expert option, and why MFU is the wrong objective
# ----------------------------------------------------------------------
hr("Part 6 -- run every expert on every token")

routed_per_tok_layer = 6 * k * 3 * d * cfg.d_ff
routed_step = routed_per_tok_layer * T * L
dense_step = routed_step * (E / k)
extra = dense_step - routed_step
print(f"routed FFN FLOPs/token/layer (fwd+bwd) = 6 x k x 3 x d x d_ff = {routed_per_tok_layer:,}")
print(f"x T x L = {routed_step / 1e12:.2f} TFLOP/step;  dense (E/k = {E // k}x) = {dense_step / 1e12:.2f};  extra = {extra / 1e12:.2f}")
assert routed_per_tok_layer == 42_467_328 and round(routed_step / 1e12, 2) == 6.96
assert round(dense_step / 1e12, 2) == 27.83 and round(extra / 1e12, 2) == 20.87

useful_step = flops_tok * T
perm_ms = measured_share_journal * eager_ms
print(f"\nuseful FLOPs/step = {flops_tok:,} x {T:,} = {useful_step / 1e12:.3f} TFLOP")
print(f"eager useful-FLOP MFU = {useful_step / (eager_ms / 1e3) / PEAK_FLOPS * 100:.2f}%   (journal: 6.3%)")
assert round(useful_step / (eager_ms / 1e3) / PEAK_FLOPS * 100, 1) == 6.3
print(f"\n{'GEMM rate assumed':>22} {'added ms':>9} {'new step ms':>12} {'speed-up':>9} {'useful MFU':>11} {'hardware MFU':>13}")
for rate in (850e12, 425e12):
    added = extra / rate * 1e3
    new_step = eager_ms - perm_ms + added
    umfu = useful_step / (new_step / 1e3) / PEAK_FLOPS * 100
    hmfu = (useful_step + extra) / (new_step / 1e3) / PEAK_FLOPS * 100
    print(f"{rate / 1e12:>19.0f} TF/s {added:>9.1f} {new_step:>12.1f} {eager_ms / new_step:>8.2f}x {umfu:>10.1f}% {hmfu:>12.1f}%")
    if rate == 850e12:
        assert round(added, 1) == 24.6 and round(eager_ms / new_step, 2) == 1.46 and round(umfu, 1) == 9.2
    else:
        assert round(added, 1) == 49.1
print("note: the journal states 9.1% for the first row; recomputed here it is 9.2% -- a 0.1-point rounding gap, the")
print("      conclusion (useful MFU 6.3% -> ~9%, wall clock ~1.45x) is unchanged. The 850 / 425 TF/s rates are the")
print("      journal's ASSUMED GEMM rates (not derived here), so these two rows are conditional on them.")

print("\nAll assertions passed -- every derived number in 044-mfu-and-the-roofline.md is real.")
