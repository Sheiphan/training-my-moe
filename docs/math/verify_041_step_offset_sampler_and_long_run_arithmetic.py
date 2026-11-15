"""Reproduces every number in `041-step-offset-sampler-and-long-run-arithmetic.md`.

Run it:  uv run python docs/math/verify_041_step_offset_sampler_and_long_run_arithmetic.py

Journal 041 made a multi-day run survivable. The parts with arithmetic in them:

  * `fold_seed` (splitmix64): worked by hand on three steps, checked against
    the real function, plus a measured avalanche statistic against the naive
    `seed + step`,
  * `StepOffsetSampler`: the resume contract (offsets are a pure function of
    (seed, step)) asserted on the real class,
  * sampling WITH replacement: the expected-duplicate and expected-coverage
    formulas, Monte-Carlo-checked at small scale, then evaluated at the real
    899,396,383-window / 2.7B-token scale. This adds one quantity the journal
    does not state -- the fraction of *tokens* a run never sees,
  * why `DataLoader(shuffle=True)` costs 48 bytes per window (measured here at
    small scale in a subprocess, then extrapolated, exactly as the journal does),
  * `strided_eval_offsets`: 144x validation coverage for identical compute,
  * checkpoint arithmetic: 12 bytes/param, 82 x 4.16GB = 341GB, tiered
    retention, driven through the real `prune_checkpoints`.

No fabricated numbers: every figure the prose quotes is printed here. Where a
journal figure is a *measurement on hardware this script cannot reproduce*
(e.g. the 8-minute stall), the doc says so and this script does not claim it.
"""
from __future__ import annotations

import dataclasses
import itertools
import math
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from configs.presets import SCALE_300M  # noqa: E402
from qwenmoe.data.sampler import (  # noqa: E402
    _GOLDEN_GAMMA,
    _MASK64,
    _MIX_A,
    _MIX_B,
    StepOffsetSampler,
    fold_seed,
    strided_eval_offsets,
)
from qwenmoe.train.trainer import prune_checkpoints  # noqa: E402
from scripts.check_capacity import estimate_params  # noqa: E402


def hr(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


# ----------------------------------------------------------------------
# Part 1 -- splitmix64 by hand
# ----------------------------------------------------------------------
hr("Part 1 -- fold_seed: splitmix64, step by step")


def fold_by_hand(seed: int, step: int, verbose: bool = False) -> int:
    z = (seed + step * _GOLDEN_GAMMA) & _MASK64
    if verbose:
        print(f"  z0 = seed + step*gamma mod 2^64        = {z:#018x}")
    z = ((z ^ (z >> 30)) * _MIX_A) & _MASK64
    if verbose:
        print(f"  z1 = (z0 ^ z0>>30) * A mod 2^64        = {z:#018x}")
    z = ((z ^ (z >> 27)) * _MIX_B) & _MASK64
    if verbose:
        print(f"  z2 = (z1 ^ z1>>27) * B mod 2^64        = {z:#018x}")
    z = z ^ (z >> 31)
    if verbose:
        print(f"  z3 = z2 ^ z2>>31                       = {z:#018x}")
    return z & 0x7FFF_FFFF_FFFF_FFFF


for step in (0, 1, 2):
    print(f"seed=0, step={step}")
    mine = fold_by_hand(0, step, verbose=True)
    real = fold_seed(0, step)
    print(f"  masked to 63 bits                      = {mine:#018x}  ({mine})")
    assert mine == real
print("PASS  hand-written mixing == real fold_seed on steps 0,1,2")
assert fold_seed(0, 0) == 0
print("note: fold_seed(0, 0) == 0 -- the all-zero input is a fixed point of the finalizer")
print("      (harmless for torch.Generator.manual_seed; step 1 onward is already well mixed)")

# avalanche: how many of 63 bits change between adjacent steps?
def popcount(x: int) -> int:
    return bin(x).count("1")


T = 10_000
naive = [popcount((0 + t) ^ (0 + t + 1)) for t in range(T)]
mixed = [popcount(fold_seed(0, t) ^ fold_seed(0, t + 1)) for t in range(T)]
print(f"adjacent steps, mean bits that differ: naive seed+step = {sum(naive) / T:.2f}, "
      f"fold_seed = {sum(mixed) / T:.2f} of 63 (ideal 31.5)")
assert sum(naive) / T < 3 and 30.5 < sum(mixed) / T < 32.5
print("PASS  fold_seed flips ~half the bits between adjacent steps; seed+step flips ~1-2")

# ----------------------------------------------------------------------
# Part 2 -- resume is a pure function
# ----------------------------------------------------------------------
hr("Part 2 -- the resume contract")

sampler = StepOffsetSampler(n_windows=1_000_003, batch_size=4, grad_accum_steps=3, seed=7)
ops = sampler.offsets_per_step
print(f"batch_size=4, grad_accum_steps=3 -> offsets_per_step = {ops} (the optimizer-step window, not a micro-batch)")
assert ops == 12
full = list(itertools.islice(iter(sampler), 10 * ops))
resumed = list(itertools.islice(iter(StepOffsetSampler(1_000_003, 4, 3, seed=7, start_step=6)), 4 * ops))
assert resumed == full[6 * ops : 10 * ops]
assert resumed[:ops] == sampler.offsets_for_step(6)
print("PASS  a sampler started at step 6 yields exactly steps 6..9 of the uninterrupted stream")
assert StepOffsetSampler(1_000_003, 4, 3, seed=8).offsets_for_step(6) != sampler.offsets_for_step(6)
print("PASS  a different seed gives different offsets for the same step")
assert all(0 <= o < 1_000_003 for o in full)
try:
    len(sampler)
    raise AssertionError("len() should raise")
except TypeError:
    print("PASS  the sampler is infinite: len() raises instead of reporting something wrong")

# ----------------------------------------------------------------------
# Part 3 -- sampling with replacement: duplicates and coverage
# ----------------------------------------------------------------------
hr("Part 3 -- with-replacement arithmetic")

W = 899_396_383          # journal 041's real train-split window count
L = 512                  # seq_len
BUDGET = 2_700_000_000   # tokens
m = BUDGET / L
print(f"windows W = {W:,};  draws m = budget/seq_len = {BUDGET:,}/{L} = {m:,.0f}")
frac_windows = m / W
distinct = W * (-math.expm1(-m / W))
dups = m - distinct
print(f"fraction of windows drawn   m/W            = {100 * frac_windows:.3f}%   (journal: 0.59%)")
print(f"expected distinct windows   W(1-e^(-m/W))  = {distinct:,.0f}")
print(f"expected duplicate draws    m - distinct   = {dups:,.0f}  ({100 * dups / m:.3f}% of draws; journal: ~0.3%)")
print(f"  closed form m^2/(2W)                     = {m * m / (2 * W):,.0f}")
assert round(100 * frac_windows, 2) == 0.59 and round(100 * dups / m, 1) == 0.3

# Monte Carlo of the duplicate formula at small scale
gen = torch.Generator().manual_seed(0)
Wsmall, msmall = 1_000_000, 20_000
dup_emp = []
for _ in range(40):
    draws = torch.randint(0, Wsmall, (msmall,), generator=gen)
    dup_emp.append(msmall - draws.unique().numel())
pred = msmall - Wsmall * (-math.expm1(-msmall / Wsmall))
print(f"Monte Carlo (W={Wsmall:,}, m={msmall:,}, 40 trials): mean duplicate draws = "
      f"{sum(dup_emp) / 40:.1f}, formula = {pred:.1f}")
assert abs(sum(dup_emp) / 40 - pred) < 0.1 * pred
print("PASS  expected-duplicates formula matches simulation")

# NEW: overlap. Two windows of length L at offsets i, j share >=1 token iff |i-j| < L.
overlap_neighbours = m * (2 * L - 1) / W
print(f"\nexpected OTHER drawn windows that overlap a given drawn window by >= 1 token: "
      f"m(2L-1)/W = {overlap_neighbours:.2f}")

# NEW: token coverage. A token position is inside L windows (those starting in [pos-L+1, pos]).
lam = m * L / W
unseen = math.exp(-lam)
print(f"lambda = expected times a given token is drawn = m*L/W = {lam:.3f}  (= budget / unique tokens)")
print(f"fraction of train tokens NEVER seen  e^(-lambda)        = {100 * unseen:.2f}%  (Poisson approximation)")
assert abs(lam - BUDGET / W) < 0.01

# Monte Carlo of the coverage formula at small scale (L=16, W=100k, lambda=3)
Ls, Ws = 16, 100_000
ms = int(3 * Ws / Ls)
n_tok = Ws + Ls
fracs = []
for trial in range(20):
    draws = torch.randint(0, Ws, (ms,), generator=gen)
    cover = torch.zeros(n_tok + 1, dtype=torch.long)
    cover.index_add_(0, draws, torch.ones(ms, dtype=torch.long))        # +1 at each window start
    cover.index_add_(0, draws + Ls, -torch.ones(ms, dtype=torch.long))  # -1 just past its end
    covered = cover.cumsum(0)[:n_tok] > 0
    fracs.append(1 - covered.float().mean().item())
lam_s = ms * Ls / Ws
print(f"Monte Carlo (L={Ls}, W={Ws:,}, lambda={lam_s:.2f}, 20 trials): unseen = "
      f"{100 * sum(fracs) / 20:.2f}%, e^-lambda = {100 * math.exp(-lam_s):.2f}%")
assert abs(sum(fracs) / 20 - math.exp(-lam_s)) < 0.005
print("PASS  Poisson coverage formula matches simulation")

# ----------------------------------------------------------------------
# Part 4 -- what DataLoader(shuffle=True) costs
# ----------------------------------------------------------------------
hr("Part 4 -- bytes per window of RandomSampler's no-replacement path")

probe = (
    "import resource, sys, torch;"
    "n=int(sys.argv[1]); b=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;"
    "p=torch.randperm(n); l=p.tolist();"
    "a=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;"
    "print(a-b)"
)
scale = 1 if sys.platform == "darwin" else 1024  # ru_maxrss is bytes on macOS, KiB on Linux
n = 10_000_000
out = subprocess.run([sys.executable, "-c", probe, str(n)], capture_output=True, text=True, check=True)
peak_delta = int(out.stdout.strip()) * scale
per = peak_delta / n
print(f"n={n:,}: peak RSS grew by {peak_delta / 1e9:.2f}GB -> {per:.1f} bytes/element (journal: 48.0 at this n)")
assert 44 <= per <= 52, per
print(f"  = 8 (int64 tensor) + 8 (list pointer) + {sys.getsizeof(2**29):d}->32 (boxed int, size-class rounded) = 48")
assert sys.getsizeof(899_396_383) == 28
full_cost = W * 48.2
print(f"extrapolated to W={W:,}: {W * 48.2 / 1e9:.1f}GB   (journal: ~43GB; extrapolated, not measured at full size)")
assert round(full_cost / 1e9) == 43
print(f"sampler instead: offsets_per_step={ops} int64 -> {ops * 8} bytes per step; O(batch), not O(W)")

# ----------------------------------------------------------------------
# Part 5 -- strided validation offsets
# ----------------------------------------------------------------------
hr("Part 5 -- validation coverage")

SEQ, BATCH, N_BATCHES = 512, 4, 50
n_eval = BATCH * N_BATCHES
val_windows = 99_932_959 - SEQ  # tail split of 999,329,599 tokens at val_fraction 0.1, minus seq_len
old_offsets = list(range(n_eval))  # shuffle=False, first 50 batches
old_positions = set()
for off in old_offsets:
    old_positions.update(range(off, off + SEQ + 1))
print(f"old: windows 0..{n_eval - 1} -> token positions [0, {max(old_positions)}] = {len(old_positions)} distinct tokens")
print(f"     predictions computed = {n_eval}*{SEQ} = {n_eval * SEQ:,}; distinct targets = {len(old_positions) - 1}")
new_offsets = strided_eval_offsets(val_windows, n_eval)
new_positions = set()
for off in new_offsets:
    new_positions.update(range(off, off + SEQ + 1))
spacing = new_offsets[1] - new_offsets[0]
print(f"new: spacing = {spacing:,} windows; distinct tokens = {len(new_positions):,}; "
      f"distinct targets = {n_eval * SEQ:,}")
assert max(old_positions) == 711 and len(old_positions) == 712
assert spacing > SEQ and len(new_positions) == n_eval * (SEQ + 1)
wasted = 1 - (len(old_positions) - 1) / (n_eval * SEQ)
ratio = n_eval * SEQ / (len(old_positions) - 1)
print(f"old scheme re-predicted the same targets {100 * wasted:.1f}% of the time; new covers {ratio:.0f}x more distinct targets "
      f"for identical compute")
assert round(100 * wasted, 1) == 99.3 and round(ratio) == 144
print("PASS  99.3% waste and 144x coverage reproduced (journal 041)")
assert strided_eval_offsets(10, 20) == list(range(10))
print("PASS  asking for more eval windows than exist clamps to all of them")

# ----------------------------------------------------------------------
# Part 6 -- checkpoint arithmetic
# ----------------------------------------------------------------------
hr("Part 6 -- checkpoint bytes and tiered retention")

cfg = dataclasses.replace(SCALE_300M, vocab_size=16385)
total, _active = estimate_params(cfg)
full_ckpt = 12 * total
weights_only = 4 * total
print(f"SCALE_300M at vocab 16,385: {total:,} params")
print(f"  full checkpoint  12 B/param (4 weight + 8 AdamW moments) = {full_ckpt / 1e9:.2f}GB   (journal: 4.16GB)")
print(f"  weights-only      4 B/param                              = {weights_only / 1e9:.2f}GB   (journal: 1.39GB)")
assert round(full_ckpt / 1e9, 2) == 4.16 and round(weights_only / 1e9, 2) == 1.39

n_ckpt = 164_000 // 2000
print(f"164,000 steps / interval 2,000 = {n_ckpt} checkpoints -> {n_ckpt * full_ckpt / 1e9:.0f}GB   (journal: 341GB)")
assert n_ckpt == 82 and round(n_ckpt * full_ckpt / 1e9) == 341

keep_last, keep_every = 2, 20_000
n_milestones = 164_000 // keep_every
tiered = keep_last * full_ckpt + n_milestones * weights_only
print(f"tiered: keep_last={keep_last} full + {n_milestones} milestones (every {keep_every:,}) weights-only")
print(f"        = {keep_last}*{full_ckpt / 1e9:.2f} + {n_milestones}*{weights_only / 1e9:.2f} = {tiered / 1e9:.1f}GB   (journal: ~19GB)")
assert round(tiered / 1e9) == 19
print(f"        reduction vs keeping everything: {n_ckpt * full_ckpt / tiered:.0f}x")

# the real prune_checkpoints, on empty files named like real ones
with tempfile.TemporaryDirectory() as td:
    d = Path(td)
    for step in range(2000, 164_001, 2000):
        (d / f"step_{step}.pt").write_bytes(b"")
    latest = d / "step_164000.pt"
    removed = prune_checkpoints(d, keep_last=keep_last, keep_every=0, protect={latest})
    left = sorted(int(p.stem.split("_")[1]) for p in d.glob("step_*.pt"))
    print(f"prune_checkpoints(keep_last=2): removed {len(removed)}, kept steps {left}")
    assert len(removed) == 80 and left == [162_000, 164_000]
print("PASS  the real pruner keeps exactly the two most recent rolling checkpoints")

print("\nAll assertions passed -- every number in 041-step-offset-sampler-and-long-run-arithmetic.md is real.")
