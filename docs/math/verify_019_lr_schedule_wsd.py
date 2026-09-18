"""Reproduces every hand-computed number in `019-lr-schedule-wsd.md`.

Run it:  uv run python docs/math/verify_019_lr_schedule_wsd.py

ONE running example, carried end to end: the smallest schedule
configuration that still exercises every branch of
`qwenmoe.train.trainer.lr_multiplier` and `_set_lr` --

    warmup_steps W = 4, total_steps T = 20, min_lr_ratio r = 0.25,
    decay_fraction f = 0.28,  peak lr = 0.01,  muon_lr = 0.02.

W = 4  ->  a real linear ramp whose last step lands on exactly 1.0.
T - W = 16  ->  cosine's `progress` hits 0, 1/4, 1/2, 3/4, 1 at steps
4, 8, 12, 16, 20, i.e. angles 0, pi/4, pi/2, 3pi/4, pi -- all exact
textbook cosines.
f = 0.28  ->  round(0.28 * 20) = round(5.6) = 6, so the WSD decay window
is 6 steps ([14, 20)) and the stable plateau is steps [4, 14).
r = 0.25  ->  1 - r = 0.75, so WSD's linear decay drops a clean 0.00125
of raw lr per step.
muon_lr = 0.02 != lr = 0.01  ->  `_set_lr` rides ONE schedule shape at TWO
peak magnitudes.

It drives the REAL `lr_multiplier` / `_set_lr` / `Trainer.fit`, prints
every intermediate, and asserts:

  * warmup: multiplier == (step + 1) / W for steps 0..3, hitting 1.0 at
    step 3,
  * the `total_steps <= warmup_steps` guard: pins the multiplier at 1.0,
  * cosine: the full steps-4..20 curve, with steps 4/8/12/16/20 checked
    against the exact closed form r + (1-r) * 0.5 * (1 + cos(pi p)),
  * cosine `progress` clamp: step 24 (past T) == step 20,
  * WSD: decay_steps = 6, decay_start = 14; flat 1.0 on [4, 14); linear
    1.0 -> 0.25 on [14, 20]; == r exactly at step 20,
  * WSD `progress'` clamp: step 23 (past T) == step 20,
  * WSD `max(1, .)` guard (tiny f) and the `decay_start = max(W, .)` clamp
    (huge f),
  * an unknown schedule name raises ValueError,
  * `_set_lr` on a real Muon optimizer: every AdamW group's lr is
    0.01 * mult, the Muon group's lr is 0.02 * mult, and the returned
    (logged) value is 0.01 * mult,
  * a real `Trainer.fit` (total_steps == 20) logs, for BOTH schedules,
    lr[logged step k] == 0.01 * lr_multiplier(k - 1, ...) EXACTLY,
  * continuity at the warmup->decay seam for both schedules,
  * min_lr_ratio = 0 recovers the pre-task decay-to-exactly-zero curve.

No fabricated numbers -- this file is the source of truth the prose quotes.
"""
from __future__ import annotations

import json
import logging
import math
import pathlib
import tempfile

import torch

from qwenmoe.config import ModelConfig, TrainConfig
from qwenmoe.data.dataset import TokenDataset
from qwenmoe.model.transformer import QwenMoEModel
from qwenmoe.train.optim import build_optimizer
from qwenmoe.train.trainer import Trainer, _set_lr, lr_multiplier

torch.set_printoptions(precision=8, sci_mode=False)


def approx(a: float, b: float, tol: float = 1e-12) -> bool:
    return abs(a - b) <= tol


# ======================================================================
# Part 1 -- the running example
# ======================================================================
print("================ Part 1 -- the running example ================")
W = 4          # warmup_steps
T = 20         # total_steps
R = 0.25       # min_lr_ratio
F = 0.28       # decay_fraction
PEAK = 0.01    # config.lr  (AdamW groups' base_lr)
MUON = 0.02    # config.muon_lr  (Muon group's base_lr)
print(f"  W (warmup_steps)   = {W}")
print(f"  T (total_steps)    = {T}")
print(f"  r (min_lr_ratio)   = {R}")
print(f"  f (decay_fraction) = {F}")
print(f"  peak lr            = {PEAK}")
print(f"  muon_lr            = {MUON}")


# ======================================================================
# Part 2 -- the warmup ramp:  mult(t) = (t + 1) / W   for t < W
# ======================================================================
print("\n================ Part 2 -- warmup ramp ================")
for t in range(W):
    got = lr_multiplier(t, W, T, schedule="cosine", min_lr_ratio=R)
    want = (t + 1) / W
    print(f"  step {t}: (t+1)/W = {t + 1}/{W} = {want:.6f}   lr = {PEAK * want:.6f}")
    assert approx(got, want), (t, got, want)
# same ramp regardless of schedule / min_lr_ratio / decay_fraction
for t in range(W):
    assert lr_multiplier(t, W, T, schedule="wsd", min_lr_ratio=R, decay_fraction=F) == (t + 1) / W
assert lr_multiplier(W - 1, W, T, schedule="cosine", min_lr_ratio=R) == 1.0
print("  last warmup step (t = W-1 = 3) lands on exactly 1.0  [OK]")


# ======================================================================
# Part 3 -- the  total_steps <= warmup_steps  guard
# ======================================================================
print("\n================ Part 3 -- total_steps <= warmup_steps guard ================")
# step past warmup, but the run is shorter than warmup itself -> no decay span
g = lr_multiplier(12, warmup_steps=10, total_steps=8, schedule="cosine", min_lr_ratio=R)
print(f"  lr_multiplier(step=12, warmup=10, total=8) = {g}   (held at peak, no decay span)")
assert g == 1.0
gw = lr_multiplier(12, warmup_steps=10, total_steps=8, schedule="wsd", min_lr_ratio=R, decay_fraction=F)
assert gw == 1.0


# ======================================================================
# Part 4 -- cosine decay:  mult(t) = r + (1-r) * 0.5 * (1 + cos(pi p))
# ======================================================================
print("\n================ Part 4 -- cosine decay ================")
print("  step |  p = (t-W)/(T-W) | angle pi*p |  cos(pi p)  |  mult      |  lr")
cos_mults = {}
for t in range(W, T + 1):
    p = min((t - W) / (T - W), 1.0)
    ang = math.pi * p
    c = 0.5 * (1.0 + math.cos(ang))
    mult = R + (1.0 - R) * c
    cos_mults[t] = mult
    got = lr_multiplier(t, W, T, schedule="cosine", min_lr_ratio=R)
    assert approx(got, mult, 1e-12), (t, got, mult)
    print(f"   {t:>3} |     {p:.6f}     |  {ang:.6f} | {math.cos(ang):+.8f} | {mult:.8f} | {PEAK * mult:.8f}")

# the exact anchors: p in {0, 1/4, 1/2, 3/4, 1}  ->  cos in {1, r2/2, 0, -r2/2, -1}
r2 = math.sqrt(2) / 2
anchor = {
    W:      R + (1 - R) * 0.5 * (1 + 1.0),
    W + 4:  R + (1 - R) * 0.5 * (1 + r2),
    W + 8:  R + (1 - R) * 0.5 * (1 + 0.0),
    W + 12: R + (1 - R) * 0.5 * (1 - r2),
    W + 16: R + (1 - R) * 0.5 * (1 - 1.0),
}
print("\n  exact anchors (textbook cosines):")
for t, m in anchor.items():
    assert approx(cos_mults[t], m, 1e-12), (t, cos_mults[t], m)
    print(f"    step {t:>2}: mult = {m:.8f}")
assert approx(cos_mults[W], 1.0)
assert approx(cos_mults[W + 8], 0.625)          # 0.25 + 0.75 * 0.5
assert approx(cos_mults[T], 0.25)               # == r exactly at t = T
# progress clamp: step past T reuses progress = 1.0
assert lr_multiplier(24, W, T, schedule="cosine", min_lr_ratio=R) == cos_mults[T]
print("\n  progress clamp: lr_multiplier(24, ...) == lr_multiplier(20, ...)  [OK]")


# ======================================================================
# Part 5 -- WSD: stable plateau + linear decay to the floor
# ======================================================================
print("\n================ Part 5 -- WSD schedule ================")
decay_steps = max(1, round(F * T))
decay_start = max(W, T - decay_steps)
print(f"  decay_steps = max(1, round(f*T)) = max(1, round({F * T})) = {decay_steps}")
print(f"  decay_start = max(W, T - decay_steps) = max({W}, {T - decay_steps}) = {decay_start}")
assert decay_steps == 6 and decay_start == 14

print("  step |   phase   | progress' | mult     | lr")
wsd_mults = {}
for t in range(W, T + 1):
    if t < decay_start:
        mult = 1.0
        phase = "stable"
        pp = 0.0
    else:
        pp = min((t - decay_start) / (T - decay_start), 1.0)
        mult = 1.0 - (1.0 - R) * pp
        phase = "decay"
    wsd_mults[t] = mult
    got = lr_multiplier(t, W, T, schedule="wsd", min_lr_ratio=R, decay_fraction=F)
    assert approx(got, mult, 1e-12), (t, got, mult)
    print(f"   {t:>3} | {phase:>9} | {pp:.6f}  | {mult:.6f} | {PEAK * mult:.6f}")

assert all(wsd_mults[t] == 1.0 for t in range(W, decay_start))     # flat plateau
assert approx(wsd_mults[14], 1.0)
assert approx(wsd_mults[17], 0.625)                                # halfway: 1 - 0.75*0.5
assert approx(wsd_mults[T], R)                                     # == r exactly at t = T
# each decay step drops raw lr by a constant 0.75 * (1/6) * 0.01
drops = [PEAK * (wsd_mults[t] - wsd_mults[t + 1]) for t in range(decay_start, T)]
print(f"\n  per-step lr drop in the decay window: {[round(d, 8) for d in drops]}")
assert all(approx(d, PEAK * (1 - R) / decay_steps, 1e-12) for d in drops)
# progress' clamp: step past T reuses progress' = 1.0
assert lr_multiplier(23, W, T, schedule="wsd", min_lr_ratio=R, decay_fraction=F) == wsd_mults[T]
print("  progress' clamp: lr_multiplier(23, ...) == lr_multiplier(20, ...)  [OK]")


# ======================================================================
# Part 6 -- WSD degenerate configs (both guards)
# ======================================================================
print("\n================ Part 6 -- WSD degenerate configs ================")
# (a) tiny f: round(0.01 * 20) = 0 -> max(1, 0) = 1 keeps T - decay_start > 0
ds_a = max(1, round(0.01 * T))
start_a = max(W, T - ds_a)
print(f"  tiny f=0.01 : decay_steps = max(1, round(0.2)) = {ds_a}, decay_start = {start_a}")
assert ds_a == 1 and start_a == T - 1
# no ZeroDivisionError, and the only in-range decay step (t = 19) is still at peak
assert lr_multiplier(19, W, T, schedule="wsd", min_lr_ratio=R, decay_fraction=0.01) == 1.0
assert lr_multiplier(20, W, T, schedule="wsd", min_lr_ratio=R, decay_fraction=0.01) == R

# (b) huge f: round(0.95 * 20) = 19 -> decay_start = max(4, 1) = 4 (plateau length 0)
ds_b = max(1, round(0.95 * T))
start_b = max(W, T - ds_b)
print(f"  huge f=0.95 : decay_steps = {ds_b}, decay_start = max({W}, {T - ds_b}) = {start_b}  (no plateau)")
assert ds_b == 19 and start_b == W
# decay starts the instant warmup ends; never overlaps warmup
assert lr_multiplier(W, W, T, schedule="wsd", min_lr_ratio=R, decay_fraction=0.95) == 1.0
assert lr_multiplier(3, W, T, schedule="wsd", min_lr_ratio=R, decay_fraction=0.95) == 1.0  # still warmup


# ======================================================================
# Part 7 -- unknown schedule name
# ======================================================================
print("\n================ Part 7 -- unknown schedule ================")
try:
    lr_multiplier(8, W, T, schedule="linear", min_lr_ratio=R)  # type: ignore[arg-type]
    raise AssertionError("expected ValueError")
except ValueError as exc:
    print(f"  lr_multiplier(..., schedule='linear') raised: {exc}")
# ...but only once past warmup with a real decay span
assert lr_multiplier(0, W, T, schedule="linear", min_lr_ratio=R) == 0.25  # type: ignore[arg-type]
print("  (during warmup the schedule name is never inspected -> returns 0.25)")


# ======================================================================
# Part 8 -- _set_lr on a real Muon optimizer: one shape, two magnitudes
# ======================================================================
print("\n================ Part 8 -- _set_lr (Muon: two peak magnitudes) ================")
cfg = TrainConfig(
    batch_size=1, seq_len=3, total_tokens=60,
    lr=PEAK, warmup_steps=W, min_lr_ratio=R, decay_fraction=F,
    lr_schedule="cosine", optimizer="muon", muon_lr=MUON,
    device="cpu", seed=0,
    checkpoint_interval_steps=20, eval_interval_steps=20, log_interval_steps=1,
    max_eval_batches=2,
)
mcfg = ModelConfig(
    vocab_size=8, d_model=8, n_layers=1, n_heads=2, n_kv_heads=1, d_ff=4,
    n_routed_experts=2, n_shared_experts=0, top_k=1, max_seq_len=16,
    tie_word_embeddings=True,
)
torch.manual_seed(0)
model = QwenMoEModel(mcfg)
opt = build_optimizer(model, cfg)
base_lrs = [g["base_lr"] for g in opt.param_groups]
print(f"  optimizer param-group base_lrs : {base_lrs}")
print("  (Muon group 0.02; AdamW no-decay group 0.01; AdamW decay group empty on this tiny model)")

step = 8
returned = _set_lr(opt, cfg.lr, step, T, cfg)
mult = lr_multiplier(step, W, T, schedule="cosine", min_lr_ratio=R)
print(f"  step {step}: multiplier = {mult:.8f}")
for g in opt.param_groups:
    print(f"    base_lr {g['base_lr']:.3f}  ->  lr = {g['lr']:.8f}   (= base_lr * mult)")
    assert approx(g["lr"], g["base_lr"] * mult, 1e-12)
print(f"  _set_lr return value (logged) : {returned:.8f}  (= config.lr * mult = {PEAK} * {mult:.8f})")
assert approx(returned, PEAK * mult, 1e-12)


# ======================================================================
# Part 9 -- a real Trainer.fit: the logged lr == the hand curve, per step
# ======================================================================
print("\n================ Part 9 -- real Trainer.fit ================")
STREAM = [1, 2, 3, 1, 2, 3, 1, 2]
train_ds = TokenDataset(STREAM, seq_len=3)
val_ds = TokenDataset(STREAM, seq_len=3)


def run_fit(schedule: str) -> list[tuple[int, float]]:
    fit_cfg = TrainConfig(
        batch_size=1, seq_len=3, total_tokens=60,
        lr=PEAK, warmup_steps=W, min_lr_ratio=R, decay_fraction=F,
        lr_schedule=schedule, optimizer="adamw",
        device="cpu", seed=1337,
        checkpoint_interval_steps=20, eval_interval_steps=20, log_interval_steps=1,
        max_eval_batches=2,
    )
    out = pathlib.Path(tempfile.mkdtemp())
    logging.disable(logging.CRITICAL)
    res = Trainer(out).fit(QwenMoEModel(mcfg), train_ds, val_ds, fit_cfg)
    logging.disable(logging.NOTSET)
    assert res.final_step == T, res.final_step
    lines = (out / "logs" / "train.jsonl").read_text().splitlines()
    steps = [json.loads(x) for x in lines if '"type": "step"' in x]
    return [(r["step"], r["lr"]) for r in steps]


for schedule in ("cosine", "wsd"):
    logged = run_fit(schedule)
    assert len(logged) == T, (schedule, len(logged))
    print(f"\n  {schedule}: logged (step, lr) vs 0.01 * lr_multiplier(step - 1, ...)")
    max_diff = 0.0
    for k, lr in logged:
        want = PEAK * lr_multiplier(k - 1, W, T, schedule=schedule, min_lr_ratio=R, decay_fraction=F)
        max_diff = max(max_diff, abs(lr - want))
        flag = "" if k in (1, 4, 5, 9, 13, 14, 17, 20) else None
        if flag is not None:
            print(f"    logged step {k:>2} (0-idx {k - 1:>2}):  lr = {lr:.8f}   hand = {want:.8f}")
        assert approx(lr, want, 1e-12), (schedule, k, lr, want)
    print(f"    max |logged - hand| over all {T} steps = {max_diff:.2e}")


# ======================================================================
# Part 10 -- continuity at the warmup -> decay seam
# ======================================================================
print("\n================ Part 10 -- seam continuity ================")
for schedule in ("cosine", "wsd"):
    left = lr_multiplier(W - 1, W, T, schedule=schedule, min_lr_ratio=R, decay_fraction=F)
    right = lr_multiplier(W, W, T, schedule=schedule, min_lr_ratio=R, decay_fraction=F)
    print(f"  {schedule}: mult(W-1) = {left}   mult(W) = {right}")
    assert left == right == 1.0


# ======================================================================
# Part 11 -- min_lr_ratio = 0 recovers the pre-task decay-to-zero curve
# ======================================================================
print("\n================ Part 11 -- min_lr_ratio = 0 reduces to the original ================")
# cosine with r = 0: floor is exactly 0 at t = T
c0 = lr_multiplier(T, W, T, schedule="cosine", min_lr_ratio=0.0)
print(f"  cosine, r=0, step T : {c0}")
assert c0 == 0.0
# wsd with r = 0: linear decay all the way to 0
w0 = lr_multiplier(T, W, T, schedule="wsd", min_lr_ratio=0.0, decay_fraction=F)
print(f"  wsd,    r=0, step T : {w0}")
assert w0 == 0.0
# and both agree with the r>0 form when the cosine/linear factor is 0
assert lr_multiplier(T, W, T, schedule="cosine", min_lr_ratio=R) == R
assert lr_multiplier(T, W, T, schedule="wsd", min_lr_ratio=R, decay_fraction=F) == R


print("\nAll assertions passed -- every number in 019-lr-schedule-wsd.md is real.")
