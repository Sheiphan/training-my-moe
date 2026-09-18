"""Reproduces every hand-computed number in `034-modernized-presets.md`.

Run it:  uv run python docs/math/verify_034_modernized_presets.py

Journal entry 034 is a *decision* entry: `BASELINE_V2` bundles three
already-shipped, already-derived axes (`qk_norm` -- doc 022; `router_scoring
= "sigmoid"` -- doc 025; `init_std = sqrt(2/(5*d_model))` -- doc 023 derives
the residual-scaling machinery it plugs into, but not this exact formula's
own origin) and `FINEGRAINED` adds one axis with **no** existing derivation:
DeepSeekMoE's fine-grained expert segmentation. This script, like its
companion doc, does not re-derive the three recapped axes -- it earns its
keep on:

  1. A **toy** MoE layer at two granularities (`E=2,d_ff=8,k=1` and its
     segmentation-factor-4 sibling `E=8,d_ff=2,k=4`, both `n_shared=1`,
     `d_model=4`) -- built as a real `QwenMoEModel`, so every weight is a
     real `_trunc_normal_`-initialized tensor, not invented. Confirms the
     segmentation-invariance algebra (routed total, routed active held;
     shared FFN and router *not* held) on real `nn.Parameter`s.
  2. Combinatorics: `math.comb` cross-checked against a from-scratch
     `n_choose_k` (factorials only), on both the toy granularities and the
     real `BASELINE` vs `FINEGRAINED` expert/top_k counts.
  3. The Nguyen & Salazar `sqrt(2/(5*d_model))` formula, derived from
     Glorot & Bengio's fan-in/fan-out compromise plus the standard
     Transformer FFN's `4*d_model` expansion, and checked bit-for-bit
     against `configs/presets.py::_V2_INIT_STD`.
  4. The real `BASELINE` / `BASELINE_V2` / `FINEGRAINED` presets run
     through the real `count_params` (doc 009), reproducing journal 034's
     own quoted parameter table exactly.
  5. A 12-seed measurement of the untrained-logit-std *ratio* between the
     three presets, checked against the ratio doc 023's own logit-std
     formula predicts from the `init_std` change alone.

No fused/alternate implementation exists for any of this (a `dataclasses
.replace` call and a parameter count are not the kind of thing with a
"fast path") -- there is nothing to reconcile two ways against, exactly
like docs 009 and 033.
"""
from __future__ import annotations

import math
import statistics

import torch

from configs.presets import BASELINE, BASELINE_V2, FINEGRAINED
import configs.presets as presets_module
from qwenmoe.config import ModelConfig
from qwenmoe.model.moe import MoELayer
from qwenmoe.model.transformer import QwenMoEModel
from qwenmoe.train.scaling import count_params

torch.set_printoptions(precision=4, sci_mode=False, linewidth=100)


def toy_cfg(**overrides) -> ModelConfig:
    """The running example's BASELINE-analog shape, d_model=4 -- small
    enough that every weight matrix in Part 1 fits on the page. Overridden
    below to produce the FINEGRAINED-analog sibling at segmentation
    factor s=4, exactly mirroring the real presets' own s=4
    (8->32 experts, d_ff 512->128, top_k 2->8)."""
    base = dict(
        vocab_size=6,
        d_model=4,
        n_layers=1,
        n_heads=2,
        n_kv_heads=1,
        d_ff=8,
        n_routed_experts=2,
        n_shared_experts=1,
        top_k=1,
        max_seq_len=8,
        activation="swiglu",
        router_scoring="softmax",
        qk_norm=False,
        init_std=0.02,
        tie_word_embeddings=True,
    )
    base.update(overrides)
    return ModelConfig(**base)


def n_choose_k(n: int, k: int) -> int:
    """From scratch: n!/(k!*(n-k)!), no library call."""
    return math.factorial(n) // (math.factorial(k) * math.factorial(n - k))


# ============================================================================
# Part 1 -- the toy running example: BASELINE-analog vs FINEGRAINED-analog,
# both real QwenMoEModel instances (real _trunc_normal_ init, seed=0).
# ============================================================================
print("=" * 78)
print("PART 1 -- toy MoE layer, two granularities, segmentation factor s=4")
print("=" * 78)

torch.manual_seed(0)
model_base = QwenMoEModel(toy_cfg())
moe_base = model_base.blocks[0].moe

torch.manual_seed(0)
model_fg = QwenMoEModel(
    toy_cfg(n_routed_experts=8, d_ff=2, top_k=4, n_shared_experts=1)
)
moe_fg = model_fg.blocks[0].moe

e0 = moe_base.routed_experts[0]
print("\nBASELINE-analog (E=2, d_ff=8, k=1, n_shared=1), d_model=4:")
print(f"  routed_experts[0].w_gate {tuple(e0.w_gate.weight.shape)} = "
      f"{e0.w_gate.weight.numel()} numbers:")
print(e0.w_gate.weight)
print(f"  routed_experts[0].w_up   {tuple(e0.w_up.weight.shape)} = "
      f"{e0.w_up.weight.numel()} numbers:")
print(e0.w_up.weight)
print(f"  routed_experts[0].w_down {tuple(e0.w_down.weight.shape)} = "
      f"{e0.w_down.weight.numel()} numbers:")
print(e0.w_down.weight)
per_expert_base = sum(p.numel() for p in e0.parameters())
print(f"  per-expert param count = 32+32+32 = {per_expert_base}"
      f"  (formula: 3*d_model*d_ff = 3*4*8 = {3*4*8})")
assert per_expert_base == 96 == 3 * 4 * 8

bound = 2 * 0.02
abs_max = e0.w_gate.weight.abs().max().item()
print(f"  |every entry| <= 2*init_std = {bound}: max observed = {abs_max:.4f} -- OK")
assert abs_max <= bound

print("\n  router.gate:", tuple(moe_base.router.gate.weight.shape),
      moe_base.router.gate.weight.numel(), "numbers")
print(moe_base.router.gate.weight)
assert moe_base.router.gate.weight.numel() == 4 * 2

e0_fg = moe_fg.routed_experts[0]
print("\nFINEGRAINED-analog (E=8, d_ff=2, k=4, n_shared=1), d_model=4:")
for i, e in enumerate(moe_fg.routed_experts):
    assert sum(p.numel() for p in e.parameters()) == 24
print(f"  routed_experts[0].w_gate {tuple(e0_fg.w_gate.weight.shape)}:")
print(e0_fg.w_gate.weight)
print(f"  routed_experts[0].w_up   {tuple(e0_fg.w_up.weight.shape)}:")
print(e0_fg.w_up.weight)
print(f"  routed_experts[0].w_down {tuple(e0_fg.w_down.weight.shape)}:")
print(e0_fg.w_down.weight)
per_expert_fg = sum(p.numel() for p in e0_fg.parameters())
print(f"  per-expert param count = {per_expert_fg}"
      f"  (formula: 3*d_model*d_ff = 3*4*2 = {3*4*2})")
assert per_expert_fg == 24 == 3 * 4 * 2

print("\n  router.gate:", tuple(moe_fg.router.gate.weight.shape),
      moe_fg.router.gate.weight.numel(), "numbers")
print(moe_fg.router.gate.weight)
assert moe_fg.router.gate.weight.numel() == 4 * 8

# ---- the segmentation-invariance algebra, on these real numbers ----
routed_total_base = 2 * per_expert_base
routed_total_fg = 8 * per_expert_fg
routed_active_base = 1 * per_expert_base
routed_active_fg = 4 * per_expert_fg
shared_base = 1 * per_expert_base
shared_fg = 1 * per_expert_fg
router_base = moe_base.router.gate.weight.numel()
router_fg = moe_fg.router.gate.weight.numel()

print("\nSegmentation-invariance check (s=4):")
print(f"  routed total : {2}x{per_expert_base}={routed_total_base}   vs  "
      f"{8}x{per_expert_fg}={routed_total_fg}   (held: {routed_total_base == routed_total_fg})")
print(f"  routed active: {1}x{per_expert_base}={routed_active_base}   vs  "
      f"{4}x{per_expert_fg}={routed_active_fg}   (held: {routed_active_base == routed_active_fg})")
print(f"  shared FFN   : {shared_base}   vs   {shared_fg}   "
      f"(ratio {shared_base/shared_fg:.1f}x -- NOT held, deliberate)")
print(f"  router       : {router_base}   vs   {router_fg}   "
      f"(ratio {router_fg/router_base:.1f}x -- grows, unavoidable)")
assert routed_total_base == routed_total_fg == 192
assert routed_active_base == routed_active_fg == 96
assert shared_base / shared_fg == 4.0
assert router_fg / router_base == 4.0

breakdown_base = count_params(model_base)
breakdown_fg = count_params(model_fg)
print(f"\n  full toy ParamBreakdown (BASELINE-analog): {breakdown_base}")
print(f"  full toy ParamBreakdown (FINEGRAINED-analog): {breakdown_fg}")
assert breakdown_base.total == 380 and breakdown_base.active == 284
assert breakdown_fg.total == 332 and breakdown_fg.active == 236
# moe_total/moe_active both shrink by exactly (shared shrink - router growth)
assert breakdown_base.moe_total - breakdown_fg.moe_total == (96 - 24) - (32 - 8) == 48
assert breakdown_base.moe_active - breakdown_fg.moe_active == 48

# ============================================================================
# Part 2 -- combinatorics: "a more flexible combination of activated
# experts" (Dai et al. 2024), quantified on this doc's own numbers.
# ============================================================================
print("\n" + "=" * 78)
print("PART 2 -- combinatorial flexibility, C(E, k), toy and real")
print("=" * 78)

c_toy_base = n_choose_k(2, 1)
c_toy_fg = n_choose_k(8, 4)
assert c_toy_base == math.comb(2, 1)
assert c_toy_fg == math.comb(8, 4)
print(f"  toy:  C(2,1) = {c_toy_base}   ->   C(8,4) = {c_toy_fg}"
      f"   ({c_toy_fg / c_toy_base:.0f}x more combinations, same FFN budget)")

c_real_base = n_choose_k(BASELINE.n_routed_experts, BASELINE.top_k)
c_real_fg = n_choose_k(FINEGRAINED.n_routed_experts, FINEGRAINED.top_k)
assert c_real_base == math.comb(8, 2) == 28
assert c_real_fg == math.comb(32, 8) == 10_518_300
print(f"  real: C(8,2) = {c_real_base:,}   ->   C(32,8) = {c_real_fg:,}"
      f"   ({c_real_fg / c_real_base:,.1f}x more combinations)")

# ============================================================================
# Part 3 -- Nguyen & Salazar's SmallInit formula, derived from Glorot &
# Bengio's compromise plus the standard 4x FFN expansion, checked against
# the real preset constant.
# ============================================================================
print("\n" + "=" * 78)
print("PART 3 -- sqrt(2/(5*d_model)): Glorot compromise + 4x FFN expansion")
print("=" * 78)

d = 256  # BASELINE's own d_model
a, b = 1 / d, 1 / (4 * d)
harmonic_mean_check = 2 * a * b / (a + b)  # harmonic mean of the two single-sided rules
xavier_via_fanin_fanout = 2 / (d + 4 * d)
print(f"  Glorot harmonic-mean identity: 2*(1/d)*(1/4d)/((1/d)+(1/4d)) = "
      f"{harmonic_mean_check:.10f}")
print(f"  2/(fan_in+fan_out) = 2/(d+4d)                                = "
      f"{xavier_via_fanin_fanout:.10f}")
assert abs(harmonic_mean_check - xavier_via_fanin_fanout) < 1e-15

exact_2_5d = 2 / (5 * d)
print(f"  2/(d+4d) == 2/(5d)? {2/(d+4*d)} == {2/(5*d)} -> "
      f"{2/(d+4*d) == 2/(5*d)}")
assert 2 / (d + 4 * d) == 2 / (5 * d)

std_v2 = math.sqrt(exact_2_5d)
print(f"  std = sqrt(2/(5*256)) = sqrt({exact_2_5d:.10f}) = {std_v2:.10f}")
print(f"  configs.presets._V2_INIT_STD (rounded to 4 dp) = "
      f"{presets_module._V2_INIT_STD}")
assert round(std_v2, 4) == presets_module._V2_INIT_STD == 0.0395
assert BASELINE_V2.init_std == presets_module._V2_INIT_STD

# the road-not-taken: this project's OWN d_ff/d_model ratio is 2 (512/256),
# not the 4 the borrowed formula assumes.
own_ratio_std = math.sqrt(2 / ((1 + BASELINE.d_ff / BASELINE.d_model) * d))
print(f"\n  BASELINE's own d_ff/d_model ratio = {BASELINE.d_ff}/{BASELINE.d_model} = "
      f"{BASELINE.d_ff / BASELINE.d_model} (not 4)")
print(f"  had the formula used THIS project's own ratio: "
      f"sqrt(2/((1+2)*256)) = {own_ratio_std:.10f}  (vs the borrowed {std_v2:.10f})")

# ============================================================================
# Part 4 -- the real BASELINE / BASELINE_V2 / FINEGRAINED presets, through
# the real count_params, reproducing journal 034's own table exactly.
# ============================================================================
print("\n" + "=" * 78)
print("PART 4 -- real presets, real count_params (journal 034's own table)")
print("=" * 78)

real_breakdowns = {}
for name, preset in [("BASELINE", BASELINE), ("BASELINE_V2", BASELINE_V2),
                      ("FINEGRAINED", FINEGRAINED)]:
    m = QwenMoEModel(preset)
    b = count_params(m)
    real_breakdowns[name] = b
    sparsity = b.total / b.active
    print(f"  {name:12s} total={b.total:>10,}  active={b.active:>10,}  "
          f"sparsity={sparsity:.2f}x")

assert real_breakdowns["BASELINE"].total == 23_477_504
assert real_breakdowns["BASELINE"].active == 9_321_728
assert real_breakdowns["BASELINE_V2"].total == 23_477_888
assert real_breakdowns["BASELINE_V2"].active == 9_322_112
assert real_breakdowns["FINEGRAINED"].total == 21_745_280
assert real_breakdowns["FINEGRAINED"].active == 7_589_504
assert round(real_breakdowns["BASELINE"].total / real_breakdowns["BASELINE"].active, 2) == 2.52
assert round(real_breakdowns["FINEGRAINED"].total / real_breakdowns["FINEGRAINED"].active, 2) == 2.87

qknorm_cost_total = real_breakdowns["BASELINE_V2"].total - real_breakdowns["BASELINE"].total
qknorm_cost_active = real_breakdowns["BASELINE_V2"].active - real_breakdowns["BASELINE"].active
head_dim = BASELINE.d_model // BASELINE.n_heads
predicted_qknorm_cost = 2 * head_dim * BASELINE.n_layers
print(f"\n  BASELINE_V2 - BASELINE: +{qknorm_cost_total} total, +{qknorm_cost_active} active")
print(f"  predicted (2 * head_dim * n_layers = 2*{head_dim}*{BASELINE.n_layers}) = "
      f"{predicted_qknorm_cost}")
assert qknorm_cost_total == qknorm_cost_active == predicted_qknorm_cost == 384

# ============================================================================
# Part 5 -- untrained logit-std RATIO across the three presets, measured
# over 12 seeds, checked against doc 023 Sec 5.5's formula applied to the
# init_std change alone.
# ============================================================================
print("\n" + "=" * 78)
print("PART 5 -- untrained logit-std ratio: predicted (from init_std) vs measured")
print("=" * 78)


def measure_logit_std(preset: ModelConfig, seed: int) -> float:
    torch.manual_seed(seed)
    model = QwenMoEModel(preset)
    torch.manual_seed(seed + 10_000)
    ids = torch.randint(0, preset.vocab_size, (4, 32))
    logits, _ = model(ids)
    return logits.std().item()


N_SEEDS = 12
ratios_v2_base = []
ratios_fg_v2 = []
for seed in range(N_SEEDS):
    lb = measure_logit_std(BASELINE, seed)
    lv2 = measure_logit_std(BASELINE_V2, seed)
    lfg = measure_logit_std(FINEGRAINED, seed)
    ratios_v2_base.append(lv2 / lb)
    ratios_fg_v2.append(lfg / lv2)
    print(f"  seed={seed:2d}  baseline={lb:.4f}  v2={lv2:.4f}  fg={lfg:.4f}"
          f"  v2/base={lv2/lb:.4f}  fg/v2={lfg/lv2:.4f}")

mean_v2_base = statistics.mean(ratios_v2_base)
stdev_v2_base = statistics.stdev(ratios_v2_base)
mean_fg_v2 = statistics.mean(ratios_fg_v2)
stdev_fg_v2 = statistics.stdev(ratios_fg_v2)

predicted_v2_base = BASELINE_V2.init_std / BASELINE.init_std
predicted_fg_v2 = 1.0  # identical init_std AND identical d_model

print(f"\n  mean v2/base ratio = {mean_v2_base:.4f} +/- {stdev_v2_base:.4f}"
      f"   predicted (init_std ratio) = {predicted_v2_base:.4f}")
print(f"  mean fg/v2 ratio   = {mean_fg_v2:.4f} +/- {stdev_fg_v2:.4f}"
      f"   predicted (identical init_std/d_model) = {predicted_fg_v2:.4f}")

rel_err_v2_base = abs(mean_v2_base - predicted_v2_base) / predicted_v2_base
rel_err_fg_v2 = abs(mean_fg_v2 - predicted_fg_v2) / predicted_fg_v2
print(f"  relative error: v2/base = {rel_err_v2_base:.2%}   fg/v2 = {rel_err_fg_v2:.2%}")
assert rel_err_v2_base < 0.01
assert rel_err_fg_v2 < 0.01

# ---- the raw (non-ratio) prediction, for honesty about its looser fit ----
alpha = 2.0
phi = math.exp(-0.5 * alpha * alpha) / math.sqrt(2.0 * math.pi)
Phi = 0.5 * (1.0 + math.erf(alpha / math.sqrt(2.0)))
std_factor = math.sqrt(1.0 - (2.0 * alpha * phi) / (2.0 * Phi - 1.0))
assert abs(std_factor - 0.8796256610342398) < 1e-12  # doc 023 Part 2's own constant

raw_pred_base = math.sqrt(BASELINE.d_model) * BASELINE.init_std * std_factor
raw_pred_v2 = math.sqrt(BASELINE_V2.d_model) * BASELINE_V2.init_std * std_factor
mean_measured_base = statistics.mean(
    measure_logit_std(BASELINE, s) for s in range(N_SEEDS)
)
mean_measured_v2 = statistics.mean(
    measure_logit_std(BASELINE_V2, s) for s in range(N_SEEDS)
)
print(f"\n  raw prediction  sqrt(d_model)*init_std*{std_factor:.4f}:"
      f"  base={raw_pred_base:.4f}  v2={raw_pred_v2:.4f}")
print(f"  raw measured (mean of {N_SEEDS} seeds):"
      f"                base={mean_measured_base:.4f}  v2={mean_measured_v2:.4f}")
print("  (looser fit than the ratio -- see doc 023 Sec 5.5's own caveats "
      "on this approximation; the RATIO cancels the systematic gap.)")

# ---- cross-check against journal 034's own quoted table (independent run) ----
journal_quoted = {"BASELINE": 0.284, "BASELINE_V2": 0.558, "FINEGRAINED": 0.557}
print(f"\n  vs. journal 034's own quoted untrained-logit-std column "
      f"(a separate, unrecorded run):")
mean_measured_fg = statistics.mean(measure_logit_std(FINEGRAINED, s) for s in range(N_SEEDS))
means_by_name = {
    "BASELINE": mean_measured_base,
    "BASELINE_V2": mean_measured_v2,
    "FINEGRAINED": mean_measured_fg,
}
for name, journal_val in journal_quoted.items():
    m = means_by_name[name]
    rel = abs(m - journal_val) / journal_val
    print(f"    {name:12s} this doc's mean({N_SEEDS} seeds)={m:.4f}"
          f"   journal 034={journal_val}   reldiff={rel:.2%}")
    assert rel < 0.02

print("\nAll assertions passed.")
