"""Named ModelConfig presets — config-as-code, no YAML/JSON schema layer.

`vocab_size` on every preset here is a placeholder sized for its typical
tokenizer — `scripts/train.py`/`scripts/run_comparison.py` override it with
the actual trained tokenizer's `vocab_size` before constructing a model.
"""
import dataclasses
import math

from qwenmoe.config import ModelConfig

# ~23.5M total / ~9.3M active params per token with vocab_size=4096 (a
# from-scratch BPE trained on TinyStories) — see requirements.md's
# "Illustrative baseline sizing" for the derivation.
BASELINE = ModelConfig(
    vocab_size=4096,
    d_model=256,
    n_layers=6,
    n_heads=8,
    n_kv_heads=4,
    d_ff=512,
    n_shared_experts=1,
    n_routed_experts=8,
    top_k=2,
    max_seq_len=256,
)

# Requirements.md's "Tier 1" smoke-test config: tiny-Shakespeare, char-level
# (~65-char vocab), sized well under 1M total params so the full data-prep
# -> train -> checkpoint pipeline runs end to end in minutes (task 10.2),
# not to produce a good model.
SMOKE_TEST = ModelConfig(
    vocab_size=65,
    d_model=64,
    n_layers=2,
    n_heads=4,
    n_kv_heads=2,
    d_ff=128,
    n_shared_experts=1,
    n_routed_experts=4,
    top_k=2,
    max_seq_len=128,
)

# Task 15 ablation-study presets (Requirement 12.1-12.3): each swaps exactly
# one architectural axis relative to SMOKE_TEST (not BASELINE) so
# `scripts/run_comparison.py` stays fast enough to actually run on a
# laptop — see design.md's illustrative preset list (POS_NOPE, POS_ALIBI,
# MOE_LOSSFREE, ACT_GEGLU, ACT_GELU) and journal entry 015.
POS_NOPE = dataclasses.replace(SMOKE_TEST, positional_scheme="nope")
POS_ALIBI = dataclasses.replace(SMOKE_TEST, positional_scheme="alibi")
ACT_GEGLU = dataclasses.replace(SMOKE_TEST, activation="geglu")
ACT_GELU = dataclasses.replace(SMOKE_TEST, activation="gelu")
MOE_LOSSFREE = dataclasses.replace(SMOKE_TEST, moe_balancing="loss_free")

# moe-modernization task 17 (Requirement 12.5): the modernization axes this
# spec added, as one-axis-off-SMOKE_TEST ModelConfig presets. The optimizer
# (OPT_MUON) and LR-schedule (SCHED_WSD) axes are TrainConfig, not
# ModelConfig, so they live only in `scripts/run_ablation_study.py::VARIANTS`
# as per-variant train-config overrides — see journal entry 033.
ROUTE_SIGMOID = dataclasses.replace(SMOKE_TEST, router_scoring="sigmoid")
ATT_QKNORM = dataclasses.replace(SMOKE_TEST, qk_norm=True)
MTP_D1 = dataclasses.replace(SMOKE_TEST, mtp_depth=1)

# Fine-grained experts (DeepSeekMoE / Requirement 5.5): more, smaller
# experts with a proportionally higher top_k, plus more shared experts to
# hold the shared-FFN width constant against the smaller d_ff. Every FFN
# parameter budget is held ~constant vs SMOKE_TEST — routed total
# 16 × 32 = 512 (vs 4 × 128), routed active top_8 × 32 = 256 (vs
# top_2 × 128), shared 4 × 32 = 128 (vs 1 × 128) — so a comparison isolates
# granularity itself, not capacity (only the router grows, ~unavoidably,
# with more experts to score). Multi-axis by nature, so it is exempt from
# the "exactly one axis" preset check.
MOE_FINEGRAINED = dataclasses.replace(
    SMOKE_TEST, n_routed_experts=16, d_ff=32, top_k=8, n_shared_experts=4
)


# ---------------------------------------------------------------------------
# moe-modernization task 18 (Requirement 12.5, 13.4): the *recommended*
# configs after this spec. BASELINE / SMOKE_TEST are deliberately left
# untouched so the parent spec's tests and trained checkpoints stay valid;
# these are the ones to reach for on a new run. See journal entry 034 for
# the change-by-change evidence from tasks 2–17.
# ---------------------------------------------------------------------------

# Width-aware "small init" from Nguyen & Salazar, "Transformers without
# Tears" (2019): std = sqrt(2 / (5 * d_model)), the smallest std among the
# Transformer's weights, designed to pair with pre-norm blocks and the
# 1/sqrt(2 * n_layers) residual-projection scaling this project already
# applies (task 7 / journal 023) -- together they enable stable training
# at a large, near-warmup-free learning rate. For d_model = 256 this is
# ~0.0395, vs the fixed 0.02 BASELINE inherits from GPT-2's d_model = 768.
_V2_INIT_STD = round(math.sqrt(2 / (5 * BASELINE.d_model)), 4)

# The recommended dense-expert config after this spec. Changes vs BASELINE,
# each with its task/journal:
#   - qk_norm=True         -- bounded attention-logit variance (task 6 / 022)
#   - router_scoring="sigmoid" -- DeepSeek-V3 independent per-expert
#                            affinities, normalized over the selected k
#                            (task 9 / 025)
#   - init_std=_V2_INIT_STD -- width-aware small init (task 7 / 023)
# The pure-win TrainConfig changes (tuned AdamW betas/wd, grad clip,
# min_lr_ratio floor) are already the defaults after tasks 2-3; pair this
# with TrainConfig(lr_schedule="wsd") for a token-budget-flexible run
# (task 3 / 021).
BASELINE_V2 = dataclasses.replace(
    BASELINE,
    qk_norm=True,
    router_scoring="sigmoid",
    init_std=_V2_INIT_STD,
)

# BASELINE_V2 + DeepSeekMoE fine-grained expert segmentation: 8 experts of
# d_ff=512 (top-2) become 32 experts of d_ff=128 (top-8), holding routed
# FFN parameters (32*128 == 8*512) and per-token active routed FFN
# (8*128 == 2*512) constant. The single shared expert also narrows to
# d_ff=128, a deliberate small capacity trade for finer routing (task 11 /
# 027, task 17 / 033).
FINEGRAINED = dataclasses.replace(
    BASELINE_V2,
    n_routed_experts=32,
    d_ff=128,
    top_k=8,
    n_shared_experts=1,
)


# ---------------------------------------------------------------------------
# Standalone scale-up request (not a moe-modernization spec task): a bigger
# dataset (tiny_stories, char-level -- ~22.5M real tokens, vs. tiny_shakespeare's
# ~1M) and a bigger model (~100M params, vs. BASELINE's ~23.5M), for smoke-testing
# scripts/prepare_data.py -> scripts/train.py at this larger scale rather than
# a full/converged training run. Built off BASELINE_V2 (qk_norm, sigmoid
# routing, width-aware init), not plain BASELINE, since journal 034/035
# recommend those as the defaults for any new run.
# ---------------------------------------------------------------------------

# BASELINE_V2's own _V2_INIT_STD is only valid at BASELINE's d_model=256;
# recomputed here for this preset's own (2x bigger) d_model.
_SCALE_100M_INIT_STD = round(math.sqrt(2 / (5 * 512)), 4)

# ~104.7M total / ~38.6M active params per token at vocab_size=91 (a
# from-scratch char tokenizer's real size over TinyStories' validation
# split -- see qwenmoe/data/download.py), sparsity 2.71x, matching
# BASELINE's own ratio exactly. Every width/head dimension is exactly 2x
# BASELINE's; n_layers is 6->7 (not a full 2x -- a full 2x depth landed
# near ~120M) so the total lands close to the requested ~100M.
SCALE_100M = dataclasses.replace(
    BASELINE_V2,
    vocab_size=91,
    d_model=512,
    n_layers=7,
    n_heads=8,
    n_kv_heads=4,
    d_ff=1024,
    n_shared_experts=1,
    n_routed_experts=8,
    top_k=2,
    max_seq_len=256,
    init_std=_SCALE_100M_INIT_STD,
)

# Follow-up scale-up: "how big a model can I actually train on this device"
# (MacBook, M4 Pro, 24GB unified memory). Empirically measured (not
# theoretical) via torch.mps.current_allocated_memory() peak sampling --
# ~104.7M and ~334M both trained fast/comfortably (<8GB peak); ~400-500M
# hit a real cliff (20-100x wall-clock slowdown from unified-memory
# swap/compression thrashing, not a clean OOM error) with this machine's
# normal everyday load (VS Code, Pylance, idle Jupyter kernels) already
# sharing the same 24GB. ~300M is the confirmed-safe ceiling at batch=16/
# seq=256, fp32 AdamW.
_SCALE_300M_INIT_STD = round(math.sqrt(2 / (5 * 768)), 4)

# ~334.4M total / ~122.0M active params per token at vocab_size=91,
# sparsity 2.74x. d_model/head_dim/d_ff are 1.5x SCALE_100M's; n_layers
# 7->10 -- the combination measured to land closest to the ~300M safe
# ceiling (a full proportional scale-up overshot it).
SCALE_300M = dataclasses.replace(
    BASELINE_V2,
    vocab_size=91,
    d_model=768,
    n_layers=10,
    n_heads=12,
    n_kv_heads=4,
    d_ff=1536,
    n_shared_experts=1,
    n_routed_experts=8,
    top_k=2,
    max_seq_len=256,
    init_std=_SCALE_300M_INIT_STD,
)

