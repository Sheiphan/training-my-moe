"""Named ModelConfig presets — config-as-code, no YAML/JSON schema layer.

`vocab_size` on every preset here is a placeholder sized for its typical
tokenizer — `scripts/train.py`/`scripts/run_comparison.py` override it with
the actual trained tokenizer's `vocab_size` before constructing a model.
"""
import dataclasses

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

