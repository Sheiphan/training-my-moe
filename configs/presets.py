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

