"""Verifies every hand-computed number in
`037-scale-presets-and-capacity-check.md` against the real
`scripts/check_capacity.py` module (and the real `qwenmoe/` model it
estimates for) -- no number in that document is asserted, only quoted
after being produced here.

Run: uv run python docs/math/verify_037_scale_presets_and_capacity_check.py
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.check_capacity import (  # noqa: E402
    BYTES_PER_PARAM_INFERENCE_FP32,
    BYTES_PER_PARAM_TRAIN_FP32,
    HardwareInfo,
    analyze,
    estimate_activation_bytes,
    estimate_params,
    kv_cache_bytes,
    recommend_max_total_params,
    verify_empirically,
)
from qwenmoe.config import ModelConfig
from qwenmoe.model.kv_cache import LatentKVCache, LayerKVCache, build_kv_cache
from qwenmoe.model.transformer import QwenMoEModel
from qwenmoe.train.scaling import count_params

import scripts.check_capacity as _cc  # noqa: E402

# Doc 037 Parts 1-9 derive the heuristic as it stood BEFORE journal 044's
# recalibration (x2.2 backward factor, plus a 3-copy logits term). Pin those two
# knobs to their pre-044 values so every number in Parts 1-9 stays reproducible;
# the Addendum at the end of this script restores the real constants and checks
# what 044 changed.
_REAL_FACTOR, _REAL_LOGIT_COPIES = _cc._ACTIVATION_BACKWARD_FACTOR, _cc._LOGIT_COPIES
_cc._ACTIVATION_BACKWARD_FACTOR, _cc._LOGIT_COPIES = 1.0, 0

PASS = "PASS"


def check(label: str, got, want) -> None:
    ok = got == want
    print(f"  [{PASS if ok else 'FAIL'}] {label}: got={got!r} want={want!r}")
    assert ok, f"{label}: got {got!r}, want {want!r}"


print("=" * 70)
print("Part 1 -- the running example TOY config, and estimate_params")
print("=" * 70)

TOY = ModelConfig(
    vocab_size=7,
    d_model=4,
    n_layers=2,
    n_heads=2,
    n_kv_heads=1,
    d_ff=3,
    n_shared_experts=1,
    n_routed_experts=3,
    top_k=2,
    qk_norm=True,
    tie_word_embeddings=True,
)
print(f"TOY = {TOY}")

total, active = estimate_params(TOY)
print(f"\nestimate_params(TOY) -> total={total}, active={active}")
check("total_params", total, 464)
check("active_params", active, 392)
print(f"  sparsity total/active = {total / active:.4f} (hand: 464/392 = {464/392:.4f})")

print("\ncross-check: estimate_params is construction-free -- does it match")
print("a REAL nn.Module's real parameter count (qwenmoe.train.scaling.count_params)?")
model = QwenMoEModel(TOY)
real = count_params(model)
print(f"  count_params(real model) -> total={real.total}, active={real.active}")
check("estimate_params vs real model: total", total, real.total)
check("estimate_params vs real model: active", active, real.active)
print(f"  max |estimate - real| = {max(abs(total - real.total), abs(active - real.active))}")

print("\naside: activation='gelu' halves the expert from 3 matrices to 2")
toy_gelu = replace(TOY, activation="gelu")
total_gelu, _ = estimate_params(toy_gelu)
model_gelu = QwenMoEModel(toy_gelu)
real_gelu = count_params(model_gelu)
expert_params_swiglu = 3 * TOY.d_model * TOY.d_ff
expert_params_gelu = 2 * TOY.d_model * TOY.d_ff
hand_total_gelu = total - TOY.n_layers * (
    TOY.n_shared_experts + TOY.n_routed_experts
) * (expert_params_swiglu - expert_params_gelu)
check("gelu total_params (hand)", total_gelu, hand_total_gelu)
check("gelu total_params vs real model", total_gelu, real_gelu.total)
print(f"  swiglu expert_params={expert_params_swiglu}, gelu expert_params={expert_params_gelu}")

print("\naside: tie_word_embeddings=False adds exactly vocab_size*d_model")
toy_untied = replace(TOY, tie_word_embeddings=False)
total_untied, _ = estimate_params(toy_untied)
check("untied total_params", total_untied, total + TOY.vocab_size * TOY.d_model)
model_untied = QwenMoEModel(toy_untied)
real_untied = count_params(model_untied)
check("untied total_params vs real model", total_untied, real_untied.total)

print("\n" + "=" * 70)
print("Part 2 -- estimate_activation_bytes and kv_cache_bytes")
print("=" * 70)

BATCH, SEQ = 2, 3
act_bytes = estimate_activation_bytes(BATCH, SEQ, TOY.n_layers, TOY.d_model, TOY.n_heads)
print(f"estimate_activation_bytes(batch={BATCH}, seq={SEQ}, ...) -> {act_bytes}")
check("activation_bytes", act_bytes, 7488)

kv_bytes_est = kv_cache_bytes(TOY, batch_size=1, seq_len=SEQ)
print(f"kv_cache_bytes(TOY, batch=1, seq={SEQ}) -> {kv_bytes_est}")
check("kv_cache_bytes (GQA)", kv_bytes_est, 96)

print("\ncross-check: kv_cache_bytes's formula vs a REAL LayerKVCache.nbytes")
caches = build_kv_cache(model)
assert isinstance(caches[0], LayerKVCache)
head_dim = TOY.d_model // TOY.n_heads
real_kv_total = 0
for cache in caches:
    k = torch.zeros(1, TOY.n_kv_heads, SEQ, head_dim)
    v = torch.zeros(1, TOY.n_kv_heads, SEQ, head_dim)
    cache.append(k, v)
    real_kv_total += cache.nbytes
print(f"  sum of real LayerKVCache.nbytes across {TOY.n_layers} layers = {real_kv_total}")
check("kv_cache_bytes formula vs real cache objects", kv_bytes_est, real_kv_total)

print("\naside: MLA cache vs. a GQA cache at the SAME head_dim (a separate tiny")
print("shape, since MLAAttention requires head_dim % 4 == 0 -- TOY's head_dim=2")
print("doesn't qualify, so this aside cannot reuse TOY unchanged)")
ASIDE_GQA = ModelConfig(
    vocab_size=7, d_model=8, n_layers=2, n_heads=2, n_kv_heads=1, d_ff=3,
    n_shared_experts=1, n_routed_experts=3, top_k=2, attention_variant="gqa",
)
ASIDE_MLA = replace(ASIDE_GQA, attention_variant="mla", mla_latent_dim=2)
aside_head_dim = ASIDE_GQA.d_model // ASIDE_GQA.n_heads
kv_bytes_aside_gqa = kv_cache_bytes(ASIDE_GQA, batch_size=1, seq_len=SEQ)
kv_bytes_aside_mla = kv_cache_bytes(ASIDE_MLA, batch_size=1, seq_len=SEQ)
print(f"  head_dim={aside_head_dim} for both; kv_cache_bytes: GQA={kv_bytes_aside_gqa}, MLA={kv_bytes_aside_mla}")
check("kv_cache_bytes (MLA) < (GQA) at equal head_dim", kv_bytes_aside_mla < kv_bytes_aside_gqa, True)
check("kv_cache_bytes (GQA aside) exact value", kv_bytes_aside_gqa, 192)
check("kv_cache_bytes (MLA aside) exact value", kv_bytes_aside_mla, 96)

model_mla = QwenMoEModel(ASIDE_MLA)
caches_mla = build_kv_cache(model_mla)
assert isinstance(caches_mla[0], LatentKVCache)
rope_head_dim = aside_head_dim // 2
real_kv_mla_total = 0
for cache in caches_mla:
    c_kv = torch.zeros(1, SEQ, ASIDE_MLA.mla_latent_dim)
    k_rope = torch.zeros(1, 1, SEQ, rope_head_dim)
    cache.append(c_kv, k_rope)
    real_kv_mla_total += cache.nbytes
check("kv_cache_bytes (MLA aside) formula vs real cache objects", kv_bytes_aside_mla, real_kv_mla_total)

print("\n" + "=" * 70)
print("Part 3 -- analyze() and recommend_max_total_params(), with an")
print("injected (deterministic, machine-independent) HardwareInfo")
print("=" * 70)

hw = HardwareInfo(device="cpu", total_system_memory=None, safe_budget_bytes=20_000)
report = analyze(TOY, batch_size=BATCH, seq_len=SEQ, hardware=hw)
print(f"CapacityReport: total_params={report.total_params}, active_params={report.active_params}")
print(f"  train_static_bytes={report.train_static_bytes}")
print(f"  train_activation_bytes={report.train_activation_bytes}")
print(f"  train_total_bytes={report.train_total_bytes}")
print(f"  inference_bytes={report.inference_bytes}")
print(f"  inference_with_kv_cache_bytes={report.inference_with_kv_cache_bytes}")
print(f"  fits(train_total_bytes) = {report.fits(report.train_total_bytes)}")

check("train_static_bytes", report.train_static_bytes, 464 * BYTES_PER_PARAM_TRAIN_FP32)
check("train_static_bytes (hand)", report.train_static_bytes, 7424)
check("train_activation_bytes", report.train_activation_bytes, 7488)
check("train_total_bytes", report.train_total_bytes, 14912)
check("inference_bytes", report.inference_bytes, 3104)
check("inference_with_kv_cache_bytes", report.inference_with_kv_cache_bytes, 1952)
check("fits(14912) at budget 20000", report.fits(report.train_total_bytes), True)

recommended = recommend_max_total_params(
    hw, batch_size=BATCH, seq_len=SEQ, d_model_hint=TOY.d_model, n_layers_hint=TOY.n_layers, n_heads_hint=TOY.n_heads
)
print(f"\nrecommend_max_total_params(same shape, same budget) -> {recommended}")
check("recommended max total params", recommended, 407)
print(
    f"  note: {recommended} < {report.total_params} even though {report.total_params} "
    f"params already 'fits' -- recommend_max_total_params applies its own extra 0.7x "
    f"safety margin on top of the raw budget, analyze().fits() does not."
)

print("\n" + "=" * 70)
print("Part 4 -- verify_empirically(): a REAL measured number, honestly")
print("compared against the rough estimate")
print("=" * 70)

real_peak = verify_empirically(TOY, batch_size=BATCH, seq_len=SEQ, device="cpu")
print(f"verify_empirically(TOY, cpu) -> {real_peak:,} bytes ({real_peak / 1e6:.1f} MB)")
print(f"  (estimate said train_total_bytes = {report.train_total_bytes:,} bytes)")
print(
    "  these are NOT expected to agree: on the CPU fallback, verify_empirically "
    "reads resource.getrusage(...).ru_maxrss -- whole-PROCESS resident memory "
    "(Python + PyTorch + libraries), which for a 464-parameter model is dominated "
    "by interpreter/library overhead, not by this model's own activations."
)
check("real measured peak is positive", real_peak > 0, True)

print("\n" + "=" * 70)
print("Part 5 -- the --verify safety guard: the SAME analyze() report that")
print("says 'DOES NOT FIT' is what stops a real hang from being triggered")
print("=" * 70)

tiny_budget_hw = HardwareInfo(device="cpu", total_system_memory=None, safe_budget_bytes=100)
tight_report = analyze(TOY, batch_size=BATCH, seq_len=SEQ, hardware=tiny_budget_hw)
guard_margin = tight_report.train_total_bytes * 1.2
print(f"synthetic tiny budget: 100 bytes; train_total_bytes={tight_report.train_total_bytes}")
print(f"  fits(train_total_bytes)              = {tight_report.fits(tight_report.train_total_bytes)}")
print(f"  fits(train_total_bytes * 1.2)         = {tight_report.fits(guard_margin)}")
print(
    "  -- this is exactly the condition scripts/check_capacity.py's main() checks "
    "before calling verify_empirically(): a config the estimate already flags as "
    "over budget is never handed to a real model construction + real training step, "
    "which is what hung (not crashed) on the too-big configs in journal 037."
)
check("tiny-budget report already fails at 1x", tight_report.fits(tight_report.train_total_bytes), False)
check("tiny-budget report fails at the 1.2x guard margin too", tight_report.fits(guard_margin), False)

print("\n" + "=" * 70)
print("Addendum (Part 10) -- what journal 044's recalibration changed")
print("=" * 70)

# Restore the REAL constants (the pin above was only to keep Parts 1-9 reproducible).
_cc._ACTIVATION_BACKWARD_FACTOR, _cc._LOGIT_COPIES = _REAL_FACTOR, _REAL_LOGIT_COPIES
check("real backward factor", _cc._ACTIVATION_BACKWARD_FACTOR, 2.2)
check("real logit copies", _cc._LOGIT_COPIES, 3)
check("real CUDA usable fraction", _cc._CUDA_USABLE_FRACTION, 0.90)

# (a) the x2.2 factor, on TOY: Part 4's 7488 becomes int(7488 * 2.2).
act_new_no_vocab = estimate_activation_bytes(BATCH, SEQ, TOY.n_layers, TOY.d_model, TOY.n_heads)
print(f"estimate_activation_bytes(no vocab_size)   = {act_new_no_vocab}   (7488 * 2.2 = {7488 * 2.2})")
check("x2.2 on the pre-044 activation estimate", act_new_no_vocab, int(7488 * 2.2))

# (b) the logits term: tokens * vocab * 4 bytes * 3 copies, added AFTER the factor.
tokens = BATCH * SEQ
logit_term = tokens * TOY.vocab_size * 4 * 3
act_new_with_vocab = estimate_activation_bytes(BATCH, SEQ, TOY.n_layers, TOY.d_model, TOY.n_heads, TOY.vocab_size)
print(f"logit term = {tokens} * {TOY.vocab_size} * 4 * 3 = {logit_term}")
print(f"estimate_activation_bytes(vocab_size={TOY.vocab_size}) = {act_new_with_vocab}")
check("logit term (hand)", logit_term, 504)
check("activation with logits = factor term + logit term", act_new_with_vocab, act_new_no_vocab + logit_term)
check("activation with logits (hand)", act_new_with_vocab, 16977)

# (c) where 2.2 comes from: the four H100 est-vs-real pairs quoted in the comment above
# `_ACTIVATION_BACKWARD_FACTOR` in check_capacity.py (QUOTED from journal 044, not re-measured here).
pairs = [(13.41, 27.86), (26.88, 59.48), (22.85, 52.87), (20.84, 45.69)]
ratios = [real / est for est, real in pairs]
mean_ratio = sum(ratios) / len(ratios)
print("measured/estimated ratios:", [round(r, 3) for r in ratios], f"mean = {mean_ratio:.4f}")
check("mean measured ratio rounds to the 2.2 factor", round(mean_ratio, 1), 2.2)

print("\n" + "=" * 70)
print("ALL CHECKS PASSED")
print("=" * 70)
