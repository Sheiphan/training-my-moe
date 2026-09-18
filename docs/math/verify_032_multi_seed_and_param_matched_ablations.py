"""Verifies docs/math/032-multi-seed-and-param-matched-ablations.md against
the real qwenmoe.train.compare module -- no fabricated numbers.

Pins one tiny ModelConfig/TrainConfig/DataConfig (d_model=4, 1 layer, 1
head, 2 routed experts, a 2-symbol "ab" corpus) and:

  1. re-derives the closed-form total(d_ff) = C + k*d_ff formula for both
     activations from count_params()'s own component breakdown, and checks
     it against a real sweep of d_ff = 1..11;
  2. hand-traces match_d_ff_to_param_budget's doubling + bisection loop and
     asserts it matches the real function's return value;
  3. runs run_comparison(..., seeds=(1, 2, 3)) twice -- once compute-matched
     (the default), once with param_matched=True -- on the identical base
     config/data/seeds, and re-derives every mean/std/noise-band number by
     hand-equivalent arithmetic, asserting it matches statistics.fmean /
     statistics.stdev / math.hypot / within_noise_of_baseline exactly;
  4. confirms bit-for-bit reproducibility: the baseline's three per-seed
     val losses come out identical across two separate run_comparison
     calls (separate temp dirs, separate process state);
  5. exercises match_d_ff_to_param_budget's raise path on a variant that
     is structurally larger than the target even at d_ff=1.

Run with: uv run python docs/math/verify_032_multi_seed_and_param_matched_ablations.py
"""
from __future__ import annotations

import dataclasses
import math
import statistics
import tempfile
import time
from pathlib import Path

import torch

from qwenmoe.config import DataConfig, ModelConfig, TrainConfig
from qwenmoe.data.download import prepared_data_dir
from qwenmoe.model.transformer import QwenMoEModel
from qwenmoe.tokenizer.char import CharTokenizer
from qwenmoe.train.compare import _total_params, match_d_ff_to_param_budget, run_comparison
from qwenmoe.train.scaling import count_params


def section(title: str) -> None:
    print(f"\n{'=' * 10} {title} {'=' * 10}")


def _write_prepared_data(cache_dir: Path) -> tuple[DataConfig, CharTokenizer]:
    """The exact tiny fixed corpus this doc uses: 300 repeats of 'ab' (600
    characters, 2 distinct symbols), written to disk in the layout
    scripts/prepare_data.py produces, so run_comparison loads it exactly
    the way scripts/train.py does -- no network download."""
    text = "ab" * 300
    tokenizer = CharTokenizer()
    tokenizer.train(text)
    token_ids = tokenizer.encode(text)

    data_config = DataConfig(corpus="tiny_shakespeare", tokenizer_type="char", cache_dir=cache_dir)
    out_dir = prepared_data_dir(data_config)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(out_dir / "tokenizer.json")
    torch.save(torch.tensor(token_ids, dtype=torch.long), out_dir / "tokens.pt")
    return data_config, tokenizer


BASE_CONFIG = ModelConfig(
    vocab_size=2,  # overwritten by run_comparison from the real tokenizer; stated here for count_params calls
    d_model=4,
    n_layers=1,
    n_heads=1,
    n_kv_heads=1,
    d_ff=4,
    n_routed_experts=2,
    n_shared_experts=0,
    top_k=1,
    max_seq_len=8,
    activation="swiglu",
)

TRAIN_CONFIG = TrainConfig(
    batch_size=2,
    seq_len=4,
    total_tokens=2 * 2 * 4,  # 2 optimizer steps
    lr=1e-2,
    warmup_steps=1,
    checkpoint_interval_steps=100,
    eval_interval_steps=100,
    log_interval_steps=100,
    device="cpu",
    seed=0,
)

SEEDS = (1, 2, 3)


# -- Part 1: the affine total(d_ff) = C + k * d_ff ledger -------------------

section("Part 1: total-parameter ledger and the C + k*d_ff formula")

breakdown = count_params(QwenMoEModel(BASE_CONFIG))
print("count_params(base, swiglu, d_ff=4):", breakdown)

model = QwenMoEModel(BASE_CONFIG)
block = model.blocks[0]
embedding = sum(p.numel() for p in model.embedding.parameters())
final_norm = sum(p.numel() for p in model.final_norm.parameters())
attn_norm = sum(p.numel() for p in block.attn_norm.parameters())
attn = sum(p.numel() for p in block.attn.parameters())
moe_norm = sum(p.numel() for p in block.moe_norm.parameters())
router = sum(p.numel() for p in block.moe.router.parameters())
shared = sum(p.numel() for p in block.moe.shared_experts.parameters())
routed_total = sum(p.numel() for p in block.moe.routed_experts.parameters())
print(
    f"embedding={embedding} final_norm={final_norm} attn_norm={attn_norm} attn={attn} "
    f"moe_norm={moe_norm} router={router} shared={shared} routed_total(d_ff=4)={routed_total}"
)
C = embedding + final_norm + attn_norm + attn + moe_norm + router + shared
print("C (d_ff-independent total) =", C)
assert C == 92, C
assert embedding + final_norm == 12
assert attn_norm + attn == 68
assert moe_norm + router + shared == 12
assert C + routed_total == breakdown.total == 188

for activation, matrices_per_expert, expected_k in [("swiglu", 3, 24), ("gelu", 2, 16)]:
    k = (BASE_CONFIG.n_shared_experts + BASE_CONFIG.n_routed_experts) * matrices_per_expert * BASE_CONFIG.d_model
    assert k == expected_k, (activation, k, expected_k)
    for d_ff in range(1, 12):
        cfg = dataclasses.replace(BASE_CONFIG, activation=activation, d_ff=d_ff)
        real_total = _total_params(cfg)
        predicted = C + k * d_ff
        assert real_total == predicted, (activation, d_ff, real_total, predicted)
    print(f"activation={activation}: total(d_ff) = {C} + {k}*d_ff  (checked d_ff=1..11 against real count_params)")

TARGET = _total_params(BASE_CONFIG)
assert TARGET == 188
GELU_AT_4 = dataclasses.replace(BASE_CONFIG, activation="gelu")
print("baseline total (swiglu, d_ff=4) =", TARGET)
print("gelu variant total at d_ff=4    =", _total_params(GELU_AT_4))


# -- Part 2: hand-tracing match_d_ff_to_param_budget -------------------------

section("Part 2: match_d_ff_to_param_budget's doubling + bisection trace")


def traced_match(variant_config: ModelConfig, target: int, tolerance: float = 0.02) -> int:
    """Line-for-line copy of match_d_ff_to_param_budget's algorithm
    (qwenmoe/train/compare.py:160-196) with a print at every step, used
    only to display the trace -- the assertions below check its answer
    against the real function, not the other way around."""
    low_total = _total_params(dataclasses.replace(variant_config, d_ff=1))
    print(f"  guard: total(d_ff=1)={low_total}  vs  target*(1+tol)={target * (1 + tolerance):.2f}")
    if low_total > target * (1 + tolerance):
        raise ValueError("variant is larger than the baseline even at d_ff=1")

    lo, hi = 1, max(2, variant_config.d_ff)
    print(f"  init: lo={lo} hi={hi}")
    while _total_params(dataclasses.replace(variant_config, d_ff=hi)) < target:
        hi *= 2
        print(f"  double: hi -> {hi}")
    while lo < hi:
        mid = (lo + hi) // 2
        total_mid = _total_params(dataclasses.replace(variant_config, d_ff=mid))
        if total_mid < target:
            lo = mid + 1
            print(f"  mid={mid} total={total_mid} < target -> lo={lo}")
        else:
            hi = mid
            print(f"  mid={mid} total={total_mid} >= target -> hi={hi}")
    candidates = [lo] + ([lo - 1] if lo > 1 else [])
    best = min(candidates, key=lambda d: abs(_total_params(dataclasses.replace(variant_config, d_ff=d)) - target))
    print(f"  candidates={candidates} -> best={best}")
    return best


print("gelu (d_ff=4) matched to baseline total 188:")
traced_gelu = traced_match(GELU_AT_4, TARGET)
real_gelu = match_d_ff_to_param_budget(GELU_AT_4, TARGET)
assert traced_gelu == real_gelu == 6, (traced_gelu, real_gelu)
assert _total_params(dataclasses.replace(GELU_AT_4, d_ff=6)) == TARGET  # exact match, not just within 2%

print("\nswiglu (d_ff=4, i.e. the baseline's own shape) matched to target 188 (a no-op case):")
traced_twin = traced_match(BASE_CONFIG, TARGET)
real_twin = match_d_ff_to_param_budget(BASE_CONFIG, TARGET)
assert traced_twin == real_twin == 4, (traced_twin, real_twin)

# the raise path: a variant with more experts is structurally bigger than
# the target even at d_ff=1
BIGGER = dataclasses.replace(BASE_CONFIG, n_routed_experts=16, activation="gelu")
try:
    match_d_ff_to_param_budget(BIGGER, TARGET)
    raise AssertionError("expected ValueError")
except ValueError as e:
    print("\nraise path (n_routed_experts=16, d_ff=1):", e)


# -- Part 3: run_comparison with seeds, twice (compute- then param-matched) -

section("Part 3a: run_comparison(seeds=(1,2,3)) -- compute-matched (default)")

# Untimed warm-up call: pays for one-time process costs (imports, DataLoader
# worker setup, PyTorch's own lazy init) so the two timed calls below measure
# actual per-Trainer.fit cost, not cold-start overhead.
tmp_warm = Path(tempfile.mkdtemp())
data_config_warm, _ = _write_prepared_data(tmp_warm / "cache")
data_config_warm = dataclasses.replace(data_config_warm, fixed_slice_tokens=200, fixed_slice_seed=0)
run_comparison(BASE_CONFIG, {"base": {}}, data_config_warm, TRAIN_CONFIG, output_dir=tmp_warm / "runs")

tmp_1seed = Path(tempfile.mkdtemp())
data_config_1seed, _ = _write_prepared_data(tmp_1seed / "cache")
data_config_1seed = dataclasses.replace(data_config_1seed, fixed_slice_tokens=200, fixed_slice_seed=0)
t0 = time.time()
run_comparison(
    BASE_CONFIG,
    {"base": {}, "twin": {}, "gelu": {"activation": "gelu"}},
    data_config_1seed,
    TRAIN_CONFIG,
    output_dir=tmp_1seed / "runs",
)  # seeds=None -> the pre-task default, one seed, 3 Trainer.fit calls
elapsed_1seed = time.time() - t0
print(f"wall-clock for 3 variants x 1 seed (the pre-task default): {elapsed_1seed:.2f}s")

tmp_a = Path(tempfile.mkdtemp())
data_config_a, tokenizer = _write_prepared_data(tmp_a / "cache")
print(f"real CharTokenizer.vocab_size on 'ab'*300 = {tokenizer.vocab_size}")
assert tokenizer.vocab_size == 2
data_config_a = dataclasses.replace(data_config_a, fixed_slice_tokens=200, fixed_slice_seed=0)

t0 = time.time()
report_a = run_comparison(
    BASE_CONFIG,
    {"base": {}, "twin": {}, "gelu": {"activation": "gelu"}},
    data_config_a,
    TRAIN_CONFIG,
    output_dir=tmp_a / "runs",
    seeds=SEEDS,
    baseline_name="base",
)
elapsed_a = time.time() - t0
print(f"wall-clock for 3 variants x 3 seeds = 9 real Trainer.fit calls: {elapsed_a:.2f}s")
print(report_a.summary_table())
for name, r in report_a.variants.items():
    print(name, "d_ff=", r.config.d_ff, "mode=", r.comparison_mode, "per_seed=", r.per_seed_val_loss)

base_r = report_a.variants["base"]
twin_r = report_a.variants["twin"]
gelu_r = report_a.variants["gelu"]

assert base_r.per_seed_val_loss == twin_r.per_seed_val_loss  # identical config+seed -> bit-identical

for name, r in [("base", base_r), ("twin", twin_r), ("gelu", gelu_r)]:
    xs = r.per_seed_val_loss
    n = len(xs)
    hand_mean = sum(xs) / n
    hand_var = sum((x - hand_mean) ** 2 for x in xs) / (n - 1)
    hand_std = math.sqrt(hand_var)
    assert hand_mean == statistics.fmean(xs), (name, hand_mean, statistics.fmean(xs))
    assert abs(hand_std - statistics.stdev(xs)) < 1e-12, (name, hand_std, statistics.stdev(xs))
    assert hand_mean == r.val_loss_mean and hand_std == r.val_loss_std
    print(f"{name}: mean={hand_mean:.10f} std={hand_std:.10f} (n={n})")

delta_twin = abs(twin_r.val_loss_mean - base_r.val_loss_mean)
band_twin = math.hypot(twin_r.val_loss_std, base_r.val_loss_std)
assert math.hypot(twin_r.val_loss_std, base_r.val_loss_std) == math.sqrt(
    twin_r.val_loss_std ** 2 + base_r.val_loss_std ** 2
)
print(f"twin: |delta|={delta_twin:.10f} band={band_twin:.10f} within_noise={twin_r.within_noise_of_baseline}")
assert (delta_twin < band_twin) == twin_r.within_noise_of_baseline is True

delta_gelu = abs(gelu_r.val_loss_mean - base_r.val_loss_mean)
band_gelu = math.hypot(gelu_r.val_loss_std, base_r.val_loss_std)
print(f"gelu: |delta|={delta_gelu:.10f} band={band_gelu:.10f} within_noise={gelu_r.within_noise_of_baseline}")
assert (delta_gelu < band_gelu) == gelu_r.within_noise_of_baseline is False

# the standard-error-of-the-difference-of-means band would be this quadrature
# sum divided by sqrt(n) -- the code's band is sqrt(n) times looser
se_band_gelu = band_gelu / math.sqrt(len(SEEDS))
print(f"(aside) standard-error band would be {se_band_gelu:.10f}  ({math.sqrt(len(SEEDS)):.7f}x tighter)")
assert delta_gelu > se_band_gelu  # differs under either band, here

assert gelu_r.comparison_mode == "compute_matched"
assert base_r.log_path == tmp_a / "runs" / "base" / "seed_1" / "logs" / "train.jsonl"
assert base_r.log_path == base_r.log_paths[0]
assert len(base_r.log_paths) == 3


section("Part 3b: run_comparison(seeds=(1,2,3), param_matched=True) -- same config/data/seeds")

tmp_b = Path(tempfile.mkdtemp())
data_config_b, _ = _write_prepared_data(tmp_b / "cache")
data_config_b = dataclasses.replace(data_config_b, fixed_slice_tokens=200, fixed_slice_seed=0)

t0 = time.time()
report_b = run_comparison(
    BASE_CONFIG,
    {"base": {}, "twin": {}, "gelu": {"activation": "gelu"}},
    data_config_b,
    TRAIN_CONFIG,
    output_dir=tmp_b / "runs",
    seeds=SEEDS,
    param_matched=True,
    baseline_name="base",
)
elapsed_b = time.time() - t0
print(f"wall-clock for 3 variants x 3 seeds (param_matched=True): {elapsed_b:.2f}s")
print(report_b.summary_table())
for name, r in report_b.variants.items():
    print(name, "d_ff=", r.config.d_ff, "mode=", r.comparison_mode, "per_seed=", r.per_seed_val_loss)

base_b = report_b.variants["base"]
twin_b = report_b.variants["twin"]
gelu_b = report_b.variants["gelu"]

# reproducibility: identical seeds + config + data -> bit-identical val losses
# across two entirely separate run_comparison calls (separate temp dirs)
assert base_b.per_seed_val_loss == base_r.per_seed_val_loss

assert base_b.comparison_mode == "compute_matched"       # baseline is never adjusted
assert twin_b.comparison_mode == "param_matched"          # label flips even though d_ff is unchanged
assert gelu_b.comparison_mode == "param_matched"
assert twin_b.config.d_ff == 4
assert gelu_b.config.d_ff == 6
gelu_b_total = _total_params(gelu_b.config)
base_b_total = _total_params(base_b.config)
assert gelu_b_total == base_b_total == 188  # exact match here, well inside the 2% tolerance

print(
    f"gelu (param-matched, d_ff=6): mean={gelu_b.val_loss_mean:.10f} std={gelu_b.val_loss_std:.10f} "
    f"within_noise={gelu_b.within_noise_of_baseline}"
)
assert gelu_b.within_noise_of_baseline is True  # the compute-matched "differs" gap closes once matched

print("\nAll assertions passed.")
