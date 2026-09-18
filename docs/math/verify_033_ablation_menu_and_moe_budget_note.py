"""Reproduces every hand-computed number in
`033-ablation-menu-and-moe-budget-note.md`.

Run it:  uv run python docs/math/verify_033_ablation_menu_and_moe_budget_note.py

Journal entry 033 covers three things: the 14-variant `VARIANTS` menu,
`run_comparison`'s `train_overrides` plumbing, and `moe_budget_note`. The
first two are software wiring with no arithmetic to derive (a 4-tuple, a
`dataclasses.replace` call) -- this script, like its companion doc, is
`moe_budget_note` (`qwenmoe/train/scaling.py:103-141`) and the
`chinchilla_budget` (`scaling.py:91-100`) it calls twice, end to end.

Six scenarios, every one driving the REAL module -- none hand-fabricated:

  1. `chinchilla_budget` on the two numbers the main example needs.
  2. `moe_budget_note`'s main worked example: (total=1,000,000,
     active=250,000, sparsity 4.0x) -- this is not a number invented for
     the doc, it is `tests/test_scaling.py::
     test_moe_budget_note_states_both_bounds_and_the_dense_constant`'s own
     fixture, reused verbatim.
  3. The near-dense branch: (total=101,000, active=100,000, 1.01x) --
     likewise lifted verbatim from
     `test_moe_budget_note_near_dense_model_says_chinchilla_applies_directly`.
  4. The three `ValueError` guards -- `test_moe_budget_note_rejects_invalid_
     args`'s own parametrization: (0,10), (100,0), (50,100).
  5. A structural proof, on a real `QwenMoEModel`: `_tiny_config()`
     (`tests/test_scaling.py`, doc 009's own model) at `top_k=2` (sparsity
     1.5224x) vs. `top_k=4=n_routed_experts` (sparsity exactly 1.0) --
     `total` is unchanged (17,904 either way); only `active` moves to meet
     it. This is why sparsity is *never* < 1, not just usually >= 1 in
     practice.
  6. The real `BASELINE` preset (`configs/presets.py`), reproducing journal
     033's own quoted paragraph (`journey_of_a_model/
     033-ablation-menu-and-moe-budget-note.md:189-193`) byte for byte.

No fused/optimized alternate path exists for this function (it is nine
lines of scalar arithmetic and an f-string) -- there is nothing to
reconcile two ways against. The "agreement" checked here is the doc's own
by-hand arithmetic against the real function's return value.
"""
from __future__ import annotations

from configs.presets import BASELINE
from qwenmoe.config import ModelConfig
from qwenmoe.model.transformer import QwenMoEModel
from qwenmoe.train.scaling import chinchilla_budget, count_params, moe_budget_note


def _tiny_config(**overrides) -> ModelConfig:
    """Identical to tests/test_scaling.py::_tiny_config -- doc 009's model."""
    base = dict(
        vocab_size=50,
        d_model=16,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        d_ff=32,
        n_routed_experts=4,
        n_shared_experts=1,
        top_k=2,
        max_seq_len=16,
    )
    base.update(overrides)
    return ModelConfig(**base)


# ----------------------------------------------------------------------
# Part 1 -- chinchilla_budget on the two numbers moe_budget_note needs.
# ----------------------------------------------------------------------
print("================ Part 1 -- chinchilla_budget ================")
active_ref = chinchilla_budget(250_000)
total_ref = chinchilla_budget(1_000_000)
print(f"  chinchilla_budget(250_000)   = {active_ref:,}")
print(f"  chinchilla_budget(1_000_000) = {total_ref:,}")
assert active_ref == 5_000_000
assert total_ref == 20_000_000

# ----------------------------------------------------------------------
# Part 2 -- moe_budget_note, the main worked example (4.0x sparse).
# ----------------------------------------------------------------------
print("\n================ Part 2 -- moe_budget_note (4.0x sparse) ================")
sparsity = 1_000_000 / 250_000
note = moe_budget_note(total_params=1_000_000, active_params=250_000)
print(f"  sparsity = 1,000,000 / 250,000 = {sparsity:.4f}x")
print(f"  note: {note}")
assert sparsity == 4.0
assert note == (
    "MoE sparsity: 250,000 active of 1,000,000 total "
    "params (4.0x). Chinchilla's 20x tokens/param is a DENSE-model "
    "constant (total == active); for a sparse MoE the compute-optimal budget "
    "lies between 20 x active = 5,000,000 and 20 x total = 20,000,000 tokens "
    "(Clark et al. 2022). Use 20 x active as a floor, not the target."
)

# ----------------------------------------------------------------------
# Part 3 -- moe_budget_note, the near-dense branch (1.01x).
# ----------------------------------------------------------------------
print("\n================ Part 3 -- moe_budget_note (1.01x near-dense) ================")
sparsity_nd = 101_000 / 100_000
note_nd = moe_budget_note(total_params=101_000, active_params=100_000)
print(f"  sparsity = 101,000 / 100,000 = {sparsity_nd:.4f}x  (< 1.05 -> other branch)")
print(f"  note: {note_nd}")
assert sparsity_nd == 1.01
assert note_nd == (
    "model is ~dense (total 101,000 ~= active 100,000); "
    "chinchilla_budget's 20 x active = 2,000,000 tokens applies directly."
)

# ----------------------------------------------------------------------
# Part 4 -- the three ValueError guards.
# ----------------------------------------------------------------------
print("\n================ Part 4 -- guards ================")
expected_errors = {
    (0, 10): "total_params and active_params must be >= 1, got 0, 10",
    (100, 0): "total_params and active_params must be >= 1, got 100, 0",
    (50, 100): "total_params (50) cannot be less than active_params (100)",
}
for (total, active), expected_message in expected_errors.items():
    try:
        moe_budget_note(total, active)
    except ValueError as exc:
        print(f"  moe_budget_note({total}, {active}) -> ValueError: {exc}")
        assert str(exc) == expected_message
    else:
        raise AssertionError(f"expected ValueError for {(total, active)}")

# ----------------------------------------------------------------------
# Part 5 -- structural proof: sparsity is never < 1, and is exactly 1 iff
# top_k == n_routed_experts, on a real QwenMoEModel (not asserted from
# algebra alone -- built and counted).
# ----------------------------------------------------------------------
print(
    "\n================ Part 5 -- sparsity >= 1 is structural ================"
)
sparse_model = QwenMoEModel(_tiny_config())  # top_k=2 < n_routed_experts=4
dense_model = QwenMoEModel(_tiny_config(top_k=4))  # top_k == n_routed_experts

sparse_breakdown = count_params(sparse_model)
dense_breakdown = count_params(dense_model)
print(f"  top_k=2 (sparse): {sparse_breakdown}")
print(f"  top_k=4 (dense) : {dense_breakdown}")

assert sparse_breakdown.total == 17_904
assert sparse_breakdown.active == 11_760
assert dense_breakdown.total == 17_904  # top_k never touches embedding/attn/moe_total
assert dense_breakdown.active == 17_904  # moe_active == moe_total when top_k == E
assert sparse_breakdown.total == dense_breakdown.total

sparsity_sparse = sparse_breakdown.total / sparse_breakdown.active
sparsity_dense = dense_breakdown.total / dense_breakdown.active
print(f"  sparsity top_k=2 = {sparsity_sparse:.4f}x   (doc 009's own example)")
print(f"  sparsity top_k=4 = {sparsity_dense:.4f}x")
assert round(sparsity_sparse, 4) == 1.5224
assert sparsity_dense == 1.0

note_dense_real = moe_budget_note(dense_breakdown.total, dense_breakdown.active)
print(f"  note (top_k=4): {note_dense_real}")
assert note_dense_real == (
    "model is ~dense (total 17,904 ~= active 17,904); "
    "chinchilla_budget's 20 x active = 358,080 tokens applies directly."
)
assert chinchilla_budget(17_904) == 358_080

# ----------------------------------------------------------------------
# Part 6 -- the real BASELINE preset: journal 033's own quoted paragraph,
# reproduced byte for byte.
# ----------------------------------------------------------------------
print(
    "\n================ Part 6 -- BASELINE (journal 033's own numbers) ================"
)
baseline_model = QwenMoEModel(BASELINE)
baseline_breakdown = count_params(baseline_model)
baseline_sparsity = baseline_breakdown.total / baseline_breakdown.active
baseline_note = moe_budget_note(baseline_breakdown.total, baseline_breakdown.active)
print(f"  {baseline_breakdown}")
print(
    f"  sparsity = {baseline_breakdown.total:,} / {baseline_breakdown.active:,} "
    f"= {baseline_sparsity:.4f}x"
)
print(f"  note: {baseline_note}")

assert baseline_breakdown.total == 23_477_504
assert baseline_breakdown.active == 9_321_728
journal_paragraph = (
    "MoE sparsity: 9,321,728 active of 23,477,504 total params (2.5x). Chinchilla's "
    "20x tokens/param is a DENSE-model constant (total == active); for a sparse MoE "
    "the compute-optimal budget lies between 20 x active = 186,434,560 and "
    "20 x total = 469,550,080 tokens (Clark et al. 2022). Use 20 x active as a floor, "
    "not the target."
)
assert baseline_note == journal_paragraph, "does not match journal 033:189-193 verbatim"

print("\nAll assertions passed.")
