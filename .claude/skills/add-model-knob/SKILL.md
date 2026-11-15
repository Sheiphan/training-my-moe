---
name: add-model-knob
description: Use when adding a new option to ModelConfig, TrainConfig or DataConfig (a new model feature, optimizer, schedule, routing or training flag). Keeps the off-by-default and bit-identical guarantees. Trigger words: add a flag, new config field, new knob, new option, modernization.
---

# Add a config knob without breaking old configs

This project's central invariant: **every new knob defaults to the pre-existing behavior**, so a config written before the knob exists trains bit-identically. Follow this order.

1. **Add the field** to the right dataclass in `qwenmoe/config.py`, defaulting to "off" / the old behavior, with a short comment saying what it does and which journal entry or task it belongs to. Add validation in `__post_init__` with a clear `ValueError`.
2. **Implement the feature** in `qwenmoe/`. For `ModelConfig` fields touching attention or MoE, there are two paths that must stay in step: the fast `forward` and the explicit `forward_with_trace` (on `Attention`, `MLAAttention`, `MoELayer`, `TransformerBlock`, `QwenMoEModel`). Change both, or state in the code why one path is exempt.
3. **Wire the CLI** for a `TrainConfig` field: add an argparse flag in `scripts/train.py`. `tests/test_train_script.py` matches parser destinations to dataclass fields and fails if a field has no flag. If the field affects ablations, check `scripts/run_comparison.py` and `scripts/run_ablation_study.py` too.
4. **Presets.** Only touch `configs/presets.py` if a preset should turn it on. Presets use `dataclasses.replace`. Any width-dependent value (like `init_std`) must be recomputed for each preset's own `d_model`.
5. **Tests**, in the matching `tests/test_*.py`:
   - Add the new default to `test_model_config_modernization_defaults_are_pre_spec_behavior` or `test_train_config_modernization_defaults` in `tests/test_config.py`.
   - Add a test that the default path is unchanged (outputs equal to before, not just "runs").
   - Add a test for the on path, and for the validation error.
   - If both forward paths exist, test that they agree numerically.
6. **Run** `uv run pytest -q`. Then, if a doc describes the changed code, run its `docs/math/verify_*.py` (see `new-math-doc`).
7. **Check the site model.** `site/shared/qwenmoe-model.js` mirrors the model's shape and parameter arithmetic for the tensor walk and model figure. If the knob changes parameter counts or tensor shapes, update it and re-run the `refresh-site` skill.
8. Record the change in a journal entry (`new-journal-entry`) if it is more than a trivial flag.
