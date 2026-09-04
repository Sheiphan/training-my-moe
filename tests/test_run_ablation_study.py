"""moe-modernization task 17 (Requirement 12.5): the refreshed ablation
menu. Every VARIANTS entry -- ModelConfig plus any per-variant TrainConfig
override -- must construct and train a few steps on the identical fixed
slice.
"""
import dataclasses
import math

import pytest
import torch

from qwenmoe.config import DataConfig, ModelConfig, TrainConfig
from qwenmoe.data.download import prepared_data_dir
from qwenmoe.tokenizer.char import CharTokenizer
from qwenmoe.train.compare import run_comparison

from scripts.run_ablation_study import BASELINE_NAME, VARIANTS

NEW_AXIS_VARIANTS = ["OPT_MUON", "SCHED_WSD", "ROUTE_SIGMOID", "ATT_QKNORM", "MTP_D1", "MOE_FINEGRAINED"]


def _write_prepared_data(cache_dir):
    text = "abcdefghij" * 300
    tokenizer = CharTokenizer()
    tokenizer.train(text)
    data_config = DataConfig(
        corpus="tiny_shakespeare",
        tokenizer_type="char",
        cache_dir=cache_dir,
        fixed_slice_tokens=400,
        fixed_slice_seed=0,
    )
    out_dir = prepared_data_dir(data_config)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(out_dir / "tokenizer.json")
    torch.save(torch.tensor(tokenizer.encode(text), dtype=torch.long), out_dir / "tokens.pt")
    return data_config


def _tiny_train_config(**overrides):
    defaults = dict(
        batch_size=4,
        seq_len=8,
        total_tokens=6 * 4 * 8,
        lr=1e-2,
        warmup_steps=2,
        checkpoint_interval_steps=6,
        eval_interval_steps=6,
        log_interval_steps=6,
        device="cpu",
        seed=0,
    )
    defaults.update(overrides)
    return TrainConfig(**defaults)


# a small SMOKE_TEST-shaped base so training is fast; every VARIANTS override
# still applies cleanly on top of it
_TINY_BASE = ModelConfig(
    vocab_size=10,
    d_model=16,
    n_layers=1,
    n_heads=2,
    n_kv_heads=1,
    d_ff=16,
    n_routed_experts=4,
    n_shared_experts=1,
    top_k=2,
    max_seq_len=8,
)


def test_every_variant_entry_is_a_valid_four_tuple():
    for name, entry in VARIANTS.items():
        axis, label, model_config, train_ov = entry
        assert isinstance(axis, str) and axis
        assert isinstance(label, str) and label
        assert isinstance(model_config, ModelConfig)
        assert isinstance(train_ov, dict)
        # a train override only names real TrainConfig fields
        for key in train_ov:
            assert key in {f.name for f in dataclasses.fields(TrainConfig)}


def _variant_overrides_on_tiny_base(name):
    """The VARIANTS entry's ModelConfig, re-expressed as a delta from
    `_TINY_BASE` (so training stays cheap) -- only the fields that differ
    from SMOKE_TEST, which are exactly the ablation axes."""
    from configs.presets import SMOKE_TEST

    _, _, model_config, train_ov = VARIANTS[name]
    model_delta = {
        f.name: getattr(model_config, f.name)
        for f in dataclasses.fields(ModelConfig)
        if getattr(model_config, f.name) != getattr(SMOKE_TEST, f.name)
    }
    return model_delta, train_ov


@pytest.mark.parametrize("name", NEW_AXIS_VARIANTS)
def test_new_ablation_variant_constructs_and_trains(tmp_path, name):
    data_config = _write_prepared_data(tmp_path / "cache")
    model_delta, train_ov = _variant_overrides_on_tiny_base(name)

    report = run_comparison(
        _TINY_BASE,
        {BASELINE_NAME: {}, name: model_delta},
        data_config,
        _tiny_train_config(),
        output_dir=tmp_path / "runs",
        train_overrides={name: train_ov} if train_ov else None,
        baseline_name=BASELINE_NAME,
    )

    result = report.variants[name]
    assert math.isfinite(result.final_val_loss)
    assert result.log_path.exists()


def test_train_overrides_reach_the_trainer(tmp_path):
    """A Muon override produces a Muon (CombinedOptimizer) checkpoint; the
    baseline without the override stays on AdamW."""
    data_config = _write_prepared_data(tmp_path / "cache")

    report = run_comparison(
        _TINY_BASE,
        {"adamw": {}, "muon": {}},
        data_config,
        _tiny_train_config(),
        output_dir=tmp_path / "runs",
        train_overrides={"muon": {"optimizer": "muon"}},
        baseline_name="adamw",
    )

    muon_ckpt = torch.load(
        tmp_path / "runs" / "muon" / "checkpoints" / "step_6.pt", weights_only=False
    )
    adamw_ckpt = torch.load(
        tmp_path / "runs" / "adamw" / "checkpoints" / "step_6.pt", weights_only=False
    )
    # CombinedOptimizer's state_dict wraps a list of sub-optimizer states;
    # a plain AdamW's has torch's "state"/"param_groups" keys.
    assert "sub_optimizers" in muon_ckpt["optimizer_state_dict"]
    assert "sub_optimizers" not in adamw_ckpt["optimizer_state_dict"]
    assert muon_ckpt["train_config"].optimizer == "muon"
    assert adamw_ckpt["train_config"].optimizer == "adamw"
    assert report.variants["muon"].config == report.variants["adamw"].config  # only train differs
