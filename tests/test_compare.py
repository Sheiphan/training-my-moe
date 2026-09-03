import dataclasses
import math

import pytest
import torch

from qwenmoe.config import DataConfig, ModelConfig, TrainConfig
from qwenmoe.data.download import prepared_data_dir
from qwenmoe.tokenizer.char import CharTokenizer
from qwenmoe.train.compare import (
    ComparisonReport,
    VariantResult,
    match_d_ff_to_param_budget,
    run_comparison,
)
from qwenmoe.train.scaling import count_params
from qwenmoe.model.transformer import QwenMoEModel


def _write_prepared_data(cache_dir) -> tuple[DataConfig, int]:
    """Fabricates a tiny on-disk `tokenizer.json` + `tokens.pt` pair in the
    exact layout `scripts/prepare_data.py` writes, so `run_comparison` can
    load it the same way `scripts/train.py` does -- without a real network
    download."""
    text = "abcdefghij" * 200  # 2000 chars -> plenty for a small fixed slice
    tokenizer = CharTokenizer()
    tokenizer.train(text)
    token_ids = tokenizer.encode(text)

    data_config = DataConfig(corpus="tiny_shakespeare", tokenizer_type="char", cache_dir=cache_dir)
    out_dir = prepared_data_dir(data_config)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(out_dir / "tokenizer.json")
    torch.save(torch.tensor(token_ids, dtype=torch.long), out_dir / "tokens.pt")
    return data_config, tokenizer.vocab_size


def _tiny_model_config(vocab_size: int, **overrides) -> ModelConfig:
    defaults = dict(
        vocab_size=vocab_size,
        d_model=16,
        n_layers=1,
        n_heads=2,
        n_kv_heads=1,
        d_ff=32,
        n_routed_experts=4,
        n_shared_experts=1,
        top_k=2,
        max_seq_len=16,
    )
    defaults.update(overrides)
    return ModelConfig(**defaults)


def _tiny_train_config(**overrides) -> TrainConfig:
    defaults = dict(
        batch_size=4,
        seq_len=8,
        total_tokens=10 * 4 * 8,
        lr=1e-2,
        warmup_steps=2,
        checkpoint_interval_steps=10,
        eval_interval_steps=10,
        log_interval_steps=10,
        device="cpu",
        seed=0,
    )
    defaults.update(overrides)
    return TrainConfig(**defaults)


def test_run_comparison_returns_report_with_every_variant(tmp_path):
    data_config, vocab_size = _write_prepared_data(tmp_path / "cache")
    data_config = dataclasses.replace(data_config, fixed_slice_tokens=300, fixed_slice_seed=0)
    base_config = _tiny_model_config(vocab_size)
    variants = {
        "rope": {"positional_scheme": "rope"},
        "nope": {"positional_scheme": "nope"},
    }

    report = run_comparison(
        base_config, variants, data_config, _tiny_train_config(), output_dir=tmp_path / "runs"
    )

    assert isinstance(report, ComparisonReport)
    assert set(report.variants) == {"rope", "nope"}
    for name, result in report.variants.items():
        assert isinstance(result, VariantResult)
        assert result.config.positional_scheme == name
        assert math.isfinite(result.final_train_loss)
        assert math.isfinite(result.final_val_loss)
        assert math.isfinite(result.val_perplexity)
        assert result.tokens_per_sec > 0
        assert result.log_path.exists()


def test_run_comparison_uses_the_identical_slice_for_every_variant(tmp_path, monkeypatch):
    """Requirement 12.6: the fixed slice must be cut exactly once and
    shared, not re-cut (even identically) per variant."""
    import qwenmoe.train.compare as compare_module

    data_config, vocab_size = _write_prepared_data(tmp_path / "cache")
    data_config = dataclasses.replace(data_config, fixed_slice_tokens=300, fixed_slice_seed=7)
    base_config = _tiny_model_config(vocab_size)
    variants = {"a": {}, "b": {"positional_scheme": "nope"}}

    calls = []
    real_make_fixed_slice = compare_module.make_fixed_slice

    def _spy(all_tokens, n_tokens, seed):
        calls.append((n_tokens, seed))
        return real_make_fixed_slice(all_tokens, n_tokens, seed)

    monkeypatch.setattr(compare_module, "make_fixed_slice", _spy)

    run_comparison(
        base_config, variants, data_config, _tiny_train_config(), output_dir=tmp_path / "runs"
    )

    assert calls == [(300, 7)]


def test_run_comparison_variant_overrides_win_over_base_config(tmp_path):
    data_config, vocab_size = _write_prepared_data(tmp_path / "cache")
    data_config = dataclasses.replace(data_config, fixed_slice_tokens=300, fixed_slice_seed=0)
    base_config = _tiny_model_config(vocab_size, activation="swiglu")
    variants = {"gelu_variant": {"activation": "gelu"}}

    report = run_comparison(
        base_config, variants, data_config, _tiny_train_config(), output_dir=tmp_path / "runs"
    )

    assert report.variants["gelu_variant"].config.activation == "gelu"
    assert base_config.activation == "swiglu"  # base_config itself is untouched


def test_run_comparison_full_preset_dict_overrides_every_field(tmp_path):
    """The CLI (task 15.3) passes `dataclasses.asdict(preset)` as the
    override dict for each variant -- a full config, not a sparse delta.
    `vocab_size` should still come from the tokenizer, not the preset."""
    data_config, vocab_size = _write_prepared_data(tmp_path / "cache")
    data_config = dataclasses.replace(data_config, fixed_slice_tokens=300, fixed_slice_seed=0)
    unrelated_base = _tiny_model_config(vocab_size, d_model=16, n_heads=2, n_kv_heads=1)
    full_preset = _tiny_model_config(999, d_model=8, n_heads=1, n_kv_heads=1, activation="geglu")
    variants = {"preset_variant": dataclasses.asdict(full_preset)}

    report = run_comparison(
        unrelated_base, variants, data_config, _tiny_train_config(), output_dir=tmp_path / "runs"
    )

    result_config = report.variants["preset_variant"].config
    assert result_config.d_model == 8
    assert result_config.activation == "geglu"
    assert result_config.vocab_size == vocab_size  # not the preset's placeholder 999


def test_run_comparison_requires_fixed_slice_tokens(tmp_path):
    data_config, vocab_size = _write_prepared_data(tmp_path / "cache")
    base_config = _tiny_model_config(vocab_size)

    with pytest.raises(ValueError, match="fixed_slice_tokens"):
        run_comparison(
            base_config, {"a": {}}, data_config, _tiny_train_config(), output_dir=tmp_path / "runs"
        )


def test_run_comparison_requires_prepared_data(tmp_path):
    data_config = DataConfig(
        corpus="tiny_shakespeare",
        tokenizer_type="char",
        cache_dir=tmp_path / "empty_cache",
        fixed_slice_tokens=100,
    )
    base_config = _tiny_model_config(vocab_size=50)

    with pytest.raises(FileNotFoundError):
        run_comparison(
            base_config, {"a": {}}, data_config, _tiny_train_config(), output_dir=tmp_path / "runs"
        )


# -- multi-seed (moe-modernization Requirement 12.1, 12.2) -----------------


def _prepared(tmp_path, **slice_overrides):
    data_config, vocab_size = _write_prepared_data(tmp_path / "cache")
    data_config = dataclasses.replace(
        data_config, fixed_slice_tokens=300, fixed_slice_seed=0, **slice_overrides
    )
    return data_config, vocab_size


def test_single_seed_default_keeps_scalar_fields_and_no_noise_flag(tmp_path):
    """Requirement 13.1 invariant: with the default (one) seed, the scalar
    metrics are that one run's numbers, std is 0, and no noise band is
    claimed."""
    data_config, vocab_size = _prepared(tmp_path)
    report = run_comparison(
        _tiny_model_config(vocab_size),
        {"base": {}, "nope": {"positional_scheme": "nope"}},
        data_config,
        _tiny_train_config(seed=0),
        output_dir=tmp_path / "runs",
    )

    assert report.seeds == (0,)
    for result in report.variants.values():
        assert result.seeds == (0,)
        assert len(result.per_seed_val_loss) == 1
        assert result.val_loss_std == 0.0
        assert result.val_loss_mean == pytest.approx(result.final_val_loss)
        assert result.within_noise_of_baseline is None
        assert result.comparison_mode == "compute_matched"
    # single-seed logs stay at the un-nested path
    assert report.variants["base"].log_path == tmp_path / "runs" / "base" / "logs" / "train.jsonl"


def test_multi_seed_reports_std_and_one_log_per_seed(tmp_path):
    """Requirement 12.1: >1 seed => a spread of per-seed losses (std > 0)
    and one training log per seed."""
    data_config, vocab_size = _prepared(tmp_path)
    seeds = (1, 2, 3)
    report = run_comparison(
        _tiny_model_config(vocab_size),
        {"base": {}, "gelu": {"activation": "gelu"}},
        data_config,
        _tiny_train_config(),
        output_dir=tmp_path / "runs",
        seeds=seeds,
    )

    assert report.seeds == seeds
    for name, result in report.variants.items():
        assert result.seeds == seeds
        assert len(result.per_seed_val_loss) == 3
        assert len(set(result.per_seed_val_loss)) > 1  # different seeds -> different runs
        assert result.val_loss_std > 0.0
        assert len(result.log_paths) == 3
        for seed in seeds:
            assert (tmp_path / "runs" / name / f"seed_{seed}" / "logs" / "train.jsonl").exists()


def test_multi_seed_sets_within_noise_flag_relative_to_baseline(tmp_path):
    data_config, vocab_size = _prepared(tmp_path)
    report = run_comparison(
        _tiny_model_config(vocab_size),
        {"base": {}, "twin": {}},  # identical config -> delta should be within noise
        data_config,
        _tiny_train_config(),
        output_dir=tmp_path / "runs",
        seeds=(1, 2),
        baseline_name="base",
    )

    assert report.variants["base"].within_noise_of_baseline is None  # baseline itself
    assert report.variants["twin"].within_noise_of_baseline is True


def test_summary_table_shows_mean_plus_minus_std_and_a_note(tmp_path):
    data_config, vocab_size = _prepared(tmp_path)
    report = run_comparison(
        _tiny_model_config(vocab_size),
        {"base": {}, "nope": {"positional_scheme": "nope"}},
        data_config,
        _tiny_train_config(),
        output_dir=tmp_path / "runs",
        seeds=(1, 2),
    )
    table = report.summary_table()
    assert "±" in table
    assert "baseline" in table
    assert ("within noise" in table) or ("differs" in table)


# -- param-matched mode (moe-modernization Requirement 12.4) ---------------


def test_match_d_ff_brings_gelu_variant_within_2pct_of_baseline_params():
    baseline = _tiny_model_config(vocab_size=50, activation="swiglu", d_ff=64)
    target = count_params(QwenMoEModel(baseline)).total

    gelu = _tiny_model_config(vocab_size=50, activation="gelu", d_ff=64)
    assert count_params(QwenMoEModel(gelu)).total < target  # 2 matrices/expert, not 3

    matched_d_ff = match_d_ff_to_param_budget(gelu, target)
    matched_total = count_params(
        QwenMoEModel(dataclasses.replace(gelu, d_ff=matched_d_ff))
    ).total
    assert abs(matched_total - target) / target <= 0.02
    assert matched_d_ff > 64  # GELU needs a wider FFN to match SwiGLU's param count


def test_run_comparison_param_matched_marks_mode_and_matches_params(tmp_path):
    data_config, vocab_size = _prepared(tmp_path)
    base_config = _tiny_model_config(vocab_size, activation="swiglu", d_ff=64)
    baseline_total = count_params(QwenMoEModel(dataclasses.replace(base_config))).total

    report = run_comparison(
        base_config,
        {"base": {}, "gelu": {"activation": "gelu"}},
        data_config,
        _tiny_train_config(),
        output_dir=tmp_path / "runs",
        param_matched=True,
        baseline_name="base",
    )

    assert report.variants["base"].comparison_mode == "compute_matched"
    gelu = report.variants["gelu"]
    assert gelu.comparison_mode == "param_matched"
    gelu_total = count_params(QwenMoEModel(gelu.config)).total
    assert abs(gelu_total - baseline_total) / baseline_total <= 0.02
