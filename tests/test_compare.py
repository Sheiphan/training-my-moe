import dataclasses
import math

import pytest
import torch

from qwenmoe.config import DataConfig, ModelConfig, TrainConfig
from qwenmoe.data.download import prepared_data_dir
from qwenmoe.tokenizer.char import CharTokenizer
from qwenmoe.train.compare import ComparisonReport, VariantResult, run_comparison


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
