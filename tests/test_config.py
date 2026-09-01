import pytest

from qwenmoe.config import DataConfig, ModelConfig, TrainConfig


def test_model_config_valid_defaults():
    cfg = ModelConfig(vocab_size=100)
    assert cfg.d_model == 256
    assert cfg.n_layers == 6
    assert cfg.n_heads == 8
    assert cfg.n_kv_heads == 4
    assert cfg.n_shared_experts == 1
    assert cfg.n_routed_experts == 8
    assert cfg.top_k == 2
    assert cfg.positional_scheme == "rope"
    assert cfg.activation == "swiglu"
    assert cfg.moe_balancing == "aux_loss"
    assert cfg.attention_variant == "gqa"
    assert cfg.tie_word_embeddings is True


def test_model_config_mla_valid():
    cfg = ModelConfig(vocab_size=100, attention_variant="mla", mla_latent_dim=64)
    assert cfg.mla_latent_dim == 64


@pytest.mark.parametrize(
    "overrides",
    [
        {"top_k": 9, "n_routed_experts": 8},                    # top_k > n_routed_experts
        {"n_heads": 8, "n_kv_heads": 3},                        # n_heads % n_kv_heads != 0
        {"n_kv_heads": 0},                                      # n_kv_heads < 1
        {"n_heads": 8, "n_kv_heads": 16},                       # n_kv_heads > n_heads
        {"d_model": 100, "n_heads": 8},                         # d_model % n_heads != 0
        {"vocab_size": 0},                                      # vocab_size < 1
        {"n_routed_experts": 0},                                # n_routed_experts < 1
        {"n_shared_experts": -1},                               # n_shared_experts < 0
        {"attention_variant": "mla", "mla_latent_dim": None},   # MLA missing latent dim
        {"mla_latent_dim": 0},                                  # mla_latent_dim < 1
        {"lora_rank": 0},                                       # lora_rank < 1
        {"max_seq_len": 0},                                     # max_seq_len < 1
    ],
)
def test_model_config_invalid_combinations_raise(overrides):
    kwargs = {"vocab_size": 100, **overrides}
    with pytest.raises(ValueError):
        ModelConfig(**kwargs)


def test_train_config_valid_defaults():
    cfg = TrainConfig(batch_size=8, seq_len=128, total_tokens=1000)
    assert cfg.lr == pytest.approx(3e-4)
    assert cfg.warmup_steps == 200
    assert cfg.max_eval_batches == 50
    assert cfg.log_interval_steps == 10
    assert cfg.device == "auto"
    assert cfg.seed == 1337


@pytest.mark.parametrize(
    "overrides",
    [
        {"batch_size": 0},
        {"seq_len": 0},
        {"total_tokens": 0},
        {"lr": 0},
        {"lr": -1},
        {"warmup_steps": -1},
        {"checkpoint_interval_steps": 0},
        {"eval_interval_steps": 0},
        {"max_eval_batches": 0},
        {"log_interval_steps": 0},
    ],
)
def test_train_config_invalid_combinations_raise(overrides):
    kwargs = {"batch_size": 8, "seq_len": 128, "total_tokens": 1000, **overrides}
    with pytest.raises(ValueError):
        TrainConfig(**kwargs)


def test_data_config_valid_defaults():
    cfg = DataConfig(corpus="tiny_shakespeare", tokenizer_type="char")
    assert cfg.bpe_vocab_size == 4096
    assert cfg.fixed_slice_tokens is None
    assert cfg.fixed_slice_seed == 0


def test_data_config_coerces_string_cache_dir():
    from pathlib import Path

    cfg = DataConfig(corpus="tiny_shakespeare", tokenizer_type="char", cache_dir="some/dir")
    assert cfg.cache_dir == Path("some/dir")


@pytest.mark.parametrize(
    "overrides",
    [
        {"bpe_vocab_size": 0},
        {"fixed_slice_tokens": 0},
    ],
)
def test_data_config_invalid_combinations_raise(overrides):
    kwargs = {"corpus": "tiny_shakespeare", "tokenizer_type": "char", **overrides}
    with pytest.raises(ValueError):
        DataConfig(**kwargs)


def test_baseline_preset_constructs():
    from configs.presets import BASELINE

    assert BASELINE.vocab_size == 4096
    assert BASELINE.d_model == 256
    assert BASELINE.n_layers == 6
    assert BASELINE.n_heads == 8
    assert BASELINE.n_kv_heads == 4
    assert BASELINE.n_shared_experts == 1
    assert BASELINE.n_routed_experts == 8
    assert BASELINE.top_k == 2


@pytest.mark.parametrize(
    "preset_name,field,expected",
    [
        ("POS_NOPE", "positional_scheme", "nope"),
        ("POS_ALIBI", "positional_scheme", "alibi"),
        ("ACT_GEGLU", "activation", "geglu"),
        ("ACT_GELU", "activation", "gelu"),
        ("MOE_LOSSFREE", "moe_balancing", "loss_free"),
    ],
)
def test_ablation_presets_swap_exactly_one_axis_from_smoke_test(preset_name, field, expected):
    """Task 15.3: each ablation preset must construct validly and differ
    from SMOKE_TEST in only the one axis it names."""
    import dataclasses

    import configs.presets as presets

    preset = getattr(presets, preset_name)
    assert getattr(preset, field) == expected

    other_fields = {f.name for f in dataclasses.fields(preset)} - {field}
    for other_field in other_fields:
        assert getattr(preset, other_field) == getattr(presets.SMOKE_TEST, other_field), (
            f"{preset_name}.{other_field} unexpectedly differs from SMOKE_TEST"
        )
