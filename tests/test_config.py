import pytest

from qwenmoe.config import DataConfig, ModelConfig, TrainConfig
from qwenmoe.data.download import prepared_data_dir


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


# -- moe-modernization spec: new config axes -------------------------------


def test_model_config_modernization_defaults_are_pre_spec_behavior():
    """Every field this spec adds must default to the old behavior so a
    config written before the spec is unchanged (Requirement 13.1)."""
    cfg = ModelConfig(vocab_size=100)
    assert cfg.router_scoring == "softmax"
    assert cfg.router_z_loss_weight == 0.0
    assert cfg.aux_loss_weight == pytest.approx(0.01)
    assert cfg.seq_balance_loss_weight == 0.0
    assert cfg.qk_norm is False
    assert cfg.mtp_depth == 0
    assert cfg.init_std == pytest.approx(0.02)


def test_train_config_modernization_defaults():
    cfg = TrainConfig(batch_size=8, seq_len=128, total_tokens=1000)
    assert cfg.optimizer == "adamw"
    assert cfg.betas == (0.9, 0.95)
    assert cfg.weight_decay == pytest.approx(0.1)
    assert cfg.grad_clip_norm == pytest.approx(1.0)
    assert cfg.lr_schedule == "cosine"
    assert cfg.min_lr_ratio == pytest.approx(0.1)
    assert cfg.decay_fraction == pytest.approx(0.2)
    assert cfg.precision == "fp32"
    assert cfg.z_loss_weight == 0.0
    assert cfg.document_masking is False
    assert cfg.grad_accum_steps == 1


def test_data_config_val_split_mode_default():
    cfg = DataConfig(corpus="tiny_shakespeare", tokenizer_type="char")
    assert cfg.val_split_mode == "tail"


def test_data_config_corpus_chars_defaults_to_none():
    cfg = DataConfig(corpus="fineweb_edu", tokenizer_type="bpe")
    assert cfg.corpus_chars is None


def test_data_config_rejects_non_positive_corpus_chars():
    with pytest.raises(ValueError):
        DataConfig(corpus="fineweb_edu", tokenizer_type="bpe", corpus_chars=0)


def test_prepared_data_dir_separates_corpus_sizes():
    """Two sizes of the same corpus must not share a prepared-data
    directory, or a bigger run silently trains on the smaller cache."""
    small = DataConfig(corpus="fineweb_edu", tokenizer_type="bpe", corpus_chars=400_000_000)
    large = DataConfig(corpus="fineweb_edu", tokenizer_type="bpe", corpus_chars=3_500_000_000)
    default = DataConfig(corpus="fineweb_edu", tokenizer_type="bpe")

    assert prepared_data_dir(small) != prepared_data_dir(large)
    assert prepared_data_dir(default) != prepared_data_dir(small)
    assert prepared_data_dir(small).name == "fineweb_edu_400Mchars_bpe4096"


@pytest.mark.parametrize(
    "overrides",
    [
        {"router_z_loss_weight": -0.1},
        {"aux_loss_weight": -1.0},
        {"seq_balance_loss_weight": -0.001},
        {"mtp_depth": -1},
        {"mtp_lambda": -0.5},
        {"init_std": 0.0},
        {"init_std": -0.02},
    ],
)
def test_model_config_modernization_invalid_raise(overrides):
    with pytest.raises(ValueError):
        ModelConfig(vocab_size=100, **overrides)


@pytest.mark.parametrize(
    "overrides",
    [
        {"betas": (0.0, 0.95)},
        {"betas": (0.9, 1.0)},
        {"adam_eps": 0.0},
        {"weight_decay": -0.1},
        {"grad_clip_norm": 0.0},
        {"muon_lr": 0.0},
        {"muon_momentum": 1.0},
        {"muon_momentum": -0.1},
        {"min_lr_ratio": 1.5},
        {"min_lr_ratio": -0.1},
        {"decay_fraction": 0.0},
        {"decay_fraction": 1.5},
        {"z_loss_weight": -1e-5},
    ],
)
def test_train_config_modernization_invalid_raise(overrides):
    kwargs = {"batch_size": 8, "seq_len": 128, "total_tokens": 1000, **overrides}
    with pytest.raises(ValueError):
        TrainConfig(**kwargs)


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
        {"grad_accum_steps": 0},
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


# -- modernized presets (moe-modernization task 18, Requirement 12.5, 13.4) --


def test_baseline_and_smoke_test_presets_are_left_untouched():
    """Requirement 13.4 / task 18: the parent-spec presets keep pre-spec
    defaults so existing tests and checkpoints stay valid."""
    from configs.presets import BASELINE, SMOKE_TEST

    for preset in (BASELINE, SMOKE_TEST):
        assert preset.qk_norm is False
        assert preset.router_scoring == "softmax"
        assert preset.init_std == pytest.approx(0.02)
        assert preset.mtp_depth == 0


def test_baseline_v2_applies_exactly_the_recommended_changes():
    import dataclasses
    import math

    from configs.presets import BASELINE, BASELINE_V2

    assert BASELINE_V2.qk_norm is True
    assert BASELINE_V2.router_scoring == "sigmoid"
    assert BASELINE_V2.init_std == pytest.approx(math.sqrt(2 / (5 * BASELINE.d_model)), abs=1e-4)

    # everything else is BASELINE
    changed = {"qk_norm", "router_scoring", "init_std"}
    for f in dataclasses.fields(BASELINE_V2):
        if f.name not in changed:
            assert getattr(BASELINE_V2, f.name) == getattr(BASELINE, f.name)


def test_finegrained_preset_is_baseline_v2_plus_expert_segmentation():
    from configs.presets import BASELINE, BASELINE_V2, FINEGRAINED

    # inherits the V2 modernization
    assert FINEGRAINED.qk_norm is True
    assert FINEGRAINED.router_scoring == "sigmoid"

    # fine-grained experts, routed FFN budget held vs BASELINE
    assert FINEGRAINED.n_routed_experts == 32
    assert FINEGRAINED.d_ff == 128
    assert FINEGRAINED.top_k == 8
    assert FINEGRAINED.n_shared_experts == 1
    assert FINEGRAINED.n_routed_experts * FINEGRAINED.d_ff == BASELINE.n_routed_experts * BASELINE.d_ff
    assert FINEGRAINED.top_k * FINEGRAINED.d_ff == BASELINE.top_k * BASELINE.d_ff


def test_modernized_presets_construct_a_valid_model():
    import dataclasses

    import torch

    from configs.presets import BASELINE_V2, FINEGRAINED
    from qwenmoe.model.transformer import QwenMoEModel

    for preset in (BASELINE_V2, FINEGRAINED):
        model = QwenMoEModel(dataclasses.replace(preset, vocab_size=128))
        ids = torch.randint(0, 128, (2, 16))
        logits, _ = model(ids)
        assert logits.shape == (2, 16, 128)
        assert torch.isfinite(logits).all()
        # Requirement 9.4: untrained logits stay O(1)
        assert logits.std().item() < 5.0


@pytest.mark.parametrize(
    "preset_name,field,expected",
    [
        ("POS_NOPE", "positional_scheme", "nope"),
        ("POS_ALIBI", "positional_scheme", "alibi"),
        ("ACT_GEGLU", "activation", "geglu"),
        ("ACT_GELU", "activation", "gelu"),
        ("MOE_LOSSFREE", "moe_balancing", "loss_free"),
        # moe-modernization task 17 (Requirement 12.5)
        ("ROUTE_SIGMOID", "router_scoring", "sigmoid"),
        ("ATT_QKNORM", "qk_norm", True),
        ("MTP_D1", "mtp_depth", 1),
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


def test_scale_100m_preset_constructs_near_100m_params():
    """Standalone scale-up request (not a moe-modernization spec task):
    ~100M params, built off BASELINE_V2's modernizations, for smoke-testing
    the pipeline on a bigger (tiny_stories, char-level) dataset."""
    import torch

    from configs.presets import SCALE_100M
    from qwenmoe.model.transformer import QwenMoEModel
    from qwenmoe.train.scaling import count_params

    assert SCALE_100M.d_model == 512
    assert SCALE_100M.n_layers == 7
    assert SCALE_100M.qk_norm is True
    assert SCALE_100M.router_scoring == "sigmoid"

    model = QwenMoEModel(SCALE_100M)
    breakdown = count_params(model)
    # measured ~104.7M total / ~38.6M active at this preset's vocab_size=91
    assert 95_000_000 <= breakdown.total <= 115_000_000
    assert breakdown.active < breakdown.total

    ids = torch.randint(0, SCALE_100M.vocab_size, (2, 16))
    logits, _ = model(ids)
    assert logits.shape == (2, 16, SCALE_100M.vocab_size)
    assert torch.isfinite(logits).all()


def test_scale_300m_preset_constructs_near_300m_params():
    """Follow-up hardware-capacity ask: ~300M params, empirically confirmed
    (torch.mps peak-memory sweep) as the safe training ceiling on this
    machine (M4 Pro, 24GB) at batch=16/seq=256 -- see repo memory notes."""
    import torch

    from configs.presets import SCALE_300M
    from qwenmoe.model.transformer import QwenMoEModel
    from qwenmoe.train.scaling import count_params

    assert SCALE_300M.d_model == 768
    assert SCALE_300M.n_layers == 10
    assert SCALE_300M.qk_norm is True
    assert SCALE_300M.router_scoring == "sigmoid"

    model = QwenMoEModel(SCALE_300M)
    breakdown = count_params(model)
    # measured ~334.4M total / ~122.0M active at this preset's vocab_size=91
    assert 310_000_000 <= breakdown.total <= 360_000_000
    assert breakdown.active < breakdown.total

    ids = torch.randint(0, SCALE_300M.vocab_size, (2, 16))
    logits, _ = model(ids)
    assert logits.shape == (2, 16, SCALE_300M.vocab_size)
    assert torch.isfinite(logits).all()


def test_moe_finegrained_preset_holds_ffn_params_constant():
    """moe-modernization task 17: MOE_FINEGRAINED changes granularity
    (more, smaller experts + higher top_k + more shared experts) while
    keeping every FFN parameter budget ~constant vs SMOKE_TEST, so a
    comparison isolates granularity, not capacity."""
    from configs.presets import MOE_FINEGRAINED, SMOKE_TEST

    def routed_total(c):
        return c.n_routed_experts * c.d_ff

    def routed_active(c):
        return c.top_k * c.d_ff

    def shared_total(c):
        return c.n_shared_experts * c.d_ff

    assert routed_total(MOE_FINEGRAINED) == routed_total(SMOKE_TEST)  # 16*32 == 4*128
    assert routed_active(MOE_FINEGRAINED) == routed_active(SMOKE_TEST)  # 8*32 == 2*128
    assert shared_total(MOE_FINEGRAINED) == shared_total(SMOKE_TEST)  # 4*32 == 1*128

    assert MOE_FINEGRAINED.n_routed_experts == 16
    assert MOE_FINEGRAINED.top_k == 8
