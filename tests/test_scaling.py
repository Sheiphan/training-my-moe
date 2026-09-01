import logging
import math

import pytest
import torch

from qwenmoe.config import ModelConfig
from qwenmoe.model.transformer import QwenMoEModel
from qwenmoe.train.scaling import (
    calibrate_throughput,
    chinchilla_budget,
    count_params,
    epochs_for_budget,
)


def _tiny_config(**overrides):
    defaults = dict(
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
    defaults.update(overrides)
    return ModelConfig(**defaults)


def test_count_params_matches_hand_computed_value_for_tiny_config():
    """Hand-computed for _tiny_config() (d_model=16, n_heads=4,
    n_kv_heads=2, d_ff=32, n_layers=2, n_routed_experts=4,
    n_shared_experts=1, top_k=2, vocab_size=50, tied embeddings):

    embedding table = 50*16 = 800; final_norm = 16 -> embedding = 816
    per-layer attention = attn_norm(16) + w_q(16*16) + w_k(16*8) + w_v(16*8)
      + w_o(16*16) = 16+256+128+128+256 = 784; x2 layers = 1568
    per-expert SwiGLU params = 16*32 (gate) + 16*32 (up) + 32*16 (down) = 1536
    per-layer moe_total = moe_norm(16) + router(16*4=64) + shared(1*1536)
      + routed(4*1536=6144) = 7760; x2 layers = 15520
    per-layer moe_active = 16 + 64 + 1536 + top_k(2)*1536 = 4688; x2 = 9376
    total = 816 + 1568 + 15520 = 17904
    active = 816 + 1568 + 9376 = 11760
    """
    config = _tiny_config()
    model = QwenMoEModel(config)

    breakdown = count_params(model)

    assert breakdown.embedding == 816
    assert breakdown.attention_total == 1568
    assert breakdown.moe_total == 15520
    assert breakdown.moe_active == 9376
    assert breakdown.total == 17904
    assert breakdown.active == 11760


def test_count_params_untied_embeddings_add_separate_output_head():
    config = _tiny_config(tie_word_embeddings=False)
    model = QwenMoEModel(config)

    breakdown = count_params(model)

    assert breakdown.embedding == 816 + 50 * 16


def test_count_params_total_matches_real_model_parameter_count():
    """Cross-checks the breakdown's total against summing every unique
    parameter tensor directly (deduplicating the tied embedding/lm_head
    tensor by identity), independent of count_params' own bucketing logic.
    """
    config = _tiny_config()
    model = QwenMoEModel(config)

    breakdown = count_params(model)

    seen_ids = set()
    real_total = 0
    for p in model.parameters():
        if id(p) not in seen_ids:
            seen_ids.add(id(p))
            real_total += p.numel()

    assert breakdown.total == real_total


def test_chinchilla_budget_arithmetic():
    assert chinchilla_budget(9_300_000) == 186_000_000
    assert chinchilla_budget(1) == 20


def test_chinchilla_budget_rejects_non_positive():
    with pytest.raises(ValueError):
        chinchilla_budget(0)


def test_epochs_for_budget_arithmetic():
    assert epochs_for_budget(total_tokens=200, unique_tokens=100) == pytest.approx(2.0)
    assert epochs_for_budget(total_tokens=50, unique_tokens=100) == pytest.approx(0.5)


def test_epochs_for_budget_warns_past_4(caplog):
    with caplog.at_level(logging.WARNING):
        result = epochs_for_budget(total_tokens=500, unique_tokens=100)
    assert result == pytest.approx(5.0)
    assert "epoch" in caplog.text.lower()


def test_epochs_for_budget_does_not_warn_at_or_below_4(caplog):
    with caplog.at_level(logging.WARNING):
        result = epochs_for_budget(total_tokens=400, unique_tokens=100)
    assert result == pytest.approx(4.0)
    assert caplog.text == ""


@pytest.mark.parametrize("overrides", [{"total_tokens": 0}, {"unique_tokens": 0}])
def test_epochs_for_budget_rejects_invalid_args(overrides):
    kwargs = {"total_tokens": 100, "unique_tokens": 100, **overrides}
    with pytest.raises(ValueError):
        epochs_for_budget(**kwargs)


def test_calibrate_throughput_returns_positive_finite_tokens_per_sec():
    config = _tiny_config()
    model = QwenMoEModel(config)
    x = torch.randint(0, config.vocab_size, (2, 8))
    y = torch.randint(0, config.vocab_size, (2, 8))

    tokens_per_sec = calibrate_throughput(model, (x, y), device="cpu", n_steps=2)

    assert tokens_per_sec > 0
    assert math.isfinite(tokens_per_sec)


def test_calibrate_throughput_rejects_invalid_n_steps():
    config = _tiny_config()
    model = QwenMoEModel(config)
    x = torch.randint(0, config.vocab_size, (2, 8))
    y = torch.randint(0, config.vocab_size, (2, 8))

    with pytest.raises(ValueError):
        calibrate_throughput(model, (x, y), device="cpu", n_steps=0)
