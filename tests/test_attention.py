import pytest
import torch

from qwenmoe.config import ModelConfig
from qwenmoe.model.attention import Attention


def _tiny_config(**overrides):
    defaults = dict(vocab_size=50, d_model=16, n_heads=4, n_kv_heads=4, max_seq_len=8)
    defaults.update(overrides)
    return ModelConfig(**defaults)


def test_attention_output_shape_and_finite_mha():
    """n_kv_heads == n_heads is the standard multi-head-attention case."""
    config = _tiny_config(n_heads=4, n_kv_heads=4)
    attn = Attention(config)
    x = torch.randn(2, 5, config.d_model)
    out = attn(x, torch.arange(5))
    assert out.shape == (2, 5, config.d_model)
    assert torch.isfinite(out).all()


def test_attention_output_shape_and_finite_gqa():
    """n_kv_heads < n_heads: 2 kv heads shared across 4 query heads."""
    config = _tiny_config(n_heads=4, n_kv_heads=2)
    attn = Attention(config)
    x = torch.randn(3, 6, config.d_model)
    out = attn(x, torch.arange(6))
    assert out.shape == (3, 6, config.d_model)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("scheme", ["rope", "nope", "alibi"])
def test_attention_works_with_each_positional_scheme(scheme):
    config = _tiny_config(n_heads=4, n_kv_heads=2, positional_scheme=scheme)
    attn = Attention(config)
    x = torch.randn(2, 5, config.d_model)
    out = attn(x, torch.arange(5))
    assert out.shape == (2, 5, config.d_model)
    assert torch.isfinite(out).all()


def test_causal_mask_blocks_future_influence():
    """Perturbing only the last token must not change any earlier
    position's output (Requirement 1.6, 9.3)."""
    torch.manual_seed(0)
    config = _tiny_config(n_heads=2, n_kv_heads=2, positional_scheme="nope")
    attn = Attention(config)
    x = torch.randn(1, 5, config.d_model)
    positions = torch.arange(5)

    with torch.no_grad():
        out_original = attn(x, positions)

    x_perturbed = x.clone()
    x_perturbed[:, -1, :] += 10.0
    with torch.no_grad():
        out_perturbed = attn(x_perturbed, positions)

    assert torch.allclose(out_original[:, :-1, :], out_perturbed[:, :-1, :], atol=1e-6)
    assert not torch.allclose(out_original[:, -1, :], out_perturbed[:, -1, :], atol=1e-6)
