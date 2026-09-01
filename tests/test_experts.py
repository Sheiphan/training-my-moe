import pytest
import torch

from qwenmoe.config import ModelConfig
from qwenmoe.model.moe import GELUExpert, GEGLUExpert, SwiGLUExpert, build_expert


def test_swiglu_expert_output_shape():
    expert = SwiGLUExpert(d_model=16, d_ff=32)
    x = torch.randn(4, 16)
    out = expert(x)
    assert out.shape == (4, 16)
    assert torch.isfinite(out).all()


def test_geglu_expert_output_shape():
    expert = GEGLUExpert(d_model=16, d_ff=32)
    x = torch.randn(4, 16)
    out = expert(x)
    assert out.shape == (4, 16)
    assert torch.isfinite(out).all()


def test_gelu_expert_output_shape():
    expert = GELUExpert(d_model=16, d_ff=32)
    x = torch.randn(4, 16)
    out = expert(x)
    assert out.shape == (4, 16)
    assert torch.isfinite(out).all()


def test_expert_output_shape_with_batch_and_seq_dims():
    """Experts operate token-independently, so a (batch, seq, d_model)
    input works the same as a flattened one."""
    expert = SwiGLUExpert(d_model=8, d_ff=16)
    x = torch.randn(2, 5, 8)
    out = expert(x)
    assert out.shape == (2, 5, 8)


def test_gelu_expert_has_no_gating_fewer_params_than_swiglu():
    """Requirement 12.2: GELU is explicitly 'non-gated', so it has 2
    weight matrices (up, down) instead of the 3 (gate, up, down) the gated
    SwiGLU/GEGLU variants use."""
    d_model, d_ff = 16, 32
    gelu_params = sum(p.numel() for p in GELUExpert(d_model, d_ff).parameters())
    swiglu_params = sum(p.numel() for p in SwiGLUExpert(d_model, d_ff).parameters())
    geglu_params = sum(p.numel() for p in GEGLUExpert(d_model, d_ff).parameters())
    assert gelu_params == 2 * d_model * d_ff
    assert swiglu_params == 3 * d_model * d_ff
    assert geglu_params == 3 * d_model * d_ff


@pytest.mark.parametrize(
    "activation,expected_cls",
    [("swiglu", SwiGLUExpert), ("geglu", GEGLUExpert), ("gelu", GELUExpert)],
)
def test_build_expert_selects_variant(activation, expected_cls):
    config = ModelConfig(vocab_size=50, d_model=16, d_ff=32, activation=activation)
    expert = build_expert(config)
    assert isinstance(expert, expected_cls)


def test_build_expert_unknown_activation_raises():
    config = ModelConfig(vocab_size=50)
    config.activation = "unknown"  # bypass __post_init__, which doesn't validate this field
    with pytest.raises(ValueError):
        build_expert(config)
