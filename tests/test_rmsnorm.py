import torch

from qwenmoe.model.rmsnorm import RMSNorm


def test_rmsnorm_output_shape():
    norm = RMSNorm(dim=16)
    x = torch.randn(4, 10, 16)
    out = norm(x)
    assert out.shape == x.shape


def test_rmsnorm_unit_rms_with_default_weight():
    dim = 32
    norm = RMSNorm(dim=dim)
    x = torch.randn(8, 20, dim) * 5.0 + 3.0
    out = norm(x)
    rms = out.pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-4)


def test_rmsnorm_scales_by_weight():
    dim = 8
    norm = RMSNorm(dim=dim)
    with torch.no_grad():
        norm.weight.fill_(2.0)
    x = torch.randn(2, 5, dim)
    out = norm(x)
    rms_before_weight = (out / 2.0).pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms_before_weight, torch.ones_like(rms_before_weight), atol=1e-4)
