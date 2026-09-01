import math

import pytest
import torch

from qwenmoe.model.positional import RotaryEmbedding


def test_rope_preserves_norm():
    torch.manual_seed(0)
    rope = RotaryEmbedding(head_dim=8, theta=10000.0)
    q = torch.randn(4, 5, 8)
    k = torch.randn(4, 5, 8)
    positions = torch.arange(5, dtype=torch.float32)
    q_rot, k_rot = rope.apply_to_qk(q, k, positions)
    assert torch.allclose(q_rot.norm(dim=-1), q.norm(dim=-1), atol=1e-4)
    assert torch.allclose(k_rot.norm(dim=-1), k.norm(dim=-1), atol=1e-4)


def test_rope_known_rotation_angle():
    # head_dim=2 -> a single frequency pair with inv_freq=1.0 regardless of
    # theta, so position p rotates (1, 0) by exactly p radians.
    rope = RotaryEmbedding(head_dim=2, theta=10000.0)
    q = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    k = torch.zeros_like(q)
    positions = torch.tensor([0.0, math.pi / 2])
    q_rot, _ = rope.apply_to_qk(q, k, positions)
    expected = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    assert torch.allclose(q_rot, expected, atol=1e-5)


def test_rope_rejects_odd_head_dim():
    with pytest.raises(ValueError):
        RotaryEmbedding(head_dim=7)


def test_rope_attention_bias_is_none():
    rope = RotaryEmbedding(head_dim=8)
    assert rope.attention_bias(10) is None
