import torch

from qwenmoe.model.positional import AlibiBias, NoPositional


def test_nopositional_is_noop():
    nope = NoPositional()
    q = torch.randn(2, 5, 8)
    k = torch.randn(2, 5, 8)
    positions = torch.arange(5, dtype=torch.float32)
    q_out, k_out = nope.apply_to_qk(q, k, positions)
    assert torch.equal(q_out, q)
    assert torch.equal(k_out, k)
    assert nope.attention_bias(5) is None


def test_alibi_bias_shape_and_monotonic_with_distance():
    n_heads = 4
    seq_len = 6
    alibi = AlibiBias(n_heads=n_heads)
    bias = alibi.attention_bias(seq_len)
    assert bias.shape == (n_heads, seq_len, seq_len)
    for h in range(n_heads):
        for i in range(seq_len):
            for j1 in range(i + 1):
                for j2 in range(j1 + 1, i + 1):
                    # j1 is further in the past than j2 (j1 < j2 <= i), so it
                    # must get an equal-or-more-negative bias.
                    assert bias[h, i, j1] <= bias[h, i, j2] + 1e-6


def test_alibi_slopes_decrease_across_heads():
    alibi = AlibiBias(n_heads=8)
    assert alibi.slopes[0] > alibi.slopes[-1]


def test_alibi_apply_is_noop_for_qk():
    alibi = AlibiBias(n_heads=4)
    q = torch.randn(2, 5, 8)
    k = torch.randn(2, 5, 8)
    positions = torch.arange(5, dtype=torch.float32)
    q_out, k_out = alibi.apply_to_qk(q, k, positions)
    assert torch.equal(q_out, q)
    assert torch.equal(k_out, k)
