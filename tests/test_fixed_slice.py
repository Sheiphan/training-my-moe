import pytest
import torch

from qwenmoe.data.dataset import make_fixed_slice


def test_make_fixed_slice_is_deterministic_for_same_seed():
    tokens = list(range(200))
    a = make_fixed_slice(tokens, n_tokens=150, seed=0)
    b = make_fixed_slice(tokens, n_tokens=150, seed=0)
    assert torch.equal(a, b)


def test_make_fixed_slice_differs_across_seeds():
    """Requirements 11.5, 12.6: verified concretely for this fixture — seed
    0 and seed 1 land on different chunk orderings."""
    tokens = list(range(200))
    a = make_fixed_slice(tokens, n_tokens=150, seed=0)
    b = make_fixed_slice(tokens, n_tokens=150, seed=1)
    assert not torch.equal(a, b)


def test_make_fixed_slice_returns_requested_length():
    s = make_fixed_slice(list(range(200)), n_tokens=150, seed=0)
    assert s.shape == (150,)


def test_make_fixed_slice_shuffles_chunks_not_individual_tokens():
    """Only chunk order is randomized; within a chunk, ascending token order
    survives, so local linguistic structure isn't destroyed by the shuffle."""
    s = make_fixed_slice(list(range(200)), n_tokens=200, seed=3).tolist()
    assert sorted(s) == list(range(200))  # a reordering of chunks, not a loss of tokens
    consecutive_steps = sum(1 for a, b in zip(s, s[1:]) if b == a + 1)
    assert consecutive_steps >= 190  # only a handful of chunk-boundary jumps


def test_make_fixed_slice_rejects_too_many_tokens():
    with pytest.raises(ValueError):
        make_fixed_slice(list(range(10)), n_tokens=20, seed=0)


def test_make_fixed_slice_rejects_invalid_n_tokens():
    with pytest.raises(ValueError):
        make_fixed_slice(list(range(10)), n_tokens=0, seed=0)
