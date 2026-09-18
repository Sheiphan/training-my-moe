"""journal 041: `StepOffsetSampler` draws window offsets per optimizer
step instead of permuting the whole dataset, and `strided_eval_offsets`
spreads validation across the whole split instead of re-reading its head.

The load-bearing test here is `test_start_step_continues_the_stream` --
that *is* the resume guarantee, asserted without running a training loop.
"""
import itertools

import pytest
import torch
from torch.utils.data import DataLoader

from qwenmoe.data.dataset import TokenDataset
from qwenmoe.data.sampler import (
    StepOffsetSampler,
    fold_seed,
    strided_eval_offsets,
)


# -- fold_seed


def test_fold_seed_is_deterministic():
    assert fold_seed(1337, 42) == fold_seed(1337, 42)


def test_fold_seed_fits_in_a_torch_generator_seed():
    """`manual_seed` takes an int64, so the sign bit isn't available."""
    for step in (0, 1, 999, 10**6, 2**31):
        seed = fold_seed(1337, step)
        assert 0 <= seed <= 0x7FFF_FFFF_FFFF_FFFF
        torch.Generator().manual_seed(seed)  # must not raise


def test_fold_seed_avalanches_between_adjacent_steps():
    """The reason this isn't `manual_seed(seed + step)`: adjacent inputs
    must not produce near-identical generator seeds. A good mixer flips
    about half the output bits for a one-bit input change."""
    for step in range(64):
        a = fold_seed(1337, step)
        b = fold_seed(1337, step + 1)
        flipped = bin(a ^ b).count("1")
        assert 16 <= flipped <= 47, f"step {step}: only {flipped}/63 bits differ"


def test_fold_seed_separates_seeds():
    assert fold_seed(1337, 7) != fold_seed(1338, 7)


# -- StepOffsetSampler


def test_offsets_for_step_is_deterministic():
    a = StepOffsetSampler(1000, batch_size=4, grad_accum_steps=2, seed=1337)
    b = StepOffsetSampler(1000, batch_size=4, grad_accum_steps=2, seed=1337)
    assert a.offsets_for_step(17) == b.offsets_for_step(17)


def test_different_seeds_give_different_offsets():
    a = StepOffsetSampler(10_000, batch_size=8, seed=1337)
    b = StepOffsetSampler(10_000, batch_size=8, seed=2024)
    assert a.offsets_for_step(0) != b.offsets_for_step(0)


def test_offsets_per_step_is_the_accumulation_window():
    sampler = StepOffsetSampler(1000, batch_size=4, grad_accum_steps=8)
    assert sampler.offsets_per_step == 32
    assert len(sampler.offsets_for_step(0)) == 32


@pytest.mark.parametrize("n_windows", [1, 2, 7, 1000])
def test_all_offsets_are_in_bounds(n_windows):
    """Out-of-bounds here would mean `TokenDataset.__getitem__` reads past
    the end of the token stream for `y`."""
    sampler = StepOffsetSampler(n_windows, batch_size=16, grad_accum_steps=4, seed=1)
    for step in range(20):
        assert all(0 <= o < n_windows for o in sampler.offsets_for_step(step))


def test_single_window_always_yields_zero():
    sampler = StepOffsetSampler(1, batch_size=3, seed=1337)
    assert sampler.offsets_for_step(0) == [0, 0, 0]


def test_start_step_continues_the_stream():
    """THE resume contract: a sampler starting at step 5 yields exactly
    what a sampler starting at step 0 would have yielded from step 5
    onward. This is what makes `--resume-from` continue the data stream
    instead of replaying it."""
    batch_size, grad_accum = 4, 2
    kwargs = dict(batch_size=batch_size, grad_accum_steps=grad_accum, seed=1337)
    consumed = 5 * batch_size * grad_accum

    fresh = StepOffsetSampler(10_000, start_step=0, **kwargs)
    resumed = StepOffsetSampler(10_000, start_step=5, **kwargs)

    expected = list(itertools.islice(iter(fresh), consumed, consumed + 40))
    assert list(itertools.islice(iter(resumed), 40)) == expected


def test_iteration_is_endless():
    """No `StopIteration` after one pass -- `_infinite_batches` relies on
    the stream never running out."""
    sampler = StepOffsetSampler(4, batch_size=2, seed=1337)
    assert len(list(itertools.islice(iter(sampler), 5000))) == 5000


def test_sampler_has_no_len():
    """An endless stream has no length; `len(loader)` must raise rather
    than report something wrong."""
    sampler = StepOffsetSampler(100, batch_size=4)
    with pytest.raises(TypeError):
        len(sampler)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(n_windows=0, batch_size=4),
        dict(n_windows=10, batch_size=0),
        dict(n_windows=10, batch_size=4, grad_accum_steps=0),
        dict(n_windows=10, batch_size=4, start_step=-1),
    ],
)
def test_sampler_rejects_invalid_arguments(kwargs):
    with pytest.raises(ValueError):
        StepOffsetSampler(**kwargs)


def test_sampler_drives_a_dataloader_over_a_real_dataset():
    tokens = torch.arange(200, dtype=torch.int32)
    dataset = TokenDataset(tokens, seq_len=8)
    sampler = StepOffsetSampler(
        len(dataset), batch_size=4, grad_accum_steps=2, seed=1337
    )
    loader = DataLoader(dataset, batch_size=4, sampler=sampler, drop_last=True)

    batches = list(itertools.islice(iter(loader), 6))
    assert len(batches) == 6
    for x, y in batches:
        assert x.shape == (4, 8) and y.shape == (4, 8)
        assert x.dtype == torch.long
        # y is x shifted one position -- the next-token target
        assert torch.equal(y[:, :-1], x[:, 1:])


def test_dataloader_batches_match_the_sampler_offsets():
    """Pins that the loader consumes the stream in order, so `start_step`
    maps onto `global_step` rather than onto some shuffled reordering."""
    tokens = torch.arange(500, dtype=torch.int32)
    dataset = TokenDataset(tokens, seq_len=4)
    sampler = StepOffsetSampler(len(dataset), batch_size=3, seed=99)
    loader = DataLoader(dataset, batch_size=3, sampler=sampler, drop_last=True)

    x, _ = next(iter(loader))
    expected_starts = sampler.offsets_for_step(0)
    assert x[:, 0].tolist() == expected_starts


# -- strided_eval_offsets


def test_strided_eval_offsets_spans_the_whole_split():
    offsets = strided_eval_offsets(100_000, 10)
    assert offsets[0] == 0
    assert offsets == sorted(offsets)
    assert len(set(offsets)) == 10
    # the last offset lands in the final tenth, not in the first 1%
    assert offsets[-1] >= 90_000


def test_strided_eval_offsets_is_deterministic():
    assert strided_eval_offsets(1234, 7) == strided_eval_offsets(1234, 7)


def test_strided_eval_offsets_are_in_bounds():
    for n_windows in (1, 3, 50, 999_999):
        for n_eval in (1, 4, 50):
            offsets = strided_eval_offsets(n_windows, n_eval)
            assert all(0 <= o < n_windows for o in offsets)


def test_strided_eval_offsets_clamps_to_available_windows():
    """Asking for more eval windows than exist must not produce
    duplicates or out-of-range offsets."""
    offsets = strided_eval_offsets(5, 50)
    assert offsets == [0, 1, 2, 3, 4]


def test_strided_eval_offsets_covers_far_more_than_a_prefix_scan():
    """The regression this function exists to fix: the old
    `shuffle=False` + `islice` loader read offsets 0..199 only."""
    n_windows, n_eval = 90_000_000, 200
    offsets = strided_eval_offsets(n_windows, n_eval)
    assert max(offsets) > 89_000_000
    assert len(set(offsets)) == n_eval


@pytest.mark.parametrize(
    "n_windows,n_eval", [(0, 5), (10, 0)]
)
def test_strided_eval_offsets_rejects_invalid_arguments(n_windows, n_eval):
    with pytest.raises(ValueError):
        strided_eval_offsets(n_windows, n_eval)
