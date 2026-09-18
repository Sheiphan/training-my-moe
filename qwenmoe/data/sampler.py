"""Window samplers that never materialize a permutation (journal 041).

`TokenDataset` is stride-1: every token position starts a window, so a
999M-token corpus is ~899M training windows. `DataLoader(shuffle=True)`
builds a `RandomSampler`, whose no-replacement path is

    yield from torch.randperm(n, generator=g).tolist()

-- an 8GB `int64` tensor *and* a fully materialized Python list of ~899M
boxed ints (~32GB), before the first training step. Journal 040 measured
~8 minutes of stall and attributed ~8GB to it; the `.tolist()` term means
that entry understated the cost.

`StepOffsetSampler` replaces it with random offsets drawn per optimizer
step, which is O(batch) memory. `strided_eval_offsets` replaces the
validation loader's `shuffle=False` prefix scan, which was evaluating the
same ~700 tokens over and over.
"""
from __future__ import annotations

import itertools
from collections.abc import Iterator

import torch
from torch.utils.data import Sampler

# splitmix64 (Steele et al., "Fast Splittable Pseudorandom Number
# Generators", OOPSLA 2014) -- the finalizer used to derive a
# well-distributed seed from a counter.
_MASK64 = 0xFFFF_FFFF_FFFF_FFFF
_GOLDEN_GAMMA = 0x9E37_79B9_7F4A_7C15
_MIX_A = 0xBF58_476D_1CE4_E5B9
_MIX_B = 0x94D0_49BB_1331_11EB


def fold_seed(seed: int, step: int) -> int:
    """Mixes `(seed, step)` into a single generator seed.

    `manual_seed(seed + step)` would be the obvious thing and is a trap:
    consecutive seeds in most PRNGs produce correlated first outputs, so
    adjacent training steps would draw overlapping-looking offsets. The
    splitmix64 finalizer avalanches every input bit across the output,
    which is the same `fold_in` idea JAX uses to derive child keys from a
    counter.

    Returned masked to 63 bits: `torch.Generator.manual_seed` takes an
    `int64`, so the sign bit is not available.
    """
    z = (seed + step * _GOLDEN_GAMMA) & _MASK64
    z = ((z ^ (z >> 30)) * _MIX_A) & _MASK64
    z = ((z ^ (z >> 27)) * _MIX_B) & _MASK64
    z = z ^ (z >> 31)
    return z & 0x7FFF_FFFF_FFFF_FFFF


class StepOffsetSampler(Sampler[int]):
    """An endless stream of random window offsets, derived from
    `(seed, optimizer_step)` rather than from a shuffled permutation.

    Two properties matter, and they are why this exists instead of
    `RandomSampler(replacement=True)` (which would also fix the memory
    blowup, since its with-replacement path already draws `torch.randint`
    in blocks of 32):

    1. **O(batch) memory.** Only one optimizer step's worth of offsets is
       ever alive.
    2. **Resume is a pure function.** The offsets for step `N` depend on
       nothing but `(seed, N)`, so continuing an interrupted run is
       `start_step=global_step` -- no generator state to pickle into a
       checkpoint, and testable in milliseconds without training anything.

    **Sampling is with replacement**, and at this scale that is not an
    approximation worth worrying about. A 2.7B-token budget at `seq_len`
    512 consumes ~5.3M of ~899M windows -- 0.59% -- for an expected
    duplicate rate near 0.3%. More to the point, the guarantee
    `randperm` buys is close to meaningless at stride 1: it prevents
    drawing the *identical* window twice while freely serving windows
    offset by one that share 511 of their 512 tokens.

    The sampler is infinite and deliberately implements no `__len__`, so
    `len(loader)` raises rather than silently reporting something wrong.
    `Trainer` is driven by `total_steps`, not by epochs, and nothing in
    the codebase asks a training loader for its length.
    """

    def __init__(
        self,
        n_windows: int,
        batch_size: int,
        grad_accum_steps: int = 1,
        seed: int = 0,
        start_step: int = 0,
    ) -> None:
        if n_windows < 1:
            raise ValueError(f"n_windows must be >= 1, got {n_windows}")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if grad_accum_steps < 1:
            raise ValueError(f"grad_accum_steps must be >= 1, got {grad_accum_steps}")
        if start_step < 0:
            raise ValueError(f"start_step must be >= 0, got {start_step}")
        self.n_windows = n_windows
        self.batch_size = batch_size
        self.grad_accum_steps = grad_accum_steps
        self.seed = seed
        self.start_step = start_step

    @property
    def offsets_per_step(self) -> int:
        """Windows consumed by one *optimizer* step -- the accumulation
        window, not one micro-batch. Chunking the stream this way is what
        makes `start_step` line up with `global_step` exactly."""
        return self.batch_size * self.grad_accum_steps

    def offsets_for_step(self, step: int) -> list[int]:
        """The offsets step `step` will see. Exposed so a test (and a
        future data-provenance check) can assert the resume contract
        without running a training loop."""
        generator = torch.Generator().manual_seed(fold_seed(self.seed, step))
        return torch.randint(
            0, self.n_windows, (self.offsets_per_step,), generator=generator
        ).tolist()

    def __iter__(self) -> Iterator[int]:
        for step in itertools.count(self.start_step):
            yield from self.offsets_for_step(step)


def strided_eval_offsets(n_windows: int, n_eval_windows: int) -> list[int]:
    """Evenly-spaced window offsets spanning the whole split.

    The validation loader used to be `shuffle=False` + `islice(..., 50)`,
    which reads dataset indices `0 .. 50*batch-1`. At `batch_size` 4 and
    `seq_len` 512 that is window offsets 0-199, covering validation token
    positions `[0, 711]` -- 102,400 predictions computed, but 99.3% of
    them re-predicting the same ~700 tokens. The validation curve for a
    billion-token run was measuring one paragraph.

    Striding instead covers `n_eval_windows * seq_len` distinct positions
    across the entire split for *identical* compute. Evenly spaced rather
    than randomly sampled so the slice is reproducible across steps,
    across runs and across checkpoints without carrying a generator.
    """
    if n_windows < 1:
        raise ValueError(f"n_windows must be >= 1, got {n_windows}")
    if n_eval_windows < 1:
        raise ValueError(f"n_eval_windows must be >= 1, got {n_eval_windows}")
    n_eval_windows = min(n_eval_windows, n_windows)
    return [i * n_windows // n_eval_windows for i in range(n_eval_windows)]
