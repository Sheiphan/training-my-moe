# 041 — A run that survives the night: samplers, checkpoints, and the memory that wasn't being measured

**Type:** tooling (direct follow-up to 040) · **Date:** 2026-09-18

## TL;DR

- **The blocker 040 named:** `DataLoader(shuffle=True)` over a stride-1
  999M-token dataset. Entry 040 costed it at ~8GB of `int64` and ~13s.
  That was an **undercount**: PyTorch's `RandomSampler` no-replacement
  path is `yield from torch.randperm(n).tolist()`, and the `.tolist()`
  builds a Python list of ~899M boxed ints on top of the tensor.
  Measured at three sizes, the pair costs a linear **48.2 bytes per
  element**, so at the real window count it is **~43GB** — on a 24GB
  machine.
- **The fix:** `StepOffsetSampler` draws offsets from `(seed, step)`
  instead of permuting anything. At the real 899,396,383 windows it
  produces a step's offsets in **0.197ms for +1MB**, and because the
  offsets are a pure function of the step, `start_step=global_step` is
  the entire resume fix — no generator state in the checkpoint.
- **The thing nobody was measuring:** chasing a hang at `seq_len` 512 turned
  up something bigger. `torch.mps.current_allocated_memory()` — the number
  `check_capacity.py` reports and journal 037 trusted — stayed flat at ~8GB
  while `torch.mps.driver_allocated_memory()` climbed **9.5 → 20.5GB over
  four steps**, and step time went **2.8s → 17.9s → 177s**. That is journal
  037's "hang" with a name. `torch.mps.empty_cache()` per step pins the
  driver figure flat for no measurable step cost.
- **Also landed:** atomic checkpoints with tiered retention (a 164k-step
  run would otherwise write **341GB**), a `latest.pt` symlink, SIGINT that
  finishes the step and checkpoints instead of truncating a 4GB file, a
  resume guard, `mmap` loading (**4.11GB → 0.20GB**, 0.014s), the 12 CLI
  flags that existed only as dataclass defaults, and a validation slice
  that covers more than one paragraph.
- **57 tests added, none changed** (492 passing, was 435).

## The sampler

### What the old path actually cost

Entry 040 inferred the `randperm` cost from a 50M-element probe and
reported the tensor. It missed that `RandomSampler.__iter__` doesn't yield
from the tensor — it yields from `.tolist()`:

```python
for _ in range(self.num_samples // n):
    yield from torch.randperm(n, generator=generator).tolist()
```

`yield from` on a list materializes the whole list first. Measured:

| n | `randperm` | `+ .tolist()` peak | bytes/element |
| ---: | ---: | ---: | ---: |
| 1,000,000 | 0.21GB | 0.25GB | 50.0 |
| 10,000,000 | 0.28GB | 0.68GB | 48.0 |
| 50,000,000 | 0.60GB | 2.61GB | 48.2 |

Linear at 48.2 bytes/element — 8 for the `int64` tensor, ~8 for the list
pointer, ~28 for each boxed `PyObject` integer (CPython's small-int cache
covers only -5..256). Extrapolated to the real train split of
**899,396,383** windows: **~43GB**. The run in journal 040 completed only
because macOS swapped, which is where its "~8 minutes before the first
step" came from.

Extrapolated, not measured at full size — running it would have swapped the
machine hard for no extra information. Three linear points and a per-element
constant that is exactly what the object layout predicts is enough.

### The replacement

```python
def __iter__(self):
    for step in itertools.count(self.start_step):
        g = torch.Generator().manual_seed(fold_seed(self.seed, step))
        yield from torch.randint(0, self.n_windows, (per_step,), generator=g).tolist()
```

Measured at the real window count: **0.197ms** for a step's 32 offsets,
**+0.001GB** peak. The corpus-sized stall is simply gone.

`fold_seed` is splitmix64's finalizer (Steele et al., OOPSLA 2014). The
obvious `manual_seed(seed + step)` is a trap — consecutive seeds in most
PRNGs produce correlated first outputs, so adjacent steps would draw
offsets that look alike. `test_fold_seed_avalanches_between_adjacent_steps`
pins that 16-47 of 63 output bits flip for a one-step input change.

### Why sampling with replacement is not a compromise

`RandomSampler(replacement=True)` would also have fixed the memory — its
with-replacement branch already draws `torch.randint` in blocks of 32. The
reason to write 30 lines instead is resume (below), not sampling quality.

At a 2.7B-token budget with `seq_len` 512 we consume **5.3M of 899M
windows — 0.59%** — for an expected duplicate-draw rate near 0.3%. But the
stronger argument is that the guarantee `randperm` buys is close to
meaningless here: **at stride 1 it prevents drawing the identical window
twice while freely serving windows offset by one that share 511 of their
512 tokens.** It was paying ~43GB to avoid exact duplicates in a set where
near-duplicates are the norm.

Non-overlapping stride-`seq_len` windows would have made `randperm` cheap
(1.76M windows, 14MB) but is strictly worse at this budget: 2.7B tokens is
~2.8 passes over a fixed 1.76M-window set, so the model would see the same
windows with the *same* boundaries three times. Random offsets re-phase
every window on every pass, for free.

## Resume that continues instead of replaying

Because offsets depend only on `(seed, step)`, resume is `start_step=global_step`.
No RNG blob, no data-position counter, no version fragility. The contract is
asserted directly, without training anything:

```python
expected = list(islice(iter(fresh), consumed, consumed + 40))
assert list(islice(iter(resumed), 40)) == expected
```

**There is no RNG state to save**, and that is a property of this
architecture rather than an omission: there is no dropout anywhere in
`qwenmoe/model/`, routing is `topk` on logits with no noise, and
`LossFreeBalancer.bias` is a registered buffer already inside
`state_dict()`. Worth writing down so the next person doesn't go looking.

### The guard, and what it deliberately does *not* guard

The checkpoint now carries a `data_fingerprint` — `seq_len`, `batch_size`,
`grad_accum_steps`, `seed`, `n_train_windows` — and a mismatched resume is
refused. These are exactly the inputs to the sampler's stream and to window
indexing; change one and the resumed run draws different data at every step
while the LR schedule shifts under a checkpoint already partway along it,
with no error and no symptom beyond a worse model.

`total_tokens` is deliberately **not** in the fingerprint. The test suite
caught this: three existing tests resume with a larger `total_tokens`,
because that is how you extend a run — and it is also how you size a WSD
cooldown. Refusing it would have blocked the exact workflow this entry
exists to enable. It gets a warning instead, naming where the resumed step
now sits on the re-sized schedule.

## The memory nobody was measuring

This one started as a hang and ended as a correction to journal 037.

Pointing the new pipeline at `seq_len` 512 produced a run that logged
`run_start`, trained two steps, and then crawled. The data path was
exonerated immediately — a direct probe at the real corpus size reached the
first batch in **0.32s at 0.23GB**, and 32 `__getitem__` calls took 0.008s.
An isolated model probe was also clean: 2.55s forward+backward at
`batch 4 / seq 512`, with `current_allocated_memory` steady at 4.15GB.

The difference showed up only when both numbers were logged together:

| step | `current_allocated` | `driver_allocated` | step time |
| ---: | ---: | ---: | ---: |
| 0 | 8.09GB | 9.52GB | 2.79s |
| 1 | 8.11GB | 11.90GB | 2.70s |
| 2 | 8.11GB | 13.22GB | 2.79s |
| 3 | 8.22GB | 14.57GB | 3.13s |
| …at accum 4 | 9.70GB | 19.44GB | **166.45s** |

**`current_allocated_memory` is flat. `driver_allocated_memory` grows
without bound.** The first is what `check_capacity.py` reports and what
journal 037 used to conclude that `SCALE_300M` at batch 16 "measured a
clean 7.26GB" — while recording, one paragraph later, that this exact
configuration *hung the real pipeline*. That contradiction now has an
explanation: the number being watched was never the number that decides
whether the machine swaps.

`torch.mps.empty_cache()` after each optimizer step pins the driver figure
(12.42GB, flat across six steps) at no measurable cost to step time.

### What this does not explain

The throughput I measured today — 524-699 tok/s at `seq 256` — is **2-4x
worse** than the 1,018-2,113 tok/s in `runs/scale_300m`'s own logs at the
same settings. Before treating that as a regression I checked the machine:

```
vm.swapusage: total = 20480.00M  used = 19559.69M  free = 920.31M
Pageins: 77496120
```

**Swap is 95% full.** Every number I took today is contaminated by that,
so none of them belongs in a throughput table, and the step path is
untouched by this entry anyway (`git diff` confirms `_run_step`,
`_forward_backward_core` and `_forward_backward_step` are byte-identical).
The honest statement is: `empty_cache` fixes unbounded driver growth, which
is real and reproducible; today's absolute tok/s figures are not
trustworthy and the calibration has to be redone on an unloaded machine.

This is the same lesson as journal 037's, arrived at from the other side:
*"The safe ceiling isn't a fixed number, even on the same machine in the
same session."*

## Checkpoints that can't fill the disk

Checkpoint size is exactly `12 bytes x total_params` — fp32 weights plus
AdamW's two moments — validated against a real file: `step_100.pt` is
4,013,033,885 bytes against `12 x 334,382,336 = 4,012,588,032` predicted,
the 445KB difference being pickle framing.

At vocab 16,385 that is 4.16GB each. A 164k-step run at
`--checkpoint-interval-steps 2000` writes 82 of them: **341GB**, against
48GB free. The run would have died on a failed `torch.save` around hour 18,
leaving a truncated 4GB file at exactly the path `--resume-from` reaches for.

Three changes:

- **Tiered retention**, because the two kinds of checkpoint exist for
  different reasons. Rolling ones exist to *resume* and are worthless once
  a newer one lands; milestones exist to *evaluate and generate from*
  across the run's history, and only the first kind needs optimizer state.
  `--keep-last 2` full (8.3GB) plus `--keep-every 20000` weights-only
  (1.39GB each) is ~19GB for a whole multi-day run.
  `scripts/generate.py` already reads only `model_state_dict` /
  `model_config` / `tokenizer_meta`, so milestones work with it unchanged.
- **Atomic writes** — `.tmp` then `os.replace` — plus a `shutil.disk_usage`
  pre-check that *skips* a checkpoint when free space is under 2x the
  estimate. A skipped checkpoint costs one interval; a half-written one
  costs the run.
- **`latest.pt`** as a relative symlink replaced via `os.replace`, so
  `--resume-from runs/x/checkpoints/latest.pt` works with no change to
  `load_checkpoint` or `generate.py`, and `runs/` stays relocatable.

### Signals

SIGINT and SIGTERM set a flag the loop checks *after* its next checkpoint,
reusing the existing save path. Saving from inside the handler is the
obvious thing and is how you get a corrupt 4GB file: `torch.save` would run
with MPS work in flight. A second signal restores `SIG_DFL` and re-raises.

Handlers are restored in a `finally` — `Trainer.fit` is called about twenty
times by the test suite and from `notebooks/05_train.ipynb`, and a leaked
handler would outlive the run that installed it. `test_signal_handlers_are_restored_after_fit`
pins that.

## Two smaller things with outsized effect

**`mmap=True`.** The token stream is read-only and randomly indexed, so it
belongs in evictable page cache rather than 4GB of dirty anonymous memory.
Measured on the real 999M-token file: **0.014s and 0.20GB RSS**, against
4.11GB resident before. `train_val_split(mode="tail")` returns views so the
mapping survives the split — one more reason `mode="documents"` is the
wrong default here, since its `torch.cat` would both defeat the mapping and
(a cost journal 040 undercounted by looking only at the val side) allocate
~3.6GB for the train half.

**The validation slice was measuring one paragraph.** `_evaluate` ran
`islice(val_loader, 50)` over a `shuffle=False` loader, which reads dataset
indices 0..199 — token positions `[0, 711]`. It computed 102,400
predictions of which **99.3% re-predicted the same ~700 tokens**.
`strided_eval_offsets` spreads the same 50 batches across the whole split:
102,400 *distinct* positions, **144x the coverage for identical compute**.
Evenly strided rather than random so the slice is reproducible across
steps, runs and checkpoints without carrying a generator.

## The flags that existed but weren't reachable

`scripts/train.py` built `TrainConfig` by hand and passed **12 of its 24
fields**. `precision`, `lr_schedule`, `weight_decay`, `z_loss_weight`,
`grad_clip_norm`, `optimizer` and the rest silently took dataclass
defaults. `--precision bf16` would have been accepted, ignored, and the run
would have trained in fp32 without saying so.

It is now a field-name match over `dataclasses.fields(TrainConfig)`, and
`test_every_train_config_field_has_a_flag_or_is_deliberately_omitted`
fails the moment a field becomes unreachable again — the bug was quiet, so
the test has to be loud.

`--lr-schedule wsd` is the one that matters most for what comes next: it
makes a checkpoint anywhere in the plateau a valid launch point for a short
cooldown, which is what lets a multi-day run be stopped when you choose and
still end as a finished model rather than an abandoned one.

## Tests

**57 added, no pre-existing test changed** (492 passing, was 435):

- **`tests/test_sampler.py`** (28): determinism, the avalanche property,
  in-bounds offsets at `n_windows` 1/2/7/1000, the `start_step` resume
  contract, endlessness, `len()` raising, DataLoader integration, and
  `strided_eval_offsets` spanning the split rather than its head.
- **`tests/test_checkpoint_hygiene.py`** (26): atomic write and no
  `.tmp` residue after a failed save, weights-only round-trip and its
  refusal on resume, numeric (not lexicographic) pruning order,
  protected paths, the `latest.pt` symlink, the fingerprint's inclusions
  *and* its deliberate exclusion of `total_tokens`, SIGINT leaving a
  resumable checkpoint, and handler restoration.
- **`tests/test_train_script.py`** (3): every `TrainConfig` field is
  reachable from the CLI, non-default flags survive the round trip, and
  `--betas` becomes a tuple.

## Fixed along the way

`docs/math/verify_008_data_pipeline.py` had been broken since journal 040
renamed `_as_long_tensor` to `_as_token_tensor` — it `ImportError`ed on
line 59. Worse, it asserted the *old* semantics (`from_tensor.dtype ==
torch.long`), which 040 deliberately inverted. Both the script and
`docs/math/008-data-pipeline.md` now describe dtype preservation and say
why, and the script runs green again. That was my regression, introduced by
a rename that the math series' own verify step would have caught if it had
been run.

## Questions that might come to mind

**Doesn't sampling with replacement mean the model never sees some data?**
In expectation it sees 1 - e^(-0.0059) ≈ 0.59% of windows at this budget,
which is the same 0.59% a permutation would reach — a permutation only
guarantees *coverage* once you consume the whole thing, and we consume
0.59% of it. At budgets past one full epoch the two converge anyway.

**Why not `num_workers > 0` to overlap loading with compute?** Because
loading is not the bottleneck: 32 windows take 0.008s against a
multi-second step. Workers also prefetch ahead of the loop, which would
break the `start_step` alignment that makes resume exact.

**Is `empty_cache()` per step wasteful?** Measured, no: at `accum 1 /
seq 512` steps ran 2.70-3.13s without it and 2.42-3.63s with it, inside
the noise, while the driver figure went from unbounded growth to flat.
Reclaiming a cache you are about to refill costs a bookkeeping pass, not a
reallocation.

**Why keep `_infinite_batches` when the sampler is already endless?** The
contract the training loop depends on is "never raises `StopIteration`".
That shouldn't silently become the sampler's private responsibility just
because today's sampler happens to satisfy it.

**Does any of this change what the model computes?** No.
`_run_step`, `_forward_backward_core` and `_forward_backward_step` are
byte-identical, and the 435 pre-existing tests pass unchanged. What changed
is which windows get fed in, how memory is released, what gets written to
disk and what gets logged.
