# Extra — Manim preview of Task 8's math (sliding windows, contiguous split, chunk-shuffle)

**Type:** task (supplementary, not a numbered task) · **Date:** 2026-08-31 ·
**Related:** Requirement 7 (visualizations), task 8's data pipeline; see
[008-data-pipeline.md](008-data-pipeline.md) for the full concept
explainers/derivations — this entry only covers the new visualization
work, not the math itself again.

## Why this isn't numbered `009`
Same reasoning as
[extra-manim-preview-task3.md](extra-manim-preview-task3.md),
[extra-manim-preview-task5.md](extra-manim-preview-task5.md),
[extra-manim-preview-task6.md](extra-manim-preview-task6.md), and
[extra-manim-preview-task7.md](extra-manim-preview-task7.md): numbered
entries track task numbers 1:1, and task 9 (scaling-law utilities) hasn't
happened yet. This is a **standalone preview** of task 8's math, requested
directly, ahead of the formal Task 13 visualization pipeline (which
consumes a `TraceBundle` captured from a real trained checkpoint via
`forward_with_trace()` — that needs tasks 9-12 first). It may inform a
future formal scene later, but isn't claiming to satisfy one now.

## What was built
`qwenmoe/viz/scenes_task8_preview.py::Task8DataPipelineScene` — a single
Manim scene with three parts, each animating **real numbers produced by
calling the actual, already-tested functions/classes**
(`TokenDataset`, `train_val_split`, `make_fixed_slice` from
`qwenmoe.data.dataset`) — cross-checked in a plain Python shell before
animating, not fabricated illustration numbers:

1. **`TokenDataset`'s sliding windows.** A real `TokenDataset(tokens=0..19,
   seq_len=5)`. A blue bracket (the `x` window) and a yellow bracket (the
   `y` window, shifted +1) slide together across the token row from `i=0`
   to `i=1` to `i=3`, each step reading out the real `ds[i]` values
   (`i=0: x=(0,1,2,3,4) y=(1,2,3,4,5)` → `i=1: x=(1,2,3,4,5)
   y=(2,3,4,5,6)` → `i=3: x=(3,4,5,6,7) y=(4,5,6,7,8)`), matching
   [008-data-pipeline.md](008-data-pipeline.md)'s worked example exactly.
2. **`train_val_split`'s contiguous split.** A real `train_val_split
   (tokens=0..19, val_fraction=0.2)` (16 train / 4 val), colored blue/
   orange either side of a boundary line at index 16. Then — the point of
   this design decision — two real, **separately constructed**
   `TokenDataset`s, one per split (`TokenDataset(train, seq_len=3)`:
   `len=13`, window 0 `x=(0,1,2)`; `TokenDataset(val, seq_len=3)`:
   `len=1`, window 0 `x=(16,17,18)`), demonstrating concretely that no
   window is ever built from tokens spanning both splits.
3. **`make_fixed_slice`'s chunk-shuffle.** A real `make_fixed_slice
   (tokens=0..19, n_tokens=20, seed=1)`. A "before" row colors each token
   by its original chunk (4 chunks of 5); an "after" row shows the same
   colors in their real post-shuffle order, with the chunk order itself
   (`[1, 3, 2, 0]`) inferred directly from the real output (see design
   note below). A second beat calls the real function twice more —
   `seed=1` called twice (`identical: True`) vs. `seed=1` vs. `seed=0`
   (`different: True`) — then closes with a schematic of one identical
   slice fanning out to three hypothetical `positional_scheme` variants,
   captioned with Requirement 12.6's point: any resulting loss difference
   can now only be explained by the one setting that actually changed.

`download_corpus` isn't part of this preview — it's a real network call
with no interesting math to animate; see 008-data-pipeline.md for its
manual, real-download verification instead.

Render command:
```
uv run manim -ql qwenmoe/viz/scenes_task8_preview.py Task8DataPipelineScene
```
Output: `media/videos/scenes_task8_preview/480p15/Task8DataPipelineScene.mp4`
(70s, 480p15; `media/` is already gitignored as fully regeneratable from
the scene file, same as the earlier previews).

## Design note: chunk-source colors inferred from the real output, not reimplemented internals
`make_fixed_slice` doesn't return its internal chunk order — only the
final flat tensor. Rather than duplicating `dataset.py`'s private
chunk-sizing formula to "replay" the shuffle step by step (risking silent
drift from the real implementation if that formula ever changes), the
scene infers each output position's source chunk directly from the real
returned values: since this demo's input tokens are exactly `0..N-1`, a
token's own value divided by `chunk_size` recovers its original chunk
index using nothing but arithmetic on the real output. This differs from
task 7 preview's approach of directly importing and replaying the real
internal helpers (`_count_pairs`/`_merge_pair`) step by step — there's no
equivalent standalone helper in `make_fixed_slice` to import, so this
preview instead proves the same "real numbers, not fabrication" property
by deriving the visualization purely from what the real function actually
returned.

## Verification
Same approach as every earlier preview: extracted a coarse 6×4
contact-sheet grid across the whole ~70s render for a first pass, then a
finer half-second-interval grid around two caption transitions that looked
"cut off"/glitchy in the coarse pass — both turned out to be a `Write`
animation still in progress at that exact sampled instant, fully legible a
second or two later (the same known "mid-animation frames look garbled"
artifact from every prior preview, not a bug). Individually-grabbed,
accurately-seeked (post-`-i`) settled frames confirm:
- Part 1: the bracket pair visibly slides from `i=0` to `i=1` to `i=3`;
  each step's readout matches the real `ds[i]` values with no overlap
  against the token row or legend above.
- Part 2: the 16/4 blue/orange split and boundary line at index 16 render
  correctly; both separately-built `TokenDataset` readouts (`len=13`/
  `window 0: x=(0,1,2)` and `len=1`/`window 0: x=(16,17,18)`) and the
  closing caption are fully legible with no overlap.
- Part 3: the before/after chunk rows keep matching colors through the
  reorder (`[1,3,2,0]`, confirmed against a direct `torch.randperm`
  cross-check done before animating); both `True`/`True` seed callouts
  render correctly; the closing rope/nope/alibi fan-out diagram and its
  caption settle fully within frame, comfortably clear of the bottom edge.

**No new rendering bug this time** — unlike every earlier preview, this
one didn't turn up a fresh issue. The two lessons already on record (avoid
`MathTex`/`Tex`/any Manim mobject that defaults to LaTeX — this machine
still has none installed; keep vertical spacing conservatively within
frame bounds rather than discovering an overflow after rendering, per task
6 preview's bar-chart bug) were simply applied from the start this time,
for both the sliding-window brackets/captions and the fan-out diagram at
the very bottom of the frame.

## Outcome
A real 70-second video (48 animations) rendered successfully, showing task
8's actual data-pipeline math (`TokenDataset`'s sliding windows,
`train_val_split`'s contiguous split, `make_fixed_slice`'s chunk-shuffle
and its compute-matched-comparison motivation) using real numbers from the
real, tested `qwenmoe.data.dataset` code.
