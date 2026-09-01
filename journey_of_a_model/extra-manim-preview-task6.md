# Extra — Manim preview of Task 6's math (pre-norm wiring, residual growth, weight tying)

**Type:** task (supplementary, not a numbered task) · **Date:** 2026-08-31 ·
**Related:** Requirement 7 (visualizations), task 6's full model assembly;
see [006-full-model-assembly.md](006-full-model-assembly.md) for the full
concept explainers/derivations — this entry only covers the new
visualization work, not the math itself again.

## Why this isn't numbered `007`
Same reasoning as
[extra-manim-preview-task3.md](extra-manim-preview-task3.md) and
[extra-manim-preview-task5.md](extra-manim-preview-task5.md): numbered
entries track task numbers 1:1, and task 7 (tokenizers) hasn't happened
yet. This is a **standalone preview** of task 6's math, requested
directly, ahead of the formal Task 13 visualization pipeline (which
consumes a `TraceBundle` captured from a real trained checkpoint via
`forward_with_trace()` — that needs tasks 7 and 12 first). It may inform
Task 13's scenes later, but isn't claiming to satisfy any of them now.

## What was built
`qwenmoe/viz/scenes_task6_preview.py::Task6ModelAssemblyScene` — a single
Manim scene with three parts, each animating **real numbers produced by
calling the actual, already-tested classes** (`TransformerBlock`,
`QwenMoEModel`) — cross-checked by running them in a plain Python shell
before animating, not fabricated illustration numbers:

1. **Pre-norm residual wiring.** A real tiny `TransformerBlock`
   (`torch.manual_seed(0)`, `d_model=4`), run stage by stage (not just
   through `forward`) so every intermediate tensor's real norm can be
   shown: `x_in` (‖x‖=3.423) → `RMSNorm` (‖x‖=3.464) → `Attention`
   (‖out‖=0.738) → `+` residual → `x_mid` (‖x‖=3.296) → `RMSNorm`
   (‖x‖=3.464) → `MoE` (‖out‖=0.580) → `+` residual → `x_out`
   (‖x‖=3.425). A caption calls out that both `RMSNorm` calls land on the
   *exact same* output norm (3.464) regardless of the residual stream's
   own (different) input norm at that point — concretely demonstrating
   what "pre-norm" buys a sublayer.
2. **Residual-stream growth across real depth.** A real 6-layer
   `QwenMoEModel` (matching this project's baseline depth), measuring the
   batch-averaged squared norm of `x` after the embedding and after each
   of its 6 real blocks (32 random sequences, `torch.manual_seed(0)`):
   `6.37, 8.07, 9.29, 9.56, 11.99, 11.63, 13.60` — an empirical,
   real-code analogue of Xiong et al. (2020)'s $E[\|x_l\|^2]=O(l\cdot d)$
   pre-norm growth result (journal entry 006), rather than only asserting
   the paper's claim.
3. **Weight tying, via the real tied `lm_head`.** A real tiny
   `QwenMoEModel` (`vocab_size=3, d_model=2, tie_word_embeddings=True`),
   with its embedding rows hand-set to reproduce journal entry 006's
   cat/dog/run worked example. Confirms `model.lm_head.weight is
   model.embedding.weight -> True` live, then calls the real, tied
   `model.lm_head(x)` directly on `x=[0.5, 0.5]` and shows the resulting
   softmax probabilities (`cat=0.2741, dog=0.2741, run=0.4519`) — matching
   the journal's hand-derived example exactly, but computed by the real
   module this time, not recomputed by hand again.

Render command:
```
uv run manim -ql qwenmoe/viz/scenes_task6_preview.py Task6ModelAssemblyScene
```
Output: `media/videos/scenes_task6_preview/480p15/Task6ModelAssemblyScene.mp4`
(57s, 480p15; `media/` is already gitignored as fully regeneratable from
the scene file, same as the earlier previews).

## Bug found and fixed: tallest bar overflowed past the frame
First render's residual-stream-growth chart used `_manual_bars(..., unit=
0.3)`. The tallest real value (block 6's `13.60`) scaled to a bar height
of `4.08` Manim units — tall enough that its value-label text landed
above the visible camera frame (and behind/above the title), confirmed by
an extracted frame showing only 6 of the chart's 7 value labels, with the
7th silently cut off past the top edge. **Fix:** reduced `unit` to `0.18`
(max bar height `2.45`, comfortably clearing the title). Re-rendered and
re-extracted the same part of the video, confirming all 7 bars/labels
(`6.37` through `13.6`) are now fully visible. **Lesson for future
bar-chart use:** `_manual_bars`'s `unit` has to be sized to the *specific*
value range being plotted — no auto-scaling/auto-fit exists, so a `unit`
that looked fine for one chart (e.g. the small `aux_loss`/probability
values in earlier previews) can silently overflow the frame for a chart
with a wider value range, and this failure mode is invisible unless a
frame is actually extracted and viewed, not just "the render didn't
crash."

## Bug found and fixed (verification-tooling, not the video itself):
`ffmpeg -ss <time> -i <file>` gives an inaccurate frame
Grabbing a frame with `-ss 44 -i video.mp4` (seeking *before* `-i`, which
seeks to the nearest preceding keyframe rather than the exact timestamp)
produced a frame that looked like a real sequencing bug: the "weight
tying" title and an `identity_label` caption were visible, but the
`shared_box` rectangle they should have already appeared alongside (per
the animation order in the code) was completely missing. Re-extracting
the *same* nominal timestamp with `-ss` moved *after* `-i` (frame-accurate
but slower, since it decodes from the start) showed `shared_box` present
exactly as expected. **Lesson:** when a frame grabbed for Manim
verification shows something that looks like an impossible object-
ordering bug (an object appears before one that the code creates earlier
in the same sequential `self.play()` chain), re-extract with accurate
(post-`-i`) seeking before concluding the *scene* is buggy — imprecise
keyframe-seek artifacts can fabricate an apparent bug that isn't really
there. This is a distinct failure mode from the already-known "mid-
animation frames look garbled" one (task 3/5 previews) — that one is
about *partial* animations; this one can make an entire, already-fully-
rendered object disappear from the extracted still.

## Verification
Same approach as the earlier previews: extracted a 4x4 contact-sheet grid
of frames sampled evenly across the whole render, plus several
individually-grabbed frames at settled (post-animation, mid-`wait()`)
timestamps, and viewed them directly. Confirmed, after the bar-overflow
fix:
- Part 1: both stages (`x_in → ... → x_mid` and `x_mid → ... → x_out`)
  show the correct real norms at every node, the residual arc is
  correctly drawn from the stage's input box to the `+` node, and the
  closing caption about both `RMSNorm` outputs sharing norm 3.464 renders
  cleanly once settled.
- Part 2: all 7 bars (`emb` through `blk6`) and their value labels
  (`6.37` through `13.6`) are fully visible with clear separation from
  the title; the Xiong et al. (2020) caption renders fully once settled.
- Part 3: `shared_box`/arrows/labels appear in the correct order (once
  re-verified with accurate seeking); the final probability bars read
  `cat=0.2741, dog=0.2741, run=0.4519`, matching the real computation
  confirmed independently in a plain Python shell beforehand.

## Outcome
A real 57-second video rendered successfully, showing task 6's actual
full-model-assembly math (pre-norm residual wiring with real per-stage
norms, real empirical residual-stream growth across a 6-layer model, and
weight tying reproduced through the real tied `lm_head`) using real
numbers from the real, tested `qwenmoe.model.block`/`qwenmoe.model.
transformer` code.
