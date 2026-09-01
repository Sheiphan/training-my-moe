# Extra — Manim preview of Task 3's math (RMSNorm + RoPE + ALiBi)

**Type:** task (supplementary, not a numbered task) · **Date:** 2026-08-31 ·
**Related:** Requirement 7 (visualizations), task 3's RMSNorm/RoPE/NoPE/
ALiBi; see [003-norm-and-positional.md](003-norm-and-positional.md) for
the full concept explainers/derivations — this entry only covers the new
visualization work, not the math itself again.

## Why this isn't numbered `005`
Numbered entries (`001`-`004`) track task numbers 1:1. Task 5 is the MoE
feed-forward layer — this Manim work is neither task 5 nor the formal
Task 13 visualization work (which consumes a `TraceBundle` captured from a
real trained checkpoint via `forward_with_trace()`, none of which exists
yet — that needs tasks 6, 7, and 12 first). It's a **standalone preview**,
requested directly, ahead of the formal pipeline. Once Task 13 is reached,
it may reuse ideas from this file or be entirely separate; this file isn't
claiming to satisfy Task 13.1/13.4.

## What was built
`qwenmoe/viz/scenes_task3_preview.py::Task3PositionalMathScene` — a single
Manim scene with three parts, each animating **real numbers produced by
calling the actual, already-tested classes** (`RMSNorm`, `RotaryEmbedding`,
`AlibiBias`) directly — not fabricated illustration numbers, consistent
with the spirit of Requirement 7.1 even though the formal `TraceBundle`
mechanism isn't built yet:

1. **RMSNorm** — bar charts of `x = [3, 4, 0, 0]` before/after
   `RMSNorm(dim=4)`, with the RMS arithmetic shown as text.
2. **RoPE** — a vector on a coordinate plane, animated rotating from
   position 0 to position $\pi/2$ using `RotaryEmbedding(head_dim=2)`, with
   the actual computed `q_rot` printed at the end.
3. **ALiBi** — a bar chart of `AlibiBias(n_heads=2).attention_bias(4)`'s
   row for query `i=3`, showing the real bias values across distance.

Render command (also how to re-render at higher quality later —
`-ql`/`-qm`/`-qh` = low/medium/high):
```
uv run manim -ql qwenmoe/viz/scenes_task3_preview.py Task3PositionalMathScene
```
Output: `media/videos/scenes_task3_preview/480p15/Task3PositionalMathScene.mp4`
(34s, 480p15). Added `media/` to `.gitignore` — it's fully regeneratable
from the scene file, no reason to commit rendered video binaries.

## Bug found and fixed: Manim's `BarChart` needs LaTeX; this machine has none
First attempt used Manim's built-in `BarChart` (with `bar_names=[...]`).
It crashed with `FileNotFoundError: ... 'latex'` — `BarChart`'s axis-label
rendering defaults to `MathTex`/`Tex` internally (even though nothing in
this project's own code called `MathTex` directly), and this machine has
no LaTeX distribution installed (confirmed: `which latex` / `which
pdflatex` both empty). **Fix:** wrote a small hand-built bar-chart helper
(`_manual_bars`, using only `Rectangle` + `Line` + `Text` — `Text` is
Pango-based via `manimpango`, already confirmed working, no LaTeX
involved) instead of Manim's `BarChart`. **Lesson for future viz tasks
(13, 14):** avoid `MathTex`/`Tex` and any higher-level Manim mobject that
defaults to them (`BarChart`, some `Axes` configurations) unless a LaTeX
distribution is confirmed installed first — stick to `Text` for all
labels/equations rendered as plain strings.

## Verification
Since this produces a video, not a testable return value, verification
was visual: extracted contact-sheet grids of frames sampled across the
render (`ffmpeg ... tile=...`) and viewed them. Confirmed, at properly
"settled" (post-animation, mid-`wait()`) timestamps:
- RMSNorm: all 4 bars + labels present, `[1.2, 1.6, 0.0, 0.0]` shown
  correctly for both charts.
- RoPE: the arrow visibly sweeps from horizontal to vertical; final label
  reads `q_rot=[-0.0000, 1.0000]` — correct (cos(π/2) is a tiny
  floating-point residual near zero, sin(π/2)=1).
- ALiBi: bars read `-0.1875, -0.125, -0.0625, 0` for `j=0..3` — matches
  the hand-worked example in entry 003 exactly.

Frames grabbed *mid*-transition (e.g. while a `Create`/`Write` animation
was still in progress) looked incomplete/garbled at first glance — this
was just an artifact of sampling an in-progress animation, not a rendering
bug; letting the video play continuously (or sampling during a `wait()`)
shows it correctly throughout.

## Outcome
A real 34-second video rendered successfully, showing the actual Task 3
math (RMSNorm rescaling, RoPE rotation, ALiBi's distance penalty) using
real numbers from the real, tested code.
