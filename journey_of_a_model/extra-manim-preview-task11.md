# Extra — Manim preview of Task 11's math (generation/sampling)

**Type:** task (supplementary, not a numbered task) · **Date:** 2026-08-31 ·
**Related:** Requirement 7 (visualizations), task 11's inference/generation
work; see [011-inference-generation.md](011-inference-generation.md) for
the full concept explainers/derivations — this entry only covers the new
visualization work, not the math itself again.

## Why this isn't numbered `012`, and won't be superseded later
Same situation as entries [extra-manim-preview-task9.md](extra-manim-preview-task9.md)
and [extra-manim-preview-task10.md](extra-manim-preview-task10.md): task 11
is already done (entry [011](011-inference-generation.md)), so this can't
be "ahead of" a future formal task the way the task 3/5/6/7/8 previews
were. Unlike task 9 (partial overlap avoided entirely) or task 10
(partially overlapping task 14's future loss-curve scene), **nothing in
the remaining formal task list ever touches `generate.py` at all**: task
12 adds `forward_with_trace`/`TraceBundle` for one forward pass's internal
tensors, task 13's six scenes animate embeddings/RoPE, attention, MoE
routing, RMSNorm/SwiGLU, and the output head — all single-forward-pass
architecture math — and task 14 animates training-run *logs* (loss curve,
expert utilization). Autoregressive decoding, temperature scaling, and
top-k/nucleus sampling never appear in any of those. So this preview is
the *only* visualization this content will ever get in this project, not
a placeholder standing in for something more thorough later.

## What was built
`qwenmoe/viz/scenes_task11_preview.py::Task11GenerationScene` — a single
Manim scene with four parts, each calling real, already-tested
`qwenmoe.model.generate` code live inside `construct()` and animating
whatever numbers come back:

1. **Temperature scaling, a real `torch.softmax(z/T)` for three
   temperatures.** Fixed logits `z = [2.0, 1.0, 0.0]`, computed live for
   `T ∈ {0.5, 1.0, 2.0}` and drawn as three side-by-side bar-chart panels
   — the exact same numbers entry 011's worked-example table reports
   (`T=0.5 → [0.867, 0.117, 0.016]`, `T=1.0 → [0.665, 0.245, 0.090]`,
   `T=2.0 → [0.506, 0.307, 0.186]`), confirming visually that lower `T`
   sharpens toward the top logit and higher `T` flattens toward uniform.
2. **`filter_top_k`, called for real.** `logits = [2.0, 0.5, 3.0, -1.0]`,
   `k=2` — the scene calls the real `filter_top_k` function (not a
   reimplementation) and shows the real "before" bars next to the real
   "after" result: `idx0=2.0` and `idx2=3.0` survive unchanged, `idx1` and
   `idx3` become `-inf`, colored red.
3. **`filter_top_p` (nucleus sampling), called for real, at two
   thresholds.** Probabilities `[0.5, 0.2, 0.15, 0.1, 0.05]` (built into
   logits via `ln(p)`, entry 011's shift-invariance trick), filtered with
   the real `filter_top_p` function at `top_p=0.7` (keeps 2 tokens,
   mass=0.70) and `top_p=0.71` (keeps 3 tokens, mass=0.85) side by side —
   the exact boundary case entry 011's derivation walks through by hand.
4. **Real `generate()` on the real task-10 checkpoint — the hero shot.**
   Loads the actual `runs/smoke_test/checkpoints/step_1000.pt` checkpoint
   (raising a clear `FileNotFoundError` with regeneration instructions if
   it's missing, mirroring `scripts/train.py::_load_prepared_data`'s own
   pattern for a missing prerequisite), rebuilds the real model +
   tokenizer, and calls the real `generate()` twice on the identical
   prompt (`"ROMEO:"`): once with `temperature=0.0` (greedy) and once with
   `temperature=1.0, top_p=0.9` (nucleus). The greedy call reproduces the
   exact repetition-loop degeneration entry 011's Outcome section
   documented (`"The the the the the the the the the the the the the the
   the"`, with every `"The"`/`"the"` colored red via Manim `Text`'s
   `t2c=` parameter), directly beside the nucleus call's non-repeating
   output (`"Goumantercosn lorde lafteer wisserterm faate be meest
   mfors"`) — a real, live, side-by-side demonstration of exactly why
   Requirement 6.2 asks for more than greedy decoding.

Render command:
```
uv run manim -ql qwenmoe/viz/scenes_task11_preview.py Task11GenerationScene
```
Output: `media/videos/scenes_task11_preview/480p15/Task11GenerationScene.mp4`
(71.1s, 480p15, 45 animations; `media/` is already gitignored as fully
regeneratable from the scene file, same as every earlier preview).

## Design notes worth calling out

**A real layout bug found and fixed during verification, not before
rendering.** The first render's `filter_top_k` part stacked its "before"
bars, an arrow, and its "after" bars *vertically* (before at one height,
after well below it), with each group's caption placed via
`next_to(group, UP)`. On paper this looked fine; rendered, the "after"
caption's actual position (computed from the group's full bounding box,
including each bar's value label sitting *above* it — taller than a naive
"baseline height" estimate suggests) landed close enough to the "before"
group's bottom labels to visibly overlap, and the closing derivation
caption crowded the "after" group's own x-axis labels the same way. This
was only caught by the same two-pass frame-verification process described
below — reading the code alone didn't reveal it, since the bug is about
`Manim`'s actual bounding-box geometry, not the code's logic. Fixed by
switching to the same **side-by-side (not stacked)** layout the
temperature and top-p parts already used successfully — before/after
groups placed at `x=-3.2`/`x=3.2` at one shared height, connected by a
horizontal arrow, with the closing caption well below both. Re-rendering
only recomputed this one part (Manim's caching left the other three
parts' identical frames untouched), and the re-verified frame confirmed
the fix. **Lesson for any future preview scene with a "before vs. after"
comparison: default to side-by-side, not stacked** — every part in this
series that used side-by-side placement rendered correctly on the first
try; the one stacked attempt didn't.

**The real checkpoint dependency is deliberate, not an oversight.** Part
4 depends on `runs/smoke_test/checkpoints/step_1000.pt` already existing
on disk (it does — task 10's manual verification run produced it, and
task 11's own manual verification reused it too). This is the one part of
the whole preview series so far that depends on an external artifact
outside the scene file itself, `.venv`, or the repo's tracked source —
a deliberate choice, not an accident: entry 011's Outcome section's most
important real finding (the greedy repetition loop, and nucleus sampling
not exhibiting it) is specifically about *that* trained checkpoint, and
re-deriving an equally convincing demonstration from a freshly-trained
throwaway model live in the scene would need enough real training steps
to reach comparable behavior — plausible, but slower to build and a
weaker narrative link back to entry 011's own already-published numbers.
The scene fails fast with an actionable message (naming the exact
`prepare_data.py`/`train.py` commands from entry 010) if the checkpoint
is ever missing, rather than crashing deep inside `torch.load` — the same
"clear, actionable error before a raw stack trace" principle design.md's
Error Handling section already applies to task 14's future log-reading
scenes (Requirement 8.3), applied here even though this scene isn't part
of that formal requirement.

**Reused the same `_title`/`_manual_bars` helpers verbatim from every
earlier preview** — no new charting utility needed, and still no
`Axes`/`BarChart`/`MathTex` anywhere (no LaTeX distribution on this
machine, confirmed again via `which latex`/`pdflatex` before writing a
line of this scene, per the by-now-standard lesson).

## Verification
Same two-pass approach as every earlier preview, run twice (once before,
once after the layout fix above). **Coarse pass:** 16 frames sampled
evenly across the whole 71.1s render (fast, `-ss`-before-`-i` seeking,
tiled into a 4x4 contact sheet via a small Pillow script — `montage`
isn't installed on this machine, so this run used Pillow directly instead
of ImageMagick, functionally identical output). The first coarse pass is
what caught the `filter_top_k` overlap in the first place — a settled
(not mid-animation) frame in that part's row was visibly cluttered, which
the "just mid-`Write`, check half a second later" explanation used by
every earlier false alarm in this series did *not* resolve, since the
frame was already well into that part's `self.wait(2.5)` pause. **Accurate
pass (post-fix), specific timestamps at native resolution:**
- `t=6.663s`: temperature part, `T=0.5`/`T=1.0` panels fully settled and
  legible and matching entry 011's table exactly (`T=2.0`'s panel still
  mid-`Create` at this exact instant — the standard, expected
  mid-animation partial-draw, not a bug — confirmed fully settled at
  `t=11.1s` in the contact sheet's next cell).
- `t=28.871s` (re-checked after the fix): `filter_top_k`'s before/after
  bars now clearly separated left/right, arrow between them, both
  x-axis label rows and the closing derivation caption fully legible with
  no overlap (one minor, pre-existing cosmetic tightness: `idx3`'s `-1.0`
  label sits close to its own `idx3` name label, since a negative bar
  height renders as a barely-visible sliver against `_manual_bars`'
  zero baseline — legible, not a new issue, and not worth a bigger
  special-case in a shared helper used unchanged since task 9).
- `t=38s`/`t=42s`: `filter_top_p`'s two threshold panels (`top_p=0.7`
  keeping 2 tokens, `top_p=0.71` keeping 3) fully legible, kept/cut
  coloring correct, closing caption clean.
- `t=68.826s`: the real-generation part fully settled — both the greedy
  (`"The the the the..."`, red-highlighted) and nucleus
  (`"Goumantercosn lorde lafteer..."`) outputs legible in full, each with
  its own caption.

**One known, pre-existing cosmetic quirk confirmed again, not newly
introduced:** the title (`.to_edge(UP)`) and each part's own setup line
(`move_to([0, 3.2-3.3, 0])`) sit close enough to visibly overlap once
both are drawn — the same standing, previously-documented-and-accepted
quirk shared by every scene in this series (entries 009 and 010 both
confirmed it too); not changed here for the same consistency reason.

## Outcome
A real ~71.1-second video (45 animations) rendered successfully after one
layout fix (caught by this series' own verification process, not left
for a viewer to notice). Shows task 11's sampling math — temperature's
real sharpening/flattening effect on a softmax distribution, `filter_top_k`
and `filter_top_p` both called live and matching entry 011's hand-derived
worked examples exactly, and real `generate()` calls against the real
task-10 checkpoint reproducing the genuine greedy-decoding repetition loop
side by side with nucleus sampling's non-repeating alternative — using the
real, tested `qwenmoe.model.generate` code directly, with no fabricated
numbers anywhere in the scene.
