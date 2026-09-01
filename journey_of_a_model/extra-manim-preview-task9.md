# Extra — Manim preview of Task 9's math (parameter counting, token budgets, throughput calibration)

**Type:** task (supplementary, not a numbered task) · **Date:** 2026-08-31 ·
**Related:** Requirement 7 (visualizations), task 9's scaling-law
utilities; see [009-scaling-law-utilities.md](009-scaling-law-utilities.md)
for the full concept explainers/derivations — this entry only covers the
new visualization work, not the math itself again.

## Why this isn't numbered `010`
Unlike the earlier `extra-manim-preview-task*` entries — where the
reasoning was "the next numbered task hasn't happened yet, so this can't
be the formal pipeline" — task 9 is already done (entry
[009](009-scaling-law-utilities.md)), and there genuinely is no future
formal task this preview will later be superseded by: Task 13's formal
scenes (13.1-13.5) are scoped specifically to *architecture* math
(embedding, attention, MoE routing, norm/activation, output projection),
driven by a `TraceBundle` captured from `forward_with_trace()` (task 12) —
none of that covers parameter counting or throughput calibration at all.
Task 14 (training/run observability) is the closest formal relative, but
that reads JSON-lines logs from a real completed `Trainer` run (task 10),
not a live call to `qwenmoe.train.scaling`'s functions. So this preview
isn't "ahead of" a future formal scene the way the earlier ones were —
it's simply outside every formal scene's planned scope, permanently. It's
still marked supplementary rather than folded into entry 009 itself, for
the same reason as always: 009 is the implementation record; this is the
separate visualization built on top of it, following the same standalone,
real-classes-directly approach as
[extra-manim-preview-task3.md](extra-manim-preview-task3.md) through
[extra-manim-preview-task8.md](extra-manim-preview-task8.md).

## What was built
`qwenmoe/viz/scenes_task9_preview.py::Task9ScalingLawScene` — a single
Manim scene with three parts, each calling the actual, already-tested
`qwenmoe.train.scaling` functions live inside `construct()` and animating
whatever numbers come back — not fabricated illustration numbers:

1. **`count_params`'s total-vs-active split.** A real tiny model (the
   exact fixture `tests/test_scaling.py` and entry 009's worked example
   use: `d_model=16, n_layers=2, n_routed_experts=4, top_k=2`, giving
   `total=17,904`/`active=11,760`). Two stacked bars share a baseline:
   the **total** bar stacks embedding/attention/moe-active/moe-skipped
   (gray/blue/green/yellow); the **active** bar stacks only the first
   three — so the active bar's top edge visibly lines up with exactly
   where the total bar's green-to-yellow transition sits, making "active
   is a strict prefix of total, missing only the skipped-expert slice"
   a directly visible fact, not just an asserted one. A zoom-in beat then
   isolates just the MoE segment as its own two-bar comparison
   (`moe total=15,520` vs `moe active=9,376`), captioned with the fact
   that embedding/attention are byte-for-byte identical either way. A
   closing beat shows *why*: four expert boxes (real per-expert param
   count, `1,536`, read directly off `model.blocks[0].moe.
   routed_experts[0]`), the first `top_k=2` colored green ("ACTIVE"), the
   remaining `n_routed_experts - top_k=2` colored gray ("skipped").
2. **`chinchilla_budget` + `epochs_for_budget`'s token-budget planning.**
   `chinchilla_budget(active_params=11,760) = 235,200`, computed live from
   part 1's same tiny model. Two "repeat bars" then divide that same
   235,200-token budget by two illustrative corpus sizes: `unique_tokens=
   40,000` gives `epochs_for_budget = 5.88` (bar outlined red, tick marks
   showing 5 full passes plus a partial one, caption reads "WARNING");
   `unique_tokens=100,000` gives `2.35` (outlined green, no warning) —
   both real calls, both crossing the threshold exactly where the real
   function's own `_EPOCH_WARNING_THRESHOLD` says they should (imported
   directly from `qwenmoe.train.scaling`, not re-typed as a separate
   magic `4`). A closing beat repeats this at this project's actual scale
   — a real `BASELINE` model built live, `chinchilla_budget(9,321,728) =
   186,434,560` tokens, checked against tiny-Shakespeare's real downloaded
   corpus size (`1,115,394`, task 8) — `epochs_for_budget(...) = 167.15`,
   firing the warning by a huge margin, captioned with *why* (task 8's
   TinyStories fixed-slice tier exists precisely because tiny-Shakespeare
   was never meant to absorb a full Chinchilla-sized budget).
3. **`calibrate_throughput`'s empirical measurement.** The same tiny
   model, a real `(4, 16)`-shaped batch, `calibrate_throughput(..., device=
   "cpu"/"mps", n_steps=15)` called live for both devices this machine
   actually has. A simple three-box flow (warm-up → `n_steps` timed
   steps → `tokens_per_sec`) illustrates the procedure, then a bar chart
   shows the real measured numbers for *this* tiny model — captioned
   against entry 009's real `BASELINE`-scale numbers (`cpu ~10,056`,
   `mps ~12,556`) for comparison.

Render command:
```
uv run manim -ql qwenmoe/viz/scenes_task9_preview.py Task9ScalingLawScene
```
Output: `media/videos/scenes_task9_preview/480p15/Task9ScalingLawScene.mp4`
(74s, 480p15, 49 animations; `media/` is already gitignored as fully
regeneratable from the scene file, same as the earlier previews).

## A genuinely interesting real finding, surfaced only by measuring
For this tiny model, **`cpu` measured faster than `mps`** (`cpu ~26,764`
tok/s vs `mps ~8,054` tok/s in this render) — the *opposite* ranking from
the real `BASELINE`-scale numbers in entry 009 (`cpu ~10,056`, `mps
~12,556`, `mps` faster there). Both numbers are real, measured live, not a
bug in either scene. The explanation is exactly what Requirement 11.4 is
about: for a model this tiny, MPS's per-step dispatch/kernel-launch
overhead is large relative to the (very small) amount of actual matrix
math each step does, so plain CPU execution — with no such dispatch
overhead — wins; at `BASELINE` scale there's enough real matrix math per
step that MPS's parallelism advantage overcomes its fixed per-step
overhead and it pulls ahead instead. Neither number could have been
predicted from a theoretical FLOPs estimate alone — which device wins
depends on where a *specific* model size sits relative to a *specific*
backend's fixed overhead, exactly the kind of fact this task's own
docstring says can only be known by measuring. This wasn't planned going
into building the scene; it fell out of actually running
`calibrate_throughput` live for the animation instead of hard-coding a
plausible-looking number.

## Design notes worth calling out

**The active bar is drawn as a literal visual subset of the total bar,
not a separately-scaled bar next to it.** Both bars share the same `unit`
(pixels-per-parameter) and the same baseline `y`, and the active bar's
segment list is exactly the total bar's segment list with the last
(moe-skipped) entry removed — so the active bar's rectangles are, by
construction, identical in position/height to the total bar's first three
segments. This means the visual "active is total minus one slice" claim
can't drift out of sync with the real numbers the way it could if the two
bars were computed/scaled independently and merely happened to look
similar.

**Reused the real `_EPOCH_WARNING_THRESHOLD` constant instead of
hard-coding `4` a second time.** It's a private (underscore-prefixed)
module constant, not part of `qwenmoe.train.scaling`'s public API — but
importing it directly means this scene's "is this a warning?" coloring
logic can never silently disagree with the real function's own threshold
if it's ever tuned later, at the cost of one slightly unusual import.

**Three independent parts, each rebuilding its own tiny model from
scratch** (same convention as `scenes_task6_preview.py`'s three
independently-built traces) — `count_params`/`chinchilla_budget`/
`calibrate_throughput` are all cheap enough on a fixture this small that
rebuilding costs nothing meaningful, and keeping each part fully
self-contained means any one part can be read, understood, or modified
without tracing state threaded through the other two.

## Verification
Same approach as every earlier preview, done in two passes. **Coarse
pass:** 16 frames sampled evenly across the whole ~74s render (fast,
`-ss`-before-`-i` seeking) tiled into a 4x4 contact sheet for a first
look. Several cells looked ambiguous (some overlapping/partial text,
some apparently "empty" content areas) — per the task 6 lesson, fast
seeking snaps to the nearest preceding keyframe rather than the exact
requested timestamp, so an ambiguous-looking coarse frame doesn't by
itself mean anything is wrong. **Accurate pass:** re-extracted 11 of the
specific suspicious timestamps using frame-accurate `select`-filter
seeking (single full decode, `-i` before any seeking) in one `ffmpeg`
invocation. Every case resolved to one of two expected, harmless causes,
not a real bug:
- A `Write`/`Create` animation genuinely still in progress at that exact
  sampled instant (Manim's default `Write` duration scales with text
  length, so longer sentences legitimately take several seconds to fully
  appear — the same well-known "mid-animation frames look partially
  drawn" artifact from every earlier preview, confirmed here for e.g. the
  `active_bar`/`active_label` pair appearing a beat after `total_bar` at
  `t=7.5s`, both fully settled together by `t=9s`).
- A genuine `self.wait()` idle gap between one beat's `FadeOut` and the
  next beat's first `Write`/`Create`, which looks "empty" only because
  there's briefly nothing on screen between two beats — not a dropped
  animation.
Two additional targeted frames (`t=14.3s`, `t=22s`) confirmed the
MoE-zoom bars (`moe total=15,520` vs `moe active=9,376`) and the
four-expert-box mechanism diagram (two green "ACTIVE" boxes, two gray
"skipped" boxes, each correctly labeled `1,536 params`) both render fully
legible once settled. Combined with the coarse pass's cleanly-rendered
frames for the remaining beats (the `chinchilla_budget`/`epochs_for_budget`
bars and their red/green warning coloring, the real-`BASELINE`-scale
callout, the warm-up/timed/result flow diagram, and the final cpu-vs-mps
bar chart), every part of the scene was visually confirmed correct.

**No rendering bug found** — same clean-first-render outcome as task 8's
preview. The by-now-standard lessons (avoid `MathTex`/`Tex`/`BarChart`/
`Axes`, which default to LaTeX; keep bar-chart `unit` sized to that
specific chart's own value range, computed dynamically rather than
guessed; keep vertical layout conservatively within frame bounds; use
accurate seeking to double-check any ambiguous fast-seek frame before
concluding there's a real bug) were simply applied from the start.

## Outcome
A real 74-second video (49 animations) rendered successfully on the first
attempt, showing task 9's actual scaling-law math (`count_params`'s
total-vs-active parameter breakdown, `chinchilla_budget`'s reference
token budget, `epochs_for_budget`'s repetition warning, and
`calibrate_throughput`'s empirical device timing) using real numbers from
the real, tested `qwenmoe.train.scaling` code — including a genuine,
previously-undocumented finding (CPU beating MPS for a tiny model, the
reverse of the `BASELINE`-scale ranking) that only turned up because the
scene calls the real function live instead of hard-coding a plausible
number.
