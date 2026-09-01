# Extra — Manim preview of Task 5's math (MoE routing, dispatch, aux loss)

**Type:** task (supplementary, not a numbered task) · **Date:** 2026-08-31 ·
**Related:** Requirement 7 (visualizations), task 5's MoE feed-forward
layer; see [005-moe-layer.md](005-moe-layer.md) for the full concept
explainers/derivations — this entry only covers the new visualization
work, not the math itself again.

## Why this isn't numbered `006`
Same reasoning as
[extra-manim-preview-task3.md](extra-manim-preview-task3.md): numbered
entries track task numbers 1:1, and task 6 (full model assembly) hasn't
happened yet. This is a **standalone preview** of task 5's math, requested
directly, ahead of the formal Task 13 visualization pipeline (which
consumes a `TraceBundle` captured from a real trained checkpoint via
`forward_with_trace()` — none of that exists until tasks 6, 7, and 12 are
done). It may inform Task 13.3 (the formal MoE routing scene) later, but
isn't claiming to satisfy it now.

## What was built
`qwenmoe/viz/scenes_task5_preview.py::Task5MoEMathScene` — a single Manim
scene with three parts, each animating **real numbers produced by calling
the actual, already-tested classes** (`Router`, `select_top_k`,
`AuxLossBalancer` from `qwenmoe.model.moe`) — the same worked examples
from [005-moe-layer.md](005-moe-layer.md), cross-checked by running them
in a plain Python shell before animating, not fabricated illustration
numbers:

1. **Router → softmax → top-k → renormalize.** A real `Router(d_model=4,
   n_routed_experts=4)`, hand-set so a one-hot input reproduces the
   journal's worked-example logits `[2.0, 0.5, -1.0, 1.0]`. Animates the
   logits as bars, then softmax probabilities, highlights the top-2
   experts selected by the real `select_top_k`, and shows their
   renormalized weights (`0.7311`, `0.2689`, summing to 1).
2. **Dispatch + combine.** A schematic diagram: a token fans out to all 4
   routed experts + the shared expert; the 2 selected routed experts (in
   yellow, using the same weights from part 1) and the shared expert (in
   blue, unweighted) converge into a "+" node and on to an output box; the
   2 unselected experts (grayed out) get a caption noting they're skipped
   with no gradient this batch (Requirement 2.5).
3. **Load-balancing auxiliary loss.** A real `AuxLossBalancer
   (n_routed_experts=4)` computes the aux loss for two scenarios: uniform
   routing (lands at exactly `alpha=0.01`, as derived in the journal
   entry) versus the journal's mildly-imbalanced worked example
   (`0.013`) — shown as a 2-bar comparison.

Render command:
```
uv run manim -ql qwenmoe/viz/scenes_task5_preview.py Task5MoEMathScene
```
Output: `media/videos/scenes_task5_preview/480p15/Task5MoEMathScene.mp4`
(47s, 480p15; `media/` is already gitignored as fully regeneratable from
the scene file, same as the task 3 preview).

## Bug found and fixed: long bar names overlap at default bar spacing
`_manual_bars` (copied from the task 3 preview rather than factored into a
shared module — see below) was designed around short names like `"e0"`/
`"e1"`. The aux-loss part's first attempt used `"uniform routing"`/`"this
example"` as bar names, which are wide enough at the default
`bar_width=0.7, gap=0.35` spacing that the two labels visually overlapped
(confirmed via an extracted frame: `"uniform routin[g][t]his example"` ran
together). **Fix:** shortened the names to single words (`"uniform"`/
`"example"`) and widened `gap` to `1.1` for this specific chart. Re-
rendered and re-extracted the frame to confirm the labels no longer touch.
**Lesson for future bar-chart use (including Task 14's expert-utilization
scene):** `_manual_bars` needs either short names or an explicitly widened
`gap`/`bar_width` whenever a label is more than a couple of characters —
it doesn't wrap or auto-size text to fit.

## Design note: `_manual_bars` was duplicated, not shared
This preview copies the same ~20-line `_manual_bars` helper from
`scenes_task3_preview.py` instead of factoring it into a shared module.
`qwenmoe/viz/common.py` is reserved by design.md for task 12's
`capture_trace(checkpoint_path, prompt) -> TraceBundle` — giving it
unrelated preview-only chart helpers now would blur that boundary for a
piece of code explicitly framed as a standalone, possibly-throwaway
preview (Requirement 7.7's "each scene runs independently, no shared
state" spirit extended here to "no shared preview-only helper module"
too). If Task 13.3 ends up wanting the same manual-bar-chart approach,
factoring it out then — once it's clear what the *formal* scenes actually
need — is one refactor, not a speculative one now.

## Verification
Same approach as the task 3 preview: extracted contact-sheet grids of
frames sampled across the whole render plus several individually-grabbed
frames at settled (post-animation, mid-`wait()`) timestamps, and viewed
them directly. Confirmed:
- Part 1: logits bars read `2, 0.5, -1, 1`; softmax bars read `0.6095,
  0.136, 0.0303, 0.2242` (matches the journal's worked example); top-2
  highlight correctly picks experts 0 and 3; renormalized weights read
  `0.7311`/`0.2689`, matching the journal exactly.
- Part 2: the dispatch diagram fans out from `token x` to all 5 boxes,
  grays out experts 1/2 with the "skipped, no gradient (Req 2.5)"
  caption, and converges only experts 0/3 (labeled with their real
  weights) plus the shared expert (labeled "unweighted") into the `+`
  node and on to `output`.
- Part 3 (after the bar-name-overlap fix): bars read `0.01` (uniform) and
  `0.013` (this example) with clearly separated labels, matching the
  journal's hand-derived numbers exactly.

## Outcome
A real 47-second video rendered successfully, showing task 5's actual MoE
math (router top-k gating with renormalization, dispatch/combine with a
skipped zero-token expert, and the load-balancing auxiliary loss) using
real numbers from the real, tested `qwenmoe.model.moe` code.
