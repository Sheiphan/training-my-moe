# Extra — Manim preview of Task 15's math (loss-free balancing + comparison framework)

**Type:** task (supplementary, not a numbered task) · **Date:** 2026-08-31 ·
**Related:** Requirement 7 (visualizations), task 15's `LossFreeBalancer`/
`run_comparison` work; see
[015-research-comparison-framework.md](015-research-comparison-framework.md)
for the full concept explainers/derivations — this entry only covers the
new visualization work, not the math itself again.

## Why this isn't numbered `016`, and won't be superseded later
Same situation as entries
[extra-manim-preview-task9.md](extra-manim-preview-task9.md),
[extra-manim-preview-task10.md](extra-manim-preview-task10.md), and
[extra-manim-preview-task11.md](extra-manim-preview-task11.md): task 15 is
already done (entry [015](015-research-comparison-framework.md)), so this
can't be "ahead of" a future formal task the way the task 3/5/6/7/8
previews were. The remaining formal tasks are 16 (MLA, a self-contained
attention-variant stretch goal), 17 (LoRA fine-tuning, a self-contained
adapter stretch goal), and 18 (project documentation) — none of them ever
touch `LossFreeBalancer` or `run_comparison` again. So this preview is the
only visualization this content will ever get, not a placeholder standing
in for something more thorough later.

## What was built
`qwenmoe/viz/scenes_task15_preview.py::Task15ComparisonScene` — a single
Manim scene with two parts, each calling real, already-tested
`qwenmoe.model.moe`/`qwenmoe.train.compare` code live inside `construct()`
and animating whatever numbers come back:

1. **`LossFreeBalancer`'s real bias-update loop, run for 40 rounds.** Fixed
   logits `[5.0, 1.0, 1.0, -5.0]` (the exact worked example from entry
   015), `update_rate=0.05` — the scene calls the real
   `LossFreeBalancer.selection_scores`/`update` and the real
   `select_top_k` every round (no reimplemented math), snapshotting the
   real bias vector at rounds 0, 10, and 39 and drawing all three as
   side-by-side bar-chart panels (bars shifted `+2.5` for visibility only;
   the printed labels show the real, unshifted bias value). The closing
   caption reports the real observed win-rate flip for expert 0.
2. **A real `run_comparison()` call, `aux_loss` vs. `loss_free`, on the
   real tiny-Shakespeare corpus.** A tiny base config (`d_model=32,
   n_layers=1`), one identical 5,000-token fixed slice, one identical
   30-step token budget, two named variants differing only in
   `moe_balancing`. The resulting real `ComparisonReport` is drawn as a
   `val_perplexity` bar chart plus a small text table (`val_loss`, `ppl`,
   `tokens_per_sec` per variant) — the exact same fields
   `scripts/run_comparison.py`'s CLI table prints, just drawn with Manim
   instead of `print()`.

Render command:
```
uv run manim -ql qwenmoe/viz/scenes_task15_preview.py Task15ComparisonScene
```
Output: `media/videos/scenes_task15_preview/480p15/Task15ComparisonScene.mp4`
(32.3s, 480p15, 23 animations; `media/` is already gitignored as fully
regeneratable from the scene file, same as every earlier preview).

## Design notes worth calling out

**No redundant "missing prerequisite" check was needed for Part 2.**
Task 11's preview had to add its own explicit `FileNotFoundError` check
for a missing checkpoint, since `torch.load` alone doesn't name the exact
regeneration command. `run_comparison` (task 15.2) already raises exactly
that kind of actionable error itself (`"no prepared data found under
{data_dir} -- run 'uv run python scripts/prepare_data.py ...' first"`),
so this scene just lets that error propagate naturally instead of
duplicating the check — one benefit of task 15's own design already
building in a clear failure message before this preview needed one.

**A genuinely interesting real dynamic showed up in Part 1 that a
hand-picked, monotonic-looking dummy trace wouldn't have revealed:**
expert 0's bias doesn't fall forever — it drops to `-0.50` by round 10
(having won every round so far), but by round 39 it's actually climbed
back up to `+0.25`. This is real, not a bug: once experts 1/2's rising
bias overtakes expert 0's falling one (they tie at a biased score of
`0.5177` right around round 10, since `softmax([5,1,1,-5]) ≈ [0.9646,
0.01766, 0.01766, 0.0000437]` and expert 0's bias has fallen exactly far
enough to erase its huge raw-probability lead), routing flips to experts
1/2 — and *they* become the newly-overloaded ones, while expert 0,
now rarely selected, starts drifting back up as a comparatively
underloaded expert. The mechanism doesn't just suppress one static
winner; it keeps re-equilibrating as the winner itself changes, an
oscillating self-correction entry 015's simpler "expert 0's win rate:
100% → lower" framing undersells slightly — worth noting here since the
real animated numbers surfaced it, and entry 015 is corrected to point
here for the fuller picture rather than restating it there.

**Reused `_title`/`_manual_bars` verbatim from every earlier preview** —
still no `Axes`/`BarChart`/`MathTex` anywhere (confirmed again via `which
latex`/`pdflatex`, both absent on this machine). The perplexity bars use
`unit=0.15` (values are in the 15-25 range, unlike earlier scenes'
0-to-1-scale probabilities) and the bias bars use a `+2.5` shift precisely
because — unlike every earlier scene's all-nonnegative values — a bias
can be legitimately negative, and `_manual_bars`' zero-baseline design
(confirmed already, entry extra-task11) renders a negative value as a
barely-visible sliver rather than a readably-sized downward bar. Shifting
the *height* while keeping the *label* unshifted (the helper already
takes `heights` and `value_labels` as independent arguments) solved this
without touching the shared helper itself.

**Part 2 writes real (throwaway) checkpoints/logs to a fresh
`tempfile.mkdtemp()`, not `runs/`.** `runs/` is this project's real
experiment-output directory (e.g. `runs/smoke_test`, referenced by
multiple other previews); a Manim preview's internal demo run has no
business cluttering it, so this is the first preview in the series to
actually write new files as a side effect of rendering, and it does so
entirely outside the repo.

## Verification
Same two-pass approach as every earlier preview. **Coarse pass:** 15
frames sampled evenly across the whole 32.3s render (`-ss`-before-`-i`
seeking, tiled into a contact sheet via the same inline Pillow script
this series has used since `montage` was found to be unavailable on this
machine). One frame (mid-`Write` of the comparison table's second line,
`t≈23.7s`) looked briefly doubled/bold — re-checked exactly per this
series' standing rule ("check ~0.5-1s later before concluding it's a
bug") at `t=24.8s` and `t=30.1s`, both fully settled and legible with no
artifact. **Accurate pass, specific timestamps at native resolution:**
- `t≈12.9s`: all three bias-round panels (round 0/10/39) fully settled,
  legible, and matching the real numbers described above exactly.
- `t≈24.8s` and `t≈30.1s`: the perplexity bar chart, the text table, and
  (at `t≈30.1s`) the closing caption all simultaneously legible with no
  overlap.

**One known, pre-existing cosmetic quirk confirmed again, not newly
introduced:** the title (`.to_edge(UP)`) and each part's own setup line
(`move_to([0, 3.2, 0])`) sit close enough to visibly overlap once both
are drawn — the same standing, previously-documented-and-accepted quirk
every scene in this series shares (most recently reconfirmed in the task
11 preview); not changed here for the same consistency reason.

## Outcome
A real ~32.3-second video (23 animations) rendered successfully on the
first attempt (no layout fixes needed this time, unlike task 11's
preview). Shows task 15's math — `LossFreeBalancer`'s real bias-update
loop across 40 rounds (surfacing a genuine oscillating self-correction
dynamic once the "winning" expert itself changes, not just a monotonic
bias decay), and a real `run_comparison()` call training two variants
(`aux_loss` vs. `loss_free`) on the real tiny-Shakespeare corpus under an
identical fixed slice and token budget — using the real, tested
`qwenmoe.model.moe`/`qwenmoe.train.compare` code directly, with no
fabricated numbers anywhere in the scene.
