# Extra — Manim preview of Task 10's math (training-loop mechanics)

**Type:** task (supplementary, not a numbered task) · **Date:** 2026-08-31 ·
**Related:** Requirement 7 (visualizations), task 10's training loop; see
[010-training-loop.md](010-training-loop.md) for the full concept
explainers/derivations — this entry only covers the new visualization
work, not the math itself again.

## Why this isn't numbered `011`
Same situation as entry [extra-manim-preview-task9.md](extra-manim-preview-task9.md):
task 10 is already done (entry [010](010-training-loop.md)), so this can't
be "ahead of" a future formal task the way the task 3/5/6/7/8 previews
were. Unlike task 9's scaling utilities, though, task 10's territory
*partially* overlaps a future formal task: Task 14 ("Training/run
observability visualization") will render a loss-curve scene and an
expert-utilization-distribution scene, both reading real JSON-lines
*logs* left behind by a completed `Trainer` run. This preview's fourth
part (checkpoint/resume) does surface a `final_train_loss` number, which
is loss-curve-adjacent — but it's produced by calling `Trainer.fit()`
*live* inside the scene on a synthetic toy sequence, not by reading a
pre-existing log file the way task 14's formal scene will. The other
three parts (the LR schedule's shape, the embedding-init bug, and AdamW's
real per-parameter update) aren't in task 14's scope at all — task 14
never touches model internals or the optimizer, only the logged
loss/expert-count numbers. So this preview is mostly, but not entirely,
outside every formal scene's planned scope; task 14 will still be the
more thorough, real-training-run treatment of "what did the loss curve
actually look like."

## What was built
`qwenmoe/viz/scenes_task10_preview.py::Task10TrainingLoopScene` — a
single Manim scene with four parts, each calling real, already-tested
`qwenmoe.train.trainer`/`qwenmoe.model.transformer` code live inside
`construct()` and animating whatever numbers come back:

1. **`lr_multiplier`'s warmup+cosine curve.** `warmup_steps=20,
   total_steps=200, base_lr=3e-4`, sampled at 151 points and drawn as a
   single manually-built polyline (`VMobject.set_points_as_corners`, no
   `Axes`/`ParametricFunction`) — a sharp linear ramp to a marked peak
   dot at `step 20 (multiplier=1.00)`, then a smooth cosine roll-off
   toward (not quite reaching) zero by `step 200`. A closing readout
   calls `lr_multiplier` again at five checkpoint steps
   (`0, 19, 20, 100, 199`) and prints the exact multiplier and resulting
   `lr` for each, live.
2. **The task-10 embedding-init bug, real before vs. after.** Two real
   `QwenMoEModel`s built from the identical seed (so every non-embedding
   weight is byte-for-byte identical), one left as-is (the fixed
   `std=0.02` init) and one with its embedding manually reset to
   PyTorch's un-fixed `nn.Embedding` default (`std=1.0`) — reproducing
   the exact bug from entry 010, not a re-typed illustration of it. Both
   get the identical random `(8, 16)` input; the scene calls
   `model(x)`/`F.cross_entropy` on both live and shows the real resulting
   logit-std and loss numbers as two side-by-side bar-chart comparisons
   (`logit std 10.98 → 0.17`; `loss 60.29 → 4.20`, against a computed
   `ln(65)=4.17` reference bar), captioned with the `sigma_z = sqrt(
   d_model) * sigma_W * sigma_h` derivation from entry 010.
3. **AdamW's real per-parameter update, captured live.** A real tiny
   model, a real `torch.optim.AdamW`, one real
   forward+backward+`optimizer.step()` on a random batch — then the
   scene reads `optimizer.state[param]["exp_avg"]`/`["exp_avg_sq"]`
   directly (PyTorch's own internal momentum/variance estimates, not a
   re-implementation) for one concrete weight entry, alongside its real
   gradient and its real before/after value, all six numbers printed
   together.
4. **`Trainer` checkpoint save + real resume.** A real `Trainer.fit()`
   call for a real `model_a` (20 steps on a period-10 repeating token
   stream, `lr=1e-2, warmup_steps=5` for genuinely visible learning in so
   few steps), followed by a **second**, differently-seeded, freshly
   constructed `model_b`, resumed via `trainer.fit(model_b, ...,
   resume_from=result_a.checkpoint_path)` for a 30-step total budget. A
   three-box-plus-arrows diagram shows the real step ranges
   (`run 1: steps 1..20` → `checkpoint: step=20` → `run 2 (resumed):
   steps 21..30`), followed by the same discriminating check
   `tests/test_trainer_integration.py::test_trainer_resume_continues_from_saved_step`
   uses: counting `"type": "step"` JSON-lines log records (30 total, not
   `20 + 30 = 50`, which is what a silently-restarted-from-0 resume would
   have produced) — a real, computed-live proof, not an assertion the
   viewer has to take on faith.

Render command:
```
uv run manim -ql qwenmoe/viz/scenes_task10_preview.py Task10TrainingLoopScene
```
Output: `media/videos/scenes_task10_preview/480p15/Task10TrainingLoopScene.mp4`
(65.5s, 480p15, 39 animations; `media/` is already gitignored as fully
regeneratable from the scene file, same as every earlier preview).

## Design notes worth calling out

**Part 2's "before" model is built by mutating a real `QwenMoEModel`
after construction, not by patching the class.** `torch.manual_seed(0)`
is called immediately before constructing *both* `model_before` and
`model_after`, so every parameter both models start with — including the
embedding, before any override — comes from the identical draw of the
global RNG stream. Only *after* `model_before` is fully constructed does
`nn.init.normal_(model_before.embedding.weight, std=1.0)` run, consuming
RNG state that doesn't feed back into any other parameter. The result:
the two models are identical in every weight except the one the bug is
about, an apples-to-apples comparison that couldn't be achieved by
building two independently-seeded models (which would also differ in
attention/MoE weights, muddying which difference caused the loss gap).

**Part 4's `lr=1e-2, warmup_steps=5` (not the config defaults).** An
early version of this part used `TrainConfig`'s plain defaults
(`warmup_steps=200`), and since both `total_tokens` budgets here produce
only 20-30 steps (`«200» ≫ 20`), the *entire* run stayed inside the
linear-warmup phase, the learning rate never got past `~4.5e-5`, and
`final_train_loss` barely moved (`2.993 → 2.992`) — technically not
wrong, but a weak, unconvincing demonstration of "training continues
correctly after resume." Bumping to `lr=1e-2, warmup_steps=5` (mirroring
`tests/test_trainer_integration.py::test_trainer_runs_and_loss_decreases`'s
own choice, for the same reason) let the schedule actually reach and
decay from its peak within so few steps, producing a real, visible
`1.743 → 1.572` improvement that continues genuinely across the resume
boundary rather than sitting flat.

**The resume "proof" counts JSON-lines log records, not just
`final_step`.** As entry 010 notes, `result_b.final_step == 30` by itself
proves nothing — a resume bug that silently restarted from step 0 would
*also* reach `final_step == 30` for the identical `total_tokens` budget.
The scene reads `trainer.log_path` directly (inside the same
`tempfile.TemporaryDirectory()` context the real `fit()` calls ran in,
before it's cleaned up) and counts `"type": "step"` records — 30, not the
`20 + 30 = 50` a broken restart-from-0 resume would have produced —
mirroring exactly what
`test_trainer_resume_continues_from_saved_step` asserts, so the video's
claim is exactly as rigorous as the test suite's.

**Reused the same `_manual_bars` helper from every earlier preview
verbatim**, and the same `_title` convention (`Text(...).to_edge(UP)`) —
no new charting utility needed for this scene's content, and no
`Axes`/`BarChart`/`MathTex` anywhere (still no LaTeX distribution on this
machine — confirmed again via `which latex`/`pdflatex` before writing a
single line of this scene, per the by-now-standard lesson).

## Verification
Same two-pass approach as every earlier preview. **Coarse pass:** 16
frames sampled evenly across the whole ~65.5s render (fast, `-ss`-before-
`-i` seeking) tiled into a 4x4 contact sheet. Every part's title and
final settled content was legible and correct at this resolution; a
couple of cells (mid-`Create` bar-chart, mid-`Write` caption) looked
partially drawn, as expected mid-animation. **Accurate pass:** re-
extracted specific timestamps at native resolution (no downscale) for
each of the four parts individually:
- Part 1 (`t=10.23s`, `t=14.32s`): the warmup-then-cosine curve and peak
  marker fully drawn; the five-checkpoint readout (`step 0/19/20/100/199`
  with exact multiplier/lr values) fully legible and arithmetically
  consistent with entry 010's hand-derivation.
- Part 2 (`t=26.59s`, `t=29.5s`): the `std=0.17` "after" bar renders as a
  genuinely thin sliver next to the `std=10.98` "before" bar — a real
  consequence of a ~64x scale difference on a linear-height bar chart,
  not a rendering bug (its `logit std=0.17` label is fully legible either
  way). At `t=26.59s` the bottom derivation caption looked partially
  drawn (mid-`Write`); at `t=29.5s`, half a second later, it's fully
  settled and clearly legible — the same "mid-`Write` looks garbled,
  settles quickly" pattern documented in every earlier preview's
  verification section.
- Part 3 (`t=38.87s`): all six real captured numbers (gradient, `m_t`,
  `v_t`, before/after weight, delta) fully legible.
- Part 4 (`t=55.23s`, `t=63.41s`): the three-box-plus-arrows diagram, the
  JSON-lines "real proof" callout, and the final `final_train_loss`
  comparison all fully legible and matching the real numbers the scene's
  own `trainer.fit()` calls produced.

**One known, pre-existing cosmetic quirk confirmed, not newly
introduced:** the title (`_title`, `.to_edge(UP)`) and each part's setup
line (`move_to([0, 3.2, 0])`) sit close enough vertically to visibly
overlap once both are drawn — checked directly against a frame from the
already-shipped `scenes_task9_preview.py` render
(`Task9ScalingLawScene.mp4` at `t=4.0s`), which shows the *identical*
title/setup overlap. This is a standing convention shared by every
preview scene in this series (never called out as a defect in any prior
entry either), not something introduced here, so it wasn't changed —
consistency with five previous scenes' accepted layout outweighs fixing
a purely cosmetic overlap in just this one.

## Outcome
A real ~65.5-second video (39 animations) rendered successfully on the
second attempt (the first attempt's part 4 used un-tuned LR-schedule
defaults that made the resumed run's loss improvement too small to be a
convincing demonstration — a presentation issue, not a correctness bug;
fixed by setting `lr=1e-2, warmup_steps=5` to match the same
already-tested tuning `tests/test_trainer_integration.py` uses for
exactly this reason). Shows task 10's training-loop mechanics — the
warmup+cosine learning-rate schedule, the embedding-init bug's real
before/after logit and loss numbers, AdamW's real per-parameter momentum/
variance state, and checkpoint save + resume proven via real JSON-lines
log counts — using the real, tested `Trainer`/`QwenMoEModel` code
directly, with no fabricated numbers anywhere in the scene.
