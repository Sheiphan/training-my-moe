# Token Journey — Deep Matrix-Math Dive: Block-by-Block Plan

## Goal
Extend `site/token-journey/` so that, inside any step that involves a real
matrix operation, you can click **"Zoom into the math"** and see the
actual numbers: input matrix/vector as a grid of real cells, the real
learned weight matrix as a grid of real cells, and an animated
multiply-accumulate sweep that fills in each output cell one at a time,
derived directly from the formula shown above it. This is additive — the
existing walkthrough (camera zoom, narrative, KaTeX formula, real
numbers panel) stays exactly as-is; "zoom into the math" is a new, optional
expansion per step.

## Is this inference or training?
**Inference.** One real forward pass, one real token, exactly like today.
Training would need a completely different data model (loss over time,
gradients, optimizer state) — this repo already covers that separately
(run-level JSON-lines logs under `runs/`), so it's out
of scope here.

## What research says about *how*
Re-confirmed from last session's fetch of poloclub.github.io/transformer-explainer:
that project renders exactly this — real weight/activation matrices as
grids of cells with a "zoom in" affordance, built with **D3.js** (already
vendored in `site/shared/lib/`). No new library, framework, or MCP is
needed. The technique itself (highlight row *i* of A + column *j* of B,
animate an accumulator, fill result cell `(i,j)`, sweep to the next cell)
is a standard, well-established visualization pattern — implemented as
one reusable component, not 17 bespoke ones (see below).

## Two new reusable viz primitives (cover every block below)
1. **`MatMul` viz** — given real matrices A (n×k) and B (k×m): renders both
   as HTML/CSS grids of real-numbered cells (large dimensions windowed to
   an honest "first 6 of 64" corner, never silently truncated), a formula
   line (`c_ij = Σ a_ik·b_kj`) with real numbers substituted in, and a
   Play/Step control that sweeps every output cell of C, highlighting the
   contributing row/column and animating the running sum.
2. **`Elementwise` viz** — one or two input grids transforming into an
   output grid via a labeled per-cell op (add, multiply, SiLU, RMSNorm's
   square→mean→rsqrt→scale, softmax's exp→normalize, RoPE's per-pair
   rotate). Simpler than MatMul: no accumulator, just an animated
   cell-by-cell transform.

Both render into the existing narrative card's real-numbers area (plain
HTML/CSS grid, not squeezed into the SVG diagram) — reuses the panel
that's already working.

## New real data required (block 0 only — the one block shown in detail)
`scripts/export_trace.py` currently exports activations only. Needs a
small, additive extension (new `weights` key, same file), loading the real
model object the same way `qwenmoe/viz/common.py::capture_trace` does
(per repo memory: "a local `_load` helper that duplicates `capture_trace`'s
body but also returns the model") so real `nn.Linear`/`RMSNorm` weights can
be read directly — never re-derived or guessed:
- `attn_norm_gamma`, `moe_norm_gamma`, `final_norm_gamma` (RMSNorm's real γ
  — today's version can't show RMSNorm's exact output because this is
  missing)
- `attn_norm_output_target`, `moe_norm_output_target` (the real
  `RMSNorm(x)` for our followed token — the actual input to Wq/Wk/Wv and
  to the router, currently not captured anywhere)
- `w_q`, `w_k`, `w_v`, `w_o` (block 0 attention)
- `router_w` (block 0 router — small, 64×4, shown in full, no windowing)
- per-expert `w_gate`/`w_up`/`w_down`, but **only for the 2 real chosen
  experts + the shared expert** (not all 4 routed ones — the 2 unchosen
  experts genuinely never ran for this token, so exporting their weights
  would be misleading busywork, not just wasted space)
- `embedding_full` (65×64 — small; reused tied for the LM-head step)

Estimated size: still well under ~1.5MB total (a few hundred thousand
floats) — fine for a local, no-server file, no network transfer involved.

## Block-by-block checklist
Existing step id -> operation type -> viz component -> new data needed.

- [x] **embed** — row lookup (not a matmul) — new `RowLookup` mini-viz:
      show the 6 real prompt-token embedding rows (already have these,
      `bundle.embeddings`, zero new export), highlight + "lift out" our
      token's row. Lightest block, good first proof-of-concept.
- [x] **attn-norm** — RMSNorm — `Elementwise` (square, mean, +ε, rsqrt,
      ×γ), needs `attn_norm_gamma` (new).
- [x] **attn-qkv** — 3× `MatMul` (x·Wq, x·Wk, x·Wv), needs
      `attn_norm_output_target` + `w_q/w_k/w_v` (new). **Build this one
      first as the concrete demonstration** — cleanest textbook matmul.
- [x] **attn-rope** — `Elementwise` pairwise rotate, using already-exported
      real angles + q/k (no new export). Verified 0.0 max diff against
      the real `RotaryEmbedding` module before wiring it up.
- [x] **attn-group** — light "duplicate this row" animation, not a full
      MatMul viz (no new export).
- [x] **attn-sdpa** — `MatMul` (Q·Kᵀ, already-exported q/k, no new export)
      chained into `Elementwise` softmax, chained into `MatMul`
      (probs·V, already-exported, no new export).
- [x] **attn-out** — `MatMul` (concat-heads · Wo), needs `w_o` (new); the
      "concat-heads" input is computed client-side from real
      `attn_scores_post_softmax` × real `v` (exact, no missing weights).
- [x] **attn-plus** — `Elementwise` add (already have both real vectors).
- [x] **moe-norm** — `Elementwise` RMSNorm, needs `moe_norm_gamma` (new).
- [x] **moe-router** — `MatMul`, small enough to show in FULL (64×4), needs
      `moe_norm_output_target` + `router_w` (new).
- [x] **moe-topk** — selection, reuse/extend the existing chip list (no new
      viz primitive needed).
- [x] **moe-experts** — `MatMul`(gate) -> `Elementwise`(SiLU) -> chained
      with `MatMul`(up) via `Elementwise`(multiply) -> `MatMul`(down), for
      each of the 2 real chosen experts + the shared expert. Needs
      per-expert `w_gate/w_up/w_down` (new, chosen+shared only).
- [x] **moe-combine** — `Elementwise` weighted sum (already have topk
      weights + expert outputs from the previous block's computation).
- [x] **moe-plus** — `Elementwise` add (no new export).
- [x] **repeat** — conceptual only, intentionally no deep-zoom.
- [x] **final-norm** — `Elementwise` RMSNorm, needs `final_norm_gamma`
      (new) — upgrades today's "real output, but can't show *how*" into a
      fully real, fully derived computation.
- [x] **lm-head** — `MatMul`, windowed to the columns for the top-6
      predicted tokens (a meaningful window, not an arbitrary first-N),
      needs `embedding_full` (new, tied).
- [x] **output** — `Elementwise` softmax (already have logits).

## Build order (this session)
1. Extend `scripts/export_trace.py` with the `weights` block above; re-run
   it; spot-check shapes/values.
2. Build the `MatMul` and `Elementwise` D3/HTML viz primitives
   (`site/token-journey/js/matrix-viz.js`).
3. Wire ONE full block end-to-end as a concrete demo: **attn-qkv** (a
   textbook `x · W = out` matmul) — verify in-browser, screenshot, confirm
   this matches what you want.
4. Pause for your confirmation before repeating the same pattern across
   the remaining ~15 blocks (cheap to replicate once the primitive +
   pattern is approved; expensive to redo 17 times if the approach needs
   adjusting).

## Completion status
All 17 computational steps now expose exact real-number math. The intro
and repeat slides are intentionally conceptual. The finished page was
verified by scripted desktop and mobile sweeps across all steps, all
expert tabs, and all matrix controls with zero console/KaTeX errors,
non-finite values, or page-level overflow.

---

> **Status note (entry 036, 2026-09-05).** This plan is complete and shipped. Since it was
> written the page grew from 4 variants to 7 (adding `BASELINE`, `BASELINE_V2` and `FINEGRAINED`),
> so the step count is now 19 or 20 depending on the architecture — a QK-norm model has an extra
> step. Traces are also lazy-loaded per variant from `data/manifest.js` rather than one `trace.js`.
> The two large modern variants ship activations but not per-expert weight matrices, so the expert
> matmul dive described here is hidden for them and the page says why.
