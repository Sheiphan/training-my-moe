# 013 — Architecture Math Manim Scenes (Task 13)

**Type:** task · **Date:** 2026-08-31 · **Related:** Requirements 7.2-7.7;
design.md `viz/` Components and Interfaces; `qwenmoe/viz/scenes_embedding.py`,
`scenes_attention.py`, `scenes_moe.py`, `scenes_norm_act.py`,
`scenes_output.py`, `scripts/render_viz.py`

No new technical concepts are introduced here — RMSNorm, RoPE, GQA
attention, MoE routing, and SwiGLU were all already explained and derived
in entries 003-005. This entry only covers the new visualization work:
five formal Manim scenes, finally built on task 12's `TraceBundle`/
`capture_trace` pipeline against a real trained checkpoint, plus the
rendering CLI.

## What was implemented

Five Manim scenes, one per architectural component named in Requirement
7.2-7.6, each sourcing every number from one real forward pass over the
real `runs/smoke_test/checkpoints/step_1000.pt` checkpoint and the prompt
`"ROMEO:"` (via `capture_trace`/`forward_with_trace`) — never illustrative
or fabricated numbers (Requirement 7.1):

- **`scenes_embedding.py::EmbeddingPositionalScene`** (Requirement 7.2) —
  the token-to-vector embedding lookup (a handful of real table rows,
  highlighting the ones this prompt actually uses) and the RoPE rotation
  of a real query vector (head 3, position 1, dims (0, 8) — picked because
  `inv_freq[0] == 1.0` always, so the real angle there equals the position
  exactly, and this specific real vector's magnitude is large enough to
  see clearly), rotated by the real angle via the real
  `RotaryEmbedding.apply_to_qk`.
- **`scenes_attention.py::AttentionScene`** (Requirement 7.3) — real
  `Q.K^T / sqrt(head_dim)` scores for one interior query row (masking
  removes exactly the future 3 of 6 keys), softmax turning masked `-inf`
  into exact `0.0`, and `probs @ V` computed from the real captured
  tensors for the last (fully unmasked) row.
- **`scenes_moe.py::MoERoutingScene`** (Requirement 7.4) — router logits
  -> softmax -> top-k -> renormalized weights for one real token, then a
  6-token x 4-expert dispatch fan-out diagram built from every real
  token's real top-k choices at once (not one hand-built example like the
  task 5 preview).
- **`scenes_norm_act.py::NormActivationScene`** (Requirement 7.5) —
  RMSNorm's real scaling steps (`x -> x/RMS(x) -> *weight`, confirming
  `RMS(x_normed) == 1` on real numbers) and the gated SwiGLU activation
  inside a real routed expert, recomputed step by step (`gate`,
  `silu(gate)`, `up`, `gated`, `down`) and verified to match
  `LayerTrace.expert_outputs` bit-for-bit.
- **`scenes_output.py::OutputProjectionScene`** (Requirement 7.6) — final
  RMSNorm output -> LM-head logits -> softmax next-character
  probabilities, surfacing a real finding: after `"ROMEO:"` this
  checkpoint predicts a newline with `p=0.8828` (matching the corpus's
  `"SPEAKER:\ndialogue"` formatting).
- **`scripts/render_viz.py`** (Requirement 7.7): an argparse CLI whose
  `--help` documents the exact `manim -pql <file> <SceneName>` invocation
  for every scene, and which can also run any one (or all) of them via
  `subprocess`.
- Manually verified (13.6): every scene was rendered independently, each
  as its own fresh `manim` process, in no particular order, with no scene
  depending on any other having run first.

## Design decisions worth calling out

**Two scenes need the real model object, not just a `TraceBundle`.**
`scenes_embedding.py` and `scenes_norm_act.py` each load the checkpoint
(mirroring `capture_trace`'s own body) and keep the `QwenMoEModel` itself,
because they need real intermediate values `LayerTrace` doesn't carry:
`model.embedding.weight` (the full lookup table a token id indexes into)
and the per-block `attn_norm`/`moe_norm`/expert-internal
(`w_gate`/`w_up`/`w_down`) submodules the SwiGLU walkthrough exposes. The
other three scenes (`scenes_attention.py`, `scenes_moe.py`,
`scenes_output.py`) only need fields already on `TraceBundle`/
`LayerTrace`, so they call `capture_trace` alone.

**Every "extra" intermediate value not on `LayerTrace` was reconstructed
from already-captured tensors or real submodule calls, never
refabricated.** `x_mid` (block 0's post-attention residual stream, the
input to `moe_norm`) is exactly `bundle.embeddings +
bundle.layers[0].attn_output` — the identical operation
`TransformerBlock.forward` performs — and this was verified bit-for-bit
(`torch.allclose`) against actually stepping the real model before
committing to it. Likewise the SwiGLU walkthrough's final `down` value
was verified to match `LayerTrace.expert_outputs[expert_id]` exactly; the
scene displays this check live as an on-screen caption, so a future edit
that broke the equivalence would be visually obvious, not just silently
wrong.

**The MoE dispatch diagram is real across the whole prompt, an upgrade
over the task 5 preview.** `scenes_task5_preview.py` (built before a
trained checkpoint existed) had to hand-seed one token's router weights
to get a worked example. `scenes_moe.py` instead reads every one of the
real 6 tokens' real `topk_indices`/`topk_weights` from one real trace and
draws all of them fanning into 4 expert boxes at once — which surfaced a
genuine finding: this specific under-trained checkpoint routes *every
single token* to expert 3, a stark real example of the load imbalance the
auxiliary loss (entry 005) exists to correct.

**A cramped first draft of the SwiGLU walkthrough was caught by the
coarse contact-sheet check, not by reading the code.** The initial
version stacked 4 bar-chart rows (`gate`, `up`, `gated`, `down`) at fixed
y-offsets that looked reasonable on paper but didn't leave the ~2.6-unit
clearance each row's bars need (they can extend +-1.3 units from their
own baseline) — adjacent rows' bars visibly overlapped once rendered.
Fixed by showing at most two rows at a time and fading each pair out
before the next — the same "sequential reveal beats simultaneous
cramming" lesson task 11's preview already recorded, now confirmed again
on a different layout shape.

**`scripts/render_viz.py` both documents and (optionally) runs each
scene.** Requirement 7.7 only asks for `--help` to document the exact
invocation, but a thin `subprocess` wrapper costs little extra and gives
one consistent entry point (`--quality`/`--preview` flags shared across
all 5 scenes) instead of five different commands to remember.

## Outcome

- 160 tests still pass (this task added no new automated tests — tasks.md
  only asks for tests on 13.1-13.5's predecessors and successors, not on
  the Manim scenes themselves; 13.6 asks for manual verification instead,
  which was done as described above).
- All 5 scenes render cleanly at `-ql` quality (480p15); every real value
  shown was checked frame-by-frame via the project's established
  coarse-contact-sheet-then-settled-frame process before being accepted.
