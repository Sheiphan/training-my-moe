# Journey of a Model — Index

A chronological lab notebook for this project: every idea considered, every
decision made, and every completed implementation task gets an entry here,
in order. See `.kiro/specs/qwen-moe-from-scratch/` for the full
requirements/design/tasks documents this journal traces back to. Numbered
entries correspond 1:1 to task numbers in `tasks.md` (`003` = task 3, etc.);
entry `001` covers spec approval plus task 1, since task 1 was literally
about setting this journal up.

## Documentation standard (Requirement 13.6-13.9)

Every entry that introduces a technical concept follows this bar — it's not
optional, and it's checked on every task, not just re-stated per entry:

1. **Beginner explainer per term:** before relying on any technical term,
   define it — what it means in plain English, why it exists / what
   problem it solves, a real-world analogy, a simple technical explanation,
   and how it relates to this project. Never define one unexplained term
   using another.
2. **At least one Mermaid diagram** per concept.
3. **A "Questions that might come to mind" section** anticipating likely
   beginner questions.
4. **Links to the concept's original paper** (found via web search, never
   guessed) and, where a good one exists, a supplementary blog/explainer.
5. **The full derivation** of any formula involved — not just the final
   result — with exact math clearly distinguished from any
   empirically-fitted constant, **plus a worked numeric example**: concrete
   dummy values run through the formula step by step by hand.

Pure process/decision entries with no new technical concept (like the one
that introduced this standard) can stay short — there's no requirement to
force a diagram or derivation where there's nothing to teach.

*Idea considered and rejected: giving this standard-upgrade decision its
own numbered entry (it briefly existed as `004-doc-standard-upgrade.md`).
Since numbered entries are meant to track task numbers 1:1, and this change
happened between tasks 3 and 4 rather than being task 4 itself, its content
was folded into this README instead — keeping `004` free for the actual
task 4 entry.*

## Index

| # | Date | Entry | Summary |
|---|------|-------|---------|
| 001 | 2026-08-31 | [001-spec-approved.md](001-spec-approved.md) | Requirements, design, and tasks approved; key decisions going into implementation. |
| 002 | 2026-08-31 | [002-config-layer.md](002-config-layer.md) | `ModelConfig`/`TrainConfig`/`DataConfig` dataclasses + `BASELINE` preset + validation tests (task 2). |
| 003 | 2026-08-31 | [003-norm-and-positional.md](003-norm-and-positional.md) | RMSNorm + RoPE/NoPE/ALiBi positional schemes, sharing one hook interface (task 3). |
| 004 | 2026-08-31 | [004-attention-gqa.md](004-attention-gqa.md) | Scaled dot-product attention + Grouped-Query Attention baseline; found & fixed an `nn.Module.apply` name collision from task 3 (task 4). |
| 005 | 2026-08-31 | [005-moe-layer.md](005-moe-layer.md) | SwiGLU/GEGLU/GELU expert variants, router + top-k gating, shared experts, and the Switch-Transformer-style load-balancing auxiliary loss (task 5). |
| 006 | 2026-08-31 | [006-full-model-assembly.md](006-full-model-assembly.md) | `TransformerBlock` (pre-norm attn + pre-norm MoE with residuals) and `QwenMoEModel` (embedding, stacked blocks, final norm, tied/untied LM head) (task 6). |
| 007 | 2026-08-31 | [007-tokenizers.md](007-tokenizers.md) | `CharTokenizer` (sorted-unique-character vocab) and `ByteBPETokenizer` (from-scratch byte-level BPE merge learning), sharing a `Tokenizer` protocol plus a `build_tokenizer` factory (task 7). |
| 008 | 2026-08-31 | [008-data-pipeline.md](008-data-pipeline.md) | `download_corpus` (tiny Shakespeare via HTTPS, TinyStories via `huggingface_hub`), `TokenDataset` sliding-window `(x, y)` pairs, `train_val_split`, and `make_fixed_slice` for compute-matched ablation comparisons (task 8). |
| 009 | 2026-08-31 | [009-scaling-law-utilities.md](009-scaling-law-utilities.md) | `count_params` (total/active `ParamBreakdown`), `chinchilla_budget`, `calibrate_throughput` (real measured tokens/sec), and `epochs_for_budget` (>4-epoch repetition warning) (task 9). |
| 010 | 2026-08-31 | [010-training-loop.md](010-training-loop.md) | `Trainer` (AdamW + warmup/cosine schedule, checkpoint save/resume, JSON-lines logging) and the `prepare_data.py`/`train.py` CLIs; found & fixed an embedding-init bug from task 6 that was blowing up untrained-model loss, and an unbounded-validation performance bug (task 10). |
| 011 | 2026-08-31 | [011-inference-generation.md](011-inference-generation.md) | `generate()` (autoregressive decoding with temperature/top-k/top-p sampling and a deterministic greedy mode) and the `scripts/generate.py` CLI; manual run against the real task-10 checkpoint reproduced a genuine "neural text degeneration" repetition loop under greedy decoding, fixed by nucleus sampling (task 11). |
| 012 | 2026-08-31 | [012-visualization-foundation.md](012-visualization-foundation.md) | `TraceBundle`/`LayerTrace` dataclasses and a `forward_with_trace` method mirrored across `Attention`/`MoELayer`/`TransformerBlock`/`QwenMoEModel`, plus `qwenmoe/viz/common.py::capture_trace(checkpoint_path, prompt)` — the real-numbers-only foundation the task 13-14 visualizations built on (task 12). |
| 013 | 2026-08-31 | *(entry removed)* | Five formal Manim scenes (embedding+RoPE, attention, MoE routing, RMSNorm+SwiGLU, output projection) animating real numbers captured via task 12's `capture_trace` pipeline, plus a rendering CLI (task 13). **All Manim scene code and this entry's doc were later deleted** — Manim was removed from the project entirely. |
| 014 | 2026-08-31 | *(entry removed)* | `LossCurveScene` and `ExpertUtilizationScene`, reading the real `Trainer` JSON-lines log directly (no `TraceBundle`); surfaced a real per-layer load-balancing asymmetry in the smoke-test run (task 14). **All Manim scene code and this entry's doc were later deleted** — Manim was removed from the project entirely. |
| 015 | 2026-08-31 | [015-research-comparison-framework.md](015-research-comparison-framework.md) | `LossFreeBalancer` (DeepSeek-V3 auxiliary-loss-free, bias-adjusted MoE balancing), `run_comparison`/`ComparisonReport` (trains named config variants on one identical fixed token slice + budget), and the `scripts/run_comparison.py` CLI + 5 new ablation presets; a real comparison run's saved checkpoint showed the learned per-layer bias echoing task 14's earlier per-layer imbalance-asymmetry finding (task 15). |
| 016 | 2026-08-31 | [016-mla-attention.md](016-mla-attention.md) | `MLAAttention` (simplified DeepSeek-V2/V3 Multi-Head Latent Attention: low-rank joint KV compression into a shared latent, plus a small decoupled-RoPE sub-dimension), selectable via `ModelConfig.attention_variant="mla"` as a drop-in alternative to GQA; real measured KV-cache savings against this project's own SMOKE_TEST/BASELINE presets (task 16, stretch). |

## Supplementary entries (not tied to a task number)

This project previously had a series of standalone Manim-video "preview" entries here
(tasks 3/5/6/7/8/9/10/11/15). **All Manim code and these docs were later deleted** — Manim
was removed from the project entirely, and nothing in this section remains.
