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
| 012 | 2026-08-31 | [012-visualization-foundation.md](012-visualization-foundation.md) | `TraceBundle`/`LayerTrace` dataclasses and a `forward_with_trace` method mirrored across `Attention`/`MoELayer`/`TransformerBlock`/`QwenMoEModel`, plus `qwenmoe/viz/common.py::capture_trace(checkpoint_path, prompt)` — the real-numbers-only foundation every Manim scene in tasks 13-14 will build on (task 12). |
| 013 | 2026-08-31 | [013-architecture-manim-scenes.md](013-architecture-manim-scenes.md) | Five formal Manim scenes (embedding+RoPE, attention, MoE routing, RMSNorm+SwiGLU, output projection), each animating real numbers captured from the real task-10 checkpoint via task 12's `capture_trace` pipeline, plus the `scripts/render_viz.py` documentation/rendering CLI (task 13). |
| 014 | 2026-08-31 | [014-training-observability-scenes.md](014-training-observability-scenes.md) | `LossCurveScene` and `ExpertUtilizationScene`, reading the real `Trainer` JSON-lines log directly (no `TraceBundle`), plus a `load_training_log` helper that fails with a clear, actionable error before any Manim rendering if the log is missing; surfaced a real per-layer load-balancing asymmetry in the smoke-test run (task 14). |
| 015 | 2026-08-31 | [015-research-comparison-framework.md](015-research-comparison-framework.md) | `LossFreeBalancer` (DeepSeek-V3 auxiliary-loss-free, bias-adjusted MoE balancing), `run_comparison`/`ComparisonReport` (trains named config variants on one identical fixed token slice + budget), and the `scripts/run_comparison.py` CLI + 5 new ablation presets; a real comparison run's saved checkpoint showed the learned per-layer bias echoing task 14's earlier per-layer imbalance-asymmetry finding (task 15). |
| 016 | 2026-08-31 | [016-mla-attention.md](016-mla-attention.md) | `MLAAttention` (simplified DeepSeek-V2/V3 Multi-Head Latent Attention: low-rank joint KV compression into a shared latent, plus a small decoupled-RoPE sub-dimension), selectable via `ModelConfig.attention_variant="mla"` as a drop-in alternative to GQA; real measured KV-cache savings against this project's own SMOKE_TEST/BASELINE presets (task 16, stretch). |
| 017 | 2026-09-01 | [017-moe-modernization-spec.md](017-moe-modernization-spec.md) | New spec `.kiro/specs/moe-modernization/` approved on branch `moe-modernization`: brings the parent spec's architecture up to 2024-2026 published practice (Muon, DeepSeek-V3 routing, QK-norm, KV cache, MTP, multi-seed ablations) as pure-win / opt-in-axis / methodology-upgrade changes, staged so the existing 213 tests keep passing throughout. |
| 018 | 2026-09-01 | [018-optimizer-hardening.md](018-optimizer-hardening.md) | `qwenmoe/train/optim.py` (`split_param_groups`, `build_optimizer`): weight-decay/no-decay parameter groups (norms + embeddings excluded), tuned AdamW `betas=(0.9, 0.95)`/`weight_decay=0.1`, and gradient-norm clipping with the pre-clip norm logged per step (moe-modernization task 2). |
| 019 | 2026-09-01 | [019-lr-schedule-wsd.md](019-lr-schedule-wsd.md) | `lr_multiplier` gains a configurable decay floor (`min_lr_ratio`, default 0.1) and an opt-in Warmup-Stable-Decay schedule alongside cosine, as a pure function unifying both curves (moe-modernization task 3). |
| 020 | 2026-09-02 | [020-muon-optimizer.md](020-muon-optimizer.md) | `Muon`/`zeropower_via_newtonschulz5`/`CombinedOptimizer` in `qwenmoe/train/optim.py`: orthogonalized-momentum optimization for 2-D hidden weight matrices, AdamW for everything else, one shared LR schedule driving two different peak rates; real measured muon/adamw parameter-routing counts and a real Newton-Schulz 2x2 worked example verified against the actual code (moe-modernization task 4). |
| 021 | 2026-09-02 | [021-fused-attention-sdpa.md](021-fused-attention-sdpa.md) | `Attention.forward`/`MLAAttention.forward` switch to fused `F.scaled_dot_product_attention` (`enable_gqa=True`, ALiBi's bias folded into an explicit `attn_mask`); `forward_with_trace` keeps the explicit math for the Manim scenes, now agreeing with `forward` within `1e-5` instead of bit-for-bit; real measured ~1.7x CPU speedup and ~1e-7 numerical agreement (moe-modernization task 5). |
| 022 | 2026-09-02 | [022-qk-norm.md](022-qk-norm.md) | `Attention` gains optional per-head `RMSNorm` on q/k before the positional transform (`ModelConfig.qk_norm`, default off, submodules not even constructed when off); the Cauchy-Schwarz attention-logit-growth derivation, RMSNorm's exact scale-invariance, and a real measured 47,000x logit-std blowup (off) vs a flat ~1.0 (on) across a 200x input scale-up (moe-modernization task 6). |
| 023 | 2026-09-02 | [023-principled-initialization.md](023-principled-initialization.md) | `QwenMoEModel` replaces PyTorch's default Kaiming-uniform with truncated-normal init at `ModelConfig.init_std`, then scales residual-path output projections (`w_o`/`w_down`/`w_2`) by `1/sqrt(2·n_layers)` (GPT-2 rule); the variance-of-a-sum derivation, a worked example, and a real measured 2.4x vs 9.1x residual-stream growth comparison over 6 blocks (moe-modernization task 7). |
| 024 | 2026-09-02 | [024-mixed-precision-and-z-loss.md](024-mixed-precision-and-z-loss.md) | `_forward_backward_step` gains optional `torch.autocast(bf16)` (no gradient scaler — bf16 has fp32's exponent range) and an optional output z-loss `z_loss_weight · mean(logsumexp(logits)²)`, both defaulting off; the fp32/fp16/bf16 bit-layout table, autocast's fp32 op-policy (measured), the cross-entropy shift-invariance derivation that motivates the z-loss, and a real 60-step fp32-vs-bf16 loss-curve comparison (moe-modernization task 8). |
| 025 | 2026-09-02 | [025-moe-routing-sigmoid-and-router-z-loss.md](025-moe-routing-sigmoid-and-router-z-loss.md) | `score_and_select` adds DeepSeek-V3 sigmoid routing (independent per-expert affinities, normalized selected for the gate) alongside softmax; `MoELayer._add_router_z_loss` adds the ST-MoE router z-loss (balancer-independent); `aux_loss_weight` becomes config-driven — all three default to the pre-spec path; softmax-vs-sigmoid identifiability and gate-sharpness, a worked 4-expert example, and measured selection-disagreement + expert-balance numbers (moe-modernization task 9). |
| 026 | 2026-09-02 | [026-per-sequence-balance-loss.md](026-per-sequence-balance-loss.md) | `LossFreeBalancer.sequence_balance_loss` adds DeepSeek-V3's complementary per-sequence `Σ fᵢPᵢ` term (folded into the aux total via `MoELayer._add_seq_balance_loss`, loss-free-only, weight defaults 0); why batch-aggregate balancing misses per-sequence routing collapse, the rearrangement-inequality minimum, a worked 2-sequence example, and a measured 1.07× vs 3.61× sensitivity gap vs the batch-wide Switch loss (moe-modernization task 10). |
| 027 | 2026-09-02 | [027-batched-expert-computation.md](027-batched-expert-computation.md) | `dispatch_batched` replaces `MoELayer.forward`'s per-expert Python loop with sorted-dispatch + 3 batched `bmm`s (experts stay an `nn.ModuleList`, so no checkpoint migration and Muon is untouched); why kernel-launch overhead makes the loop O(n_experts), the padded-capacity-grid idea worked by hand, an idle expert keeping `grad=None`, and measured wall-clock (batched stays ~flat 8→64 experts while the loop climbs) + 6e-8 numerical agreement (moe-modernization task 11). |
| 028 | 2026-09-02 | [028-multi-token-prediction.md](028-multi-token-prediction.md) | `qwenmoe/model/mtp.py::MTPHead` + `QwenMoEModel.forward(input_ids, targets=None)`: optional training-only heads predicting tokens at offset +2, +3… via DeepSeek-V3's causal chain (`mtp_depth` empty ModuleList = byte-identical no-op; skipped at inference so `generate()` is untouched); teacher-forcing myopia, the slice-based alignment, a worked 2-token example, Gloeckle-vs-DeepSeek head designs, and measured +4.8%/depth param cost + per-depth loss curves (moe-modernization task 12). |

## Supplementary entries (not tied to a task number)

| Entry | Summary |
|-------|---------|
| [extra-manim-preview-task3.md](extra-manim-preview-task3.md) | A standalone Manim video previewing Task 3's math (RMSNorm/RoPE/ALiBi) using the real, tested classes directly — ahead of the formal `TraceBundle`-based Task 13 pipeline. |
| [extra-manim-preview-task5.md](extra-manim-preview-task5.md) | A standalone Manim video previewing Task 5's math (router top-k gating + renormalization, dispatch/combine with a skipped zero-token expert, load-balancing auxiliary loss) using the real, tested classes directly. |
| [extra-manim-preview-task6.md](extra-manim-preview-task6.md) | A standalone Manim video previewing Task 6's math (pre-norm residual wiring with real per-stage norms, empirical residual-stream growth across a real 6-layer model, weight tying via the real tied `lm_head`) using the real, tested classes directly. |
| [extra-manim-preview-task7.md](extra-manim-preview-task7.md) | A standalone Manim video previewing Task 7's math (`CharTokenizer`'s sorted-character vocab and out-of-vocabulary handling, `ByteBPETokenizer`'s greedy merge-pair training loop step by step, byte-level BPE's compression behavior and universal-coverage guarantee) using the real, tested classes directly. |
| [extra-manim-preview-task8.md](extra-manim-preview-task8.md) | A standalone Manim video previewing Task 8's math (`TokenDataset`'s sliding `(x, y)` windows, `train_val_split`'s contiguous split via two separately-built datasets, `make_fixed_slice`'s chunk-shuffle and its compute-matched-comparison motivation) using the real, tested functions directly. |
| [extra-manim-preview-task9.md](extra-manim-preview-task9.md) | A standalone Manim video previewing Task 9's math (`count_params`'s total-vs-active parameter breakdown, `chinchilla_budget`/`epochs_for_budget`'s token-budget planning, `calibrate_throughput`'s empirical device timing) using the real, tested functions directly — surfaced a genuine finding that CPU beats MPS for a tiny model, the reverse of the `BASELINE`-scale ranking. |
| [extra-manim-preview-task10.md](extra-manim-preview-task10.md) | A standalone Manim video previewing Task 10's training-loop mechanics (`lr_multiplier`'s warmup+cosine curve, the real embedding-init bug's before/after logits and loss, AdamW's real captured momentum/variance state, `Trainer` checkpoint save + resume proven via real JSON-lines log counts) using the real, tested `Trainer`/`QwenMoEModel` code directly. |
| [extra-manim-preview-task11.md](extra-manim-preview-task11.md) | A standalone Manim video previewing Task 11's sampling math (temperature's real sharpening/flattening effect on a softmax distribution, `filter_top_k`/`filter_top_p` called live and matching entry 011's worked examples, and real `generate()` calls against the real task-10 checkpoint reproducing the genuine greedy-decoding repetition loop side by side with nucleus sampling's fix) using the real, tested `qwenmoe.model.generate` code directly — the only visualization this content will ever get, since no later formal task touches generation/sampling. |
| [extra-manim-preview-task15.md](extra-manim-preview-task15.md) | A standalone Manim video previewing Task 15's math (`LossFreeBalancer`'s real 40-round bias-update loop, surfacing a genuine oscillating self-correction dynamic once the "winning" expert changes, and a real `run_comparison()` call training `aux_loss` vs. `loss_free` variants on the real tiny-Shakespeare corpus under an identical fixed slice/budget) using the real, tested `qwenmoe.model.moe`/`qwenmoe.train.compare` code directly — the only visualization this content will ever get, since no later formal task touches either. |
