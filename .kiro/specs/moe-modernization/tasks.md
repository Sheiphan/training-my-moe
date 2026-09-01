# Implementation Plan

> Convention (parent Requirement 13.2, 13.6–13.9): a task is not "done"
> until a dated `journey_of_a_model/NNN-slug.md` beginner-explainer entry is
> added and indexed in `journey_of_a_model/README.md` — every term defined
> before use, ≥1 Mermaid diagram, a "Questions that might come to mind"
> section, the full derivation of any formula with a hand-worked numeric
> example, and a link to the source paper found via web search. Numbering
> continues the parent journal's sequence (017+).
>
> Invariant for every task (Requirement 13.1, 13.3): `uv run pytest -q`
> passes with a test count `>=` the pre-task count, and any changed
> numerical path has a test pinning "flag off ⇒ identical to today"
> within `1e-5`.

- [x] 1. Config scaffolding for every new axis
  - Add all `ModelConfig` fields from design.md (`router_scoring`,
    `router_z_loss_weight`, `aux_loss_weight`, `seq_balance_loss_weight`,
    `qk_norm`, `mtp_depth`, `mtp_lambda`, `init_std`) and all `TrainConfig` /
    `DataConfig` fields (`optimizer`, `betas`, `adam_eps`, `weight_decay`,
    `grad_clip_norm`, `muon_lr`, `muon_momentum`, `lr_schedule`,
    `min_lr_ratio`, `decay_fraction`, `precision`, `z_loss_weight`,
    `document_masking`, `val_split_mode`), each with its `__post_init__`
    validation.
  - No behavior wired yet — every field is inert until its task below.
  - Extend `tests/test_config.py`: each new field's default constructs; each
    invalid value raises `ValueError`.
  - _Requirements: 1, 2, 3, 4, 6, 7, 9, 10, 13.2_

- [x] 2. Optimizer pure-wins: param groups, tuned AdamW, gradient clipping
  - New `qwenmoe/train/optim.py::split_param_groups(model)` and
    `build_optimizer(model, train_config)` (AdamW branch only for now: two
    param groups, `betas`/`eps`/`weight_decay` from config).
  - `Trainer.fit` uses `build_optimizer`; `_forward_backward_step` clips the
    global grad norm to `grad_clip_norm` and returns it; step-log records
    gain `grad_norm` (additive key).
  - Journal the default change (`betas (0.9, 0.999)→(0.9, 0.95)`,
    `wd 0.01→0.1`, no decay on norms/embeddings) with the AdamW update rule
    derivation.
  - Tests: no-decay group excludes every 1-D param and the embedding; grad
    norm is logged and finite; the integration sequence still converges and
    final loss is within a loose band of the pre-change run.
  - _Requirements: 1.1, 1.2, 1.3, 13.1, 13.3_

- [x] 3. Learning-rate schedule: min-LR floor + WSD
  - `lr_multiplier(step, config)` (or a `Schedule` object) branches
    `cosine` vs `wsd`, reads `min_lr_ratio` and `decay_fraction`, stays a
    pure function returning `[0, 1]`.
  - Update `test_trainer_integration.py`'s `lr_multiplier` tests to the new
    signature; add WSD tests (flat stable phase; hits `min_lr_ratio` at the
    end; warmup unchanged).
  - Journal: the WSD curve, why it decouples the token budget, the
    cooldown-branch idea, with a worked multiplier calculation.
  - _Requirements: 2.1, 2.2, 2.3, 2.4_

- [x] 4. Muon optimizer
  - `qwenmoe/train/optim.py`: `zeropower_via_newtonschulz5`, `Muon`
    (`torch.optim.Optimizer`), `CombinedOptimizer` (fan-out
    `step`/`zero_grad`/`state_dict`/`load_state_dict`, concatenated
    `param_groups`). `build_optimizer` grows a `muon` branch.
  - `Trainer` resume path restores a `CombinedOptimizer`.
  - Tests: the orthogonalized update has singular values ≈ 1; a `muon` run
    on the integration sequence converges; checkpoint save/resume round-trips
    for `optimizer="muon"`.
  - Journal: matrix-vs-scalar optimization, the Newton–Schulz iteration
    derived, a 2×2 worked example, links to the Muon reports.
  - _Requirements: 1.4, 1.5, 1.6_

- [ ] 5. Attention: fused SDPA on the non-trace path
  - `Attention.forward` and `MLAAttention.forward` call
    `F.scaled_dot_product_attention(is_causal=True, enable_gqa=True)`;
    drop the explicit `-inf` mask and `repeat_interleave`. ALiBi passes its
    bias as an additive `attn_mask`.
  - `forward_with_trace` keeps the explicit math **plus** nothing else
    changes; add a test asserting `forward` vs `forward_with_trace` agree
    within `1e-5` for `rope`/`nope`/`alibi`.
  - Journal: what SDPA fuses, why the explicit path stays for teaching.
  - _Requirements: 7.1, 7.2, 7.4, 7.5, 13.3_

- [ ] 6. Attention: QK-norm
  - `Attention` gains `q_norm`/`k_norm` (`RMSNorm(head_dim)`) applied per
    head before the positional transform, in both `forward` and
    `forward_with_trace`, gated on `ModelConfig.qk_norm`.
  - Tests: `qk_norm=False` ⇒ output bit-comparable to today; `qk_norm=True`
    ⇒ output changes and attention logits have bounded variance; config test.
  - Journal: attention-logit growth, why RMSNorm on q/k fixes it, Qwen3/
    Gemma link, a worked before/after logit-scale example.
  - _Requirements: 7.3, 7.5_

- [ ] 7. Principled initialization
  - `QwenMoEModel._init_weights` via `self.apply`: truncated-normal linears
    at `init_std`; residual-path projections (`w_o`, expert down-proj) scaled
    by `1/sqrt(2·n_layers)`; embedding unchanged; tied head skipped.
  - Tests: untrained-logit std is `O(1)` for `SMOKE_TEST`; a documented
    tolerance note vs the pre-change init.
  - Journal: init variance through a residual stream, the `1/sqrt(2N)`
    derivation, a worked 2-layer example, GPT-2 link.
  - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5_

- [ ] 8. Mixed precision + output z-loss
  - `_forward_backward_step`: optional `torch.autocast(bfloat16)`; loss and
    z-loss in `float32`; add `z_loss_weight × mean(logsumexp(logits)²)`;
    step-log gains `z_loss`.
  - Startup log line states effective precision.
  - Tests: `precision="bf16"` run stays finite and converges; `z_loss=0` ⇒
    loss identical to today; z-loss term is logged and non-negative.
  - Journal: what bf16 buys and its range vs fp16, why the router stays
    fp32, the z-loss derivation (softmax normalizer control), PaLM/ST-MoE
    links.
  - _Requirements: 3.1, 3.2, 3.3, 3.4_

- [ ] 9. MoE routing: config-driven weights, router z-loss, sigmoid scoring
  - `AuxLossBalancer` reads `aux_loss_weight`; add the optional router
    z-loss term. New `score_and_select(...)` handling
    `router_scoring in {"softmax", "sigmoid"}`; wire through `MoELayer`
    and `forward_with_trace`.
  - Tests: sigmoid path selects exactly `k`, weights sum to 1, zero-token
    expert still skipped; `softmax` + `router_z_loss_weight=0` ⇒ identical to
    today; router z-loss responds to induced logit blow-up.
  - Journal: softmax vs normalized-sigmoid gating (identifiability, gradient
    flow), router z-loss, DeepSeek-V3 link, worked routing example.
  - _Requirements: 4.1, 4.2, 4.3, 4.5, 4.6_

- [ ] 10. MoE routing: loss-free complementary sequence-balance loss
  - `LossFreeBalancer.sequence_balance_loss(...)`, added at
    `seq_balance_loss_weight` when `> 0`; surfaced in the aux-loss total and
    the step log.
  - Tests: weight `0` ⇒ loss-free behavior identical to today; a single
    imbalanced sequence produces a positive penalty; batch-balanced but
    sequence-imbalanced input is still penalized.
  - Journal: why "auxiliary-loss-free" still carries a tiny loss, the
    per-sequence vs per-batch distinction, DeepSeek-V3 §2.1.2, worked
    example.
  - _Requirements: 4.4, 4.6_

- [ ] 11. MoE: batched expert computation + checkpoint migration
  - `RoutedExperts` module holding expert weights as 3-D `nn.Parameter`s;
    `forward(x_flat, topk_indices, topk_weights)` via argsort +
    batched/grouped matmul + scatter-add. Shared experts stay a
    `nn.ModuleList`. `load_state_dict` pre-hook remaps parent-spec keys
    (`routed_experts.N.w_gate.weight` → new layout).
  - `forward_with_trace` keeps the explicit per-expert loop.
  - Tests: batched output matches the loop within `1e-5` incl. a zero-token
    expert; a parent-spec `MoELayer` state-dict loads via the hook; a
    32-expert config trains a few steps within a small constant factor of
    the 8-expert wall-clock.
  - Journal: why the naïve loop is slow, the sorted-dispatch / grouped-GEMM
    idea, MegaBlocks link, a small dispatch worked by hand.
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

- [ ] 12. Multi-Token Prediction head
  - `qwenmoe/model/mtp.py::MTPHead`; `QwenMoEModel` wires `mtp_depth` heads,
    training-only, returning `extras["mtp_loss"]`; `Trainer` adds
    `mtp_lambda × mtp_loss`; step-log gains per-depth losses.
  - `forward`/`generate` signatures unchanged when `mtp_depth == 0` or at
    inference.
  - Tests: `mtp_depth=0` is a structural no-op; `mtp_depth=1` adds one loss
    term and extra params that round-trip through a checkpoint; `generate()`
    works unchanged on an MTP model.
  - Journal: teacher-forcing myopia, how MTP densifies the signal, the
    causal-chain head design, DeepSeek-V3 / Gloeckle links, worked 2-token
    example.
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

- [ ] 13. KV cache for inference
  - `qwenmoe/model/kv_cache.py` (`LayerKVCache`, `LatentKVCache`,
    `build_kv_cache`); `Attention.forward`/`MLAAttention.forward` accept an
    optional cache; `generate()` gains `use_kv_cache=True` with prefill +
    incremental steps.
  - Tests: cached vs uncached generation produce identical tokens from the
    same seed/prompt; measured retained bytes for MLA < GQA at matched
    shape; `max_seq_len` stop still holds.
  - Journal: the KV-cache concept, O(n²)→O(n) per token, GQA vs MLA cache
    shapes, DeepSeek-V2 link, a worked cache-growth table.
  - _Requirements: 8.1, 8.2, 8.3, 8.4_

- [ ] 14. Inference sampling: min-p + repetition penalty
  - `filter_min_p`, `apply_repetition_penalty` in `generate.py`, mirroring
    the existing filters; new `generate()` params compose with
    `top_k`/`top_p`.
  - Tests: min-p never masks all tokens; neutral params ⇒ identical to
    today; repetition penalty lowers a repeated token's probability.
  - Journal: min-p's adaptive truncation vs top-p, the repetition-loop the
    parent journal hit, min-p paper link, worked distribution example.
  - _Requirements: 11.1, 11.2, 11.3, 11.4_

- [ ] 15. Data pipeline: document boundaries
  - BPE `<eos>`/separator id; `prepare_data.py` joins docs with it and
    records `eos_id`. `TokenDataset(return_segments=True)`;
    `build_document_mask`; `Attention` combines it with the causal mask when
    `document_masking`. `train_val_split(mode="documents", eos_id=...)`.
  - Tests: `document_masking=False` ⇒ identical to today; with it on, a
    token can't attend across a separator; document-mode val split contains
    only whole documents.
  - Journal: cross-document contamination, sequence packing without
    contamination, Llama-3 doc-mask, a worked mask example.
  - _Requirements: 10.1, 10.2, 10.3, 10.4_

- [ ] 16. Ablation methodology: multi-seed, param-matched, mean ± std
  - `run_comparison(..., seeds=(1337,))` loops seeds; extended
    `VariantResult`; summary shows `mean ± std` and a `within_noise` flag.
  - `param_matched` mode bisects `d_ff` to match baseline total params.
  - Tests: `len(seeds) > 1` ⇒ `std > 0` and one log per seed; `param_matched`
    brings a GELU variant's param count within 2% of baseline.
  - Journal: why single-seed ablations mislead, pooled-std noise bands,
    compute- vs param-matched comparisons.
  - _Requirements: 12.1, 12.2, 12.4_

- [ ] 17. Ablation methodology: refreshed menu, MoE budget note, TinyStories default
  - Extend `scripts/run_ablation_study.py::VARIANTS` with `OPT_MUON`,
    `ROUTE_SIGMOID`, `ATT_QKNORM`, `SCHED_WSD`, `MTP_D1`, `MOE_FINEGRAINED`.
    Default corpus → TinyStories BPE (parent Tier 2); keep a
    `--smoke` flag for the fast Shakespeare run.
  - `scaling.py::moe_budget_note(total, active)`; call it in the training
    CLIs' startup summary.
  - Tests: `moe_budget_note` arithmetic and wording; each new preset
    constructs and trains a few steps.
  - Journal: the 2026 axis menu and why NoPE/ALiBi/GEGLU are settled
    questions; MoE vs dense compute-optimal ratios (Clark et al.).
  - _Requirements: 12.3, 12.5, 12.6_

- [ ] 18. Modernized presets
  - `configs/presets.py`: a `BASELINE_V2` (QK-norm on, sigmoid routing,
    WSD-friendly, `init_std` scaled) documented as "the recommended config
    after this spec", and `FINEGRAINED` (e.g. 32 experts, `d_ff≈128`,
    top-8, 1 shared). Keep `BASELINE`/`SMOKE_TEST` untouched for parent-spec
    tests.
  - Journal: a short decision entry — what `BASELINE_V2` changes vs
    `BASELINE` and the evidence from tasks 2–17 for each.
  - _Requirements: 12.5, 13.4_
