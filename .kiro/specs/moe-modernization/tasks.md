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

- [x] 5. Attention: fused SDPA on the non-trace path
  - `Attention.forward` and `MLAAttention.forward` call
    `F.scaled_dot_product_attention(is_causal=True, enable_gqa=True)`;
    drop the explicit `-inf` mask and `repeat_interleave`. ALiBi passes its
    bias as an additive `attn_mask`.
  - `forward_with_trace` keeps the explicit math **plus** nothing else
    changes; add a test asserting `forward` vs `forward_with_trace` agree
    within `1e-5` for `rope`/`nope`/`alibi`.
  - Journal: what SDPA fuses, why the explicit path stays for teaching.
  - _Requirements: 7.1, 7.2, 7.4, 7.5, 13.3_

- [x] 6. Attention: QK-norm
  - `Attention` gains `q_norm`/`k_norm` (`RMSNorm(head_dim)`) applied per
    head before the positional transform, in both `forward` and
    `forward_with_trace`, gated on `ModelConfig.qk_norm`.
  - Tests: `qk_norm=False` ⇒ output bit-comparable to today; `qk_norm=True`
    ⇒ output changes and attention logits have bounded variance; config test.
  - Journal: attention-logit growth, why RMSNorm on q/k fixes it, Qwen3/
    Gemma link, a worked before/after logit-scale example.
  - _Requirements: 7.3, 7.5_

- [x] 7. Principled initialization
  - `QwenMoEModel._init_weights` via `self.apply`: truncated-normal linears
    at `init_std`; residual-path projections (`w_o`, expert down-proj) scaled
    by `1/sqrt(2·n_layers)`; embedding unchanged; tied head skipped.
  - Tests: untrained-logit std is `O(1)` for `SMOKE_TEST`; a documented
    tolerance note vs the pre-change init.
  - Journal: init variance through a residual stream, the `1/sqrt(2N)`
    derivation, a worked 2-layer example, GPT-2 link.
  - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5_

- [x] 8. Mixed precision + output z-loss
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

- [x] 9. MoE routing: config-driven weights, router z-loss, sigmoid scoring
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

- [x] 10. MoE routing: loss-free complementary sequence-balance loss
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

- [x] 11. MoE: batched expert computation
  - IMPLEMENTED via `design.md`'s *alternative* (not `RoutedExperts` 3-D
    params): `dispatch_batched(x_flat, topk_indices, topk_weights,
    routed_experts, activation)` — argsort dispatch pairs by expert, pad to
    a `(K, G, d_model)` capacity grid, 3 batched `bmm`s, mask + scatter-add.
    Routed experts **stay an `nn.ModuleList` of `nn.Linear`-based experts**,
    so state-dict keys are unchanged (no migration hook needed), Muon
    (task 4) is untouched, and `moe.routed_experts[i]` still works for the
    trace path / `scaling.py` / viz. Only active experts' weights are
    stacked per forward, so an idle expert's `.grad` stays `None`.
  - `forward_with_trace` keeps the explicit per-expert loop.
  - Tests: batched output matches the loop within `1e-5` incl. a zero-token
    expert (all 3 activations); an idle expert's params get `grad is None`;
    `MoELayer.forward` matches a reference loop; a parent-spec state-dict
    loads with `strict=True` and no migration; wall-clock scales
    sub-linearly with `n_experts` (measured: batched ~flat 8→64 experts,
    loop climbs). See journal entry 027 for the rationale of the deviation.
  - Journal: why the naïve loop is slow, the sorted-dispatch / capacity-grid
    idea, MegaBlocks/GShard links, a small dispatch worked by hand.
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

- [x] 12. Multi-Token Prediction head
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

- [x] 13. KV cache for inference
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

- [x] 14. Inference sampling: min-p + repetition penalty
  - `filter_min_p`, `apply_repetition_penalty` in `generate.py`, mirroring
    the existing filters; new `generate()` params compose with
    `top_k`/`top_p`.
  - Tests: min-p never masks all tokens; neutral params ⇒ identical to
    today; repetition penalty lowers a repeated token's probability.
  - Journal: min-p's adaptive truncation vs top-p, the repetition-loop the
    parent journal hit, min-p paper link, worked distribution example.
  - _Requirements: 11.1, 11.2, 11.3, 11.4_

- [x] 15. Data pipeline: document boundaries
  - Opt-in `ByteBPETokenizer(eos_token=True)` reserves a separator id (one
    past the learned vocab, exposed as `.eos_id`, dropped on `decode`,
    round-tripped through `save`/`load`); `scripts/prepare_data.py
    --split-documents` splits the raw corpus on `<|endoftext|>`, encodes
    each document independently, and joins with `eos_id` (pure helpers
    `split_documents` / `encode_corpus`). `TokenDataset(return_segments=
    True, eos_id=...)` yields `(x, y, segment_ids)` with per-window,
    rebased document ids (`<eos>` = last token of the doc it ends).
    `build_document_mask(segment_ids)` in `model/attention.py` →
    `(B, 1, S, S)` additive `0`/`-inf` mask; `Attention.forward` /
    `MLAAttention.forward` / `TransformerBlock.forward` /
    `QwenMoEModel.forward` thread `document_mask` / `segment_ids` through
    (one mask built per forward, shared across blocks). `Trainer` consumes
    3-tuples when `TrainConfig.document_masking` and raises eagerly
    otherwise; `train_val_split(tokens, *, mode="documents", eos_id=...)`
    holds out whole documents from the tail. `tokenizer_meta` carries
    `eos_id`; `scripts/train.py` gains `--val-split-mode` /
    `--document-masking`.
  - Tests (+27, 332 → 359): `document_masking`/`segment_ids` off ⇒
    `torch.equal` to today (attention + full model); a document-1 token is
    bit-identical under a document-0 perturbation (`Attention`, all
    positional schemes, MLA, full model); `train_val_split(mode=
    "documents")` splits only on separators; BPE `eos_token` reserves one
    id and is dropped on decode; `encode_corpus` inserts exactly one
    separator per document.
  - Journal: `journey_of_a_model/031-document-boundaries.md` — packing,
    cross-document contamination (measured 78.4% → 0% attention leakage),
    the additive two-mask composition, worked `segment_ids` + mask
    examples, Llama-3 / Zhao et al. 2024 / Krell et al. 2021.
  - _Requirements: 10.1, 10.2, 10.3, 10.4_

- [x] 16. Ablation methodology: multi-seed, param-matched, mean ± std
  - `run_comparison(..., *, seeds=None, param_matched=False,
    baseline_name=None)` — trains each variant once per seed on the same
    fixed slice (default `(train_config.seed,)` = today's single run),
    nesting logs under `seed_<n>/` only when >1 seed. `VariantResult`
    gains `seeds` / `per_seed_val_loss` / `per_seed_val_perplexity` /
    `val_loss_mean` / `val_loss_std` / `val_perplexity_mean` /
    `val_perplexity_std` / `log_paths` / `comparison_mode` /
    `within_noise_of_baseline` (all additive — every pre-task field and
    single-seed value unchanged). `ComparisonReport` gains `baseline_name`,
    `seeds`, `summary_table()` (`mean ± std` + `baseline`/`within
    noise`/`differs`/`n/a (1 seed)` note). `within_noise_of_baseline` =
    `|Δmean| < hypot(std, std_baseline)`, only when ≥2 seeds.
    `match_d_ff_to_param_budget(config, target, tol=0.02)` bisects `d_ff`
    (params monotone in it) to the baseline total within 2%;
    `param_matched=True` applies it to every non-baseline variant and
    stamps `comparison_mode`. `scripts/run_comparison.py` gains `--seeds`
    / `--param-matched`, prints `summary_table()`, JSON records per-seed
    losses + noise flag + mode.
  - Tests (+6, 359 → 365): single-seed default keeps scalars + no noise
    flag + un-nested log path; 3 seeds ⇒ `std > 0` + 3 distinct per-seed
    losses + one log dir per seed; an identical "twin" variant flags
    `within noise`; `summary_table()` shows `±` + a note;
    `match_d_ff_to_param_budget` + `param_matched=True` hit the 2% budget
    for a GELU variant and stamp the mode.
  - Journal: `journey_of_a_model/032-multi-seed-and-param-matched-ablations.md`
    — seeds as samples, the `√(σ² + σ_base²)` noise band derived + worked
    on measured numbers (NoPE Δ −0.0002 inside a ±0.011 band vs GELU
    Δ −0.048 at 4× the band), the linear-in-`d_ff` param count + bisection
    (SwiGLU 128 → GELU 192), compute- vs param-matched (Switch
    Transformer), Bouthillier et al. 2021 / Dodge et al. 2019.
  - _Requirements: 12.1, 12.2, 12.4_

- [x] 17. Ablation methodology: refreshed menu, MoE budget note, TinyStories default
  - `scripts/run_ablation_study.py::VARIANTS` entries are now 4-tuples
    `(axis, label, ModelConfig, train_overrides)`; +6 rows: `OPT_MUON`
    (`train_overrides={"optimizer":"muon"}`), `SCHED_WSD`
    (`{"lr_schedule":"wsd"}`), `ROUTE_SIGMOID`, `ATT_QKNORM`, `MTP_D1`,
    `MOE_FINEGRAINED`. `configs/presets.py` gains `ROUTE_SIGMOID` /
    `ATT_QKNORM` / `MTP_D1` (one axis off `SMOKE_TEST`) and
    `MOE_FINEGRAINED` (16 experts × `d_ff=32`, `top_k=8`, 4 shared — every
    FFN budget held ~constant, +0.6% total params from the wider router).
    `run_comparison(..., train_overrides={name: {TrainConfig field}})`
    applies per-variant `TrainConfig` deltas (`replace(train_config,
    **ov, seed=seed)`), keeping the token budget shared. Default corpus →
    `tiny_stories` / `bpe` with `--smoke` restoring char tiny-Shakespeare.
  - `qwenmoe/train/scaling.py::moe_budget_note(total, active)` — states
    `20×` is a dense constant, the compute-optimal MoE budget lies in
    `[20·active, 20·total]` (Clark et al. 2022), `20·active` is a floor;
    logged at startup by `scripts/train.py`, `scripts/run_comparison.py`,
    `scripts/run_ablation_study.py` alongside `count_params` /
    `chinchilla_budget`.
  - Tests (+17, 365 → 382): `moe_budget_note` bounds/wording/validation;
    `ROUTE_SIGMOID`/`ATT_QKNORM`/`MTP_D1` in the one-axis check +
    `MOE_FINEGRAINED` FFN-budget test; new `test_run_ablation_study.py` —
    every `VARIANTS` entry is a valid 4-tuple naming real `TrainConfig`
    fields, each new axis constructs + trains via `run_comparison`, a Muon
    `train_overrides` delta produces a Muon checkpoint with an unchanged
    `ModelConfig`.
  - Journal: `journey_of_a_model/033-ablation-menu-and-moe-budget-note.md`
    — the `ModelConfig` vs `TrainConfig` axis split + `train_overrides`,
    fine-grained segmentation with FFN budgets tabulated (DeepSeekMoE),
    the `[20·active, 20·total]` budget band derived on `BASELINE`'s real
    23.5M/9.3M split (Clark et al. 2022), why TinyStories is the real
    tier; measured single-seed run of all 6 new axes (illustrative, with
    the entry-032 caveat).
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
