# Implementation Plan

> **Status update (post-task-16):** tasks 13 and 14 below implemented Manim
> scenes; all Manim scene code (`qwenmoe/viz/scenes_*.py`,
> `scripts/render_viz.py`, `tests/test_scenes_training_errors.py`) and the
> `manim` dependency were later removed from the project entirely. Task
> items below are left checked as a historical record of what was built and
> later removed, not as a description of the current codebase.
>
> Convention (Requirement 13.2, 13.6-13.9): finishing any task below is not
> "done" until a dated journal entry is added under `/journey_of_a_model/`
> (`NNN-short-slug.md`, indexed in `journey_of_a_model/README.md`)
> summarizing what was implemented and why, and, where relevant, any idea
> considered/rejected or config decision made along the way. Whenever an
> entry introduces a technical concept, write it as a beginner-friendly
> explainer (define every term before using it: plain-English meaning, the
> problem it solves, a real-world analogy, a simple technical explanation),
> include at least one Mermaid diagram, a "Questions that might come to
> mind" section, the full mathematical derivation for any concept backed by
> a formula plus a worked numeric example run by hand on dummy values, and
> link the concept's original paper/blog found via web search. This applies
> to every task; it is not repeated in each item below.

- [x] 1. Project scaffolding
  - Add `manim`, `huggingface_hub`, and `pytest` (dev) via `uv add` /
    `uv add --dev`; keep `torch` as already declared.
  - Create the `qwenmoe/` package skeleton (empty `__init__.py` files for
    `qwenmoe`, `qwenmoe.tokenizer`, `qwenmoe.data`, `qwenmoe.model`,
    `qwenmoe.train`, `qwenmoe.viz`) and declare it as the build target
    (`[tool.hatch.build.targets.wheel] packages = ["qwenmoe"]` in
    `pyproject.toml`).
  - Create `scripts/`, `configs/`, `tests/`, and `journey_of_a_model/`
    directories; add `journey_of_a_model/README.md` (empty index table) and
    a first entry describing the spec/design decisions made so far.
  - Verify `uv sync` installs cleanly and `uv run python -c "import qwenmoe"`
    succeeds.
  - _Requirements: 10.2, 13.5_

- [x] 2. Config layer
  - Implement `qwenmoe/config.py` with `ModelConfig`, `TrainConfig`,
    `DataConfig` dataclasses exactly as specified in design.md, including
    `ModelConfig.__post_init__` validation (`top_k <= n_routed_experts`,
    `n_heads % n_kv_heads == 0`, `mla_latent_dim` required when
    `attention_variant == "mla"`).
  - Implement `configs/presets.py` with a `BASELINE` preset matching the
    illustrative sizing in requirements.md (d_model=256, 6 layers, 8
    heads/4 kv-heads, 1 shared + 8 routed experts, top-2).
  - Write `tests/test_config.py` covering each valid default construction
    and each invalid combination raising `ValueError`.
  - _Requirements: 3.1, 3.3, 12.1, 12.2, 12.3_

- [x] 3. Normalization and positional schemes
- [x] 3.1 RMSNorm
  - Implement `qwenmoe/model/rmsnorm.py::RMSNorm`.
  - Write `tests/test_rmsnorm.py` asserting output shape and unit
    root-mean-square within floating-point tolerance.
  - _Requirements: 1.4, 9.1_
- [x] 3.2 RoPE
  - Implement `qwenmoe/model/positional.py::RotaryEmbedding` with
    `apply(q, k, positions) -> (q, k)`, configurable `rope_theta`.
  - Write `tests/test_rope.py` asserting norm preservation and a known
    rotation angle for a hand-picked input.
  - _Requirements: 1.3, 9.2_
- [x] 3.3 NoPE and ALiBi
  - Implement `NoPositional` (identity `apply`, no `attention_bias`) and
    `AlibiBias` (`attention_bias(seq_len) -> Tensor` linear-slope bias,
    identity `apply`) in `qwenmoe/model/positional.py`.
  - Write `tests/test_positional_alt.py` checking `NoPositional` is a
    true no-op and `AlibiBias` produces a monotonically-increasing-with-
    distance bias matrix.
  - _Requirements: 12.1_

- [x] 4. Attention (GQA baseline)
  - Implement `qwenmoe/model/attention.py::Attention` (scaled dot-product,
    GQA via `n_kv_heads`, causal mask, calls the configured positional
    scheme's `apply`/`attention_bias` hooks).
  - Write `tests/test_attention.py` verifying: `num_kv_heads == num_heads`
    behaves as standard MHA; causal masking means perturbing a later
    token's input leaves an earlier position's output unchanged; output
    shape is `(batch, seq_len, d_model)`.
  - _Requirements: 1.5, 1.6, 9.3_

- [x] 5. Mixture-of-Experts feed-forward layer (baseline)
- [x] 5.1 Expert activation variants
  - Implement `qwenmoe/model/moe.py::SwiGLUExpert`, `GEGLUExpert`,
    `GELUExpert` behind a common `Expert` interface selected by
    `ModelConfig.activation`.
  - Write `tests/test_experts.py` checking output shape for each variant.
  - _Requirements: 2.1, 12.2_
- [x] 5.2 Router, auxiliary-loss balancing, and MoE layer
  - Implement `Router`, `AuxLossBalancer`, and `MoELayer` (top-k routed
    experts + always-on shared experts, renormalized gate combination,
    skip-if-zero-tokens per expert) in `qwenmoe/model/moe.py`.
  - Write `tests/test_moe.py` verifying: exactly `k` experts selected per
    token; selected gate weights sum to 1; an expert receiving zero tokens
    doesn't raise; aux loss is a finite scalar that responds to induced
    imbalance.
  - _Requirements: 2.2, 2.3, 2.4, 2.5, 9.4_

- [x] 6. Full model assembly
  - Implement `qwenmoe/model/block.py::TransformerBlock` (pre-norm attn +
    pre-norm MoE with residuals) and `qwenmoe/model/transformer.py::
    QwenMoEModel` (embedding, stacked blocks, final norm, tied/untied LM
    head via `tie_word_embeddings`), exposing
    `forward(input_ids) -> (logits, {"aux_loss": ...})`.
  - Write `tests/test_model.py` verifying output shape
    `(batch, seq_len, vocab_size)` for at least two different
    batch/seq-len combinations, and that tied-embedding mode shares the
    same weight tensor.
  - _Requirements: 1.1, 1.2, 1.7, 9.5_

- [x] 7. Tokenizers
- [x] 7.1 Character-level tokenizer
  - Implement `qwenmoe/tokenizer/char.py::CharTokenizer` (`train`,
    `encode`, `decode`, `save`/`load`, `vocab_size`).
  - Write `tests/test_char_tokenizer.py` asserting lossless round-trip for
    arbitrary corpus text.
  - _Requirements: 4.2, 4.3_
- [x] 7.2 From-scratch byte-level BPE tokenizer
  - Implement `qwenmoe/tokenizer/bpe.py::ByteBPETokenizer` (merge-pair
    learning loop implemented in this project, `encode`/`decode`/
    `save`/`load`).
  - Write `tests/test_bpe_tokenizer.py` asserting lossless round-trip and
    that trained vocab size matches the configured target.
  - _Requirements: 4.2, 4.3_

- [x] 8. Data pipeline
- [x] 8.1 Corpus download and dataset windows
  - Implement `qwenmoe/data/download.py::download_corpus` for
    `tiny_shakespeare` (direct HTTPS text file) and `tiny_stories`
    (`huggingface_hub.hf_hub_download` of the plain-text release file),
    caching under `DataConfig.cache_dir`.
  - Implement `qwenmoe/data/dataset.py::TokenDataset` (fixed-length
    `(x, y)` windows) and a train/val split helper.
  - Write `tests/test_dataset.py` using a small in-memory token list,
    checking window shapes and split proportions.
  - _Requirements: 4.1, 4.4_
- [x] 8.2 Fixed-slice sampling for fair comparisons
  - Implement `qwenmoe/data/dataset.py::make_fixed_slice(all_tokens,
    n_tokens, seed)`.
  - Write a test asserting identical `(all_tokens, n_tokens, seed)` always
    yields an identical slice, and different seeds yield different slices.
  - _Requirements: 11.5, 12.6_

- [x] 9. Scaling-law utilities
- [x] 9.1 Parameter counting and Chinchilla reference budget
  - Implement `qwenmoe/train/scaling.py::count_params(model) ->
    ParamBreakdown` (embedding/attention/moe total/moe active/total/active)
    and `chinchilla_budget(active_params) -> int` (`20 × active_params`).
  - Write `tests/test_scaling.py` checking `count_params` against a
    hand-computed value for a tiny fixed config, and `chinchilla_budget`'s
    arithmetic.
  - _Requirements: 11.1, 11.2_
- [x] 9.2 Throughput calibration and epoch/repetition warning
  - Implement `calibrate_throughput(model, sample_batch, device,
    n_steps=20) -> tokens_per_sec` (runs real steps, times them) and
    `epochs_for_budget(total_tokens, unique_tokens) -> float`, logging a
    `logging.warning` when the result exceeds 4.
  - Write `tests/test_scaling.py` (extend) checking `epochs_for_budget`'s
    arithmetic and that the warning fires past 4 and not below it.
  - _Requirements: 11.3, 11.4_

- [x] 10. Training loop
- [x] 10.1 Trainer
  - Implement `qwenmoe/train/trainer.py::Trainer` — AdamW with
    warmup+cosine schedule, device auto-selection (`mps` else `cpu`),
    console logging of step/loss/aux-loss/elapsed at a configurable
    interval, periodic val loss/perplexity, checkpoint save (model +
    optimizer + config + step) and resume, and structured JSON-lines
    logging of per-step metrics and per-expert token counts (consumed
    later by the training-observability scenes).
  - Write a small integration test (`tests/test_trainer_integration.py`):
    train a tiny config for ~50 steps on a short repeating synthetic
    sequence and assert loss is finite and lower at the end than at the
    start.
  - _Requirements: 3.2, 5.1, 5.2, 5.3, 5.4, 8.1, 8.2_
- [x] 10.2 Data-prep and training CLIs
  - Implement `scripts/prepare_data.py` (downloads/tokenizes a corpus,
    reports token counts) and `scripts/train.py` (argparse: preset name,
    corpus, token budget, output dir; wires config → data → model →
    `Trainer`).
  - Manually run `scripts/prepare_data.py` + `scripts/train.py` end-to-end
    on the tiny-Shakespeare smoke-test config to confirm the full pipeline
    executes and produces a checkpoint with decreasing validation loss.
  - _Requirements: 5.5, 10.1_

- [x] 11. Inference / generation
- [x] 11.1 Sampling
  - Implement `qwenmoe/model/generate.py::generate(model, tokenizer,
    prompt, max_new_tokens, temperature, top_k, top_p)` with autoregressive
    decoding and a max-context/max-new-tokens stop condition.
  - Write `tests/test_generate.py` checking: greedy (temperature→0-like)
    decoding is deterministic; output never exceeds the configured length;
    runs correctly with `device="cpu"`.
  - _Requirements: 6.2, 6.3, 6.4_
- [x] 11.2 Generation CLI
  - Implement `scripts/generate.py` (argparse: checkpoint path, prompt,
    sampling params) that loads a checkpoint, tokenizes, generates, and
    prints decoded text.
  - _Requirements: 6.1_

- [x] 12. Visualization foundation
  - Add `forward_with_trace(input_ids) -> (logits, TraceBundle)` to
    `QwenMoEModel`, and the `TraceBundle`/`LayerTrace` dataclasses (per
    design.md) populated with embeddings, per-layer q/k/v, attention
    scores pre/post-softmax, router logits/probs/top-k, expert outputs,
    and final logits/probs.
  - Implement `qwenmoe/viz/common.py::capture_trace(checkpoint_path,
    prompt) -> TraceBundle` (loads checkpoint, runs `forward_with_trace`
    under `torch.no_grad()`).
  - Write `tests/test_trace.py` asserting every `TraceBundle`/`LayerTrace`
    field is populated with the expected shape for a tiny model.
  - _Requirements: 7.1_

- [x] 13. Architecture math Manim scenes
- [x] 13.1 Embedding + positional-encoding scene
  - Implement `qwenmoe/viz/scenes_embedding.py`, animating the
    token-to-vector lookup and RoPE rotation of a query/key vector from a
    captured `TraceBundle`.
  - _Requirements: 7.2_
- [x] 13.2 Attention scene
  - Implement `qwenmoe/viz/scenes_attention.py`, animating the Q·Kᵀ score
    matrix, causal mask application, softmax normalization, and weighted
    sum with V.
  - _Requirements: 7.3_
- [x] 13.3 MoE routing scene
  - Implement `qwenmoe/viz/scenes_moe.py`, animating router-logit
    computation, softmax gate scores, top-k selection, and token dispatch.
  - _Requirements: 7.4_
- [x] 13.4 RMSNorm/SwiGLU scene
  - Implement `qwenmoe/viz/scenes_norm_act.py`, animating the RMSNorm
    scaling step and the gated SwiGLU activation inside an expert.
  - _Requirements: 7.5_
- [x] 13.5 Output projection scene
  - Implement `qwenmoe/viz/scenes_output.py`, animating the final linear
    projection to vocabulary logits and the softmax producing next-token
    probabilities.
  - _Requirements: 7.6_
- [x] 13.6 Rendering CLI and independence check
  - Implement `scripts/render_viz.py` documenting the exact `manim -pql
    <file> <SceneName>` invocation for each scene in `--help` output.
  - Manually verify each scene renders on its own (fresh process, no
    shared state) without running any other scene first.
  - _Requirements: 7.7_
  - **(Removed post-task-16: all Manim scene code and the `manim` dependency were later deleted from the project.)**

- [x] 14. Training/run observability visualization
  - Implement `qwenmoe/viz/scenes_training.py` with a loss-curve scene and
    an expert-utilization-distribution scene, both reading the JSON-lines
    logs written by `Trainer` (task 10.1).
  - Write `tests/test_scenes_training_errors.py` asserting a missing log
    file raises a clear, actionable error (not a raw stack trace) before
    any Manim rendering is attempted.
  - _Requirements: 8.1, 8.2, 8.3_
  - **(Removed post-task-16: all Manim scene code and the `manim` dependency were later deleted from the project.)**

- [x] 15. Research-style comparison framework
- [x] 15.1 Auxiliary-loss-free MoE balancing
  - Implement `LossFreeBalancer` in `qwenmoe/model/moe.py`: a
    non-trainable per-expert bias buffer added to router logits for
    top-k selection only (combination weights stay un-biased), updated
    after each step by nudging above/below-average-load experts down/up
    by a fixed step.
  - Write `tests/test_moe.py` (extend) verifying exactly `k` experts are
    still selected, weights still sum to 1, and repeated imbalanced
    batches measurably flatten expert-selection frequency over steps.
  - _Requirements: 12.3_
- [x] 15.2 Comparison runner
  - Implement `qwenmoe/train/compare.py::run_comparison(base_config,
    variants, data_config, train_config) -> ComparisonReport` (trains each
    named variant on the identical fixed slice from task 8.2, writing
    per-variant logs).
  - Write `tests/test_compare.py` with two tiny variants over a handful of
    steps, asserting both variants consumed the exact same token slice and
    a `ComparisonReport` with both results is returned.
  - _Requirements: 11.5, 12.6_
- [x] 15.3 Comparison CLI and summary output
  - Implement `scripts/run_comparison.py` (argparse: list of preset names
    to compare, shared token budget) that calls `run_comparison` and
    prints/saves a final validation loss/perplexity comparison table.
  - _Requirements: 12.7_

- [x] 16. Stretch: Multi-Head Latent Attention (MLA)
  - Implement `qwenmoe/model/attention.py::MLAAttention` (low-rank joint
    KV compression into `mla_latent_dim`, decoupled RoPE on a small
    sub-dimension, own internal positional handling), selected via
    `ModelConfig.attention_variant == "mla"`.
  - Write `tests/test_mla.py` verifying output shape parity with
    `Attention` for the same `(batch, seq_len, d_model)` input, and that
    the KV cache/latent representation is smaller than the equivalent GQA
    K+V size.
  - _Requirements: 12.4_

- [ ] 17. Stretch: LoRA adapter fine-tuning demo
- [ ] 17.1 LoRA module
  - Implement `qwenmoe/model/lora.py::LoRALinear` and
    `attach_lora(model, rank) -> model`.
  - Write `tests/test_lora.py` verifying the wrapped module's frozen base
    weight doesn't receive gradients while the low-rank `A`/`B` matrices do.
  - _Requirements: 12.5_
- [ ] 17.2 Fine-tune comparison script
  - Implement `scripts/finetune_compare.py` fine-tuning a pretrained
    checkpoint on a small new dataset both fully and via `attach_lora`,
    reporting trainable-parameter count, loss, and wall-clock time for
    both.
  - _Requirements: 12.5_

- [ ] 18. Project documentation
  - Write the top-level `README.md`: `uv sync` setup, and exact commands
    for data prep, training (including the tiny-Shakespeare smoke test),
    inference, running the test suite, rendering each Manim scene, and
    running the comparison study.
  - _Requirements: 10.1, 10.2_
