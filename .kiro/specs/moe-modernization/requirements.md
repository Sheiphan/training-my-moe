# Requirements Document

## Introduction

The `qwenmoe` package (see `.kiro/specs/qwen-moe-from-scratch/`) implements a
correct, well-tested Qwen-/DeepSeek-V2-era decoder-only MoE transformer. This
spec **modernizes** that implementation against published 2024–2026 practice,
staged as a research/ablation project rather than a rewrite. Every change is
one of:

- a **pure win** — mathematically equivalent or strictly-better behavior with
  no downside (fused attention kernels, gradient clipping, tuned AdamW
  constants). These change the default path directly.
- a **new axis** — an alternative technique added behind a config flag,
  **defaulting to today's behavior**, so the existing 213-test suite stays
  green and the `scripts/run_comparison.py` framework can measure it (Muon,
  sigmoid routing, QK-norm, WSD schedule, MTP, …).
- a **methodology upgrade** — changes to how the comparison study is *run*
  (multiple seeds, the TinyStories tier, param-matched variants) so its
  conclusions are defensible.

The audience and constraints are unchanged from the parent spec: a single
learner, an Apple-Silicon laptop (M4 Pro, 24 GB, `mps`/`cpu`), no discrete
GPU, `uv`-managed, dependency-minimal (`torch`, `manim`, `huggingface_hub`,
`pytest` — **no** new heavyweight deps; Muon and µP-style scaling are
implemented from scratch in-repo, matching the "from scratch" rule).

Source for every technique below: `journey_of_a_model/017-*` onward and the
"MoE Upgrade Path" review these requirements were written from.

## Assumptions & Defaults (flag anything wrong here at the approval gate)

- **Backward compatibility is a hard requirement.** After every task, `uv run
  pytest -q` passes with the same count or higher. New model/train config
  fields default such that a config written before this spec produces
  bit-comparable behavior (within floating-point tolerance) to today —
  *except* where a requirement is explicitly a pure-win default change, which
  is called out per requirement.
- **`forward_with_trace` parity.** Any change to `Attention`, `MoELayer`,
  `TransformerBlock`, or `QwenMoEModel` keeps the trace path working and
  numerically consistent with `forward` (parent Requirement 7.1). New
  architectural options that the Manim scenes don't yet visualize may raise a
  clear `NotImplementedError` from `forward_with_trace` (the documented
  pattern MLA already uses), never silently diverge.
- **No new runtime dependencies.** `pyproject.toml`'s dependency list does not
  grow. `torch.compile`, `torch.autocast`, and
  `F.scaled_dot_product_attention` are stdlib-of-torch and allowed.
- **Config-as-code.** Every new knob is a field on `ModelConfig` /
  `TrainConfig` with validation in `__post_init__`; new named presets go in
  `configs/presets.py`. No YAML/JSON schema layer (parent decision).
- **Journal convention carries over unchanged** (parent Requirement 13.2,
  13.6–13.9): every completed task gets a dated
  `journey_of_a_model/NNN-slug.md` beginner-explainer entry with a Mermaid
  diagram, a "Questions that might come to mind" section, the full derivation
  of any formula with a hand-worked numeric example, and a link to the
  source paper.
- **Scale for correctness work stays tiny** (unit tests run in seconds on
  CPU); the headline ablation study moves to the parent spec's already-planned
  Tier 2 (TinyStories, from-scratch BPE).

## Requirements

### Requirement 1: Optimizer modernization (Muon + tuned AdamW)

**User Story:** As a learner running many small training runs, I want a
modern optimizer setup so that each run reaches a given loss in fewer steps
and without hand-tuning, and so that "Muon vs AdamW" becomes a measurable
axis.

#### Acceptance Criteria

1. WHEN the trainer builds its optimizer THEN the system SHALL split
   parameters into a **decay group** (tensors with `ndim >= 2`) and a
   **no-decay group** (biases, RMSNorm gains, embeddings), applying weight
   decay only to the former.
2. WHEN AdamW is used THEN the system SHALL read `betas`, `eps`, and
   `weight_decay` from `TrainConfig` with defaults `betas=(0.9, 0.95)`,
   `eps=1e-8`, `weight_decay=0.1` (a **pure-win default change** from
   PyTorch's `(0.9, 0.999)` / `0.01`).
3. WHEN a training step completes the backward pass THEN the system SHALL
   clip the global gradient norm to `TrainConfig.grad_clip_norm` (default
   `1.0`) before `optimizer.step()`, and SHALL record the pre-clip norm in
   the JSON-lines step log.
4. WHERE `TrainConfig.optimizer == "muon"` THE SYSTEM SHALL optimize every
   2-D weight (attention projections, expert `w_gate`/`w_up`/`w_down`, the
   router) with a from-scratch Muon implementation (momentum + Newton–Schulz
   orthogonalization of the update) and every remaining parameter
   (embeddings, norms, the LM head, any 1-D tensor) with AdamW.
5. WHEN `TrainConfig.optimizer == "adamw"` (the default) THEN the system
   SHALL behave exactly as criteria 1–3 describe, with no Muon code path
   active.
6. WHEN a Muon run and an AdamW run are compared via
   `scripts/run_comparison.py` on an identical fixed slice THEN both SHALL
   train and produce a `VariantResult`, and the checkpoint SHALL round-trip
   through save/resume for both optimizers.

### Requirement 2: Learning-rate schedule (WSD)

**User Story:** As a learner who often wants to extend or branch a run, I
want a schedule that doesn't hard-commit the token budget up front.

#### Acceptance Criteria

1. WHEN `TrainConfig.lr_schedule == "cosine"` (the default) THEN the system
   SHALL behave as today (linear warmup → cosine decay) **except** the decay
   floor SHALL be `TrainConfig.min_lr_ratio` (default `0.1`) of peak rather
   than `0.0` (a **pure-win default change**).
2. WHERE `TrainConfig.lr_schedule == "wsd"` THE SYSTEM SHALL run linear
   warmup, then hold the peak rate constant, then decay to
   `min_lr_ratio × peak` over the final `TrainConfig.decay_fraction` (default
   `0.2`) of total steps.
3. WHEN either schedule is selected THEN `lr_multiplier` SHALL remain a pure
   function of `(step, config)` returning a multiplier in `[0, 1]`, unit-
   testable without a model.
4. WHEN a WSD run reaches any step in its stable phase THEN a checkpoint
   saved there SHALL be resumable such that a *shorter* subsequent run can
   apply a fresh decay phase from it (a "cooldown branch").

### Requirement 3: Mixed precision and numerical stability

**User Story:** As a learner on a memory-limited laptop, I want optional
bf16 training and the small stability terms modern models rely on, so that
runs are faster and don't spike.

#### Acceptance Criteria

1. WHERE `TrainConfig.precision == "bf16"` THE SYSTEM SHALL wrap the forward
   pass in `torch.autocast` with `dtype=torch.bfloat16`, keep the router
   logits/softmax and the loss computation in `float32`, and require no
   gradient scaler.
2. WHEN `TrainConfig.precision == "fp32"` (the default) THEN no autocast
   context SHALL be entered.
3. WHERE `TrainConfig.z_loss_weight > 0` (default `0.0`, i.e. off) THE SYSTEM
   SHALL add `z_loss_weight × mean(logsumexp(logits, dim=-1) ** 2)` to the
   training loss, exposed as its own field in the step log alongside
   `train_loss` and `aux_loss`.
4. WHEN mixed precision is active THEN the full-model forward SHALL still
   produce finite logits and a decreasing loss on the trainer integration
   test's synthetic sequence.

### Requirement 4: MoE routing modernization

**User Story:** As a learner studying expert routing, I want the DeepSeek-V3
routing refinements available as flags so I can see their effect on expert
utilization.

#### Acceptance Criteria

1. WHERE `ModelConfig.router_scoring == "sigmoid"` THE SYSTEM SHALL compute
   per-expert affinities as `sigmoid(router_logits)`, select the top-`k` by
   affinity (plus the loss-free bias when active), and form combination
   weights by normalizing the selected affinities to sum to 1.
2. WHEN `ModelConfig.router_scoring == "softmax"` (the default) THEN routing
   SHALL be exactly as today.
3. WHERE `ModelConfig.router_z_loss_weight > 0` (default `0.0`) THE SYSTEM
   SHALL add `weight × mean(logsumexp(router_logits, dim=-1) ** 2)` to the
   MoE auxiliary loss.
4. WHERE `ModelConfig.moe_balancing == "loss_free"` AND
   `ModelConfig.seq_balance_loss_weight > 0` (default `0.0`) THE SYSTEM SHALL
   additionally compute the DeepSeek-V3 complementary **per-sequence**
   balance loss at that small weight, so single-sequence routing collapse is
   still penalized.
5. WHEN `AuxLossBalancer` is used THEN its `loss_weight` SHALL be readable
   from `ModelConfig.aux_loss_weight` (default `0.01`, unchanged) rather than
   hard-coded.
6. WHEN any routing flag is toggled THEN the router SHALL still select
   exactly `k` experts per token, combination weights SHALL still sum to 1
   (±1e-5), and a zero-token expert SHALL still be skipped without error
   (parent Requirements 2.2, 2.3, 2.5).

### Requirement 5: Efficient expert computation

**User Story:** As a learner who wants to try many-expert configs, I want the
MoE layer to not be O(num_experts) Python calls, so that fine-grained-expert
runs are practical on a laptop.

#### Acceptance Criteria

1. WHEN `MoELayer.forward` dispatches tokens to routed experts THEN it SHALL
   compute all experts of a given activation type via a batched operation
   (sorted-token grouped matmul or stacked-weight `bmm`), not a per-expert
   Python loop with one `index_add_` each.
2. WHEN the batched path runs on the same input and weights as the current
   loop THEN its output SHALL match the current implementation within
   `1e-5` (it is a performance refactor, not a behavior change).
3. IF a routed expert receives zero tokens THEN the batched path SHALL still
   produce no gradient for that expert and SHALL not error (parent
   Requirement 2.5).
4. WHEN `forward_with_trace` runs THEN per-expert `expert_outputs` SHALL
   still be captured for the dispatched experts (parent Requirement 7.1) —
   the trace path MAY keep the explicit loop for clarity if the fast path is
   hard to instrument.
5. WHEN a fine-grained preset (e.g. 32 experts, `d_ff` reduced
   proportionally, `top_k` raised) is trained for a few steps THEN it SHALL
   complete without error and its per-step wall-clock SHALL be within a
   small constant factor of the 8-expert baseline, not ~4×.

### Requirement 6: Multi-Token Prediction (MTP)

**User Story:** As a learner, I want an optional multi-token-prediction head
so I can study how predicting two tokens ahead densifies the training signal.

#### Acceptance Criteria

1. WHERE `ModelConfig.mtp_depth >= 1` (default `0`, i.e. off) THE SYSTEM
   SHALL add `mtp_depth` lightweight prediction module(s), each taking the
   main trunk's final hidden state and predicting the token at offset
   `+2, +3, …` with its own cross-entropy term.
2. WHEN `mtp_depth == 0` THEN the model SHALL be structurally identical to
   today (no extra parameters, no extra loss term).
3. WHEN MTP is active THEN the combined loss SHALL be
   `lm_loss + mtp_lambda × mean(mtp_losses)` with `mtp_lambda` a
   `ModelConfig` field (default `0.3`), and each depth's loss SHALL appear in
   the step log.
4. WHEN MTP is active THEN `forward` SHALL still return
   `(next_token_logits, {"aux_loss": ...})` with the primary head's logits
   unchanged in shape and meaning, so `generate()` and every existing caller
   work without modification.
5. WHEN a checkpoint of an MTP model is saved and resumed THEN the MTP
   modules SHALL round-trip.

### Requirement 7: Attention modernization

**User Story:** As a learner, I want the attention path to use the fused
kernel and the QK-norm stability trick, while the visualization path keeps
the explicit math.

#### Acceptance Criteria

1. WHEN `Attention.forward` (GQA) and `MLAAttention.forward` compute
   attention THEN they SHALL call `F.scaled_dot_product_attention` with
   `is_causal=True` rather than materializing an explicit `(seq, seq)`
   `-inf` mask and a manual softmax (a **pure-win change** to the non-trace
   path).
2. WHEN `Attention.forward_with_trace` runs THEN it SHALL keep the explicit
   `Q·Kᵀ → mask → softmax → ·V` steps and capture `attn_scores_pre_mask`
   and `attn_scores_post_softmax` (parent Requirement 7.3) — its output
   SHALL match `forward`'s within `1e-5`.
3. WHERE `ModelConfig.qk_norm` is `True` (default `False`) THE SYSTEM SHALL
   apply a per-head `RMSNorm` to `q` and to `k` (learnable scale of size
   `head_dim`) before the positional transform, in both `forward` and
   `forward_with_trace`.
4. WHEN ALiBi is the positional scheme THEN the additive bias SHALL still be
   applied (passed to SDPA as an attention-mask tensor, or retained on the
   explicit path).
5. WHEN `qk_norm` is `False` THEN attention output SHALL be numerically
   unchanged from today.

### Requirement 8: KV cache for inference

**User Story:** As a learner, I want `generate()` to use an incremental KV
cache so that generation isn't O(n²) per token and so my MLA variant can
demonstrate a real memory advantage.

#### Acceptance Criteria

1. WHEN `generate()` produces tokens THEN attention SHALL consume a
   per-layer cache of past keys/values (GQA) or past latent + decoupled key
   (MLA), appending one position per step instead of re-encoding the whole
   prefix.
2. WHEN cached and uncached generation are run from the same checkpoint,
   prompt, and seed THEN they SHALL produce identical token sequences.
3. WHEN a GQA model and an MLA model of matched `d_model`/`n_heads`/
   `n_layers` generate the same number of tokens THEN the measured cache
   memory (bytes actually retained) SHALL be reported for both, and MLA's
   SHALL be smaller — a **measured** result, replacing the parent spec's
   reporting-only `kv_cache_elements_per_token`.
4. WHEN `max_seq_len` is reached THEN generation SHALL stop as today (parent
   Requirement 6.3).

### Requirement 9: Principled initialization

**User Story:** As a learner, I want the weight initialization to follow the
depth-aware scheme real transformers use, so deep configs train stably and
so hyperparameter reasoning is sound.

#### Acceptance Criteria

1. WHEN `QwenMoEModel` is constructed THEN every `nn.Linear` SHALL be
   initialized from a truncated normal with `std = ModelConfig.init_std`
   (default `0.02`), replacing PyTorch's default Kaiming-uniform.
2. WHEN a block's residual-path output projections (`Attention.w_o`, each
   expert's `w_down` / `w_2`) are initialized THEN their `std` SHALL be
   scaled by `1 / sqrt(2 × n_layers)` (the GPT-2/Llama residual-scaling
   rule).
3. WHEN `tie_word_embeddings` is `True` THEN the shared matrix SHALL keep the
   embedding init (`std = init_std`), not the scaled-linear init.
4. WHEN an untrained model's logits are computed on a random batch THEN
   their standard deviation SHALL be within a small factor of `1.0` (no
   50×-style blow-up — the failure mode the parent journal already hit
   once).
5. WHEN `init_std` and the scaling rule reproduce today's numbers for the
   `SMOKE_TEST` shape within a documented tolerance THEN existing trained
   checkpoints and tests remain valid; otherwise the change is noted as a
   deliberate, journaled default change.

### Requirement 10: Data pipeline — document boundaries

**User Story:** As a learner training on TinyStories, I want documents kept
separate so the model doesn't learn cross-story correlations and perplexity
isn't measured on nonsense contexts.

#### Acceptance Criteria

1. WHEN a multi-document corpus is tokenized THEN `scripts/prepare_data.py`
   SHALL insert a dedicated separator/`<eos>` token id between documents and
   record that id in `tokenizer.json` / `tokenizer_meta`.
2. WHERE `TrainConfig.document_masking` is `True` (default `False` for
   backward compatibility) THE SYSTEM SHALL build a per-window attention
   mask that blocks attention across separator boundaries and pass it
   through `Attention`/`MLAAttention` (via SDPA's `attn_mask`).
3. WHEN `document_masking` is `False` THEN behavior SHALL be identical to
   today (flat stream, causal-only mask).
4. WHEN the validation split is built for a multi-document corpus THEN it
   SHALL be possible to hold out **whole documents** (split on separator
   boundaries) rather than a contiguous token tail, via a
   `DataConfig.val_split_mode` field (`"tail"` default, `"documents"` opt-in).

### Requirement 11: Inference sampling

**User Story:** As a learner, I want the current-best sampling options so my
generated text is coherent without hand-tuning `top_p`.

#### Acceptance Criteria

1. WHEN `generate()` is called with `min_p` set THEN the system SHALL keep
   only tokens with probability `>= min_p × max_token_probability` before
   sampling, always retaining at least the argmax token.
2. WHEN `generate()` is called with `repetition_penalty != 1.0` THEN the
   system SHALL divide the logit of every already-generated token id by the
   penalty before sampling.
3. WHEN `min_p` and `repetition_penalty` are both unset/neutral THEN
   generation SHALL be identical to today (`temperature`/`top_k`/`top_p`
   path unchanged).
4. WHEN `min_p` is combined with `top_k`/`top_p` THEN the filters SHALL
   compose (each further restricts the candidate set) and the result SHALL
   never mask every token.

### Requirement 12: Ablation methodology

**User Story:** As a learner simulating a real research process, I want the
comparison study to produce defensible conclusions, not single-run point
estimates.

#### Acceptance Criteria

1. WHEN `run_comparison` is invoked with `seeds=[...]` THEN it SHALL train
   each variant once per seed on the identical fixed slice and return, per
   variant, the mean and standard deviation of final val loss / perplexity
   across seeds.
2. WHEN the comparison summary table/JSON is produced THEN each variant's
   metric SHALL be shown as `mean ± std` and the baseline delta SHALL be
   flagged as "within noise" when it is smaller than the pooled std.
3. WHEN the headline ablation study is run THEN its default corpus SHALL be
   TinyStories with the from-scratch BPE tokenizer (parent Tier 2), not
   char-level tiny-Shakespeare; the Shakespeare run SHALL remain available
   as a fast smoke check.
4. WHERE a variant changes the parameter count (e.g. non-gated GELU experts)
   THE SYSTEM SHALL support a `param_matched` mode that adjusts `d_ff` so
   total parameters match the baseline within 2%, and the report SHALL state
   for each variant whether the comparison is compute-matched or
   param-matched.
5. WHEN the `VARIANTS` menu is defined THEN it SHALL include the
   modernization axes introduced by this spec (`optimizer`,
   `router_scoring`, `qk_norm`, `lr_schedule`, `mtp_depth`, fine-grained
   experts) in addition to the parent spec's original axes.
6. WHEN `chinchilla_budget` is reported for an MoE model THEN the output
   SHALL also state that `20 × active_params` is the dense-model constant and
   that sparse-MoE compute-optimal ratios differ (parent Requirement 11.2,
   Clark et al. 2022), via a companion `moe_budget_note(total, active)`
   helper.

### Requirement 13: Tests and journal

**User Story:** As a learner, I want the same evidence bar as the parent
project so I can trust each modernization before running the study on it.

#### Acceptance Criteria

1. WHEN any task in this spec is completed THEN `uv run pytest -q` SHALL
   pass with a test count `>=` the pre-task count, including new tests that
   cover the added behavior AND assert the default-path behavior is
   unchanged.
2. WHEN a task adds a config flag THEN `tests/test_config.py` SHALL cover its
   validation (valid defaults construct; invalid values raise `ValueError`).
3. WHEN a task changes a numerical path THEN a test SHALL pin the
   "flag off ⇒ identical to today" invariant with an explicit tolerance.
4. WHEN a task is completed THEN a dated `journey_of_a_model/NNN-slug.md`
   entry SHALL be added and indexed in `journey_of_a_model/README.md`,
   following the parent spec's beginner-explainer standard (Requirement
   13.6–13.9): every term defined before use, ≥1 Mermaid diagram, a
   "Questions that might come to mind" section, the full derivation of any
   formula with a hand-worked numeric example, and a link to the source
   paper found via web search.
5. WHEN an entry describes a technique THEN it SHALL link the specific paper
   it comes from (DeepSeek-V3, Muon reports, MiniCPM/WSD, Qwen3, MTP,
   MegaBlocks, min-p, StreamingLLM, etc. as applicable).
