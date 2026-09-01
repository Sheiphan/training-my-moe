# Requirements Document

## Introduction
This project builds a small Qwen-style decoder-only transformer language model
**from scratch in PyTorch** — no `transformers`/`AutoModel` shortcuts — with a
Mixture-of-Experts (MoE) feed-forward layer instead of a dense MLP. It covers
the full educational loop: implement the architecture, train it on real text
on the user's own laptop, run inference on the trained checkpoint, and render
Manim animations of the actual mathematical operations happening in every
part of the model (embeddings, RoPE, attention, MoE routing, norms,
activations, output head) using real numbers pulled from real forward passes.
The audience is a single learner (the repo owner) who wants to deeply
understand how an LLM works end-to-end, not to produce a production-grade or
state-of-the-art model.

## Assumptions & Defaults (flag anything wrong here at the approval gate)
- **Hardware detected:** Apple M4 Pro, 14 cores, 24GB unified memory, arm64
  macOS. PyTorch will use the `mps` backend when available, falling back to
  `cpu`.
- **Env/tooling:** `uv`-managed project (already initialized), Python 3.12,
  PyTorch already declared in `pyproject.toml`. Manim (Community Edition)
  will be added as a new dependency.
- **Training corpora (tiered), see Requirement 11/12 for the derivation:**
  1. *Smoke test* — tiny Shakespeare (~1MB, char-level) to prove the full
     pipeline runs end to end in minutes, on a tiny (<1M param) model.
  2. *Comparison/ablation study* — a fixed, shuffled-once token slice of
     TinyStories (Eldan & Li, 2023 — a corpus of simple synthetic children's
     stories specifically designed so small (<50M param) models can produce
     coherent text), sized to the scaling-law token budget and reused
     identically across every ablation variant for a fair, compute-matched
     comparison.
  3. *Optional confirmation run* — most/all of TinyStories (~3-4 epochs,
     within the empirically-safe repetition zone per Muennighoff et al. 2023)
     on the winning config from step 2, for the best showcase checkpoint.
  All three are swappable via config, not hard-coded.
- **Tokenizer:** built from scratch in this repo (byte/character-level BPE),
  not a pretrained tokenizer — vocab is derived from the training corpus.
- **Model scale target:** under 100M total parameters / under 30M active
  parameters per token by default, sized so a full training run finishes on
  this laptop without a discrete GPU. Exact layer/expert counts are a design
  decision, not fixed here.
- **Illustrative baseline sizing** (finalized in design.md): d_model=256, 6
  layers, 8 heads/4 kv-heads (GQA), 1 shared + 8 routed experts (top-2) ⇒
  ≈23.5M total / ≈9.3M active params/token — comfortably under budget.
  Chinchilla-style reference budget `D ≈ 20 × active_params` ≈ 186M tokens
  (Hoffmann et al. 2022); MoE-specific compute-optimal ratios can shift this
  constant (Clark et al. 2022 "Unified Scaling Laws for Routed Language
  Models"; newer 2025/2026 MoE scaling-law work), so treat it as a planning
  heuristic, not a hard target — see Requirement 11.
- **Qwen-style features in scope:** RMSNorm, RoPE, causal self-attention with
  optional Grouped-Query Attention, SwiGLU experts, MoE router with top-k
  gating, optional always-on "shared experts", load-balancing auxiliary loss.
- **Research techniques in scope for comparison** (Requirement 12):
  positional embedding (RoPE vs NoPE vs ALiBi), expert activation (SwiGLU vs
  GEGLU vs GELU), MoE balancing (auxiliary-loss vs DeepSeek-V3-style
  auxiliary-loss-free bias balancing, Wang et al. 2024). Stretch goals:
  simplified Multi-Head Latent Attention (MLA, DeepSeek-V2/V3) and a LoRA
  adapter fine-tuning demo.
- **Process convention:** every idea, decision, and implementation step gets
  a dated journal entry under `/journey_of_a_model/`, written as a
  beginner-friendly explainer with diagrams and links to source material
  (Requirement 13).
- **Scope is one combined spec** covering architecture, training, inference,
  visualization, comparison studies, and tests, since it's one cohesive
  learning project.

## Requirements

### Requirement 1: From-Scratch Transformer Backbone
**User Story:** As a learner building an LLM from scratch, I want a
Qwen-style transformer decoder implemented in raw PyTorch, so that I can see
and understand every matrix operation involved in a forward pass.

#### Acceptance Criteria
1. WHEN the model package is imported THEN the system SHALL expose an
   `nn.Module` implementing a decoder-only transformer built only from
   primitive PyTorch ops (`nn.Linear`, `nn.Embedding`, tensor ops) — no calls
   to `transformers`, `AutoModel`, or other pretrained-model libraries.
2. WHEN a token sequence tensor of shape `(batch, seq_len)` is passed to the
   model THEN the system SHALL return logits of shape
   `(batch, seq_len, vocab_size)`.
3. WHEN computing attention THEN the system SHALL apply Rotary Position
   Embeddings (RoPE) to query and key vectors via explicit rotation math
   implemented in this project (no third-party RoPE library).
4. WHEN normalizing activations THEN the system SHALL use a from-scratch
   RMSNorm implementation, applied in pre-norm position around the attention
   and MoE sublayers.
5. WHEN computing self-attention THEN the system SHALL support
   Grouped-Query Attention with a configurable `num_kv_heads <= num_heads`,
   including the `num_kv_heads == num_heads` (standard multi-head) case.
6. WHEN computing attention scores for query position `i` THEN the system
   SHALL mask out all key positions `j > i` so no future token influences
   the prediction at position `i`.
7. IF `tie_word_embeddings` is enabled in the model config THEN the system
   SHALL reuse the input embedding matrix as the output projection weight.

### Requirement 2: Mixture-of-Experts Feed-Forward Layer
**User Story:** As a learner, I want the transformer's feed-forward sublayer
replaced by a Mixture-of-Experts layer with a trainable router, so that I
can study sparse expert routing the way Qwen's MoE models do it.

#### Acceptance Criteria
1. WHEN an MoE layer is constructed THEN the system SHALL create
   `num_routed_experts` independent SwiGLU expert networks plus an optional
   `num_shared_experts` experts that are always active for every token.
2. WHEN a token's hidden state reaches the MoE layer THEN the system SHALL
   compute router logits via a learned linear layer, apply softmax, and
   select the top-`k` routed experts for that token (`k` configurable,
   default 2).
3. WHEN the top-`k` routed experts are selected THEN the system SHALL
   combine their outputs weighted by their renormalized softmax gate
   values, and add the output of any shared experts unweighted.
4. WHEN a training batch is processed THEN the system SHALL compute a
   load-balancing auxiliary loss penalizing uneven routing across experts,
   exposed separately from the main language-modeling loss.
5. IF a routed expert receives zero tokens in a batch THEN the system SHALL
   NOT error, and that expert's parameters SHALL simply receive no gradient
   for that batch.

### Requirement 3: Laptop-Friendly, Configurable Model Sizing
**User Story:** As a learner training on my own laptop (no cluster, no
dedicated training GPU), I want the model size and compute device to be
config-driven, so that I can run the whole pipeline locally in a bounded
amount of time.

#### Acceptance Criteria
1. WHEN the model is constructed THEN the system SHALL read all
   architectural hyperparameters (layer count, hidden size, head count,
   kv-head count, expert count, top-k, context length, vocab size) from a
   single config object rather than hard-coded literals.
2. WHEN no device is explicitly specified THEN the system SHALL select
   `mps` if `torch.backends.mps.is_available()`, else `cpu`.
3. WHEN using the default configuration THEN the total parameter count SHALL
   stay under 100M total / under 30M active parameters per token, so that a
   full training run (Requirement 5) completes on the detected hardware
   without a discrete/cloud GPU.
4. WHERE a lower-memory environment is detected THE SYSTEM SHALL still run
   the default config without out-of-memory errors on a machine with 16GB+
   system memory.

### Requirement 4: From-Scratch Tokenizer & Data Pipeline
**User Story:** As a learner, I want a tokenizer and data-loading pipeline
that I build myself rather than importing a pretrained tokenizer, so that I
understand how raw text becomes model input.

#### Acceptance Criteria
1. WHEN the data pipeline is run for the first time THEN the system SHALL
   download or generate a small (under 10MB) laptop-friendly text corpus
   suitable for language-model pretraining (default: tiny Shakespeare) and
   cache it locally.
2. WHEN training the tokenizer THEN the system SHALL derive its vocabulary
   from the corpus itself via a byte/character-level BPE implemented in this
   project, with no dependency on a pretrained vocab file.
3. WHEN encoding text THEN the system SHALL convert a string into a list of
   integer token IDs; WHEN decoding THEN the system SHALL convert token IDs
   back into the original string losslessly for all characters in the
   corpus.
4. WHEN preparing training data THEN the system SHALL split the tokenized
   corpus into train/validation sets and serve fixed-length context windows
   via a PyTorch `Dataset`/`DataLoader`.

### Requirement 5: Training Loop
**User Story:** As a learner, I want a training script that optimizes the
model on the prepared dataset with visible progress, so that I can watch the
model learn and reproduce the run later.

#### Acceptance Criteria
1. WHEN training starts THEN the system SHALL run an AdamW optimizer with a
   configurable learning-rate schedule (e.g. warmup + cosine decay).
2. WHEN each logging interval elapses THEN the system SHALL log step number,
   language-modeling loss, MoE auxiliary loss, and elapsed time.
3. WHEN a configurable number of steps has elapsed THEN the system SHALL
   evaluate on the validation split and log validation loss/perplexity.
4. WHEN a configurable checkpoint interval is reached THEN the system SHALL
   save model weights, optimizer state, and config to disk, and SHALL be
   able to resume training from such a checkpoint.
5. WHEN training completes on the default configuration THEN the resulting
   checkpoint SHALL generate recognizably non-random text, with validation
   loss trending downward from its initial value.

### Requirement 6: Text Generation / Inference
**User Story:** As a learner, I want to load a trained checkpoint and
generate text from a prompt, so that I can see the model I built actually
produce output.

#### Acceptance Criteria
1. WHEN a checkpoint path and text prompt are provided THEN the system SHALL
   tokenize the prompt, run autoregressive generation, and print the decoded
   text.
2. WHEN generating THEN the system SHALL support configurable temperature,
   top-k, and top-p (nucleus) sampling.
3. WHEN the generated sequence reaches the model's max context length or a
   configured max-new-tokens limit THEN the system SHALL stop generation.
4. WHEN no GPU/MPS is available THEN the system SHALL still run generation
   correctly on CPU.

### Requirement 7: Manim Visualizations of Architectural Math Operations
**User Story:** As a learner, I want an animated, step-by-step visualization
of the actual math happening in each part of the architecture, so that I can
build intuition instead of just reading formulas.

#### Acceptance Criteria
1. WHEN a visualization scene is rendered for an architectural component
   THEN the system SHALL use Manim (Community Edition) and SHALL source the
   numbers it animates from a real forward pass on a small example input
   (not illustrative/fabricated numbers).
2. WHEN the embedding/positional-encoding scene is rendered THEN the system
   SHALL animate the token-to-vector lookup and the RoPE rotation applied to
   a query/key vector.
3. WHEN the attention scene is rendered THEN the system SHALL animate the
   Q·Kᵀ score matrix, the causal mask being applied, the softmax
   normalization, and the weighted sum with V.
4. WHEN the MoE routing scene is rendered THEN the system SHALL animate
   router-logit computation, softmax gate scores, top-k expert selection,
   and token dispatch to the chosen experts.
5. WHEN the RMSNorm/SwiGLU scene is rendered THEN the system SHALL animate
   the normalization scaling step and the gated SwiGLU activation inside an
   expert.
6. WHEN the output scene is rendered THEN the system SHALL animate the final
   linear projection to vocabulary logits and the softmax producing
   next-token probabilities.
7. WHEN any scene is invoked THEN the system SHALL be runnable independently
   via a documented `manim` CLI command, without requiring other scenes to
   run first.

### Requirement 8: Training/Run Observability Visualizations
**User Story:** As a learner, I want to see how training progresses and how
load gets balanced across experts, so that I can connect the math to real
training dynamics.

#### Acceptance Criteria
1. WHEN a training run has produced logged metrics THEN the system SHALL
   generate an animation of the loss curve (train and validation) over
   steps.
2. WHEN a training run has produced per-expert token counts THEN the system
   SHALL generate an animation showing expert-utilization distribution and
   how load-balancing changes it over time.
3. IF no training logs exist yet THEN the visualization scripts SHALL fail
   with a clear, actionable error message rather than a raw stack trace.

### Requirement 9: Correctness Unit Tests for Core Math
**User Story:** As a learner, I want automated tests for the from-scratch
math, so that I can trust the implementation before spending time training
it.

#### Acceptance Criteria
1. WHEN the test suite is run THEN the system SHALL verify RMSNorm output
   has the expected shape and unit root-mean-square (within floating-point
   tolerance).
2. WHEN the test suite is run THEN the system SHALL verify RoPE preserves
   vector norm and rotates a known input by the expected angle.
3. WHEN the test suite is run THEN the system SHALL verify the causal
   attention mask prevents a change to a later token from affecting an
   earlier position's output.
4. WHEN the test suite is run THEN the system SHALL verify the MoE router
   always selects exactly `k` experts per token and that selected gate
   weights sum to 1.
5. WHEN the test suite is run THEN the system SHALL verify the full model
   produces the correct output shape for at least two different
   batch/sequence-length combinations.

### Requirement 10: Project Setup & Run Instructions
**User Story:** As a learner returning to this project later, I want clear
instructions for environment setup and for running each stage, so that I
don't have to reverse-engineer my own project.

#### Acceptance Criteria
1. WHEN a contributor reads the README THEN the system SHALL document how
   to install dependencies with `uv` and how to run data prep, training,
   inference, tests, and each visualization scene.
2. WHEN dependencies are declared THEN the system SHALL manage them via
   `pyproject.toml`/`uv` (PyTorch, Manim, and any tokenizer/dataset helper
   libraries) so that `uv sync` alone reproduces a working environment.

### Requirement 11: Scaling-Law-Informed Training Budget
**User Story:** As a learner simulating a real research process, I want the
training token/epoch budget for each run derived from scaling-law reasoning
applied to the model's actual parameter counts, so that training choices are
principled rather than arbitrary.

#### Acceptance Criteria
1. WHEN a model config is finalized THEN the system SHALL compute and report
   its total parameter count and active parameters per token, broken down by
   component (embedding, attention, MoE experts, router).
2. WHEN planning a training run THEN the system SHALL compute a
   Chinchilla-style reference token budget as `D_ref ≈ 20 × active_params`
   and report it alongside the run's actually configured token budget for
   comparison.
3. WHEN the configured token budget requires more passes over the dataset
   than there are unique tokens available THEN the system SHALL compute and
   log the resulting epoch count, and SHALL warn if it exceeds 4 (the point
   past which repeated-data returns are known to diminish sharply, per
   Muennighoff et al. 2023).
4. WHEN estimating wall-clock time for a planned run THEN the system SHALL
   derive the estimate from a short empirical throughput calibration
   (measured tokens/sec on the actual device for the actual config) rather
   than a theoretical FLOPs estimate, since real MPS throughput is not
   reliably known in advance.
5. WHEN the comparison/ablation study (Requirement 12) is run THEN every
   compared configuration SHALL use the same configured token budget so
   comparisons are compute-matched.

### Requirement 12: Research-Style Architecture Comparison Framework
**User Story:** As a learner who wants to simulate a genuine research/
ablation study, I want to swap in alternative techniques for specific
architectural axes and compare them under a matched compute budget, so that
I can see how each choice actually changes model behavior.

#### Acceptance Criteria
1. WHEN a model config selects a positional-embedding strategy THEN the
   system SHALL support at least RoPE (default/baseline), NoPE (no
   positional embedding, relying on causal masking alone), and ALiBi
   (linear attention-bias positional encoding), selectable independently of
   other components.
2. WHEN a model config selects an expert/MLP activation THEN the system
   SHALL support at least SwiGLU (default/baseline), GEGLU, and plain
   (non-gated) GELU, selectable independently of other components.
3. WHEN a model config selects an MoE load-balancing strategy THEN the
   system SHALL support at least auxiliary-loss-based balancing
   (default/baseline, Requirement 2.4) and DeepSeek-V3-style
   auxiliary-loss-free bias-adjusted balancing (a per-expert bias added to
   routing scores and updated from observed expert load, requiring no
   gradient-based auxiliary loss).
4. WHERE the learner enables the attention stretch-goal THE SYSTEM SHALL
   additionally support a simplified Multi-Head Latent Attention (MLA)
   variant (low-rank joint KV compression into a latent vector, plus
   decoupled RoPE on a small sub-dimension) as an alternative to GQA.
5. WHERE the learner enables the adapter stretch-goal THE SYSTEM SHALL
   support fine-tuning a pretrained checkpoint with a LoRA (low-rank
   adapter) module on a small new dataset, and SHALL report a comparison
   against full fine-tuning on the same data (trainable-parameter count,
   loss, wall-clock time).
6. WHEN a comparison/ablation run is launched THEN the system SHALL train
   each selected configuration variant on an identical, fixed slice of
   tokens (Requirement 11.5) and log per-variant training/validation loss
   curves and expert-utilization stats to a common location.
7. WHEN all variants in a comparison run complete THEN the system SHALL
   produce a single summary (table and/or chart) comparing final validation
   loss/perplexity across variants.

### Requirement 13: Development Journal (`/journey_of_a_model`)
**User Story:** As a learner treating this like a real research project, I
want a running journal of every idea, decision, and implementation step, so
that the repo itself tells the story of how and why the project evolved,
like a lab notebook.

#### Acceptance Criteria
1. WHEN an architectural or experimental idea is considered (whether
   adopted or rejected) THEN the system SHALL add a dated markdown entry
   under `/journey_of_a_model/` describing the idea, the reasoning, and the
   outcome.
2. WHEN a coding task from `tasks.md` is completed THEN completing that task
   SHALL include adding a corresponding journal entry summarizing what was
   implemented and why.
3. WHEN a design or configuration decision is made that affects the model,
   training, or comparison methodology THEN the system SHALL record the
   options considered, trade-offs, and final choice as a journal entry.
4. WHEN a training or comparison run completes THEN the system SHALL record
   its config, token budget, and result summary as a journal entry.
5. WHEN journal entries accumulate THEN the system SHALL keep them as
   individually named, chronologically ordered files (e.g.
   `NNN-short-slug.md`) rather than one growing file, with a top-level
   `journey_of_a_model/README.md` indexing them in order.
6. WHEN a journal entry introduces a technical concept THEN the system
   SHALL write the explanation for a complete beginner: define every
   technical term before relying on it, and for each term give (a) what it
   means in plain English, (b) why it exists / what problem it solves, (c)
   a real-world analogy, (d) a simple technical explanation, and (e) how it
   relates to this project — never defining one unexplained term using
   another.
7. WHEN a journal entry introduces a technical concept THEN the system
   SHALL include at least one Mermaid diagram illustrating that concept,
   and a "Questions that might come to mind" section anticipating and
   answering likely beginner questions.
8. WHEN a journal entry describes a concept drawn from published research
   (a paper, technique, or named method) THEN the system SHALL include a
   link to that concept's original paper and, where a clear one was found
   via web search, a supplementary blog/explainer link, so the learner can
   read further.
9. WHEN a journal entry introduces a concept backed by a mathematical
   formula THEN the system SHALL derive that formula from first principles
   or from its stated design goal (not just present the final result),
   showing the intermediate steps, and SHALL state plainly which parts of
   the derivation are exact math versus an empirically-fitted constant.
   The entry SHALL also include a worked numeric example: concrete dummy
   values run through the formula step by step by hand, arriving at (and
   where practical, checking) a final numeric result.
