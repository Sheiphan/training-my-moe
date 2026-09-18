# training-my-moe

A Qwen-/DeepSeek-V2-era **Mixture-of-Experts decoder-only transformer, written from scratch** in PyTorch —
then modernized, ablated, trained, and visualized.

Nothing here wraps a model library. Every component (RMSNorm, RoPE/ALiBi/NoPE, GQA and MLA attention,
SwiGLU/GEGLU/GELU experts, the MoE router and its load balancers, the training loop, the tokenizers, the
sampling code) is implemented in `qwenmoe/` and covered by tests.

The other half of the project is **explaining** it: a 41-entry engineering journal, an 9-notebook runnable
tour, and three interactive pages that open straight from disk with no server and no build step.

> **One convention runs through everything: no fabricated numbers.** Every figure in the journal, the
> notebooks and the visualizations is read from a real training log or captured from a real forward pass
> through a real trained checkpoint. Where a value is illustrative or a comparison is not apples-to-apples,
> the text says so at the point of use.

---

## Quickstart

Requires Python ≥ 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                                  # install
uv run pytest -q                         # 492 tests

# prepare data, train the fast smoke config, generate from it
uv run python scripts/prepare_data.py --corpus tiny_shakespeare --tokenizer-type char
uv run python scripts/train.py --preset SMOKE_TEST --corpus tiny_shakespeare \
    --tokenizer-type char --total-tokens 200000 --output-dir runs/smoke_test
uv run python scripts/generate.py --checkpoint runs/smoke_test/checkpoints/step_1000.pt \
    --prompt "ROMEO:" --max-new-tokens 120 --top-p 0.9
```

To just *look* at the thing, open `token_journey/index.html` in a browser — no install required.

---

## The three interactive pages

Plain HTML + vendored D3/KaTeX. Open the file directly; they work offline and over `file://`.

| Page | Question it answers |
|---|---|
| [`token_journey/index.html`](token_journey/index.html) | **What happens to one token?** Step through a real forward pass of any of **7 real trained architectures**, one operation at a time, with the real activations, attention weights and routing decisions at every step. The walkthrough itself changes shape per architecture — a QK-norm model has a step a non-QK-norm model does not. |
| [`token_journey/architecture.html`](token_journey/architecture.html) | **What shape is this model?** A live schematic of the whole network driven by togglable `ModelConfig` fields, with exact parameter-count and KV-cache arithmetic. Needs no checkpoint, so you can build any combination — and when a combination *does* have a trained checkpoint, it hands off into the walkthrough for it. |
| [`ablation_study/index.html`](ablation_study/index.html) | **What does each choice actually cost?** 14 real compute-matched training runs that each change exactly one architectural axis, plus a separate multi-seed showdown of the three recommended presets with real seed-noise bands. |

Regenerate their data with:

```bash
uv run python scripts/run_ablation_study.py --smoke   # -> ablation_study/data/
uv run python scripts/export_trace.py                 # -> token_journey/data/
```

---

## Repository map

```
qwenmoe/            the model and training code
  config.py           ModelConfig / DataConfig / TrainConfig -- every knob, one place
  model/              rmsnorm, positional, attention, moe, block, transformer,
                      kv_cache, mtp, generate, trace
  tokenizer/          char + from-scratch byte-level BPE
  data/               download, tokenize, cache, fixed-slice sampling
  train/              trainer, optimizers (AdamW/Muon), schedules, scaling laws, compare
  viz/                trace capture + scene definitions
configs/presets.py  named ModelConfigs (see below)
scripts/            prepare_data, train, generate, run_comparison,
                    run_ablation_study, export_trace, render_viz
notebooks/          runnable code tour, 00 is the map
journey_of_a_model/ 41 engineering-journal entries + figures
tests/              492 tests
```

## Presets

`configs/presets.py`. `SMOKE_TEST` and `BASELINE` are the original spec's; the two below them are the
recommended configurations after the `moe-modernization` spec.

| Preset | Shape | Notes |
|---|---|---|
| `SMOKE_TEST` | d_model 64, 2 layers, 4 experts top-2 | Sized so the whole pipeline runs end to end in minutes. Every ablation variant extends this. |
| `BASELINE` | d_model 256, 6 layers, 8 experts top-2 | ~23.5M total / ~9.3M active params at vocab 4096. |
| **`BASELINE_V2`** | same shape as `BASELINE` | **+ QK-norm, sigmoid routing, width-aware init.** Beats `BASELINE` by **−0.072 validation loss** (3× the measured seed-noise band) for **+384 parameters**. |
| `FINEGRAINED` | 32 experts of d_ff 128, top-8 | `BASELINE_V2` + DeepSeekMoE expert segmentation, holding both the total and per-token-active FFN budget constant. Measured *worse* at this scale — see journal 035 for why that is the honest result. |

Plus one-axis-off variants used by the ablation study (`POS_NOPE`, `POS_ALIBI`, `ACT_GEGLU`, `ACT_GELU`,
`MOE_LOSSFREE`, `ROUTE_SIGMOID`, `ATT_QKNORM`, `MTP_D1`, `MOE_FINEGRAINED`).

---

## The notebooks

A runnable companion to the codebase — start at [`notebooks/00_overview.ipynb`](notebooks/00_overview.ipynb),
which maps the repo and runs a 60-second end-to-end tour.

| # | Covers |
|---|---|
| 00 | project map + end-to-end tour + the modernized presets |
| 01–03 | `config.py` / `presets.py`, tokenizers, data pipeline |
| 04–06 | the model internals, the training stack, trace capture and rendering |
| 07 | the real `BASELINE` vs `BASELINE_V2` vs `FINEGRAINED` training run behind journal 035 |
| — | [`rope_deep_dive.ipynb`](notebooks/rope_deep_dive.ipynb), a standalone interactive look at rotary embeddings |

## The journal

[`journey_of_a_model/`](journey_of_a_model/) — one entry per implementation step, in order, each defining its
terms, deriving its formulas with worked numeric examples, linking the source papers, and answering the
questions the work actually raised. [Its README](journey_of_a_model/README.md) is the index.

Recent entries: **034** picks the modernized defaults and says why; **035** trains them head to head and
reports that one of the two loses; **036** rebuilds the visualizations around them.

## The specs

This was built spec-first, and both specs are complete:

- [`.kiro/specs/qwen-moe-from-scratch/`](.kiro/specs/qwen-moe-from-scratch/) — the original model.
- [`.kiro/specs/moe-modernization/`](.kiro/specs/moe-modernization/) — 18/18 tasks bringing it up to
  2024–2026 practice: Muon, WSD schedules, bf16, sigmoid routing and router z-loss, batched expert dispatch,
  multi-token prediction, QK-norm, a KV cache, principled init, document-boundary masking, and multi-seed /
  parameter-matched ablations. **Every one of them defaults to the pre-spec behavior**, so a config written
  before the spec still trains bit-identically.

## Testing

```bash
uv run pytest -q      # 386 passed
```

The suite pins the invariant that matters most here: every modernization knob is off by default, and a model
built from a pre-spec config produces bit-identical results to before the spec landed.
