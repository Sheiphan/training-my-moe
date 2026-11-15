# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Qwen-/DeepSeek-V2-era Mixture-of-Experts decoder-only transformer written from scratch in PyTorch (`qwenmoe/`), plus the material that explains it: an engineering journal (`journey_of_a_model/`), hand-derived math docs (`docs/math/`), notebooks, and static HTML pages (`site/`). No model library is wrapped. Python >= 3.12, managed with `uv`.

**Core convention: no fabricated numbers.** Every figure in the docs, notebooks and site comes from a real training log or a real forward pass through a real checkpoint. When a value is illustrative or a comparison is not apples-to-apples, the text says so where it is used. Keep to this when editing any doc.

## Commands

```bash
uv sync
uv run pytest -q                              # ~543 tests
uv run pytest tests/test_moe.py -q            # one file
uv run pytest tests/test_moe.py::test_name -q # one test

# end-to-end smoke path (char-level tiny_shakespeare)
uv run python scripts/prepare_data.py --corpus tiny_shakespeare --tokenizer-type char
uv run python scripts/train.py --preset SMOKE_TEST --corpus tiny_shakespeare \
    --tokenizer-type char --total-tokens 200000 --output-dir runs/smoke_test
uv run python scripts/generate.py --checkpoint runs/smoke_test/checkpoints/step_1000.pt --prompt "ROMEO:"

uv run python docs/math/verify_005_moe_layer.py   # one math-doc verify script (run from repo root)
uv run python docs/math/run_all_verify.py         # all of them; rewrites docs/math/verify-status.md, exits 1 on failure
uv run python site/check_site.py                  # every file the site/ pages reference exists and is git-tracked
uv run python scripts/check_capacity.py --preset SCALE_300M   # will this config fit in memory on this machine?
uv run python scripts/watch_run.py runs/<name> [--once]       # health check of a live run's logs/train.jsonl
```

There is no linter or formatter configured. `main.py` is a placeholder.

`scripts/train.py` documents its long-run flags (WSD schedule, `--keep-last`/`--keep-every` checkpoint pruning, `--resume-from`, `--allow-schedule-change`) in its module docstring. Read that before launching a long run.

## Architecture

**Config is the single source of truth.** `qwenmoe/config.py` holds `ModelConfig`, `TrainConfig`, `DataConfig` as plain dataclasses with validation in `__post_init__`. There is no YAML or JSON layer. Named configs live in `configs/presets.py` as plain Python built with `dataclasses.replace`: `SMOKE_TEST` and `BASELINE` (original spec), `BASELINE_V2` and `FINEGRAINED` (modernized), `SCALE_100M` and `SCALE_300M` (scale-ups), and one-axis-off ablation variants (`POS_NOPE`, `ACT_GEGLU`, `MOE_LOSSFREE`, ...). Sizing constants such as `init_std` are recomputed per preset, so check them when adding a preset.

**Every modernization knob defaults to the pre-spec behavior.** Muon, WSD, bf16, sigmoid routing, router z-loss, MTP, QK-norm, KV cache, document masking, gradient accumulation and the rest are all off by default. A model built from an old config must stay bit-identical. Tests pin this. When adding a feature, add it as an off-by-default field and add a test that the default path is unchanged.

**Two forward paths per attention/MoE module.** `forward` is the fast production path (fused `F.scaled_dot_product_attention`, batched expert dispatch). `forward_with_trace` is an explicit, slower mirror that records intermediates into `TraceBundle`/`LayerTrace` (`qwenmoe/model/trace.py`). It is used by `qwenmoe/viz/`, `scripts/export_trace.py`, and the math-doc verify scripts. Both are mirrored across `Attention`, `MLAAttention`, `MoELayer`, `TransformerBlock` and `QwenMoEModel`. A change to the math in one path must be made in the other, and the two are tested to agree numerically.

**Layout of `qwenmoe/`:**
- `model/`: `rmsnorm`, `positional` (RoPE/NoPE/ALiBi behind one hook interface), `attention` (GQA and MLA), `moe` (router, experts, shared experts, `AuxLossBalancer` and `LossFreeBalancer`), `block`, `transformer`, `mtp`, `kv_cache`, `generate`
- `tokenizer/`: `CharTokenizer` and a from-scratch byte-level BPE, behind a `build_tokenizer` factory
- `data/`: corpus download, tokenize, cache, `TokenDataset`, fixed-slice and step-offset samplers
- `train/`: `trainer.py` (the large one: `Trainer`, `lr_multiplier`, checkpointing and pruning, JSON-lines logging), `optim.py` (AdamW param groups, Muon, `CombinedOptimizer`), `scaling.py` (param counts, Chinchilla budget), `compare.py` (compute-matched comparisons)
- `viz/`: trace capture (`capture_trace`), used by `scripts/export_trace.py` and the trace tests

`scripts/train.py` builds `TrainConfig` by matching argparse destinations to dataclass field names, and a test fails if any `TrainConfig` field becomes unreachable from the CLI. Adding a `TrainConfig` field means adding a flag for it.

**Gitignored inputs.** `runs/` (checkpoints and logs), `.cache/` (prepared corpora) and `content/` are not tracked. The generated site data is committed, but regenerating it (`scripts/render_model_figure.py`, `render_tensor_walk.py`, `run_ablation_study.py`, `export_trace.py`, `docs/math/build_architecture_footer.py`) needs a local training run first. The two `render_*` scripts also shell out to `node`. `check_site.py` should pass after any change under `site/`.

## Branches are versions

Branches stack as project versions: root -> `moe-modernization` -> `scaling-improvements` -> `cuda-h100` -> `docs-and-site`, with `model-development` as a separate side branch. `main` is the GitHub default branch and follows the latest tip (`docs-and-site`). Manim was removed from the project's latest code; the older version branches still contain it as history. The plan is one squashed commit per version, which rewrites every branch above the one squashed. Use the `branch-versions` skill before squashing, rebasing or adding a branch.

## Docs conventions (journal and math docs)

- `journey_of_a_model/NNN-*.md` is one numbered entry per implementation step, indexed in its `README.md`. Entries 001-044 map 1:1 to tasks in `.kiro/specs/*/tasks.md`, which hold the requirements, design and task lists the project was built from. The journal's documentation standard is in that README: define every term, include a Mermaid diagram, a "Questions that might come to mind" section and sourced paper links, and give full derivations with worked numeric examples.
- `docs/math/NNN-*.md` mirrors a journal number and carries a runnable `verify_NNN_*.py` that pins the doc's hand-computed numbers into the real `qwenmoe/` module and asserts them. A number in a math doc that its verify script doesn't reproduce is a bug in the doc. After editing a doc's numbers or the code it describes, run its verify script.
- Docs link to code as relative paths with a line anchor (e.g. `qwenmoe/model/block.py#L37`). Moving code can break those line anchors, so check the links in docs that cite code you changed.
- `site/` pages are plain HTML with vendored D3/KaTeX under `site/shared/lib/` and open from `file://` with no server or build step. Don't introduce a bundler or runtime fetches that break offline use.
