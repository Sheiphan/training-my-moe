---
name: refresh-site
description: Use when regenerating or validating the static pages in site/ (model figure, tensor walk, token journey, ablation, architecture map) after changing the model, docs, or site files. Trigger words: refresh site, regenerate site, check_site, model figure, tensor walk.
---

# Regenerate and validate `site/`

`site/` is plain HTML with vendored D3/KaTeX (`site/shared/lib/`). It must open from `file://` with no server, no build step and no runtime network fetch (the architecture map's Mermaid CDN script is the one existing exception). The generated files are committed.

## Which generator for which page

| Changed | Run |
|---|---|
| a `docs/math` doc, journal link, or code line a doc cites | `uv run python docs/math/build_architecture_footer.py` (-> `site/architecture-map/`) |
| model shape, parameter arithmetic, `site/shared/qwenmoe-model.js` | `uv run python scripts/render_model_figure.py` and `uv run python scripts/render_tensor_walk.py` (both need `node`, and verify parameter counts against the real model) |
| ablation results | `uv run python scripts/run_ablation_study.py --smoke` (-> `site/ablation/data/`) |
| token-journey traces | `uv run python scripts/export_trace.py` (-> `site/token-journey/data/`) |

The ablation and trace generators read git-ignored inputs (`runs/`, `.cache/data/`), so they need a local training run first. If those inputs are missing, say so and do not hand-edit generated data or invent numbers. Every figure on the site must come from a real run.

## Then always

1. `uv run python site/check_site.py` (add `--list` to see every referenced file). It checks that every referenced file exists **and is tracked by git** (staged counts), so `git add` new generated files before judging a failure.
2. Open `site/index.html` (or the changed page) in a browser from disk and look at it. The checker cannot see layout or JS errors.
3. Commit generated output together with the change that caused it.
