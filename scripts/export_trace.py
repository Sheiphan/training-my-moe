"""Exports real captured forward-pass traces to JS for `site/token-journey/`'s
web walkthrough -- one file per variant so the page can render a
structurally different diagram with correct real math for whichever
architecture is selected.

Sources every number from a real trained checkpoint -- activations via
`QwenMoEModel.forward_with_trace` (the project's one trace-capture path)
plus, for the matrix-multiplication deep-dive, real block-0 weight
matrices and real RMSNorm outputs read directly off the live model (the
same "local helper that also returns the model" pattern
`qwenmoe/viz/scenes_embedding.py` already established) -- never a
fabricated or illustrative number (Requirement 7.1 convention).

Two families of variant ship side by side:

- **classic** -- the four `SMOKE_TEST`-derived checkpoints
  `scripts/run_ablation_study.py` trains (d_model=64, 2 layers, 600
  steps), covering every real (attention, positional) pairing that study
  has real trained weights for: GQA x {RoPE, NoPE, ALiBi} and MLA x RoPE
  (MLA always applies its own decoupled RoPE regardless of
  `positional_scheme`, per design.md, so MLA x NoPE/ALiBi combinations
  don't exist as distinct trained variants).
- **modern** -- the three `moe-modernization` presets trained head to head
  in `notebooks/07_modernized_presets.ipynb` / journal entry 035
  (d_model=256, 6 layers, 305 steps): `BASELINE`, `BASELINE_V2`
  (QK-norm + sigmoid routing + width-aware init) and `FINEGRAINED`
  (DeepSeekMoE segmentation, 32 experts top-8). These are what make the
  spec's architectural changes visible in a real trace rather than only
  in a loss number.

Each variant is written to its own `data/trace_<key>.js` and listed in a
small `data/manifest.js`, so the page loads one variant at a time on
demand instead of parsing every variant's weights up front (the previous
single-file export reached 8.6 MB before the modern presets were added).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from qwenmoe.model.transformer import QwenMoEModel
from qwenmoe.tokenizer.base import build_tokenizer_from_meta
from qwenmoe.train.trainer import load_checkpoint

# key -> (label, family, checkpoint, blurb, expert_weights)
#
# `expert_weights` gates only the per-expert FFN weight matrices used by the
# matmul deep-dive. At d_model=256/d_ff=512 a single expert's three matrices
# are ~393K floats, so shipping them for the modern family would add tens of
# MB per variant to explain a step the classic family already covers; the
# modern family's own deep-dives (QK-norm, sigmoid routing, fine-grained
# dispatch) need none of them. Every *activation* is exported for every
# variant either way -- this drops weights, never real measured values.
VARIANTS: dict[str, tuple[str, str, str, str, bool]] = {
    "gqa_rope": (
        "GQA + RoPE",
        "classic",
        "runs/ablation_study/SMOKE_TEST/checkpoints/step_600.pt",
        "The from-scratch baseline: grouped-query attention with rotary position embedding.",
        True,
    ),
    "gqa_nope": (
        "GQA + NoPE",
        "classic",
        "runs/ablation_study/POS_NOPE/checkpoints/step_600.pt",
        "No positional operation at all -- order is carried only by the causal mask.",
        True,
    ),
    "gqa_alibi": (
        "GQA + ALiBi",
        "classic",
        "runs/ablation_study/POS_ALIBI/checkpoints/step_600.pt",
        "A fixed per-head distance penalty added straight to the attention scores.",
        True,
    ),
    "mla_rope": (
        "MLA",
        "classic",
        "runs/ablation_study/ATTN_MLA/checkpoints/step_600.pt",
        "Multi-head latent attention: K/V compressed to one small cached latent.",
        True,
    ),
    "baseline": (
        "BASELINE",
        "modern",
        "runs/modernized_presets/BASELINE/seed_1337/checkpoints/step_305.pt",
        "The original spec's full-size model: softmax routing, no QK-norm, init_std 0.02.",
        False,
    ),
    "baseline_v2": (
        "BASELINE_V2",
        "modern",
        "runs/modernized_presets/BASELINE_V2/seed_1337/checkpoints/step_305.pt",
        "Same shape, three changes: QK-norm, sigmoid routing, width-aware init.",
        False,
    ),
    "finegrained": (
        "FINEGRAINED",
        "modern",
        "runs/modernized_presets/FINEGRAINED/seed_1337/checkpoints/step_305.pt",
        "BASELINE_V2 plus DeepSeekMoE segmentation: 32 narrow experts, top-8.",
        False,
    ),
}
DEFAULT_VARIANT = "baseline_v2"

# Activations and weights are all O(1)-magnitude; 6 decimals is far below
# float32's own precision at this scale and roughly halves the file size.
_PLACES = 6


def _round(value):
    """Recursively rounds floats in nested lists to `_PLACES` decimals."""
    if isinstance(value, float):
        return round(value, _PLACES)
    if isinstance(value, list):
        return [_round(v) for v in value]
    return value


def _list(value):
    """Recursively converts tensors (and dicts of tensors) to plain lists."""
    if isinstance(value, torch.Tensor):
        return _round(value.tolist())
    if isinstance(value, dict):
        return {str(k): _list(v) for k, v in value.items()}
    return value


def _wT(linear):
    """(in_features, out_features) -- transposed from PyTorch's own
    (out, in) storage, so JS can do a plain row-vector * matrix."""
    return _list(linear.weight.T)


def _rms(vector: torch.Tensor) -> float:
    """Root-mean-square of a 1-D tensor -- exactly the quantity RMSNorm
    divides by, so `_rms(post_norm) == 1` (up to gamma) is the direct
    measurement of what QK-norm did."""
    return float(vector.pow(2).mean().sqrt())


def _qk_norm_effect(bundle, target: int) -> list | None:
    """Per-layer, per-head RMS of the followed token's q and k vectors
    *before* and *after* QK-norm (moe-modernization Requirement 7.3).

    This is the whole point of QK-norm made measurable: the pre-norm RMS
    drifts head to head and layer to layer, the post-norm RMS is pinned to
    the learned gamma scale, and the attention logits downstream inherit
    that bounded scale. `None` when the model has no QK-norm, in which case
    `LayerTrace.q_pre_norm` is `None` too.
    """
    if bundle.layers[0].q_pre_norm is None:
        return None
    out = []
    for layer in bundle.layers:
        out.append({
            "q_rms_pre": _round([_rms(h[target]) for h in layer.q_pre_norm]),
            "q_rms_post": _round([_rms(h[target]) for h in layer.q]),
            "k_rms_pre": _round([_rms(h[target]) for h in layer.k_pre_norm]),
            "k_rms_post": _round([_rms(h[target]) for h in layer.k]),
        })
    return out


def _routing_compare(bundle, target: int, scoring: str) -> list:
    """Both scoring rules evaluated on the *same real* router logits, per
    layer, for the followed token (moe-modernization Requirement 4.1-4.2).

    `softmax` couples every expert through one shared denominator;
    `sigmoid` scores each expert independently and only then renormalizes.
    Showing both from one set of real logits is what makes the difference
    legible -- neither column is fabricated, they are two real functions of
    the same real numbers, and `scoring` records which one this checkpoint
    actually trained with and used.
    """
    out = []
    for layer in bundle.layers:
        logits = layer.router_logits[target]
        softmax_probs = torch.softmax(logits, dim=-1)
        affinities = torch.sigmoid(logits)
        sigmoid_probs = affinities / affinities.sum()
        out.append({
            "logits": _list(logits),
            "softmax_probs": _list(softmax_probs),
            "sigmoid_probs": _list(sigmoid_probs),
            "used": scoring,
            "topk_indices": _list(layer.topk_indices[target]),
            "topk_weights": _list(layer.topk_weights[target]),
        })
    return out


def _layer_stats(bundle, n_routed_experts: int) -> list:
    """Per-layer summary statistics computed from this same real forward
    pass -- the numbers journal entry 035 reports, recomputed here so the
    page can plot them per architecture:

    - `attn_logit_std` / `attn_logit_absmax`: spread of the pre-mask
      attention scores. QK-norm's payoff is that these stay flat across
      depth instead of growing with the residual stream.
    - `router_entropy_norm`: Shannon entropy of the mean routing
      distribution, divided by `ln(n_experts)` so it lands in [0, 1] and
      stays comparable between an 8-expert and a 32-expert model.
    - `expert_load`: how many of this prompt's (token, slot) dispatch pairs
      each routed expert received.
    """
    out = []
    for layer in bundle.layers:
        scores = layer.attn_scores_pre_mask
        mean_probs = layer.router_probs.mean(dim=0)
        entropy = float(-(mean_probs * (mean_probs + 1e-12).log()).sum())
        load = torch.bincount(
            layer.topk_indices.flatten(), minlength=n_routed_experts
        )
        out.append({
            "attn_logit_std": round(float(scores.std()), _PLACES),
            "attn_logit_absmax": round(float(scores.abs().max()), _PLACES),
            "router_entropy_norm": round(entropy / math.log(n_routed_experts), _PLACES),
            "expert_load": load.tolist(),
        })
    return out


def _build_variant_data(checkpoint_path: str, prompt: str, expert_weights: bool) -> dict:
    checkpoint = load_checkpoint(checkpoint_path)
    cfg = checkpoint["model_config"]
    model = QwenMoEModel(cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    tokenizer = build_tokenizer_from_meta(checkpoint["tokenizer_meta"])
    vocab = [tokenizer.decode([i]) for i in range(tokenizer.vocab_size)]

    token_ids = tokenizer.encode(prompt)
    if len(token_ids) == 0:
        raise ValueError("prompt must encode to at least one token")
    input_ids = torch.tensor([token_ids], dtype=torch.long)
    with torch.no_grad():
        _, bundle = model.forward_with_trace(input_ids)

    # Follow the LAST token position -- its logits are what the model uses
    # to predict what comes next.
    target = len(bundle.token_ids) - 1

    layers = []
    for layer in bundle.layers:
        rope = layer.rope_angles_or_alibi_bias
        if rope is None:
            rope_full, rope_target = None, None
        elif rope.dim() == 2:  # RoPE (GQA full head_dim, or MLA's rope_head_dim slice)
            rope_full, rope_target = _list(rope), _list(rope[target])
        else:  # ALiBi: (n_heads, seq_len, seq_len)
            rope_full, rope_target = _list(rope), _list(rope[:, target, :])

        layers.append({
            "rope_angles_or_alibi_bias": rope_full,
            "q": _list(layer.q),
            "k": _list(layer.k),
            "v": _list(layer.v),
            "attn_scores_pre_mask": _list(layer.attn_scores_pre_mask),
            "attn_scores_post_softmax": _list(layer.attn_scores_post_softmax),
            "attn_output": _list(layer.attn_output),
            "router_logits": _list(layer.router_logits),
            "router_probs": _list(layer.router_probs),
            "topk_indices": _list(layer.topk_indices),
            "topk_weights": _list(layer.topk_weights),
            "expert_outputs": _list(layer.expert_outputs),
            "moe_output": _list(layer.moe_output),
            # Convenience slices for the ONE followed token -- saves the JS
            # from re-deriving index math the Python side already knows.
            "target": {
                "rope_angles_or_alibi_bias": rope_target,
                "q": _list(layer.q[:, target, :]),
                "k": _list(layer.k[:, target, :]),
                "v": _list(layer.v[:, target, :]),
                "q_pre_norm": (
                    None if layer.q_pre_norm is None else _list(layer.q_pre_norm[:, target, :])
                ),
                "k_pre_norm": (
                    None if layer.k_pre_norm is None else _list(layer.k_pre_norm[:, target, :])
                ),
                "attn_scores_post_softmax": _list(layer.attn_scores_post_softmax[:, target, :]),
                "attn_output": _list(layer.attn_output[target]),
                "router_probs": _list(layer.router_probs[target]),
                "topk_indices": _list(layer.topk_indices[target]),
                "topk_weights": _list(layer.topk_weights[target]),
                "moe_output": _list(layer.moe_output[target]),
            },
        })

    # Real block-0 weight matrices + real RMSNorm outputs, for the
    # matrix-multiplication deep-dive (DEEP_DIVE_PLAN.md). Read directly
    # off the live model -- never re-derived or approximated.
    with torch.no_grad():
        block0 = model.blocks[0]
        x0 = bundle.embeddings[target : target + 1]
        attn_norm_out = block0.attn_norm(x0).squeeze(0)
        x_after_attn = bundle.embeddings[target] + bundle.layers[0].attn_output[target]
        moe_norm_out = block0.moe_norm(x_after_attn.unsqueeze(0)).squeeze(0)

    chosen_experts = sorted(set(bundle.layers[0].topk_indices[target].tolist()))
    experts_weights = {}
    if expert_weights:
        for idx in chosen_experts:
            e = block0.moe.routed_experts[idx]
            experts_weights[str(idx)] = {
                "w_gate": _wT(e.w_gate), "w_up": _wT(e.w_up), "w_down": _wT(e.w_down)
            }
        for i, e in enumerate(block0.moe.shared_experts):
            experts_weights[f"shared{i}"] = {
                "w_gate": _wT(e.w_gate), "w_up": _wT(e.w_up), "w_down": _wT(e.w_down)
            }

    attn = block0.attn
    if cfg.attention_variant == "mla":
        attn_weights = {
            "w_dkv": _wT(attn.w_dkv),
            "w_uk": _wT(attn.w_uk),
            "w_uv": _wT(attn.w_uv),
            "w_q": _wT(attn.w_q),
            "w_qr": _wT(attn.w_qr),
            "w_kr": _wT(attn.w_kr),
            "w_o": _wT(attn.w_o),
        }
    else:
        attn_weights = {
            "w_q": _wT(attn.w_q),
            "w_k": _wT(attn.w_k),
            "w_v": _wT(attn.w_v),
            "w_o": _wT(attn.w_o),
        }

    weights = {
        "attn_norm_gamma": _list(block0.attn_norm.weight),
        "moe_norm_gamma": _list(block0.moe_norm.weight),
        "final_norm_gamma": _list(model.final_norm.weight),
        "attn_norm_output_target": _list(attn_norm_out),
        "moe_norm_output_target": _list(moe_norm_out),
        **attn_weights,
        "router_w": _wT(block0.moe.router.gate),
        "chosen_experts": chosen_experts,
        "experts": experts_weights,
        "has_expert_weights": expert_weights,
        "embedding_full": _list(model.embedding.weight),
    }
    # QK-norm's own learned per-head gamma, when present -- the scale the
    # post-norm RMS is pinned to.
    if getattr(attn, "qk_norm", False):
        weights["q_norm_gamma"] = _list(attn.q_norm.weight)
        weights["k_norm_gamma"] = _list(attn.k_norm.weight)

    return {
        "prompt": prompt,
        "vocab": vocab,
        "config": {
            "vocab_size": cfg.vocab_size,
            "d_model": cfg.d_model,
            "n_layers": cfg.n_layers,
            "n_heads": cfg.n_heads,
            "n_kv_heads": cfg.n_kv_heads,
            "d_ff": cfg.d_ff,
            "n_shared_experts": cfg.n_shared_experts,
            "n_routed_experts": cfg.n_routed_experts,
            "top_k": cfg.top_k,
            "positional_scheme": cfg.positional_scheme,
            "activation": cfg.activation,
            "moe_balancing": cfg.moe_balancing,
            "attention_variant": cfg.attention_variant,
            "mla_latent_dim": cfg.mla_latent_dim,
            "tie_word_embeddings": cfg.tie_word_embeddings,
            "rope_theta": cfg.rope_theta,
            "router_scoring": cfg.router_scoring,
            "qk_norm": cfg.qk_norm,
            "init_std": cfg.init_std,
            "mtp_depth": cfg.mtp_depth,
            "aux_loss_weight": cfg.aux_loss_weight,
        },
        "token_ids": bundle.token_ids,
        "tokens": [vocab[i] for i in bundle.token_ids],
        "target_position": target,
        "target_token": vocab[bundle.token_ids[target]],
        "embeddings": _list(bundle.embeddings),
        "layers": layers,
        "weights": weights,
        "final_norm_output": _list(bundle.final_norm_output),
        "logits": _list(bundle.logits),
        "probs": _list(bundle.probs),
        "target_embedding": _list(bundle.embeddings[target]),
        "target_final_norm_output": _list(bundle.final_norm_output[target]),
        # -- analysis blocks: real measurements of what the modernization
        # changed, recomputed from this same forward pass --
        "qk_norm_effect": _qk_norm_effect(bundle, target),
        "routing_compare": _routing_compare(bundle, target, cfg.router_scoring),
        "layer_stats": _layer_stats(bundle, cfg.n_routed_experts),
    }


def export_trace(prompt: str, output_dir: str) -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_variants = {}
    for key, (label, family, checkpoint_path, blurb, expert_weights) in VARIANTS.items():
        if not Path(checkpoint_path).exists():
            print(f"  SKIP {key}: no checkpoint at {checkpoint_path}")
            continue
        data = _build_variant_data(checkpoint_path, prompt, expert_weights)
        filename = f"trace_{key}.js"
        # A plain .json file can't be `fetch()`-ed from a page opened via
        # file:// (browsers block XHR/fetch across file:// origins) -- so
        # each variant ships as a <script>-loadable global instead, the same
        # trick as every other no-server, no-build file in this project.
        # Unlike `fetch`, a dynamically injected <script> IS allowed under
        # file://, which is what lets the page load these on demand.
        body = json.dumps(data)
        (out_dir / filename).write_text(
            "window.TRACE_VARIANTS = window.TRACE_VARIANTS || {};\n"
            f"window.TRACE_VARIANTS[{json.dumps(key)}] = {body};\n"
        )
        size_kb = (out_dir / filename).stat().st_size / 1024
        cfg = data["config"]
        manifest_variants[key] = {
            "label": label,
            "family": family,
            "blurb": blurb,
            "file": filename,
            "size_kb": round(size_kb, 1),
            "checkpoint": checkpoint_path,
            "config": cfg,
            "has_expert_weights": expert_weights,
        }
        probs = data["probs"]
        top_id = max(range(len(probs)), key=lambda i: probs[i])
        print(
            f"  {key:<12} {family:<8} {size_kb:>8.1f} KB  "
            f"qk_norm={str(cfg['qk_norm']):<5} routing={cfg['router_scoring']:<7} "
            f"experts={cfg['n_routed_experts']:>2}/top-{cfg['top_k']}  "
            f"next token {data['vocab'][top_id]!r} p={probs[top_id]:.4f}"
        )

    if DEFAULT_VARIANT in manifest_variants:
        default = DEFAULT_VARIANT
    else:
        default = next(iter(manifest_variants))
    manifest = {"prompt": prompt, "default": default, "variants": manifest_variants}
    (out_dir / "manifest.js").write_text(
        f"window.TRACE_MANIFEST = {json.dumps(manifest, indent=1)};\n"
    )
    print(f"wrote {out_dir}/manifest.js ({len(manifest_variants)} variants, default {default})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default="ROMEO:")
    parser.add_argument("--output-dir", default="site/token-journey/data")
    args = parser.parse_args()
    export_trace(args.prompt, args.output_dir)


if __name__ == "__main__":
    main()
