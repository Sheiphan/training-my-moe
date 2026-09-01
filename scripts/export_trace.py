"""Exports real captured forward-pass traces to JSON for
`token_journey/`'s web walkthrough -- one per (attention, positional)
variant so the page can render a structurally different diagram with
correct real math for whichever combination is toggled.

Sources every number from a real trained checkpoint -- activations via
`QwenMoEModel.forward_with_trace` (the same path every Manim scene uses)
plus, for the matrix-multiplication deep-dive, real block-0 weight
matrices and real RMSNorm outputs read directly off the live model (the
same "local helper that also returns the model" pattern
`qwenmoe/viz/scenes_embedding.py` already established) -- never a
fabricated or illustrative number (Requirement 7.1 convention).

The 4 variants below reuse the real checkpoints `scripts/run_ablation_
study.py` already trained (identical fixed token slice, identical
600-step budget) -- covering every real (attention, positional) pairing
that study has real trained weights for: GQA×{RoPE,NoPE,ALiBi} and
MLA×RoPE (MLA always applies its own decoupled RoPE regardless of
`positional_scheme`, per design.md, so MLA×NoPE/ALiBi combinations don't
exist as distinct trained variants).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from qwenmoe.model.transformer import QwenMoEModel
from qwenmoe.tokenizer.base import build_tokenizer_from_meta
from qwenmoe.train.trainer import load_checkpoint

VARIANTS = {
    "gqa_rope": "runs/ablation_study/SMOKE_TEST/checkpoints/step_600.pt",
    "gqa_nope": "runs/ablation_study/POS_NOPE/checkpoints/step_600.pt",
    "gqa_alibi": "runs/ablation_study/POS_ALIBI/checkpoints/step_600.pt",
    "mla_rope": "runs/ablation_study/ATTN_MLA/checkpoints/step_600.pt",
}
DEFAULT_VARIANT = "gqa_rope"


def _list(value):
    """Recursively converts tensors (and dicts of tensors) to plain lists."""
    if isinstance(value, torch.Tensor):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _list(v) for k, v in value.items()}
    return value


def _wT(linear):
    """(in_features, out_features) -- transposed from PyTorch's own
    (out, in) storage, so JS can do a plain row-vector * matrix."""
    return _list(linear.weight.T)


def _build_variant_data(checkpoint_path: str, prompt: str) -> dict:
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
    for idx in chosen_experts:
        e = block0.moe.routed_experts[idx]
        experts_weights[str(idx)] = {"w_gate": _wT(e.w_gate), "w_up": _wT(e.w_up), "w_down": _wT(e.w_down)}
    for i, e in enumerate(block0.moe.shared_experts):
        experts_weights[f"shared{i}"] = {"w_gate": _wT(e.w_gate), "w_up": _wT(e.w_up), "w_down": _wT(e.w_down)}

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
        "embedding_full": _list(model.embedding.weight),
    }

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
    }


def export_trace(prompt: str, output_path: str) -> None:
    variants_out = {}
    for key, checkpoint_path in VARIANTS.items():
        variants_out[key] = _build_variant_data(checkpoint_path, prompt)

    data = {"prompt": prompt, "default": DEFAULT_VARIANT, "variants": variants_out}

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # A plain .json file can't be `fetch()`-ed from a page opened via
    # file:// (browsers block XHR/fetch across file:// origins) -- so this
    # ships as a plain <script>-loadable global instead, same trick as
    # every other no-server, no-build file in this project.
    out_path.write_text(f"window.TRACE_DATA = {json.dumps(data)};\n")
    size_kb = out_path.stat().st_size / 1024
    print(f"wrote {out_path} ({size_kb:.1f} KB, {len(variants_out)} variants)")
    for key, v in variants_out.items():
        probs = v["probs"]
        top_id = max(range(len(probs)), key=lambda i: probs[i])
        print(
            f"  {key:<10} attention={v['config']['attention_variant']:<4} "
            f"positional={v['config']['positional_scheme']:<6} "
            f"top predicted next token: {v['vocab'][top_id]!r} at p={probs[top_id]:.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default="ROMEO:")
    parser.add_argument("--output", default="token_journey/data/trace.js")
    args = parser.parse_args()
    export_trace(args.prompt, args.output)


if __name__ == "__main__":
    main()
