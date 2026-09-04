#!/usr/bin/env python
"""Runs a real, compute-matched ablation study across the model's main
architectural axes and exports the results for `ablation_study/`'s web
page -- same "never fabricate a number" convention as
`scripts/export_trace.py`.

Every variant changes exactly ONE axis relative to `SMOKE_TEST` and trains
on the identical fixed token slice / token budget via
`qwenmoe.train.compare.run_comparison` (Requirements 11.5, 12.6), so any
metric difference reflects only that axis. Axes that live on `TrainConfig`
rather than `ModelConfig` (`OPT_MUON` -> `optimizer="muon"`, `SCHED_WSD` ->
`lr_schedule="wsd"`) are passed as `run_comparison`'s per-variant
`train_overrides` (moe-modernization Requirement 12.5). `ATTN_MLA` and
`TIE_FALSE` are ad-hoc overrides covering two axes the architecture toggle
bar (architecture_diagram.html) exposes; `mla_latent_dim=16` matches the
value it illustrates for SMOKE_TEST (`latentDim = headDim = 16`).

The default corpus is TinyStories with the from-scratch BPE tokenizer
(parent spec Tier 2, Requirement 12.3); `--smoke` runs the fast
char-level tiny-Shakespeare version instead.

Usage:
    uv run python scripts/prepare_data.py --corpus tiny_stories --tokenizer-type bpe --split-documents
    uv run python scripts/run_ablation_study.py
    # or, the fast smoke version:
    uv run python scripts/prepare_data.py --corpus tiny_shakespeare --tokenizer-type char
    uv run python scripts/run_ablation_study.py --smoke
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
from pathlib import Path

import torch

from configs.presets import (
    ACT_GEGLU,
    ACT_GELU,
    ATT_QKNORM,
    MOE_FINEGRAINED,
    MOE_LOSSFREE,
    MTP_D1,
    POS_ALIBI,
    POS_NOPE,
    ROUTE_SIGMOID,
    SMOKE_TEST,
)
from qwenmoe.config import DataConfig, ModelConfig, TrainConfig
from qwenmoe.model.moe import LossFreeBalancer, select_top_k
from qwenmoe.model.transformer import QwenMoEModel
from qwenmoe.train.compare import run_comparison
from qwenmoe.train.scaling import chinchilla_budget, count_params, moe_budget_note

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

BASELINE_NAME = "SMOKE_TEST"

# name -> (axis, human label, full ModelConfig, TrainConfig overrides)
VARIANTS: dict[str, tuple[str, str, ModelConfig, dict]] = {
    "SMOKE_TEST": ("baseline", "SMOKE_TEST (baseline)", SMOKE_TEST, {}),
    "POS_NOPE": ("positional", "NoPE", POS_NOPE, {}),
    "POS_ALIBI": ("positional", "ALiBi", POS_ALIBI, {}),
    "ACT_GEGLU": ("activation", "GEGLU", ACT_GEGLU, {}),
    "ACT_GELU": ("activation", "GELU", ACT_GELU, {}),
    "MOE_LOSSFREE": ("moe_balancing", "Loss-Free", MOE_LOSSFREE, {}),
    "ATTN_MLA": (
        "attention",
        "MLA",
        dataclasses.replace(SMOKE_TEST, attention_variant="mla", mla_latent_dim=16),
        {},
    ),
    "TIE_FALSE": (
        "tie_word_embeddings",
        "Untied embeddings",
        dataclasses.replace(SMOKE_TEST, tie_word_embeddings=False),
        {},
    ),
    # -- moe-modernization axes (task 17, Requirement 12.5) --
    "OPT_MUON": ("optimizer", "Muon", SMOKE_TEST, {"optimizer": "muon"}),
    "SCHED_WSD": ("lr_schedule", "WSD schedule", SMOKE_TEST, {"lr_schedule": "wsd"}),
    "ROUTE_SIGMOID": ("router_scoring", "Sigmoid routing", ROUTE_SIGMOID, {}),
    "ATT_QKNORM": ("qk_norm", "QK-norm", ATT_QKNORM, {}),
    "MTP_D1": ("mtp_depth", "MTP depth 1", MTP_D1, {}),
    "MOE_FINEGRAINED": ("expert_granularity", "Fine-grained experts (16, top-8)", MOE_FINEGRAINED, {}),
}


def _read_log(log_path: Path) -> dict:
    """Parses a `Trainer`-written `train.jsonl` -- same record schema
    `qwenmoe/viz/scenes_training.py::load_training_log` reads, reimplemented
    locally so this script doesn't need to import `manim`. Returns the
    step-by-step curve data PLUS the final metrics and a real measured
    tokens/sec (mean of every logged step's `elapsed_s`, the actual
    per-step wall-clock time `Trainer.fit` recorded during this exact run
    -- not a separate post-hoc calibration pass), so the whole export can
    be rebuilt from this durable on-disk log alone, without needing the
    in-memory `VariantResult` (or a live model) at all.
    """
    steps, train_loss, aux_loss, elapsed_s = [], [], [], []
    eval_steps, val_loss, val_ppl = [], [], []
    expert_steps, expert_counts = [], []
    with log_path.open(encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record["type"] == "step":
                steps.append(record["step"])
                train_loss.append(record["train_loss"])
                aux_loss.append(record.get("aux_loss"))
                elapsed_s.append(record["elapsed_s"])
                counts = record.get("expert_token_counts")
                if counts is not None:
                    expert_steps.append(record["step"])
                    expert_counts.append(counts)
            elif record["type"] == "eval":
                eval_steps.append(record["step"])
                val_loss.append(record["val_loss"])
                val_ppl.append(record["val_perplexity"])

    if not steps or not eval_steps:
        raise ValueError(f"'{log_path}' has no complete step/eval records")

    mean_elapsed = sum(elapsed_s) / len(elapsed_s)
    return {
        "final_train_loss": train_loss[-1],
        "final_val_loss": val_loss[-1],
        "val_perplexity": val_ppl[-1],
        "tokens_per_sec": None,  # filled in by caller, which knows batch_size*seq_len
        "mean_step_elapsed_s": mean_elapsed,
        "curve": {
            "steps": steps,
            "train_loss": train_loss,
            "aux_loss": aux_loss,
            "eval_steps": eval_steps,
            "val_loss": val_loss,
            "val_perplexity": val_ppl,
            "expert_steps": expert_steps,
            "expert_token_counts": expert_counts,
        },
    }


def _bias_mechanism_demo() -> dict:
    """Real `LossFreeBalancer` bias-nudge loop over 40 rounds -- the same
    mechanism `qwenmoe/viz/scenes_task15_preview.py` already visualizes in
    Manim, captured here as real numbers for the web deep-dive rather than
    re-derived or approximated."""
    balancer = LossFreeBalancer(n_routed_experts=4, update_rate=0.05)
    logits = torch.tensor([[5.0, 1.0, 1.0, -5.0]])
    probs = torch.softmax(logits, dim=-1)
    snapshots: dict[int, list[float]] = {}
    winners: list[int] = []
    for round_idx in range(40):
        if round_idx in (0, 10, 39):
            snapshots[round_idx] = balancer.bias.clone().tolist()
        scores = balancer.selection_scores(logits)
        topk_indices, _ = select_top_k(scores, probs, top_k=1)
        winners.append(topk_indices.item())
        counts = torch.bincount(topk_indices.flatten(), minlength=4)
        balancer.update(counts)
    return {
        "initial_logits": logits.tolist()[0],
        "snapshots": {str(k): v for k, v in snapshots.items()},
        "expert0_win_rate_early": winners[:5].count(0) / 5,
        "expert0_win_rate_late": winners[-5:].count(0) / 5,
    }


def _startup_budget_note(config: ModelConfig, vocab_size: int) -> None:
    """Log the Chinchilla reference budget plus the sparse-MoE caveat
    (moe-modernization Requirement 12.6) for the baseline shape."""
    breakdown = count_params(QwenMoEModel(dataclasses.replace(config, vocab_size=vocab_size)))
    logger.info(
        "baseline params: %s total / %s active per token | chinchilla 20x active = %s tokens",
        f"{breakdown.total:,}",
        f"{breakdown.active:,}",
        f"{chinchilla_budget(breakdown.active):,}",
    )
    logger.info("MoE budget note: %s", moe_budget_note(breakdown.total, breakdown.active))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--finalize-only",
        action="store_true",
        help="skip (re-)training; rebuild ablation_study/data/ablation.js from each "
        "variant's already-written runs/ablation_study/<name>/logs/train.jsonl",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="use the fast char-level tiny-Shakespeare corpus instead of the default "
        "TinyStories BPE tier (Requirement 12.3)",
    )
    args = parser.parse_args()

    if args.smoke:
        data_config = DataConfig(
            corpus="tiny_shakespeare",
            tokenizer_type="char",
            cache_dir=Path(".cache/data"),
            fixed_slice_tokens=200_000,
            fixed_slice_seed=0,
        )
    else:
        data_config = DataConfig(
            corpus="tiny_stories",
            tokenizer_type="bpe",
            bpe_vocab_size=4096,
            cache_dir=Path(".cache/data"),
            fixed_slice_tokens=1_000_000,
            fixed_slice_seed=0,
        )
    train_config = TrainConfig(
        batch_size=64,
        seq_len=128,
        total_tokens=64 * 128 * 600,
        lr=3e-4,
        warmup_steps=60,
        eval_interval_steps=30,
        checkpoint_interval_steps=600,
        log_interval_steps=30,
        device="cpu",
        seed=1337,
    )
    output_dir = Path("runs/ablation_study")
    tokens_per_step = train_config.batch_size * train_config.seq_len

    from qwenmoe.data.download import prepared_data_dir
    from qwenmoe.tokenizer.base import build_tokenizer

    tokenizer = build_tokenizer(data_config)
    tokenizer.load(prepared_data_dir(data_config) / "tokenizer.json")
    vocab_size = tokenizer.vocab_size
    _startup_budget_note(SMOKE_TEST, vocab_size)

    if args.finalize_only:
        configs_by_name = {
            name: dataclasses.replace(cfg, vocab_size=vocab_size)
            for name, (_, _, cfg, _) in VARIANTS.items()
        }
        log_paths = {name: output_dir / name / "logs" / "train.jsonl" for name in VARIANTS}
    else:
        variants_for_run = {name: dataclasses.asdict(cfg) for name, (_, _, cfg, _) in VARIANTS.items()}
        train_overrides = {name: ov for name, (_, _, _, ov) in VARIANTS.items() if ov}
        report = run_comparison(
            SMOKE_TEST,
            variants_for_run,
            data_config,
            train_config,
            output_dir,
            train_overrides=train_overrides,
            baseline_name=BASELINE_NAME,
        )
        configs_by_name = {name: report.variants[name].config for name in VARIANTS}
        log_paths = {name: report.variants[name].log_path for name in VARIANTS}

    variants_out = {}
    for name, (axis, label, _cfg, train_ov) in VARIANTS.items():
        parsed = _read_log(log_paths[name])
        parsed["tokens_per_sec"] = tokens_per_step / parsed["mean_step_elapsed_s"]
        variants_out[name] = {
            "axis": axis,
            "label": label,
            "config": dataclasses.asdict(configs_by_name[name]),
            "train_overrides": train_ov,
            **parsed,
        }

    data = {
        "baseline": BASELINE_NAME,
        "data_config": {
            "corpus": data_config.corpus,
            "tokenizer_type": data_config.tokenizer_type,
            "fixed_slice_tokens": data_config.fixed_slice_tokens,
        },
        "train_config": {
            "batch_size": train_config.batch_size,
            "seq_len": train_config.seq_len,
            "total_tokens": train_config.total_tokens,
            "total_steps": train_config.total_tokens // (train_config.batch_size * train_config.seq_len),
            "lr": train_config.lr,
            "device_requested": train_config.device,
        },
        "variants": variants_out,
        "loss_free_demo": _bias_mechanism_demo(),
    }

    out_path = Path("ablation_study/data/ablation.js")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(f"window.ABLATION_DATA = {json.dumps(data)};\n")

    print(f"wrote {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")
    for name in VARIANTS:
        r = variants_out[name]
        print(
            f"{name:<14} val_loss={r['final_val_loss']:.4f}  "
            f"ppl={r['val_perplexity']:.3f}  tok/s={r['tokens_per_sec']:.0f}"
        )


if __name__ == "__main__":
    main()
