#!/usr/bin/env python
"""Trains multiple named `ModelConfig` presets on an identical, fixed token
slice under a shared token budget, and prints/saves a final validation
loss/perplexity comparison table (Requirement 12.6-12.7).

Usage (compares the baseline positional scheme against two alternatives,
using tiny-Shakespeare's already-prepared char data):
    uv run python scripts/prepare_data.py --corpus tiny_shakespeare --tokenizer-type char
    uv run python scripts/run_comparison.py --presets SMOKE_TEST POS_NOPE POS_ALIBI \\
        --corpus tiny_shakespeare --tokenizer-type char \\
        --fixed-slice-tokens 200000 --total-tokens 200000 --output-dir runs/comparison
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
from pathlib import Path

import configs.presets as presets
from qwenmoe.config import DataConfig, ModelConfig, TrainConfig
from qwenmoe.model.transformer import QwenMoEModel
from qwenmoe.train.compare import ComparisonReport, run_comparison
from qwenmoe.train.scaling import chinchilla_budget, count_params, moe_budget_note

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def _log_baseline_budget(base_config: ModelConfig) -> None:
    """moe-modernization Requirement 12.6: print the Chinchilla reference
    budget for the baseline preset plus the sparse-MoE caveat."""
    breakdown = count_params(QwenMoEModel(base_config))
    logger.info(
        "baseline params: %s total / %s active | chinchilla 20x active = %s tokens",
        f"{breakdown.total:,}",
        f"{breakdown.active:,}",
        f"{chinchilla_budget(breakdown.active):,}",
    )
    logger.info("MoE budget note: %s", moe_budget_note(breakdown.total, breakdown.active))


def _report_to_json(report: ComparisonReport) -> dict:
    return {
        "baseline": report.baseline_name,
        "seeds": list(report.seeds),
        "variants": {
            name: {
                "final_train_loss": result.final_train_loss,
                "final_val_loss": result.final_val_loss,
                "val_perplexity": result.val_perplexity,
                "val_loss_mean": result.val_loss_mean,
                "val_loss_std": result.val_loss_std,
                "val_perplexity_mean": result.val_perplexity_mean,
                "val_perplexity_std": result.val_perplexity_std,
                "per_seed_val_loss": result.per_seed_val_loss,
                "within_noise_of_baseline": result.within_noise_of_baseline,
                "comparison_mode": result.comparison_mode,
                "tokens_per_sec": result.tokens_per_sec,
                "log_paths": [str(p) for p in result.log_paths],
            }
            for name, result in report.variants.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--presets",
        nargs="+",
        required=True,
        help="ModelConfig preset names from configs/presets.py to compare, e.g. "
        "SMOKE_TEST POS_NOPE POS_ALIBI",
    )
    parser.add_argument(
        "--corpus", choices=["tiny_shakespeare", "tiny_stories"], default="tiny_shakespeare"
    )
    parser.add_argument("--tokenizer-type", choices=["char", "bpe"], default="char")
    parser.add_argument("--bpe-vocab-size", type=int, default=4096)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/data"))
    parser.add_argument(
        "--fixed-slice-tokens",
        type=int,
        required=True,
        help="identical token slice every variant trains on (Requirement 12.6)",
    )
    parser.add_argument("--fixed-slice-seed", type=int, default=0)
    parser.add_argument(
        "--total-tokens", type=int, required=True, help="shared training token budget (Requirement 11.5)"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--eval-interval-steps", type=int, default=100)
    parser.add_argument("--max-eval-batches", type=int, default=50)
    parser.add_argument("--checkpoint-interval-steps", type=int, default=500)
    parser.add_argument("--log-interval-steps", type=int, default=10)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="train every variant once per seed and report mean ± std across seeds "
        "(moe-modernization Requirement 12.1); defaults to just --seed",
    )
    parser.add_argument(
        "--param-matched",
        action="store_true",
        help="bisect each variant's d_ff so its total parameter count matches the "
        "baseline (first preset) within 2% (moe-modernization Requirement 12.4)",
    )
    args = parser.parse_args()

    for name in args.presets:
        if not hasattr(presets, name):
            raise ValueError(f"no preset named {name!r} in configs/presets.py")

    base_config = getattr(presets, args.presets[0])
    variants = {name: dataclasses.asdict(getattr(presets, name)) for name in args.presets}

    data_config = DataConfig(
        corpus=args.corpus,
        tokenizer_type=args.tokenizer_type,
        bpe_vocab_size=args.bpe_vocab_size,
        cache_dir=args.cache_dir,
        fixed_slice_tokens=args.fixed_slice_tokens,
        fixed_slice_seed=args.fixed_slice_seed,
    )
    train_config = TrainConfig(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        total_tokens=args.total_tokens,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        eval_interval_steps=args.eval_interval_steps,
        max_eval_batches=args.max_eval_batches,
        checkpoint_interval_steps=args.checkpoint_interval_steps,
        log_interval_steps=args.log_interval_steps,
        device=args.device,
        seed=args.seed,
    )

    seeds = tuple(args.seeds) if args.seeds else (args.seed,)
    logger.info("comparing presets: %s", ", ".join(args.presets))
    _log_baseline_budget(base_config)
    logger.info(
        "fixed slice: %d tokens (seed=%d), budget: %d tokens per variant, seeds: %s%s",
        args.fixed_slice_tokens,
        args.fixed_slice_seed,
        args.total_tokens,
        list(seeds),
        " [param-matched]" if args.param_matched else "",
    )

    report = run_comparison(
        base_config,
        variants,
        data_config,
        train_config,
        args.output_dir,
        seeds=seeds,
        param_matched=args.param_matched,
        baseline_name=args.presets[0],
    )

    table = report.summary_table()
    print(table)

    summary_path = Path(args.output_dir) / "comparison_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(_report_to_json(report), indent=2) + "\n", encoding="utf-8")
    logger.info("summary written to %s", summary_path)


if __name__ == "__main__":
    main()
