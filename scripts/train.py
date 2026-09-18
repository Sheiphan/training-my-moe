#!/usr/bin/env python
"""Trains a QwenMoE model on a corpus already prepared by
`scripts/prepare_data.py`, writing checkpoints and JSON-lines logs under
`--output-dir` (Requirement 5.5, 10.1).

Usage (tiny-Shakespeare smoke test):
    uv run python scripts/prepare_data.py --corpus tiny_shakespeare --tokenizer-type char
    uv run python scripts/train.py --preset SMOKE_TEST --corpus tiny_shakespeare \\
        --tokenizer-type char --total-tokens 200000 --output-dir runs/smoke_test

Usage (gradient accumulation -- match a bigger model's memory-constrained
--batch-size to a smaller model's effective batch, journal 038):
    uv run python scripts/train.py --preset SCALE_300M --batch-size 4 \\
        --grad-accum-steps 4 ... # effective batch 16, same peak memory as batch=4 alone

Usage (fineweb_edu -- real, diverse web text instead of tiny_stories'
synthetic children's stories, journal 039):
    uv run python scripts/prepare_data.py --corpus fineweb_edu --tokenizer-type bpe \\
        --bpe-vocab-size 4096 --split-documents
    uv run python scripts/train.py --preset SCALE_300M --corpus fineweb_edu \\
        --tokenizer-type bpe --bpe-vocab-size 4096 --total-tokens 200000000 \\
        --output-dir runs/fineweb_300m
"""
from __future__ import annotations

import argparse
import dataclasses
import logging
from pathlib import Path

import torch

import configs.presets as presets
from qwenmoe.config import DataConfig, TrainConfig
from qwenmoe.data.dataset import TokenDataset, train_val_split
from qwenmoe.data.download import prepared_data_dir
from qwenmoe.model.transformer import QwenMoEModel
from qwenmoe.tokenizer.base import build_tokenizer
from qwenmoe.train.scaling import chinchilla_budget, count_params, moe_budget_note
from qwenmoe.train.trainer import Trainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def _load_prepared_data(config: DataConfig) -> tuple[torch.Tensor, dict]:
    """Loads the `tokens.pt` + `tokenizer.json` pair `prepare_data.py`
    wrote for `config`, failing with an actionable message (not a raw
    traceback) if that step hasn't been run yet."""
    data_dir = prepared_data_dir(config)
    tokens_path = data_dir / "tokens.pt"
    tokenizer_path = data_dir / "tokenizer.json"
    if not tokens_path.exists() or not tokenizer_path.exists():
        raise FileNotFoundError(
            f"no prepared data found under {data_dir} -- run "
            f"'uv run python scripts/prepare_data.py --corpus {config.corpus} "
            f"--tokenizer-type {config.tokenizer_type}' first"
        )

    tokens = torch.load(tokens_path, weights_only=True)
    tokenizer = build_tokenizer(config)
    tokenizer.load(tokenizer_path)
    tokenizer_meta = {
        "tokenizer_type": config.tokenizer_type,
        "tokenizer_path": str(tokenizer_path),
        "vocab_size": tokenizer.vocab_size,
        # moe-modernization Requirement 10.1: the document separator id, or
        # None unless prepared with `--split-documents`.
        "eos_id": getattr(tokenizer, "eos_id", None),
    }
    return tokens, tokenizer_meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset", default="BASELINE", help="ModelConfig preset name from configs/presets.py"
    )
    parser.add_argument(
        "--corpus",
        choices=["tiny_shakespeare", "tiny_stories", "fineweb_edu"],
        default="tiny_shakespeare",
    )
    parser.add_argument("--tokenizer-type", choices=["char", "bpe"], default="char")
    parser.add_argument("--bpe-vocab-size", type=int, default=4096)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/data"))
    parser.add_argument(
        "--corpus-chars",
        type=int,
        default=None,
        help="must match the --corpus-chars prepare_data.py was run with, since "
        "it selects which prepared-data directory to load",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--total-tokens", type=int, required=True, help="training token budget")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=1,
        help="micro-batches of --batch-size accumulated per optimizer step -- an "
        "effective batch of batch-size * grad-accum-steps at the peak memory cost "
        "of just batch-size (journal 038; lets a memory-constrained --batch-size "
        "still match a bigger run's effective batch for a fair comparison)",
    )
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--eval-interval-steps", type=int, default=100)
    parser.add_argument("--max-eval-batches", type=int, default=50)
    parser.add_argument("--checkpoint-interval-steps", type=int, default=500)
    parser.add_argument("--log-interval-steps", type=int, default=10)
    parser.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument(
        "--val-split-mode",
        choices=["tail", "documents"],
        default="tail",
        help="'documents' holds out whole <eos>-separated documents "
        "(moe-modernization Requirement 10.4; needs data prepared with --split-documents)",
    )
    parser.add_argument(
        "--document-masking",
        action="store_true",
        help="block attention across packed-document boundaries "
        "(moe-modernization Requirement 10.2; needs data prepared with --split-documents)",
    )
    args = parser.parse_args()

    data_config = DataConfig(
        corpus=args.corpus,
        tokenizer_type=args.tokenizer_type,
        bpe_vocab_size=args.bpe_vocab_size,
        cache_dir=args.cache_dir,
        corpus_chars=args.corpus_chars,
        val_split_mode=args.val_split_mode,
    )
    tokens, tokenizer_meta = _load_prepared_data(data_config)
    eos_id = tokenizer_meta["eos_id"]
    if (args.document_masking or args.val_split_mode == "documents") and eos_id is None:
        raise ValueError(
            "--document-masking / --val-split-mode=documents need a corpus prepared with "
            "'uv run python scripts/prepare_data.py ... --split-documents' (no eos_id found)"
        )

    train_tokens, val_tokens = train_val_split(
        tokens, mode=args.val_split_mode, eos_id=eos_id
    )
    segment_kwargs = dict(return_segments=args.document_masking, eos_id=eos_id)
    train_dataset = TokenDataset(train_tokens, args.seq_len, **segment_kwargs)
    val_dataset = TokenDataset(val_tokens, args.seq_len, **segment_kwargs)

    if not hasattr(presets, args.preset):
        raise ValueError(f"no preset named {args.preset!r} in configs/presets.py")
    model_config = dataclasses.replace(
        getattr(presets, args.preset),
        vocab_size=tokenizer_meta["vocab_size"],
        max_seq_len=max(getattr(presets, args.preset).max_seq_len, args.seq_len),
    )

    train_config = TrainConfig(
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
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
        document_masking=args.document_masking,
    )

    torch.manual_seed(args.seed)
    model = QwenMoEModel(model_config)
    logger.info(
        "model: preset=%s vocab_size=%d d_model=%d n_layers=%d",
        args.preset,
        model_config.vocab_size,
        model_config.d_model,
        model_config.n_layers,
    )
    breakdown = count_params(model)
    logger.info(
        "params: %s total / %s active per token | chinchilla 20x active = %s tokens (budget: %s)",
        f"{breakdown.total:,}",
        f"{breakdown.active:,}",
        f"{chinchilla_budget(breakdown.active):,}",
        f"{args.total_tokens:,}",
    )
    logger.info("MoE budget note: %s", moe_budget_note(breakdown.total, breakdown.active))
    logger.info(
        "data: corpus=%s tokenizer=%s train_tokens=%d val_tokens=%d",
        args.corpus,
        args.tokenizer_type,
        len(train_tokens),
        len(val_tokens),
    )
    if args.grad_accum_steps > 1:
        logger.info(
            "gradient accumulation: batch_size=%d x grad_accum_steps=%d -> effective batch %d",
            args.batch_size,
            args.grad_accum_steps,
            args.batch_size * args.grad_accum_steps,
        )

    trainer = Trainer(args.output_dir, tokenizer_meta=tokenizer_meta)
    result = trainer.fit(model, train_dataset, val_dataset, train_config, resume_from=args.resume_from)

    logger.info(
        "done: step=%d train_loss=%.4f val_loss=%.4f val_perplexity=%.4f",
        result.final_step,
        result.final_train_loss,
        result.final_val_loss,
        result.val_perplexity,
    )
    logger.info("checkpoint: %s", result.checkpoint_path)
    logger.info("log: %s", result.log_path)


if __name__ == "__main__":
    main()
