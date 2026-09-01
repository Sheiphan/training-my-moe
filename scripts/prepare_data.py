#!/usr/bin/env python
"""Downloads a corpus, trains its tokenizer from scratch on the full text,
encodes the corpus, and caches both to disk (Requirement 4.1-4.3) so
`scripts/train.py` doesn't need to redo this every run.

Usage:
    uv run python scripts/prepare_data.py --corpus tiny_shakespeare --tokenizer-type char
    uv run python scripts/prepare_data.py --corpus tiny_stories --tokenizer-type bpe --bpe-vocab-size 4096
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from qwenmoe.config import DataConfig
from qwenmoe.data.download import download_corpus, prepared_data_dir
from qwenmoe.tokenizer.base import build_tokenizer

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def prepare_data(config: DataConfig) -> Path:
    """Downloads `config.corpus`, trains its tokenizer, encodes the full
    text, and writes `tokenizer.json` + `tokens.pt` under
    `prepared_data_dir(config)`. Returns that directory."""
    raw_path = download_corpus(config.corpus, config.cache_dir)
    text = raw_path.read_text(encoding="utf-8")

    tokenizer = build_tokenizer(config)
    tokenizer.train(text)
    token_ids = tokenizer.encode(text)

    out_dir = prepared_data_dir(config)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(out_dir / "tokenizer.json")
    torch.save(torch.tensor(token_ids, dtype=torch.long), out_dir / "tokens.pt")

    logger.info("corpus: %s (source: %s)", config.corpus, raw_path)
    logger.info("characters: %d", len(text))
    logger.info("tokenizer: %s, vocab_size=%d", config.tokenizer_type, tokenizer.vocab_size)
    logger.info("tokens: %d", len(token_ids))
    logger.info("compression ratio (characters/token): %.3f", len(text) / max(1, len(token_ids)))
    logger.info("wrote: %s", out_dir)
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus", choices=["tiny_shakespeare", "tiny_stories"], default="tiny_shakespeare"
    )
    parser.add_argument("--tokenizer-type", choices=["char", "bpe"], default="char")
    parser.add_argument("--bpe-vocab-size", type=int, default=4096)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/data"))
    args = parser.parse_args()

    config = DataConfig(
        corpus=args.corpus,
        tokenizer_type=args.tokenizer_type,
        bpe_vocab_size=args.bpe_vocab_size,
        cache_dir=args.cache_dir,
    )
    prepare_data(config)


if __name__ == "__main__":
    main()
