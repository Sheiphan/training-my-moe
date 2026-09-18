#!/usr/bin/env python
"""Downloads a corpus, trains its tokenizer from scratch on the full text,
encodes the corpus, and caches both to disk (Requirement 4.1-4.3) so
`scripts/train.py` doesn't need to redo this every run.

Usage:
    uv run python scripts/prepare_data.py --corpus tiny_shakespeare --tokenizer-type char
    uv run python scripts/prepare_data.py --corpus tiny_stories --tokenizer-type bpe \\
        --bpe-vocab-size 4096 --split-documents
    uv run python scripts/prepare_data.py --corpus fineweb_edu --tokenizer-type bpe \\
        --bpe-vocab-size 4096 --split-documents
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

# The literal string TinyStories (and most GPT-2-lineage corpora) uses to
# separate documents in its raw text release.
_DOCUMENT_SEPARATOR = "<|endoftext|>"


def split_documents(text: str, separator: str = _DOCUMENT_SEPARATOR) -> list[str]:
    """Splits a raw corpus on its document separator, dropping empty
    fragments (leading/trailing/repeated separators). A corpus with no
    separator comes back as a single document."""
    return [chunk.strip() for chunk in text.split(separator) if chunk.strip()]


def encode_corpus(text: str, tokenizer, *, document_aware: bool) -> list[int]:
    """Encodes `text` to a flat id list. With `document_aware`, the raw text
    is split on `_DOCUMENT_SEPARATOR` and each document is encoded
    independently, then the pieces are joined by the tokenizer's reserved
    `eos_id` (moe-modernization Requirement 10.1) -- so a merge is never
    learnt or applied across a boundary and every document is followed by
    exactly one separator token. Without it, today's behavior: one
    `encode` call over the whole string.
    """
    if not document_aware:
        return tokenizer.encode(text)
    eos_id = tokenizer.eos_id
    if eos_id is None:
        raise ValueError("document_aware encoding needs a tokenizer built with eos_token=True")
    token_ids: list[int] = []
    for document in split_documents(text):
        token_ids.extend(tokenizer.encode(document))
        token_ids.append(eos_id)
    return token_ids


def prepare_data(config: DataConfig, *, document_aware: bool = False) -> Path:
    """Downloads `config.corpus`, trains its tokenizer, encodes the full
    text, and writes `tokenizer.json` + `tokens.pt` under
    `prepared_data_dir(config)`. Returns that directory.

    `document_aware` (moe-modernization Requirement 10.1): reserve a
    separator/`<eos>` id and insert it between documents, so
    `TrainConfig.document_masking` and `DataConfig.val_split_mode
    == "documents"` have real boundaries to work with. `bpe` only.
    """
    if document_aware and config.tokenizer_type != "bpe":
        raise ValueError("--split-documents requires --tokenizer-type bpe")

    raw_path = download_corpus(config.corpus, config.cache_dir)
    text = raw_path.read_text(encoding="utf-8")

    tokenizer = build_tokenizer(config, eos_token=document_aware)
    # Merges are learnt over the separator-free text so no merge spans a
    # document boundary (matches how encode_corpus then applies them).
    tokenizer.train(text.replace(_DOCUMENT_SEPARATOR, "\n") if document_aware else text)
    token_ids = encode_corpus(text, tokenizer, document_aware=document_aware)

    out_dir = prepared_data_dir(config)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(out_dir / "tokenizer.json")
    torch.save(torch.tensor(token_ids, dtype=torch.long), out_dir / "tokens.pt")

    logger.info("corpus: %s (source: %s)", config.corpus, raw_path)
    logger.info("characters: %d", len(text))
    logger.info("tokenizer: %s, vocab_size=%d", config.tokenizer_type, tokenizer.vocab_size)
    if document_aware:
        logger.info(
            "documents: %d, separator id (eos_id): %d",
            token_ids.count(tokenizer.eos_id),
            tokenizer.eos_id,
        )
    logger.info("tokens: %d", len(token_ids))
    logger.info("compression ratio (characters/token): %.3f", len(text) / max(1, len(token_ids)))
    logger.info("wrote: %s", out_dir)
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        choices=["tiny_shakespeare", "tiny_stories", "fineweb_edu"],
        default="tiny_shakespeare",
    )
    parser.add_argument("--tokenizer-type", choices=["char", "bpe"], default="char")
    parser.add_argument("--bpe-vocab-size", type=int, default=4096)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/data"))
    parser.add_argument(
        "--split-documents",
        action="store_true",
        help="reserve a <eos> id and insert it between documents (bpe only, "
        "moe-modernization Requirement 10.1) -- needed for TrainConfig.document_masking "
        "and DataConfig.val_split_mode='documents'",
    )
    args = parser.parse_args()

    config = DataConfig(
        corpus=args.corpus,
        tokenizer_type=args.tokenizer_type,
        bpe_vocab_size=args.bpe_vocab_size,
        cache_dir=args.cache_dir,
    )
    prepare_data(config, document_aware=args.split_documents)


if __name__ == "__main__":
    main()
