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
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch

from qwenmoe.config import DataConfig
from qwenmoe.data.download import download_corpus, prepared_data_dir
from qwenmoe.tokenizer.base import build_tokenizer

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# The literal string TinyStories (and most GPT-2-lineage corpora) uses to
# separate documents in its raw text release.
_DOCUMENT_SEPARATOR = "<|endoftext|>"

# How much text is resident at once while streaming (journal 040). Big
# enough that per-block overhead is irrelevant, small enough that a corpus
# of any size costs the same memory to walk.
_BLOCK_CHARS = 16 * 1024 * 1024


def split_documents(text: str, separator: str = _DOCUMENT_SEPARATOR) -> list[str]:
    """Splits a raw corpus on its document separator, dropping empty
    fragments (leading/trailing/repeated separators). A corpus with no
    separator comes back as a single document."""
    return [chunk.strip() for chunk in text.split(separator) if chunk.strip()]


def _encode_documents(documents, tokenizer, out) -> tuple[int, int]:
    """Encodes each document into `out` (anything with `extend`/`append` --
    a list, or a `_TokenBuffer`), following each with the tokenizer's
    reserved `eos_id` (moe-modernization Requirement 10.1), so every
    document is terminated by exactly one separator token. Returns
    `(document count, character count)`; the character count covers the
    documents themselves, not the separators between them."""
    eos_id = tokenizer.eos_id
    if eos_id is None:
        raise ValueError("document_aware encoding needs a tokenizer built with eos_token=True")
    n_documents = 0
    n_chars = 0
    for document in documents:
        out.extend(tokenizer.encode(document))
        out.append(eos_id)
        n_documents += 1
        n_chars += len(document)
    return n_documents, n_chars


def encode_corpus(text: str, tokenizer, *, document_aware: bool) -> list[int]:
    """Encodes an in-memory `text` to a flat id list. With `document_aware`,
    the raw text is split on `_DOCUMENT_SEPARATOR` and each document is
    encoded independently, joined by the reserved `eos_id`; without it, one
    `encode` call over the whole string.

    `prepare_data` streams instead of calling this (journal 040), but the
    semantics it defines are the same ones the streaming path produces.
    """
    if not document_aware:
        return tokenizer.encode(text)
    token_ids: list[int] = []
    _encode_documents(split_documents(text), tokenizer, token_ids)
    return token_ids


def iter_text_blocks(path: Path, block_chars: int = _BLOCK_CHARS) -> Iterator[str]:
    """Yields `path`'s text in blocks cut at newline boundaries, so no
    pre-token (and therefore no BPE merge) is ever split across a block.
    That is what makes block-by-block encoding produce exactly the ids a
    single whole-corpus `encode` call would.
    """
    with path.open(encoding="utf-8") as handle:
        remainder = ""
        while True:
            block = handle.read(block_chars)
            if not block:
                if remainder:
                    yield remainder
                return
            block = remainder + block
            cut = block.rfind("\n")
            if cut == -1:  # no newline in sight; keep reading before cutting
                remainder = block
                continue
            yield block[: cut + 1]
            remainder = block[cut + 1 :]


def iter_documents(
    path: Path, block_chars: int = _BLOCK_CHARS, separator: str = _DOCUMENT_SEPARATOR
) -> Iterator[str]:
    """Streams `path` and yields whole documents, so a corpus far larger
    than memory can still be split on its separator. Only one document plus
    one block is ever resident."""
    pending = ""
    for block in iter_text_blocks(path, block_chars):
        pending += block
        *complete, pending = pending.split(separator)
        for document in complete:
            document = document.strip()
            if document:
                yield document
    tail = pending.strip()
    if tail:
        yield tail


class _TokenBuffer:
    """Append-only `int32` token array (journal 040).

    Replaces building a Python `list[int]` and handing it to
    `torch.tensor`: at ~1B tokens that list costs ~8GB of pointers alone
    and the tensor another ~8GB alongside it, which does not fit in this
    machine's 24GB. `int32` holds any vocab this project will train, halves
    the tensor against `torch.long`, and `torch.from_numpy` wraps the array
    without copying it.
    """

    def __init__(self, capacity: int):
        self._buffer = np.empty(max(capacity, 1024), dtype=np.int32)
        self._size = 0

    def _ensure(self, extra: int) -> None:
        needed = self._size + extra
        if needed > self._buffer.size:
            grown = np.empty(max(needed, self._buffer.size * 2), dtype=np.int32)
            grown[: self._size] = self._buffer[: self._size]
            self._buffer = grown

    def extend(self, token_ids: list[int]) -> None:
        self._ensure(len(token_ids))
        self._buffer[self._size : self._size + len(token_ids)] = token_ids
        self._size += len(token_ids)

    def append(self, token_id: int) -> None:
        self._ensure(1)
        self._buffer[self._size] = token_id
        self._size += 1

    def as_tensor(self) -> torch.Tensor:
        # Shrink in place first: a *view* of a larger array would make
        # torch.save serialize the whole unused capacity too.
        self._buffer.resize(self._size, refcheck=False)
        return torch.from_numpy(self._buffer)


def _training_blocks(raw_path: Path, block_chars: int, document_aware: bool) -> Iterator[str]:
    """The text the tokenizer learns merges from. Under `document_aware`
    the separator is replaced by a newline so no merge is learnt across a
    document boundary (matching how `_encode_documents` then applies them);
    a separator is never split across a block, since blocks are cut at
    newlines and the separator contains none."""
    for block in iter_text_blocks(raw_path, block_chars):
        yield block.replace(_DOCUMENT_SEPARATOR, "\n") if document_aware else block


def prepare_data(
    config: DataConfig, *, document_aware: bool = False, block_chars: int = _BLOCK_CHARS
) -> Path:
    """Downloads `config.corpus`, trains its tokenizer, encodes the full
    text, and writes `tokenizer.json` + `tokens.pt` under
    `prepared_data_dir(config)`. Returns that directory.

    `document_aware` (moe-modernization Requirement 10.1): reserve a
    separator/`<eos>` id and insert it between documents, so
    `TrainConfig.document_masking` and `DataConfig.val_split_mode
    == "documents"` have real boundaries to work with. `bpe` only.

    The corpus is streamed, never held whole (journal 040): the tokenizer
    trains from a block iterator, and encoded ids go straight into an
    `int32` `_TokenBuffer`. Peak memory is therefore set by the *output*
    token array, not by the corpus, which is what makes a ~1B-token
    `fineweb_edu` preparable on a 24GB machine.
    """
    if document_aware and config.tokenizer_type != "bpe":
        raise ValueError("--split-documents requires --tokenizer-type bpe")

    raw_path = download_corpus(config.corpus, config.cache_dir, target_chars=config.corpus_chars)

    tokenizer = build_tokenizer(config, eos_token=document_aware)
    tokenizer.train(_training_blocks(raw_path, block_chars, document_aware))

    # One id per character for `char`; ~3.5 characters per id for `bpe`, so
    # a third is a deliberate over-estimate -- `_TokenBuffer` still grows if
    # it is wrong, this only avoids paying for the copy.
    raw_bytes = raw_path.stat().st_size
    buffer = _TokenBuffer(raw_bytes if config.tokenizer_type == "char" else raw_bytes // 3)

    n_chars = 0
    n_documents = 0
    if document_aware:
        n_documents, n_chars = _encode_documents(
            iter_documents(raw_path, block_chars), tokenizer, buffer
        )
    else:
        for block in iter_text_blocks(raw_path, block_chars):
            n_chars += len(block)
            buffer.extend(tokenizer.encode(block))

    tokens = buffer.as_tensor()

    out_dir = prepared_data_dir(config)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(out_dir / "tokenizer.json")
    torch.save(tokens, out_dir / "tokens.pt")

    logger.info("corpus: %s (source: %s)", config.corpus, raw_path)
    logger.info("characters: %d", n_chars)
    logger.info("tokenizer: %s, vocab_size=%d", config.tokenizer_type, tokenizer.vocab_size)
    if document_aware:
        logger.info("documents: %d, separator id (eos_id): %d", n_documents, tokenizer.eos_id)
    logger.info("tokens: %d (%s)", tokens.numel(), tokens.dtype)
    logger.info("compression ratio (characters/token): %.3f", n_chars / max(1, tokens.numel()))
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
        "--corpus-chars",
        type=int,
        default=None,
        help="raw-character cap for fineweb_edu (default: the corpus's own "
        "400M). The whole downloaded shard is ~3.5B characters / ~1B tokens; "
        "it is part of the prepared-data directory name, so sizes never "
        "overwrite each other",
    )
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
        corpus_chars=args.corpus_chars,
    )
    prepare_data(config, document_aware=args.split_documents)


if __name__ == "__main__":
    main()
