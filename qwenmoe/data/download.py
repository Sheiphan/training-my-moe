"""Corpus download and local caching (Requirement 4.1) — the only
network-touching module in the project.

`tiny_shakespeare` is fetched as a direct HTTPS text file (the standard
~1.1MB tiny-Shakespeare release from Karpathy's char-rnn, used by nanoGPT
and countless other from-scratch LM tutorials). `tiny_stories` uses
`huggingface_hub.hf_hub_download` to fetch one plain-text release file
(not the parquet default, to avoid a `pyarrow`/`pandas` dependency) from
the TinyStories dataset (Eldan & Li, 2023,
https://huggingface.co/datasets/roneneldan/TinyStories) — specifically the
V2 (GPT-4-only generations, higher quality than the original's GPT-3.5
mix) validation split, ~22.5MB, small enough to download and to run this
project's from-scratch `ByteBPETokenizer` merge-learning loop over in
reasonable time on a laptop. See the task-8 journal entry for the full
file-choice rationale.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Literal

from huggingface_hub import hf_hub_download

from qwenmoe.config import DataConfig

_TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)
_TINY_STORIES_REPO_ID = "roneneldan/TinyStories"
_TINY_STORIES_FILENAME = "TinyStoriesV2-GPT4-valid.txt"
_DOWNLOAD_TIMEOUT_SECONDS = 30


def download_corpus(name: Literal["tiny_shakespeare", "tiny_stories"], cache_dir: str | Path) -> Path:
    """Downloads (if not already cached) `name`'s raw text corpus under
    `cache_dir`, and returns the local path to the cached text file."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if name == "tiny_shakespeare":
        dest = cache_dir / "tiny_shakespeare.txt"
        if not dest.exists():
            with urllib.request.urlopen(
                _TINY_SHAKESPEARE_URL, timeout=_DOWNLOAD_TIMEOUT_SECONDS
            ) as response:
                dest.write_bytes(response.read())
        return dest

    if name == "tiny_stories":
        path = hf_hub_download(
            repo_id=_TINY_STORIES_REPO_ID,
            filename=_TINY_STORIES_FILENAME,
            repo_type="dataset",
            cache_dir=cache_dir,
        )
        return Path(path)

    raise ValueError(f"Unknown corpus: {name!r}")


def prepared_data_dir(config: DataConfig) -> Path:
    """Where `scripts/prepare_data.py` writes (and `scripts/train.py` later
    reads) a corpus's trained tokenizer + encoded tokens -- named so that
    different `bpe_vocab_size` targets for the same corpus don't collide.
    """
    suffix = f"bpe{config.bpe_vocab_size}" if config.tokenizer_type == "bpe" else "char"
    return Path(config.cache_dir) / f"{config.corpus}_{suffix}"
