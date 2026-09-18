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

`fineweb_edu` trades TinyStories' narrow, synthetic, children's-story
domain for real, diverse, quality-filtered web text — see the journal-039
entry for the dataset comparison and the scaling-law sizing behind the
~400M-character target. FineWeb-Edu (Penedo et al., 2024,
https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu, odc-by license)
ships only as parquet, so this path needs `pyarrow` (already a project
dependency) to read the `text` column; there's no plain-text release to
avoid it with, unlike TinyStories.
"""
from __future__ import annotations

import unicodedata
import urllib.request
from pathlib import Path
from typing import Literal

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from qwenmoe.config import DataConfig

_TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)
_TINY_STORIES_REPO_ID = "roneneldan/TinyStories"
_TINY_STORIES_FILENAME = "TinyStoriesV2-GPT4-valid.txt"
_DOWNLOAD_TIMEOUT_SECONDS = 30

_FINEWEB_EDU_REPO_ID = "HuggingFaceFW/fineweb-edu"
# The smallest official FineWeb-Edu release is the 10B-token "sample-10BT"
# split across 14 shards of ~2.15GB each (~700M tokens/shard); one shard is
# already far more raw material than this project needs, so only the first
# is ever downloaded.
_FINEWEB_EDU_SHARD_FILENAME = "sample/10BT/000_00000.parquet"
# ~18x tiny_stories' ~22.5M characters, chosen so that even a full
# Chinchilla-optimal training budget for this repo's biggest preset
# (SCALE_300M, ~122M active params -> chinchilla_budget ~= 2.44B tokens,
# see qwenmoe/train/scaling.py) stays under ~6 epochs over this corpus, and
# a realistic laptop-scale run (tens to a few hundred million tokens,
# journal-037's measured wall-clock budget) stays under 1 epoch --
# comfortably below epochs_for_budget's 4-epoch repetition warning either
# way. Also sized to keep char-tokenizer encoding's peak memory (a Python
# str + an int list + a torch.long tensor of this many elements) well clear
# of the ~24GB unified-memory swap cliff journal-037 measured on this
# machine.
_FINEWEB_EDU_TARGET_CHARS = 400_000_000

# A raw sample of FineWeb-Edu documents (empirically checked, journal-039)
# pulls in a long tail of foreign-script/symbol/emoji characters from
# incidental non-English content: even filtering out documents that are
# *mostly* non-ASCII still left ~3700 distinct characters over a 400M-
# character sample, most appearing so rarely their embedding rows would
# barely ever get a gradient -- and CharTokenizer.encode raises on any
# character outside the trained vocab, so a fatter tail is also a more
# fragile tokenizer (ByteBPETokenizer has no such failure mode: it falls
# back to raw bytes). `_to_ascii` folds common typographic punctuation and
# accented letters to their plain-ASCII equivalents (so "café"/"naïve"
# survive as "cafe"/"naive", and smart quotes/em-dashes survive as their
# straight/hyphen forms) and drops anything left over (CJK, Cyrillic,
# emoji, symbols) character by character, matching how tiny_stories/
# tiny_shakespeare's char vocabs stay in the ~60-90 range.
_TYPOGRAPHIC_ASCII_MAP = str.maketrans(
    {
        "‘": "'", "’": "'",  # ‘ ’
        "“": '"', "”": '"',  # “ ”
        "–": "-", "—": "-",  # – —
        "…": "...",  # …
        "\xa0": " ",  # non-breaking space
    }
)

# A document that's mostly a different script turns into near-nothing once
# ASCII-folded; skip it rather than spend the character budget on the
# handful of stray Latin characters (page furniture, numbers) left behind.
_FINEWEB_EDU_MIN_RETAINED_FRACTION = 0.8


def _to_ascii(text: str) -> str:
    text = text.translate(_TYPOGRAPHIC_ASCII_MAP)
    text = unicodedata.normalize("NFKD", text)
    return text.encode("ascii", "ignore").decode("ascii")


def _write_fineweb_edu_text(shard_path: Path, dest: Path, target_chars: int) -> int:
    """Reads `shard_path`'s `text` column row-group by row-group,
    ASCII-folding each document (`_to_ascii`) and keeping it only if that
    didn't hollow it out, writing straight out to `dest` and stopping as
    soon as `target_chars` characters have been written. Returns the
    character count.

    Streamed rather than accumulated (journal 040): at this module's
    original 400M-character cap a list-of-documents plus its `join` was
    affordable, but a whole ~3.5B-character shard would need several times
    24GB to hold the same way. Nothing larger than one row group is ever
    resident.
    """
    parquet_file = pq.ParquetFile(shard_path)
    total_chars = 0
    with dest.open("w", encoding="utf-8") as out:
        for batch in parquet_file.iter_batches(columns=["text"]):
            for text in batch.column("text").to_pylist():
                ascii_text = _to_ascii(text)
                if len(ascii_text) < _FINEWEB_EDU_MIN_RETAINED_FRACTION * len(text):
                    continue
                if total_chars:
                    out.write(_DOCUMENT_SEPARATOR)
                out.write(ascii_text)
                total_chars += len(ascii_text)
                if total_chars >= target_chars:
                    return total_chars
    return total_chars


# Matches scripts/prepare_data.py's own document separator so
# --split-documents works on this corpus exactly like tiny_stories.
_DOCUMENT_SEPARATOR = "<|endoftext|>"


def download_corpus(
    name: Literal["tiny_shakespeare", "tiny_stories", "fineweb_edu"],
    cache_dir: str | Path,
    target_chars: int | None = None,
) -> Path:
    """Downloads (if not already cached) `name`'s raw text corpus under
    `cache_dir`, and returns the local path to the cached text file.

    `target_chars` caps how much of a size-parameterized corpus
    (`fineweb_edu`) is kept, defaulting to `_FINEWEB_EDU_TARGET_CHARS`. It
    is part of the cache filename, so asking for a different size prepares
    a different file instead of silently reusing the previous one.
    """
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

    if name == "fineweb_edu":
        chars = target_chars if target_chars is not None else _FINEWEB_EDU_TARGET_CHARS
        dest = cache_dir / f"fineweb_edu_{chars // 1_000_000}Mchars.txt"
        if not dest.exists():
            shard_path = hf_hub_download(
                repo_id=_FINEWEB_EDU_REPO_ID,
                filename=_FINEWEB_EDU_SHARD_FILENAME,
                repo_type="dataset",
                cache_dir=cache_dir,
            )
            # Written to a temporary name first so an interrupted run can't
            # leave a short file behind that later looks like a valid cache.
            partial = dest.with_suffix(".partial")
            _write_fineweb_edu_text(Path(shard_path), partial, chars)
            partial.replace(dest)
        return dest

    raise ValueError(f"Unknown corpus: {name!r}")


def prepared_data_dir(config: DataConfig) -> Path:
    """Where `scripts/prepare_data.py` writes (and `scripts/train.py` later
    reads) a corpus's trained tokenizer + encoded tokens -- named so that
    different `bpe_vocab_size` targets, or different `corpus_chars` sizes
    of the same corpus, don't collide.
    """
    suffix = f"bpe{config.bpe_vocab_size}" if config.tokenizer_type == "bpe" else "char"
    size = f"_{config.corpus_chars // 1_000_000}Mchars" if config.corpus_chars else ""
    return Path(config.cache_dir) / f"{config.corpus}{size}_{suffix}"
