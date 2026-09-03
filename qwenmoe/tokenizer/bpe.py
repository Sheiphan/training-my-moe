"""Byte-level BPE tokenizer, merge-pair learning loop implemented from
scratch (no `tokenizers`/`tiktoken` dependency) — Requirement 4.2-4.3.

Byte-level (Radford et al., 2019 / GPT-2) rather than the original
character-level BPE (Sennrich, Haddow & Birch, 2015) means the base
vocabulary is the 256 possible byte values, so *any* UTF-8 string can be
encoded with zero unknown-token cases, unlike `CharTokenizer`. Used for the
TinyStories comparison study.

Simplification vs. GPT-2's own tokenizer: merges are learned over the raw
byte stream with no regex pre-tokenization step, so (unlike GPT-2) a merge
can span a word/whitespace boundary. This keeps the from-scratch merge loop
simple (mirrors Karpathy's minbpe `BasicTokenizer`) at the cost of some
vocab efficiency; see the journal entry for the full tradeoff discussion.
"""
from __future__ import annotations

import json
from pathlib import Path

_BASE_VOCAB_SIZE = 256  # every possible byte value 0-255


def _count_pairs(ids: list[int]) -> dict[tuple[int, int], int]:
    counts: dict[tuple[int, int], int] = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def _merge_pair(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    merged: list[int] = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
            merged.append(new_id)
            i += 2
        else:
            merged.append(ids[i])
            i += 1
    return merged


class ByteBPETokenizer:
    """`target_vocab_size` (the "configured target" in tasks.md) is fixed at
    construction; `train` learns merges up to that size, stopping early if
    the corpus runs out of repeated (frequency >= 2) pairs first. Before
    `train` is called, encode/decode still work via the base byte vocab
    alone (zero merges learned yet).
    """

    def __init__(self, target_vocab_size: int = 4096, *, eos_token: bool = False):
        if target_vocab_size < _BASE_VOCAB_SIZE:
            raise ValueError(
                f"target_vocab_size must be >= {_BASE_VOCAB_SIZE} (the byte-level base "
                f"vocabulary), got {target_vocab_size}"
            )
        self._target_vocab_size = target_vocab_size
        # moe-modernization Requirement 10.1: when `eos_token` is on, one id
        # past the learned byte+merge vocab is reserved as a dedicated
        # document separator. `encode` never emits it (it isn't a byte
        # value); `scripts/prepare_data.py` splices it between packed
        # documents explicitly, and `decode` drops it. Off by default, so a
        # tokenizer written before this spec is unchanged (same
        # `vocab_size`, same ids).
        self._eos_token = eos_token
        self._merges: dict[tuple[int, int], int] = {}
        self._vocab: dict[int, bytes] = {i: bytes([i]) for i in range(_BASE_VOCAB_SIZE)}

    @property
    def vocab_size(self) -> int:
        return len(self._vocab) + (1 if self._eos_token else 0)

    @property
    def eos_id(self) -> int | None:
        """The reserved document-separator id (`None` unless constructed
        with `eos_token=True`): the first id past the byte+merge vocab."""
        return len(self._vocab) if self._eos_token else None

    def train(self, text: str) -> None:
        ids = list(text.encode("utf-8"))
        self._merges = {}
        self._vocab = {i: bytes([i]) for i in range(_BASE_VOCAB_SIZE)}

        num_merges = self._target_vocab_size - _BASE_VOCAB_SIZE
        for i in range(num_merges):
            pair_counts = _count_pairs(ids)
            if not pair_counts:
                break
            best_pair = max(pair_counts, key=pair_counts.get)
            if pair_counts[best_pair] < 2:
                break  # no repeated pair left worth merging
            new_id = _BASE_VOCAB_SIZE + i
            ids = _merge_pair(ids, best_pair, new_id)
            self._merges[best_pair] = new_id
            self._vocab[new_id] = self._vocab[best_pair[0]] + self._vocab[best_pair[1]]

    def encode(self, text: str) -> list[int]:
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            pair_counts = _count_pairs(ids)
            mergeable = [p for p in pair_counts if p in self._merges]
            if not mergeable:
                break
            best_pair = min(mergeable, key=lambda p: self._merges[p])
            ids = _merge_pair(ids, best_pair, self._merges[best_pair])
        return ids

    def decode(self, ids: list[int]) -> str:
        eos_id = self.eos_id
        try:
            byte_seq = b"".join(self._vocab[i] for i in ids if i != eos_id)
        except KeyError as e:
            raise ValueError(
                f"token id {e.args[0]} out of range for vocab_size={self.vocab_size}"
            ) from e
        return byte_seq.decode("utf-8", errors="replace")

    def save(self, path: str | Path) -> None:
        ordered_merges = sorted(self._merges.items(), key=lambda item: item[1])
        payload = {
            "target_vocab_size": self._target_vocab_size,
            "merges": [[a, b] for (a, b), _new_id in ordered_merges],
            "eos_token": self._eos_token,
        }
        Path(path).write_text(json.dumps(payload))

    def load(self, path: str | Path) -> None:
        data = json.loads(Path(path).read_text())
        self._target_vocab_size = data["target_vocab_size"]
        self._eos_token = data.get("eos_token", False)  # pre-spec files have no eos
        self._merges = {}
        self._vocab = {i: bytes([i]) for i in range(_BASE_VOCAB_SIZE)}
        for i, (a, b) in enumerate(data["merges"]):
            new_id = _BASE_VOCAB_SIZE + i
            self._merges[(a, b)] = new_id
            self._vocab[new_id] = self._vocab[a] + self._vocab[b]
