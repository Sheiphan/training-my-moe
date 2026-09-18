"""Byte-level BPE tokenizer, merge-pair learning loop implemented from
scratch (no `tokenizers`/`tiktoken` dependency) — Requirement 4.2-4.3.

Byte-level (Radford et al., 2019 / GPT-2) rather than the original
character-level BPE (Sennrich, Haddow & Birch, 2015) means the base
vocabulary is the 256 possible byte values, so *any* UTF-8 string can be
encoded with zero unknown-token cases, unlike `CharTokenizer`.

Two things make this fast enough to train on a ~400M-character corpus
(journal 039) where the previous implementation would have taken tens of
hours:

1. **Pre-tokenization** (GPT-2's regex, approximated with the stdlib `re`
   module so no `regex` dependency is needed): text is first cut into
   word-ish chunks, and merges are learned and applied *within* a chunk
   only. This is what GPT-2 itself does, and it replaces this module's
   earlier deliberate simplification — merges learned over the raw byte
   stream, free to span word/whitespace boundaries — which cost some vocab
   efficiency (a merge could spend an id on `"the the"`) and, far worse,
   forced every count to be taken over the whole corpus.
2. **A frequency table with incremental pair counts** (Sennrich et al.'s
   own approach): identical chunks are collapsed into one entry with a
   count, shrinking the working set from every position in the corpus to
   the unique chunks (millions, not hundreds of millions). Each merge then
   touches only the chunks that actually contain the merged pair, updating
   the affected pair counts in place, instead of re-scanning the entire
   corpus once per merge.

An alternative considered and not taken (journal 039): keep the naive
whole-stream trainer, but train it on a small *sample* of the corpus and
apply the merges it learns to encode the full corpus. Encoding is much
cheaper than training, so that also runs in reasonable time, and it is what
production tokenizers effectively do (a tokenizer is fitted on a sample,
not the whole pretraining set). It was rejected here only because the
merges would then be fitted to a narrower slice of text than the model
trains on; the approach above makes training on the full corpus cheap
enough that the tradeoff isn't needed.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

_BASE_VOCAB_SIZE = 256  # every possible byte value 0-255

# GPT-2's pre-tokenization pattern, translated from the `regex` module's
# `\p{L}`/`\p{N}` Unicode property classes to stdlib-`re` equivalents:
# `[^\W\d_]` is "word character, but not a digit and not underscore", i.e.
# a letter. Contractions are split off first, then letter runs, digit runs,
# and symbol runs each optionally keep one leading space (so " the" is a
# single chunk and the tokenizer learns word-with-space units), and finally
# whitespace runs are kept whole except for the last space before a word.
#
# The symbol class is `[^\s\w]|_` rather than a plain `[^\s\w]`: GPT-2's
# `[^\s\p{L}\p{N}]` counts `_` as a symbol, but Python's `\w` *includes*
# `_`, so the direct translation would leave underscore matched by no
# alternative at all -- and `finditer` silently skips what it cannot match,
# which drops the character from the text entirely. `_covers_all_text` in
# the tests pins the invariant that every character lands in some chunk.
_PRE_TOKEN_PATTERN = re.compile(
    r"'s|'t|'re|'ve|'m|'ll|'d| ?[^\W\d_]+| ?\d+| ?(?:[^\s\w]|_)+|\s+(?!\S)|\s+"
)


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
        # chunk -> its encoded ids. Natural text repeats the same words
        # constantly, so this turns most of `encode` into a dict hit.
        # Invalidated whenever the merge table changes.
        self._encode_cache: dict[str, list[int]] = {}

    @property
    def vocab_size(self) -> int:
        return len(self._vocab) + (1 if self._eos_token else 0)

    @property
    def eos_id(self) -> int | None:
        """The reserved document-separator id (`None` unless constructed
        with `eos_token=True`): the first id past the byte+merge vocab."""
        return len(self._vocab) if self._eos_token else None

    def train(self, text: str) -> None:
        self._merges = {}
        self._vocab = {i: bytes([i]) for i in range(_BASE_VOCAB_SIZE)}
        self._encode_cache = {}

        chunk_freqs = Counter(
            match.group().encode("utf-8") for match in _PRE_TOKEN_PATTERN.finditer(text)
        )
        chunks = [list(chunk) for chunk in chunk_freqs]
        freqs = list(chunk_freqs.values())

        # Pair -> total frequency-weighted count, and pair -> which chunks
        # contain it, so a merge only has to revisit those chunks.
        pair_counts: dict[tuple[int, int], int] = {}
        pair_chunks: dict[tuple[int, int], set[int]] = {}
        for index, chunk in enumerate(chunks):
            frequency = freqs[index]
            for pair in zip(chunk, chunk[1:]):
                pair_counts[pair] = pair_counts.get(pair, 0) + frequency
                pair_chunks.setdefault(pair, set()).add(index)

        num_merges = self._target_vocab_size - _BASE_VOCAB_SIZE
        for i in range(num_merges):
            if not pair_counts:
                break
            best_pair = max(pair_counts, key=pair_counts.get)
            if pair_counts[best_pair] < 2:
                break  # no repeated pair left worth merging
            new_id = _BASE_VOCAB_SIZE + i

            for index in list(pair_chunks[best_pair]):
                chunk = chunks[index]
                frequency = freqs[index]
                for pair in zip(chunk, chunk[1:]):
                    remaining = pair_counts.get(pair, 0) - frequency
                    if remaining > 0:
                        pair_counts[pair] = remaining
                    else:
                        pair_counts.pop(pair, None)
                    holders = pair_chunks.get(pair)
                    if holders is not None:
                        holders.discard(index)

                merged = _merge_pair(chunk, best_pair, new_id)
                chunks[index] = merged
                for pair in zip(merged, merged[1:]):
                    pair_counts[pair] = pair_counts.get(pair, 0) + frequency
                    pair_chunks.setdefault(pair, set()).add(index)

            pair_counts.pop(best_pair, None)
            pair_chunks.pop(best_pair, None)
            self._merges[best_pair] = new_id
            self._vocab[new_id] = self._vocab[best_pair[0]] + self._vocab[best_pair[1]]

    def _encode_chunk(self, raw: bytes) -> list[int]:
        """Applies the learned merges to one pre-token, always taking the
        earliest-learned applicable merge first (its id doubles as its
        rank), which is what makes encoding reproduce training's merge
        order."""
        ids = list(raw)
        while len(ids) >= 2:
            best_pair = None
            best_rank = None
            for pair in zip(ids, ids[1:]):
                rank = self._merges.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_pair, best_rank = pair, rank
            if best_pair is None:
                break
            ids = _merge_pair(ids, best_pair, best_rank)
        return ids

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for match in _PRE_TOKEN_PATTERN.finditer(text):
            chunk = match.group()
            encoded = self._encode_cache.get(chunk)
            if encoded is None:
                encoded = self._encode_chunk(chunk.encode("utf-8"))
                self._encode_cache[chunk] = encoded
            ids.extend(encoded)
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
        self._encode_cache = {}
        for i, (a, b) in enumerate(data["merges"]):
            new_id = _BASE_VOCAB_SIZE + i
            self._merges[(a, b)] = new_id
            self._vocab[new_id] = self._vocab[a] + self._vocab[b]
