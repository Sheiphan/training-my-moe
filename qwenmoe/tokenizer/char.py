"""Character-level tokenizer: vocab = every unique character seen during
training, sorted for a deterministic id assignment (Requirement 4.2-4.3).
Used for the tiny-Shakespeare smoke test. Unlike `ByteBPETokenizer`, vocab
size here is emergent from the corpus, not a configured target.
"""
from __future__ import annotations

import json
from pathlib import Path


class CharTokenizer:
    def __init__(self):
        self._stoi: dict[str, int] = {}
        self._itos: dict[int, str] = {}

    @property
    def vocab_size(self) -> int:
        return len(self._stoi)

    def train(self, text: str) -> None:
        chars = sorted(set(text))
        self._stoi = {ch: i for i, ch in enumerate(chars)}
        self._itos = dict(enumerate(chars))

    def encode(self, text: str) -> list[int]:
        try:
            return [self._stoi[ch] for ch in text]
        except KeyError as e:
            raise ValueError(f"character {e.args[0]!r} is not in the trained vocabulary") from e

    def decode(self, ids: list[int]) -> str:
        try:
            return "".join(self._itos[i] for i in ids)
        except KeyError as e:
            raise ValueError(
                f"token id {e.args[0]} out of range for vocab_size={self.vocab_size}"
            ) from e

    def save(self, path: str | Path) -> None:
        chars = [self._itos[i] for i in range(self.vocab_size)]
        Path(path).write_text(json.dumps({"chars": chars}))

    def load(self, path: str | Path) -> None:
        data = json.loads(Path(path).read_text())
        chars = data["chars"]
        self._stoi = {ch: i for i, ch in enumerate(chars)}
        self._itos = dict(enumerate(chars))
