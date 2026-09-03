"""Shared `Tokenizer` protocol both tokenizer implementations satisfy, plus
the `build_tokenizer` factory selecting between them by
`DataConfig.tokenizer_type` (Requirement 4.2-4.3) — mirrors the
`build_expert`/`build_attention`/`build_balancer`/`build_positional_scheme`
factories in `qwenmoe/model/`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from qwenmoe.config import DataConfig
from qwenmoe.tokenizer.bpe import ByteBPETokenizer
from qwenmoe.tokenizer.char import CharTokenizer


@runtime_checkable
class Tokenizer(Protocol):
    """`train`/`encode`/`decode`/`save`/`load`/`vocab_size`, implemented by
    both `CharTokenizer` and `ByteBPETokenizer` (design.md `tokenizer/`)."""

    @property
    def vocab_size(self) -> int: ...

    def train(self, text: str) -> None: ...
    def encode(self, text: str) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...
    def save(self, path: str | Path) -> None: ...
    def load(self, path: str | Path) -> None: ...


def build_tokenizer(config: DataConfig, *, eos_token: bool = False) -> Tokenizer:
    """Constructs the tokenizer named by `config.tokenizer_type`.

    `eos_token` (moe-modernization Requirement 10.1, only meaningful for
    `"bpe"`): reserve a dedicated document-separator id -- set by
    `scripts/prepare_data.py --split-documents` when packing a
    multi-document corpus. `CharTokenizer` (tiny-Shakespeare, a single
    document) ignores it.
    """
    if config.tokenizer_type == "char":
        return CharTokenizer()
    if config.tokenizer_type == "bpe":
        return ByteBPETokenizer(config.bpe_vocab_size, eos_token=eos_token)
    raise ValueError(f"Unknown tokenizer_type: {config.tokenizer_type!r}")


def build_tokenizer_from_meta(tokenizer_meta: dict | None) -> Tokenizer:
    """Reconstructs the exact tokenizer a checkpoint was trained with, from
    the `tokenizer_meta` dict every checkpoint carries (`tokenizer_type` +
    `tokenizer_path`) -- deliberately separate from `build_tokenizer`
    above, which needs a full `DataConfig` a checkpoint doesn't have.
    Shared by `scripts/generate.py` and `qwenmoe/viz/common.py::capture_trace`.
    """
    if not tokenizer_meta or "tokenizer_path" not in tokenizer_meta:
        raise ValueError(
            "checkpoint has no usable tokenizer_meta (missing tokenizer_path) -- it wasn't "
            "produced by 'uv run python scripts/train.py'"
        )
    tokenizer_type = tokenizer_meta["tokenizer_type"]
    if tokenizer_type == "char":
        tokenizer = CharTokenizer()
    elif tokenizer_type == "bpe":
        tokenizer = ByteBPETokenizer()
    else:
        raise ValueError(f"unknown tokenizer_type in checkpoint: {tokenizer_type!r}")
    tokenizer.load(tokenizer_meta["tokenizer_path"])
    return tokenizer
