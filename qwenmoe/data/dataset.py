"""Fixed-length `(x, y)` windows for next-token prediction, a train/val
split helper (Requirement 4.4), and `make_fixed_slice` for reproducible
cross-variant comparisons (Requirements 11.5, 12.6).
"""
from __future__ import annotations

import torch
from torch import Tensor
from torch.utils.data import Dataset

# "documents" for make_fixed_slice's shuffle are approximated as fixed-size
# contiguous chunks of the flat token stream (see task-8 journal entry) —
# real document boundaries aren't available from an already-tokenized,
# already-concatenated `all_tokens` stream alone.
_DEFAULT_CHUNK_SIZE = 1024


def _as_long_tensor(tokens: list[int] | Tensor) -> Tensor:
    if isinstance(tokens, Tensor):
        return tokens.to(dtype=torch.long)
    return torch.tensor(tokens, dtype=torch.long)


class TokenDataset(Dataset):
    """`(x, y)` windows of length `seq_len` for next-token prediction: `y`
    is the same span of the token stream shifted right by one position
    (Requirement 4.4)."""

    def __init__(self, tokens: list[int] | Tensor, seq_len: int) -> None:
        if seq_len < 1:
            raise ValueError(f"seq_len must be >= 1, got {seq_len}")
        tokens = _as_long_tensor(tokens)
        if tokens.numel() < seq_len + 1:
            raise ValueError(
                f"need at least seq_len + 1 = {seq_len + 1} tokens, got {tokens.numel()}"
            )
        self._tokens = tokens
        self._seq_len = seq_len

    def __len__(self) -> int:
        return self._tokens.numel() - self._seq_len

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        x = self._tokens[idx : idx + self._seq_len]
        y = self._tokens[idx + 1 : idx + self._seq_len + 1]
        return x, y


def train_val_split(
    tokens: list[int] | Tensor, val_fraction: float = 0.1
) -> tuple[Tensor, Tensor]:
    """Splits a flat token stream into contiguous `(train, val)` segments,
    the last `val_fraction` of tokens reserved for validation (Requirement
    4.4)."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")
    tokens = _as_long_tensor(tokens)
    n_val = int(tokens.numel() * val_fraction)
    n_train = tokens.numel() - n_val
    return tokens[:n_train], tokens[n_train:]


def make_fixed_slice(all_tokens: list[int] | Tensor, n_tokens: int, seed: int) -> Tensor:
    """Reproducibly selects `n_tokens` from `all_tokens`: splits into
    fixed-size chunks, shuffles the chunk order once using `seed`,
    concatenates, then truncates to exactly `n_tokens` (Requirements 11.5,
    12.6). Calling this with the same `(all_tokens, n_tokens, seed)` always
    returns the identical slice, so every ablation variant trains on
    exactly the same data; a different `seed` reshuffles the chunk order.
    Token order *within* each chunk is preserved — only chunk order is
    randomized — so local linguistic structure survives the shuffle.
    """
    if n_tokens < 1:
        raise ValueError(f"n_tokens must be >= 1, got {n_tokens}")
    all_tokens = _as_long_tensor(all_tokens)
    if n_tokens > all_tokens.numel():
        raise ValueError(
            f"n_tokens ({n_tokens}) exceeds available tokens ({all_tokens.numel()})"
        )

    chunk_size = max(1, min(_DEFAULT_CHUNK_SIZE, all_tokens.numel() // 4))
    chunks = list(torch.split(all_tokens, chunk_size))
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(chunks), generator=generator)
    shuffled = torch.cat([chunks[i] for i in order.tolist()])
    return shuffled[:n_tokens]
