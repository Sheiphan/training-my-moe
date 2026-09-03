"""Fixed-length `(x, y)` windows for next-token prediction, a train/val
split helper (Requirement 4.4; moe-modernization Requirement 10.4), and
`make_fixed_slice` for reproducible cross-variant comparisons
(Requirements 11.5, 12.6).
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


def _document_ids(tokens: Tensor, eos_id: int) -> Tensor:
    """Per-position document index for a flat, `eos_id`-separated stream: a
    new document begins at the position *after* each separator, so the
    `<eos>` token is the last token of the document it terminates (the
    Llama-3 convention). `cumsum` of "the previous token was a separator".
    """
    starts = torch.zeros_like(tokens)
    starts[1:] = (tokens[:-1] == eos_id).to(tokens.dtype)
    return torch.cumsum(starts, dim=0)


class TokenDataset(Dataset):
    """`(x, y)` windows of length `seq_len` for next-token prediction: `y`
    is the same span of the token stream shifted right by one position
    (Requirement 4.4).

    `return_segments` (moe-modernization Requirement 10.2, default `False`
    -> unchanged 2-tuple output): when `True`, `__getitem__` also returns a
    `(seq_len,)` integer tensor of per-position document ids (rebased so
    the window starts at document `0`), which the trainer turns into an
    attention mask that blocks cross-document attention. Requires `eos_id`
    -- the separator token `scripts/prepare_data.py` inserts between packed
    documents (Requirement 10.1).
    """

    def __init__(
        self,
        tokens: list[int] | Tensor,
        seq_len: int,
        *,
        return_segments: bool = False,
        eos_id: int | None = None,
    ) -> None:
        if seq_len < 1:
            raise ValueError(f"seq_len must be >= 1, got {seq_len}")
        if return_segments and eos_id is None:
            raise ValueError(
                "TokenDataset(return_segments=True) requires eos_id -- the document "
                "separator id recorded in tokenizer.json by scripts/prepare_data.py "
                "(run it with --split-documents)"
            )
        tokens = _as_long_tensor(tokens)
        if tokens.numel() < seq_len + 1:
            raise ValueError(
                f"need at least seq_len + 1 = {seq_len + 1} tokens, got {tokens.numel()}"
            )
        self._tokens = tokens
        self._seq_len = seq_len
        self._return_segments = return_segments
        self._eos_id = eos_id
        self._doc_ids = _document_ids(tokens, eos_id) if return_segments else None

    def __len__(self) -> int:
        return self._tokens.numel() - self._seq_len

    def __getitem__(self, idx: int):
        x = self._tokens[idx : idx + self._seq_len]
        y = self._tokens[idx + 1 : idx + self._seq_len + 1]
        if not self._return_segments:
            return x, y
        window = self._doc_ids[idx : idx + self._seq_len]
        segments = window - window[0]  # rebase so this window starts at document 0
        return x, y, segments


def _split_into_documents(tokens: Tensor, eos_id: int) -> list[Tensor]:
    """Cuts a flat `eos_id`-separated stream into a list of document
    tensors, each *including* its terminating separator (a trailing
    document with no separator is kept as-is)."""
    sep_positions = (tokens == eos_id).nonzero(as_tuple=True)[0].tolist()
    documents: list[Tensor] = []
    start = 0
    for pos in sep_positions:
        documents.append(tokens[start : pos + 1])
        start = pos + 1
    if start < tokens.numel():
        documents.append(tokens[start:])
    return documents


def train_val_split(
    tokens: list[int] | Tensor,
    val_fraction: float = 0.1,
    *,
    mode: str = "tail",
    eos_id: int | None = None,
) -> tuple[Tensor, Tensor]:
    """Splits a flat token stream into `(train, val)` (Requirement 4.4;
    moe-modernization Requirement 10.4).

    - `mode="tail"` (default, today's behavior): the last `val_fraction` of
      *tokens* is the validation segment -- a contiguous cut that may fall
      in the middle of a document.
    - `mode="documents"`: whole documents (split on `eos_id`) are held out,
      taking documents from the end of the stream until at least
      `val_fraction` of the tokens are in validation. No document is ever
      split between train and val, so validation perplexity is never
      measured on a truncated-context document. Requires `eos_id`.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")
    tokens = _as_long_tensor(tokens)

    if mode == "tail":
        n_val = int(tokens.numel() * val_fraction)
        n_train = tokens.numel() - n_val
        return tokens[:n_train], tokens[n_train:]

    if mode == "documents":
        if eos_id is None:
            raise ValueError("train_val_split(mode='documents') requires eos_id")
        documents = _split_into_documents(tokens, eos_id)
        if len(documents) < 2:
            raise ValueError(
                f"need >= 2 documents to split on boundaries, found {len(documents)} "
                f"(is eos_id={eos_id} the correct separator?)"
            )
        target_val_tokens = tokens.numel() * val_fraction
        val_documents: list[Tensor] = []
        accumulated = 0
        for document in reversed(documents):
            if accumulated >= target_val_tokens:
                break
            val_documents.append(document)
            accumulated += document.numel()
        val_documents.reverse()
        n_train_documents = len(documents) - len(val_documents)
        if n_train_documents < 1:
            raise ValueError(
                f"val_fraction={val_fraction} leaves no whole documents for training"
            )
        train_documents = documents[:n_train_documents]
        return torch.cat(train_documents), torch.cat(val_documents)

    raise ValueError(f"Unknown val_split_mode: {mode!r}")


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
