"""Autoregressive text generation from a trained checkpoint (Requirement 6).

No KV cache: every new token re-runs a full forward pass over the entire
sequence generated so far, reusing `QwenMoEModel.forward` exactly as every
other module in this project does for any `seq_len` -- simpler than adding
an incremental cache, at the cost of some speed. Fine at the laptop/tiny-
model scale this project targets.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from qwenmoe.model.kv_cache import build_kv_cache
from qwenmoe.model.transformer import QwenMoEModel
from qwenmoe.tokenizer.base import Tokenizer


def filter_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Keeps only the `k` highest logits per row, setting the rest to
    `-inf` so softmax assigns them exactly zero probability (top-k
    sampling, Fan, Lewis & Dauphin 2018). Standalone (not private) so it's
    directly testable with hand-built logits tensors, mirroring
    `moe.py::select_top_k`.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    k = min(k, logits.size(-1))
    threshold = torch.topk(logits, k, dim=-1).values[..., -1, None]
    return torch.where(logits < threshold, torch.full_like(logits, float("-inf")), logits)


def filter_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Nucleus sampling (Holtzman et al. 2019): keeps the smallest,
    highest-probability prefix (in descending-probability order) whose
    cumulative probability is `>= top_p`, setting every other logit to
    `-inf`. Always keeps at least the single highest-probability token, so
    the result never has every entry masked out.
    """
    if not (0.0 < top_p <= 1.0):
        raise ValueError(f"top_p must be in (0, 1], got {top_p}")
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    mass_before = torch.cumsum(sorted_probs, dim=-1) - sorted_probs  # prob strictly before this token
    sorted_logits = torch.where(
        mass_before >= top_p, torch.full_like(sorted_logits, float("-inf")), sorted_logits
    )
    filtered = torch.full_like(logits, float("-inf"))
    filtered.scatter_(-1, sorted_indices, sorted_logits)
    return filtered


def _sample_next(
    next_logits: torch.Tensor,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
) -> torch.Tensor:
    """`(batch, vocab)` last-position logits -> `(batch, 1)` next token id.
    `temperature <= 0` is deterministic greedy (argmax); otherwise scale,
    optionally `top_k`/`top_p` filter, softmax, multinomial-sample. Shared
    verbatim by the cached and uncached decode loops so they can't drift."""
    if temperature <= 0:
        return torch.argmax(next_logits, dim=-1, keepdim=True)
    scaled_logits = next_logits / temperature
    if top_k is not None:
        scaled_logits = filter_top_k(scaled_logits, top_k)
    if top_p is not None:
        scaled_logits = filter_top_p(scaled_logits, top_p)
    probs = F.softmax(scaled_logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def generate(
    model: QwenMoEModel,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    device: str = "cpu",
    use_kv_cache: bool = False,
) -> str:
    """Encodes `prompt`, autoregressively samples up to `max_new_tokens`
    more tokens, and decodes the full resulting sequence back to text
    (Requirement 6.1, 6.2). Stops early if the sequence would otherwise
    exceed `model.config.max_seq_len` (Requirement 6.3). `temperature <=
    0` performs deterministic greedy (argmax) decoding instead of
    sampling -- an explicit mode this project adds on top of Requirement
    6.2's "configurable temperature" so there's a decoding path with zero
    randomness to test determinism against.

    `use_kv_cache` (moe-modernization Requirement 8): with the cache on,
    the prompt is encoded once (a "prefill") and every subsequent step
    feeds *only* the newest token, attending against each layer's stored
    keys/values -- `O(n)` total attention work to emit `n` tokens instead
    of `O(n^2)`. The cached and uncached paths compute the same attention
    (each token still attends over the identical prefix at its true
    positions), so greedy decoding produces an identical token sequence
    (Requirement 8.2).
    """
    if max_new_tokens < 0:
        raise ValueError(f"max_new_tokens must be >= 0, got {max_new_tokens}")
    if temperature < 0:
        raise ValueError(f"temperature must be >= 0, got {temperature}")
    if top_k is not None and top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}")
    if top_p is not None and not (0.0 < top_p <= 1.0):
        raise ValueError(f"top_p must be in (0, 1], got {top_p}")

    token_ids = tokenizer.encode(prompt)
    if len(token_ids) == 0:
        raise ValueError("prompt must encode to at least one token")

    model = model.to(device)
    model.eval()

    sequence = torch.tensor([token_ids], dtype=torch.long, device=device)
    max_seq_len = model.config.max_seq_len
    caches = build_kv_cache(model) if use_kv_cache else None

    for _ in range(max_new_tokens):
        if sequence.size(1) >= max_seq_len:
            break  # Requirement 6.3 / 8.4: stop at the model's max context length

        if caches is None:
            logits, _ = model(sequence)
        elif caches[0].length == 0:
            logits, _ = model(sequence, kv_caches=caches)  # prefill: encode the whole prompt once
        else:
            logits, _ = model(
                sequence[:, -1:], kv_caches=caches, position_offset=sequence.size(1) - 1
            )

        next_token = _sample_next(logits[:, -1, :], temperature, top_k, top_p)
        sequence = torch.cat([sequence, next_token], dim=1)

    return tokenizer.decode(sequence[0].tolist())
