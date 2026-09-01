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


def filter_min_p(logits: torch.Tensor, min_p: float) -> torch.Tensor:
    """min-p sampling (Nguyen et al., 2024): keep only tokens whose
    probability is at least `min_p` times the single most likely token's
    probability; `-inf` the rest.

    Adaptive where top-p is fixed: when the model is confident (one token
    holds most of the mass) the threshold `min_p · p_max` is high and only
    a few tokens survive; when it's unsure (a flat distribution) `p_max` is
    small so the threshold is low and many tokens survive. Since `min_p <=
    1`, the argmax always clears its own threshold, so the result never
    masks every token (Requirement 11.1, 11.4). Standalone and mirroring
    `filter_top_k`/`filter_top_p` so it's testable on hand-built logits.
    """
    if not (0.0 < min_p <= 1.0):
        raise ValueError(f"min_p must be in (0, 1], got {min_p}")
    probs = F.softmax(logits, dim=-1)
    threshold = min_p * probs.max(dim=-1, keepdim=True).values
    return torch.where(probs < threshold, torch.full_like(logits, float("-inf")), logits)


def apply_repetition_penalty(
    logits: torch.Tensor, generated_ids: torch.Tensor, penalty: float
) -> torch.Tensor:
    """CTRL-style repetition penalty (Keskar et al., 2019): push the logit
    of every token id that already appears in `generated_ids` toward
    `-inf` by a factor of `penalty` (`> 1` discourages repeats, `1.0` is a
    no-op).

    Sign-aware, per the CTRL / Hugging Face implementation: a *positive*
    seen-token logit is **divided** by `penalty` (shrinks toward 0), a
    *negative* one is **multiplied** (grows more negative). Both lower that
    token's probability -- a naive "always divide" would instead *raise*
    the probability of a seen token whose logit is negative, which is the
    opposite of the intent behind Requirement 11.2.
    """
    if penalty <= 0:
        raise ValueError(f"repetition_penalty must be > 0, got {penalty}")
    if penalty == 1.0:
        return logits
    seen = torch.unique(generated_ids)
    logits = logits.clone()
    current = logits[..., seen]
    logits[..., seen] = torch.where(current > 0, current / penalty, current * penalty)
    return logits


def _sample_next(
    next_logits: torch.Tensor,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    min_p: float | None = None,
    repetition_penalty: float = 1.0,
    generated_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """`(batch, vocab)` last-position logits -> `(batch, 1)` next token id.

    The repetition penalty (if any) is applied to the raw logits first, so
    it shapes greedy decoding too. Then `temperature <= 0` is deterministic
    greedy (argmax); otherwise scale by temperature and apply
    `top_k` -> `top_p` -> `min_p` in sequence -- each further restricts the
    surviving candidate set (Requirement 11.4) and every one keeps at least
    the argmax, so the set is never empty -- then softmax + multinomial.
    Shared verbatim by the cached and uncached decode loops so they can't
    drift.
    """
    if repetition_penalty != 1.0 and generated_ids is not None:
        next_logits = apply_repetition_penalty(next_logits, generated_ids, repetition_penalty)
    if temperature <= 0:
        return torch.argmax(next_logits, dim=-1, keepdim=True)
    scaled_logits = next_logits / temperature
    if top_k is not None:
        scaled_logits = filter_top_k(scaled_logits, top_k)
    if top_p is not None:
        scaled_logits = filter_top_p(scaled_logits, top_p)
    if min_p is not None:
        scaled_logits = filter_min_p(scaled_logits, min_p)
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
    min_p: float | None = None,
    repetition_penalty: float = 1.0,
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

    `min_p` and `repetition_penalty` (moe-modernization Requirement 11):
    `min_p` keeps only tokens with probability `>= min_p * p_max` (an
    adaptive alternative to `top_p`), composing after `top_k`/`top_p`;
    `repetition_penalty > 1.0` pushes down the logits of tokens already in
    the sequence. Both neutral by default (`None` / `1.0`) -> generation is
    identical to before this task (Requirement 11.3).
    """
    if max_new_tokens < 0:
        raise ValueError(f"max_new_tokens must be >= 0, got {max_new_tokens}")
    if temperature < 0:
        raise ValueError(f"temperature must be >= 0, got {temperature}")
    if top_k is not None and top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}")
    if top_p is not None and not (0.0 < top_p <= 1.0):
        raise ValueError(f"top_p must be in (0, 1], got {top_p}")
    if min_p is not None and not (0.0 < min_p <= 1.0):
        raise ValueError(f"min_p must be in (0, 1], got {min_p}")
    if repetition_penalty <= 0:
        raise ValueError(f"repetition_penalty must be > 0, got {repetition_penalty}")

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

        next_token = _sample_next(
            logits[:, -1, :], temperature, top_k, top_p,
            min_p=min_p, repetition_penalty=repetition_penalty, generated_ids=sequence,
        )
        sequence = torch.cat([sequence, next_token], dim=1)

    return tokenizer.decode(sequence[0].tolist())
