"""Scaling-law utilities (Requirement 11): parameter counting, a
Chinchilla-style reference token budget, empirical throughput calibration,
and the epoch/repetition warning — the numbers a training run's token
budget should be planned from, rather than picked arbitrarily.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from qwenmoe.model.transformer import QwenMoEModel

logger = logging.getLogger(__name__)

_EPOCH_WARNING_THRESHOLD = 4


@dataclass
class ParamBreakdown:
    """Per-component parameter counts (Requirement 11.1).

    `attention_total` has no separate "active" figure because attention is
    fully dense — every parameter is used for every token, unlike the
    routed half of `moe_total`/`moe_active`. `embedding` bundles the token
    embedding table, the final RMSNorm, and (only when
    `tie_word_embeddings=False`) the separate output-head weight — the
    once-per-model pieces that sit outside every repeated `TransformerBlock`.
    """

    embedding: int
    attention_total: int
    moe_total: int
    moe_active: int
    total: int
    active: int


def _numel(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def count_params(model: QwenMoEModel) -> ParamBreakdown:
    """Counts total and per-token-active parameters, broken down by
    component (Requirement 11.1). "Active" mirrors exactly which weights a
    real forward pass multiplies against for one token: every dense piece
    in full (embedding, attention, MoE norm/router/shared-experts), plus
    only `top_k` of each layer's `n_routed_experts` routed experts —
    assumed equally sized, which `build_expert` guarantees.
    """
    config = model.config

    attention_total = 0
    moe_total = 0
    moe_active = 0
    for block in model.blocks:
        attention_total += _numel(block.attn_norm) + _numel(block.attn)

        moe = block.moe
        moe_norm_params = _numel(block.moe_norm)
        router_params = _numel(moe.router)
        shared_params = _numel(moe.shared_experts)
        routed_total = _numel(moe.routed_experts)
        routed_per_expert = routed_total // len(moe.routed_experts)

        moe_total += moe_norm_params + router_params + shared_params + routed_total
        moe_active += (
            moe_norm_params + router_params + shared_params + config.top_k * routed_per_expert
        )

    embedding = _numel(model.embedding) + _numel(model.final_norm)
    if not config.tie_word_embeddings:
        embedding += _numel(model.lm_head)

    total = embedding + attention_total + moe_total
    active = embedding + attention_total + moe_active
    return ParamBreakdown(
        embedding=embedding,
        attention_total=attention_total,
        moe_total=moe_total,
        moe_active=moe_active,
        total=total,
        active=active,
    )


def chinchilla_budget(active_params: int) -> int:
    """Chinchilla-style reference token budget `D_ref ~= 20 * active_params`
    (Hoffmann et al., 2022), Requirement 11.2 — a planning heuristic, not a
    hard target. See journal entry 001 for the full compute-optimal
    derivation of why this ratio is a constant multiplier independent of
    model size.
    """
    if active_params < 1:
        raise ValueError(f"active_params must be >= 1, got {active_params}")
    return 20 * active_params


def _training_step(model: QwenMoEModel, x: torch.Tensor, y: torch.Tensor) -> None:
    model.zero_grad(set_to_none=True)
    logits, extras = model(x)
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
    aux_loss = extras.get("aux_loss")
    if aux_loss is not None:
        loss = loss + aux_loss
    loss.backward()


def _synchronize(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()


def calibrate_throughput(
    model: QwenMoEModel,
    sample_batch: tuple[torch.Tensor, torch.Tensor],
    device: str,
    n_steps: int = 20,
) -> float:
    """Empirically measures real tokens/sec for `model` on `device` by
    timing `n_steps` real forward+backward training steps on
    `sample_batch` (Requirement 11.4) — deliberately not a theoretical
    FLOPs estimate, since real Apple-Silicon MPS throughput isn't reliably
    known in advance. One extra, untimed warm-up step runs first so a
    backend's first-call overhead (e.g. MPS's first Metal shader
    compilation) isn't counted against the timed average.
    """
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")

    x, y = sample_batch
    model = model.to(device)
    x = x.to(device)
    y = y.to(device)
    model.train()
    tokens_per_step = x.shape[0] * x.shape[1]

    _training_step(model, x, y)
    _synchronize(device)

    start = time.perf_counter()
    for _ in range(n_steps):
        _training_step(model, x, y)
    _synchronize(device)
    elapsed = time.perf_counter() - start

    return tokens_per_step * n_steps / elapsed


def epochs_for_budget(total_tokens: int, unique_tokens: int) -> float:
    """How many passes over the available unique tokens a `total_tokens`
    training budget requires; logs a warning past 4 epochs, the point past
    which repeated-data returns are known to diminish sharply (Muennighoff
    et al., 2023), Requirement 11.3.
    """
    if total_tokens < 1:
        raise ValueError(f"total_tokens must be >= 1, got {total_tokens}")
    if unique_tokens < 1:
        raise ValueError(f"unique_tokens must be >= 1, got {unique_tokens}")

    epochs = total_tokens / unique_tokens
    if epochs > _EPOCH_WARNING_THRESHOLD:
        logger.warning(
            "training budget of %d tokens over %d unique tokens is %.2f epochs "
            "(> %d) -- repeated-data returns diminish sharply past this point "
            "(Muennighoff et al., 2023)",
            total_tokens,
            unique_tokens,
            epochs,
            _EPOCH_WARNING_THRESHOLD,
        )
    return epochs
