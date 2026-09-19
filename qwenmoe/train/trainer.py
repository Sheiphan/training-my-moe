"""Training loop (Requirement 5): AdamW or Muon (`TrainConfig.optimizer`,
moe-modernization Requirement 1) with a warmup + cosine-or-WSD
learning-rate schedule, gradient clipping, optional bf16 autocast +
output z-loss (moe-modernization Requirement 3), optional cross-document
attention masking (moe-modernization Requirement 10, `TrainConfig.
document_masking` -- consumes `(x, y, segment_ids)` batches from a
`TokenDataset(return_segments=True)`), device auto-selection (`mps` else
`cpu`), periodic console + JSON-lines logging of per-step metrics (now
including gradient norm and z-loss) and per-expert token counts (consumed
later by the training-observability scenes, task 14), periodic validation
loss/perplexity, checkpoint save/resume, and gradient accumulation
(`TrainConfig.grad_accum_steps` -- journal 038: an effective batch of
`batch_size * grad_accum_steps` at the peak memory cost of just
`batch_size`, so a memory-constrained model size can still be trained at
the same effective batch as a smaller model it's being compared against).
"""
from __future__ import annotations

import collections
import contextlib
import dataclasses
import itertools
import json
import logging
import math
import os
import resource
import shutil
import signal
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from qwenmoe.config import ModelConfig, TrainConfig
from qwenmoe.data.sampler import StepOffsetSampler, strided_eval_offsets
from qwenmoe.model.transformer import QwenMoEModel
from qwenmoe.train.optim import build_optimizer
from qwenmoe.train.scaling import (
    chinchilla_budget,
    count_params,
    epochs_for_budget,
    moe_budget_note,
)

logger = logging.getLogger(__name__)


@dataclass
class TrainResult:
    """What a completed (or checkpointed-partway-through) `Trainer.fit`
    call produced. Not part of design.md's Data Models section (written
    before `Trainer` existed) -- added here so task 15.2's
    `run_comparison` has a single object to build a per-variant
    `VariantResult` from.
    """

    final_step: int
    final_train_loss: float
    final_val_loss: float
    val_perplexity: float
    log_path: Path
    checkpoint_path: Path | None


def select_device(requested: Literal["auto", "cuda", "mps", "cpu"]) -> str:
    """Requirement 3.2, extended to CUDA in journal 042: `cuda` if
    available, else `mps`, else `cpu`; `requested="cpu"` always forces
    `cpu`. Warns (doesn't raise) if an accelerator was explicitly requested
    but isn't available (design.md Error Handling: a missing backend falls
    back rather than crashing).

    Returns the **bare** string `"cuda"`, never `"cuda:0"`. That is not
    cosmetic: `torch.autocast(device_type=device)` in
    `_forward_backward_core` requires the bare form, and passing the
    trainer's own device string straight through is the whole reason bf16
    needed no separate CUDA plumbing.
    """
    if requested == "cpu":
        return "cpu"
    if requested != "mps" and torch.cuda.is_available():
        return "cuda"
    if requested != "cuda" and torch.backends.mps.is_available():
        return "mps"
    if requested in ("cuda", "mps"):
        logger.warning("device=%r requested but unavailable; falling back to cpu", requested)
    return "cpu"


def lr_multiplier(
    step: int,
    warmup_steps: int,
    total_steps: int,
    schedule: Literal["cosine", "wsd"] = "cosine",
    min_lr_ratio: float = 0.0,
    decay_fraction: float = 0.2,
) -> float:
    """Linear warmup from `1/warmup_steps` to `1.0` over `warmup_steps`,
    then one of two post-warmup decay curves (Requirement 5.1;
    moe-modernization Requirement 2), both landing on `min_lr_ratio`
    (default `0.0`, today's exact behavior) rather than `0.0` when set
    above zero. `step` is 0-indexed (the step about to run); a
    `warmup_steps` of 0 skips straight to the decay phase.

    - `schedule="cosine"` (default): smooth cosine decay across the *entire*
      post-warmup span, `1.0` at `step == warmup_steps` down to
      `min_lr_ratio` at `step == total_steps`.
    - `schedule="wsd"` (Hu et al. 2024 / MiniCPM's Warmup-Stable-Decay):
      holds the multiplier flat at `1.0` for a "stable" plateau, then
      decays *linearly* to `min_lr_ratio` only over the final
      `decay_fraction` of `total_steps` -- so a checkpoint saved anywhere
      in the plateau is a valid launch point for a fresh, shorter decay
      ("cooldown branch"), which a schedule that decays from the very
      first post-warmup step can't offer.
    """
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    if total_steps <= warmup_steps:
        return 1.0

    if schedule == "wsd":
        decay_steps = max(1, round(decay_fraction * total_steps))
        decay_start = max(warmup_steps, total_steps - decay_steps)
        if step < decay_start:
            return 1.0  # stable phase: held at peak
        progress = min((step - decay_start) / (total_steps - decay_start), 1.0)
        return 1.0 - (1.0 - min_lr_ratio) * progress  # linear decay to the floor

    if schedule != "cosine":
        raise ValueError(f"Unknown lr_schedule: {schedule!r}")
    progress = min((step - warmup_steps) / (total_steps - warmup_steps), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


def _set_lr(
    optimizer: torch.optim.Optimizer,
    base_lr: float,
    step: int,
    total_steps: int,
    config: TrainConfig,
) -> float:
    """Applies one shared `lr_multiplier` to every param group, but scaled
    from *that group's own* peak learning rate rather than a single global
    one -- `build_optimizer` stamps each group with `"base_lr"` at
    construction (`config.lr` for every AdamW group, `config.muon_lr or
    config.lr` for the Muon group when `optimizer="muon"`), so a Muon run's
    two sub-optimizers can ride the same warmup/decay *shape* at two
    different peak magnitudes (Requirement 1.6 / design.md: "the multiplier
    from lr_multiplier applies to both"). Groups built before this field
    existed (there are none left in this codebase, but this keeps the
    function total) fall back to the passed-in `base_lr`. Returns the
    applied rate for the *default* (`config.lr`-based) group, which is what
    the step log records.
    """
    multiplier = lr_multiplier(
        step,
        config.warmup_steps,
        total_steps,
        schedule=config.lr_schedule,
        min_lr_ratio=config.min_lr_ratio,
        decay_fraction=config.decay_fraction,
    )
    for group in optimizer.param_groups:
        group["lr"] = group.get("base_lr", base_lr) * multiplier
    return base_lr * multiplier


def _move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: str) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def _infinite_batches(loader: DataLoader):
    """Yields batches forever, so `Trainer` is driven purely by step count
    (`TrainConfig.total_tokens`), not by epoch count.

    Since journal 041 the training loader is driven by `StepOffsetSampler`,
    which is itself endless, so the `while True` never actually comes
    around a second time. It's kept because the contract this function
    provides -- "never raises StopIteration" -- is what the training loop
    depends on, and that shouldn't silently become the sampler's private
    responsibility.
    """
    while True:
        yield from loader


def _collect_expert_counts(model: QwenMoEModel) -> torch.Tensor:
    """Per-layer, per-routed-expert dispatched-token counts from the most
    recent forward call (Requirement 8.2), read from the
    `MoELayer.last_expert_counts` cache each block's MoE layer populates.

    Returns a `(n_layers, n_routed_experts)` tensor **on the model's
    device**, deliberately not a list of lists (journal 042). This is
    called after every micro-batch, and the previous `.tolist()` forced one
    host-device synchronization *per layer* -- ten pipeline stalls per
    micro-batch on `SCALE_300M`, on every step, even though the result is
    discarded on the ~90% of steps that aren't logging steps. Converting is
    now the caller's job, done once inside the logging branch.
    """
    return torch.stack([block.moe.last_expert_counts for block in model.blocks])


def _forward_backward_core(
    model: QwenMoEModel,
    x: torch.Tensor,
    y: torch.Tensor,
    device: str,
    precision: Literal["fp32", "bf16"],
    z_loss_weight: float,
    segment_ids: torch.Tensor | None,
    loss_scale: float,
) -> dict:
    """Forward pass + `(loss_scale * total_loss).backward()` for ONE
    micro-batch -- the piece shared by `_forward_backward_step` (a full
    optimizer step, `loss_scale=1.0`, called once) and gradient
    accumulation's `_run_step` (`loss_scale = 1/grad_accum_steps`, called
    `grad_accum_steps` times per optimizer step with gradients accumulating
    across calls -- zero_grad/clip/optimizer.step() are the *caller's*
    responsibility, not this function's, since they bracket the whole
    accumulation window rather than one micro-batch).

    Returns a dict of Python floats (or `None` / `list[float]`) with keys
    `lm_loss`, `aux_loss`, `z_loss`, `mtp_loss`, `mtp_losses` -- each `None`
    when that term is inactive (no balancer, `z_loss_weight == 0`,
    `mtp_depth == 0`). These are always the *unscaled* per-token loss
    values -- `loss_scale` only rescales the backward pass so accumulated
    gradients average correctly over the effective batch; it never changes
    what's reported (averaging *reported* losses across micro-batches is
    `_average_metrics`'s job, not this function's).

    Precision (Requirement 3.1, 3.2): with `precision == "bf16"` the forward
    pass runs under `torch.autocast(bfloat16)` -- matmuls drop to bf16 while
    autocast's own op policy keeps softmax / normalization / cross-entropy in
    fp32, so no gradient scaler is needed (bf16 shares fp32's exponent
    range). The logits are then cast up to fp32 before the loss and z-loss so
    the whole loss subgraph is fp32 regardless of `precision`; in the
    `"fp32"` path `.float()` on an already-fp32 tensor is a no-op, keeping
    that path numerically identical to before this task.

    MTP (moe-modernization Requirement 6): `y` is passed to `model` as
    `targets` so the Multi-Token-Prediction heads (if any) can compute their
    own cross-entropy terms; the combined loss gains
    `model.config.mtp_lambda * mean(mtp_losses)`. With `mtp_depth == 0` the
    model has no MTP heads and `targets` is ignored -- the forward is
    byte-identical to before this task.

    Document masking (moe-modernization Requirement 10.2): `segment_ids`
    (`(batch, seq_len)` int, or `None`) is forwarded to the model, which
    turns it into an attention mask blocking cross-document attention.
    `None` (the trainer passes it only when `TrainConfig.document_masking`)
    -> forward byte-identical to before this task (Requirement 10.3).
    """
    autocast_ctx: contextlib.AbstractContextManager = (
        torch.autocast(device_type=device, dtype=torch.bfloat16)
        if precision == "bf16"
        else contextlib.nullcontext()
    )
    with autocast_ctx:
        logits, extras = model(x, targets=y, segment_ids=segment_ids)

    logits = logits.float()  # loss + z-loss always in fp32 (Requirement 3.1)
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
    aux_loss = extras.get("aux_loss")
    total_loss = loss if aux_loss is None else loss + aux_loss

    z_loss: torch.Tensor | None = None
    if z_loss_weight > 0:
        # `logsumexp(logits)` is `log(sum_v exp(logit_v))` -- the log of the
        # softmax normalizer Z. Cross-entropy is invariant to adding a
        # constant to every logit, so nothing else constrains the logits'
        # absolute scale; left free it drifts up and bf16 then rounds the
        # large `exp` sum badly. Penalizing `(log Z)^2` pulls Z back toward
        # 1 without touching the logit *differences* that set the
        # probabilities (PaLM Sec. 5 / ST-MoE, typical weight ~1e-4).
        log_z = torch.logsumexp(logits.reshape(-1, logits.size(-1)), dim=-1)
        z_loss_term = z_loss_weight * log_z.pow(2).mean()
        total_loss = total_loss + z_loss_term
        # journal 042: deliberately NOT `.item()` here. This line sits
        # *before* `backward()` is enqueued, so reading the value drains the
        # pipeline and leaves the accelerator idle while Python walks to the
        # backward call -- strictly worse than a sync after it. The value is
        # detached and converted by `_average_metrics`, on logging steps only.
        z_loss = z_loss_term.detach()

    mtp_loss: torch.Tensor | None = None
    mtp_losses: list[torch.Tensor] | None = None
    mtp_mean = extras.get("mtp_loss")
    if mtp_mean is not None:
        total_loss = total_loss + model.config.mtp_lambda * mtp_mean
        mtp_loss = mtp_mean.detach()
        mtp_losses = [t.detach() for t in extras["mtp_losses"]]

    (total_loss * loss_scale).backward()
    return {
        "lm_loss": loss.detach(),
        "aux_loss": aux_loss.detach() if aux_loss is not None else None,
        "z_loss": z_loss,
        "mtp_loss": mtp_loss,
        "mtp_losses": mtp_losses,
    }


def _forward_backward_step(
    model: QwenMoEModel,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    y: torch.Tensor,
    grad_clip_norm: float,
    device: str = "cpu",
    precision: Literal["fp32", "bf16"] = "fp32",
    z_loss_weight: float = 0.0,
    segment_ids: torch.Tensor | None = None,
) -> tuple[dict, float]:
    """One full optimizer step on a single micro-batch (the
    `grad_accum_steps == 1` case -- gradient accumulation's multi-micro-batch
    path lives in `_run_step`, which calls `_forward_backward_core` directly
    instead of this wrapper). Returns `(metrics, grad_norm)` -- see
    `_forward_backward_core` for what `metrics` contains and how precision/
    MTP/document-masking are handled.
    """
    model.train()
    optimizer.zero_grad(set_to_none=True)
    metrics = _forward_backward_core(
        model, x, y, device, precision, z_loss_weight, segment_ids, loss_scale=1.0
    )
    # Returns the *pre-clip* global norm (Requirement 1.3) -- clip_grad_norm_
    # computes it before rescaling, so this doubles as a training-health
    # signal: a rising norm here typically precedes a loss spike.
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
    optimizer.step()
    return _average_metrics([metrics]), float(grad_norm)


def _average_metrics(metrics_list: list[dict]) -> dict:
    """Means each numeric field across accumulated micro-batches' metrics
    dicts (as produced by `_forward_backward_core`) -- e.g. `lm_loss`
    becomes the mean per-token loss over the whole accumulation window,
    matching what it would have been had the micro-batches been one real
    batch. `None`-valued fields (an inactive loss term) are skipped rather
    than treated as 0, and stay `None` if every micro-batch had `None`.
    `mtp_losses` (a list, one entry per MTP depth) is averaged elementwise.

    With a single-element `metrics_list` (`grad_accum_steps == 1`, the
    default) this is the identity on that one dict -- every existing
    single-micro-batch test's exact values are unaffected.

    Since journal 042 `_forward_backward_core` returns **detached tensors**
    rather than Python floats, so the averaging happens on-device and the
    single host-device synchronization lands here -- once per *optimizer*
    step, after every micro-batch's `backward()` has been enqueued, instead
    of two-to-five times per micro-batch with two of them landing before
    `backward()`. The returned dict still holds plain floats, so every
    caller and every existing test is unchanged.
    """

    def _to_float(value):
        return float(value) if isinstance(value, torch.Tensor) else value

    def _mean(key: str) -> float | None:
        values = [m[key] for m in metrics_list if m[key] is not None]
        return _to_float(sum(values) / len(values)) if values else None

    mtp_losses_lists = [m["mtp_losses"] for m in metrics_list if m["mtp_losses"] is not None]
    mtp_losses = (
        [
            _to_float(sum(depth_losses) / len(depth_losses))
            for depth_losses in zip(*mtp_losses_lists)
        ]
        if mtp_losses_lists
        else None
    )

    return {
        "lm_loss": _mean("lm_loss"),
        "aux_loss": _mean("aux_loss"),
        "z_loss": _mean("z_loss"),
        "mtp_loss": _mean("mtp_loss"),
        "mtp_losses": mtp_losses,
    }


def _sum_expert_counts(
    per_microbatch: list[torch.Tensor] | list[list[list[int]]],
) -> torch.Tensor | list[list[int]]:
    """Elementwise-sums `_collect_expert_counts` readings taken after each
    micro-batch in an accumulation window, so the logged per-expert token
    counts reflect the *whole* effective batch rather than just the last
    micro-batch (`MoELayer.last_expert_counts` is overwritten by every
    forward call). With one micro-batch (`grad_accum_steps == 1`) this is
    the identity on that single reading.

    Accepts either a list of `(n_layers, n_experts)` tensors (what
    `_collect_expert_counts` produces since journal 042, summed on-device
    with no synchronization) or the nested-list form, which the
    `docs/math/verify_*.py` scripts pass directly. Output mirrors the input
    kind.
    """
    if isinstance(per_microbatch[0], torch.Tensor):
        # The hot path since journal 042: stays on-device, no sync.
        return torch.stack(per_microbatch).sum(dim=0)
    n_layers = len(per_microbatch[0])
    return [
        [sum(counts[layer_idx][expert_idx] for counts in per_microbatch)
         for expert_idx in range(len(per_microbatch[0][layer_idx]))]
        for layer_idx in range(n_layers)
    ]


def _run_step(
    model: QwenMoEModel,
    optimizer: torch.optim.Optimizer,
    micro_batches: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]],
    device: str,
    step: int,
    total_steps: int,
    config: TrainConfig,
) -> tuple[dict, float, float, str, QwenMoEModel, list[list[int]]]:
    """Runs one *optimizer* step, accumulating gradients over every
    `(x, y, segment_ids)` triple in `micro_batches` (moe-modernization
    follow-up, journal 038: gradient accumulation) before a single clip +
    `optimizer.step()`. `len(micro_batches) == 1` (`config.grad_accum_steps
    == 1`, the default) reduces to exactly the old one-micro-batch-per-step
    behavior (`loss_scale=1.0`, a single `_forward_backward_core` call), so
    every pre-existing call site and test is unaffected.

    Falls back to `cpu` if a real `RuntimeError` hits mid-window on `mps`
    (design.md Error Handling: "MPS unavailable/errors during a step -> fall
    back to cpu with a logged warning, never a hard crash", Requirement 3.2
    -- this also catches a backend that can't run the bf16 autocast path).
    On fallback the *whole* accumulation window is restarted from a fresh
    zero_grad on cpu rather than resumed partway through -- simpler than
    threading "which micro-batch failed" through the retry, and just as
    correct given how rare this path is (a warned, not silent, fallback).

    Returns `(metrics, grad_norm, current_lr, device, model, expert_counts)`
    -- the (possibly updated) `device`/`model` so the caller keeps using
    them for subsequent steps, `metrics` averaged across the window
    (`_average_metrics`), and `expert_counts` summed across it
    (`_sum_expert_counts`) so it reflects the whole effective batch, not
    just the last micro-batch.

    `segment_ids` (moe-modernization Requirement 10.2) is moved to `device`
    alongside `x`/`y` and passed through for document masking; `None` when
    `TrainConfig.document_masking` is off.
    """
    current_lr = _set_lr(optimizer, config.lr, step, total_steps, config)
    loss_scale = 1.0 / len(micro_batches)

    def _accumulate(device: str) -> tuple[list[dict], list[list[list[int]]]]:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        step_metrics = []
        expert_counts_per_microbatch = []
        for x, y, segment_ids in micro_batches:
            seg = segment_ids.to(device) if segment_ids is not None else None
            step_metrics.append(
                _forward_backward_core(
                    model,
                    x.to(device),
                    y.to(device),
                    device,
                    config.precision,
                    config.z_loss_weight,
                    seg,
                    loss_scale,
                )
            )
            expert_counts_per_microbatch.append(_collect_expert_counts(model))
        return step_metrics, expert_counts_per_microbatch

    try:
        step_metrics, expert_counts_per_microbatch = _accumulate(device)
    except RuntimeError as exc:
        if device != "mps":
            raise
        logger.warning(
            "RuntimeError on mps at step %d (%s); falling back to cpu for the rest of this run",
            step,
            exc,
        )
        model = model.to("cpu")
        _move_optimizer_state_to_device(optimizer, "cpu")
        device = "cpu"
        step_metrics, expert_counts_per_microbatch = _accumulate(device)

    # Returns the *pre-clip* global norm (Requirement 1.3) -- clip_grad_norm_
    # computes it before rescaling, so this doubles as a training-health
    # signal: a rising norm here typically precedes a loss spike.
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
    optimizer.step()
    metrics = _average_metrics(step_metrics)
    expert_counts = _sum_expert_counts(expert_counts_per_microbatch)
    # journal 042: `grad_norm` stays a tensor here. `float()` on it is a
    # host-device sync, and the caller only needs the number on logging
    # steps -- so the conversion moved there.
    return metrics, grad_norm, current_lr, device, model, expert_counts


def _evaluate(
    model: QwenMoEModel, val_loader: DataLoader, device: str, max_eval_batches: int
) -> tuple[float, float]:
    """Mean validation loss and perplexity `exp(loss)` over at most
    `max_eval_batches` of `val_loader` (Requirement 5.3) -- capped so
    evaluating doesn't require a full pass over a potentially huge
    validation split every `eval_interval_steps`.

    Since journal 041 `val_loader` is driven by `strided_eval_offsets`, so
    those batches are spread evenly across the whole split rather than
    being its first `max_eval_batches * batch_size` windows. The cap and
    the sampler are sized from the same number, so the `islice` below is
    normally a no-op; it stays as the guard that gives this function its
    name.

    Deliberately *not* run under `torch.autocast`: measuring validation
    loss in fp32 keeps it comparable across precision changes and against
    this project's earlier runs, at a cost of a few seconds per eval.

    A `val_loader` over a `TokenDataset(return_segments=True)` yields
    `(x, y, segment_ids)`; the segment ids are forwarded so validation
    perplexity is measured under the same document masking as training
    (moe-modernization Requirement 10.2). A plain 2-tuple loader is
    unchanged.
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for batch in itertools.islice(val_loader, max_eval_batches):
            x, y = batch[0], batch[1]
            segment_ids = batch[2].to(device) if len(batch) == 3 else None
            logits, _ = model(x.to(device), segment_ids=segment_ids)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.to(device).reshape(-1))
            total_loss += loss.item()
            n_batches += 1
    model.train()
    mean_loss = total_loss / n_batches
    return mean_loss, math.exp(mean_loss)


def _write_log_line(log_file, record: dict) -> None:
    log_file.write(json.dumps(record) + "\n")
    log_file.flush()  # a killed run should still leave a durable, readable log


def _format_duration(seconds: float) -> str:
    """`12d 04h`, `4h 07m`, `7m 12s` -- an ETA measured in days is only
    misleading when printed to the second."""
    if not math.isfinite(seconds) or seconds < 0:
        return "n/a"
    seconds = int(seconds)
    days, rem = divmod(seconds, 86_400)
    hours, rem = divmod(rem, 3_600)
    minutes, secs = divmod(rem, 60)
    if days:
        return f"{days}d {hours:02d}h"
    if hours:
        return f"{hours}h {minutes:02d}m"
    return f"{minutes}m {secs:02d}s"


def _rss_bytes() -> int:
    """Peak resident set size of *this* process. `ru_maxrss` is bytes on
    macOS and kilobytes on Linux. Journal 040 established the in-process
    reading as the trustworthy one here: `/usr/bin/time -l` was measuring
    the `uv` wrapper and reported 2.72GB for a run whose output alone was
    4GB.
    """
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw if sys.platform == "darwin" else raw * 1024


def _mps_allocated_bytes() -> int | None:
    if not torch.backends.mps.is_available():
        return None
    with contextlib.suppress(Exception):
        return int(torch.mps.current_allocated_memory())
    return None


def _mps_driver_bytes() -> int | None:
    """Total memory the Metal driver holds, which is *not* the same as
    `current_allocated_memory` and is the figure that actually decides
    whether this machine starts swapping (journal 041). Logged separately
    because watching only the first one is what made journal 037's cliff
    look like a clean fit right up until the run hung.
    """
    if not torch.backends.mps.is_available():
        return None
    with contextlib.suppress(Exception):
        return int(torch.mps.driver_allocated_memory())
    return None


def _device_memory_bytes(device: str) -> tuple[int | None, int | None]:
    """`(allocated, reserved)` for whichever accelerator is in use
    (journal 042).

    The second figure is the one that matters, and the parallel between
    the backends is exact: CUDA's `memory_reserved` is what the caching
    allocator holds from the driver, just as MPS's `driver_allocated_memory`
    is. Journal 041 found the hard way that watching only *allocated* is
    how a run looks like a clean fit right up until it starts swapping --
    on MPS that was a 9.5 -> 20.5GB climb the allocated figure never
    showed. Reserved is the number that predicts an OOM.

    `(None, None)` on cpu, and on any backend whose probe raises.
    """
    if device.startswith("cuda"):
        with contextlib.suppress(Exception):
            return int(torch.cuda.memory_allocated()), int(torch.cuda.memory_reserved())
        return None, None
    if device == "mps":
        return _mps_allocated_bytes(), _mps_driver_bytes()
    return None, None


def _cuda_hardware() -> dict:
    """Identifies the GPU a run actually executed on, so a journal entry
    can cite hardware rather than assert it. Empty off CUDA."""
    if not torch.cuda.is_available():
        return {}
    info: dict = {}
    with contextlib.suppress(Exception):
        info["cuda_device_name"] = torch.cuda.get_device_name()
        info["cuda_capability"] = ".".join(str(v) for v in torch.cuda.get_device_capability())
        info["cuda_total_memory_bytes"] = int(
            torch.cuda.get_device_properties(0).total_memory
        )
        info["cuda_version"] = torch.version.cuda
        # Whether SDPA is allowed to pick the flash backend -- the
        # difference between a fast and a very slow attention.
        info["flash_sdp_enabled"] = bool(torch.backends.cuda.flash_sdp_enabled())
        info["tf32_matmul"] = torch.get_float32_matmul_precision()
    return info


def _git_sha() -> str | None:
    """The commit the run started from, so a log can be tied back to the
    code that produced it. Best-effort: a run from a tarball is still a
    valid run."""
    with contextlib.suppress(Exception):
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=Path(__file__).resolve().parent,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    return None


def _run_start_record(
    *,
    model: QwenMoEModel,
    config: TrainConfig,
    device: str,
    total_steps: int,
    tokens_per_step: int,
    start_step: int,
    n_train_windows: int,
    n_val_windows: int,
    tokenizer_meta: dict | None,
    fingerprint: dict,
) -> dict:
    """One self-describing record at the head of every run (journal 041).

    `scripts/train.py` already computed most of this and logged it to a
    console that scrolls away. Putting it in the log instead means the
    journal entry for a run can be written *from the run*, which is what
    makes this project's "no fabricated numbers" rule structural rather
    than a matter of discipline.
    """
    breakdown = count_params(model)
    unique_train_tokens = n_train_windows + config.seq_len
    return {
        "type": "run_start",
        "wall_clock": time.time(),
        "device": device,
        "precision": config.precision,
        "torch_version": torch.__version__,
        "git_sha": _git_sha(),
        **_cuda_hardware(),  # journal 042: cite the GPU, don't assert it
        "start_step": start_step,
        "total_steps": total_steps,
        "tokens_per_step": tokens_per_step,
        "total_tokens": config.total_tokens,
        "params_total": breakdown.total,
        "params_active": breakdown.active,
        "params_embedding": breakdown.embedding,
        "chinchilla_budget_tokens": chinchilla_budget(breakdown.active),
        "moe_budget_note": moe_budget_note(breakdown.total, breakdown.active),
        "epochs_over_train_split": epochs_for_budget(
            config.total_tokens, unique_train_tokens
        ),
        "n_train_windows": n_train_windows,
        "n_val_windows": n_val_windows,
        "unique_train_tokens": unique_train_tokens,
        "data_fingerprint": fingerprint,
        "tokenizer_meta": tokenizer_meta,
        "train_config": dataclasses.asdict(config),
        "model_config": dataclasses.asdict(model.config),
    }


def data_fingerprint(config: TrainConfig, n_train_windows: int) -> dict:
    """The knobs that, if changed across a resume, would silently corrupt
    the run rather than fail it (journal 041).

    These are exactly the inputs to `StepOffsetSampler`'s stream and to
    window indexing: change one and the resumed run draws different data
    at every step *and* moves the learning-rate schedule under a
    checkpoint already partway along it, with no error and no visible
    symptom beyond a worse model. Over a multi-day run with a dozen
    restarts, mistyping one of these is a question of when, not if.

    `total_tokens` is deliberately **not** here. It changes `total_steps`
    and therefore the schedule, but changing it is also the *intended* way
    to extend a run or to size a WSD cooldown -- so it gets a warning
    rather than a refusal.
    """
    return {
        "seq_len": config.seq_len,
        "batch_size": config.batch_size,
        "grad_accum_steps": config.grad_accum_steps,
        "seed": config.seed,
        "n_train_windows": n_train_windows,
    }


def save_checkpoint(
    path: str | Path,
    model: QwenMoEModel,
    optimizer: torch.optim.Optimizer | None,
    model_config: ModelConfig,
    train_config: TrainConfig,
    step: int,
    tokenizer_meta: dict | None,
    *,
    total_steps: int | None = None,
    fingerprint: dict | None = None,
) -> None:
    """Model + optimizer + config + step (Requirement 5.4), per design.md's
    checkpoint format, written atomically (journal 041).

    `optimizer=None` writes a **weights-only** checkpoint: enough to
    generate from, evaluate, or visualize, but not to resume. At ~346M
    parameters a full checkpoint is `12 bytes x params` (fp32 weights plus
    AdamW's two moments) = 4.16GB, against 1.39GB weights-only -- which is
    what makes it affordable to keep a milestone every 20k steps for the
    whole run.

    The write goes to a `.tmp` sibling and is then `os.replace`d, which is
    atomic within a filesystem. Without that, a Ctrl-C or a full disk
    partway through `torch.save` leaves a truncated multi-gigabyte file
    sitting at exactly the path `--resume-from` will reach for.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "model_config": model_config,
        "train_config": train_config,
        "step": step,
        "tokenizer_meta": tokenizer_meta,
        "total_steps": total_steps,
        "data_fingerprint": fingerprint,
    }
    tmp_path = path.with_name(path.name + ".tmp")
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def estimate_checkpoint_bytes(model: QwenMoEModel, *, with_optimizer: bool) -> int:
    """Bytes a checkpoint of this model will occupy: 4 per fp32 parameter,
    plus 8 more for AdamW's `exp_avg`/`exp_avg_sq` when optimizer state is
    included. Validated against a real file -- `runs/scale_300m`'s
    `step_100.pt` is 4,013,033,885 bytes against `12 x 334,382,336 =
    4,012,588,032` predicted, the 445KB difference being pickle framing.
    """
    n_params = sum(p.numel() for p in model.parameters())
    return n_params * (12 if with_optimizer else 4)


def prune_checkpoints(
    checkpoint_dir: Path, *, keep_last: int, keep_every: int, protect: set[Path]
) -> list[Path]:
    """Deletes rolling checkpoints that are neither recent nor milestones,
    and returns what was removed (journal 041).

    Retention is tiered because the two kinds of checkpoint exist for
    different reasons: rolling ones exist to *resume* and are worthless
    once a newer one lands, while milestones exist to *evaluate and
    generate from* across the run's history. Only the first kind needs
    optimizer state.

    Without this, a 164k-step run at `checkpoint_interval_steps=2000`
    writes 82 x 4.16GB = 341GB. The run would die on a failed `torch.save`
    around 18 hours in.
    """
    checkpoints: list[tuple[int, Path]] = []
    for path in checkpoint_dir.glob("step_*.pt"):
        if path.is_symlink():
            continue
        try:
            checkpoints.append((int(path.stem.removeprefix("step_")), path))
        except ValueError:
            continue  # not ours; leave it alone
    checkpoints.sort()

    recent = {path for _, path in checkpoints[-keep_last:]} if keep_last > 0 else set()
    milestones = {
        path for step, path in checkpoints if keep_every > 0 and step % keep_every == 0
    }
    keep = recent | milestones | protect

    removed = []
    for _, path in checkpoints:
        if path not in keep:
            path.unlink(missing_ok=True)
            removed.append(path)
    return removed


def update_latest_pointer(checkpoint_dir: Path, target: Path) -> None:
    """Points `latest.pt` at `target` via an atomically-replaced relative
    symlink, so `--resume-from runs/x/checkpoints/latest.pt` works with no
    change to `load_checkpoint` or `scripts/generate.py`, and `runs/` stays
    relocatable. Falls back to a copy-free no-op if the filesystem refuses
    symlinks -- a missing convenience pointer must never fail a run.
    """
    link = checkpoint_dir / "latest.pt"
    tmp = checkpoint_dir / "latest.pt.tmp"
    try:
        tmp.unlink(missing_ok=True)
        tmp.symlink_to(target.name)
        os.replace(tmp, link)
    except OSError as exc:
        logger.warning("could not update %s: %s", link, exc)
        tmp.unlink(missing_ok=True)


def load_checkpoint(path: str | Path) -> dict:
    """Loads a checkpoint written by `save_checkpoint`. `weights_only=False`
    is required -- the checkpoint's `model_config`/`train_config` are plain
    dataclasses, not just tensors -- and isn't a security concern here since
    every checkpoint this project loads is one it wrote itself locally,
    never a downloaded or user-uploaded file.
    """
    return torch.load(Path(path), map_location="cpu", weights_only=False)


class Trainer:
    """Owns the optimizer/schedule/logging/checkpointing lifecycle around a
    `QwenMoEModel` training run. `output_dir` holds `checkpoints/` (one
    `step_{N}.pt` file per checkpoint interval, plus a `latest.pt` symlink)
    and `logs/train.jsonl` (append-mode, so a resumed run's log history
    stays contiguous).

    `keep_last`/`keep_every` bound what `checkpoints/` costs on disk
    (journal 041); `0` for either disables that tier, and `keep_last=0`
    with `keep_every=0` restores the pre-041 behaviour of keeping
    everything.
    """

    def __init__(
        self,
        output_dir: str | Path,
        tokenizer_meta: dict | None = None,
        *,
        keep_last: int = 0,
        keep_every: int = 0,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.log_path = self.output_dir / "logs" / "train.jsonl"
        self.tokenizer_meta = tokenizer_meta
        self.keep_last = keep_last
        self.keep_every = keep_every
        self._stop_requested = False

    @contextlib.contextmanager
    def _graceful_stop(self):
        """Turns SIGINT/SIGTERM into a flag the training loop checks after
        its next checkpoint, rather than an exception mid-step.

        Saving from inside the handler is the obvious thing and is how you
        get a corrupt multi-gigabyte checkpoint: `torch.save` would run
        with MPS work still in flight and the optimizer possibly halfway
        through a step. A second signal restores the default handler and
        re-raises, so an impatient Ctrl-C still works.

        Handlers are restored on the way out -- `Trainer.fit` is called
        ~20 times by the test suite and from notebooks, and a leaked
        handler would outlive the run that installed it.
        """
        previous: dict[int, object] = {}

        def request_stop(signum, _frame):
            if self._stop_requested:  # second signal: let it through
                signal.signal(signum, signal.SIG_DFL)
                os.kill(os.getpid(), signum)
                return
            self._stop_requested = True
            logger.warning(
                "signal %s received -- finishing this step, then checkpointing and exiting "
                "(send it again to abort immediately)",
                signal.Signals(signum).name,
            )

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                previous[sig] = signal.signal(sig, request_stop)
            except (ValueError, OSError):
                pass  # not the main thread; the run just isn't interruptible this way
        try:
            yield
        finally:
            for sig, handler in previous.items():
                with contextlib.suppress(ValueError, OSError, TypeError):
                    signal.signal(sig, handler)

    def _checkpoint(
        self,
        *,
        model: QwenMoEModel,
        optimizer: torch.optim.Optimizer,
        config: TrainConfig,
        step: int,
        total_steps: int,
        fingerprint: dict,
        log_file,
    ) -> Path | None:
        """Writes the rolling checkpoint, a weights-only milestone when one
        is due, repoints `latest.pt`, and prunes (journal 041).

        Returns the rolling checkpoint's path, or `None` if it was skipped
        for lack of disk. Skipping costs one interval of progress; writing
        into a full disk costs the run, because `torch.save` raises from
        inside the training loop and leaves a truncated file behind.
        """
        path = self.checkpoint_dir / f"step_{step}.pt"
        needed = estimate_checkpoint_bytes(model, with_optimizer=True)
        free = shutil.disk_usage(self.checkpoint_dir).free
        if free < 2 * needed:
            logger.error(
                "skipping checkpoint at step %d: %.1fGB free, need %.1fGB headroom "
                "(2x the %.1fGB checkpoint). Free disk space or lower --keep-every.",
                step,
                free / 1e9,
                2 * needed / 1e9,
                needed / 1e9,
            )
            _write_log_line(
                log_file,
                {
                    "type": "checkpoint_skipped",
                    "step": step,
                    "wall_clock": time.time(),
                    "free_bytes": free,
                    "needed_bytes": needed,
                },
            )
            return None

        save_checkpoint(
            path,
            model,
            optimizer,
            model.config,
            config,
            step,
            self.tokenizer_meta,
            total_steps=total_steps,
            fingerprint=fingerprint,
        )
        update_latest_pointer(self.checkpoint_dir, path)

        # Milestones carry no optimizer state: at ~346M params that is
        # 1.39GB against 4.16GB, which is what makes it affordable to keep
        # one every `keep_every` steps for a whole multi-day run.
        milestone: Path | None = None
        if self.keep_every > 0 and step % self.keep_every == 0:
            milestone = self.checkpoint_dir / f"milestone_{step}.pt"
            save_checkpoint(
                milestone,
                model,
                None,
                model.config,
                config,
                step,
                self.tokenizer_meta,
                total_steps=total_steps,
                fingerprint=fingerprint,
            )

        if self.keep_last > 0:
            removed = prune_checkpoints(
                self.checkpoint_dir,
                keep_last=self.keep_last,
                keep_every=0,  # milestones live under their own prefix
                protect={path},
            )
            if removed:
                logger.info(
                    "pruned %d old checkpoint(s): %s",
                    len(removed),
                    ", ".join(p.name for p in removed),
                )
        _write_log_line(
            log_file,
            {
                "type": "checkpoint",
                "step": step,
                "wall_clock": time.time(),
                "path": str(path),
                "milestone": str(milestone) if milestone else None,
                "bytes": path.stat().st_size if path.exists() else None,
                "free_bytes_after": shutil.disk_usage(self.checkpoint_dir).free,
            },
        )
        return path

    def fit(
        self,
        model: QwenMoEModel,
        train_dataset: Dataset,
        val_dataset: Dataset,
        config: TrainConfig,
        resume_from: str | Path | None = None,
        *,
        allow_schedule_change: bool = False,
    ) -> TrainResult:
        if len(train_dataset) < config.batch_size:
            raise ValueError(
                f"train_dataset has {len(train_dataset)} windows, fewer than "
                f"batch_size ({config.batch_size}); cannot form a single batch"
            )
        if len(val_dataset) < config.batch_size:
            raise ValueError(
                f"val_dataset has {len(val_dataset)} windows, fewer than "
                f"batch_size ({config.batch_size}); cannot form a single batch"
            )
        if config.document_masking and len(train_dataset[0]) != 3:
            raise ValueError(
                "TrainConfig.document_masking=True requires a train_dataset that yields "
                "(x, y, segment_ids) -- construct TokenDataset with return_segments=True "
                "(and an eos_id)"
            )

        device = select_device(config.device)
        torch.manual_seed(config.seed)

        if device.startswith("cuda") and config.tf32:
            # journal 042: lets cuBLAS use tensor cores for the fp32
            # matmuls that bf16 autocast leaves alone. Small under bf16,
            # decisive under --precision fp32 (~50 vs ~400 TFLOP/s on an
            # H100). No effect on mps/cpu, so the laptop path is unchanged.
            torch.set_float32_matmul_precision("high")

        # Requirement 3.2 / design.md Error Handling: state the effective
        # precision up front. bf16 means the forward pass runs under
        # `torch.autocast`; if the backend can't, `_run_step` catches the
        # RuntimeError and drops to cpu for the rest of the run.
        effective_precision = (
            "bf16 (torch.autocast)" if config.precision == "bf16" else "fp32"
        )
        logger.info(
            "effective training precision: %s | device: %s | z_loss_weight: %g",
            effective_precision,
            device,
            config.z_loss_weight,
        )
        if device.startswith("cuda"):
            hardware = _cuda_hardware()
            logger.info(
                "cuda: %s (sm_%s, %.1fGB) | flash sdp %s | tf32 matmul %r",
                hardware.get("cuda_device_name", "?"),
                hardware.get("cuda_capability", "?").replace(".", ""),
                hardware.get("cuda_total_memory_bytes", 0) / 1e9,
                "on" if hardware.get("flash_sdp_enabled") else "OFF",
                hardware.get("tf32_matmul"),
            )

        model = model.to(device)
        optimizer = build_optimizer(model, config)

        fingerprint = data_fingerprint(config, len(train_dataset))

        # Gradient accumulation (moe-modernization follow-up, journal 038):
        # each optimizer step consumes `grad_accum_steps` micro-batches, so
        # the *effective* batch that sets step count is `batch_size *
        # grad_accum_steps`, not `batch_size` alone. `grad_accum_steps == 1`
        # (the default) reduces this to the pre-existing formula exactly.
        effective_batch_size = config.batch_size * config.grad_accum_steps
        total_steps = max(1, config.total_tokens // (effective_batch_size * config.seq_len))
        tokens_per_step = effective_batch_size * config.seq_len
        # A stride-1 dataset has one window per token position, so the
        # unique token count is the window count plus the final window's
        # tail -- this is what `epoch` in the step log is measured against.
        unique_train_tokens = len(train_dataset) + config.seq_len

        global_step = 0
        if resume_from is not None:
            checkpoint = load_checkpoint(resume_from)
            if checkpoint.get("optimizer_state_dict") is None:
                raise ValueError(
                    f"{resume_from} is a weights-only milestone checkpoint and carries no "
                    "optimizer state, so a run cannot be continued from it. Resume from "
                    "checkpoints/latest.pt (or another full checkpoint) instead."
                )
            saved_fingerprint = checkpoint.get("data_fingerprint")
            if (
                saved_fingerprint is not None
                and saved_fingerprint != fingerprint
                and not allow_schedule_change
            ):
                changed = {
                    key: (saved_fingerprint.get(key), fingerprint.get(key))
                    for key in fingerprint
                    if saved_fingerprint.get(key) != fingerprint.get(key)
                }
                raise ValueError(
                    "resuming with different data settings would re-align the data stream "
                    "and move the learning-rate schedule under a partly-trained checkpoint. "
                    f"Changed (was -> now): {changed}. Pass allow_schedule_change=True "
                    "(--allow-schedule-change) if this is deliberate."
                )
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            _move_optimizer_state_to_device(optimizer, device)
            global_step = checkpoint["step"]
            saved_total_steps = checkpoint.get("total_steps")
            if saved_total_steps is not None and saved_total_steps != total_steps:
                # Not an error: extending a run and sizing a WSD cooldown
                # both work exactly this way. But the schedule a resumed
                # step sits on has moved, so say so out loud.
                logger.warning(
                    "total_steps changed across resume (%d -> %d): the learning-rate "
                    "schedule has been re-sized, so step %d now sits at %.1f%% of the "
                    "schedule instead of %.1f%%.",
                    saved_total_steps,
                    total_steps,
                    global_step,
                    100 * global_step / total_steps,
                    100 * global_step / saved_total_steps,
                )

        # journal 041: offsets drawn per optimizer step, not a permutation
        # of every window. `start_step=global_step` is the whole of the
        # resume fix -- it must therefore be built *after* the resume block
        # above has restored `global_step`.
        train_sampler = StepOffsetSampler(
            len(train_dataset),
            batch_size=config.batch_size,
            grad_accum_steps=config.grad_accum_steps,
            seed=config.seed,
            start_step=global_step,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            sampler=train_sampler,
            drop_last=True,
            num_workers=0,  # workers prefetch ahead of the loop, breaking resume alignment
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            sampler=strided_eval_offsets(
                len(val_dataset), config.max_eval_batches * config.batch_size
            ),
            drop_last=True,
        )
        train_iter = _infinite_batches(train_loader)

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        final_train_loss = math.nan
        final_val_loss = math.nan
        final_val_ppl = math.nan
        checkpoint_path: Path | None = None
        recent_elapsed: collections.deque[float] = collections.deque(maxlen=50)

        with self.log_path.open("a", encoding="utf-8") as log_file, self._graceful_stop():
            _write_log_line(
                log_file,
                _run_start_record(
                    model=model,
                    config=config,
                    device=device,
                    total_steps=total_steps,
                    tokens_per_step=tokens_per_step,
                    start_step=global_step,
                    n_train_windows=len(train_dataset),
                    n_val_windows=len(val_dataset),
                    tokenizer_meta=self.tokenizer_meta,
                    fingerprint=fingerprint,
                ),
            )
            while global_step < total_steps:
                # One optimizer step consumes `grad_accum_steps` micro-batches
                # (moe-modernization follow-up, journal 038); with the default
                # of 1 this is exactly the old single-batch-per-step pull.
                micro_batches = []
                for _ in range(config.grad_accum_steps):
                    batch = next(train_iter)
                    x, y = batch[0], batch[1]
                    # (x, y, segment_ids) from a return_segments dataset --
                    # only consumed when document masking is on (Requirement
                    # 10.2/10.3).
                    segment_ids = (
                        batch[2] if (config.document_masking and len(batch) == 3) else None
                    )
                    micro_batches.append((x, y, segment_ids))

                step_start = time.perf_counter()
                device_before = device
                try:
                    metrics, grad_norm, current_lr, device, model, expert_counts = _run_step(
                        model,
                        optimizer,
                        micro_batches,
                        device,
                        global_step,
                        total_steps,
                        config,
                    )
                except torch.OutOfMemoryError:
                    # journal 042: CUDA deliberately has no cpu fallback --
                    # journal 041's lesson is that silently continuing at a
                    # fraction of speed is the worst unattended failure
                    # mode, and on CUDA that fraction is ~1/1000. So record
                    # what the allocator held at the moment it died, then
                    # re-raise. The retry loop resumes from latest.pt.
                    oom_allocated, oom_reserved = _device_memory_bytes(device)
                    _write_log_line(
                        log_file,
                        {
                            "type": "device_oom",
                            "step": global_step,
                            "wall_clock": time.time(),
                            "device": device,
                            "device_allocated_bytes": oom_allocated,
                            "device_reserved_bytes": oom_reserved,
                            "batch_size": config.batch_size,
                            "seq_len": config.seq_len,
                        },
                    )
                    logger.error(
                        "CUDA OOM at step %d (allocated %.2fGB, reserved %.2fGB). "
                        "Resume from checkpoints/latest.pt, or lower --batch-size "
                        "(note that changes the data fingerprint).",
                        global_step,
                        (oom_allocated or 0) / 1e9,
                        (oom_reserved or 0) / 1e9,
                    )
                    raise
                if device != device_before:
                    # `_run_step` drops to cpu on an mps RuntimeError and
                    # logs a warning. On an unattended multi-day run that
                    # warning scrolls away and the run silently continues
                    # at roughly 1% speed, which is the worst available
                    # failure mode (journal 041) -- so it goes in the log
                    # too, where the watcher can alert on it.
                    _write_log_line(
                        log_file,
                        {
                            "type": "device_fallback",
                            "step": global_step,
                            "wall_clock": time.time(),
                            "from": device_before,
                            "to": device,
                        },
                    )
                # journal 041: MPS's caching allocator holds freed blocks,
                # and across steps `driver_allocated_memory` climbs without
                # bound even while `current_allocated_memory` stays flat --
                # measured 9.5GB -> 20.5GB over four steps at seq_len 512,
                # with step time going 2.8s -> 17.9s -> 177s as the machine
                # started swapping. Releasing the cache each step pins the
                # driver figure (12.4GB, flat) for no measurable step cost.
                # This is the cliff journal 037 hit and could only describe
                # as "hung": `current_allocated_memory` never showed it,
                # which is why `check_capacity.py` called it a clean fit.
                if device == "mps":
                    torch.mps.empty_cache()
                elapsed = time.perf_counter() - step_start

                global_step += 1
                recent_elapsed.append(elapsed)
                train_loss = metrics["lm_loss"]
                aux_loss_value = metrics["aux_loss"]
                final_train_loss = train_loss
                is_last_step = global_step == total_steps
                tokens_seen = global_step * tokens_per_step

                if global_step % config.log_interval_steps == 0 or is_last_step:
                    # Median over a rolling window, not the last step alone:
                    # a single slow step (an eval, a checkpoint, a thermal
                    # blip) shouldn't move the ETA by hours.
                    median_elapsed = statistics.median(recent_elapsed)
                    tokens_per_sec = tokens_per_step / elapsed if elapsed > 0 else None
                    device_allocated, device_reserved = _device_memory_bytes(device)
                    logger.info(
                        "step %d/%d | loss %.4f | aux_loss %s | z_loss %s | mtp_loss %s "
                        "| lr %.2e | elapsed %.3fs | %s tok/s | eta %s",
                        global_step,
                        total_steps,
                        train_loss,
                        "n/a" if aux_loss_value is None else f"{aux_loss_value:.4f}",
                        "n/a" if metrics["z_loss"] is None else f"{metrics['z_loss']:.4f}",
                        "n/a" if metrics["mtp_loss"] is None else f"{metrics['mtp_loss']:.4f}",
                        current_lr,
                        elapsed,
                        "n/a" if tokens_per_sec is None else f"{tokens_per_sec:,.0f}",
                        _format_duration((total_steps - global_step) * median_elapsed),
                    )
                    _write_log_line(
                        log_file,
                        {
                            "type": "step",
                            "step": global_step,
                            "train_loss": train_loss,
                            "aux_loss": aux_loss_value,
                            "z_loss": metrics["z_loss"],
                            "mtp_loss": metrics["mtp_loss"],
                            "mtp_losses": metrics["mtp_losses"],
                            "lr": current_lr,
                            "grad_norm": float(grad_norm),
                            "elapsed_s": elapsed,
                            # journal 041: `elapsed_s` alone can't tell "the
                            # run was down for six hours" from "these steps
                            # were slow" -- on a multi-day run that is the
                            # distinction that matters most.
                            "wall_clock": time.time(),
                            "tokens": tokens_seen,
                            "tokens_per_sec": tokens_per_sec,
                            "eta_s": (total_steps - global_step) * median_elapsed,
                            "rss_bytes": _rss_bytes(),
                            # journal 042: the `mps_*` keys stay so every
                            # log already on disk -- and every journal
                            # figure quoting one -- keeps its meaning. The
                            # device-neutral pair is what new readers use.
                            "mps_allocated_bytes": _mps_allocated_bytes(),
                            "mps_driver_bytes": _mps_driver_bytes(),
                            "device_allocated_bytes": device_allocated,
                            "device_reserved_bytes": device_reserved,
                            "device": device,  # changes if the cpu fallback fires
                            "epoch": tokens_seen / unique_train_tokens,
                            # Summed across the accumulation window by
                            # `_run_step` (`_sum_expert_counts`), not just the
                            # last micro-batch's reading.
                            "expert_token_counts": (
                                expert_counts.tolist()
                                if isinstance(expert_counts, torch.Tensor)
                                else expert_counts
                            ),
                        },
                    )

                if global_step % config.eval_interval_steps == 0 or is_last_step:
                    final_val_loss, final_val_ppl = _evaluate(
                        model, val_loader, device, config.max_eval_batches
                    )
                    logger.info(
                        "step %d/%d | val_loss %.4f | val_ppl %.4f",
                        global_step,
                        total_steps,
                        final_val_loss,
                        final_val_ppl,
                    )
                    _write_log_line(
                        log_file,
                        {
                            "type": "eval",
                            "step": global_step,
                            "tokens": tokens_seen,
                            "wall_clock": time.time(),
                            "val_loss": final_val_loss,
                            "val_perplexity": final_val_ppl,
                        },
                    )

                due_for_checkpoint = (
                    global_step % config.checkpoint_interval_steps == 0
                    or is_last_step
                    or self._stop_requested
                )
                if due_for_checkpoint:
                    saved = self._checkpoint(
                        model=model,
                        optimizer=optimizer,
                        config=config,
                        step=global_step,
                        total_steps=total_steps,
                        fingerprint=fingerprint,
                        log_file=log_file,
                    )
                    checkpoint_path = saved or checkpoint_path

                if self._stop_requested:
                    logger.warning(
                        "stopping at step %d/%d on request; checkpoint written",
                        global_step,
                        total_steps,
                    )
                    _write_log_line(
                        log_file,
                        {
                            "type": "interrupt",
                            "step": global_step,
                            "tokens": tokens_seen,
                            "wall_clock": time.time(),
                            "checkpoint": str(checkpoint_path) if checkpoint_path else None,
                        },
                    )
                    break

        return TrainResult(
            final_step=global_step,
            final_train_loss=final_train_loss,
            final_val_loss=final_val_loss,
            val_perplexity=final_val_ppl,
            log_path=self.log_path,
            checkpoint_path=checkpoint_path,
        )
