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

import contextlib
import itertools
import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from qwenmoe.config import ModelConfig, TrainConfig
from qwenmoe.model.transformer import QwenMoEModel
from qwenmoe.train.optim import build_optimizer

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


def select_device(requested: Literal["auto", "mps", "cpu"]) -> str:
    """Requirement 3.2: `mps` if available, else `cpu`; `requested="cpu"`
    always forces `cpu`. Warns (doesn't raise) if `mps` was explicitly
    requested but isn't available (design.md Error Handling: MPS
    unavailable falls back rather than crashing).
    """
    if requested == "cpu":
        return "cpu"
    if torch.backends.mps.is_available():
        return "mps"
    if requested == "mps":
        logger.warning("device='mps' requested but unavailable; falling back to cpu")
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
    """Yields batches forever, starting a fresh (re-shuffled, since the
    loader is built with `shuffle=True`) epoch each time it's exhausted --
    lets `Trainer` be driven purely by step count
    (`TrainConfig.total_tokens`), not by epoch count.
    """
    while True:
        yield from loader


def _collect_expert_counts(model: QwenMoEModel) -> list[list[int]]:
    """Per-layer, per-routed-expert dispatched-token counts from the most
    recent forward call (Requirement 8.2), read from the
    `MoELayer.last_expert_counts` cache each block's MoE layer populates.
    """
    return [block.moe.last_expert_counts.tolist() for block in model.blocks]


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

    z_loss: float | None = None
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
        z_loss = z_loss_term.item()

    mtp_loss: float | None = None
    mtp_losses: list[float] | None = None
    mtp_mean = extras.get("mtp_loss")
    if mtp_mean is not None:
        total_loss = total_loss + model.config.mtp_lambda * mtp_mean
        mtp_loss = mtp_mean.item()
        mtp_losses = [t.item() for t in extras["mtp_losses"]]

    (total_loss * loss_scale).backward()
    return {
        "lm_loss": loss.item(),
        "aux_loss": aux_loss.item() if aux_loss is not None else None,
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
    return metrics, float(grad_norm)


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
    """

    def _mean(key: str) -> float | None:
        values = [m[key] for m in metrics_list if m[key] is not None]
        return sum(values) / len(values) if values else None

    mtp_losses_lists = [m["mtp_losses"] for m in metrics_list if m["mtp_losses"] is not None]
    mtp_losses = (
        [sum(depth_losses) / len(depth_losses) for depth_losses in zip(*mtp_losses_lists)]
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


def _sum_expert_counts(per_microbatch: list[list[list[int]]]) -> list[list[int]]:
    """Elementwise-sums `_collect_expert_counts` readings taken after each
    micro-batch in an accumulation window, so the logged per-expert token
    counts reflect the *whole* effective batch rather than just the last
    micro-batch (`MoELayer.last_expert_counts` is overwritten by every
    forward call). With one micro-batch (`grad_accum_steps == 1`) this is
    the identity on that single reading.
    """
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
    return metrics, float(grad_norm), current_lr, device, model, expert_counts


def _evaluate(
    model: QwenMoEModel, val_loader: DataLoader, device: str, max_eval_batches: int
) -> tuple[float, float]:
    """Mean validation loss and perplexity `exp(loss)` over at most
    `max_eval_batches` of `val_loader` (Requirement 5.3) -- capped so
    evaluating doesn't require a full pass over a potentially huge
    validation split every `eval_interval_steps`.

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


def save_checkpoint(
    path: str | Path,
    model: QwenMoEModel,
    optimizer: torch.optim.Optimizer,
    model_config: ModelConfig,
    train_config: TrainConfig,
    step: int,
    tokenizer_meta: dict | None,
) -> None:
    """Model + optimizer + config + step (Requirement 5.4), per design.md's
    checkpoint format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_config": model_config,
            "train_config": train_config,
            "step": step,
            "tokenizer_meta": tokenizer_meta,
        },
        path,
    )


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
    `step_{N}.pt` file per checkpoint interval) and `logs/train.jsonl`
    (append-mode, so a resumed run's log history stays contiguous).
    """

    def __init__(self, output_dir: str | Path, tokenizer_meta: dict | None = None) -> None:
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.log_path = self.output_dir / "logs" / "train.jsonl"
        self.tokenizer_meta = tokenizer_meta

    def fit(
        self,
        model: QwenMoEModel,
        train_dataset: Dataset,
        val_dataset: Dataset,
        config: TrainConfig,
        resume_from: str | Path | None = None,
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

        model = model.to(device)
        optimizer = build_optimizer(model, config)

        global_step = 0
        if resume_from is not None:
            checkpoint = load_checkpoint(resume_from)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            _move_optimizer_state_to_device(optimizer, device)
            global_step = checkpoint["step"]

        # Gradient accumulation (moe-modernization follow-up, journal 038):
        # each optimizer step consumes `grad_accum_steps` micro-batches, so
        # the *effective* batch that sets step count is `batch_size *
        # grad_accum_steps`, not `batch_size` alone. `grad_accum_steps == 1`
        # (the default) reduces this to the pre-existing formula exactly.
        effective_batch_size = config.batch_size * config.grad_accum_steps
        total_steps = max(1, config.total_tokens // (effective_batch_size * config.seq_len))

        train_loader = DataLoader(
            train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=config.batch_size, shuffle=False, drop_last=True
        )
        train_iter = _infinite_batches(train_loader)

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        final_train_loss = math.nan
        final_val_loss = math.nan
        final_val_ppl = math.nan
        checkpoint_path: Path | None = None

        with self.log_path.open("a", encoding="utf-8") as log_file:
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
                metrics, grad_norm, current_lr, device, model, expert_counts = _run_step(
                    model,
                    optimizer,
                    micro_batches,
                    device,
                    global_step,
                    total_steps,
                    config,
                )
                elapsed = time.perf_counter() - step_start

                global_step += 1
                train_loss = metrics["lm_loss"]
                aux_loss_value = metrics["aux_loss"]
                final_train_loss = train_loss
                is_last_step = global_step == total_steps

                if global_step % config.log_interval_steps == 0 or is_last_step:
                    logger.info(
                        "step %d/%d | loss %.4f | aux_loss %s | z_loss %s | mtp_loss %s "
                        "| lr %.2e | elapsed %.3fs",
                        global_step,
                        total_steps,
                        train_loss,
                        "n/a" if aux_loss_value is None else f"{aux_loss_value:.4f}",
                        "n/a" if metrics["z_loss"] is None else f"{metrics['z_loss']:.4f}",
                        "n/a" if metrics["mtp_loss"] is None else f"{metrics['mtp_loss']:.4f}",
                        current_lr,
                        elapsed,
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
                            "grad_norm": grad_norm,
                            "elapsed_s": elapsed,
                            # Summed across the accumulation window by
                            # `_run_step` (`_sum_expert_counts`), not just the
                            # last micro-batch's reading.
                            "expert_token_counts": expert_counts,
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
                            "val_loss": final_val_loss,
                            "val_perplexity": final_val_ppl,
                        },
                    )

                if global_step % config.checkpoint_interval_steps == 0 or is_last_step:
                    checkpoint_path = self.checkpoint_dir / f"step_{global_step}.pt"
                    save_checkpoint(
                        checkpoint_path,
                        model,
                        optimizer,
                        model.config,
                        config,
                        global_step,
                        self.tokenizer_meta,
                    )

        return TrainResult(
            final_step=global_step,
            final_train_loss=final_train_loss,
            final_val_loss=final_val_loss,
            val_perplexity=final_val_ppl,
            log_path=self.log_path,
            checkpoint_path=checkpoint_path,
        )
