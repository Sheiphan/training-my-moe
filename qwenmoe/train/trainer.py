"""Training loop (Requirement 5): AdamW with a warmup+cosine learning-rate
schedule, device auto-selection (`mps` else `cpu`), periodic console +
JSON-lines logging of per-step metrics and per-expert token counts (consumed
later by the training-observability scenes, task 14), periodic validation
loss/perplexity, and checkpoint save/resume.
"""
from __future__ import annotations

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


def lr_multiplier(step: int, warmup_steps: int, total_steps: int) -> float:
    """Linear warmup from `1/warmup_steps` to `1.0` over `warmup_steps`,
    then cosine decay from `1.0` towards `0.0` by `total_steps`
    (Requirement 5.1). `step` is 0-indexed (the step about to run); a
    `warmup_steps` of 0 skips straight to the cosine phase.
    """
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    if total_steps <= warmup_steps:
        return 1.0
    progress = min((step - warmup_steps) / (total_steps - warmup_steps), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def _set_lr(
    optimizer: torch.optim.Optimizer, base_lr: float, step: int, warmup_steps: int, total_steps: int
) -> float:
    lr = base_lr * lr_multiplier(step, warmup_steps, total_steps)
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


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


def _forward_backward_step(
    model: QwenMoEModel, optimizer: torch.optim.Optimizer, x: torch.Tensor, y: torch.Tensor
) -> tuple[float, float | None]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits, extras = model(x)
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
    aux_loss = extras.get("aux_loss")
    total_loss = loss if aux_loss is None else loss + aux_loss
    total_loss.backward()
    optimizer.step()
    return loss.item(), (aux_loss.item() if aux_loss is not None else None)


def _run_step(
    model: QwenMoEModel,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    y: torch.Tensor,
    device: str,
    base_lr: float,
    step: int,
    total_steps: int,
    warmup_steps: int,
) -> tuple[float, float | None, float, str, QwenMoEModel]:
    """Runs one training step, falling back to `cpu` if a real `RuntimeError`
    hits mid-step on `mps` (design.md Error Handling: "MPS unavailable/errors
    during a step -> fall back to cpu with a logged warning, never a hard
    crash", Requirement 3.2). Returns the (possibly updated) `device`/`model`
    so the caller keeps using them for subsequent steps.
    """
    current_lr = _set_lr(optimizer, base_lr, step, warmup_steps, total_steps)
    try:
        train_loss, aux_loss_value = _forward_backward_step(model, optimizer, x.to(device), y.to(device))
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
        train_loss, aux_loss_value = _forward_backward_step(model, optimizer, x.to(device), y.to(device))
    return train_loss, aux_loss_value, current_lr, device, model


def _evaluate(
    model: QwenMoEModel, val_loader: DataLoader, device: str, max_eval_batches: int
) -> tuple[float, float]:
    """Mean validation loss and perplexity `exp(loss)` over at most
    `max_eval_batches` of `val_loader` (Requirement 5.3) -- capped so
    evaluating doesn't require a full pass over a potentially huge
    validation split every `eval_interval_steps`.
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for x, y in itertools.islice(val_loader, max_eval_batches):
            logits, _ = model(x.to(device))
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

        device = select_device(config.device)
        torch.manual_seed(config.seed)

        model = model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

        global_step = 0
        if resume_from is not None:
            checkpoint = load_checkpoint(resume_from)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            _move_optimizer_state_to_device(optimizer, device)
            global_step = checkpoint["step"]

        total_steps = max(1, config.total_tokens // (config.batch_size * config.seq_len))

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
                x, y = next(train_iter)

                step_start = time.perf_counter()
                train_loss, aux_loss_value, current_lr, device, model = _run_step(
                    model,
                    optimizer,
                    x,
                    y,
                    device,
                    config.lr,
                    global_step,
                    total_steps,
                    config.warmup_steps,
                )
                elapsed = time.perf_counter() - step_start

                global_step += 1
                final_train_loss = train_loss
                is_last_step = global_step == total_steps

                if global_step % config.log_interval_steps == 0 or is_last_step:
                    logger.info(
                        "step %d/%d | loss %.4f | aux_loss %s | lr %.2e | elapsed %.3fs",
                        global_step,
                        total_steps,
                        train_loss,
                        "n/a" if aux_loss_value is None else f"{aux_loss_value:.4f}",
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
                            "lr": current_lr,
                            "elapsed_s": elapsed,
                            "expert_token_counts": _collect_expert_counts(model),
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
