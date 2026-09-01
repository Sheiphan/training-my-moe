"""Research-style architecture comparison framework (Requirement 12): trains
multiple named `ModelConfig` variants on an identical, fixed token slice
under an identical token budget, so any difference in final metrics
reflects the architectural change being studied, not different training
data or compute (Requirements 11.5, 12.6).
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from qwenmoe.config import DataConfig, ModelConfig, TrainConfig
from qwenmoe.data.dataset import TokenDataset, make_fixed_slice, train_val_split
from qwenmoe.data.download import prepared_data_dir
from qwenmoe.model.transformer import QwenMoEModel
from qwenmoe.tokenizer.base import build_tokenizer
from qwenmoe.train.scaling import calibrate_throughput
from qwenmoe.train.trainer import Trainer, select_device


@dataclass
class VariantResult:
    """One compared variant's outcome (Requirement 12.7)."""

    config: ModelConfig
    final_train_loss: float
    final_val_loss: float
    val_perplexity: float
    tokens_per_sec: float
    log_path: Path


@dataclass
class ComparisonReport:
    """`run_comparison`'s return value: every variant's `VariantResult`,
    keyed by the same name used in the `variants` dict passed in."""

    variants: dict[str, VariantResult]


def _load_fixed_slice_datasets(
    data_config: DataConfig, seq_len: int
) -> tuple[TokenDataset, TokenDataset, dict]:
    """Loads the corpus `scripts/prepare_data.py` already prepared for
    `data_config`, cuts the one `make_fixed_slice` (Requirement 12.6) every
    variant will train on, and returns `(train_dataset, val_dataset,
    tokenizer_meta)`. Raises eagerly (not mid-run) if the ablation-mode
    slice size wasn't configured, or if prep hasn't been run yet.
    """
    if data_config.fixed_slice_tokens is None:
        raise ValueError(
            "run_comparison requires DataConfig.fixed_slice_tokens to be set, so "
            "every variant trains on an identical slice (Requirement 12.6)"
        )

    data_dir = prepared_data_dir(data_config)
    tokens_path = data_dir / "tokens.pt"
    tokenizer_path = data_dir / "tokenizer.json"
    if not tokens_path.exists() or not tokenizer_path.exists():
        raise FileNotFoundError(
            f"no prepared data found under {data_dir} -- run "
            f"'uv run python scripts/prepare_data.py --corpus {data_config.corpus} "
            f"--tokenizer-type {data_config.tokenizer_type}' first"
        )

    all_tokens = torch.load(tokens_path, weights_only=True)
    tokenizer = build_tokenizer(data_config)
    tokenizer.load(tokenizer_path)
    tokenizer_meta = {
        "tokenizer_type": data_config.tokenizer_type,
        "tokenizer_path": str(tokenizer_path),
        "vocab_size": tokenizer.vocab_size,
    }

    fixed_slice = make_fixed_slice(
        all_tokens, data_config.fixed_slice_tokens, data_config.fixed_slice_seed
    )
    train_tokens, val_tokens = train_val_split(fixed_slice)
    train_dataset = TokenDataset(train_tokens, seq_len)
    val_dataset = TokenDataset(val_tokens, seq_len)
    return train_dataset, val_dataset, tokenizer_meta


def run_comparison(
    base_config: ModelConfig,
    variants: dict[str, dict],
    data_config: DataConfig,
    train_config: TrainConfig,
    output_dir: str | Path,
) -> ComparisonReport:
    """Trains one `QwenMoEModel` per `variants` entry -- `name ->
    dataclasses.replace(base_config, **overrides)` -- on the identical
    fixed token slice and the identical `train_config` token budget
    (Requirements 11.5, 12.6), writing each variant's checkpoints/logs
    under `output_dir/<name>/`.

    `overrides` may be a sparse dict of just the fields that differ from
    `base_config` (an ablation-style delta, e.g.
    `{"positional_scheme": "nope"}`), or a full
    `dataclasses.asdict(some_preset)` (the CLI convenience `scripts/
    run_comparison.py` uses to compare independent named presets) --
    either way every field named in `overrides` wins, and `vocab_size`
    always comes from the prepared tokenizer, never the caller, so a
    preset's placeholder value can never silently mismatch the real one.
    """
    train_dataset, val_dataset, tokenizer_meta = _load_fixed_slice_datasets(
        data_config, train_config.seq_len
    )

    output_dir = Path(output_dir)
    device = select_device(train_config.device)
    sample_loader = DataLoader(
        train_dataset, batch_size=train_config.batch_size, shuffle=False, drop_last=True
    )
    sample_batch = next(iter(sample_loader))

    results: dict[str, VariantResult] = {}
    for name, overrides in variants.items():
        variant_config = dataclasses.replace(
            base_config, **{**overrides, "vocab_size": tokenizer_meta["vocab_size"]}
        )

        # Same seed before each variant's construction: identical initial
        # weights wherever two variants' shapes actually match (e.g. a
        # positional-scheme swap adds no parameters), isolating the
        # architectural change from random-init luck as much as possible.
        torch.manual_seed(train_config.seed)
        model = QwenMoEModel(variant_config)

        trainer = Trainer(output_dir / name, tokenizer_meta=tokenizer_meta)
        train_result = trainer.fit(model, train_dataset, val_dataset, train_config)
        tokens_per_sec = calibrate_throughput(model, sample_batch, device, n_steps=5)

        results[name] = VariantResult(
            config=variant_config,
            final_train_loss=train_result.final_train_loss,
            final_val_loss=train_result.final_val_loss,
            val_perplexity=train_result.val_perplexity,
            tokens_per_sec=tokens_per_sec,
            log_path=train_result.log_path,
        )

    return ComparisonReport(variants=results)
