"""Research-style architecture comparison framework (Requirement 12): trains
multiple named `ModelConfig` variants on an identical, fixed token slice
under an identical token budget, so any difference in final metrics
reflects the architectural change being studied, not different training
data or compute (Requirements 11.5, 12.6).

moe-modernization task 16 (Requirement 12.1, 12.2, 12.4): each variant can
be trained once *per seed* in a `seeds` tuple, so its reported metric is a
`mean ± std` over seeds rather than a single-run point estimate, and a
variant whose delta from the baseline is smaller than the pooled
seed-to-seed noise is flagged `within_noise_of_baseline`. A `param_matched`
mode adjusts each variant's `d_ff` (bisection) so its total parameter count
matches the baseline within 2%, for comparisons where a raw compute-matched
run would confound "better architecture" with "more parameters".
"""
from __future__ import annotations

import dataclasses
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from qwenmoe.config import DataConfig, ModelConfig, TrainConfig
from qwenmoe.data.dataset import TokenDataset, make_fixed_slice, train_val_split
from qwenmoe.data.download import prepared_data_dir
from qwenmoe.model.transformer import QwenMoEModel
from qwenmoe.tokenizer.base import build_tokenizer
from qwenmoe.train.scaling import calibrate_throughput, count_params
from qwenmoe.train.trainer import Trainer, select_device

_PARAM_MATCH_TOLERANCE = 0.02  # Requirement 12.4: total params within 2% of baseline


@dataclass
class VariantResult:
    """One compared variant's outcome (Requirement 12.7; moe-modernization
    Requirement 12.1, 12.2).

    The scalar `final_train_loss` / `final_val_loss` / `val_perplexity`
    fields are the **mean across seeds** (with a single seed -- the default
    -- they are exactly that one run's numbers, so every pre-task reader is
    unchanged). `per_seed_*`, `*_mean`, `*_std` expose the full multi-seed
    picture; `within_noise_of_baseline` is set once every variant has run
    (`None` for the baseline itself and whenever there is only one seed, so
    no noise band can be estimated).
    """

    config: ModelConfig
    final_train_loss: float
    final_val_loss: float
    val_perplexity: float
    tokens_per_sec: float
    log_path: Path

    seeds: tuple[int, ...] = ()
    per_seed_val_loss: list[float] = field(default_factory=list)
    per_seed_val_perplexity: list[float] = field(default_factory=list)
    val_loss_mean: float = math.nan
    val_loss_std: float = 0.0
    val_perplexity_mean: float = math.nan
    val_perplexity_std: float = 0.0
    log_paths: list[Path] = field(default_factory=list)
    comparison_mode: str = "compute_matched"  # or "param_matched"
    within_noise_of_baseline: bool | None = None


@dataclass
class ComparisonReport:
    """`run_comparison`'s return value: every variant's `VariantResult`,
    keyed by the same name used in the `variants` dict passed in, plus the
    name treated as the baseline and the seed list every variant used."""

    variants: dict[str, VariantResult]
    baseline_name: str
    seeds: tuple[int, ...]

    def summary_table(self) -> str:
        """A `mean ± std` comparison table (moe-modernization Requirement
        12.2): each non-baseline variant's val-loss delta is annotated
        `within noise` when it is smaller than the pooled seed-to-seed
        standard deviation, `differs` otherwise, `n/a (1 seed)` when no
        noise band could be estimated.
        """
        header = (
            f"{'variant':<20}{'val_loss (mean±std)':>24}{'val_ppl (mean±std)':>24}"
            f"{'tokens/sec':>13}{'mode':>16}  note"
        )
        lines = [header, "-" * len(header)]
        for name, r in self.variants.items():
            if name == self.baseline_name:
                note = "baseline"
            elif r.within_noise_of_baseline is None:
                note = "n/a (1 seed)"
            elif r.within_noise_of_baseline:
                note = "within noise"
            else:
                note = "differs"
            lines.append(
                f"{name:<20}"
                f"{f'{r.val_loss_mean:.4f} ± {r.val_loss_std:.4f}':>24}"
                f"{f'{r.val_perplexity_mean:.3f} ± {r.val_perplexity_std:.3f}':>24}"
                f"{r.tokens_per_sec:>13.1f}{r.comparison_mode:>16}  {note}"
            )
        return "\n".join(lines)


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


def _total_params(config: ModelConfig) -> int:
    """Total parameter count for a config, via a throwaway model so
    `count_params` (Requirement 11.1) stays the single source of truth."""
    return count_params(QwenMoEModel(config)).total


def match_d_ff_to_param_budget(
    variant_config: ModelConfig,
    target_total_params: int,
    tolerance: float = _PARAM_MATCH_TOLERANCE,
) -> int:
    """Smallest-error `d_ff` such that `variant_config`'s total parameter
    count is within `tolerance` of `target_total_params` (Requirement
    12.4). Total params are monotone increasing in `d_ff` (every expert's
    matrices are `d_model x d_ff`), so a plain integer bisection converges;
    the two `d_ff` straddling the target are compared and the closer one
    returned. Raises if even `d_ff = 1` already exceeds the target (the
    variant is structurally larger than the baseline and can't be shrunk
    into range by `d_ff` alone).
    """
    low_total = _total_params(dataclasses.replace(variant_config, d_ff=1))
    if low_total > target_total_params * (1 + tolerance):
        raise ValueError(
            f"variant is larger than the baseline even at d_ff=1 "
            f"({low_total} > {target_total_params}); cannot param-match via d_ff"
        )

    lo, hi = 1, max(2, variant_config.d_ff)
    while _total_params(dataclasses.replace(variant_config, d_ff=hi)) < target_total_params:
        hi *= 2
    while lo < hi:
        mid = (lo + hi) // 2
        if _total_params(dataclasses.replace(variant_config, d_ff=mid)) < target_total_params:
            lo = mid + 1
        else:
            hi = mid
    # `lo` is the smallest d_ff whose total >= target; its predecessor is the
    # largest whose total < target -- return whichever is closer.
    candidates = [lo] + ([lo - 1] if lo > 1 else [])
    return min(
        candidates,
        key=lambda d: abs(_total_params(dataclasses.replace(variant_config, d_ff=d)) - target_total_params),
    )


def run_comparison(
    base_config: ModelConfig,
    variants: dict[str, dict],
    data_config: DataConfig,
    train_config: TrainConfig,
    output_dir: str | Path,
    *,
    seeds: tuple[int, ...] | list[int] | None = None,
    param_matched: bool = False,
    baseline_name: str | None = None,
    train_overrides: dict[str, dict] | None = None,
) -> ComparisonReport:
    """Trains one `QwenMoEModel` per `variants` entry -- `name ->
    dataclasses.replace(base_config, **overrides)` -- on the identical
    fixed token slice and the identical `train_config` token budget
    (Requirements 11.5, 12.6), writing each variant's checkpoints/logs
    under `output_dir/<name>/` (or `output_dir/<name>/seed_<n>/` with more
    than one seed).

    `overrides` may be a sparse dict of just the fields that differ from
    `base_config` (an ablation-style delta, e.g.
    `{"positional_scheme": "nope"}`), or a full
    `dataclasses.asdict(some_preset)` (the CLI convenience `scripts/
    run_comparison.py` uses to compare independent named presets) --
    either way every field named in `overrides` wins, and `vocab_size`
    always comes from the prepared tokenizer, never the caller.

    `seeds` (moe-modernization Requirement 12.1, default `(train_config.
    seed,)` -- exactly today's single run): every variant is trained once
    per seed on the *same* fixed slice, and its `VariantResult` reports the
    mean and standard deviation of final val loss / perplexity across
    seeds. With >= 2 seeds, `within_noise_of_baseline` is set on each
    non-baseline variant (`abs(mean - baseline_mean) < hypot(std,
    baseline_std)`).

    `param_matched` (Requirement 12.4): before training, each non-baseline
    variant's `d_ff` is bisected so its total parameter count matches the
    baseline's within 2%. `comparison_mode` on each result records which
    mode was used. `baseline_name` defaults to the first key of `variants`.

    `train_overrides` (moe-modernization Requirement 12.5): `{name: {field:
    value}}` `TrainConfig` deltas per variant, so axes that live on
    `TrainConfig` rather than `ModelConfig` -- `optimizer="muon"`,
    `lr_schedule="wsd"` -- can be compared too. The shared token budget
    (`total_tokens`, `batch_size`, `seq_len`) must not be overridden or the
    comparison stops being compute-matched (Requirement 12.6); this is the
    caller's responsibility.
    """
    seed_tuple: tuple[int, ...] = tuple(seeds) if seeds is not None else (train_config.seed,)
    if not seed_tuple:
        raise ValueError("seeds must be non-empty")
    baseline = baseline_name if baseline_name is not None else next(iter(variants))
    train_overrides = train_overrides or {}

    train_dataset, val_dataset, tokenizer_meta = _load_fixed_slice_datasets(
        data_config, train_config.seq_len
    )

    output_dir = Path(output_dir)
    device = select_device(train_config.device)
    sample_loader = DataLoader(
        train_dataset, batch_size=train_config.batch_size, shuffle=False, drop_last=True
    )
    sample_batch = next(iter(sample_loader))
    vocab_size = tokenizer_meta["vocab_size"]

    def _resolve(overrides: dict) -> ModelConfig:
        return dataclasses.replace(base_config, **{**overrides, "vocab_size": vocab_size})

    baseline_total = _total_params(_resolve(variants[baseline])) if param_matched else None

    results: dict[str, VariantResult] = {}
    for name, overrides in variants.items():
        variant_config = _resolve(overrides)
        comparison_mode = "compute_matched"
        if param_matched and name != baseline and baseline_total is not None:
            matched = match_d_ff_to_param_budget(variant_config, baseline_total)
            variant_config = dataclasses.replace(variant_config, d_ff=matched)
            comparison_mode = "param_matched"

        per_seed_val_loss: list[float] = []
        per_seed_val_ppl: list[float] = []
        per_seed_train_loss: list[float] = []
        seed_log_paths: list[Path] = []
        model = None
        for seed in seed_tuple:
            run_dir = output_dir / name if len(seed_tuple) == 1 else output_dir / name / f"seed_{seed}"

            # Same seed before construction: identical initial weights
            # wherever two variants' shapes match, isolating the
            # architectural change from random-init luck.
            torch.manual_seed(seed)
            model = QwenMoEModel(variant_config)

            trainer = Trainer(run_dir, tokenizer_meta=tokenizer_meta)
            seed_train_config = dataclasses.replace(
                train_config, **train_overrides.get(name, {}), seed=seed
            )
            train_result = trainer.fit(
                model, train_dataset, val_dataset, seed_train_config
            )
            per_seed_val_loss.append(train_result.final_val_loss)
            per_seed_val_ppl.append(train_result.val_perplexity)
            per_seed_train_loss.append(train_result.final_train_loss)
            seed_log_paths.append(train_result.log_path)

        tokens_per_sec = calibrate_throughput(model, sample_batch, device, n_steps=5)

        val_loss_std = statistics.stdev(per_seed_val_loss) if len(seed_tuple) > 1 else 0.0
        val_ppl_std = statistics.stdev(per_seed_val_ppl) if len(seed_tuple) > 1 else 0.0
        val_loss_mean = statistics.fmean(per_seed_val_loss)
        val_ppl_mean = statistics.fmean(per_seed_val_ppl)

        results[name] = VariantResult(
            config=variant_config,
            final_train_loss=statistics.fmean(per_seed_train_loss),
            final_val_loss=val_loss_mean,
            val_perplexity=val_ppl_mean,
            tokens_per_sec=tokens_per_sec,
            log_path=seed_log_paths[0],
            seeds=seed_tuple,
            per_seed_val_loss=per_seed_val_loss,
            per_seed_val_perplexity=per_seed_val_ppl,
            val_loss_mean=val_loss_mean,
            val_loss_std=val_loss_std,
            val_perplexity_mean=val_ppl_mean,
            val_perplexity_std=val_ppl_std,
            log_paths=seed_log_paths,
            comparison_mode=comparison_mode,
        )

    if len(seed_tuple) >= 2 and baseline in results:
        base_result = results[baseline]
        for name, result in results.items():
            if name == baseline:
                continue
            noise_band = math.hypot(result.val_loss_std, base_result.val_loss_std)
            result.within_noise_of_baseline = (
                abs(result.val_loss_mean - base_result.val_loss_mean) < noise_band
            )

    return ComparisonReport(variants=results, baseline_name=baseline, seeds=seed_tuple)
