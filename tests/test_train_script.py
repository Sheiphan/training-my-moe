"""journal 041: `scripts/train.py` builds `TrainConfig` by matching
argparse destinations against dataclass field names.

The bug this replaced was quiet rather than loud: the hand-written
constructor passed 12 of `TrainConfig`'s 24 fields, so `--precision bf16`
would have been accepted, ignored, and the run would have trained in fp32
without saying so. The test below fails the moment a field goes
unreachable again.
"""
import argparse
import dataclasses

import pytest

from qwenmoe.config import TrainConfig
from scripts.train import _build_train_config, main  # noqa: F401  (import smoke)


def _parser_destinations() -> set[str]:
    """Every `dest` `scripts/train.py`'s parser defines, read from the
    parser itself rather than from a hand-maintained list."""
    import sys
    from unittest import mock

    import scripts.train as train_script

    captured = {}
    real_parse = argparse.ArgumentParser.parse_args

    def capture(self, *args, **kwargs):
        captured["parser"] = self
        raise SystemExit(0)  # stop before the script does any real work

    with mock.patch.object(argparse.ArgumentParser, "parse_args", capture):
        with mock.patch.object(sys, "argv", ["train.py", "--output-dir", "x", "--total-tokens", "1"]):
            with pytest.raises(SystemExit):
                train_script.main()

    return {action.dest for action in captured["parser"]._actions}


def test_every_train_config_field_has_a_flag_or_is_deliberately_omitted():
    """`TrainConfig` fields with no CLI flag silently take their default.
    If you add a field, either add a flag or add it to the exempt set
    here, so the choice is recorded rather than accidental."""
    fields = {f.name for f in dataclasses.fields(TrainConfig)}
    destinations = _parser_destinations()

    # `total_tokens` etc. are all flags; nothing is currently exempt.
    exempt: set[str] = set()
    missing = fields - destinations - exempt
    assert not missing, f"TrainConfig fields unreachable from the CLI: {sorted(missing)}"


def test_build_train_config_passes_the_non_default_flags_through():
    args = argparse.Namespace(
        batch_size=4,
        grad_accum_steps=8,
        seq_len=512,
        total_tokens=1_000,
        lr=6e-4,
        warmup_steps=2000,
        eval_interval_steps=500,
        max_eval_batches=50,
        checkpoint_interval_steps=2000,
        log_interval_steps=10,
        device="cpu",
        seed=99,
        document_masking=False,
        optimizer="muon",
        lr_schedule="wsd",
        min_lr_ratio=0.0,
        decay_fraction=0.15,
        precision="bf16",
        weight_decay=0.05,
        grad_clip_norm=0.5,
        z_loss_weight=1e-4,
        adam_eps=1e-9,
        betas=[0.9, 0.98],
        muon_lr=1e-3,
        muon_momentum=0.9,
        # extras that are NOT TrainConfig fields must be ignored, not crash
        output_dir="runs/x",
        keep_last=2,
        corpus="fineweb_edu",
    )
    config = _build_train_config(args)

    assert config.precision == "bf16"
    assert config.lr_schedule == "wsd"
    assert config.optimizer == "muon"
    assert config.weight_decay == 0.05
    assert config.grad_clip_norm == 0.5
    assert config.z_loss_weight == 1e-4
    assert config.min_lr_ratio == 0.0
    assert config.decay_fraction == 0.15
    assert config.muon_lr == 1e-3
    assert config.seed == 99


def test_betas_becomes_a_tuple():
    """argparse yields a list for nargs=2; `TrainConfig.__post_init__`
    indexes it and downstream code unpacks it as a pair."""
    args = argparse.Namespace(
        batch_size=1, seq_len=1, total_tokens=1, betas=[0.9, 0.99]
    )
    config = _build_train_config(args)
    assert config.betas == (0.9, 0.99)
    assert isinstance(config.betas, tuple)
