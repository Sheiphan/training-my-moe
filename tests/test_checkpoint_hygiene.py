"""journal 041: the machinery that makes a multi-day run survivable --
atomic checkpoint writes, tiered retention, a `latest.pt` pointer, a
resume guard against silently re-aligned data, and SIGINT that finishes
the step and checkpoints instead of leaving a truncated 4GB file.

The scenarios here are the ones that only ever happen at hour 40 of an
unattended run, which is exactly why they need to be pinned at second 1
of a test suite.
"""
import json
import os
import signal
import threading
import time

import pytest
import torch

from qwenmoe.config import ModelConfig, TrainConfig
from qwenmoe.data.dataset import TokenDataset, train_val_split
from qwenmoe.model.transformer import QwenMoEModel
from qwenmoe.train.optim import build_optimizer
from qwenmoe.train.trainer import (
    Trainer,
    data_fingerprint,
    estimate_checkpoint_bytes,
    load_checkpoint,
    prune_checkpoints,
    save_checkpoint,
    update_latest_pointer,
)


def _tiny_model():
    return QwenMoEModel(
        ModelConfig(
            vocab_size=12,
            d_model=16,
            n_layers=2,
            n_heads=4,
            n_kv_heads=2,
            d_ff=32,
            n_routed_experts=4,
            n_shared_experts=1,
            top_k=2,
            max_seq_len=32,
        )
    )


def _datasets(seq_len=16, n_repeats=300):
    tokens = list(range(10)) * n_repeats
    train_tokens, val_tokens = train_val_split(tokens, val_fraction=0.1)
    return TokenDataset(train_tokens, seq_len), TokenDataset(val_tokens, seq_len)


def _config(**overrides):
    defaults = dict(
        batch_size=4,
        seq_len=16,
        total_tokens=4 * 16 * 10,
        device="cpu",
        eval_interval_steps=100,
        log_interval_steps=1,
        checkpoint_interval_steps=2,
        warmup_steps=1,
        max_eval_batches=2,
    )
    defaults.update(overrides)
    return TrainConfig(**defaults)


def _step_once(model, optimizer):
    """One real optimizer step, so AdamW actually allocates the two moment
    buffers that make a full checkpoint 3x a weights-only one."""
    logits, _ = model(torch.zeros(2, 8, dtype=torch.long))
    logits.square().mean().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def _read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


# -- save_checkpoint: atomicity and weights-only ----------------------------


def test_save_checkpoint_leaves_no_temp_file(tmp_path):
    model = _tiny_model()
    optimizer = build_optimizer(model, _config())
    path = tmp_path / "step_1.pt"
    save_checkpoint(path, model, optimizer, model.config, _config(), 1, None)

    assert path.exists()
    assert list(tmp_path.glob("*.tmp")) == []


def test_failed_save_leaves_nothing_at_the_target_path(tmp_path, monkeypatch):
    """The failure this guards: `torch.save` dying partway (a full disk, a
    signal) must not leave a truncated file exactly where `--resume-from`
    will look."""
    model = _tiny_model()
    optimizer = build_optimizer(model, _config())
    path = tmp_path / "step_1.pt"

    def boom(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(torch, "save", boom)
    with pytest.raises(RuntimeError):
        save_checkpoint(path, model, optimizer, model.config, _config(), 1, None)

    assert not path.exists()
    assert list(tmp_path.glob("*.tmp")) == []


def test_weights_only_checkpoint_omits_optimizer_state(tmp_path):
    model = _tiny_model()
    path = tmp_path / "milestone_1.pt"
    save_checkpoint(path, model, None, model.config, _config(), 1, None)

    checkpoint = load_checkpoint(path)
    assert checkpoint["optimizer_state_dict"] is None
    assert checkpoint["model_state_dict"]  # still fully usable for generation


def test_weights_only_checkpoint_is_much_smaller(tmp_path):
    model = _tiny_model()
    config = _config()
    optimizer = build_optimizer(model, config)
    _step_once(model, optimizer)

    full = tmp_path / "full.pt"
    light = tmp_path / "light.pt"
    save_checkpoint(full, model, optimizer, model.config, config, 1, None)
    save_checkpoint(light, model, None, model.config, config, 1, None)

    assert light.stat().st_size < full.stat().st_size / 2


def test_estimate_checkpoint_bytes_tracks_the_real_file(tmp_path):
    """Pins the `12 bytes x param` rule the disk budget is planned against.

    It is deliberately an *upper* bound, which is what a pre-flight disk
    check wants. Two things make a real file come in slightly under:
    routing is dropless, so an expert that received no tokens gets no
    gradient and AdamW never allocates its two moment buffers; and tied
    embeddings share one tensor between the input table and the output
    head. Both effects shrink with scale -- at 300M parameters with every
    expert hit, the rule was accurate to 445KB in 4.01GB on a real
    `runs/scale_300m` checkpoint.
    """
    model = _tiny_model()
    config = _config()
    optimizer = build_optimizer(model, config)
    _step_once(model, optimizer)

    path = tmp_path / "full.pt"
    save_checkpoint(path, model, optimizer, model.config, config, 1, None)
    estimated = estimate_checkpoint_bytes(model, with_optimizer=True)
    actual = path.stat().st_size

    assert actual <= estimated + 100_000, "estimate must not under-count the real file"
    assert actual >= estimated * 0.5, "estimate must still be the right order"


# -- retention ---------------------------------------------------------------


def _touch_checkpoints(tmp_path, steps):
    for step in steps:
        (tmp_path / f"step_{step}.pt").write_bytes(b"x")
    return tmp_path


def test_prune_keeps_only_the_most_recent(tmp_path):
    _touch_checkpoints(tmp_path, [100, 200, 300, 400])
    removed = prune_checkpoints(tmp_path, keep_last=2, keep_every=0, protect=set())

    remaining = sorted(p.name for p in tmp_path.glob("step_*.pt"))
    assert remaining == ["step_300.pt", "step_400.pt"]
    assert len(removed) == 2


def test_prune_keeps_milestones(tmp_path):
    _touch_checkpoints(tmp_path, [100, 200, 300, 400, 500])
    prune_checkpoints(tmp_path, keep_last=1, keep_every=200, protect=set())

    remaining = sorted(int(p.stem.removeprefix("step_")) for p in tmp_path.glob("step_*.pt"))
    assert remaining == [200, 400, 500]


def test_prune_never_deletes_a_protected_path(tmp_path):
    _touch_checkpoints(tmp_path, [100, 200, 300])
    protected = tmp_path / "step_100.pt"
    prune_checkpoints(tmp_path, keep_last=1, keep_every=0, protect={protected})

    assert protected.exists()


def test_prune_orders_numerically_not_lexicographically(tmp_path):
    """step_9 vs step_10: sorting these as strings keeps the wrong one."""
    _touch_checkpoints(tmp_path, [9, 10, 11])
    prune_checkpoints(tmp_path, keep_last=1, keep_every=0, protect=set())

    remaining = [p.name for p in tmp_path.glob("step_*.pt")]
    assert remaining == ["step_11.pt"]


def test_prune_ignores_files_it_does_not_own(tmp_path):
    _touch_checkpoints(tmp_path, [100, 200])
    stranger = tmp_path / "step_notanumber.pt"
    stranger.write_bytes(b"x")
    other = tmp_path / "milestone_100.pt"
    other.write_bytes(b"x")

    prune_checkpoints(tmp_path, keep_last=1, keep_every=0, protect=set())
    assert stranger.exists() and other.exists()


# -- latest.pt ---------------------------------------------------------------


def test_latest_pointer_resolves_to_the_checkpoint(tmp_path):
    target = tmp_path / "step_500.pt"
    target.write_bytes(b"payload")
    update_latest_pointer(tmp_path, target)

    link = tmp_path / "latest.pt"
    assert link.read_bytes() == b"payload"
    assert os.readlink(link) == "step_500.pt"  # relative, so runs/ stays movable


def test_latest_pointer_is_replaced_not_stacked(tmp_path):
    for step in (100, 200):
        target = tmp_path / f"step_{step}.pt"
        target.write_bytes(str(step).encode())
        update_latest_pointer(tmp_path, target)

    assert (tmp_path / "latest.pt").read_bytes() == b"200"
    assert list(tmp_path.glob("*.tmp")) == []


# -- the resume guard --------------------------------------------------------


def test_fingerprint_excludes_total_tokens():
    """Extending a run and sizing a WSD cooldown both change
    `total_tokens`; refusing that would block the intended workflow."""
    a = data_fingerprint(_config(total_tokens=1000), 500)
    b = data_fingerprint(_config(total_tokens=9999), 500)
    assert a == b
    assert "total_tokens" not in a


def test_fingerprint_catches_a_changed_batch_size():
    a = data_fingerprint(_config(batch_size=4), 500)
    b = data_fingerprint(_config(batch_size=8), 500)
    assert a != b


def test_resume_refuses_a_changed_batch_size(tmp_path):
    train_ds, val_ds = _datasets()
    trainer = Trainer(tmp_path)
    trainer.fit(_tiny_model(), train_ds, val_ds, _config(), None)
    checkpoint = tmp_path / "checkpoints" / "latest.pt"

    with pytest.raises(ValueError, match="re-align the data stream"):
        Trainer(tmp_path).fit(
            _tiny_model(), train_ds, val_ds, _config(batch_size=2), checkpoint
        )


def test_allow_schedule_change_permits_it(tmp_path):
    train_ds, val_ds = _datasets()
    Trainer(tmp_path).fit(_tiny_model(), train_ds, val_ds, _config(), None)
    checkpoint = tmp_path / "checkpoints" / "latest.pt"

    result = Trainer(tmp_path).fit(
        _tiny_model(),
        train_ds,
        val_ds,
        _config(batch_size=2, total_tokens=2 * 16 * 20),
        checkpoint,
        allow_schedule_change=True,
    )
    assert result.final_step > 0


def test_resume_from_a_weights_only_checkpoint_is_refused(tmp_path):
    train_ds, val_ds = _datasets()
    model = _tiny_model()
    path = tmp_path / "milestone_1.pt"
    save_checkpoint(path, model, None, model.config, _config(), 1, None)

    with pytest.raises(ValueError, match="weights-only"):
        Trainer(tmp_path).fit(_tiny_model(), train_ds, val_ds, _config(), path)


def test_resume_continues_the_data_stream(tmp_path):
    """The point of `start_step=global_step`: the resumed run must not
    re-serve the windows the first run already trained on."""
    train_ds, _ = _datasets()
    config = _config()
    from qwenmoe.data.sampler import StepOffsetSampler

    first = StepOffsetSampler(
        len(train_ds), config.batch_size, config.grad_accum_steps, config.seed, 0
    )
    resumed = StepOffsetSampler(
        len(train_ds), config.batch_size, config.grad_accum_steps, config.seed, 3
    )
    assert resumed.offsets_for_step(3) == first.offsets_for_step(3)
    assert resumed.offsets_for_step(3) != first.offsets_for_step(0)


# -- the run_start record ----------------------------------------------------


def test_run_start_record_is_written_and_self_describing(tmp_path):
    train_ds, val_ds = _datasets()
    Trainer(tmp_path, tokenizer_meta={"tokenizer_type": "char"}).fit(
        _tiny_model(), train_ds, val_ds, _config(), None
    )

    records = _read_jsonl(tmp_path / "logs" / "train.jsonl")
    start = records[0]
    assert start["type"] == "run_start"
    for key in (
        "params_total",
        "params_active",
        "chinchilla_budget_tokens",
        "epochs_over_train_split",
        "total_steps",
        "tokens_per_step",
        "torch_version",
        "model_config",
        "train_config",
        "data_fingerprint",
    ):
        assert key in start, f"run_start is missing {key}"
    assert start["chinchilla_budget_tokens"] == 20 * start["params_active"]


def test_step_records_carry_throughput_and_memory(tmp_path):
    train_ds, val_ds = _datasets()
    Trainer(tmp_path).fit(_tiny_model(), train_ds, val_ds, _config(), None)

    steps = [r for r in _read_jsonl(tmp_path / "logs" / "train.jsonl") if r["type"] == "step"]
    assert steps
    for record in steps:
        assert record["tokens"] == record["step"] * 4 * 16
        assert record["tokens_per_sec"] > 0
        assert record["rss_bytes"] > 0
        assert record["wall_clock"] > 1_600_000_000  # a real unix timestamp
        assert record["epoch"] > 0


def test_checkpoint_records_are_written(tmp_path):
    train_ds, val_ds = _datasets()
    Trainer(tmp_path).fit(_tiny_model(), train_ds, val_ds, _config(), None)

    records = [
        r for r in _read_jsonl(tmp_path / "logs" / "train.jsonl") if r["type"] == "checkpoint"
    ]
    assert records
    assert records[-1]["bytes"] > 0
    assert records[-1]["free_bytes_after"] > 0


# -- retention wired through Trainer ----------------------------------------


def test_trainer_prunes_while_training(tmp_path):
    train_ds, val_ds = _datasets()
    Trainer(tmp_path, keep_last=2).fit(
        _tiny_model(),
        train_ds,
        val_ds,
        _config(total_tokens=4 * 16 * 10, checkpoint_interval_steps=1),
        None,
    )

    rolling = sorted((tmp_path / "checkpoints").glob("step_*.pt"))
    assert len(rolling) == 2, [p.name for p in rolling]
    assert (tmp_path / "checkpoints" / "latest.pt").exists()


def test_trainer_writes_milestones_that_survive_pruning(tmp_path):
    train_ds, val_ds = _datasets()
    Trainer(tmp_path, keep_last=1, keep_every=2).fit(
        _tiny_model(),
        train_ds,
        val_ds,
        _config(total_tokens=4 * 16 * 10, checkpoint_interval_steps=1),
        None,
    )

    milestones = sorted((tmp_path / "checkpoints").glob("milestone_*.pt"))
    assert len(milestones) >= 2
    # milestones are weights-only, so they must be unusable for resume
    assert load_checkpoint(milestones[0])["optimizer_state_dict"] is None


def test_checkpoint_is_skipped_rather_than_half_written_when_disk_is_full(
    tmp_path, monkeypatch
):
    import shutil as shutil_module

    from qwenmoe.train import trainer as trainer_module

    real_disk_usage = shutil_module.disk_usage

    class TinyDisk:
        free = 1  # one byte

    monkeypatch.setattr(
        trainer_module.shutil, "disk_usage", lambda p: TinyDisk() if "checkpoints" in str(p) else real_disk_usage(p)
    )

    train_ds, val_ds = _datasets()
    result = Trainer(tmp_path).fit(_tiny_model(), train_ds, val_ds, _config(), None)

    assert result.checkpoint_path is None
    assert list((tmp_path / "checkpoints").glob("*.pt")) == []
    skipped = [
        r
        for r in _read_jsonl(tmp_path / "logs" / "train.jsonl")
        if r["type"] == "checkpoint_skipped"
    ]
    assert skipped, "a skipped checkpoint must be recorded, not silent"


# -- graceful stop -----------------------------------------------------------


def test_sigint_finishes_the_step_then_checkpoints_and_exits(tmp_path):
    """A Ctrl-C on a multi-day run must end with a valid checkpoint and a
    clean log, not a traceback and a truncated file."""
    train_ds, val_ds = _datasets()
    config = _config(total_tokens=4 * 16 * 10_000, checkpoint_interval_steps=100_000)

    def interrupt_soon():
        time.sleep(0.4)
        os.kill(os.getpid(), signal.SIGINT)

    threading.Thread(target=interrupt_soon, daemon=True).start()
    result = Trainer(tmp_path).fit(_tiny_model(), train_ds, val_ds, config, None)

    total_steps = config.total_tokens // (config.batch_size * config.seq_len)
    assert 0 < result.final_step < total_steps, "should have stopped early"

    records = _read_jsonl(tmp_path / "logs" / "train.jsonl")
    interrupts = [r for r in records if r["type"] == "interrupt"]
    assert len(interrupts) == 1
    assert interrupts[0]["step"] == result.final_step

    # and the checkpoint it left behind must actually be resumable
    checkpoint = load_checkpoint(tmp_path / "checkpoints" / "latest.pt")
    assert checkpoint["step"] == result.final_step
    assert checkpoint["optimizer_state_dict"] is not None


def test_signal_handlers_are_restored_after_fit(tmp_path):
    """`Trainer.fit` runs ~20 times in this suite and from notebooks; a
    leaked handler would outlive the run that installed it."""
    before_int = signal.getsignal(signal.SIGINT)
    before_term = signal.getsignal(signal.SIGTERM)

    train_ds, val_ds = _datasets()
    Trainer(tmp_path).fit(_tiny_model(), train_ds, val_ds, _config(), None)

    assert signal.getsignal(signal.SIGINT) is before_int
    assert signal.getsignal(signal.SIGTERM) is before_term
