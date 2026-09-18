"""journal 041: `scripts/watch_run.py` turns a run's `train.jsonl` into an
answer to "is it still going, is it still fast, is it about to die".

The alert conditions are the point -- `--once` exits non-zero on them so a
cron job can watch a multi-day run instead of a person. Each one is pinned
here, because an alert that doesn't fire is worse than no alert at all.
"""
import json
import time

import pytest

from scripts.watch_run import expert_balance, read_records, summarize


def _write_log(tmp_path, records):
    log = tmp_path / "logs" / "train.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    return tmp_path


def _step(step, **overrides):
    record = {
        "type": "step",
        "step": step,
        "train_loss": 5.0 - step * 0.01,
        "tokens": step * 16384,
        "tokens_per_sec": 2500.0,
        "elapsed_s": 6.5,
        "wall_clock": time.time(),
        "eta_s": 3600.0,
        "rss_bytes": 2_000_000_000,
        "mps_allocated_bytes": 8_000_000_000,
        "mps_driver_bytes": 12_000_000_000,
        "epoch": step * 0.001,
        "device": "mps",
        "expert_token_counts": [[100, 100, 100, 100]],
    }
    record.update(overrides)
    return record


def _run_start(**overrides):
    record = {
        "type": "run_start",
        "total_steps": 1000,
        "params_total": 346_896_128,
        "params_active": 134_559_488,
        "chinchilla_budget_tokens": 2_691_189_760,
        "device": "mps",
        "precision": "fp32",
    }
    record.update(overrides)
    return record


def _summarize(tmp_path, **kwargs):
    kwargs.setdefault("stale_minutes", 10.0)
    kwargs.setdefault("min_free_gb", 10.0)
    return summarize(tmp_path, **kwargs)


# -- reading


def test_read_records_tolerates_a_partial_trailing_line(tmp_path):
    """The trainer flushes per record, but a watcher can still arrive
    mid-write."""
    log = tmp_path / "train.jsonl"
    log.write_text('{"type": "step", "step": 1}\n{"type": "step", "st')
    assert read_records(log) == [{"type": "step", "step": 1}]


def test_missing_log_is_an_alert_not_a_crash(tmp_path):
    report, alerts = _summarize(tmp_path)
    assert "no log" in report
    assert alerts


def test_run_with_no_steps_yet_is_not_an_alert(tmp_path):
    """Model construction alone takes ~24s for SCALE_300M; a watcher that
    cried wolf during startup would be useless."""
    _write_log(tmp_path, [_run_start()])
    report, alerts = _summarize(tmp_path)
    assert "no steps logged yet" in report
    assert alerts == []


# -- the normal report


def test_healthy_run_reports_progress_and_raises_nothing(tmp_path):
    _write_log(tmp_path, [_run_start()] + [_step(i) for i in range(1, 51)])
    report, alerts = _summarize(tmp_path)

    assert alerts == []
    assert "step 50/1,000" in report
    assert "5.0%" in report
    assert "2,500 tok/s" in report
    assert "steady" in report


def test_report_includes_chinchilla_progress(tmp_path):
    _write_log(tmp_path, [_run_start()] + [_step(i) for i in range(1, 11)])
    report, _ = _summarize(tmp_path)
    assert "2,691,189,760 chinchilla floor" in report


def test_eval_and_checkpoint_records_are_surfaced(tmp_path):
    _write_log(
        tmp_path,
        [_run_start()]
        + [_step(i) for i in range(1, 11)]
        + [
            {"type": "eval", "step": 10, "val_loss": 4.2, "val_perplexity": 66.7},
            {
                "type": "checkpoint",
                "step": 10,
                "wall_clock": time.time(),
                "bytes": 4_162_753_536,
                "milestone": None,
            },
        ],
    )
    report, alerts = _summarize(tmp_path)
    assert "ppl 66.7" in report
    assert "4.16GB" in report
    assert alerts == []


# -- the alerts, one by one


def test_device_fallback_alerts(tmp_path):
    """The worst unattended failure mode: the run keeps going, ~100x
    slower, and only a scrolled-away warning said so."""
    _write_log(
        tmp_path,
        [_run_start()]
        + [_step(i) for i in range(1, 11)]
        + [{"type": "device_fallback", "step": 7, "from": "mps", "to": "cpu"}],
    )
    _, alerts = _summarize(tmp_path)
    assert any("DEVICE FALLBACK" in a for a in alerts)


def test_throughput_collapse_alerts(tmp_path):
    fast = [_step(i) for i in range(1, 60)]
    slow = [_step(i, tokens_per_sec=200.0) for i in range(60, 120)]
    _write_log(tmp_path, [_run_start()] + fast + slow)
    _, alerts = _summarize(tmp_path)
    assert any("under half the run median" in a for a in alerts)


def test_a_sustained_slowdown_does_not_silence_its_own_alert(tmp_path):
    """The baseline must be the rate the run established, not its all-time
    median. Against the median, a slowdown that lasts long enough to
    dominate the sample drags the baseline down to meet it and the run
    reads as "steady" at a fifth of its original speed."""
    fast = [_step(i) for i in range(1, 30)]
    slow = [_step(i, tokens_per_sec=200.0) for i in range(30, 400)]
    _write_log(tmp_path, [_run_start()] + fast + slow)
    _, alerts = _summarize(tmp_path)
    assert any("under half" in a for a in alerts), "alert went silent as the slowdown persisted"


def test_stale_run_alerts(tmp_path):
    old = time.time() - 3600
    _write_log(tmp_path, [_run_start()] + [_step(i, wall_clock=old) for i in range(1, 11)])
    _, alerts = _summarize(tmp_path, stale_minutes=10.0)
    assert any("no step logged for" in a for a in alerts)


def test_non_finite_loss_alerts(tmp_path):
    _write_log(
        tmp_path,
        [_run_start()] + [_step(i) for i in range(1, 10)] + [_step(10, train_loss=float("nan"))],
    )
    _, alerts = _summarize(tmp_path)
    assert any("train_loss" in a for a in alerts)


def test_low_disk_alerts(tmp_path):
    _write_log(tmp_path, [_run_start()] + [_step(i) for i in range(1, 11)])
    _, alerts = _summarize(tmp_path, min_free_gb=10_000_000.0)
    assert any("free" in a for a in alerts)


def test_skipped_checkpoint_alerts(tmp_path):
    _write_log(
        tmp_path,
        [_run_start()]
        + [_step(i) for i in range(1, 11)]
        + [{"type": "checkpoint_skipped", "step": 10, "free_bytes": 1, "needed_bytes": 2}],
    )
    _, alerts = _summarize(tmp_path)
    assert any("SKIPPED" in a for a in alerts)


def test_interrupt_is_reported_but_is_not_an_alert(tmp_path):
    """A deliberate stop is not a failure."""
    _write_log(
        tmp_path,
        [_run_start()]
        + [_step(i) for i in range(1, 11)]
        + [{"type": "interrupt", "step": 10, "wall_clock": time.time()}],
    )
    report, alerts = _summarize(tmp_path)
    assert "stopped on request" in report
    assert alerts == []


# -- expert balance


def test_expert_balance_is_one_when_perfectly_balanced():
    assert expert_balance([[100, 100, 100, 100]]) == pytest.approx(1.0)


def test_expert_balance_reports_the_worst_layer():
    balanced = [10, 10, 10, 10]
    collapsed = [40, 0, 0, 0]  # one expert takes everything -> max/mean = 4
    assert expert_balance([balanced, collapsed]) == pytest.approx(4.0)


def test_expert_balance_ignores_layers_with_no_dispatched_tokens():
    assert expert_balance([[0, 0], [5, 5]]) == pytest.approx(1.0)


def test_expert_balance_is_none_without_counts():
    assert expert_balance([]) is None
