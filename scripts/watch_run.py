#!/usr/bin/env python
"""Reads a run's `logs/train.jsonl` and says what's happening (journal 041).

A multi-day run can't be supervised by watching a terminal, and `elapsed_s`
alone doesn't answer the questions that matter at hour 40: is it still
going, is it still fast, is it about to run out of disk, and when will it
finish. Stdlib only, read-only, safe to run against a live run.

Usage:
    uv run python scripts/watch_run.py runs/fineweb_300m           # live
    uv run python scripts/watch_run.py runs/fineweb_300m --once    # one snapshot

`--once` exits non-zero if anything looks wrong, so it can drive a cron
job or a notification hook instead of a watched terminal:

    */10 * * * * cd ... && uv run python scripts/watch_run.py runs/x --once || notify

Alert conditions: a device fallback (the run silently dropped to cpu and is
now ~100x slower), throughput under half the run's own median, no new step
record for --stale-minutes, a non-finite loss, or free disk under
--min-free-gb.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import statistics
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

_SLOW_FRACTION = 0.5
_RECENT_STEPS = 50


def read_records(path: Path) -> list[dict]:
    """Tolerates a partial trailing line: the trainer flushes after every
    record, but a reader can still arrive mid-write."""
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def _fmt_duration(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0:
        return "n/a"
    seconds = int(seconds)
    days, rem = divmod(seconds, 86_400)
    hours, rem = divmod(rem, 3_600)
    minutes, _ = divmod(rem, 60)
    if days:
        return f"{days}d {hours:02d}h"
    if hours:
        return f"{hours}h {minutes:02d}m"
    return f"{minutes}m"


def _bar(fraction: float, width: int = 28) -> str:
    filled = int(max(0.0, min(1.0, fraction)) * width)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def expert_balance(counts: list[list[int]]) -> float | None:
    """Worst per-layer `max/mean` dispatched-token ratio -- 1.0 is perfect
    balance, and a persistently high value means the router has collapsed
    onto a few experts and the other experts' parameters are dead weight.
    Computed here from counts the trainer already logs, so watching costs
    the run nothing.
    """
    worst = None
    for layer in counts:
        total = sum(layer)
        if not layer or total == 0:
            continue
        ratio = max(layer) / (total / len(layer))
        worst = ratio if worst is None else max(worst, ratio)
    return worst


def summarize(run_dir: Path, *, stale_minutes: float, min_free_gb: float) -> tuple[str, list[str]]:
    """Returns `(report, alerts)`."""
    log_path = run_dir / "logs" / "train.jsonl"
    if not log_path.exists():
        return f"no log at {log_path}", [f"missing log: {log_path}"]

    records = read_records(log_path)
    steps = [r for r in records if r.get("type") == "step"]
    evals = [r for r in records if r.get("type") == "eval"]
    checkpoints = [r for r in records if r.get("type") == "checkpoint"]
    starts = [r for r in records if r.get("type") == "run_start"]
    fallbacks = [r for r in records if r.get("type") == "device_fallback"]
    ooms = [r for r in records if r.get("type") == "device_oom"]
    interrupts = [r for r in records if r.get("type") == "interrupt"]
    skipped = [r for r in records if r.get("type") == "checkpoint_skipped"]

    if not steps:
        started = "; run_start seen" if starts else ""
        return f"{run_dir}: no steps logged yet{started}", []

    last = steps[-1]
    start = starts[-1] if starts else {}
    total_steps = start.get("total_steps") or last["step"]
    alerts: list[str] = []
    lines: list[str] = []

    fraction = last["step"] / total_steps if total_steps else 0.0
    lines.append(f"{run_dir}")
    if start:
        lines.append(
            f"  {start.get('params_total', 0):,} params "
            f"({start.get('params_active', 0):,} active) on {start.get('device', '?')}"
            f" / {start.get('precision', '?')}"
        )
    lines.append(f"  {_bar(fraction)} step {last['step']:,}/{total_steps:,} ({fraction:.1%})")

    tokens = last.get("tokens")
    if tokens is not None:
        budget = start.get("chinchilla_budget_tokens")
        against = f" / {budget:,} chinchilla floor ({tokens / budget:.1%})" if budget else ""
        lines.append(f"  tokens {tokens:,}{against}")
    if last.get("epoch") is not None:
        lines.append(f"  epoch  {last['epoch']:.3f} over the train split")

    # Throughput: recent against the rate this run *established*, not
    # against its all-time median. Using the median would make the alert
    # silence itself -- once a slowdown has run long enough to dominate the
    # sample, the baseline sinks to meet it and the run looks "steady" at
    # a fifth of its original speed. The baseline is therefore the first
    # window of steps, which is the rate the machine actually sustained.
    rates = [r["tokens_per_sec"] for r in steps if r.get("tokens_per_sec")]
    if rates:
        recent = statistics.median(rates[-_RECENT_STEPS:])
        baseline = statistics.median(rates[:_RECENT_STEPS])
        overall = statistics.median(rates)
        trend = "steady"
        if recent < baseline * _SLOW_FRACTION:
            trend = "SLOW"
            alerts.append(
                f"throughput {recent:,.0f} tok/s is under half the run median {baseline:,.0f} "
                f"tok/s this run established"
            )
        elif recent > baseline * 1.2:
            trend = "faster"
        lines.append(
            f"  speed  {recent:,.0f} tok/s recent / {baseline:,.0f} baseline "
            f"/ {overall:,.0f} median  ({trend})"
        )

    losses = [r["train_loss"] for r in steps if r.get("train_loss") is not None]
    if losses:
        window = losses[-_RECENT_STEPS:]
        lines.append(f"  loss   {losses[-1]:.4f} (mean of last {len(window)}: {statistics.mean(window):.4f})")
        if not math.isfinite(losses[-1]):
            alerts.append(f"train_loss is {losses[-1]} at step {last['step']}")
    if evals:
        last_eval = evals[-1]
        lines.append(
            f"  val    {last_eval['val_loss']:.4f} (ppl {last_eval['val_perplexity']:,.1f})"
            f" at step {last_eval['step']:,}"
        )

    if last.get("eta_s") is not None:
        finish = datetime.now() + timedelta(seconds=last["eta_s"])
        lines.append(f"  eta    {_fmt_duration(last['eta_s'])} -> {finish:%a %d %b %H:%M}")

    mem = []
    if last.get("rss_bytes"):
        mem.append(f"rss {last['rss_bytes'] / 1e9:.2f}GB")
    # journal 042: prefer the device-neutral pair, fall back to the mps-only
    # keys so logs written before the CUDA port still render.
    label = (last.get("device") or "device").split(":")[0]
    allocated = last.get("device_allocated_bytes") or last.get("mps_allocated_bytes")
    reserved = last.get("device_reserved_bytes") or last.get("mps_driver_bytes")
    if allocated:
        mem.append(f"{label} {allocated / 1e9:.2f}GB")
    if reserved:
        # CUDA's memory_reserved / MPS's driver_allocated_memory: what the
        # allocator holds, which is the figure that predicts an OOM
        mem.append(f"reserved {reserved / 1e9:.2f}GB")
    if mem:
        lines.append("  memory " + "  ".join(mem))

    balance = expert_balance(last.get("expert_token_counts") or [])
    if balance is not None:
        lines.append(f"  expert max/mean {balance:.2f} (1.00 is perfectly balanced)")

    if checkpoints:
        ck = checkpoints[-1]
        age = time.time() - ck.get("wall_clock", time.time())
        lines.append(
            f"  ckpt   step {ck['step']:,} ({(ck.get('bytes') or 0) / 1e9:.2f}GB, {_fmt_duration(age)} ago)"
        )

    free_gb = shutil.disk_usage(run_dir).free / 1e9
    lines.append(f"  disk   {free_gb:.1f}GB free")
    if free_gb < min_free_gb:
        alerts.append(f"only {free_gb:.1f}GB free, under the {min_free_gb:.0f}GB floor")

    if last.get("wall_clock"):
        stale_s = time.time() - last["wall_clock"]
        if stale_s > stale_minutes * 60:
            alerts.append(
                f"no step logged for {_fmt_duration(stale_s)} (last was step {last['step']:,})"
            )

    for oom in ooms:
        alerts.append(
            f"CUDA OOM at step {oom['step']:,} "
            f"(allocated {(oom.get('device_allocated_bytes') or 0) / 1e9:.1f}GB, "
            f"reserved {(oom.get('device_reserved_bytes') or 0) / 1e9:.1f}GB) -- "
            "the run died; resume from checkpoints/latest.pt"
        )
    for fallback in fallbacks:
        alerts.append(
            f"DEVICE FALLBACK at step {fallback['step']:,}: "
            f"{fallback.get('from')} -> {fallback.get('to')} (the run is now far slower)"
        )
    for skip in skipped:
        alerts.append(f"checkpoint SKIPPED at step {skip['step']:,} for lack of disk")
    if interrupts:
        lines.append(f"  stopped on request at step {interrupts[-1]['step']:,}")

    return "\n".join(lines), alerts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--once", action="store_true", help="one snapshot, exit non-zero on alerts")
    parser.add_argument("--interval", type=float, default=30.0, help="seconds between refreshes")
    parser.add_argument("--stale-minutes", type=float, default=10.0)
    parser.add_argument("--min-free-gb", type=float, default=10.0)
    args = parser.parse_args()

    while True:
        report, alerts = summarize(
            args.run_dir, stale_minutes=args.stale_minutes, min_free_gb=args.min_free_gb
        )
        if args.once:
            print(report)
            for alert in alerts:
                print(f"  ALERT: {alert}", file=sys.stderr)
            sys.exit(1 if alerts else 0)

        print("\033[2J\033[H", end="")  # clear, home
        print(f"{datetime.now():%H:%M:%S}\n{report}")
        for alert in alerts:
            print(f"  ALERT: {alert}")
        try:
            time.sleep(args.interval)
        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    main()
