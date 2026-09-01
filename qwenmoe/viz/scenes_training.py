"""Task 14 -- training/run observability scenes (Requirement 8.1, 8.2): a
loss-curve scene and an expert-utilization-distribution scene, both
sourced from the real JSON-lines log `Trainer.fit` writes to
`<output_dir>/logs/train.jsonl` (`qwenmoe/train/trainer.py`, task 10.1).
Unlike every task-13 scene, these do NOT go through
`capture_trace`/`TraceBundle` at all -- they read run-level metrics
accumulated across many steps, not a single forward pass (design.md
`viz/` Components: "reads the JSON/CSV logs Trainer writes ... rather
than a TraceBundle"). See journey_of_a_model/010-training-loop.md for the
log format itself and journey_of_a_model/014-training-observability-
scenes.md for this task's own derivations.

`load_training_log` raises a clear, actionable `FileNotFoundError` if the
log doesn't exist yet (Requirement 8.3). Both Scene subclasses call it as
the very first step of `construct()`, before any Manim primitive is
created, so a missing log fails fast with nothing partially drawn.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from manim import *

_LOG_PATH = Path("runs/smoke_test/logs/train.jsonl")
_LAYER_COLORS = [BLUE, ORANGE, GREEN, PURPLE, TEAL, YELLOW]


@dataclass
class TrainingLog:
    """Parsed content of one `Trainer`-written `train.jsonl` file."""

    steps: list[int]
    train_loss: list[float]
    eval_steps: list[int]
    val_loss: list[float]
    val_perplexity: list[float]
    expert_steps: list[int]
    expert_token_counts: list[list[list[int]]]  # [logged step][layer][expert] -> token count


def load_training_log(log_path: str | Path) -> TrainingLog:
    """Reads a `Trainer`-written JSON-lines log (Requirement 8.1, 8.2).

    Raises `FileNotFoundError` with an actionable message -- not a raw
    stack trace -- if `log_path` doesn't exist yet (Requirement 8.3), and
    `ValueError` for a present-but-unusable file (empty, or malformed
    JSON on some line), since both are "clear, actionable error" cases a
    caller should see before any Manim rendering is attempted.
    """
    log_path = Path(log_path)
    if not log_path.exists():
        raise FileNotFoundError(
            f"no training log found at '{log_path}'. Trainer.fit() writes this file as "
            "training runs, so generate one first, e.g.:\n"
            "  uv run python scripts/prepare_data.py --corpus tiny_shakespeare "
            "--tokenizer-type char\n"
            "  uv run python scripts/train.py --preset SMOKE_TEST --corpus tiny_shakespeare "
            "--tokenizer-type char --total-tokens 2048000 --output-dir runs/smoke_test\n"
            "(see journey_of_a_model/010-training-loop.md), which writes "
            "<output-dir>/logs/train.jsonl."
        )

    steps: list[int] = []
    train_loss: list[float] = []
    eval_steps: list[int] = []
    val_loss: list[float] = []
    val_perplexity: list[float] = []
    expert_steps: list[int] = []
    expert_token_counts: list[list[list[int]]] = []

    with log_path.open(encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"'{log_path}' line {line_no} is not valid JSON: {exc}") from exc

            if record["type"] == "step":
                steps.append(record["step"])
                train_loss.append(record["train_loss"])
                counts = record.get("expert_token_counts")
                if counts is not None:
                    expert_steps.append(record["step"])
                    expert_token_counts.append(counts)
            elif record["type"] == "eval":
                eval_steps.append(record["step"])
                val_loss.append(record["val_loss"])
                val_perplexity.append(record["val_perplexity"])

    if not steps:
        raise ValueError(f"'{log_path}' has no 'step' records -- nothing to animate")

    return TrainingLog(
        steps=steps,
        train_loss=train_loss,
        eval_steps=eval_steps,
        val_loss=val_loss,
        val_perplexity=val_perplexity,
        expert_steps=expert_steps,
        expert_token_counts=expert_token_counts,
    )


def _title(scene: Scene, text: str) -> Text:
    title = Text(text, font_size=26)
    title.to_edge(UP)
    scene.play(Write(title))
    scene.wait(0.5)
    return title


def _auto_unit(values, target=1.3):
    peak = max((abs(v) for v in values), default=1.0)
    return target / peak if peak > 1e-9 else 1.0


def _manual_bars(heights, names, colors, value_labels, unit, bar_width=1.1, gap=0.7):
    """Hand-built bar chart (Manim's `BarChart`/`Axes` default to
    LaTeX-rendered labels, which fails with no LaTeX installed)."""
    group = VGroup()
    n = len(heights)
    total_width = n * bar_width + (n - 1) * gap
    start_x = -total_width / 2 + bar_width / 2
    end_x = start_x + (n - 1) * (bar_width + gap)
    baseline = Line(
        [start_x - bar_width / 2 - 0.2, 0, 0], [end_x + bar_width / 2 + 0.2, 0, 0], color=GRAY
    )
    group.add(baseline)
    for i, (h_val, name, color, label) in enumerate(zip(heights, names, colors, value_labels)):
        x = start_x + i * (bar_width + gap)
        h = h_val * unit
        rect = Rectangle(
            width=bar_width, height=max(abs(h), 0.03), fill_color=color, fill_opacity=0.85,
            stroke_width=0,
        )
        rect.move_to([x, h / 2, 0])
        name_label = Text(name, font_size=14).next_to([x, 0, 0], DOWN, buff=0.15)
        val_label = Text(label, font_size=13).next_to(rect, UP if h >= 0 else DOWN, buff=0.1)
        group.add(rect, name_label, val_label)
    return group


def _line_chart(series, colors, legend_labels, origin=(-4.3, -2.0), width=8.6, height=4.2, y_fmt="{:.2f}"):
    """Hand-built line chart (same LaTeX-avoidance reason as
    `_manual_bars` -- Manim's `Axes` defaults to LaTeX-rendered tick
    labels). `series` is a list of `(x_values, y_values)` pairs sharing
    one pair of axes scaled to fit every series' combined range; each is
    drawn as a connected polyline (`VMobject.set_points_as_corners`) plus
    a `Dot` per real logged data point.
    """
    ox, oy = origin
    all_x = [x for xs, _ in series for x in xs]
    all_y = [y for _, ys in series for y in ys]
    x_min, x_max = min(all_x), max(all_x)
    x_span = (x_max - x_min) or 1.0
    y_min, y_max = min(all_y), max(all_y)
    y_pad = (y_max - y_min) * 0.1 or max(abs(y_max), 1.0) * 0.1
    y_min, y_max = y_min - y_pad, y_max + y_pad
    y_span = y_max - y_min

    def to_point(x, y):
        return [ox + (x - x_min) / x_span * width, oy + (y - y_min) / y_span * height, 0]

    group = VGroup()
    group.add(Line([ox, oy, 0], [ox + width, oy, 0], color=GRAY))
    group.add(Line([ox, oy, 0], [ox, oy + height, 0], color=GRAY))

    for xv in sorted(set(all_x)):
        px, py, _ = to_point(xv, y_min)
        group.add(Text(str(xv), font_size=12).next_to([px, py, 0], DOWN, buff=0.12))
    for frac in (0.0, 0.5, 1.0):
        yv = y_min + frac * y_span
        px, py, _ = to_point(x_min, yv)
        group.add(Text(y_fmt.format(yv), font_size=12).next_to([ox, py, 0], LEFT, buff=0.12))

    for (xs, ys), color in zip(series, colors):
        points = [to_point(x, y) for x, y in zip(xs, ys)]
        curve = VMobject(color=color, stroke_width=3)
        curve.set_points_as_corners(points)
        dots = VGroup(*(Dot(p, radius=0.045, color=color) for p in points))
        group.add(curve, dots)

    legend = VGroup()
    for i, (label, color) in enumerate(zip(legend_labels, colors)):
        swatch = Line([0, 0, 0], [0.35, 0, 0], color=color, stroke_width=4)
        text = Text(label, font_size=14).next_to(swatch, RIGHT, buff=0.1)
        entry = VGroup(swatch, text)
        entry.move_to([ox + width - 1.7, oy + height - 0.3 - i * 0.35, 0])
        legend.add(entry)
    group.add(legend)
    return group


class LossCurveScene(Scene):
    """Requirement 8.1: real train + validation loss over logged training
    steps, read from a real `Trainer` JSON-lines log -- no `TraceBundle`,
    no checkpoint load, just the log file."""

    def __init__(self, log_path: str | Path = _LOG_PATH, **kwargs):
        self.log_path = Path(log_path)
        super().__init__(**kwargs)

    def construct(self):
        log = load_training_log(self.log_path)
        self._loss_curve_part(log)

    def _loss_curve_part(self, log: TrainingLog):
        title = _title(self, "Training loss: real train + validation loss over logged steps")
        setup = Text(
            f"{self.log_path}  --  {len(log.steps)} logged steps, {len(log.eval_steps)} evals",
            font_size=17,
        ).move_to([0, 3.3, 0])
        self.play(Write(setup))

        chart = _line_chart(
            [(log.steps, log.train_loss), (log.eval_steps, log.val_loss)],
            [BLUE, ORANGE],
            ["train_loss", "val_loss"],
        )
        self.play(Create(chart))
        self.wait(2)

        start_step, end_step = log.steps[0], log.steps[-1]
        start_loss, end_loss = log.train_loss[0], log.train_loss[-1]
        caption = Text(
            f"real drop: train_loss {start_loss:.3f} -> {end_loss:.3f} from step {start_step} "
            f"to {end_step} (final val_perplexity={log.val_perplexity[-1]:.2f})",
            font_size=15,
            color=GRAY,
        ).move_to([0, -3.3, 0])
        self.play(Write(caption))
        self.wait(3)
        self.play(FadeOut(title, setup, chart, caption))


class ExpertUtilizationScene(Scene):
    """Requirement 8.2: real per-expert token-count distribution, and how
    load-balancing changes it across the course of a real training run."""

    def __init__(self, log_path: str | Path = _LOG_PATH, **kwargs):
        self.log_path = Path(log_path)
        super().__init__(**kwargs)

    def construct(self):
        log = load_training_log(self.log_path)
        if not log.expert_token_counts:
            raise ValueError(
                f"'{self.log_path}' has 'step' records but none carry 'expert_token_counts' "
                "-- nothing to animate"
            )
        self._distribution_part(log)
        self._trend_part(log)

    # ---- Part 1: one layer's real distribution, first vs. last logged step ----

    def _distribution_part(self, log: TrainingLog):
        title = _title(self, "Expert utilization: real per-expert token counts, early vs. late")
        n_layers = len(log.expert_token_counts[0])
        layer = 1 if n_layers > 1 else 0
        first_counts = log.expert_token_counts[0][layer]
        last_counts = log.expert_token_counts[-1][layer]
        n_experts = len(first_counts)
        names = [f"e{i}" for i in range(n_experts)]

        setup = Text(
            f"layer {layer} -- expert_token_counts at step {log.expert_steps[0]} vs. step "
            f"{log.expert_steps[-1]}",
            font_size=17,
        ).move_to([0, 3.3, 0])
        self.play(Write(setup))

        unit = _auto_unit(first_counts + last_counts)
        first_bars = _manual_bars(
            first_counts, names, [BLUE] * n_experts, [str(v) for v in first_counts], unit=unit,
            bar_width=0.6, gap=0.3,
        ).move_to([-3.2, 0.2, 0])
        first_label = Text(f"step {log.expert_steps[0]}", font_size=16, color=BLUE).next_to(
            first_bars, UP, buff=0.3
        )
        last_bars = _manual_bars(
            last_counts, names, [ORANGE] * n_experts, [str(v) for v in last_counts], unit=unit,
            bar_width=0.6, gap=0.3,
        ).move_to([3.2, 0.2, 0])
        last_label = Text(f"step {log.expert_steps[-1]}", font_size=16, color=ORANGE).next_to(
            last_bars, UP, buff=0.3
        )
        self.play(Create(first_bars), Write(first_label), Create(last_bars), Write(last_label))
        self.wait(2)

        first_ratio = max(first_counts) / (sum(first_counts) / n_experts)
        last_ratio = max(last_counts) / (sum(last_counts) / n_experts)
        caption = Text(
            f"max/mean token-count ratio (1.0 = perfectly even): {first_ratio:.2f} -> "
            f"{last_ratio:.2f}",
            font_size=15,
            color=GRAY,
        ).move_to([0, -3.2, 0])
        self.play(Write(caption))
        self.wait(2.5)
        self.play(FadeOut(title, setup, first_bars, first_label, last_bars, last_label, caption))

    # ---- Part 2: max/mean imbalance ratio per layer, across every logged step ----

    def _trend_part(self, log: TrainingLog):
        title = _title(self, "Does load-balancing actually improve? Real per-layer trend")
        n_layers = len(log.expert_token_counts[0])

        setup = Text(
            "max/mean expert token-count ratio per layer, over every logged step "
            "(1.0 = perfectly even load)",
            font_size=17,
        ).move_to([0, 3.3, 0])
        self.play(Write(setup))

        series = []
        for layer_idx in range(n_layers):
            ratios = [
                max(counts_at_step[layer_idx])
                / (sum(counts_at_step[layer_idx]) / len(counts_at_step[layer_idx]))
                for counts_at_step in log.expert_token_counts
            ]
            series.append((log.expert_steps, ratios))

        colors = _LAYER_COLORS[:n_layers]
        chart = _line_chart(series, colors, [f"layer {i}" for i in range(n_layers)])
        self.play(Create(chart))
        self.wait(2)

        starts = [ys[0] for _, ys in series]
        finals = [ys[-1] for _, ys in series]
        most_improved = max(range(n_layers), key=lambda i: starts[i] - finals[i])
        caption = Text(
            f"real asymmetry: layer {most_improved} improves the most ({starts[most_improved]:.2f} "
            f"-> {finals[most_improved]:.2f}) -- not every layer balances at the same rate",
            font_size=15,
            color=GRAY,
        ).move_to([0, -3.3, 0])
        self.play(Write(caption))
        self.wait(3)
        self.play(FadeOut(title, setup, chart, caption))
