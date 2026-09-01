"""Manim preview for Task 9's math (`count_params`'s total-vs-active
parameter breakdown, `chinchilla_budget`/`epochs_for_budget`'s token-
budget planning, and `calibrate_throughput`'s empirical device timing),
animated from the real, already-tested `qwenmoe.train.scaling` functions
rather than fabricated numbers (Requirement 7.1's spirit) -- built ahead
of the formal TraceBundle-based pipeline (tasks 12-13), same approach as
`scenes_task3_preview.py`/`scenes_task5_preview.py`/
`scenes_task6_preview.py`/`scenes_task7_preview.py`/
`scenes_task8_preview.py`.
"""
from __future__ import annotations

import torch
from manim import *

from configs.presets import BASELINE
from qwenmoe.config import ModelConfig
from qwenmoe.model.transformer import QwenMoEModel
from qwenmoe.train.scaling import (
    _EPOCH_WARNING_THRESHOLD,  # the real threshold, not re-typed as a magic "4"
    calibrate_throughput,
    chinchilla_budget,
    count_params,
    epochs_for_budget,
)

# Same tiny fixture as tests/test_scaling.py, so every number this scene
# shows matches journal entry 009's hand-computed worked example exactly.
_TINY_CONFIG = dict(
    vocab_size=50, d_model=16, n_layers=2, n_heads=4, n_kv_heads=2,
    d_ff=32, n_routed_experts=4, n_shared_experts=1, top_k=2, max_seq_len=16,
)

# real tiny-Shakespeare byte/character count downloaded and verified in task 8
_TINY_SHAKESPEARE_TOKENS = 1_115_394


class Task9ScalingLawScene(Scene):
    def construct(self):
        self._param_breakdown_part()
        self._token_budget_part()
        self._throughput_part()

    def _title(self, text):
        title = Text(text, font_size=26)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(0.5)
        return title

    def _manual_bars(self, heights, names, colors, value_labels, unit, bar_width=1.1, gap=0.7):
        """Hand-built bar chart (Manim's `BarChart`/`Axes` default to
        LaTeX-rendered labels, which fails with no LaTeX installed -- same
        approach as every earlier preview's `_manual_bars`)."""
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
                width=bar_width, height=max(h, 0.03), fill_color=color, fill_opacity=0.85, stroke_width=0
            )
            rect.move_to([x, h / 2, 0])
            name_label = Text(name, font_size=16).next_to([x, 0, 0], DOWN, buff=0.15)
            val_label = Text(label, font_size=15).next_to(rect, UP, buff=0.1)
            group.add(rect, name_label, val_label)
        return group

    def _stacked_bar(self, x, base_y, segments, unit, bar_width=1.6):
        """`segments`: list of (value, color), stacked bottom-to-top from `base_y`."""
        group = VGroup()
        y = base_y
        for value, color in segments:
            h = value * unit
            rect = Rectangle(
                width=bar_width, height=h, fill_color=color, fill_opacity=0.9,
                stroke_width=1, stroke_color=WHITE,
            )
            rect.move_to([x, y + h / 2, 0])
            group.add(rect)
            y += h
        return group

    def _repeat_bar(self, y, budget, unique_tokens, tick_color, width=9.0, height=0.55):
        """A `budget`-long bar divided into `unique_tokens`-sized repeats --
        each tick is one full pass (epoch) over the available corpus."""
        epochs = budget / unique_tokens
        warn = epochs > _EPOCH_WARNING_THRESHOLD
        outline = Rectangle(width=width, height=height, color=RED if warn else GREEN).move_to([0, y, 0])
        fill = Rectangle(
            width=width, height=height, fill_color=tick_color, fill_opacity=0.2, stroke_width=0,
        ).move_to([0, y, 0])
        seg_width = width * (unique_tokens / budget)
        ticks = VGroup()
        x = -width / 2 + seg_width
        guard = 0
        while x < width / 2 - 1e-6 and guard < 200:
            ticks.add(Line([x, y - height / 2, 0], [x, y + height / 2, 0], color=tick_color, stroke_width=2.5))
            x += seg_width
            guard += 1
        return VGroup(fill, outline, ticks), epochs, warn

    # ---- Part 1: count_params -- total vs. active parameters ----

    def _param_breakdown_part(self):
        title = self._title("count_params: total vs. active parameters (Requirement 11.1)")

        config = ModelConfig(**_TINY_CONFIG)
        model = QwenMoEModel(config)
        b = count_params(model)
        per_expert = sum(p.numel() for p in model.blocks[0].moe.routed_experts[0].parameters())

        setup = Text(
            f"tiny config: d_model=16, n_layers=2, n_routed_experts=4, top_k=2  ->  "
            f"total={b.total:,}   active={b.active:,}",
            font_size=18,
        ).move_to([0, 3.2, 0])
        self.play(Write(setup))

        legend = Text(
            "gray=embedding   blue=attention   green=moe (active this token)   "
            "yellow=moe (skipped this token)",
            font_size=15, color=GRAY,
        ).move_to([0, 2.5, 0])
        self.play(Write(legend))

        base_y = -2.1
        unit = 2.6 / b.total
        total_segments = [
            (b.embedding, GRAY), (b.attention_total, BLUE),
            (b.moe_active, GREEN), (b.moe_total - b.moe_active, YELLOW),
        ]
        active_segments = [(b.embedding, GRAY), (b.attention_total, BLUE), (b.moe_active, GREEN)]
        total_bar = self._stacked_bar(-2.3, base_y, total_segments, unit)
        active_bar = self._stacked_bar(2.3, base_y, active_segments, unit)
        total_label = Text(f"total = {b.total:,}", font_size=18).next_to(total_bar, DOWN, buff=0.3)
        active_label = Text(f"active = {b.active:,}", font_size=18).next_to(active_bar, DOWN, buff=0.3)

        self.play(Create(total_bar), Write(total_label))
        self.play(Create(active_bar), Write(active_label))
        self.wait(1.5)
        self.play(FadeOut(setup, legend, total_bar, total_label, active_bar, active_label))

        zoom_caption = Text(
            f"embedding ({b.embedding:,}) and attention ({b.attention_total:,}) are IDENTICAL "
            "either way -- both are fully dense. Only the MoE segment differs:",
            font_size=17,
        ).move_to([0, 2.6, 0])
        self.play(Write(zoom_caption))

        moe_bars = self._manual_bars(
            [b.moe_total, b.moe_active], ["moe total", "moe active"], [YELLOW, GREEN],
            [f"{b.moe_total:,}", f"{b.moe_active:,}"], unit=2.2 / b.moe_total,
        )
        moe_bars.move_to([0, 0.1, 0])
        self.play(Create(moe_bars))
        self.wait(1.3)
        self.play(FadeOut(zoom_caption, moe_bars))

        why_caption = Text(
            f"why: each layer has {config.n_routed_experts} routed experts "
            f"({per_expert:,} params each), but only top_k={config.top_k} "
            "run for any single token",
            font_size=18,
        ).move_to([0, 2.6, 0])
        self.play(Write(why_caption))

        n = config.n_routed_experts
        box_w, gap = 1.7, 0.35
        total_w = n * box_w + (n - 1) * gap
        start_x = -total_w / 2 + box_w / 2
        experts = VGroup()
        for i in range(n):
            x = start_x + i * (box_w + gap)
            active_expert = i < config.top_k
            color = GREEN if active_expert else GRAY
            box = Rectangle(
                width=box_w, height=1.1, color=color, fill_color=color,
                fill_opacity=0.35 if active_expert else 0.15,
            ).move_to([x, 0.9, 0])
            label = Text(
                f"expert {i}\n{per_expert:,} params\n{'ACTIVE' if active_expert else 'skipped'}",
                font_size=14, color=color if active_expert else GRAY,
            ).move_to(box.get_center())
            experts.add(box, label)
        self.play(Create(experts))
        self.wait(1.5)

        mech_caption = Text(
            "these 2 skipped experts still count toward `total` -- they just never "
            "touch THIS token's computation, so they're excluded from `active`",
            font_size=15, color=GRAY,
        ).move_to([0, -1.4, 0])
        self.play(Write(mech_caption))
        self.wait(2)
        self.play(FadeOut(title, why_caption, experts, mech_caption))

    # ---- Part 2: chinchilla_budget + epochs_for_budget ----

    def _token_budget_part(self):
        title = self._title(
            "chinchilla_budget & epochs_for_budget: planning a token budget (Req 11.2, 11.3)"
        )

        config = ModelConfig(**_TINY_CONFIG)
        active = count_params(QwenMoEModel(config)).active
        budget = chinchilla_budget(active)

        setup = Text(
            f"chinchilla_budget(active_params={active:,}) = 20 x {active:,} = {budget:,}",
            font_size=19,
        ).move_to([0, 3.1, 0])
        self.play(Write(setup))
        self.wait(0.8)

        small_corpus, big_corpus = 40_000, 100_000
        bar1, epochs1, warn1 = self._repeat_bar(1.55, budget, small_corpus, BLUE)
        text1 = Text(
            f"unique_tokens={small_corpus:,}  ->  epochs_for_budget = {epochs1:.2f} "
            f"{'(> ' + str(_EPOCH_WARNING_THRESHOLD) + ' -> WARNING)' if warn1 else '(no warning)'}",
            font_size=16, color=RED if warn1 else GREEN,
        ).next_to(bar1, DOWN, buff=0.25)
        self.play(Create(bar1), Write(text1))
        self.wait(1)

        bar2, epochs2, warn2 = self._repeat_bar(-0.55, budget, big_corpus, BLUE)
        text2 = Text(
            f"unique_tokens={big_corpus:,}  ->  epochs_for_budget = {epochs2:.2f} "
            f"{'(> ' + str(_EPOCH_WARNING_THRESHOLD) + ' -> WARNING)' if warn2 else '(no warning)'}",
            font_size=16, color=RED if warn2 else GREEN,
        ).next_to(bar2, DOWN, buff=0.25)
        self.play(Create(bar2), Write(text2))
        self.wait(1.5)
        self.play(FadeOut(setup, bar1, text1, bar2, text2))

        # real project scale: this project's actual BASELINE preset (journal entry 009)
        baseline_active = count_params(QwenMoEModel(BASELINE)).active
        baseline_budget = chinchilla_budget(baseline_active)
        baseline_epochs = epochs_for_budget(baseline_budget, _TINY_SHAKESPEARE_TOKENS)

        real_title = Text("at this project's real scale (BASELINE preset):", font_size=19).move_to([0, 2.4, 0])
        real_numbers = Text(
            f"chinchilla_budget({baseline_active:,}) = {baseline_budget:,} tokens",
            font_size=18,
        ).move_to([0, 1.7, 0])
        real_corpus = Text(
            f"tiny-Shakespeare's real downloaded corpus (task 8): only "
            f"{_TINY_SHAKESPEARE_TOKENS:,} unique tokens",
            font_size=18,
        ).move_to([0, 1.1, 0])
        real_epochs = Text(
            f"epochs_for_budget(...) = {baseline_epochs:,.2f}  ->  fires the warning by a huge margin",
            font_size=19, color=RED,
        ).move_to([0, 0.4, 0])
        real_moral = Text(
            "exactly why task 8's TinyStories fixed-slice tier exists, not the "
            "tiny-Shakespeare smoke-test corpus, for the actual comparison study",
            font_size=15, color=GRAY,
        ).move_to([0, -0.5, 0])
        self.play(Write(real_title))
        self.play(Write(real_numbers))
        self.play(Write(real_corpus))
        self.play(Write(real_epochs))
        self.wait(1)
        self.play(Write(real_moral))
        self.wait(2.5)
        self.play(FadeOut(title, real_title, real_numbers, real_corpus, real_epochs, real_moral))

    # ---- Part 3: calibrate_throughput -- measure, don't guess ----

    def _throughput_part(self):
        title = self._title("calibrate_throughput: measured tokens/sec, not a FLOPs guess (Req 11.4)")

        torch.manual_seed(0)
        config = ModelConfig(**_TINY_CONFIG)
        model = QwenMoEModel(config)
        batch_size, seq_len, n_steps = 4, 16, 15
        x = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        y = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        devices = ["cpu"]
        if torch.backends.mps.is_available():
            devices.append("mps")
        results = {d: calibrate_throughput(model, (x, y), device=d, n_steps=n_steps) for d in devices}

        setup = Text(
            f"batch_size={batch_size}, seq_len={seq_len}, n_steps={n_steps} on: {', '.join(devices)}",
            font_size=18,
        ).move_to([0, 3.1, 0])
        self.play(Write(setup))

        formula = Text(
            "tokens/sec = (batch_size x seq_len x n_steps) / elapsed_seconds", font_size=17, color=GRAY,
        ).move_to([0, 2.5, 0])
        self.play(Write(formula))

        warmup = Rectangle(width=2.6, height=0.9, color=GRAY).move_to([-4.3, 1.0, 0])
        warmup_label = Text("1 warm-up step\n(not timed)", font_size=14).move_to(warmup.get_center())
        timed = Rectangle(width=3.2, height=0.9, color=BLUE).move_to([-0.6, 1.0, 0])
        timed_label = Text(f"{n_steps} timed steps\n(forward + backward)", font_size=14).move_to(timed.get_center())
        result_box = Rectangle(width=2.6, height=0.9, color=GREEN).move_to([3.3, 1.0, 0])
        result_label = Text("tokens_per_sec", font_size=16).move_to(result_box.get_center())
        arrow1 = Arrow(warmup.get_right(), timed.get_left(), buff=0.1)
        arrow2 = Arrow(timed.get_right(), result_box.get_left(), buff=0.1)
        self.play(Create(warmup), Write(warmup_label))
        self.play(Create(arrow1), Create(timed), Write(timed_label))
        self.play(Create(arrow2), Create(result_box), Write(result_label))
        self.wait(1.3)
        self.play(FadeOut(warmup, warmup_label, timed, timed_label, result_box, result_label, arrow1, arrow2))

        unit = 2.4 / max(results.values())
        bars = self._manual_bars(
            [results[d] for d in devices], devices, [BLUE, GREEN][: len(devices)],
            [f"{results[d]:,.0f}/s" for d in devices], unit=unit, bar_width=1.6, gap=1.6,
        )
        bars.move_to([0, -0.7, 0])
        self.play(Create(bars))
        self.wait(1.3)

        real_ref = Text(
            "this project's real BASELINE preset (journal entry 009, measured on "
            "this same laptop): cpu ~10,056 tok/s, mps ~12,556 tok/s",
            font_size=15, color=GRAY,
        ).move_to([0, -2.6, 0])
        self.play(Write(real_ref))
        self.wait(2.5)
        self.play(FadeOut(title, setup, formula, bars, real_ref))
