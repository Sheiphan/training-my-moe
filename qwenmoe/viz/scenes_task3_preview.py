"""Manim preview for Task 3's math (RMSNorm + RoPE + ALiBi), animated from
the real, already-tested `qwenmoe.model` classes rather than fabricated
numbers (Requirement 7.1's spirit) — built ahead of the formal
TraceBundle-based pipeline (tasks 12-13), which will re-do this properly
once a full trained model exists to capture a trace from.
"""
from __future__ import annotations

import torch
from manim import *

from qwenmoe.model.positional import AlibiBias, RotaryEmbedding
from qwenmoe.model.rmsnorm import RMSNorm


class Task3PositionalMathScene(Scene):
    def construct(self):
        self._rmsnorm_part()
        self._rope_part()
        self._alibi_part()

    def _title(self, text):
        title = Text(text, font_size=36)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(0.5)
        return title

    def _manual_bars(self, values, names, color, unit=1.0, bar_width=0.7, gap=0.35):
        """A minimal hand-built bar chart (Rectangle + Text). Manim's own
        `BarChart`/`Axes` default to LaTeX-rendered (`MathTex`) axis labels,
        which fails on a machine with no LaTeX install — this sidesteps
        that entirely by using only `Text` (Pango-based, no LaTeX needed).
        """
        group = VGroup()
        n = len(values)
        total_width = n * bar_width + (n - 1) * gap
        start_x = -total_width / 2 + bar_width / 2
        end_x = start_x + (n - 1) * (bar_width + gap)
        baseline = Line(
            [start_x - bar_width / 2 - 0.2, 0, 0], [end_x + bar_width / 2 + 0.2, 0, 0], color=GRAY
        )
        group.add(baseline)
        for i, (name, v) in enumerate(zip(names, values)):
            x = start_x + i * (bar_width + gap)
            h = v * unit
            rect_h = max(abs(h), 0.03)
            rect = Rectangle(
                width=bar_width, height=rect_h, fill_color=color, fill_opacity=0.85, stroke_width=0
            )
            rect.move_to([x, h / 2, 0])
            name_label = Text(name, font_size=18).next_to([x, 0, 0], DOWN, buff=0.15)
            value_label = Text(f"{v:.4g}", font_size=16).next_to(
                rect, UP if h >= 0 else DOWN, buff=0.1
            )
            group.add(rect, name_label, value_label)
        return group

    def _rmsnorm_part(self):
        title = self._title("RMSNorm: rescaling a vector")

        norm = RMSNorm(dim=4)
        x = torch.tensor([3.0, 4.0, 0.0, 0.0])
        with torch.no_grad():
            output = norm(x)

        names = ["x0", "x1", "x2", "x3"]
        in_bars = self._manual_bars(x.tolist(), names, "#58C4DD", unit=0.4).shift(LEFT * 3.3)
        in_label = Text("input x = [3, 4, 0, 0]", font_size=22).next_to(in_bars, DOWN, buff=0.6)

        out_vals = [round(v, 2) for v in output.tolist()]
        out_bars = self._manual_bars(out_vals, names, "#83C167", unit=0.4).shift(RIGHT * 3.3)
        out_label = Text(f"RMSNorm(x) = {out_vals}", font_size=22).next_to(out_bars, DOWN, buff=0.6)

        rms_text = Text(
            "RMS = sqrt(mean(x^2)) = sqrt(6.25) = 2.5", font_size=26
        ).next_to(title, DOWN, buff=0.4)

        self.play(Create(in_bars), Write(in_label))
        self.wait(0.5)
        self.play(Write(rms_text))
        self.wait(1)
        self.play(Create(out_bars), Write(out_label))
        self.wait(1.5)
        self.play(FadeOut(title, in_bars, in_label, out_bars, out_label, rms_text))

    def _rope_part(self):
        title = self._title("RoPE: rotating a vector by its position")

        rope = RotaryEmbedding(head_dim=2, theta=10000.0)
        q = torch.tensor([[1.0, 0.0]])
        k = torch.zeros_like(q)
        final_angle = 1.5708  # radians, ~pi/2
        with torch.no_grad():
            q_rot, _ = rope.apply_to_qk(q, k, torch.tensor([final_angle]))

        plane = NumberPlane(x_range=[-2, 2, 1], y_range=[-2, 2, 1]).scale(0.9)
        vec = Arrow(ORIGIN, [1, 0, 0], buff=0, color="#FC6255")
        vec_label = Text("q at position 0", font_size=22).next_to(vec.get_end(), RIGHT)

        self.play(Create(plane))
        self.play(GrowArrow(vec), Write(vec_label))
        self.wait(0.5)
        self.play(FadeOut(vec_label))

        self.play(Rotate(vec, angle=final_angle, about_point=ORIGIN), run_time=3)
        final_label = Text(
            f"position=pi/2 -> q_rot=[{q_rot[0, 0]:.4f}, {q_rot[0, 1]:.4f}]",
            font_size=22,
        ).next_to(plane, DOWN)
        self.play(Write(final_label))
        self.wait(2)
        self.play(FadeOut(title, plane, vec, final_label))

    def _alibi_part(self):
        title = self._title("ALiBi: a distance penalty added to attention scores")

        alibi = AlibiBias(n_heads=2)
        seq_len = 4
        with torch.no_grad():
            bias = alibi.attention_bias(seq_len)
        head0_row = [round(v, 4) for v in bias[0, seq_len - 1, :].tolist()]

        bars = self._manual_bars(head0_row, ["j=0", "j=1", "j=2", "j=3"], "#FC6255", unit=8.0)
        label = Text(
            "head 0 bias for query i=3 (more negative = further in the past)",
            font_size=22,
        ).next_to(bars, DOWN, buff=0.6)

        self.play(Create(bars), Write(label))
        self.wait(2.5)
        self.play(FadeOut(title, bars, label))
