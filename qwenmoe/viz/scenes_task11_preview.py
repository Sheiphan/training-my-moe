"""Manim preview for Task 11's math (`generate.py`'s temperature scaling,
top-k/nucleus logit filtering, and the real autoregressive decode loop),
animated by calling the real, already-tested `qwenmoe.model.generate`
functions live rather than fabricated numbers (Requirement 7.1's spirit).
Unlike tasks 3/5/6/7/8's previews (built *ahead of* the formal Task 13
architecture scenes) or tasks 9/10's previews (which partially overlap a
later formal task's territory), nothing in the formal task list ever
visualizes generation/sampling specifically -- tasks 12-13 cover one
forward pass's internals (embeddings, attention, MoE routing, norms,
output head) and task 14 covers training-run logs, but none of them touch
`generate.py` at all. So this preview is the only visualization this
content will ever get, not a placeholder ahead of anything. See
`011-inference-generation.md` for the full concept explainers/derivations
this scene only illustrates.
"""
from __future__ import annotations

from pathlib import Path

import torch
from manim import *

from qwenmoe.model.generate import filter_top_k, filter_top_p, generate
from qwenmoe.model.transformer import QwenMoEModel
from qwenmoe.tokenizer.char import CharTokenizer
from qwenmoe.train.trainer import load_checkpoint

_CHECKPOINT_PATH = Path("runs/smoke_test/checkpoints/step_1000.pt")


class Task11GenerationScene(Scene):
    def construct(self):
        self._temperature_part()
        self._top_k_part()
        self._top_p_part()
        self._real_generation_part()

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
            name_label = Text(name, font_size=15).next_to([x, 0, 0], DOWN, buff=0.15)
            val_label = Text(label, font_size=15).next_to(rect, UP, buff=0.1)
            group.add(rect, name_label, val_label)
        return group

    # ---- Part 1: temperature scaling reshapes a real softmax output ----

    def _temperature_part(self):
        title = self._title("Temperature scaling: softmax(z / T) for real logits (Requirement 6.2)")

        z = torch.tensor([2.0, 1.0, 0.0])
        setup = Text("logits z = [2.0, 1.0, 0.0]  --  same logits, three temperatures", font_size=18)
        setup.move_to([0, 3.2, 0])
        self.play(Write(setup))

        temperatures = [0.5, 1.0, 2.0]
        colors = [RED, BLUE, GREEN]
        x_positions = [-3.8, 0, 3.8]
        groups = VGroup()
        for T, color, x in zip(temperatures, colors, x_positions):
            probs = torch.softmax(z / T, dim=-1)  # real computation, not fabricated
            bars = self._manual_bars(
                probs.tolist(), ["z=2", "z=1", "z=0"], [color, color, color],
                [f"{p:.3f}" for p in probs.tolist()], unit=2.0, bar_width=0.55, gap=0.35,
            )
            bars.move_to([x, -0.3, 0])
            caption = Text(f"T = {T}", font_size=20, color=color).next_to(bars, UP, buff=0.35)
            groups.add(bars, caption)
            self.play(Create(bars), Write(caption))
        self.wait(1.5)

        caption = Text(
            "softmax(z/T)_i = e^(z_i/T) / sum_j e^(z_j/T)  --  lower T sharpens toward\n"
            "argmax (T->0 gives a one-hot, exactly greedy); higher T flattens toward uniform",
            font_size=15, color=GRAY,
        ).move_to([0, -2.6, 0])
        self.play(Write(caption))
        self.wait(2.5)
        self.play(FadeOut(title, setup, groups, caption))

    # ---- Part 2: filter_top_k, the real function, live ----

    def _top_k_part(self):
        title = self._title("filter_top_k: keep only the k highest logits (real function call)")

        logits = torch.tensor([[2.0, 0.5, 3.0, -1.0]])
        k = 2
        setup = Text(f"logits = {logits.tolist()[0]},  k = {k}", font_size=18).move_to([0, 3.2, 0])
        self.play(Write(setup))

        # Side-by-side (not stacked), same layout as the temperature/top_p parts --
        # avoids the vertical-crowding overlap a stacked before/after layout ran into.
        before_bars = self._manual_bars(
            logits[0].tolist(), ["idx0", "idx1", "idx2", "idx3"], [BLUE] * 4,
            [f"{v:.1f}" for v in logits[0].tolist()], unit=0.5, bar_width=0.6, gap=0.35,
        )
        before_bars.move_to([-3.2, -0.5, 0])
        before_caption = Text("before", font_size=18, color=BLUE).next_to(before_bars, UP, buff=0.35)
        self.play(Create(before_bars), Write(before_caption))
        self.wait(1)

        filtered = filter_top_k(logits, k=k)  # real function, not reimplemented
        kept_mask = torch.isfinite(filtered[0])
        after_colors = [GREEN if keep else RED for keep in kept_mask.tolist()]
        after_labels = [
            f"{v:.1f}" if keep else "-inf" for v, keep in zip(filtered[0].tolist(), kept_mask.tolist())
        ]
        after_heights = [max(v, 0.0) if keep else 0.05 for v, keep in zip(logits[0].tolist(), kept_mask.tolist())]
        after_bars = self._manual_bars(
            after_heights, ["idx0", "idx1", "idx2", "idx3"], after_colors, after_labels,
            unit=0.5, bar_width=0.6, gap=0.35,
        )
        after_bars.move_to([3.2, -0.5, 0])
        after_caption = Text(
            "after filter_top_k(logits, k=2)", font_size=18, color=GREEN
        ).next_to(after_bars, UP, buff=0.35)
        arrow = Arrow(before_bars.get_right(), after_bars.get_left(), buff=0.3)
        self.play(Create(arrow), Create(after_bars), Write(after_caption))
        self.wait(1.5)

        caption = Text(
            "idx1 and idx3 fall outside the top-2 -> set to -inf -> softmax gives them\n"
            "exactly 0 probability (entry 004's softmax(-inf)=0 property, reused directly)",
            font_size=15, color=GRAY,
        ).move_to([0, -2.6, 0])
        self.play(Write(caption))
        self.wait(2.5)
        self.play(FadeOut(title, setup, before_bars, before_caption, arrow, after_bars, after_caption, caption))

    # ---- Part 3: filter_top_p (nucleus sampling), the real function, live ----

    def _top_p_part(self):
        title = self._title("filter_top_p (nucleus): smallest prefix with cumulative prob >= p")

        probs = torch.tensor([[0.5, 0.2, 0.15, 0.1, 0.05]])
        logits = torch.log(probs)  # softmax(ln p) = p exactly, since p already sums to 1
        setup = Text(
            "probabilities (already sorted) = [0.5, 0.2, 0.15, 0.1, 0.05], logits = ln(p)",
            font_size=17,
        ).move_to([0, 3.2, 0])
        self.play(Write(setup))

        names = ["0.5", "0.2", "0.15", "0.1", "0.05"]
        x_positions = [-3.0, 3.0]
        top_ps = [0.7, 0.71]
        groups = VGroup()
        for top_p, x in zip(top_ps, x_positions):
            filtered = filter_top_p(logits, top_p=top_p)  # real function, not reimplemented
            kept_mask = torch.isfinite(filtered[0])
            colors = [GREEN if keep else RED for keep in kept_mask.tolist()]
            bars = self._manual_bars(
                probs[0].tolist(), names, colors,
                ["kept" if keep else "cut" for keep in kept_mask.tolist()],
                unit=2.6, bar_width=0.5, gap=0.25,
            )
            bars.move_to([x, -0.4, 0])
            n_kept = int(kept_mask.sum().item())
            mass = sum(v for v, keep in zip(probs[0].tolist(), kept_mask.tolist()) if keep)
            caption = Text(
                f"top_p = {top_p}\nkeeps {n_kept} tokens, mass={mass:.2f}", font_size=17
            ).next_to(bars, UP, buff=0.35)
            groups.add(bars, caption)
            self.play(Create(bars), Write(caption))
            self.wait(0.5)
        self.wait(1)

        caption = Text(
            "{0.5, 0.2} already sums to exactly 0.7 -- at p=0.7 that's sufficient, but at\n"
            "p=0.71 it falls just short (0.70 < 0.71), so the 3rd token (0.15) is also kept",
            font_size=15, color=GRAY,
        ).move_to([0, -3.2, 0])
        self.play(Write(caption))
        self.wait(2.5)
        self.play(FadeOut(title, setup, groups, caption))

    # ---- Part 4: real generate() on the real task-10 checkpoint ----

    def _real_generation_part(self):
        title = self._title("Real generate(): greedy degenerates, nucleus sampling doesn't")

        if not _CHECKPOINT_PATH.exists():
            raise FileNotFoundError(
                f"{_CHECKPOINT_PATH} not found -- regenerate it first via the exact commands in "
                "journey_of_a_model/010-training-loop.md's Outcome section (prepare_data.py "
                "--corpus tiny_shakespeare --tokenizer-type char, then train.py --preset "
                "SMOKE_TEST --corpus tiny_shakespeare --tokenizer-type char --total-tokens "
                "2048000 --output-dir runs/smoke_test)"
            )

        checkpoint = load_checkpoint(_CHECKPOINT_PATH)
        model = QwenMoEModel(checkpoint["model_config"])
        model.load_state_dict(checkpoint["model_state_dict"])
        tokenizer = CharTokenizer()
        tokenizer.load(checkpoint["tokenizer_meta"]["tokenizer_path"])

        setup = Text(
            f"real checkpoint: {_CHECKPOINT_PATH} (step {checkpoint['step']}, "
            "tiny-Shakespeare char tokenizer)  --  prompt = \"ROMEO:\"",
            font_size=15,
        ).move_to([0, 3.3, 0])
        self.play(Write(setup))

        torch.manual_seed(1337)
        greedy_text = generate(model, tokenizer, "ROMEO:", max_new_tokens=60, temperature=0.0)
        torch.manual_seed(7)
        nucleus_text = generate(
            model, tokenizer, "ROMEO:", max_new_tokens=60, temperature=1.0, top_p=0.9
        )

        greedy_label = Text("temperature=0.0 (greedy)", font_size=18, color=RED).move_to([0, 1.9, 0])
        greedy_box = Text(
            greedy_text, font_size=17, color=WHITE, t2c={"The": RED, "the": RED}
        ).move_to([0, 1.1, 0])
        self.play(Write(greedy_label))
        self.play(Write(greedy_box))
        self.wait(2)

        greedy_caption = Text(
            "stuck in a repetition loop -- \"neural text degeneration\" (Holtzman et al. 2019)",
            font_size=15, color=RED,
        ).next_to(greedy_box, DOWN, buff=0.25)
        self.play(Write(greedy_caption))
        self.wait(1.5)

        nucleus_label = Text(
            "temperature=1.0, top_p=0.9 (nucleus)", font_size=18, color=GREEN
        ).move_to([0, -1.3, 0])
        nucleus_box = Text(nucleus_text, font_size=17, color=WHITE).move_to([0, -2.1, 0])
        self.play(Write(nucleus_label))
        self.play(Write(nucleus_box))
        self.wait(2)

        nucleus_caption = Text(
            "no repetition loop -- same checkpoint, same prompt, different decoding rule",
            font_size=15, color=GREEN,
        ).next_to(nucleus_box, DOWN, buff=0.25)
        self.play(Write(nucleus_caption))
        self.wait(2.5)
        self.play(FadeOut(
            title, setup, greedy_label, greedy_box, greedy_caption,
            nucleus_label, nucleus_box, nucleus_caption,
        ))
