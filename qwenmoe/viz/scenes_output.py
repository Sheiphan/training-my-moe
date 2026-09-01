"""Task 13.5 — output projection scene (Requirement 7.6): the final linear
projection to vocabulary logits, and the softmax producing next-token
probabilities. Sourced entirely from `TraceBundle` fields already
populated by a real `capture_trace` call on the real
`runs/smoke_test/checkpoints/step_1000.pt` checkpoint (Requirement 7.1) --
no extra model load needed (only the tokenizer, to decode predicted ids
into characters). See journey_of_a_model/006-full-model-assembly.md (LM
head) for the full derivation; this scene only animates it with real data.
"""
from __future__ import annotations

from pathlib import Path

import torch
from manim import *

from qwenmoe.tokenizer.base import build_tokenizer_from_meta
from qwenmoe.train.trainer import load_checkpoint
from qwenmoe.viz.common import capture_trace

_CHECKPOINT_PATH = Path("runs/smoke_test/checkpoints/step_1000.pt")
_PROMPT = "ROMEO:"
_DIMS_SHOWN = 8
_TOP_N = 8


def _auto_unit(values, target=1.3):
    peak = max((abs(v) for v in values), default=1.0)
    return target / peak if peak > 1e-9 else 1.0


def _char_label(ch: str) -> str:
    if ch == "\n":
        return "\\n"
    if ch == " ":
        return "<space>"
    return ch


class OutputProjectionScene(Scene):
    """Requirement 7.6: final RMSNorm output -> LM-head logits -> softmax
    next-token probabilities, for the real prompt's last position."""

    def construct(self):
        if not _CHECKPOINT_PATH.exists():
            raise FileNotFoundError(
                f"{_CHECKPOINT_PATH} not found -- regenerate it first via the exact commands in "
                "journey_of_a_model/010-training-loop.md's Outcome section (prepare_data.py "
                "--corpus tiny_shakespeare --tokenizer-type char, then train.py --preset "
                "SMOKE_TEST --corpus tiny_shakespeare --tokenizer-type char --total-tokens "
                "2048000 --output-dir runs/smoke_test)"
            )
        checkpoint = load_checkpoint(_CHECKPOINT_PATH)
        tokenizer = build_tokenizer_from_meta(checkpoint["tokenizer_meta"])
        bundle = capture_trace(_CHECKPOINT_PATH, _PROMPT)
        self._projection_part(tokenizer, bundle)
        self._softmax_part(tokenizer, bundle)

    def _title(self, text):
        title = Text(text, font_size=26)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(0.5)
        return title

    def _manual_bars(self, heights, names, colors, value_labels, unit, bar_width=1.1, gap=0.7):
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

    # ---- Part 1: final_norm_output -> lm_head -> logits (Requirement 7.6) ----

    def _projection_part(self, tokenizer, bundle):
        title = self._title("Output projection: final RMSNorm output -> LM-head logits")
        pos = len(bundle.token_ids) - 1
        d_model = bundle.final_norm_output.shape[-1]
        vocab_size = bundle.logits.shape[-1]

        setup = Text(
            f'prompt = "{_PROMPT}" -> predicting the token after position {pos} '
            f"(d_model={d_model}, vocab_size={vocab_size})",
            font_size=17,
        ).move_to([0, 3.3, 0])
        self.play(Write(setup))

        x = bundle.final_norm_output[pos][:_DIMS_SHOWN].tolist()
        in_bars = self._manual_bars(
            x, [f"d{i}" for i in range(_DIMS_SHOWN)], [BLUE] * _DIMS_SHOWN, [f"{v:.3f}" for v in x],
            unit=_auto_unit(x), bar_width=0.6, gap=0.3,
        ).move_to([-3.2, 0.3, 0])
        in_label = Text(
            f"final_norm_output[{pos}]  (first {_DIMS_SHOWN} of {d_model})", font_size=15, color=BLUE
        ).next_to(in_bars, UP, buff=0.3)
        self.play(Create(in_bars), Write(in_label))
        self.wait(1.2)

        top_vals, top_idx = torch.topk(bundle.logits[pos], _TOP_N)
        names = [_char_label(tokenizer.decode([i.item()])) for i in top_idx]
        vals = top_vals.tolist()
        out_bars = self._manual_bars(
            vals, names, [ORANGE] * _TOP_N, [f"{v:.2f}" for v in vals], unit=_auto_unit(vals),
            bar_width=0.55, gap=0.25,
        ).move_to([3.0, 0.3, 0])
        out_label = Text(
            f"top {_TOP_N} of {vocab_size} logits = lm_head(final_norm_output)", font_size=15, color=ORANGE
        ).next_to(out_bars, UP, buff=0.3)
        arrow = Arrow(in_bars.get_right(), out_bars.get_left(), buff=0.3)
        self.play(Create(arrow), Create(out_bars), Write(out_label))
        self.wait(2)

        caption = Text(
            "lm_head is the (tied) transpose of the embedding table -- same real weight "
            "matrix as the embedding lookup scene, used in the other direction",
            font_size=14, color=GRAY,
        ).move_to([0, -3.1, 0])
        self.play(Write(caption))
        self.wait(2.5)
        self.play(FadeOut(title, setup, in_bars, in_label, arrow, out_bars, out_label, caption))

    # ---- Part 2: softmax -> next-token probabilities (Requirement 7.6) ----

    def _softmax_part(self, tokenizer, bundle):
        title = self._title("Softmax: logits -> a real next-character probability distribution")
        pos = len(bundle.token_ids) - 1
        vocab_size = bundle.logits.shape[-1]

        setup = Text(
            f'prompt = "{_PROMPT}"  --  probs = softmax(logits[{pos}]), over all {vocab_size} '
            "characters in the tiny-Shakespeare vocabulary",
            font_size=17,
        ).move_to([0, 3.3, 0])
        self.play(Write(setup))

        top_vals, top_idx = torch.topk(bundle.probs, _TOP_N)
        names = [_char_label(tokenizer.decode([i.item()])) for i in top_idx]
        vals = top_vals.tolist()
        bars = self._manual_bars(
            vals, names, ["#83C167"] * _TOP_N, [f"{v:.4f}" for v in vals], unit=_auto_unit(vals),
        )
        label = Text(
            f"top {_TOP_N} of {vocab_size} -- full distribution sums to {bundle.probs.sum().item():.4f}",
            font_size=18,
        ).next_to(bars, DOWN, buff=0.5)
        self.play(Create(bars), Write(label))
        self.wait(2)

        top1_char = _char_label(tokenizer.decode([top_idx[0].item()]))
        prediction = Text(
            f'greedy next character: "{top1_char}"  (p={top_vals[0].item():.4f})',
            font_size=20, color=YELLOW,
        ).next_to(label, DOWN, buff=0.4)
        self.play(Write(prediction))
        self.wait(1.5)

        if top_idx[0].item() == tokenizer.encode("\n")[0]:
            caption_text = (
                'a real, if unsurprising, finding: after "ROMEO:" the model has learned that a '
                "newline usually follows -- matching this corpus's \"SPEAKER:\\ndialogue\" format"
            )
        else:
            caption_text = (
                f"real numbers from this checkpoint's actual forward pass, not a fabricated "
                f"or illustrative distribution"
            )
        caption = Text(caption_text, font_size=14, color=GRAY).move_to([0, -3.2, 0])
        self.play(Write(caption))
        self.wait(3)
        self.play(FadeOut(title, setup, bars, label, prediction, caption))
