"""Task 13.1 — embedding lookup + RoPE rotation scene (Requirement 7.2).

Unlike `scenes_task3_preview.py` (built ahead of the formal pipeline, using
tiny hand-built modules), this is the formal scene: it sources every number
from `capture_trace`/`forward_with_trace` on the real
`runs/smoke_test/checkpoints/step_1000.pt` checkpoint (Requirement 7.1). See
journey_of_a_model/003-norm-and-positional.md for RoPE's full derivation;
this scene only animates it with real captured data.
"""
from __future__ import annotations

from pathlib import Path

import torch
from manim import *

from qwenmoe.model.transformer import QwenMoEModel
from qwenmoe.tokenizer.base import build_tokenizer_from_meta
from qwenmoe.train.trainer import load_checkpoint

_CHECKPOINT_PATH = Path("runs/smoke_test/checkpoints/step_1000.pt")
_PROMPT = "ROMEO:"
_DIMS_SHOWN = 8


def _char_label(ch: str) -> str:
    """Manim `Text` renders a literal newline as an actual line break, and
    a bare space is invisible -- both need a visible stand-in label."""
    if ch == "\n":
        return "\\n"
    if ch == " ":
        return "<space>"
    return ch


def _auto_unit(values, target=1.3):
    """Picks a bar-height multiplier so the largest real value in `values`
    plots at `target` units tall, regardless of the underlying magnitude
    (embeddings here are ~0.1-scale; other scenes' values are not)."""
    peak = max((abs(v) for v in values), default=1.0)
    return target / peak if peak > 1e-9 else 1.0


def _load(checkpoint_path: Path, prompt: str):
    """Loads the real trained checkpoint and runs a real forward pass with
    trace capture (Requirement 7.1) -- mirrors `viz/common.py::capture_trace`
    but also returns the model and tokenizer, since this scene needs
    `model.embedding.weight` directly (to show the lookup-table framing
    `TraceBundle` alone doesn't carry) and the tokenizer (to label tokens
    with their real characters).
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"{checkpoint_path} not found -- regenerate it first via the exact commands in "
            "journey_of_a_model/010-training-loop.md's Outcome section (prepare_data.py "
            "--corpus tiny_shakespeare --tokenizer-type char, then train.py --preset "
            "SMOKE_TEST --corpus tiny_shakespeare --tokenizer-type char --total-tokens "
            "2048000 --output-dir runs/smoke_test)"
        )
    checkpoint = load_checkpoint(checkpoint_path)
    model = QwenMoEModel(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    tokenizer = build_tokenizer_from_meta(checkpoint["tokenizer_meta"])
    token_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([token_ids], dtype=torch.long)
    with torch.no_grad():
        _, bundle = model.forward_with_trace(input_ids)
    return model, tokenizer, bundle


class EmbeddingPositionalScene(Scene):
    """Requirement 7.2: the token-to-vector embedding lookup, and the RoPE
    rotation applied to a real query vector."""

    def construct(self):
        model, tokenizer, bundle = _load(_CHECKPOINT_PATH, _PROMPT)
        self._lookup_part(model, tokenizer, bundle)
        self._rope_part(model, bundle)

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
            name_label = Text(name, font_size=15).next_to([x, 0, 0], DOWN, buff=0.15)
            val_label = Text(label, font_size=15).next_to(rect, UP if h >= 0 else DOWN, buff=0.1)
            group.add(rect, name_label, val_label)
        return group

    # ---- Part 1: token id -> real embedding-table row (Requirement 7.2) ----

    def _lookup_part(self, model, tokenizer, bundle):
        title = self._title("Embedding lookup: a token id selects a row of the real embedding table")

        setup = Text(
            f'prompt = "{_PROMPT}" -> token_ids = {bundle.token_ids} '
            f"(vocab_size={model.config.vocab_size}, d_model={model.config.d_model})",
            font_size=17,
        ).move_to([0, 3.3, 0])
        self.play(Write(setup))

        # Every distinct id this prompt actually uses -- never the full
        # 65-row table (too many to fit on screen at once).
        seen: list[int] = []
        for tid in bundle.token_ids:
            if tid not in seen:
                seen.append(tid)
        table = VGroup()
        row_boxes = {}
        for i, tid in enumerate(seen):
            y = 1.8 - i * 0.75
            box = Rectangle(width=3.2, height=0.6, color=BLUE).move_to([-4.0, y, 0])
            ch = _char_label(tokenizer.decode([tid]))
            label = Text(f"id {tid}  ('{ch}')", font_size=16).move_to(box.get_center())
            table.add(box, label)
            row_boxes[tid] = box
        self.play(Create(table))
        self.wait(0.5)

        first_id = bundle.token_ids[0]
        arrow = Arrow(row_boxes[first_id].get_right(), [-0.6, row_boxes[first_id].get_y(), 0], buff=0.1)
        vec = model.embedding.weight[first_id].detach()
        shown = vec[:_DIMS_SHOWN].tolist()
        bars = self._manual_bars(
            shown, [f"d{i}" for i in range(_DIMS_SHOWN)], [BLUE] * _DIMS_SHOWN,
            [f"{v:.3f}" for v in shown], unit=_auto_unit(shown), bar_width=0.55, gap=0.3,
        ).move_to([2.9, 0.3, 0])
        vec_label = Text(
            f"model.embedding.weight[{first_id}]  (first {_DIMS_SHOWN} of {model.config.d_model} dims)",
            font_size=15,
        ).next_to(bars, UP, buff=0.3)
        self.play(Create(arrow), Create(bars), Write(vec_label))
        self.wait(1.5)

        caption = Text(
            "Every position in the prompt looks up its own row the same way -- these are "
            "the model's real trained parameters, not fabricated numbers",
            font_size=15, color=GRAY,
        ).move_to([0, -3.1, 0])
        self.play(Write(caption))
        self.wait(2)
        self.play(FadeOut(title, setup, table, arrow, bars, vec_label, caption))

    # ---- Part 2: RoPE rotation of a real query vector (Requirement 7.2) ----

    def _rope_part(self, model, bundle):
        title = self._title("RoPE: rotating a real query vector by its position")

        # head=3, position=1: inv_freq[0] == 1.0 always (theta^0 == 1), so this
        # pair's real angle is exactly `position` radians -- a clean, verifiable
        # example, and this specific (head, position) pair has a clearly visible
        # magnitude (found by scanning the real trace for the most legible pair).
        head, position, dim_a = 3, 1, 0
        layer = bundle.layers[0]
        q_pre = layer.q[head, position, :]
        head_dim = q_pre.shape[0]
        dim_b = dim_a + head_dim // 2

        positional = model.blocks[0].attn.positional
        q_slice = q_pre.view(1, 1, head_dim)
        k_dummy = torch.zeros_like(q_slice)
        with torch.no_grad():
            q_rot, _ = positional.apply_to_qk(q_slice, k_dummy, torch.tensor([position]))
        q_rot = q_rot[0, 0]

        theta = layer.rope_angles_or_alibi_bias[position, dim_a].item()
        q0, q1 = q_pre[dim_a].item(), q_pre[dim_b].item()
        r0, r1 = q_rot[dim_a].item(), q_rot[dim_b].item()

        setup = Text(
            f"head {head}, position {position} (of {len(bundle.token_ids)}), "
            f"dims ({dim_a}, {dim_b}) of {head_dim} -- real pre-RoPE q from the trace",
            font_size=16,
        ).move_to([0, 3.3, 0])
        self.play(Write(setup))

        plane = NumberPlane(x_range=[-4, 4, 1], y_range=[-4, 4, 1]).scale(0.85).shift(DOWN * 0.3)
        origin = plane.c2p(0, 0)
        vec = Arrow(origin, plane.c2p(q0, q1), buff=0, color=BLUE)
        vec_label = Text(f"q pre-RoPE = ({q0:.3f}, {q1:.3f})", font_size=17, color=BLUE).to_edge(
            LEFT, buff=0.6
        ).shift(UP * 2.6)

        self.play(Create(plane))
        self.play(GrowArrow(vec), Write(vec_label))
        self.wait(1)

        self.play(Rotate(vec, angle=theta, about_point=origin), run_time=3)
        final_label = Text(
            f"theta = position * inv_freq[0] = {position} * 1.0 = {theta:.4f} rad -> "
            f"q_rot = ({r0:.3f}, {r1:.3f})",
            font_size=16, color=GREEN,
        ).next_to(plane, DOWN, buff=0.25)
        self.play(Write(final_label))
        self.wait(2.5)

        caption = Text(
            "q_rot = q*cos(theta) + rotate_half(q)*sin(theta) -- computed here by the real, "
            "already-tested RotaryEmbedding.apply_to_qk; full derivation in "
            "journey_of_a_model/003-norm-and-positional.md",
            font_size=13, color=GRAY,
        ).next_to(final_label, DOWN, buff=0.25)
        self.play(Write(caption))
        self.wait(2)
        self.play(FadeOut(title, setup, plane, vec, vec_label, final_label, caption))
