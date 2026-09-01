"""Task 13.4 — RMSNorm/SwiGLU scene (Requirement 7.5): the normalization
scaling step and the gated SwiGLU activation inside one real routed
expert. Needs the model itself (not just a `TraceBundle`), since
`LayerTrace` doesn't carry the per-block RMSNorm outputs or the
intermediate gate/up/down splits inside an expert -- this scene
recomputes those by calling the real submodules directly on real
(reconstructed-exactly, verified below) intermediate tensors. See
journey_of_a_model/003-norm-and-positional.md (RMSNorm) and
journey_of_a_model/005-moe-layer.md (SwiGLU) for the full derivations;
this scene only animates them with real captured/recomputed data.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from manim import *

from qwenmoe.model.transformer import QwenMoEModel
from qwenmoe.tokenizer.base import build_tokenizer_from_meta
from qwenmoe.train.trainer import load_checkpoint

_CHECKPOINT_PATH = Path("runs/smoke_test/checkpoints/step_1000.pt")
_PROMPT = "ROMEO:"
_LAYER = 0
_DIMS_SHOWN = 8


def _auto_unit(values, target=1.3):
    peak = max((abs(v) for v in values), default=1.0)
    return target / peak if peak > 1e-9 else 1.0


def _load(checkpoint_path: Path, prompt: str):
    """Same pattern as `scenes_embedding.py::_load` -- returns the model
    too, since this scene calls real submodules (`attn_norm`, `moe_norm`,
    an expert's `w_gate`/`w_up`/`w_down`) directly."""
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
    return model, bundle


class NormActivationScene(Scene):
    """Requirement 7.5: RMSNorm's scaling step, and the gated SwiGLU
    activation inside one real dispatched expert."""

    def construct(self):
        model, bundle = _load(_CHECKPOINT_PATH, _PROMPT)
        self._rmsnorm_part(model, bundle)
        self._swiglu_part(model, bundle)

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
            val_label = Text(label, font_size=14).next_to(rect, UP if h >= 0 else DOWN, buff=0.1)
            group.add(rect, name_label, val_label)
        return group

    def _bar_step(self, values, color, y, caption, caption_color=WHITE):
        shown = values[:_DIMS_SHOWN]
        bars = self._manual_bars(
            shown, [f"d{i}" for i in range(_DIMS_SHOWN)], [color] * _DIMS_SHOWN,
            [f"{v:.3f}" for v in shown], unit=_auto_unit(shown), bar_width=0.75, gap=0.35,
        ).move_to([0, y, 0])
        label = Text(caption, font_size=17, color=caption_color).next_to(bars, UP, buff=0.25)
        self.play(Create(bars), Write(label))
        return VGroup(bars, label)

    # ---- Part 1: RMSNorm's real scaling step (Requirement 7.5) ----

    def _rmsnorm_part(self, model, bundle):
        title = self._title("RMSNorm: rescaling a real residual-stream vector to unit RMS")
        pos = len(bundle.token_ids) - 1
        x = bundle.embeddings[pos]
        d_model = x.shape[0]
        weight = model.blocks[0].attn_norm.weight.detach()

        rms = x.pow(2).mean().sqrt().item()
        x_normed = x / rms
        output = x_normed * weight
        normed_rms = x_normed.pow(2).mean().sqrt().item()

        setup = Text(
            f"real embedding for token position {pos} (d_model={d_model}) -> "
            "block 0's attn_norm -- attn_norm.weight is a real learned parameter",
            font_size=16,
        ).move_to([0, 3.3, 0])
        self.play(Write(setup))

        group1 = self._bar_step(x.tolist(), BLUE, 1.3, f"x  (first {_DIMS_SHOWN} of {d_model} dims)")
        self.wait(1)
        rms_text = Text(
            f"RMS(x) = sqrt(mean(x^2)) = {rms:.5f}  (over all {d_model} real dims)", font_size=18
        ).move_to([0, -0.6, 0])
        self.play(Write(rms_text))
        self.wait(1.5)
        self.play(FadeOut(group1))

        group2 = self._bar_step(
            x_normed.tolist(), "#83C167", 1.3, f"x_normed = x / RMS(x)  -- RMS(x_normed) = {normed_rms:.4f}"
        )
        self.wait(1.5)
        self.play(FadeOut(group2, rms_text))

        group3 = self._bar_step(
            output.tolist(), ORANGE, 1.3, "output = x_normed * attn_norm.weight (real learned scale)"
        )
        self.wait(2)

        caption = Text(
            "full derivation in journey_of_a_model/003-norm-and-positional.md",
            font_size=14, color=GRAY,
        ).move_to([0, -3.2, 0])
        self.play(Write(caption))
        self.wait(2)
        self.play(FadeOut(title, setup, group3, caption))

    # ---- Part 2: the gated SwiGLU activation inside a real expert (Requirement 7.5) ----

    def _swiglu_part(self, model, bundle):
        title = self._title("SwiGLU inside a real routed expert: down(silu(gate(x)) * up(x))")
        layer = bundle.layers[_LAYER]

        # Reconstructed exactly (bit-for-bit, verified) from already-captured tensors:
        # TransformerBlock.forward computes x_mid = x + attn(attn_norm(x)) for block 0,
        # where x is bundle.embeddings -- no separate model forward pass needed here.
        x_mid = (bundle.embeddings + layer.attn_output).unsqueeze(0)
        with torch.no_grad():
            normed = model.blocks[_LAYER].moe_norm(x_mid)

        # Pick whichever routed expert this real batch dispatched the fewest tokens to,
        # so the walkthrough below has exactly one token to show (Requirement 2.5's
        # zero-token skip means not every expert id is guaranteed to appear at all).
        expert_id = min(layer.expert_outputs, key=lambda e: layer.expert_outputs[e].shape[0])
        token_pos = (layer.topk_indices == expert_id).nonzero(as_tuple=True)[0][0].item()
        expert = model.blocks[_LAYER].moe.routed_experts[expert_id]

        with torch.no_grad():
            x_tok = normed[0, token_pos]
            gate = expert.w_gate(x_tok)
            gate_act = F.silu(gate)
            up = expert.w_up(x_tok)
            gated = gate_act * up
            down = expert.w_down(gated)

        d_ff = gate.shape[0]
        matches = torch.allclose(down, layer.expert_outputs[expert_id][0], atol=1e-5)

        setup = Text(
            f"token position {token_pos} -- one of the {layer.topk_indices.shape[-1]} experts this token "
            f"selected was expert {expert_id} (moe_norm output, real d_ff={d_ff})",
            font_size=16,
        ).move_to([0, 3.35, 0])
        self.play(Write(setup))

        # Two rows max on screen at once (each bar group can stand up to ~1.3 units tall
        # off its own baseline) -- sequential reveal instead of cramming all 4 steps in
        # together, so no row's bars ever overlap another row's labels.
        top_y, bottom_y = 1.7, -1.7

        g1 = self._manual_bars(
            gate[:_DIMS_SHOWN].tolist(), [f"d{i}" for i in range(_DIMS_SHOWN)], [BLUE] * _DIMS_SHOWN,
            [f"{v:.3f}" for v in gate[:_DIMS_SHOWN].tolist()], unit=_auto_unit(gate[:_DIMS_SHOWN].tolist()),
            bar_width=0.5, gap=0.25,
        ).move_to([-3.4, top_y, 0])
        g1_label = Text("gate = w_gate(x)", font_size=15, color=BLUE).next_to(g1, UP, buff=0.2)

        g2 = self._manual_bars(
            gate_act[:_DIMS_SHOWN].tolist(), [f"d{i}" for i in range(_DIMS_SHOWN)], [GREEN] * _DIMS_SHOWN,
            [f"{v:.3f}" for v in gate_act[:_DIMS_SHOWN].tolist()],
            unit=_auto_unit(gate_act[:_DIMS_SHOWN].tolist()), bar_width=0.5, gap=0.25,
        ).move_to([3.4, top_y, 0])
        g2_label = Text("silu(gate)", font_size=15, color=GREEN).next_to(g2, UP, buff=0.2)
        self.play(Create(g1), Write(g1_label), Create(g2), Write(g2_label))
        self.wait(1.2)

        g3 = self._manual_bars(
            up[:_DIMS_SHOWN].tolist(), [f"d{i}" for i in range(_DIMS_SHOWN)], [BLUE] * _DIMS_SHOWN,
            [f"{v:.3f}" for v in up[:_DIMS_SHOWN].tolist()], unit=_auto_unit(up[:_DIMS_SHOWN].tolist()),
            bar_width=0.5, gap=0.25,
        ).move_to([-3.4, bottom_y, 0])
        g3_label = Text("up = w_up(x)", font_size=15, color=BLUE).next_to(g3, UP, buff=0.2)
        self.play(Create(g3), Write(g3_label))
        self.wait(1.5)
        self.play(FadeOut(g1, g1_label, g2, g2_label, g3, g3_label))

        g4 = self._manual_bars(
            gated[:_DIMS_SHOWN].tolist(), [f"d{i}" for i in range(_DIMS_SHOWN)], [ORANGE] * _DIMS_SHOWN,
            [f"{v:.3f}" for v in gated[:_DIMS_SHOWN].tolist()], unit=_auto_unit(gated[:_DIMS_SHOWN].tolist()),
            bar_width=0.55, gap=0.3,
        ).move_to([0, 0.3, 0])
        g4_label = Text(
            f"gated = silu(gate) * up  (first {_DIMS_SHOWN} of {d_ff} d_ff)", font_size=16, color=ORANGE
        ).next_to(g4, UP, buff=0.3)
        self.play(Create(g4), Write(g4_label))
        self.wait(1.8)
        self.play(FadeOut(g4, g4_label))

        d_model = down.shape[0]
        g5 = self._manual_bars(
            down[:_DIMS_SHOWN].tolist(), [f"d{i}" for i in range(_DIMS_SHOWN)], ["#83C167"] * _DIMS_SHOWN,
            [f"{v:.3f}" for v in down[:_DIMS_SHOWN].tolist()], unit=_auto_unit(down[:_DIMS_SHOWN].tolist()),
            bar_width=0.55, gap=0.3,
        ).move_to([0, 0.3, 0])
        g5_label = Text(
            f"down = w_down(gated)  (first {_DIMS_SHOWN} of {d_model})", font_size=16, color="#83C167"
        ).next_to(g5, UP, buff=0.3)
        self.play(Create(g5), Write(g5_label))
        self.wait(1.8)

        check = "matches" if matches else "does NOT match"
        caption = Text(
            f"this real 'down' output {check} LayerTrace.expert_outputs[{expert_id}] exactly -- "
            "same real computation, just with the gate/up/down steps exposed",
            font_size=14, color=GRAY,
        ).move_to([0, -3.3, 0])
        self.play(Write(caption))
        self.wait(2.5)
        self.play(FadeOut(title, setup, g5, g5_label, caption))
