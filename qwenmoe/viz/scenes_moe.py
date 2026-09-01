"""Task 13.3 — MoE routing scene (Requirement 7.4): router-logit
computation, softmax gate scores, top-k expert selection, and token
dispatch to the chosen experts. Sourced from a real `capture_trace` call
on the real `runs/smoke_test/checkpoints/step_1000.pt` checkpoint
(Requirement 7.1). See journey_of_a_model/005-moe-layer.md for the full
router/dispatch/combine derivation; this scene only animates it with real
data across a real 6-token prompt.
"""
from __future__ import annotations

from pathlib import Path

from manim import *

from qwenmoe.viz.common import capture_trace

_CHECKPOINT_PATH = Path("runs/smoke_test/checkpoints/step_1000.pt")
_PROMPT = "ROMEO:"
_LAYER = 0
_TOKEN_COLORS = [BLUE, GREEN, YELLOW, ORANGE, PURPLE, TEAL]


def _auto_unit(values, target=1.3):
    peak = max((abs(v) for v in values), default=1.0)
    return target / peak if peak > 1e-9 else 1.0


def _require_checkpoint(checkpoint_path: Path) -> None:
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"{checkpoint_path} not found -- regenerate it first via the exact commands in "
            "journey_of_a_model/010-training-loop.md's Outcome section (prepare_data.py "
            "--corpus tiny_shakespeare --tokenizer-type char, then train.py --preset "
            "SMOKE_TEST --corpus tiny_shakespeare --tokenizer-type char --total-tokens "
            "2048000 --output-dir runs/smoke_test)"
        )


class MoERoutingScene(Scene):
    """Requirement 7.4: router logits -> softmax -> top-k -> dispatch,
    using one real trace across a real 6-token prompt."""

    def construct(self):
        _require_checkpoint(_CHECKPOINT_PATH)
        bundle = capture_trace(_CHECKPOINT_PATH, _PROMPT)
        self._router_topk_part(bundle)
        self._dispatch_part(bundle)

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

    # ---- Part 1: router logits -> softmax -> top-k for one real token ----

    def _router_topk_part(self, bundle):
        title = self._title("Router: real logits -> softmax -> top-k -> renormalize")
        layer = bundle.layers[_LAYER]
        pos = len(bundle.token_ids) - 1  # last token (":")
        n_experts = layer.router_logits.shape[-1]
        names = [f"e{i}" for i in range(n_experts)]

        setup = Text(
            f'prompt = "{_PROMPT}"  --  layer {_LAYER}, token position {pos} '
            f"(of {len(bundle.token_ids)}); router_logits[{pos}]",
            font_size=17,
        ).move_to([0, 3.3, 0])
        self.play(Write(setup))

        logits = layer.router_logits[pos].tolist()
        logit_bars = self._manual_bars(
            logits, names, [BLUE] * n_experts, [f"{v:.3f}" for v in logits], unit=_auto_unit(logits),
        )
        logit_label = Text("router_logits = Router(x)", font_size=20).next_to(logit_bars, DOWN, buff=0.5)
        self.play(Create(logit_bars), Write(logit_label))
        self.wait(1.5)
        self.play(FadeOut(logit_bars, logit_label))

        probs = layer.router_probs[pos].tolist()
        prob_bars = self._manual_bars(
            probs, names, [BLUE] * n_experts, [f"{v:.4f}" for v in probs], unit=_auto_unit(probs),
        )
        prob_label = Text(
            f"softmax(logits) -- sums to {sum(probs):.4f}", font_size=20
        ).next_to(prob_bars, DOWN, buff=0.5)
        self.play(Create(prob_bars), Write(prob_label))
        self.wait(1.5)

        selected = layer.topk_indices[pos].tolist()
        rects = [prob_bars[1 + 3 * i] for i in range(n_experts)]
        self.play(*[
            rect.animate.set_fill(YELLOW, opacity=1.0) if i in selected
            else rect.animate.set_fill(GRAY, opacity=0.25)
            for i, rect in enumerate(rects)
        ])
        selected_label = Text(
            f"top-{len(selected)} selected: experts {selected}", font_size=20
        ).next_to(prob_label, DOWN, buff=0.3)
        self.play(Write(selected_label))
        self.wait(1.5)
        self.play(FadeOut(prob_bars, prob_label, selected_label))

        weights = layer.topk_weights[pos].tolist()
        renorm_names = [f"e{i}" for i in selected]
        renorm_bars = self._manual_bars(
            weights, renorm_names, [YELLOW] * len(selected), [f"{v:.4f}" for v in weights],
            unit=_auto_unit(weights),
        )
        renorm_label = Text(
            f"renormalized gate weights -- sum to {sum(weights):.4f}", font_size=20
        ).next_to(renorm_bars, DOWN, buff=0.5)
        self.play(Create(renorm_bars), Write(renorm_label))
        self.wait(2)
        self.play(FadeOut(title, setup, renorm_bars, renorm_label))

    # ---- Part 2: token dispatch across the whole real prompt ----

    def _dispatch_part(self, bundle):
        title = self._title("Token dispatch: every real token's top-k routes to its own experts")
        layer = bundle.layers[_LAYER]
        n_tokens = len(bundle.token_ids)
        n_experts = layer.router_logits.shape[-1]

        token_x = [-5 + 2 * i for i in range(n_tokens)]
        expert_x = [-3 + 2 * i for i in range(n_experts)]

        token_boxes, token_labels = [], []
        for i in range(n_tokens):
            box = Rectangle(width=1.5, height=0.7, color=WHITE).move_to([token_x[i], 2.6, 0])
            lbl = Text(f"tok {i}", font_size=15).move_to(box.get_center())
            token_boxes.append(box)
            token_labels.append(lbl)
        self.play(*[Create(b) for b in token_boxes], *[Write(l) for l in token_labels])

        expert_boxes, expert_labels = [], []
        for e in range(n_experts):
            box = Rectangle(width=1.6, height=0.8, color=GRAY).move_to([expert_x[e], -1.0, 0])
            lbl = Text(f"expert {e}", font_size=16).move_to(box.get_center())
            expert_boxes.append(box)
            expert_labels.append(lbl)
        self.play(*[Create(b) for b in expert_boxes], *[Write(l) for l in expert_labels])
        self.wait(0.5)

        all_arrows = VGroup()
        for i in range(n_tokens):
            color = _TOKEN_COLORS[i % len(_TOKEN_COLORS)]
            arrows_i = VGroup()
            for e in layer.topk_indices[i].tolist():
                arrow = Arrow(
                    token_boxes[i].get_bottom(), expert_boxes[e].get_top(), buff=0.08,
                    stroke_width=2.5, color=color,
                )
                arrows_i.add(arrow)
            self.play(*[Create(a) for a in arrows_i], run_time=0.5)
            all_arrows.add(arrows_i)

        counts = [int((layer.topk_indices == e).sum().item()) for e in range(n_experts)]
        for e, box in enumerate(expert_boxes):
            unit = "token" if counts[e] == 1 else "tokens"
            count_label = Text(f"{counts[e]} {unit}", font_size=15, color=YELLOW).next_to(
                box, DOWN, buff=0.2
            )
            all_arrows.add(count_label)
            self.play(Write(count_label), run_time=0.3)
        self.wait(1)

        imbalance_note = Text(
            f"real imbalance in this under-trained model: expert {counts.index(max(counts))} "
            f"is in every single token's top-{layer.topk_indices.shape[-1]} "
            f"({max(counts)} of {n_tokens} tokens) -- exactly what the load-balancing "
            "auxiliary loss (entry 005) is meant to push back against",
            font_size=15, color=GRAY,
        ).move_to([0, -3.3, 0])
        self.play(Write(imbalance_note))
        self.wait(3)
        self.play(FadeOut(
            title, *token_boxes, *token_labels, *expert_boxes, *expert_labels, all_arrows, imbalance_note
        ))
