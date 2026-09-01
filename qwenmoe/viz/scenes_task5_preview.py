"""Manim preview for Task 5's math (MoE router/top-k gating, dispatch +
combine, and the load-balancing auxiliary loss), animated from the real,
already-tested `qwenmoe.model.moe` classes rather than fabricated numbers
(Requirement 7.1's spirit) — built ahead of the formal TraceBundle-based
pipeline (tasks 12-13), same approach as
`scenes_task3_preview.py::Task3PositionalMathScene`.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from manim import *

from qwenmoe.model.moe import AuxLossBalancer, Router, select_top_k


class Task5MoEMathScene(Scene):
    def construct(self):
        self._router_and_topk_part()
        self._dispatch_part()
        self._aux_loss_part()

    def _title(self, text):
        title = Text(text, font_size=32)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(0.5)
        return title

    def _manual_bars(self, values, names, color, unit=1.0, bar_width=0.7, gap=0.35):
        """Same hand-built bar chart as the Task 3 preview: Manim's own
        `BarChart`/`Axes` default to LaTeX-rendered labels, which fails on
        a machine with no LaTeX install. `Text` (Pango-based) sidesteps
        that entirely."""
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

    def _make_router_example(self):
        """The exact worked example from journal entry 005: a real
        `Router`, hand-set so a one-hot input reproduces
        logits=[2.0, 0.5, -1.0, 1.0], then routed through the real
        `select_top_k` (top_k=2)."""
        router = Router(d_model=4, n_routed_experts=4)
        with torch.no_grad():
            router.gate.weight.zero_()
            router.gate.weight[:, 0] = torch.tensor([2.0, 0.5, -1.0, 1.0])
        x = torch.tensor([1.0, 0.0, 0.0, 0.0])
        with torch.no_grad():
            logits = router(x)
            probs = F.softmax(logits, dim=-1)
            topk_indices, topk_weights = select_top_k(logits, probs, top_k=2)
        return logits, probs, topk_indices, topk_weights

    def _router_and_topk_part(self):
        title = self._title("Router: logits -> softmax -> top-k -> renormalize")
        logits, probs, topk_indices, topk_weights = self._make_router_example()
        names = ["e0", "e1", "e2", "e3"]

        logit_bars = self._manual_bars(logits.tolist(), names, "#58C4DD", unit=0.7)
        logit_label = Text("router_logits = Router(x)", font_size=22).next_to(
            logit_bars, DOWN, buff=0.5
        )
        self.play(Create(logit_bars), Write(logit_label))
        self.wait(1.5)
        self.play(FadeOut(logit_bars, logit_label))

        prob_vals = [round(v, 4) for v in probs.tolist()]
        prob_bars = self._manual_bars(prob_vals, names, "#83C167", unit=3.0)
        prob_label = Text("softmax(logits) -- sums to 1", font_size=22).next_to(
            prob_bars, DOWN, buff=0.5
        )
        self.play(Create(prob_bars), Write(prob_label))
        self.wait(1.5)

        selected = topk_indices.tolist()
        rects = [prob_bars[1 + 3 * i] for i in range(4)]
        highlight_anims = [
            rect.animate.set_fill(YELLOW, opacity=1.0)
            if i in selected
            else rect.animate.set_fill(GRAY, opacity=0.25)
            for i, rect in enumerate(rects)
        ]
        self.play(*highlight_anims)
        selected_label = Text(f"top-2 selected: experts {selected}", font_size=22).next_to(
            prob_label, DOWN, buff=0.3
        )
        self.play(Write(selected_label))
        self.wait(1.5)
        self.play(FadeOut(prob_bars, prob_label, selected_label))

        renorm_names = [f"e{i}" for i in selected]
        renorm_vals = [round(v, 4) for v in topk_weights.tolist()]
        renorm_bars = self._manual_bars(renorm_vals, renorm_names, "#FC6255", unit=3.0)
        renorm_label = Text("renormalized gate weights (sum to 1)", font_size=22).next_to(
            renorm_bars, DOWN, buff=0.5
        )
        self.play(Create(renorm_bars), Write(renorm_label))
        self.wait(2)
        self.play(FadeOut(title, renorm_bars, renorm_label))

    def _dispatch_part(self):
        title = self._title("Dispatch: renormalized routed experts + always-on shared expert")
        _, _, topk_indices, topk_weights = self._make_router_example()
        selected = topk_indices.tolist()
        weights = [round(v, 4) for v in topk_weights.tolist()]

        token_box = Rectangle(width=1.6, height=0.7, color=WHITE).move_to([0, 2.4, 0])
        token_label = Text("token x", font_size=20).move_to(token_box.get_center())

        row2_x = [-4, -2, 0, 2, 4]
        row2_names = ["expert 0", "expert 1", "expert 2", "expert 3", "shared\n(always on)"]
        boxes, labels = [], []
        for i in range(4):
            active = i in selected
            color = YELLOW if active else GRAY
            box = Rectangle(
                width=1.5,
                height=0.8,
                color=color,
                fill_color=color,
                fill_opacity=0.9 if active else 0.2,
            ).move_to([row2_x[i], 0.4, 0])
            label = Text(row2_names[i], font_size=16, color=BLACK if active else WHITE)
            label.move_to(box.get_center())
            boxes.append(box)
            labels.append(label)
        shared_box = Rectangle(
            width=1.5, height=0.8, color=BLUE, fill_color=BLUE, fill_opacity=0.5
        ).move_to([row2_x[4], 0.4, 0])
        shared_label = Text(row2_names[4], font_size=16).move_to(shared_box.get_center())
        boxes.append(shared_box)
        labels.append(shared_label)

        fan_out = [Arrow(token_box.get_bottom(), b.get_top(), buff=0.1, stroke_width=2) for b in boxes]

        combine = Circle(radius=0.4, color=WHITE).move_to([0, -1.6, 0])
        combine_label = Text("+", font_size=28).move_to(combine.get_center())
        output_box = Rectangle(width=1.8, height=0.7, color=WHITE).move_to([0, -2.9, 0])
        output_label = Text("output", font_size=20).move_to(output_box.get_center())

        self.play(Create(token_box), Write(token_label))
        self.play(
            *[Create(b) for b in boxes], *[Write(lbl) for lbl in labels], *[Create(a) for a in fan_out]
        )
        self.wait(1)

        skip_note = Text(
            "experts 1, 2: zero tokens this batch -> skipped, no gradient (Req 2.5)",
            font_size=18,
            color=GRAY,
        ).to_edge(DOWN, buff=1.2)
        self.play(Write(skip_note))
        self.wait(1)

        converge = []
        for idx, w in zip(selected, weights):
            arrow = Arrow(boxes[idx].get_bottom(), combine.get_top(), buff=0.1, color=YELLOW)
            w_label = Text(f"w={w}", font_size=16, color=YELLOW).next_to(arrow, LEFT, buff=0.05)
            converge.append(VGroup(arrow, w_label))
        shared_arrow = Arrow(shared_box.get_bottom(), combine.get_top(), buff=0.1, color=BLUE)
        shared_w_label = Text("unweighted", font_size=16, color=BLUE).next_to(
            shared_arrow, RIGHT, buff=0.05
        )
        self.play(Create(combine), Write(combine_label))
        self.play(*[Create(c) for c in converge], Create(shared_arrow), Write(shared_w_label))
        self.wait(1)

        final_arrow = Arrow(combine.get_bottom(), output_box.get_top(), buff=0.1)
        self.play(Create(final_arrow), Create(output_box), Write(output_label))
        self.wait(2)
        self.play(
            FadeOut(
                title,
                token_box,
                token_label,
                *boxes,
                *labels,
                *fan_out,
                skip_note,
                combine,
                combine_label,
                *converge,
                shared_arrow,
                shared_w_label,
                final_arrow,
                output_box,
                output_label,
            )
        )

    def _aux_loss_part(self):
        title = self._title("Load-balancing auxiliary loss: uniform vs. this example's routing")
        formula = Text(
            "aux_loss = alpha * N * sum_i(f_i * P_i)   (alpha=0.01, N=4 experts)", font_size=22
        ).next_to(title, DOWN, buff=0.4)
        self.play(Write(formula))

        balancer = AuxLossBalancer(n_routed_experts=4)

        imbalanced_probs = torch.tensor(
            [
                [0.7, 0.1, 0.1, 0.1],
                [0.6, 0.2, 0.1, 0.1],
                [0.1, 0.6, 0.2, 0.1],
                [0.2, 0.1, 0.6, 0.1],
            ]
        )
        imbalanced_indices = imbalanced_probs.argmax(dim=-1, keepdim=True)
        with torch.no_grad():
            imbalanced_loss = balancer.aux_loss(imbalanced_probs, imbalanced_indices).item()

        uniform_probs = torch.full((4, 4), 0.25)
        uniform_indices = torch.arange(4).remainder(4).unsqueeze(-1)
        with torch.no_grad():
            uniform_loss = balancer.aux_loss(uniform_probs, uniform_indices).item()

        vals = [round(uniform_loss, 4), round(imbalanced_loss, 4)]
        bars = self._manual_bars(vals, ["uniform", "example"], "#FC6255", unit=100.0, gap=1.1)
        label = Text(
            "uniform routing lands exactly at alpha; this example's mild "
            "imbalance pushes the loss above it",
            font_size=18,
        ).next_to(bars, DOWN, buff=0.6)

        self.play(Create(bars), Write(label))
        self.wait(2.5)
        self.play(FadeOut(title, formula, bars, label))
