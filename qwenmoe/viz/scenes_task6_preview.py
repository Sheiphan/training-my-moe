"""Manim preview for Task 6's math (pre-norm residual wiring inside a
`TransformerBlock`, the resulting growth of the residual stream's norm
across a real multi-layer `QwenMoEModel`, and weight tying), animated
from the real, already-tested `qwenmoe.model.block`/`qwenmoe.model.
transformer` classes rather than fabricated numbers (Requirement 7.1's
spirit) — built ahead of the formal TraceBundle-based pipeline (tasks
12-13), same approach as `scenes_task3_preview.py`/`scenes_task5_preview.py`.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from manim import *

from qwenmoe.config import ModelConfig
from qwenmoe.model.block import TransformerBlock
from qwenmoe.model.transformer import QwenMoEModel


class Task6ModelAssemblyScene(Scene):
    def construct(self):
        self._block_wiring_part()
        self._residual_growth_part()
        self._weight_tying_part()

    def _title(self, text):
        title = Text(text, font_size=30)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(0.5)
        return title

    def _manual_bars(self, values, names, color, unit=1.0, bar_width=0.7, gap=0.35):
        """Same hand-built bar chart as the task 3/5 previews: Manim's own
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

    # ---- Part 1: pre-norm residual wiring inside one real TransformerBlock ----

    def _make_block_trace(self):
        """Real `TransformerBlock`, tiny dims, one hand-seeded forward pass
        with every intermediate tensor's norm kept, so both pre-norm
        sublayer stages (Requirement 1.4) can be shown with real numbers."""
        torch.manual_seed(0)
        config = ModelConfig(
            vocab_size=10,
            d_model=4,
            n_heads=2,
            n_kv_heads=2,
            d_ff=8,
            n_routed_experts=2,
            n_shared_experts=1,
            top_k=1,
            max_seq_len=4,
            positional_scheme="nope",
        )
        block = TransformerBlock(config)
        x_in = torch.randn(1, 3, config.d_model)
        positions = torch.arange(3)
        with torch.no_grad():
            normed1 = block.attn_norm(x_in)
            attn_out = block.attn(normed1, positions)
            x_mid = x_in + attn_out
            normed2 = block.moe_norm(x_mid)
            moe_out, _ = block.moe(normed2)
            x_out = x_mid + moe_out
        return {
            "x_in": x_in.norm().item(),
            "normed1": normed1.norm().item(),
            "attn_out": attn_out.norm().item(),
            "x_mid": x_mid.norm().item(),
            "normed2": normed2.norm().item(),
            "moe_out": moe_out.norm().item(),
            "x_out": x_out.norm().item(),
        }

    def _flow_stage(self, left_name, left_norm, sub_name, sub_norm, norm_out, right_name, right_norm):
        """One `x + Sublayer(RMSNorm(x))` stage: boxes + arrows, with the
        residual skip connection drawn as an arc above — annotated with
        real computed norms rather than illustrative placeholders."""
        left = Rectangle(width=2.0, height=0.9, color=WHITE).move_to([-5, 0, 0])
        left_label = Text(left_name, font_size=20).move_to(left.get_center())
        left_norm_label = Text(f"||x||={left_norm:.4g}", font_size=16, color=GRAY).next_to(
            left, DOWN, buff=0.25
        )

        norm_box = Rectangle(width=1.8, height=0.9, color=BLUE).move_to([-2.3, 0, 0])
        norm_label = Text("RMSNorm", font_size=18).move_to(norm_box.get_center())
        norm_out_label = Text(f"||x||={norm_out:.4g}", font_size=16, color=GRAY).next_to(
            norm_box, DOWN, buff=0.25
        )

        sub_box = Rectangle(width=1.8, height=0.9, color=YELLOW).move_to([0.4, 0, 0])
        sub_label = Text(sub_name, font_size=18).move_to(sub_box.get_center())
        sub_norm_label = Text(f"||out||={sub_norm:.4g}", font_size=16, color=GRAY).next_to(
            sub_box, DOWN, buff=0.25
        )

        plus = Circle(radius=0.35, color=WHITE).move_to([2.7, 0, 0])
        plus_label = Text("+", font_size=26).move_to(plus.get_center())

        right = Rectangle(width=2.0, height=0.9, color=WHITE).move_to([5.2, 0, 0])
        right_label = Text(right_name, font_size=20).move_to(right.get_center())
        right_norm_label = Text(f"||x||={right_norm:.4g}", font_size=16, color=GRAY).next_to(
            right, DOWN, buff=0.25
        )

        arrow1 = Arrow(left.get_right(), norm_box.get_left(), buff=0.1)
        arrow2 = Arrow(norm_box.get_right(), sub_box.get_left(), buff=0.1)
        arrow3 = Arrow(sub_box.get_right(), plus.get_left(), buff=0.1)
        arrow4 = Arrow(plus.get_right(), right.get_left(), buff=0.1)

        skip = ArcBetweenPoints(left.get_top(), plus.get_top(), angle=-TAU / 6)
        skip_label = Text("residual (identity)", font_size=16, color=GRAY).next_to(
            skip, UP, buff=0.15
        )

        self.play(Create(left), Write(left_label), Write(left_norm_label))
        self.play(Create(arrow1), Create(norm_box), Write(norm_label), Write(norm_out_label))
        self.play(Create(arrow2), Create(sub_box), Write(sub_label), Write(sub_norm_label))
        self.play(Create(arrow3), Create(plus), Write(plus_label))
        self.play(Create(skip), Write(skip_label))
        self.play(Create(arrow4), Create(right), Write(right_label), Write(right_norm_label))
        self.wait(1.5)
        return VGroup(
            left, left_label, left_norm_label, norm_box, norm_label, norm_out_label, sub_box,
            sub_label, sub_norm_label, plus, plus_label, right, right_label, right_norm_label,
            arrow1, arrow2, arrow3, arrow4, skip, skip_label,
        )

    def _block_wiring_part(self):
        title = self._title("TransformerBlock: pre-norm residual wiring (Requirement 1.4)")
        t = self._make_block_trace()

        group1 = self._flow_stage(
            "x_in", t["x_in"], "Attention", t["attn_out"], t["normed1"], "x_mid", t["x_mid"]
        )
        self.play(FadeOut(group1))

        group2 = self._flow_stage(
            "x_mid", t["x_mid"], "MoE", t["moe_out"], t["normed2"], "x_out", t["x_out"]
        )
        same_scale_note = Text(
            "Both RMSNorm calls output ||x||=3.464 -- same reference scale "
            "every time, regardless of the residual stream's own growing norm",
            font_size=16,
            color=GRAY,
        ).to_edge(DOWN, buff=0.4)
        self.play(Write(same_scale_note))
        self.wait(2)
        self.play(FadeOut(title, group2, same_scale_note))

    # ---- Part 2: the residual stream's norm growing across real depth ----

    def _make_residual_growth_trace(self):
        """Real 6-layer `QwenMoEModel` (matching this project's baseline
        depth). Batch-averaged squared norm of the residual stream after
        the embedding and after every block — the real, empirical
        analogue of the pre-norm E[||x_l||^2] = O(l . d) growth result
        derived in journal entry 006, measured on this project's own code
        rather than asserted from the paper alone."""
        torch.manual_seed(0)
        config = ModelConfig(
            vocab_size=20,
            d_model=8,
            n_layers=6,
            n_heads=2,
            n_kv_heads=2,
            d_ff=16,
            n_routed_experts=2,
            n_shared_experts=1,
            top_k=1,
            max_seq_len=8,
            positional_scheme="nope",
        )
        model = QwenMoEModel(config)
        input_ids = torch.randint(0, config.vocab_size, (32, 4))
        positions = torch.arange(4)
        with torch.no_grad():
            x = model.embedding(input_ids)
            norms = [x.pow(2).sum(dim=-1).mean().item()]
            for block in model.blocks:
                x, _ = block(x, positions)
                norms.append(x.pow(2).sum(dim=-1).mean().item())
        return norms

    def _residual_growth_part(self):
        title = self._title("Pre-norm's residual stream grows with depth (real 6-layer model)")
        norms = self._make_residual_growth_trace()
        names = ["emb"] + [f"blk{i}" for i in range(1, len(norms))]

        bars = self._manual_bars(
            [round(v, 2) for v in norms], names, "#58C4DD", unit=0.18, bar_width=0.55, gap=0.25
        )
        label = Text(
            "mean ||x||^2 over 32 random sequences, measured after each real block",
            font_size=17,
        ).next_to(bars, DOWN, buff=0.5)
        formula = Text(
            "Xiong et al. (2020): E[||x_l||^2] = O(l . d) for pre-norm "
            "-- an upward trend, not a strict step-by-step guarantee",
            font_size=16,
            color=GRAY,
        ).next_to(label, DOWN, buff=0.3)

        self.play(Create(bars), Write(label))
        self.wait(1)
        self.play(Write(formula))
        self.wait(2.5)
        self.play(FadeOut(title, bars, label, formula))

    # ---- Part 3: weight tying, via the real tied lm_head ----

    def _make_tied_lm_head_trace(self):
        """A real tiny `QwenMoEModel`, hand-set embedding rows so this
        reproduces journal entry 006's cat/dog/run worked example exactly
        -- but computed by calling the real, tied `lm_head` directly,
        not by recomputing the dot products by hand again."""
        config = ModelConfig(
            vocab_size=3,
            d_model=2,
            n_layers=1,
            n_heads=1,
            n_kv_heads=1,
            d_ff=4,
            n_routed_experts=1,
            n_shared_experts=1,
            top_k=1,
            max_seq_len=4,
            tie_word_embeddings=True,
        )
        model = QwenMoEModel(config)
        with torch.no_grad():
            model.embedding.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]))
        is_tied = model.lm_head.weight is model.embedding.weight
        x = torch.tensor([0.5, 0.5])
        with torch.no_grad():
            logits = model.lm_head(x)
            probs = F.softmax(logits, dim=-1)
        return is_tied, probs.tolist()

    def _weight_tying_part(self):
        title = self._title("Weight tying: one matrix, two jobs (Requirement 1.7)")
        is_tied, probs = self._make_tied_lm_head_trace()

        shared_box = Rectangle(
            width=3.2, height=1.1, color=BLUE, fill_color=BLUE, fill_opacity=0.25
        ).move_to([0, 1.4, 0])
        shared_label = Text(
            "shared matrix E\ncat=[1,0]  dog=[0,1]  run=[1,1]", font_size=18
        ).move_to(shared_box.get_center())

        emb_box = Rectangle(width=2.4, height=0.8, color=WHITE).move_to([-3.6, -0.6, 0])
        emb_label = Text("embedding lookup", font_size=16).move_to(emb_box.get_center())
        head_box = Rectangle(width=2.4, height=0.8, color=WHITE).move_to([3.6, -0.6, 0])
        head_label = Text("lm_head (output)", font_size=16).move_to(head_box.get_center())

        arrow1 = Arrow(shared_box.get_bottom(), emb_box.get_top(), buff=0.15)
        arrow2 = Arrow(shared_box.get_bottom(), head_box.get_top(), buff=0.15)
        identity_label = Text(
            f"model.lm_head.weight is model.embedding.weight -> {is_tied}",
            font_size=18,
            color=GREEN if is_tied else RED,
        ).next_to(shared_box, UP, buff=0.3)

        self.play(Create(shared_box), Write(shared_label))
        self.play(Write(identity_label))
        self.play(Create(arrow1), Create(emb_box), Write(emb_label))
        self.play(Create(arrow2), Create(head_box), Write(head_label))
        self.wait(1.5)
        self.play(
            FadeOut(
                shared_box, shared_label, emb_box, emb_label, head_box, head_label,
                arrow1, arrow2, identity_label,
            )
        )

        x_label = Text(
            "x = [0.5, 0.5]  ->  model.lm_head(x)  (the real, tied nn.Linear)", font_size=20
        ).next_to(title, DOWN, buff=0.4)
        self.play(Write(x_label))

        prob_vals = [round(p, 4) for p in probs]
        bars = self._manual_bars(prob_vals, ["cat", "dog", "run"], "#83C167", unit=4.0)
        label = Text(
            "softmax(logits) -- 'run' wins: x=[0.5,0.5] looks most like run's own embedding",
            font_size=16,
        ).next_to(bars, DOWN, buff=0.5)
        self.play(Create(bars), Write(label))
        self.wait(2.5)
        self.play(FadeOut(title, x_label, bars, label))
