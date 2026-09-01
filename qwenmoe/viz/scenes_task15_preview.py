"""Manim preview for Task 15's math (`LossFreeBalancer`'s bias-adjusted
routing and `run_comparison`'s compute-matched ablation study), animated
by calling the real, already-tested `qwenmoe.model.moe`/`qwenmoe.train.compare`
code live rather than fabricated numbers -- same convention as every
earlier preview in this series (tasks 3/5/6/7/8/9/10/11). Like those
previews, nothing in the remaining formal task list (16: MLA, 17: LoRA,
18: docs) ever touches `LossFreeBalancer`/`run_comparison` again, so this
is the only visualization this content will ever get. See
`015-research-comparison-framework.md` for the full concept explainers/
derivations this scene only illustrates.
"""
from __future__ import annotations

import tempfile

import torch
from manim import *

from qwenmoe.config import DataConfig, ModelConfig, TrainConfig
from qwenmoe.model.moe import LossFreeBalancer, select_top_k
from qwenmoe.train.compare import run_comparison


class Task15ComparisonScene(Scene):
    def construct(self):
        self._bias_mechanism_part()
        self._real_comparison_part()

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

    # ---- Part 1: LossFreeBalancer's real bias-nudge loop, many rounds ----

    def _bias_mechanism_part(self):
        title = self._title("LossFreeBalancer: a per-expert bias, nudged after every step (Req 12.3)")

        balancer = LossFreeBalancer(n_routed_experts=4, update_rate=0.05)
        logits = torch.tensor([[5.0, 1.0, 1.0, -5.0]])
        probs = torch.softmax(logits, dim=-1)
        setup = Text(
            "logits = [5.0, 1.0, 1.0, -5.0] -- expert 0 wins every round unless something intervenes",
            font_size=17,
        ).move_to([0, 3.2, 0])
        self.play(Write(setup))

        # Real update loop -- calls the real LossFreeBalancer.selection_scores/
        # update, not a reimplementation.
        winners = []
        snapshots = {}
        for round_idx in range(40):
            if round_idx in (0, 10, 39):
                snapshots[round_idx] = balancer.bias.clone().tolist()
            scores = balancer.selection_scores(logits)
            topk_indices, _ = select_top_k(scores, probs, top_k=1)
            winners.append(topk_indices.item())
            counts = torch.bincount(topk_indices.flatten(), minlength=4)
            balancer.update(counts)

        names = ["expert0", "expert1", "expert2", "expert3"]
        colors = [RED, BLUE, GREEN, YELLOW]
        x_positions = [-4.4, 0, 4.4]
        groups = VGroup()
        for round_idx, x in zip((0, 10, 39), x_positions):
            values = snapshots[round_idx]
            bars = self._manual_bars(
                [v + 2.5 for v in values],  # shift so a negative bias still draws a visible bar
                names, colors, [f"{v:+.2f}" for v in values],
                unit=0.6, bar_width=0.55, gap=0.3,
            )
            bars.move_to([x, -0.6, 0])
            caption = Text(f"round {round_idx}", font_size=18).next_to(bars, UP, buff=0.35)
            groups.add(bars, caption)
            self.play(Create(bars), Write(caption))
            self.wait(0.5)
        self.wait(1)

        early_rate = winners[:5].count(0) / 5
        late_rate = winners[-5:].count(0) / 5
        caption = Text(
            f"expert0's win rate: {early_rate:.0%} in rounds 0-4  ->  {late_rate:.0%} in rounds 35-39\n"
            "bars are shifted +2.5 for visibility -- labels show the real (unshifted) bias value",
            font_size=15, color=GRAY,
        ).move_to([0, -3.0, 0])
        self.play(Write(caption))
        self.wait(2.5)
        self.play(FadeOut(title, setup, groups, caption))

    # ---- Part 2: real run_comparison(), aux_loss vs loss_free, real data ----

    def _real_comparison_part(self):
        title = self._title("run_comparison: one fixed slice, one budget, two named variants (Req 12.6-12.7)")

        setup = Text(
            "real tiny_shakespeare corpus, base config d_model=32/n_layers=1 -- "
            "comparing moe_balancing='aux_loss' vs 'loss_free'",
            font_size=16,
        ).move_to([0, 3.2, 0])
        self.play(Write(setup))

        # Real call -- run_comparison itself raises a clear, actionable
        # FileNotFoundError (naming the exact prepare_data.py command) if this
        # corpus hasn't been prepared yet, so no redundant check is needed here.
        data_config = DataConfig(
            corpus="tiny_shakespeare", tokenizer_type="char",
            cache_dir=".cache/data", fixed_slice_tokens=5000, fixed_slice_seed=0,
        )
        base_config = ModelConfig(
            vocab_size=65, d_model=32, n_layers=1, n_heads=2, n_kv_heads=1,
            d_ff=64, n_routed_experts=4, n_shared_experts=1, top_k=2, max_seq_len=32,
        )
        variants = {
            "aux_loss": {"moe_balancing": "aux_loss"},
            "loss_free": {"moe_balancing": "loss_free"},
        }
        train_config = TrainConfig(
            batch_size=8, seq_len=32, total_tokens=30 * 8 * 32, lr=1e-2, warmup_steps=3,
            eval_interval_steps=30, checkpoint_interval_steps=30, log_interval_steps=30,
            device="cpu", seed=0,
        )
        output_dir = tempfile.mkdtemp(prefix="manim_task15_comparison_")
        report = run_comparison(base_config, variants, data_config, train_config, output_dir)

        names = list(report.variants)
        colors = [BLUE, GREEN]
        ppl_values = [report.variants[n].val_perplexity for n in names]
        bars = self._manual_bars(
            ppl_values, names, colors, [f"{v:.2f}" for v in ppl_values],
            unit=0.15, bar_width=1.0, gap=0.8,
        )
        bars.move_to([-3.0, -0.6, 0])
        bars_caption = Text("val_perplexity (lower is better)", font_size=17).next_to(bars, UP, buff=0.35)
        self.play(Create(bars), Write(bars_caption))
        self.wait(1)

        table_lines = [
            f"{name:<10} val_loss={report.variants[name].final_val_loss:.3f}  "
            f"ppl={report.variants[name].val_perplexity:.2f}  "
            f"tok/s={report.variants[name].tokens_per_sec:.0f}"
            for name in names
        ]
        table = Text("\n".join(table_lines), font_size=16).move_to([3.0, -0.6, 0])
        self.play(Write(table))
        self.wait(2)

        caption = Text(
            "identical fixed token slice + identical token budget (Requirements 11.5, 12.6) -- \n"
            "the only difference left between these two bars is moe_balancing itself",
            font_size=15, color=GRAY,
        ).move_to([0, -3.0, 0])
        self.play(Write(caption))
        self.wait(2.5)
        self.play(FadeOut(title, setup, bars, bars_caption, table, caption))
