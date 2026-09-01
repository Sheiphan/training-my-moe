"""Manim preview for Task 10's math (`Trainer`'s warmup+cosine LR schedule,
AdamW's real per-parameter update, checkpoint save/resume) and its most
important real finding -- the unscaled-tied-embedding logit blowup bug --
animated from the real, already-tested `qwenmoe.train.trainer`/
`qwenmoe.model.transformer` code rather than fabricated numbers
(Requirement 7.1's spirit). Task 14 (training/run observability) will
later render a loss curve and expert-utilization chart from real JSON-lines
*logs* of a completed run; this scene instead animates the training-loop
*mechanics themselves* (the schedule formula, the optimizer's real internal
state, the embedding-init bug, checkpoint/resume) that task 14 doesn't
cover -- same standalone, real-classes-directly approach as
`scenes_task3_preview.py` through `scenes_task9_preview.py`. See
`010-training-loop.md` for the full concept explainers/derivations this
scene only illustrates.
"""
from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from manim import *

from qwenmoe.config import ModelConfig, TrainConfig
from qwenmoe.data.dataset import TokenDataset, train_val_split
from qwenmoe.model.transformer import QwenMoEModel
from qwenmoe.train.trainer import Trainer, lr_multiplier

# Same tiny-fixture spirit as tests/test_trainer_integration.py -- small
# enough to construct/train live inside a Manim scene in well under a second.
_TINY_CONFIG = dict(
    vocab_size=20, d_model=16, n_layers=1, n_heads=2, n_kv_heads=2, d_ff=32,
    n_routed_experts=2, n_shared_experts=1, top_k=1, max_seq_len=16,
)


class Task10TrainingLoopScene(Scene):
    def construct(self):
        self._lr_schedule_part()
        self._embedding_init_bug_part()
        self._adamw_part()
        self._checkpoint_resume_part()

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

    # ---- Part 1: lr_multiplier -- linear warmup, then cosine decay ----

    def _lr_schedule_part(self):
        title = self._title("lr_multiplier: linear warmup, then cosine decay (Requirement 5.1)")

        warmup_steps, total_steps, base_lr = 20, 200, 3e-4
        setup = Text(
            f"warmup_steps={warmup_steps}, total_steps={total_steps}, base_lr={base_lr:.0e}",
            font_size=18,
        ).move_to([0, 3.2, 0])
        self.play(Write(setup))

        x0, x1, y0, y1 = -5.2, 5.2, -1.6, 1.6
        x_axis = Line([x0, y0, 0], [x1, y0, 0], color=GRAY)
        y_axis = Line([x0, y0, 0], [x0, y1, 0], color=GRAY)
        x_label_0 = Text("step 0", font_size=14, color=GRAY).next_to([x0, y0, 0], DOWN, buff=0.15)
        x_label_end = Text(f"step {total_steps}", font_size=14, color=GRAY).next_to(
            [x1, y0, 0], DOWN, buff=0.15
        )
        y_label_0 = Text("mult=0", font_size=14, color=GRAY).next_to([x0, y0, 0], LEFT, buff=0.2)
        y_label_1 = Text("mult=1", font_size=14, color=GRAY).next_to([x0, y1, 0], LEFT, buff=0.2)
        self.play(
            Create(x_axis), Create(y_axis),
            Write(x_label_0), Write(x_label_end), Write(y_label_0), Write(y_label_1),
        )

        n_samples = 150
        points = []
        for i in range(n_samples + 1):
            step = i * (total_steps - 1) / n_samples
            mult = lr_multiplier(int(round(step)), warmup_steps, total_steps)
            x = x0 + (x1 - x0) * (step / (total_steps - 1))
            y = y0 + (y1 - y0) * mult
            points.append([x, y, 0])
        curve = VMobject(color=BLUE, stroke_width=4)
        curve.set_points_as_corners(points)
        self.play(Create(curve), run_time=2)

        boundary_mult = lr_multiplier(warmup_steps, warmup_steps, total_steps)
        bx = x0 + (x1 - x0) * (warmup_steps / (total_steps - 1))
        by = y0 + (y1 - y0) * boundary_mult
        boundary_dot = Dot([bx, by, 0], color=YELLOW)
        boundary_label = Text(
            f"step {warmup_steps}: peak, multiplier={boundary_mult:.2f}", font_size=15, color=YELLOW
        ).next_to([bx, by, 0], UP, buff=0.2)
        self.play(FadeIn(boundary_dot), Write(boundary_label))
        self.wait(1)

        checkpoints = [0, warmup_steps - 1, warmup_steps, total_steps // 2, total_steps - 1]
        lines = []
        for s in checkpoints:
            mult = lr_multiplier(s, warmup_steps, total_steps)
            lr = base_lr * mult
            lines.append(f"step {s:>3}:  multiplier={mult:.4f}   ->   lr={lr:.2e}")
        readout = Text("\n".join(lines), font_size=16, color=GREEN).move_to([0, -2.7, 0])
        self.play(Write(readout))
        self.wait(2.5)
        self.play(FadeOut(
            title, setup, x_axis, y_axis, x_label_0, x_label_end, y_label_0, y_label_1,
            curve, boundary_dot, boundary_label, readout,
        ))

    # ---- Part 2: the task-10 embedding-init bug, real before vs. after ----

    def _embedding_init_bug_part(self):
        title = self._title("Bug found building task 10: unscaled embedding init blows up logits")

        config = ModelConfig(
            vocab_size=65, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2, d_ff=128,
            n_routed_experts=4, n_shared_experts=1, top_k=2, max_seq_len=32,
        )

        # "after": the real QwenMoEModel exactly as it's built today (std=0.02 fix applied).
        torch.manual_seed(0)
        model_after = QwenMoEModel(config)

        # "before": identical architecture and identical non-embedding weights (same seed),
        # with the embedding manually reset to PyTorch's un-fixed nn.Embedding default
        # (std=1) -- reproducing the exact bug found while validating task 10.
        torch.manual_seed(0)
        model_before = QwenMoEModel(config)
        with torch.no_grad():
            nn.init.normal_(model_before.embedding.weight, mean=0.0, std=1.0)

        x = torch.randint(0, config.vocab_size, (8, 16))
        y = torch.randint(0, config.vocab_size, (8, 16))

        def std_and_loss(model):
            logits, _ = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            return logits.std().item(), loss.item()

        std_before, loss_before = std_and_loss(model_before)
        std_after, loss_after = std_and_loss(model_after)
        ln_v = math.log(config.vocab_size)

        setup = Text(
            f"same random input, same architecture (d_model={config.d_model}), "
            "same non-embedding weights -- only embedding init std differs",
            font_size=16,
        ).move_to([0, 3.2, 0])
        self.play(Write(setup))

        unit_std = 2.3 / max(std_before, std_after, 1e-6)
        std_bars = self._manual_bars(
            [std_before, std_after], ["std_W=1.0 (bug)", "std_W=0.02 (fixed)"], [RED, GREEN],
            [f"logit std={std_before:.2f}", f"logit std={std_after:.2f}"], unit=unit_std,
        )
        std_bars.move_to([-3.3, 0.3, 0])
        std_caption = Text("logit standard deviation", font_size=15, color=GRAY).next_to(
            std_bars, UP, buff=0.35
        )

        unit_loss = 2.3 / max(loss_before, loss_after, ln_v)
        loss_bars = self._manual_bars(
            [loss_before, loss_after, ln_v], ["before", "after", "ln(V) ideal"], [RED, GREEN, BLUE],
            [f"{loss_before:.2f}", f"{loss_after:.2f}", f"{ln_v:.2f}"], unit=unit_loss, bar_width=0.9,
        )
        loss_bars.move_to([3.3, 0.3, 0])
        loss_caption = Text("untrained cross-entropy loss", font_size=15, color=GRAY).next_to(
            loss_bars, UP, buff=0.35
        )

        self.play(Create(std_bars), Write(std_caption))
        self.play(Create(loss_bars), Write(loss_caption))
        self.wait(2)

        caption = Text(
            "sigma_z = sqrt(d_model) * sigma_W * sigma_h -- an unscaled tied embedding\n"
            "(sigma_W=1) blows logits up by a factor of sqrt(d_model); std=0.02 (GPT-2's\n"
            "convention, reused verbatim by nanoGPT) brings untrained loss back to ln(V)",
            font_size=15, color=GRAY,
        ).move_to([0, -2.5, 0])
        self.play(Write(caption))
        self.wait(2.5)
        self.play(FadeOut(title, setup, std_bars, std_caption, loss_bars, loss_caption, caption))

    # ---- Part 3: AdamW's real per-parameter update, captured live ----

    def _adamw_part(self):
        title = self._title("AdamW: real momentum + variance estimates after one real step")

        torch.manual_seed(0)
        config = ModelConfig(**_TINY_CONFIG)
        model = QwenMoEModel(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

        x = torch.randint(0, config.vocab_size, (4, 8))
        y = torch.randint(0, config.vocab_size, (4, 8))

        param = model.blocks[0].attn.w_q.weight
        before = param.detach().clone()

        optimizer.zero_grad(set_to_none=True)
        logits, extras = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        loss.backward()
        optimizer.step()

        state = optimizer.state[param]
        m_t, v_t = state["exp_avg"], state["exp_avg_sq"]
        after = param.detach()
        idx = (0, 0)

        setup = Text(
            "one real forward + backward + optimizer.step() on attn.w_q's weight[0,0]",
            font_size=17,
        ).move_to([0, 3.2, 0])
        self.play(Write(setup))

        rows = [
            f"gradient this step   = {param.grad[idx].item(): .6f}",
            f"m_t (exp_avg)        = {m_t[idx].item(): .6f}",
            f"v_t (exp_avg_sq)     = {v_t[idx].item(): .6f}",
            f"weight BEFORE step   = {before[idx].item(): .6f}",
            f"weight AFTER step    = {after[idx].item(): .6f}",
            f"change (delta)       = {(after[idx] - before[idx]).item(): .6f}",
        ]
        readout = Text("\n".join(rows), font_size=19, color=GREEN).move_to([0, 0.5, 0])
        self.play(Write(readout))
        self.wait(2)

        caption = Text(
            "AdamW combines the gradient's running mean (m_t) and running squared\n"
            "mean (v_t) into an adaptive per-parameter step, THEN separately shrinks\n"
            "every weight toward zero (decoupled weight decay) -- all real numbers above",
            font_size=15, color=GRAY,
        ).move_to([0, -2.6, 0])
        self.play(Write(caption))
        self.wait(2.5)
        self.play(FadeOut(title, setup, readout, caption))

    # ---- Part 4: Trainer checkpoint save + real resume ----

    def _checkpoint_resume_part(self):
        title = self._title("Trainer: resume continues global_step, not step 0 (Requirement 5.4)")

        config = ModelConfig(**_TINY_CONFIG)
        tokens = list(range(10)) * 40
        train_tokens, val_tokens = train_val_split(tokens, val_fraction=0.1)
        train_ds = TokenDataset(train_tokens, seq_len=8)
        val_ds = TokenDataset(val_tokens, seq_len=8)

        with tempfile.TemporaryDirectory() as tmp_dir:
            trainer = Trainer(tmp_dir)

            torch.manual_seed(0)
            model_a = QwenMoEModel(config)
            config_a = TrainConfig(
                batch_size=4, seq_len=8, total_tokens=20 * 4 * 8, lr=1e-2, warmup_steps=5,
                checkpoint_interval_steps=20, eval_interval_steps=20, log_interval_steps=1,
                device="cpu",
            )
            result_a = trainer.fit(model_a, train_ds, val_ds, config_a)

            torch.manual_seed(999)
            model_b = QwenMoEModel(config)  # freshly, differently initialized -- not model_a
            config_b = TrainConfig(
                batch_size=4, seq_len=8, total_tokens=30 * 4 * 8, lr=1e-2, warmup_steps=5,
                checkpoint_interval_steps=10, eval_interval_steps=30, log_interval_steps=1,
                device="cpu",
            )
            result_b = trainer.fit(
                model_b, train_ds, val_ds, config_b, resume_from=result_a.checkpoint_path
            )

            step_lines = sum(
                1 for line in Path(trainer.log_path).read_text().splitlines()
                if json.loads(line)["type"] == "step"
            )

        setup = Text(
            f"run 1: fit() for {result_a.final_step} steps, checkpoint saved  ->  run 2: a "
            f"FRESH model (seed=999), resume_from=that checkpoint, budget {result_b.final_step} steps",
            font_size=15,
        ).move_to([0, 3.3, 0])
        self.play(Write(setup))

        box1 = Rectangle(width=3.1, height=1.1, color=BLUE).move_to([-3.4, 1.2, 0])
        box1_label = Text(f"run 1\nsteps 1..{result_a.final_step}", font_size=15).move_to(box1.get_center())
        disk = Rectangle(width=2.4, height=1.1, color=GRAY).move_to([0, 1.2, 0])
        disk_label = Text(f"checkpoint\nstep={result_a.final_step}", font_size=14).move_to(disk.get_center())
        box2 = Rectangle(width=3.6, height=1.1, color=GREEN).move_to([3.5, 1.2, 0])
        box2_label = Text(
            f"run 2 (resumed)\nsteps {result_a.final_step + 1}..{result_b.final_step}", font_size=14
        ).move_to(box2.get_center())
        arrow1 = Arrow(box1.get_right(), disk.get_left(), buff=0.12)
        arrow2 = Arrow(disk.get_right(), box2.get_left(), buff=0.12)
        self.play(Create(box1), Write(box1_label))
        self.play(Create(arrow1), Create(disk), Write(disk_label))
        self.play(Create(arrow2), Create(box2), Write(box2_label))
        self.wait(1.3)

        would_be_if_broken = result_a.final_step + result_b.final_step
        proof = Text(
            f"real proof it didn't silently restart from 0: the JSON-lines log gained exactly "
            f"{step_lines - result_a.final_step} new 'step' records in run 2\n"
            f"({step_lines} total -- NOT {would_be_if_broken}, which is what a broken "
            "restart-from-0 resume would have produced instead)",
            font_size=15, color=YELLOW,
        ).move_to([0, -1.1, 0])
        self.play(Write(proof))
        self.wait(1.5)

        losses = Text(
            f"final_train_loss:  run 1 = {result_a.final_train_loss:.3f}   "
            f"run 2 (continuing run 1's weights, not model_b's original init) = {result_b.final_train_loss:.3f}",
            font_size=15, color=GREEN,
        ).move_to([0, -2.4, 0])
        self.play(Write(losses))
        self.wait(2.5)
        self.play(FadeOut(
            title, setup, box1, box1_label, arrow1, disk, disk_label, arrow2, box2, box2_label,
            proof, losses,
        ))
