"""Manim preview for Task 8's math (TokenDataset's sliding windows,
train_val_split's contiguous split, and make_fixed_slice's chunk-shuffle
for compute-matched ablation comparisons), animated from the real,
already-tested `qwenmoe.data.dataset` classes/functions rather than
fabricated numbers (Requirement 7.1's spirit) — built ahead of the formal
TraceBundle-based pipeline (tasks 12-13), same approach as
`scenes_task3_preview.py`/`scenes_task5_preview.py`/
`scenes_task6_preview.py`/`scenes_task7_preview.py`. `download_corpus`
isn't included here: it's a real network call with no interesting math to
animate (see the task 8 journal entry for its manual verification).
"""
from __future__ import annotations

import torch
from manim import *

from qwenmoe.data.dataset import TokenDataset, make_fixed_slice, train_val_split


class Task8DataPipelineScene(Scene):
    def construct(self):
        self._sliding_window_part()
        self._train_val_split_part()
        self._fixed_slice_part()

    def _title(self, text):
        title = Text(text, font_size=26)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(0.5)
        return title

    def _box_center_x(self, idx, n_total, box_width, gap):
        total_width = n_total * box_width + (n_total - 1) * gap
        start_x = -total_width / 2 + box_width / 2
        return start_x + idx * (box_width + gap)

    def _token_row(self, labels, box_width=0.5, box_height=0.62, gap=0.08, color=WHITE,
                    highlight=None, font_size=15):
        """Row of boxes, one per token -- same manual-box approach as the
        task 7 preview's `_token_row` (duplicated deliberately, not shared;
        see task 5 preview's design note on why)."""
        highlight = highlight or {}
        n = len(labels)
        group = VGroup()
        for i, label in enumerate(labels):
            x = self._box_center_x(i, n, box_width, gap)
            box_color = highlight.get(i, color)
            box = Rectangle(
                width=box_width, height=box_height, color=box_color,
                fill_color=box_color, fill_opacity=0.35 if i in highlight else 0.0,
            )
            box.move_to([x, 0, 0])
            text = Text(str(label), font_size=font_size).move_to(box.get_center())
            group.add(box, text)
        return group

    def _window_bracket(self, start_idx, end_idx_inclusive, n_total, box_width, gap, y, color):
        """A thin bar spanning boxes [start_idx, end_idx_inclusive] of a row
        built with the same box_width/gap arithmetic as `_token_row` --
        highlights the current x/y window without recoloring overlapping
        box fills (the two windows share seq_len - 1 boxes)."""
        x_start = self._box_center_x(start_idx, n_total, box_width, gap) - box_width / 2
        x_end = self._box_center_x(end_idx_inclusive, n_total, box_width, gap) + box_width / 2
        bracket = Rectangle(
            width=x_end - x_start, height=0.14, color=color,
            fill_color=color, fill_opacity=1.0, stroke_width=0,
        )
        bracket.move_to([(x_start + x_end) / 2, y, 0])
        return bracket

    # ---- Part 1: TokenDataset's sliding windows ----

    def _sliding_window_part(self):
        title = self._title("TokenDataset: sliding (x, y) windows for next-token prediction (Req 4.4)")

        tokens = list(range(20))
        seq_len = 5
        ds = TokenDataset(tokens, seq_len)

        setup = Text(
            f"TokenDataset(tokens=0..19, seq_len={seq_len}) -> len(dataset) = {len(ds)}",
            font_size=18,
        ).move_to([0, 2.7, 0])
        self.play(Write(setup))

        row = self._token_row([str(t) for t in tokens])
        row.move_to([0, 1.3, 0])
        self.play(Create(row))

        legend = Text(
            "blue bracket = x (input window)    yellow bracket = y (target, shifted +1)",
            font_size=15, color=GRAY,
        ).move_to([0, 0.45, 0])
        self.play(Write(legend))

        def window_visuals(i):
            x_i, y_i = ds[i]
            x_bracket = self._window_bracket(i, i + seq_len - 1, len(tokens), 0.5, 0.08, y=0.85, color=BLUE)
            y_bracket = self._window_bracket(i + 1, i + seq_len, len(tokens), 0.5, 0.08, y=1.75, color=YELLOW)
            readout = Text(
                f"i={i}:   x={tuple(x_i.tolist())}   y={tuple(y_i.tolist())}",
                font_size=20, color=GREEN,
            ).move_to([0, -1.6, 0])
            return x_bracket, y_bracket, readout

        x_bracket, y_bracket, readout = window_visuals(0)
        self.play(Create(x_bracket), Create(y_bracket), Write(readout))
        self.wait(1.2)
        for i in [1, 3]:
            new_x, new_y, new_readout = window_visuals(i)
            self.play(Transform(x_bracket, new_x), Transform(y_bracket, new_y), Transform(readout, new_readout))
            self.wait(1.2)

        self.wait(0.3)
        self.play(FadeOut(title, setup, row, legend, x_bracket, y_bracket, readout))

    # ---- Part 2: train_val_split's contiguous, order-preserving split ----

    def _train_val_split_part(self):
        title = self._title("train_val_split: contiguous split, not random (Req 4.4)")

        tokens = list(range(20))
        val_fraction = 0.2
        train, val = train_val_split(tokens, val_fraction=val_fraction)
        n_train = len(train)

        setup = Text(
            f"train_val_split(tokens=0..19, val_fraction={val_fraction}) "
            f"-> train={n_train} tokens, val={len(val)} tokens",
            font_size=17,
        ).move_to([0, 2.7, 0])
        self.play(Write(setup))

        highlight = {i: BLUE for i in range(n_train)}
        highlight.update({i: ORANGE for i in range(n_train, len(tokens))})
        row = self._token_row([str(t) for t in tokens], highlight=highlight)
        row.move_to([0, 1.4, 0])
        self.play(Create(row))

        boundary_x = (
            self._box_center_x(n_train - 1, len(tokens), 0.5, 0.08)
            + self._box_center_x(n_train, len(tokens), 0.5, 0.08)
        ) / 2
        boundary_line = DashedLine([boundary_x, 1.85, 0], [boundary_x, 0.95, 0], color=WHITE)
        boundary_label = Text(
            f"split boundary: index {n_train}", font_size=15, color=GRAY
        ).next_to(boundary_line, UP, buff=0.1)
        legend = Text("blue = train    orange = val", font_size=15, color=GRAY).move_to([0, 0.55, 0])
        self.play(Create(boundary_line), Write(boundary_label), Write(legend))
        self.wait(1.2)

        train_ds = TokenDataset(train, seq_len=3)
        val_ds = TokenDataset(val, seq_len=3)
        tx, ty = train_ds[0]
        vx, vy = val_ds[0]
        result = Text(
            f"TokenDataset(train, seq_len=3): len={len(train_ds)}, window 0: x={tuple(tx.tolist())}\n"
            f"TokenDataset(val, seq_len=3):   len={len(val_ds)}, window 0: x={tuple(vx.tolist())}",
            font_size=18, color=GREEN,
        ).move_to([0, -1.2, 0])
        self.play(Write(result))
        self.wait(1.3)

        caption = Text(
            "each split gets its own separate TokenDataset -- every window uses only\n"
            "genuinely adjacent original tokens; none ever mixes train and val tokens",
            font_size=15, color=GRAY,
        ).move_to([0, -2.6, 0])
        self.play(Write(caption))
        self.wait(2)

        self.play(FadeOut(title, setup, row, boundary_line, boundary_label, legend, result, caption))

    # ---- Part 3: make_fixed_slice's chunk-shuffle ----

    def _fixed_slice_part(self):
        title = self._title("make_fixed_slice: chunk-shuffle for compute-matched comparisons (Req 11.5, 12.6)")

        tokens = list(range(20))
        seed = 1
        sliced = make_fixed_slice(tokens, n_tokens=20, seed=seed)
        sliced_list = sliced.tolist()
        chunk_size = max(1, min(1024, len(tokens) // 4))  # mirrors dataset.py's own chunk sizing
        n_chunks = len(tokens) // chunk_size
        chunk_colors = [RED, BLUE, GREEN, PURPLE][:n_chunks]

        # original tokens are 0..N-1, so a value's own chunk is recoverable
        # directly from the real output -- no need to reach into internals
        orig_highlight = {i: chunk_colors[i // chunk_size] for i in range(len(tokens))}
        sliced_highlight = {i: chunk_colors[v // chunk_size] for i, v in enumerate(sliced_list)}
        order = [sliced_list[k * chunk_size] // chunk_size for k in range(n_chunks)]

        setup = Text(
            f"make_fixed_slice(tokens=0..19, n_tokens=20, seed={seed})", font_size=19,
        ).move_to([0, 3.0, 0])
        self.play(Write(setup))

        before_label = Text("before: 4 chunks of 5", font_size=15, color=GRAY).move_to([0, 2.0, 0])
        before_row = self._token_row([str(t) for t in tokens], highlight=orig_highlight)
        before_row.move_to([0, 1.4, 0])
        self.play(Write(before_label), Create(before_row))
        self.wait(1)

        order_label = Text(
            f"chunk order after shuffling with seed={seed}: {order}", font_size=17, color=YELLOW,
        ).move_to([0, 0.3, 0])
        self.play(Write(order_label))
        self.wait(0.8)

        after_label = Text("after: same chunk colors, reordered", font_size=15, color=GRAY).move_to([0, -0.6, 0])
        after_row = self._token_row([str(t) for t in sliced_list], highlight=sliced_highlight)
        after_row.move_to([0, -1.2, 0])
        self.play(Write(after_label), Create(after_row))
        self.wait(1.2)

        caption = Text(
            "only chunk ORDER is randomized -- within each chunk, ascending token\n"
            "order survives, so local structure isn't destroyed by the shuffle",
            font_size=14, color=GRAY,
        ).move_to([0, -2.3, 0])
        self.play(Write(caption))
        self.wait(2)
        self.play(FadeOut(title, setup, before_label, before_row, order_label, after_label, after_row, caption))

        title2 = self._title("same seed -> identical slice; different seed -> different slice")
        a = make_fixed_slice(tokens, n_tokens=20, seed=1)
        b = make_fixed_slice(tokens, n_tokens=20, seed=1)
        c = make_fixed_slice(tokens, n_tokens=20, seed=0)
        same_seed_label = Text(
            f"seed=1 called twice -> identical: {torch.equal(a, b)}", font_size=20, color=GREEN,
        ).move_to([0, 1.6, 0])
        diff_seed_label = Text(
            f"seed=1 vs seed=0 -> different: {not torch.equal(a, c)}", font_size=20, color=GREEN,
        ).move_to([0, 0.8, 0])
        self.play(Write(same_seed_label))
        self.wait(0.8)
        self.play(Write(diff_seed_label))
        self.wait(1.3)

        identical_box = Rectangle(width=4.4, height=0.75, color=WHITE).move_to([0, -0.5, 0])
        identical_label = Text("one identical fixed slice", font_size=17).move_to(identical_box.get_center())
        variants = VGroup()
        for i, name in enumerate(["rope", "nope", "alibi"]):
            vx = -3.6 + i * 3.6
            box = Rectangle(width=2.8, height=0.75, color=BLUE).move_to([vx, -2.0, 0])
            label = Text(f"positional_scheme={name}", font_size=14).move_to(box.get_center())
            arrow = Arrow(
                identical_box.get_bottom(), box.get_top(), buff=0.08, color=GRAY, stroke_width=2.5,
            )
            variants.add(arrow, box, label)
        caption2 = Text(
            "same tokens for every variant -> any loss difference isolates\n"
            "ONLY the one setting that changed (Req 12.6)",
            font_size=14, color=GRAY,
        ).move_to([0, -3.2, 0])
        self.play(Create(identical_box), Write(identical_label))
        self.play(Create(variants))
        self.wait(1)
        self.play(Write(caption2))
        self.wait(2.5)
        self.play(FadeOut(
            title2, same_seed_label, diff_seed_label, identical_box, identical_label, variants, caption2
        ))
