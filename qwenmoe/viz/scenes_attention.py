"""Task 13.2 — attention scene (Requirement 7.3): the real Q.K^T score
matrix, the causal mask, the softmax normalization, and the weighted sum
with V. Everything here comes straight from a real `capture_trace` call
on the real `runs/smoke_test/checkpoints/step_1000.pt` checkpoint
(Requirement 7.1) -- no extra model load needed, unlike
`scenes_embedding.py`/`scenes_norm_act.py`, since every field this scene
needs already lives on `LayerTrace`. See
journey_of_a_model/004-attention-gqa.md for the full derivation of scaled
dot-product attention and GQA; this scene only animates it with real data.
"""
from __future__ import annotations

from pathlib import Path

from manim import *

from qwenmoe.viz.common import capture_trace

_CHECKPOINT_PATH = Path("runs/smoke_test/checkpoints/step_1000.pt")
_PROMPT = "ROMEO:"
_HEAD = 0


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


class AttentionScene(Scene):
    """Requirement 7.3: Q.K^T scores, causal masking, softmax, weighted
    sum with V -- one real head, one real prompt."""

    def construct(self):
        _require_checkpoint(_CHECKPOINT_PATH)
        bundle = capture_trace(_CHECKPOINT_PATH, _PROMPT)
        self._mask_part(bundle)
        self._softmax_part(bundle)
        self._weighted_sum_part(bundle)

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

    # ---- Part 1: real pre-mask scores -> causal mask (Requirement 7.3) ----

    def _mask_part(self, bundle):
        title = self._title("Q.K^T scores (real, post-scale) and the causal mask")
        seq_len = len(bundle.token_ids)
        query_pos = 2  # interior position: masking removes 3 of its 6 real scores
        scores = bundle.layers[0].attn_scores_pre_mask[_HEAD, query_pos]
        names = [f"j={j}" for j in range(seq_len)]

        setup = Text(
            f'prompt = "{_PROMPT}"  --  layer 0, head {_HEAD}, query position i={query_pos} '
            f"(of {seq_len}); attn_scores_pre_mask[{_HEAD}, {query_pos}]",
            font_size=16,
        ).move_to([0, 3.3, 0])
        self.play(Write(setup))

        raw_vals = scores.tolist()
        before_bars = self._manual_bars(
            raw_vals, names, [BLUE] * seq_len, [f"{v:.2f}" for v in raw_vals],
            unit=_auto_unit(raw_vals), bar_width=0.7, gap=0.35,
        ).move_to([-3.3, -0.3, 0])
        before_caption = Text("real Q.K^T / sqrt(head_dim)", font_size=17, color=BLUE).next_to(
            before_bars, UP, buff=0.35
        )
        self.play(Create(before_bars), Write(before_caption))
        self.wait(1.5)

        masked_colors = [RED if j > query_pos else GREEN for j in range(seq_len)]
        masked_labels = [
            "-inf" if j > query_pos else f"{v:.2f}" for j, v in enumerate(raw_vals)
        ]
        masked_heights = [0.05 if j > query_pos else v for j, v in enumerate(raw_vals)]
        after_bars = self._manual_bars(
            masked_heights, names, masked_colors, masked_labels,
            unit=_auto_unit(raw_vals), bar_width=0.7, gap=0.35,
        ).move_to([3.3, -0.3, 0])
        after_caption = Text(
            "+ causal mask (upper triangle -> -inf)", font_size=17, color=RED
        ).next_to(after_bars, UP, buff=0.35)
        arrow = Arrow(before_bars.get_right(), after_bars.get_left(), buff=0.3)
        self.play(Create(arrow), Create(after_bars), Write(after_caption))
        self.wait(1.5)

        caption = Text(
            f"key positions j > i={query_pos} are still in the future at query position "
            f"{query_pos} -- masked so they can never influence this row's output",
            font_size=15, color=GRAY,
        ).move_to([0, -3.1, 0])
        self.play(Write(caption))
        self.wait(2.5)
        self.play(FadeOut(title, setup, before_bars, before_caption, arrow, after_bars, after_caption, caption))

    # ---- Part 2: softmax normalization (Requirement 7.3) ----

    def _softmax_part(self, bundle):
        title = self._title("Softmax normalization: masked scores -> a real probability row")
        seq_len = len(bundle.token_ids)
        query_pos = 2
        scores = bundle.layers[0].attn_scores_pre_mask[_HEAD, query_pos].tolist()
        probs = bundle.layers[0].attn_scores_post_softmax[_HEAD, query_pos].tolist()
        names = [f"j={j}" for j in range(seq_len)]

        setup = Text(
            f"same row (i={query_pos}) -- attn_scores_post_softmax[{_HEAD}, {query_pos}]",
            font_size=17,
        ).move_to([0, 3.3, 0])
        self.play(Write(setup))

        masked_colors = [RED if j > query_pos else GREEN for j in range(seq_len)]
        masked_labels = ["-inf" if j > query_pos else f"{v:.2f}" for j, v in enumerate(scores)]
        masked_heights = [0.05 if j > query_pos else v for j, v in enumerate(scores)]
        before_bars = self._manual_bars(
            masked_heights, names, masked_colors, masked_labels,
            unit=_auto_unit(scores), bar_width=0.7, gap=0.35,
        ).move_to([-3.3, -0.3, 0])
        before_caption = Text("masked scores", font_size=17, color=RED).next_to(
            before_bars, UP, buff=0.35
        )
        self.play(Create(before_bars), Write(before_caption))
        self.wait(1)

        after_colors = [GRAY if j > query_pos else "#83C167" for j in range(seq_len)]
        after_bars = self._manual_bars(
            probs, names, after_colors, [f"{v:.3f}" for v in probs],
            unit=_auto_unit(probs), bar_width=0.7, gap=0.35,
        ).move_to([3.3, -0.3, 0])
        after_caption = Text(
            f"softmax(scores) -- sums to {sum(probs):.4f}", font_size=17, color="#83C167"
        ).next_to(after_bars, UP, buff=0.35)
        arrow = Arrow(before_bars.get_right(), after_bars.get_left(), buff=0.3)
        self.play(Create(arrow), Create(after_bars), Write(after_caption))
        self.wait(1.5)

        caption = Text(
            "softmax(-inf) = 0 exactly -- the masked positions get exactly zero weight, "
            "not just a small one",
            font_size=15, color=GRAY,
        ).move_to([0, -3.1, 0])
        self.play(Write(caption))
        self.wait(2.5)
        self.play(FadeOut(title, setup, before_bars, before_caption, arrow, after_bars, after_caption, caption))

    # ---- Part 3: weighted sum with V (Requirement 7.3) ----

    def _weighted_sum_part(self, bundle):
        title = self._title("Weighted sum with V: probs @ V for one real head")
        seq_len = len(bundle.token_ids)
        query_pos = seq_len - 1  # last position: unmasked, attends to every real key
        layer = bundle.layers[0]
        probs = layer.attn_scores_post_softmax[_HEAD, query_pos]
        n_rep = 4 // 2  # n_heads // n_kv_heads for the SMOKE_TEST checkpoint (GQA sharing)
        kv_head = _HEAD // n_rep
        v = layer.v[kv_head]  # (seq_len, head_dim), this head's real (pre-repeat) V

        setup = Text(
            f"query position i={query_pos} (last token, attends to all {seq_len} real keys) "
            f"-- head {_HEAD} shares kv-head {kv_head} of 2 (GQA)",
            font_size=16,
        ).move_to([0, 3.3, 0])
        self.play(Write(setup))

        names = [f"j={j}" for j in range(seq_len)]
        w_vals = probs.tolist()
        w_bars = self._manual_bars(
            w_vals, names, [BLUE] * seq_len, [f"{v:.3f}" for v in w_vals],
            unit=_auto_unit(w_vals), bar_width=0.7, gap=0.35,
        ).move_to([-3.3, -0.3, 0])
        w_caption = Text("attention weights (this row)", font_size=16, color=BLUE).next_to(
            w_bars, UP, buff=0.35
        )
        self.play(Create(w_bars), Write(w_caption))
        self.wait(1.5)

        manual_out = probs @ v  # exactly what Attention.forward computes per head, real tensors
        shown = manual_out[:8].tolist()
        out_bars = self._manual_bars(
            shown, [f"d{i}" for i in range(8)], [ORANGE] * 8, [f"{v:.3f}" for v in shown],
            unit=_auto_unit(shown), bar_width=0.55, gap=0.3,
        ).move_to([3.3, -0.3, 0])
        out_caption = Text(
            "sum_j weight_j * V[j]  (first 8 of 16 head_dim)", font_size=16, color=ORANGE
        ).next_to(out_bars, UP, buff=0.35)
        arrow = Arrow(w_bars.get_right(), out_bars.get_left(), buff=0.3)
        self.play(Create(arrow), Create(out_bars), Write(out_caption))
        self.wait(2)

        caption = Text(
            "this is one head's contribution, before concatenating all 4 heads and the w_o "
            "projection (that combined result is LayerTrace.attn_output)",
            font_size=14, color=GRAY,
        ).move_to([0, -3.2, 0])
        self.play(Write(caption))
        self.wait(2.5)
        self.play(FadeOut(title, setup, w_bars, w_caption, arrow, out_bars, out_caption, caption))
