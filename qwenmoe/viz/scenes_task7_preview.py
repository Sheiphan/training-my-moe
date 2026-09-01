"""Manim preview for Task 7's math (CharTokenizer's sorted-character vocab,
ByteBPETokenizer's greedy merge-pair training loop, and byte-level BPE's
compression + universal-coverage properties), animated from the real,
already-tested `qwenmoe.tokenizer.char`/`qwenmoe.tokenizer.bpe` classes
rather than fabricated numbers (Requirement 7.1's spirit) — built ahead of
the formal TraceBundle-based pipeline (tasks 12-13), same approach as
`scenes_task3_preview.py`/`scenes_task5_preview.py`/`scenes_task6_preview.py`.
"""
from __future__ import annotations

from manim import *

from qwenmoe.tokenizer.bpe import ByteBPETokenizer, _count_pairs, _merge_pair
from qwenmoe.tokenizer.char import CharTokenizer


class Task7TokenizerMathScene(Scene):
    def construct(self):
        self._char_tokenizer_part()
        self._bpe_training_part()
        self._compression_universality_part()

    def _title(self, text):
        title = Text(text, font_size=28)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(0.5)
        return title

    def _token_row(
        self,
        labels,
        sub_labels=None,
        box_width=0.9,
        box_height=0.8,
        gap=0.18,
        color=WHITE,
        highlight=None,
        font_size=20,
        sub_font_size=14,
    ):
        """A horizontal row of boxes, each showing `labels[i]` (and
        optionally a smaller `sub_labels[i]` beneath) -- used for the
        char-vocab strip, BPE token-sequence steps, and the final
        token-length breakdown, so every "sequence of tokens" shown is a
        real row of real values rather than a schematic illustration."""
        highlight = highlight or {}
        n = len(labels)
        total_width = n * box_width + (n - 1) * gap
        start_x = -total_width / 2 + box_width / 2
        group = VGroup()
        for i, label in enumerate(labels):
            x = start_x + i * (box_width + gap)
            box_color = highlight.get(i, color)
            box = Rectangle(
                width=box_width,
                height=box_height,
                color=box_color,
                fill_color=box_color,
                fill_opacity=0.25 if i in highlight else 0.0,
            )
            box.move_to([x, 0, 0])
            text = Text(str(label), font_size=font_size).move_to(box.get_center())
            group.add(box, text)
            if sub_labels is not None:
                sub = Text(str(sub_labels[i]), font_size=sub_font_size, color=GRAY).next_to(
                    box, DOWN, buff=0.12
                )
                group.add(sub)
        return group

    # ---- Part 1: CharTokenizer's sorted-unique-character vocab ----

    def _make_char_tokenizer_trace(self):
        """Real `CharTokenizer`, trained on a short real phrase -- the
        vocab, an encode/decode round trip, and a real out-of-vocabulary
        `ValueError` are all produced by calling the actual class."""
        phrase = "to be or not to be"
        tok = CharTokenizer()
        tok.train(phrase)
        sample = "to be"
        ids = tok.encode(sample)
        decoded = tok.decode(ids)
        try:
            tok.encode("to be!")
            oov_message = None
        except ValueError as e:
            oov_message = str(e)
        vocab_items = sorted(tok._stoi.items(), key=lambda kv: kv[1])
        return {
            "phrase": phrase,
            "vocab_items": vocab_items,
            "sample": sample,
            "ids": ids,
            "decoded": decoded,
            "oov_message": oov_message,
        }

    def _char_tokenizer_part(self):
        title = self._title("CharTokenizer: vocab = sorted unique characters (Req 4.2-4.3)")
        t = self._make_char_tokenizer_trace()

        phrase_label = Text(
            f"train(text) with text = {t['phrase']!r}", font_size=20
        ).move_to([0, 2.6, 0])
        self.play(Write(phrase_label))
        self.wait(0.8)

        chars = [repr(ch) for ch, _ in t["vocab_items"]]
        ids_labels = [str(i) for _, i in t["vocab_items"]]
        vocab_group = self._token_row(chars, sub_labels=ids_labels, box_width=0.75, gap=0.2, color=BLUE)
        vocab_group.move_to([0, 1.1, 0])
        vocab_caption = Text(
            "vocab_size = 7 -- sorted unique characters, id shown below each",
            font_size=16,
            color=GRAY,
        ).next_to(vocab_group, DOWN, buff=0.5)
        self.play(Create(vocab_group), Write(vocab_caption))
        self.wait(1.5)
        self.play(FadeOut(phrase_label, vocab_caption))

        sample_label = Text(f"encode({t['sample']!r})", font_size=20).move_to([0, 2.6, 0])
        self.play(Write(sample_label))
        ids_group = self._token_row([str(i) for i in t["ids"]], box_width=0.75, gap=0.2, color=YELLOW)
        ids_group.move_to([0, -0.2, 0])
        self.play(Create(ids_group))
        self.wait(1)

        decode_label = Text(
            f"decode({t['ids']}) = {t['decoded']!r}", font_size=20, color=GREEN
        ).move_to([0, -1.6, 0])
        self.play(Write(decode_label))
        self.wait(1.5)
        self.play(FadeOut(sample_label, ids_group, decode_label, vocab_group))

        oov_label = Text(
            "encode(\"to be!\") -- '!' never appeared during train()", font_size=20
        ).move_to([0, 1.0, 0])
        self.play(Write(oov_label))
        oov_result = Text(f"ValueError: {t['oov_message']}", font_size=18, color=RED).move_to(
            [0, -0.2, 0]
        )
        self.play(Write(oov_result))
        self.wait(2)
        self.play(FadeOut(title, oov_label, oov_result))

    # ---- Part 2: ByteBPETokenizer's greedy merge-pair training loop ----

    def _make_bpe_training_trace(self):
        """Replays the exact 3-merge training loop `ByteBPETokenizer.train`
        runs internally, using the real `_count_pairs`/`_merge_pair`
        helpers from `qwenmoe.tokenizer.bpe` at every step, so the
        intermediate pair-counts/argmax/merge shown are real, not
        reconstructed after the fact from just the final result."""
        ids = list("aaabdaaabac".encode("utf-8"))
        names = {97: "a", 98: "b", 99: "c", 100: "d"}
        steps = []
        next_id = 256
        for _ in range(3):
            counts = _count_pairs(ids)
            readable_counts = {(names[a], names[b]): c for (a, b), c in counts.items()}
            best_pair = max(counts, key=counts.get)
            best_count = counts[best_pair]
            seq_before = [names[i] for i in ids]
            best_pair_labels = (names[best_pair[0]], names[best_pair[1]])
            ids = _merge_pair(ids, best_pair, next_id)
            names[next_id] = names[best_pair[0]] + names[best_pair[1]]
            steps.append(
                {
                    "seq_before": seq_before,
                    "readable_counts": readable_counts,
                    "best_pair_labels": best_pair_labels,
                    "best_count": best_count,
                    "new_id": next_id,
                    "new_label": names[next_id],
                    "seq_after": [names[i] for i in ids],
                }
            )
            next_id += 1
        return steps

    def _bpe_step_visual(self, step, step_num):
        seq_before = step["seq_before"]
        best_labels = step["best_pair_labels"]
        highlight_idx = next(
            i
            for i in range(len(seq_before) - 1)
            if (seq_before[i], seq_before[i + 1]) == best_labels
        )
        highlight = {highlight_idx: YELLOW, highlight_idx + 1: YELLOW}

        step_label = Text(
            f"Step {step_num}: most frequent adjacent pair {best_labels}, "
            f"count={step['best_count']}",
            font_size=20,
        ).move_to([0, 2.5, 0])
        row = self._token_row(
            seq_before, box_width=0.6, gap=0.12, color=WHITE, highlight=highlight, font_size=16
        )
        row.move_to([0, 1.1, 0])
        merge_label = Text(
            f"merge -> new token {step['new_id']} = {step['new_label']!r}",
            font_size=20,
            color=YELLOW,
        ).move_to([0, -0.3, 0])
        after_row = self._token_row(
            step["seq_after"], box_width=0.6, gap=0.12, color=BLUE, font_size=16
        )
        after_row.move_to([0, -1.7, 0])

        self.play(Write(step_label))
        self.play(Create(row))
        self.wait(1)
        self.play(Write(merge_label))
        self.play(Create(after_row))
        self.wait(1.5)
        self.play(FadeOut(step_label, row, merge_label, after_row))

    def _bpe_training_part(self):
        title = self._title("ByteBPETokenizer: greedy merge-pair training loop (Req 4.2)")

        setup_label = Text(
            "train(text) with text = 'aaabdaaabac', target_vocab_size = 256 + 3",
            font_size=20,
        ).move_to([0, 3.0, 0])
        self.play(Write(setup_label))
        self.wait(1)
        self.play(FadeOut(setup_label))

        steps = self._make_bpe_training_trace()
        for i, step in enumerate(steps, start=1):
            self._bpe_step_visual(step, i)

        tok = ByteBPETokenizer(target_vocab_size=259)
        tok.train("aaabdaaabac")
        encoded = tok.encode("aaabdaaabac")
        decoded = tok.decode(encoded)
        confirm = Text(
            f"encode('aaabdaaabac') = {encoded}\ndecode(...) = {decoded!r}",
            font_size=20,
            color=GREEN,
        ).move_to([0, 0, 0])
        self.play(Write(confirm))
        self.wait(2)
        self.play(FadeOut(title, confirm))

    # ---- Part 3: compression on repetitive text + universal byte coverage ----

    def _make_compression_trace(self):
        """Real `ByteBPETokenizer` trained on a highly repetitive corpus
        (to make the merge mechanism's effect visible), plus the same
        tokenizer encoding/decoding text it never trained on, contrasted
        with a real `CharTokenizer`'s out-of-vocabulary failure on that
        same unseen text."""
        corpus = "the quick brown fox jumps over the lazy dog. " * 40
        tok = ByteBPETokenizer(target_vocab_size=300)
        tok.train(corpus)
        encoded = tok.encode(corpus)
        raw_len = len(corpus.encode("utf-8"))
        token_lengths = [len(tok._vocab[i]) for i in encoded]

        unseen = "héllo wörld 你好 🎉"
        unseen_encoded = tok.encode(unseen)
        unseen_decoded = tok.decode(unseen_encoded)

        char_tok = CharTokenizer()
        char_tok.train("to be or not to be")
        try:
            char_tok.encode(unseen)
            char_oov = None
        except ValueError as e:
            char_oov = str(e)

        return {
            "raw_len": raw_len,
            "num_tokens": len(encoded),
            "token_lengths": token_lengths,
            "unseen": unseen,
            "unseen_encoded_len": len(unseen_encoded),
            "unseen_raw_len": len(unseen.encode("utf-8")),
            "unseen_round_trip_ok": unseen_decoded == unseen,
            "char_oov": char_oov,
        }

    def _compression_universality_part(self):
        title = self._title(
            "Byte-level BPE: extreme compression on repetitive text + universal coverage (Req 4.3)"
        )
        t = self._make_compression_trace()

        setup_label = Text(
            "corpus = 'the quick brown fox jumps over the lazy dog. ' x 40, "
            "target_vocab_size = 300",
            font_size=17,
        ).move_to([0, 3.0, 0])
        self.play(Write(setup_label))
        self.wait(1)

        stat_label = Text(
            f"{t['raw_len']} raw bytes -> {t['num_tokens']} tokens "
            f"({t['raw_len'] / t['num_tokens']:.0f}x fewer)",
            font_size=24,
            color=YELLOW,
        ).move_to([0, 2.0, 0])
        self.play(Write(stat_label))
        self.wait(1)

        lengths_row = self._token_row(
            [str(n) for n in t["token_lengths"]], box_width=0.9, gap=0.18, color=BLUE, font_size=18
        )
        lengths_row.move_to([0, 0.5, 0])
        lengths_caption = Text(
            "bytes represented by each of the 8 final tokens", font_size=16, color=GRAY
        ).next_to(lengths_row, DOWN, buff=0.3)
        self.play(Create(lengths_row), Write(lengths_caption))
        self.wait(1)

        doubling_note = Text(
            "greedy pairwise merging compounds like doubling on perfectly repetitive\n"
            "input (single tokens end up worth 1, 2, 4, 16 sentence-repeats each) --\n"
            "an extreme case chosen to make the mechanism visible, not a realistic\n"
            "compression ratio for natural text",
            font_size=15,
            color=GRAY,
        ).move_to([0, -1.8, 0])
        self.play(Write(doubling_note))
        self.wait(2.5)
        self.play(FadeOut(setup_label, stat_label, lengths_row, lengths_caption, doubling_note))

        unseen_label = Text(
            f"encode({t['unseen']!r}) -- never seen during training", font_size=20
        ).move_to([0, 2.2, 0])
        self.play(Write(unseen_label))
        unseen_result = Text(
            f"{t['unseen_raw_len']} bytes in -> {t['unseen_encoded_len']} tokens out "
            f"(zero merges apply) -> decodes back exactly: {t['unseen_round_trip_ok']}",
            font_size=18,
            color=GREEN,
        ).move_to([0, 1.0, 0])
        self.play(Write(unseen_result))
        self.wait(1.5)

        contrast_label = Text(
            "CharTokenizer on that same text (7-char vocab from 'to be or not to be'):",
            font_size=18,
        ).move_to([0, -0.4, 0])
        contrast_result = Text(f"ValueError: {t['char_oov']}", font_size=16, color=RED).move_to(
            [0, -1.2, 0]
        )
        self.play(Write(contrast_label))
        self.play(Write(contrast_result))
        self.wait(2.5)
        self.play(FadeOut(title, unseen_label, unseen_result, contrast_label, contrast_result))
