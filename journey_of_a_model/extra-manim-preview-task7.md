# Extra — Manim preview of Task 7's math (CharTokenizer vocab, BPE merge training, byte-level universality)

**Type:** task (supplementary, not a numbered task) · **Date:** 2026-08-31 ·
**Related:** Requirement 7 (visualizations), task 7's tokenizers; see
[007-tokenizers.md](007-tokenizers.md) for the full concept explainers/
derivations — this entry only covers the new visualization work, not the
math itself again.

## Why this isn't numbered `008`
Same reasoning as
[extra-manim-preview-task3.md](extra-manim-preview-task3.md),
[extra-manim-preview-task5.md](extra-manim-preview-task5.md), and
[extra-manim-preview-task6.md](extra-manim-preview-task6.md): numbered
entries track task numbers 1:1, and task 8 (data pipeline) hasn't happened
yet. This is a **standalone preview** of task 7's math, requested
directly, ahead of the formal Task 13 visualization pipeline (which
consumes a `TraceBundle` captured from a real trained checkpoint via
`forward_with_trace()` — that needs tasks 8-12 first). It may inform a
future formal scene, but isn't claiming to satisfy one now.

## What was built
`qwenmoe/viz/scenes_task7_preview.py::Task7TokenizerMathScene` — a single
Manim scene with three parts, each animating **real numbers produced by
calling the actual, already-tested classes** (`CharTokenizer`,
`ByteBPETokenizer`, plus its internal `_count_pairs`/`_merge_pair` helpers
from `qwenmoe.tokenizer.bpe`) — cross-checked in a plain Python shell
before animating, not fabricated illustration numbers:

1. **`CharTokenizer`'s vocab.** A real `CharTokenizer` trained on
   `"to be or not to be"`, showing its actual 7-entry vocab (`' '=0,
   'b'=1, 'e'=2, 'n'=3, 'o'=4, 'r'=5, 't'=6` — sorted-unique-character
   order), a real `encode("to be") = [6, 4, 0, 1, 2]` /
   `decode(...) = "to be"` round trip, and a real out-of-vocabulary
   `ValueError` from `encode("to be!")` (`"character '!' is not in the
   trained vocabulary"`).
2. **`ByteBPETokenizer`'s greedy merge-pair training loop.** The exact
   3-merge trace from journal entry 007's worked example
   (`"aaabdaaabac"`, `target_vocab_size=256+3`), replayed step by step by
   calling the real `_count_pairs`/`_merge_pair` functions directly:
   `('a','a')` count 4 → `256="aa"`; `('aa','a')` count 2 (the tie
   documented in 007) → `257="aaa"`; `('aaa','b')` count 2 → `258="aaab"`.
   Each step shows the current token sequence (as boxes), highlights the
   winning pair, and shows the collapsed sequence after the merge. Closes
   with a real `encode`/`decode` round trip on the fully-trained
   tokenizer, confirmed to match.
3. **Compression on repetitive text + universal byte-level coverage.** A
   real `ByteBPETokenizer(target_vocab_size=300)` trained on `"the quick
   brown fox jumps over the lazy dog. " * 40` (1800 raw bytes), showing
   the real result — just **8 tokens** (225x fewer) — broken down by each
   final token's real byte length (`4, 720, 720, 180, 90, 45, 40, 1`),
   with a caption explaining this as greedy pairwise merging compounding
   like doubling on perfectly repetitive input (single tokens end up
   worth 1, 2, 4, or 16 whole sentence-repeats each), not a realistic
   natural-text compression ratio. Then the same trained tokenizer
   encodes/decodes `"héllo wörld 你好 🎉"` — text containing characters it
   never saw during training — losslessly (25 bytes in, 25 tokens out,
   zero merges apply, decodes back exactly), contrasted with a real
   `CharTokenizer` (trained on the part-1 phrase) raising `ValueError:
   character 'h' is not in the trained vocabulary` on that same text.

Render command:
```
uv run manim -ql qwenmoe/viz/scenes_task7_preview.py Task7TokenizerMathScene
```
Output: `media/videos/scenes_task7_preview/480p15/Task7TokenizerMathScene.mp4`
(94s, 480p15; `media/` is already gitignored as fully regeneratable from
the scene file, same as the earlier previews).

## Bug found (cosmetic, not fixed): the 🎉 emoji glyph doesn't render
Part 3's unseen-text demo encodes/decodes `"héllo wörld 你好 🎉"` — the `é`,
`ö`, and `你好` characters all render correctly in the video, but `🎉`
renders as blank space. Checked `fc-list | grep -i emoji` on this machine
first (expecting a missing-font explanation, per the project's earlier
LaTeX-availability lesson) — **Apple Color Emoji is actually installed**,
so the cause isn't a missing font. The real cause is that Manim's `Text`
renders via Pango (`manimpango`), and Pango's rendering pipeline doesn't
support color/bitmap emoji glyphs the way a browser or native macOS text
view does — a rendering-pipeline limitation, not a missing-asset one.
**Not fixed**, because it doesn't affect what this demo is actually
proving: the real byte/token counts and the `decodes back exactly: True`
result shown directly below are computed by the real tokenizer on the
real string (emoji included) and are unaffected by whether the glyph is
*visible* — `ByteBPETokenizer.encode`/`decode` still correctly round-trip
the emoji's 4 UTF-8 bytes, this is purely a display gap in this preview
video. **Lesson for future scenes:** verify a quick "does this glyph
actually render" check (same category as the LaTeX-availability check
already in memory) before relying on any Manim `Text` string containing
emoji specifically — plain accented Latin (`é`, `ö`) and CJK (`你好`) both
rendered fine here, so this is specifically a color-emoji gap, not a
general non-ASCII one.

## Verification
Same approach as the earlier previews: extracted a coarse contact-sheet
grid (one frame every 4s) across the whole render for a first pass, then
a finer 1-frame-per-second grid to precisely locate "settled"
(post-animation, mid-`wait()`) timestamps, then individually-grabbed
frames at those timestamps with accurate (post-`-i`) seeking. Confirmed:
- Part 1: the 7-box vocab strip reads `' ' 'b' 'e' 'n' 'o' 'r' 't'` with
  ids `0-6` below each, matching sorted-character order; the encode/decode
  round trip and the red `ValueError` message both render cleanly with no
  overlap.
- Part 2: all three steps' "before" rows (11, 9, 7 boxes) correctly
  highlight the winning pair in yellow, the merge label names the right
  new token each time, and the "after" rows (9, 7, 5 boxes) show the
  correctly-collapsed sequence — step 3's after-row settles to exactly
  `aaab, d, aaab, a, c`, matching journal 007's verified trace. The final
  green confirmation text matches the real `encode`/`decode` call.
- Part 3: the `1800 raw bytes -> 8 tokens (225x fewer)` stat and the
  8-box length breakdown (`4, 720, 720, 180, 90, 45, 40, 1`, summing to
  1800) render with no overlap against the caption below or the
  explanatory note beneath that; the unseen-text and `CharTokenizer`
  contrast messages both render fully within the frame with correct
  real values (aside from the emoji-glyph gap noted above).
- No part's title/content bled into the next part — each part's `FadeOut`
  fully clears before the next part's title is written, matching the
  reliable sequential-fade pattern already used in the task 5/6 previews.

## Outcome
A real 94-second video rendered successfully, showing task 7's actual
tokenizer math (`CharTokenizer`'s sorted-character vocab and
out-of-vocabulary handling, `ByteBPETokenizer`'s greedy merge-pair
training loop step by step, and byte-level BPE's compression behavior
plus universal-coverage guarantee) using real numbers from the real,
tested `qwenmoe.tokenizer.char`/`qwenmoe.tokenizer.bpe` code.
