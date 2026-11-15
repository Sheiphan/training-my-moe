"""Reproduces every number in `039-bpe-incremental-counts-and-streaming.md`.

Run it:  uv run python docs/math/verify_039_bpe_incremental_counts_and_streaming.py

Covers journal 039 (the BPE trainer rewrite) and journal 040 (streaming
corpus preparation). ONE running example for the BPE half: the tiny text
below, pre-tokenized by the real `_PRE_TOKEN_PATTERN`, trained by the real
`ByteBPETokenizer.train`, and cross-checked against a from-scratch recount at
every merge.

It asserts:

  * the real pre-tokenizer cuts the example into the chunks the doc lists, and
    the naive `[^\\s\\w]` translation of GPT-2's pattern really does delete `_`
    (the journal-039 regex bug), while the real pattern covers every character,
  * the frequency table (unique chunk -> count) shrinks the working set,
  * at EVERY merge, the incrementally-maintained pair table equals a
    from-scratch recount of the current chunks (the correctness invariant),
  * the merges learned by an instrumented mirror of the incremental loop equal
    the real tokenizer's `_merges`,
  * the work counted (symbol positions touched) by naive-recount vs
    incremental, on the example and on real repo text,
  * block-wise chunking equals whole-corpus chunking exactly when the cut is
    "safe" -- the rule is derived by brute force over 100k+ cases -- and does
    NOT otherwise (a real counterexample to the unconditional reading of
    journal 040's "exactly identical"); measured on real repo text AND, when
    cached locally, on the real 3.46GB FineWeb-Edu corpus (every cut is safe),
  * the journal-040 memory ledger (19.4GB -> int32 buffer) and the Chinchilla
    epoch arithmetic, read off the real preset and `count_params`.

No fabricated numbers: every figure the prose quotes is printed here.
"""
from __future__ import annotations

import re
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from configs.presets import SCALE_300M  # noqa: E402
from qwenmoe.model.transformer import QwenMoEModel  # noqa: E402
from qwenmoe.tokenizer.bpe import (  # noqa: E402
    _PRE_TOKEN_PATTERN,
    ByteBPETokenizer,
    _count_pairs,
    _merge_pair,
)
from qwenmoe.train.scaling import chinchilla_budget, count_params, epochs_for_budget  # noqa: E402
from scripts.prepare_data import iter_text_blocks  # noqa: E402

import dataclasses  # noqa: E402
import logging  # noqa: E402

logging.disable(logging.WARNING)


def chunks_of(text: str) -> list[str]:
    return [m.group() for m in _PRE_TOKEN_PATTERN.finditer(text)]


def hr(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


# ----------------------------------------------------------------------
# Part 1 -- pre-tokenization
# ----------------------------------------------------------------------
hr("Part 1 -- pre-tokenization: the running example")

TEXT = "low low low low low lower lower newest newest newest widest_ widest_"
print(f"text = {TEXT!r}")
chunks = chunks_of(TEXT)
print(f"chunks ({len(chunks)}) = {chunks}")
assert "".join(chunks) == TEXT, "every character must land in some chunk"
assert chunks == ["low", " low", " low", " low", " low", " lower", " lower",
                  " newest", " newest", " newest", " widest", "_", " widest", "_"]
print("PASS  chunks reassemble the exact text")

# The journal-039 bug: a direct translation of GPT-2's `[^\s\p{L}\p{N}]`.
NAIVE = re.compile(r"'s|'t|'re|'ve|'m|'ll|'d| ?[^\W\d_]+| ?\d+| ?[^\s\w]+|\s+(?!\S)|\s+")
naive_chunks = [m.group() for m in NAIVE.finditer(TEXT)]
print(f"naive pattern drops {len(TEXT) - len(''.join(naive_chunks))} characters: "
      f"{set(TEXT) - set(''.join(naive_chunks))}")
assert "".join(naive_chunks) != TEXT and "_" not in "".join(naive_chunks)
print("PASS  the naive `[^\\s\\w]` translation silently deletes '_' (finditer skips unmatched text)")

# ----------------------------------------------------------------------
# Part 2 -- the frequency table
# ----------------------------------------------------------------------
hr("Part 2 -- frequency table: unique chunks")

freq = Counter(c.encode("utf-8") for c in chunks)
for k, v in freq.items():
    print(f"  {k!r:12} x{v}")
n_positions_stream = sum(len(c) for c in freq.elements())
n_positions_table = sum(len(c) for c in freq)
print(f"symbol positions: whole stream = {n_positions_stream}, frequency table = {n_positions_table}")
assert n_positions_stream == len(TEXT.encode()) == 68 and n_positions_table == 28

# ----------------------------------------------------------------------
# Part 3 -- incremental pair counts == recount, at every merge
# ----------------------------------------------------------------------
hr("Part 3 -- incremental updates vs a from-scratch recount")


def recount(chunk_list, freqs):
    out: dict[tuple[int, int], int] = {}
    for ch, f in zip(chunk_list, freqs):
        for p in zip(ch, ch[1:]):
            out[p] = out.get(p, 0) + f
    return out


def train_instrumented(freq_table: Counter, n_merges: int):
    """Line-for-line mirror of `ByteBPETokenizer.train`'s loop, plus (a) the
    recount invariant after every merge and (b) a work counter."""
    ch = [list(c) for c in freq_table]
    fr = list(freq_table.values())
    pair_counts, pair_chunks = {}, {}
    for i, c in enumerate(ch):
        for p in zip(c, c[1:]):
            pair_counts[p] = pair_counts.get(p, 0) + fr[i]
            pair_chunks.setdefault(p, set()).add(i)
    merges, work_incremental, work_naive_table, work_naive_stream, trace = {}, 0, 0, 0, []
    stream_len = sum(len(c) * f for c, f in zip(ch, fr))
    for i in range(n_merges):
        if not pair_counts:
            break
        best = max(pair_counts, key=pair_counts.get)
        if pair_counts[best] < 2:
            break
        new_id = 256 + i
        affected = list(pair_chunks[best])
        for idx in affected:
            c, f = ch[idx], fr[idx]
            work_incremental += len(c)
            before = len(c)
            for p in zip(c, c[1:]):
                rem = pair_counts.get(p, 0) - f
                if rem > 0:
                    pair_counts[p] = rem
                else:
                    pair_counts.pop(p, None)
                h = pair_chunks.get(p)
                if h is not None:
                    h.discard(idx)
            m = _merge_pair(c, best, new_id)
            ch[idx] = m
            stream_len -= (before - len(m)) * f
            for p in zip(m, m[1:]):
                pair_counts[p] = pair_counts.get(p, 0) + f
                pair_chunks.setdefault(p, set()).add(idx)
        pair_counts.pop(best, None)
        pair_chunks.pop(best, None)
        merges[best] = new_id
        # THE INVARIANT: incremental table == fresh recount of current chunks
        assert pair_counts == recount(ch, fr), f"incremental != recount after merge {i}"
        work_naive_table += sum(len(c) for c in ch)  # recount of the frequency table
        work_naive_stream += stream_len  # recount of the whole stream (the pre-039 trainer)
        trace.append((best, new_id, affected))
    return merges, work_incremental, work_naive_table, work_naive_stream, trace


N_MERGES = 8
merges_mirror, w_inc, w_tab, w_stream, trace = train_instrumented(freq, N_MERGES)
for best, new_id, affected in trace:
    print(f"  merge -> id {new_id}: {best}  touches {len(affected)} chunk(s)")
print(f"PASS  incremental pair table == from-scratch recount after each of {len(trace)} merges")

tok = ByteBPETokenizer(target_vocab_size=256 + N_MERGES)
tok.train(TEXT)
assert tok._merges == merges_mirror, (tok._merges, merges_mirror)
print(f"PASS  mirror's merges == real ByteBPETokenizer merges ({len(tok._merges)} merges)")
print(f"vocab bytes of last merges: {[tok._vocab[i] for i in range(256, 256 + len(tok._merges))]}")

# the reference single-sequence count agrees with the weighted count on one chunk
ref = _count_pairs(list(b" lower"))
assert ref == {(32, 108): 1, (108, 111): 1, (111, 119): 1, (119, 101): 1, (101, 114): 1}
print("PASS  `_count_pairs` (the textbook form doc 007 walks) agrees on one chunk")

print(f"\nwork on the example ({len(trace)} merges), in symbol positions scanned:")
print(f"  recount the whole stream each merge (pre-039): {w_stream}")
print(f"  recount the frequency table each merge:        {w_tab}")
print(f"  incremental (what train() does):               {w_inc}")
assert w_stream > w_tab > w_inc

# ----------------------------------------------------------------------
# Part 4 -- the same work comparison on real text (no timing, just counts)
# ----------------------------------------------------------------------
hr("Part 4 -- work counts on real repo text")

journal_text = "\n".join(
    p.read_text(encoding="utf-8") for p in sorted((ROOT / "journey_of_a_model").glob("0*.md"))
)
print(f"corpus: {len(journal_text):,} characters of the journal itself")
jf = Counter(m.group().encode() for m in _PRE_TOKEN_PATTERN.finditer(journal_text))
tokens_whole = sum(len(k) * v for k, v in jf.items())
print(f"stream positions {tokens_whole:,}; unique chunks {len(jf):,}; "
      f"table positions {sum(len(k) for k in jf):,}")
J_MERGES = 600
# the recount invariant is O(table) per merge; run it only on a slice, count work over all merges
jm, _wi, _wt, _ws, tr = train_instrumented(Counter({k: v for k, v in list(jf.items())[:400]}), 60)
print(f"invariant re-checked on a 400-chunk slice, 60 merges: PASS ({len(tr)} merges)")

# work for the FULL table with the (cheap) incremental trainer only
def incremental_only_work(freq_table, n_merges):
    ch = [list(c) for c in freq_table]; fr = list(freq_table.values())
    pc, pch = {}, {}
    for i, c in enumerate(ch):
        for p in zip(c, c[1:]):
            pc[p] = pc.get(p, 0) + fr[i]; pch.setdefault(p, set()).add(i)
    work_inc = 0; stream_len = sum(len(c) * f for c, f in zip(ch, fr)); work_naive = 0; done = 0
    for i in range(n_merges):
        if not pc: break
        best = max(pc, key=pc.get)
        if pc[best] < 2: break
        done += 1
        for idx in list(pch[best]):
            c, f = ch[idx], fr[idx]; work_inc += len(c)
            before = len(c)
            for p in zip(c, c[1:]):
                r = pc.get(p, 0) - f
                (pc.__setitem__(p, r) if r > 0 else pc.pop(p, None))
                h = pch.get(p)
                if h is not None: h.discard(idx)
            m = _merge_pair(c, best, 256 + i); ch[idx] = m
            stream_len -= (before - len(m)) * f
            for p in zip(m, m[1:]):
                pc[p] = pc.get(p, 0) + f; pch.setdefault(p, set()).add(idx)
        pc.pop(best, None); pch.pop(best, None)
        work_naive += stream_len  # a whole-stream recount scans every remaining symbol
    return done, work_inc, work_naive


done, w_inc_full, w_naive_full = incremental_only_work(jf, J_MERGES)
print(f"{done} merges: naive whole-stream recount scans {w_naive_full:,} symbol positions; "
      f"incremental touches {w_inc_full:,}  ->  {w_naive_full / w_inc_full:,.1f}x fewer")
assert w_naive_full > w_inc_full

# ----------------------------------------------------------------------
# Part 5 -- block-wise vs whole-corpus pre-tokenization (journal 040)
# ----------------------------------------------------------------------
hr("Part 5 -- does cutting at newlines preserve the chunks?")

for whole in ["a\nb", "a\n b", "a\n  b", "a\n\nb", "a  \nb", "x\n\t y"]:
    cut = whole.rindex("\n") + 1  # `iter_text_blocks` cuts just after the LAST newline of a block
    w, b = chunks_of(whole), chunks_of(whole[:cut]) + chunks_of(whole[cut:])
    print(f"  {whole!r:12} whole={w!s:36} blockwise={b!s:38} {'same' if w == b else 'DIFFERENT'}")
assert chunks_of("a\n  b") == ["a", "\n ", " b"]
assert chunks_of("a\n") + chunks_of("  b") == ["a", "\n", " ", " b"]
assert chunks_of("a\n\nb") == ["a", "\n", "\n", "b"] and chunks_of("a\n\n") == ["a", "\n\n"]
print("PASS  counterexamples: the whitespace run straddling the cut is split differently by the two sides")
print("      (the `\\s+(?!\\S)` branch gives a run's last character to the NEXT chunk)")

# The exact rule. Let a = length of the whitespace run that ENDS at the cut (it
# includes the cut's own newline) and b = length of the whitespace run that
# STARTS at the cut. Claim: the chunk list is unchanged iff b <= 1 and (b == 1 or a == 1).
# Checked by brute force over every short string on {a, ' ', \n, \t}, every cut after a \n.
import itertools  # noqa: E402


def ws_runs(s: str, cut: int) -> tuple[int, int]:
    j = cut - 1
    while j >= 0 and s[j].isspace():
        j -= 1
    k = cut
    while k < len(s) and s[k].isspace():
        k += 1
    return cut - 1 - j, k - cut


def cut_is_safe(a: int, b: int) -> bool:
    return b <= 1 and (b == 1 or a == 1)


n_cases = 0
for length in range(1, 9):
    for mid in itertools.product("a \n\t", repeat=length):
        s = "a" + "".join(mid) + "a"  # anchored by letters, so the run is always followed by a non-space
        for i, ch in enumerate(s):
            if ch != "\n":
                continue
            cut = i + 1
            a_len, b_len = ws_runs(s, cut)
            same = chunks_of(s) == chunks_of(s[:cut]) + chunks_of(s[cut:])
            assert same == cut_is_safe(a_len, b_len), (s, cut, a_len, b_len, same)
            n_cases += 1
print(f"PASS  rule 'safe iff b<=1 and (b==1 or a==1)' holds on all {n_cases:,} (string, cut) cases")

# Real measurement on the repo's own markdown (indentation- and blank-line-heavy: a pessimistic corpus).
with tempfile.TemporaryDirectory() as td:
    path = Path(td) / "corpus.txt"
    path.write_text(journal_text, encoding="utf-8")
    for block_chars in (2048, 8192, 65536):
        blocks = list(iter_text_blocks(path, block_chars))
        assert "".join(blocks) == journal_text
        whole_freq = Counter(chunks_of(journal_text))
        block_freq: Counter = Counter()
        for bl in blocks:
            block_freq.update(chunks_of(bl))
        cuts = len(blocks) - 1
        unsafe = 0
        for i in range(1, len(blocks)):
            a_len = len(blocks[i - 1]) - len(blocks[i - 1].rstrip())
            b_len = len(blocks[i]) - len(blocks[i].lstrip())
            unsafe += not cut_is_safe(a_len, b_len)
        diff = sum((whole_freq - block_freq).values()) + sum((block_freq - whole_freq).values())
        print(f"  block_chars={block_chars:>6}: {cuts:>4} cuts, {unsafe:>3} unsafe, "
              f"chunk-count disagreements = {diff}")
        # an unsafe cut turns one whole-corpus chunk into two (1 + 2 = at most 3 Counter differences)
        assert (unsafe > 0 or diff == 0) and diff <= 3 * unsafe
print("PASS  no unsafe cut -> no disagreement; each unsafe cut costs at most 3 chunk-count differences")

# Real measurement on the real FineWeb-Edu corpus, when it is cached locally (gitignored, 3.46GB).
real = ROOT / ".cache" / "data" / "fineweb_edu_3500Mchars.txt"
if real.exists():
    from scripts.prepare_data import _BLOCK_CHARS  # noqa: E402

    prev = None
    n_cuts = n_unsafe = 0
    kinds: Counter = Counter()
    for block in iter_text_blocks(real, _BLOCK_CHARS):
        if prev is not None:
            a_len = len(prev) - len(prev.rstrip())
            b_len = len(block) - len(block.lstrip())
            n_cuts += 1
            n_unsafe += not cut_is_safe(a_len, b_len)
            kinds[(a_len, b_len)] += 1
        prev = block[-16:]
    print(f"real FineWeb-Edu corpus at {_BLOCK_CHARS // 2**20} MiB blocks: {n_cuts} cuts, "
          f"{n_unsafe} unsafe; (a,b) histogram = {dict(sorted(kinds.items()))}")
    assert n_unsafe == 0
    print("PASS  every real cut is safe -> journal 040's token-for-token equality is not in question")
else:
    print("SKIP  real-corpus cut measurement (.cache/data/fineweb_edu_3500Mchars.txt not present)")

# ----------------------------------------------------------------------
# Part 6 -- the streaming memory ledger (journal 040) and epoch arithmetic
# ----------------------------------------------------------------------
hr("Part 6 -- memory ledger and the epoch count")

N_TOK = 999_329_599          # journal 040's measured token count for the full shard
CORPUS_CHARS = 3.46e9        # journal 040's measured corpus size (ASCII-folded)
GB = 1e9
list_ptrs = N_TOK * 8
int64_tensor = N_TOK * 8
int32_buf = N_TOK * 4
old_total = CORPUS_CHARS + list_ptrs + int64_tensor
print(f"corpus string     {CORPUS_CHARS / GB:6.2f} GB (1 byte per ASCII char; journal 040's figure)")
print(f"list of pointers  {list_ptrs / GB:6.2f} GB  = {N_TOK:,} x 8 B")
print(f"int64 tensor      {int64_tensor / GB:6.2f} GB  = {N_TOK:,} x 8 B")
print(f"old peak (all three live at once) = {old_total / GB:.2f} GB   (24 GB machine)")
print(f"new: one int32 buffer             = {int32_buf / GB:.2f} GB   (torch.from_numpy wraps it, no copy)")
assert round(old_total / GB, 1) == 19.4
assert int32_buf * 2 == int64_tensor and round(int32_buf / GB, 2) == 4.00

cfg = dataclasses.replace(SCALE_300M, vocab_size=4097)
active = count_params(QwenMoEModel(cfg)).active
floor = chinchilla_budget(active)
print(f"\nSCALE_300M at vocab 4097: active params = {active:,}")
print(f"Chinchilla floor 20 x active = {floor:,} tokens; shard = {N_TOK:,} tokens")
epochs = epochs_for_budget(floor, N_TOK)
print(f"epochs over the whole shard = {epochs:.2f}")
assert round(epochs, 2) == 2.50
print("PASS  2.50 epochs (journal 040), computed from the real preset and real count_params")

print("\nAll assertions passed -- every number in 039-bpe-incremental-counts-and-streaming.md is real.")
