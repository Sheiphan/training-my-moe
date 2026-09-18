"""Reproduces every hand-computed number in `007-tokenizers.md`.

Run it:  uv run python docs/math/verify_007_tokenizers.py

It trains the exact tiny `ByteBPETokenizer` the derivation uses
(target_vocab_size=262, corpus "aaacaaacabab", eos_token off), driving the
REAL module functions `_count_pairs` / `_merge_pair` and the real
`ByteBPETokenizer.train` / `.encode` / `.decode` / `.save` / `.load`,
prints every intermediate (each merge round's pair table, the argmax, the
merged id list, every new vocab entry), and `assert`s:

  * the 4 learned merges and the final vocab_size (260, *below* the 262
    target -- the corpus ran out of frequency>=2 pairs: an early stop),
  * encode(corpus) == [258, 258, 259, 259], which decodes back byte-for-byte,
  * the priority-order tie-break (`min` by learned id), shown on encode("aab"),
  * greedy-encode suboptimality, shown on encode("aaaac"),
  * byte-level coverage of an unseen-Unicode string ("café <coffee>"),
  * decode(errors="replace") on a non-round-trip id list,
  * decode of an out-of-range id raising ValueError,
  * save/load round-tripping the merges and the eos flag,
  * the eos_token=True variant reserving exactly one id and dropping it on decode,
  * CharTokenizer's sorted-set vocab on the same corpus, for contrast.

There is no fused/optimised code path for the tokenizer -- the module *is*
the explicit path -- so, unlike the attention/MoE verifiers, there is no
two-path agreement check. No fabricated numbers: this is the source of
truth the prose quotes.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from qwenmoe.config import DataConfig
from qwenmoe.tokenizer.base import build_tokenizer
from qwenmoe.tokenizer.bpe import ByteBPETokenizer, _count_pairs, _merge_pair
from qwenmoe.tokenizer.char import CharTokenizer

CORPUS = "aaacaaacabab"
TARGET = 262


def show(label: str, ids: list[int]) -> None:
    print(f"  {label}: {ids}   (length {len(ids)})")


def piece(vocab: dict[int, bytes], i: int) -> str:
    return vocab[i].decode("utf-8", errors="replace")


# ----------------------------------------------------------------------
# Part 2 -- train(), one merge round at a time, through the real
# _count_pairs / _merge_pair helpers.
# ----------------------------------------------------------------------
print("================ running example: ByteBPETokenizer ================")
print(f"corpus = {CORPUS!r}    target_vocab_size = {TARGET}")

ids0 = list(CORPUS.encode("utf-8"))
print(f"\nStep 1  text.encode('utf-8') -> {ids0}   (length {len(ids0)})")
assert ids0 == [97, 97, 97, 99, 97, 97, 97, 99, 97, 98, 97, 98]

vocab = {i: bytes([i]) for i in range(256)}
merges: dict[tuple[int, int], int] = {}
work = list(ids0)

for r in range(TARGET - 256):
    counts = _count_pairs(work)
    if not counts:
        print(f"\nRound {r + 1}: sequence has no pairs left -> stop")
        break
    best = max(counts, key=counts.get)
    print(f"\nRound {r + 1}:  (num_merges budget still open)")
    print("  pair counts, in dict insertion order == first left-to-right occurrence:")
    for p, c in counts.items():
        print(f"    ({p[0]:>3}, {p[1]:>3})  {piece(vocab, p[0])!r:>7} + {piece(vocab, p[1])!r:<7} -> {c}")
    if counts[best] < 2:
        print(f"  argmax = ({best[0]}, {best[1]}) has count {counts[best]} < 2  ->  EARLY STOP")
        break
    new_id = 256 + r
    print(f"  argmax = ({best[0]}, {best[1]})  count {counts[best]}   (first key reaching the max)")
    print(f"  new id {new_id};  vocab[{new_id}] = vocab[{best[0]}] + vocab[{best[1]}] "
          f"= {vocab[best[0]]!r} + {vocab[best[1]]!r} = {(vocab[best[0]] + vocab[best[1]])!r}")
    work = _merge_pair(work, best, new_id)
    merges[best] = new_id
    vocab[new_id] = vocab[best[0]] + vocab[best[1]]
    show("sequence after merge", work)

print("\nlearned merges (pair -> new id):")
for p, nid in merges.items():
    print(f"  ({p[0]}, {p[1]}) -> {nid}   ({piece(vocab, nid)!r})")
print(f"vocab_size = {len(vocab)}   (target was {TARGET}: {len(vocab)} < {TARGET} -> early stop)")

# --- assert the hand trace against the real .train() ------------------
real = ByteBPETokenizer(target_vocab_size=TARGET)
real.train(CORPUS)
assert real._merges == merges == {
    (97, 97): 256, (256, 97): 257, (257, 99): 258, (97, 98): 259
}, real._merges
assert real._vocab == vocab
assert real.vocab_size == 260 < TARGET
assert vocab[256] == b"aa" and vocab[257] == b"aaa"
assert vocab[258] == b"aaac" and vocab[259] == b"ab"


# ----------------------------------------------------------------------
# Part 3 -- encode(corpus): priority-ordered replay of the merge rules.
# ----------------------------------------------------------------------
print("\n================ encode(corpus) -- priority-ordered replay ================")
e = list(CORPUS.encode("utf-8"))
show("start (raw bytes)", e)
p = 0
while len(e) >= 2:
    p += 1
    counts = _count_pairs(e)
    mergeable = [pr for pr in counts if pr in merges]
    if not mergeable:
        print(f"  pass {p}: no adjacent pair is a learned merge -> stop")
        break
    best = min(mergeable, key=lambda pr: merges[pr])
    print(f"  pass {p}: mergeable pairs present = {mergeable}")
    print(f"          min by learned id -> ({best[0]}, {best[1]}) = id {merges[best]}")
    e = _merge_pair(e, best, merges[best])
    show(f"        after pass {p}", e)

assert e == [258, 258, 259, 259], e
assert real.encode(CORPUS) == e
decoded = real.decode(e)
print(f"\n  decode({e}) = {decoded!r}")
assert decoded == CORPUS


# ----------------------------------------------------------------------
# Part 4 -- the behaviours the spine touches once each.
# ----------------------------------------------------------------------
print("\n================ Part 4 -- single-shot behaviours ================")

# 4a. priority tie-break decides the RESULT on overlapping merges.
#     "aab": (97,97)->256 and (97,98)->259 both apply and overlap on the
#     middle 'a'. min(256, 259) = 256 wins, giving [256, 98] ("aa","b").
assert real.encode("aab") == [256, 98]
alt = _merge_pair(list("aab".encode("utf-8")), (97, 98), 259)
assert alt == [97, 259] and alt != [256, 98]
print(f"  encode('aab')          = {real.encode('aab')}   ('aa','b')")
print(f"  if (97,98) went first  = {alt}   ('a','ab')  <- different, so order matters")

# 4b. a learned merge is a real, directly usable token, not just a
#     stepping stone: encode('aa') -> [256], encode('aad') -> [256, 100].
assert real.encode("aa") == [256]
assert real.encode("aad") == [256, 100]
print(f"  encode('aa')           = {real.encode('aa')}       (token 256 used on its own)")
print(f"  encode('aad')          = {real.encode('aad')}   ('aa','d')")

# 4c. greedy-encode suboptimality: "aaaac" -> [256,256,99]; the 2-token
#     [97, 258] ('a','aaac') is valid and shorter but unreachable, because
#     the leftmost (97,97) merge fires first and strands the 'c'.
assert real.encode("aaaac") == [256, 256, 99]
assert real.decode([97, 258]) == "aaaac"
print(f"  encode('aaaac')        = {real.encode('aaaac')}   (3 tokens)")
print(f"  decode([97, 258])      = {real.decode([97, 258])!r}   (2 tokens, valid, not found by greedy replay)")

# 4d. byte-level coverage of characters never seen in training.
unseen = "café ☕"  # 'caf' + e-acute (U+00E9) + ' ' + HOT BEVERAGE (U+2615)
enc = real.encode(unseen)
raw = list(unseen.encode("utf-8"))
print(f"\n  encode({unseen!r}) = {enc}")
print(f"    == raw utf-8 bytes  {raw}   (no learned merge applies)")
assert enc == raw
assert real.decode(enc) == unseen
assert len("é".encode("utf-8")) == 2      # 'e' with acute accent: 2 bytes
assert len("☕".encode("utf-8")) == 3      # hot-beverage sign: 3 bytes

# 4e. decode(errors="replace") on ids that don't form valid UTF-8.
assert real.decode([128]) == "�"
assert real.decode([256, 128]) == "aa�"
print(f"\n  decode([256, 128])     = {real.decode([256, 128])!r}   (0x80 alone is not valid UTF-8)")

# 4f. an out-of-range id is a ValueError, not a raw KeyError.
try:
    real.decode([260])
except ValueError as ex:
    print(f"  decode([260])          -> ValueError: {ex}")
else:
    raise AssertionError("expected ValueError for id 260")

# 4g. save / load round trip.
with tempfile.TemporaryDirectory() as d:
    path = Path(d) / "bpe.json"
    real.save(path)
    payload = json.loads(path.read_text())
    print(f"\n  saved JSON: {payload}")
    assert payload == {
        "target_vocab_size": 262,
        "merges": [[97, 97], [256, 97], [257, 99], [97, 98]],
        "eos_token": False,
    }
    loaded = ByteBPETokenizer()  # default target 4096; load() must override
    loaded.load(path)
    assert loaded._merges == merges
    assert loaded.vocab_size == 260
    assert loaded.encode(CORPUS) == [258, 258, 259, 259]
    assert loaded.decode(loaded.encode(CORPUS)) == CORPUS

# 4h. eos_token=True: exactly one id past the byte+merge vocab, never
#     emitted by encode, dropped by decode.
eos = ByteBPETokenizer(target_vocab_size=TARGET, eos_token=True)
eos.train(CORPUS)
assert eos.vocab_size == 261 and eos.eos_id == 260
assert eos.eos_id not in eos.encode(CORPUS)
mixed = eos.encode("aa") + [eos.eos_id] + eos.encode("bb")
print(f"\n  eos: encode('aa') + [eos_id] + encode('bb') = {mixed}")
assert eos.decode(mixed) == "aabb"

# 4i. CharTokenizer on the same corpus, for contrast.
ch = CharTokenizer()
ch.train(CORPUS)
print(f"\n  CharTokenizer.train({CORPUS!r}) -> sorted vocab {sorted(set(CORPUS))}, "
      f"vocab_size {ch.vocab_size}")
assert ch.vocab_size == 3
assert ch.encode("cab") == [2, 0, 1]        # sorted: 'a'<'b'<'c' -> 0,1,2
assert ch.decode([2, 0, 1]) == "cab"
try:
    ch.encode("d")
except ValueError as ex:
    print(f"  CharTokenizer.encode('d') -> ValueError: {ex}")
else:
    raise AssertionError("expected ValueError for unseen character 'd'")

# 4j. the build_tokenizer factory wires bpe_vocab_size through as the target.
cfg = DataConfig(corpus="tiny_stories", tokenizer_type="bpe", bpe_vocab_size=TARGET)
built = build_tokenizer(cfg)
assert isinstance(built, ByteBPETokenizer)
built.train(CORPUS)
assert built.encode(CORPUS) == [258, 258, 259, 259]

print("\nAll assertions passed -- every number in 007-tokenizers.md is real.")
