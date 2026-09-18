"""Reproduces every hand-computed number in `008-data-pipeline.md`.

Run it:  uv run python docs/math/verify_008_data_pipeline.py

It carries ONE running example -- four toy "documents" packed into a single
flat, `<eos>`-separated token stream of 24 ids -- through every function in
`qwenmoe/data/dataset.py` (and the caching branch of
`qwenmoe/data/download.py`), driving the REAL module code, printing every
intermediate, and `assert`ing:

  * `_document_ids`: the cumsum-of-"previous-token-was-a-separator" prefix
    sum, checked against a hand-rolled Python prefix-sum loop (the two
    "paths" -- vectorised `torch.cumsum` and the scalar definition -- must
    agree exactly),
  * `_split_into_documents`: 4 documents of lengths [5, 6, 5, 8], the last
    kept as-is because it has no terminating separator,
  * `train_val_split(mode="tail")`: floor(N * val_fraction) held out as one
    contiguous slice from the end (N=24: 0.25 -> 6, 0.1 -> 2), and
    train ++ val reconstructs the input byte-for-byte,
  * `train_val_split(mode="documents")`: whole documents taken from the
    tail until >= val_fraction of tokens are held out (overshoots to 8/24
    here because no document is ever split), plus the >=2-document and
    eos_id guards,
  * `TokenDataset.__len__` == N - seq_len, checked against enumerating
    every window, and `__getitem__`'s y = x shifted one position right,
  * `TokenDataset(return_segments=True)`: per-window document ids rebased
    to start at 0, on windows inside / straddling a boundary,
  * `make_fixed_slice`: chunk_size = max(1, min(1024, N // 4)) = 6, four
    chunks, `torch.randperm(4)` under a seeded generator == [1, 3, 2, 0]
    for seed 1 (the unique Fisher-Yates swap path (1, 2, 0) is replayed by
    hand and shown to land there), determinism for a fixed seed,
    divergence for seed 0 vs seed 1, a seed-0/seed-2 collision, within-chunk
    order preserved, and the n_tokens guards,
  * the whole pipeline end to end: make_fixed_slice(24 -> 20, seed 1) ->
    train_val_split(0.25, tail) -> TokenDataset(seq_len=4) -> DataLoader
    (batch_size=2) -> the exact first (x, y) batch,
  * `download_corpus`: the `dest.exists()` short-circuit returns the cached
    path WITHOUT any network call (urllib is monkeypatched to explode), and
    an unknown corpus name raises ValueError.

There is no fused / optimised code path for the data pipeline -- the module
*is* the explicit path -- so, unlike the attention / MoE verifiers, the
only "two paths agree" check is the vectorised-cumsum vs. scalar-prefix-sum
one in Part 6. No fabricated numbers: this is the source of truth the prose
quotes.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from qwenmoe.data import download as download_mod
from qwenmoe.data.dataset import (
    _DEFAULT_CHUNK_SIZE,
    TokenDataset,
    _as_token_tensor,
    _document_ids,
    _split_into_documents,
    make_fixed_slice,
    train_val_split,
)
from qwenmoe.data.download import download_corpus

torch.set_printoptions(precision=6, sci_mode=False)

# ----------------------------------------------------------------------
# Part 1 -- the running example.
# Four toy documents, packed into one flat stream separated by <eos> = 0.
# The last document has NO trailing separator (exercises the "trailing
# document kept as-is" branch).  No content token is 0.
# ----------------------------------------------------------------------
EOS_ID = 0
DOC_A = [3, 1, 4, 1]
DOC_B = [5, 9, 2, 6, 2]
DOC_C = [8, 5, 3, 5]
DOC_D = [6, 1, 4, 1, 3, 5, 2, 4]
ALL_TOKENS = DOC_A + [EOS_ID] + DOC_B + [EOS_ID] + DOC_C + [EOS_ID] + DOC_D
N = len(ALL_TOKENS)

print("================ running example: the data pipeline ================")
print(f"documents (pre-packing): {DOC_A}, {DOC_B}, {DOC_C}, {DOC_D}")
print(f"eos_id = {EOS_ID}")
print(f"ALL_TOKENS = {ALL_TOKENS}   (N = {N})")
assert ALL_TOKENS == [3, 1, 4, 1, 0, 5, 9, 2, 6, 2, 0, 8, 5, 3, 5, 0, 6, 1, 4, 1, 3, 5, 2, 4]
assert N == 24


# ----------------------------------------------------------------------
# Part 3 -- _as_token_tensor: the dtype gate.
#
# Renamed from _as_long_tensor in journal 040, and the semantics changed
# with it: a tensor now keeps the dtype it arrives in. `prepare_data.py`
# writes int32, and upcasting a ~1B-token stream to int64 here would
# double it to ~8GB for no benefit -- TokenDataset.__getitem__ casts each
# seq_len window instead, which is all the embedding lookup needs. A
# plain Python list has no dtype to preserve, so it still becomes int64.
# ----------------------------------------------------------------------
print("\n================ Part 3 -- _as_token_tensor ================")
from_list = _as_token_tensor(ALL_TOKENS)
from_tensor = _as_token_tensor(torch.tensor(ALL_TOKENS, dtype=torch.int32))
print(f"  list  -> {from_list.dtype}, shape {tuple(from_list.shape)}")
print(f"  int32 tensor -> {from_tensor.dtype}  (dtype preserved)")
assert from_list.dtype == torch.long
assert from_tensor.dtype == torch.int32
assert from_list.tolist() == ALL_TOKENS

T = _as_token_tensor(ALL_TOKENS)


# ----------------------------------------------------------------------
# Part 6 -- _document_ids: prefix sum of "the previous token was <eos>".
# (placed before the split parts because TokenDataset(return_segments) and
#  the prose both build on it)
# ----------------------------------------------------------------------
print("\n================ Part 6 -- _document_ids ================")
doc_ids = _document_ids(T, EOS_ID)
print(f"  _document_ids(ALL_TOKENS, {EOS_ID}) = {doc_ids.tolist()}")

# hand-rolled scalar prefix sum from the same definition:
#   starts[0] = 0;  starts[i] = 1 iff tokens[i-1] == eos_id
#   doc_id[i] = starts[0] + starts[1] + ... + starts[i]
starts = [0] + [1 if ALL_TOKENS[i - 1] == EOS_ID else 0 for i in range(1, N)]
hand = []
running = 0
for s in starts:
    running += s
    hand.append(running)
print(f"  hand scalar prefix sum        = {hand}")
assert doc_ids.tolist() == hand == [
    0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3
]
# the <eos> at positions 4/10/15 belongs to the document it terminates
assert doc_ids[4].item() == 0 and doc_ids[10].item() == 1 and doc_ids[15].item() == 2


# ----------------------------------------------------------------------
# Part 5a -- _split_into_documents.
# ----------------------------------------------------------------------
print("\n================ Part 5a -- _split_into_documents ================")
docs = _split_into_documents(T, EOS_ID)
for i, d in enumerate(docs):
    print(f"  document {i}: {d.tolist()}   (length {d.numel()})")
assert [d.tolist() for d in docs] == [
    [3, 1, 4, 1, 0],
    [5, 9, 2, 6, 2, 0],
    [8, 5, 3, 5, 0],
    [6, 1, 4, 1, 3, 5, 2, 4],
]
assert [d.numel() for d in docs] == [5, 6, 5, 8]
assert sum(d.numel() for d in docs) == N


# ----------------------------------------------------------------------
# Part 4 -- train_val_split, mode="tail".
# ----------------------------------------------------------------------
print("\n================ Part 4 -- train_val_split(mode='tail') ================")
for vf in (0.25, 0.1):
    n_val = int(N * vf)
    tr, va = train_val_split(ALL_TOKENS, val_fraction=vf)  # tail is the default
    print(f"  val_fraction={vf}:  N*vf = {N * vf!r} -> floor = {n_val};  "
          f"len(train)={len(tr)} len(val)={len(va)}")
    assert len(va) == n_val
    assert len(tr) == N - n_val
    assert torch.equal(torch.cat([tr, va]), T)  # contiguous, order-preserving

tr25, va25 = train_val_split(ALL_TOKENS, val_fraction=0.25)
print(f"  train(0.25) = {tr25.tolist()}")
print(f"  val(0.25)   = {va25.tolist()}   <- a contiguous cut, mid-document D")
assert va25.tolist() == [4, 1, 3, 5, 2, 4]
assert train_val_split(ALL_TOKENS, 0.1)[1].tolist() == [2, 4]

for bad in (0.0, 1.0, -0.1, 1.1):
    try:
        train_val_split(ALL_TOKENS, val_fraction=bad)
    except ValueError:
        pass
    else:
        raise AssertionError(f"expected ValueError for val_fraction={bad}")


# ----------------------------------------------------------------------
# Part 5b -- train_val_split, mode="documents".
# ----------------------------------------------------------------------
print("\n================ Part 5b -- train_val_split(mode='documents') ================")
target = N * 0.25
print(f"  target_val_tokens = N * 0.25 = {target}")
tr_d, va_d = train_val_split(ALL_TOKENS, val_fraction=0.25, mode="documents", eos_id=EOS_ID)
print(f"  walking documents from the tail: D(8) -> accumulated 8 >= {target} -> stop")
print(f"  train = {tr_d.tolist()}   (documents A+B+C, {tr_d.numel()} tokens)")
print(f"  val   = {va_d.tolist()}   (document D, {va_d.numel()} tokens = "
      f"{va_d.numel() / N:.1%} >= 25%)")
assert tr_d.tolist() == [3, 1, 4, 1, 0, 5, 9, 2, 6, 2, 0, 8, 5, 3, 5, 0]
assert va_d.tolist() == DOC_D
assert tr_d.numel() + va_d.numel() == N
assert tr_d[-1].item() == EOS_ID          # train ends exactly on a separator
assert va_d.numel() >= 0.25 * N

try:
    train_val_split(ALL_TOKENS, mode="documents")  # no eos_id
except ValueError:
    pass
else:
    raise AssertionError("expected ValueError: documents mode needs eos_id")

try:
    train_val_split([7, 7, 7, 7], mode="documents", eos_id=EOS_ID)  # < 2 documents
except ValueError:
    pass
else:
    raise AssertionError("expected ValueError: < 2 documents")

try:
    train_val_split(ALL_TOKENS, mode="sideways")
except ValueError:
    pass
else:
    raise AssertionError("expected ValueError: unknown mode")


# ----------------------------------------------------------------------
# Part 7 -- TokenDataset.
# ----------------------------------------------------------------------
print("\n================ Part 7 -- TokenDataset ================")
SEQ_LEN = 4
ds_full = TokenDataset(ALL_TOKENS, seq_len=SEQ_LEN)
print(f"  TokenDataset(ALL_TOKENS, seq_len={SEQ_LEN}):  len = {len(ds_full)}  "
      f"(= N - seq_len = {N} - {SEQ_LEN})")
assert len(ds_full) == N - SEQ_LEN == 20

# __len__ vs. enumerating valid windows: window i needs x = tokens[i : i+L]
# AND y = tokens[i+1 : i+L+1], so i + L + 1 <= N, i.e. i in 0 .. N-L-1.
valid_starts = [i for i in range(N) if i + SEQ_LEN + 1 <= N]
assert len(valid_starts) == len(ds_full) == N - SEQ_LEN

x0, y0 = ds_full[0]
x3, y3 = ds_full[3]
print(f"  window 0:  x = {x0.tolist()}   y = {y0.tolist()}")
print(f"  window 3:  x = {x3.tolist()}   y = {y3.tolist()}")
assert x0.tolist() == [3, 1, 4, 1] and y0.tolist() == [1, 4, 1, 0]
assert x3.tolist() == [1, 0, 5, 9] and y3.tolist() == [0, 5, 9, 2]
# y is x shifted one position right: y[j] == x[j+1] for j < seq_len - 1
for xi, yi in (ds_full[k] for k in range(len(ds_full))):
    assert yi[:-1].tolist() == xi[1:].tolist()
assert x0.dtype == torch.long and y0.dtype == torch.long

for bad_seq_len in (0, -1):
    try:
        TokenDataset(ALL_TOKENS, seq_len=bad_seq_len)
    except ValueError:
        pass
    else:
        raise AssertionError(f"expected ValueError for seq_len={bad_seq_len}")
try:
    TokenDataset([1, 2, 3], seq_len=4)  # numel < seq_len + 1
except ValueError:
    pass
else:
    raise AssertionError("expected ValueError: too few tokens")

# --- return_segments=True: per-window document ids, rebased to start at 0 ---
print("\n  -- return_segments=True --")
ds_seg = TokenDataset(ALL_TOKENS, seq_len=SEQ_LEN, return_segments=True, eos_id=EOS_ID)
for idx in (0, 3, 8):
    x, y, seg = ds_seg[idx]
    raw = doc_ids[idx: idx + SEQ_LEN].tolist()
    print(f"  window {idx}: x = {x.tolist()}  doc_ids {raw}  -> segments {seg.tolist()}")
assert ds_seg[0][2].tolist() == [0, 0, 0, 0]      # wholly inside document 0
assert ds_seg[3][2].tolist() == [0, 0, 1, 1]      # straddles the A|B boundary
assert ds_seg[8][2].tolist() == [0, 0, 0, 1]      # straddles the B|C boundary
assert len(ds_full[0]) == 2 and len(ds_seg[0]) == 3

try:
    TokenDataset(ALL_TOKENS, seq_len=SEQ_LEN, return_segments=True)  # no eos_id
except ValueError:
    pass
else:
    raise AssertionError("expected ValueError: return_segments needs eos_id")


# ----------------------------------------------------------------------
# Part 8 -- make_fixed_slice.
# ----------------------------------------------------------------------
print("\n================ Part 8 -- make_fixed_slice ================")
chunk_size = max(1, min(_DEFAULT_CHUNK_SIZE, N // 4))
print(f"  chunk_size = max(1, min({_DEFAULT_CHUNK_SIZE}, {N} // 4)) = {chunk_size}")
assert chunk_size == 6
chunks = list(torch.split(T, chunk_size))
for i, c in enumerate(chunks):
    print(f"  chunk {i}: {c.tolist()}")
assert [c.tolist() for c in chunks] == [
    [3, 1, 4, 1, 0, 5],
    [9, 2, 6, 2, 0, 8],
    [5, 3, 5, 0, 6, 1],
    [4, 1, 3, 5, 2, 4],
]

print("\n  torch.randperm(4) under a seeded generator:")
perms = {}
for seed in (0, 1, 2, 3):
    g = torch.Generator().manual_seed(seed)
    perms[seed] = torch.randperm(len(chunks), generator=g).tolist()
    print(f"    seed {seed}: {perms[seed]}")
assert perms[1] == [1, 3, 2, 0]
assert perms[0] == perms[2]          # a collision: distinct seeds, same ordering
assert perms[0] != perms[1]

# the UNIQUE Fisher-Yates swap path to [1, 3, 2, 0]:
#   for i in 0..n-2:  z = random() % (n - i);  swap(a[i], a[i + z])
# back-solving the output gives (z0, z1, z2) = (1, 2, 0):
a = [0, 1, 2, 3]
swap_trace = []
for i, z in enumerate((1, 2, 0)):
    j = i + z
    a[i], a[j] = a[j], a[i]
    swap_trace.append((i, z, j, list(a)))
    print(f"    i={i}  z={z}  swap(a[{i}], a[{j}]) -> {a}")
assert a == perms[1] == [1, 3, 2, 0]

# determinism, divergence, collision at the slice level
s1a = make_fixed_slice(ALL_TOKENS, n_tokens=20, seed=1)
s1b = make_fixed_slice(ALL_TOKENS, n_tokens=20, seed=1)
s0 = make_fixed_slice(ALL_TOKENS, n_tokens=20, seed=0)
print(f"\n  make_fixed_slice(ALL_TOKENS, 20, seed=1) = {s1a.tolist()}")
print(f"  make_fixed_slice(ALL_TOKENS, 20, seed=0) = {s0.tolist()}")
assert torch.equal(s1a, s1b)                       # same seed -> identical
assert not torch.equal(s1a, s0)                    # seed 0 vs 1 -> different
assert s1a.tolist() == [9, 2, 6, 2, 0, 8, 4, 1, 3, 5, 2, 4, 5, 3, 5, 0, 6, 1, 3, 1]
assert s1a.shape == (20,)

# within-chunk order preserved: the full (untruncated) shuffle is exactly a
# reordering of the four original chunks.
full_shuffle = make_fixed_slice(ALL_TOKENS, n_tokens=N, seed=1)
blocks = [b.tolist() for b in torch.split(full_shuffle, chunk_size)]
assert blocks == [chunks[i].tolist() for i in perms[1]]
print(f"  full shuffle (n_tokens={N}) splits back into the original chunks, "
      f"in order {perms[1]}")

for bad in (0, -3):
    try:
        make_fixed_slice(ALL_TOKENS, n_tokens=bad, seed=0)
    except ValueError:
        pass
    else:
        raise AssertionError(f"expected ValueError for n_tokens={bad}")
try:
    make_fixed_slice(ALL_TOKENS, n_tokens=N + 1, seed=0)
except ValueError:
    pass
else:
    raise AssertionError("expected ValueError: n_tokens > available")


# ----------------------------------------------------------------------
# Part 9 -- the whole pipeline, end to end.
# ----------------------------------------------------------------------
print("\n================ Part 9 -- end to end ================")
sliced = make_fixed_slice(ALL_TOKENS, n_tokens=20, seed=1)
train_tokens, val_tokens = train_val_split(sliced, val_fraction=0.25)  # tail
print(f"  fixed slice (20)      = {sliced.tolist()}")
print(f"  train (15)            = {train_tokens.tolist()}")
print(f"  val   (5)             = {val_tokens.tolist()}")
assert train_tokens.tolist() == [9, 2, 6, 2, 0, 8, 4, 1, 3, 5, 2, 4, 5, 3, 5]
assert val_tokens.tolist() == [0, 6, 1, 3, 1]

train_ds = TokenDataset(train_tokens, seq_len=SEQ_LEN)
val_ds = TokenDataset(val_tokens, seq_len=SEQ_LEN)
print(f"  len(train_ds)         = {len(train_ds)}   (15 - 4)")
print(f"  len(val_ds)           = {len(val_ds)}   (5 - 4)")
assert len(train_ds) == 11 and len(val_ds) == 1

loader = DataLoader(train_ds, batch_size=2, shuffle=False)
xb, yb = next(iter(loader))
print(f"  first batch x         = {xb.tolist()}   shape {tuple(xb.shape)}")
print(f"  first batch y         = {yb.tolist()}   shape {tuple(yb.shape)}")
assert xb.shape == (2, SEQ_LEN) and yb.shape == (2, SEQ_LEN)
assert xb.tolist() == [[9, 2, 6, 2], [2, 6, 2, 0]]
assert yb.tolist() == [[2, 6, 2, 0], [6, 2, 0, 8]]

xv, yv = val_ds[0]
print(f"  val window 0          = x {xv.tolist()}  y {yv.tolist()}")
assert xv.tolist() == [0, 6, 1, 3] and yv.tolist() == [6, 1, 3, 1]
try:
    TokenDataset(val_tokens, seq_len=5)  # 5 tokens, needs 6
except ValueError:
    pass
else:
    raise AssertionError("expected ValueError: val split too short for seq_len=5")


# ----------------------------------------------------------------------
# Part 2 -- download_corpus: the caching short-circuit, no network.
# ----------------------------------------------------------------------
print("\n================ Part 2 -- download_corpus (offline) ================")


def _explode(*_a, **_k):  # any network call is a test failure
    raise AssertionError("download_corpus touched the network on a cache hit")


original_urlopen = download_mod.urllib.request.urlopen
download_mod.urllib.request.urlopen = _explode
try:
    with tempfile.TemporaryDirectory() as d:
        cache = Path(d)
        (cache / "tiny_shakespeare.txt").write_text("FAKE CACHED SHAKESPEARE")
        got = download_corpus("tiny_shakespeare", cache)
        print(f"  cache hit -> returned {got.name!r} without downloading")
        assert got == cache / "tiny_shakespeare.txt"
        assert got.read_text() == "FAKE CACHED SHAKESPEARE"

        try:
            download_corpus("tiny_homer", cache)
        except ValueError as ex:
            print(f"  unknown corpus -> ValueError: {ex}")
        else:
            raise AssertionError("expected ValueError for unknown corpus")
finally:
    download_mod.urllib.request.urlopen = original_urlopen

print(f"  (constants: url ...{download_mod._TINY_SHAKESPEARE_URL[-32:]!r}, "
      f"hf file {download_mod._TINY_STORIES_FILENAME!r})")


print("\nAll assertions passed -- every number in 008-data-pipeline.md is real.")
