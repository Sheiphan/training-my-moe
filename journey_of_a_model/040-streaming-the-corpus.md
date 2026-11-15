# 040 — Streaming the corpus: preparing a billion tokens on a 24GB laptop

**Type:** tooling (direct follow-up to 039) · **Date:** 2026-09-18

> **Full derivation:** [`docs/math/039-bpe-incremental-counts-and-streaming.md`](../docs/math/039-bpe-incremental-counts-and-streaming.md)
> (Parts 5–6) — the "exactly identical" claim stated as a condition on where the cut falls, checked against every real
> cut, and the 19.45 GB → 4.00 GB memory ledger.

## TL;DR

- **The blocker entry 039 left behind:** the shard on disk holds just under
  1B tokens (039 estimated 1.01B from average document length; the real
  figure turned out to be 999,329,599), enough to train `SCALE_300M` at its
  Chinchilla floor inside the 4-epoch repetition guidance — but
  `prepare_data.py` could not build a dataset that size. It read the whole
  corpus into one string, encoded it
  into one Python `list[int]`, and handed that to `torch.tensor`. At the
  999M tokens this shard actually produces, that is 3.46GB of corpus string
  + 7.99GB of list pointers + 7.99GB of `int64` tensor, all alive at once:
  **19.4GB before any per-object overhead**, on a 24GB machine that journal
  037 already measured hitting a swap cliff with ordinary apps open. Not a
  tight fit — a hard stop.
- **The fix:** stream everything. The corpus is walked in newline-aligned
  blocks, the tokenizer trains from that block iterator instead of a
  string, and encoded ids go straight into a preallocated `int32` array
  that `torch.from_numpy` wraps without copying. Peak memory is now set by
  the **output** token array (4.0GB at 1B tokens), not by the corpus.
- **The result:** the full shard prepares into **999,329,599 tokens** in
  11m14s — 8.6x the previous corpus, and enough to put `SCALE_300M`'s
  compute-optimal floor at **2.5 epochs**, inside the repetition guidance
  for the first time.
- **The check that mattered:** re-preparing the old 400M-character corpus
  through the new streaming path produces **exactly 115,936,101
  tokens**, the same count as the in-memory path did in entry 039. The rewrite
  changed the memory profile, not the data.

## What actually cost the memory

Four separate places assumed the corpus fit in RAM, and all four had to go:

| step | before | after |
| --- | --- | --- |
| extract shard to text | build `list[str]`, then `"".join` | write each document straight to the file |
| read corpus | `raw_path.read_text()` — one 3.5GB `str` | 16MB blocks, newline-aligned |
| train tokenizer | needs the whole `str` | consumes a block iterator |
| encode | `list[int]` (~8GB) then `torch.tensor` (~8GB) | preallocated `int32` array (4.0GB), zero-copy wrap |

A fifth was waiting at training time: `TokenDataset` called
`tokens.to(dtype=torch.long)` on the whole stream, which would have undone
the `int32` saving the moment a run started. It now keeps whatever dtype it
is given and casts **per window** in `__getitem__` — `nn.Embedding` needs
`int64` indices, but only for the `seq_len` positions actually being looked
up, not for all billion.

## Why block-wise encoding is safe

Cutting a corpus into blocks and encoding each one separately is only
equivalent to encoding the whole thing if no token can span a block
boundary. Two facts make that true here:

1. Blocks are cut at **newlines** (`iter_text_blocks` reads a fixed number
   of characters, then backs up to the last `\n`).
2. After entry 039's rewrite, BPE merges never span a **pre-token**, and a
   newline always ends one.

So a merge can never straddle a cut. This is asserted rather than assumed:
`test_streaming_encode_matches_whole_corpus_encode` encodes a deliberately
awkward corpus at block sizes of 7, 64 and 10,000 characters and requires
the ids to equal a single whole-corpus `encode` call, and
`test_iter_text_blocks_reassembles_the_file_exactly` pins the more basic
property that the blocks partition the file — down to `block_chars=1`.

The same reasoning covers the document separator: `<|endoftext|>` contains
no newline, so it can never be split across a block, and `iter_documents`
can stream whole documents out of a corpus far larger than memory.

## `_TokenBuffer`

The encoded stream goes into a small append-only wrapper around a numpy
array:

- **`int32`, not `int64`** — halves the output. Any vocabulary this project
  will plausibly train fits, and the dataset casts per window anyway.
- **Preallocated** from the corpus size (`raw_bytes` for `char`, `//3` for
  `bpe`, a deliberate over-estimate) so the geometric growth path is
  normally never taken. It still exists, because an estimate that is wrong
  should cost a copy, not a crash.
- **Shrunk in place before wrapping.** `as_tensor` calls
  `resize(self._size, refcheck=False)` first. Without it the tensor is a
  *view* of an over-allocated array, and `torch.save` serializes the whole
  underlying storage — writing hundreds of megabytes of unused capacity to
  disk next to the real tokens. `test_token_buffer_tensor_holds_no_unused_capacity`
  saves a 3-token buffer allocated at 1M and asserts the file is under 10KB.

## Sizes are now part of the cache identity

`--corpus-chars` selects how much of the shard to keep, and it is baked
into both the cached text filename (`fineweb_edu_3500Mchars.txt`) and the
prepared-data directory (`fineweb_edu_3500Mchars_bpe4096`). Without that,
asking for a bigger corpus would silently reuse the smaller prepared cache
and train on it while reporting the larger number — the kind of bug that
produces a confusing result rather than an error.
`scripts/train.py` takes the same flag so it loads the matching directory.

The extraction also writes to a `.partial` file and renames it on
completion, so an interrupted download can't leave a short file behind that
later looks like a complete cache.

## The numbers

Both corpora, prepared through the same streaming path:

| | 400M chars | full shard |
| --- | --- | --- |
| characters | 400,001,266 | 3,450,710,480 |
| documents | 82,999 | 725,980 |
| tokens | 115,936,101 | 999,329,599 |
| chars/token | 3.450 | 3.453 |
| `tokens.pt` | 464MB | 3,997,319,906 B (3.7GiB) |
| tokenize + encode | 1m44s | 9m22s |
| peak RSS | 1.74GB | 3.75GB |

Both wall-clock figures are for tokenizer training + encoding with the text
already extracted. Going all the way from the parquet shard the first time
took 11m14s for the full corpus, so the parquet read and ASCII
normalisation account for the remaining ~1m50s. The full-shard run was done
twice and produced an identical 999,329,599 tokens both times.

Character counts are of the documents themselves; the separators between
them are not included, which is why the full-shard figure is slightly under
the 3.5B cap (the extraction stopped when the shard ran out, at
3,460,154,481 bytes on disk).

Peak RSS was measured in-process with `resource.getrusage`, not with
`/usr/bin/time -l` — the latter was tried first and reported 2.72GB for the
full-shard run, which cannot be right when simply *loading* the resulting
`tokens.pt` costs 4.11GB. It was measuring the `uv` wrapper rather than the
work.

3.75GB is slightly *under* the 4.0GB the finished token array occupies,
which is the expected shape of this: `np.empty` reserves 4.61GB of address
space up front but commits pages only as they are written, so the peak
tracks the tokens actually produced rather than the capacity allocated —
and the corpus itself, all 3.46GB of it, never contributes.

The 3,997,319,906-byte output is itself the proof that `as_tensor`'s shrink
works: 999,329,599 `int32` tokens are exactly 3,997,318,396 bytes, so the
file carries 1,510 bytes of pickle header and nothing else. The buffer was
preallocated at 1.15B elements (4.6GB); had the unused capacity been
serialized, the file would have been ~600MB larger.

And against the scaling-law targets from entry 039 — the point of the
whole exercise:

| target | tokens | epochs over 999M |
| --- | --- | --- |
| `SCALE_100M` floor (20x active) | 813,391,360 | **0.81** |
| `SCALE_300M` floor (20x active) | 2,502,430,720 | **2.50** |
| `SCALE_300M`, 20x *total* params | 6,749,163,520 | 6.75 |

`SCALE_300M` can now be trained to its compute-optimal floor at 2.50
epochs, comfortably inside `epochs_for_budget`'s 4-epoch warning threshold
(Muennighoff et al., 2023) — and `SCALE_100M`'s floor fits in a single pass
with room left over. The upper end of Clark et al.'s MoE range
(`20 x total`) would still need 6.75 passes, so a second shard is the next
lever if that turns out to matter — but the floor is what the budget is
planned against, and it is now reachable. `sample-10BT` has 13 more shards.

## Proving it end to end

A dataset that prepares but won't train is not a result, so `SCALE_300M`
was pointed at the 999M-token corpus for a short run
(`--batch-size 4 --seq-len 256 --total-tokens 200000`, 195 steps):

```
step  50/195 | loss 7.0887 | elapsed 1.527s
step 100/195 | loss 6.6994 | val_loss 6.9331 | val_ppl 1025.7128
step 150/195 | loss 6.5353
step 195/195 | loss 6.3163 | val_loss 6.3656 | val_ppl  581.5215
```

Loss falls, perplexity nearly halves in 195 steps, and the checkpoint
writes. The `int32` stream feeds `nn.Embedding` correctly through
`__getitem__`'s per-window cast, which was the change most likely to break
something quietly.

### The next bottleneck, already visible

It took **~8 minutes before the first step** — the steps themselves run at
~1.2-1.5s. Two things happen in that window: `torch.load` pulls in the 4GB
token tensor, and then `DataLoader(shuffle=True)` builds a `RandomSampler`,
which calls `torch.randperm(999_329_343)`. Measured directly,
`randperm(50M)` takes 0.66s and 0.40GB, so the full-size permutation is
**~8GB of `int64` and ~13s** — a permutation twice the size of the dataset
it indexes, allocated purely to decide visiting order.

So the memory won back on the prepare side is partly given away again at
train time. This does not block anything — the run above completed fine —
but a real multi-hour run on this corpus should not be paying 8GB for a
shuffle. The fix is a sampler that draws random offsets instead of
materializing a permutation, which for a corpus seen for well under one
epoch is statistically equivalent anyway. That is the next entry, and it is
the same lesson as this one: every default in the stack was written when
the dataset was 1MB.

## Tests

21 added, no pre-existing test changed (435 passing, was 414):

- **`tests/test_prepare_data.py`** (18, several parameterized over block
  sizes; 5 tests before, 23 now): blocks reassemble the file exactly;
  streamed documents match `split_documents`; streamed encoding matches
  whole-corpus encoding; streamed document encoding matches
  `encode_corpus`; `_TokenBuffer` grows past its capacity and holds no
  unused capacity when saved; both tokenizers train identically from blocks
  and from one string.
- **`tests/test_config.py`** (3): `corpus_chars` default and validation,
  and that two corpus sizes never share a prepared-data directory.

The streaming iterators are exercised at `block_chars=1` as well as at
realistic sizes, since boundary handling is exactly where this kind of code
fails.

## Questions that might come to mind

**Does `int32` limit the vocabulary?** Not in any way that matters — it
holds ids up to ~2.1 billion. The practical ceiling on vocabulary here is
the embedding table, long before the token dtype.

**Why not `int16`, since the vocab is 4097?** It would fit today and break
the moment anyone trains a 32k-vocab tokenizer, which is an ordinary thing
to want. `int32` costs 2 bytes more per token and removes the question.

**Is the whole corpus still held anywhere?** The prepared token array, yes
— 4.0GB, and that is the thing being written to disk, so it has to exist.
What no longer exists is a second and third copy of it.

**What about `--val-split-mode=documents` at this size?** It calls
`torch.cat` over the held-out documents, which briefly doubles the
validation slice. At 10% of 1.01B tokens that is ~400MB extra — fine. The
`tail` default is pure slicing and copies nothing.

**Does document masking still work at this size?** Yes, but note
`TokenDataset(return_segments=True)` builds a full-length document-id
tensor via `cumsum`, which is another ~4GB at 1B tokens. Worth knowing
before turning it on for a full-corpus run — and it belongs on the same
list as the `randperm` above.

**Is the 400M corpus now obsolete?** No, and it is kept. It prepares in
under two minutes and fits any quick experiment; the full shard is for runs
that are actually chasing the Chinchilla budget. The two live side by side
because `--corpus-chars` puts them in separate directories.
