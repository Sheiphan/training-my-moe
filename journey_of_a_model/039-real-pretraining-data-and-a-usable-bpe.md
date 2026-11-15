# 039 — Real pretraining data, and the BPE trainer that had to get fast first

**Type:** data + tooling (standalone scale-up request, not a spec task) · **Date:** 2026-09-18

> **Full derivation:** [`docs/math/039-bpe-incremental-counts-and-streaming.md`](../docs/math/039-bpe-incremental-counts-and-streaming.md)
> — the incremental pair table proved equal to a recount at every merge, the work counted (not timed), and the exact
> condition under which block-wise encoding (entry 040) equals whole-corpus encoding.

## TL;DR

- **The complaint:** the biggest preset (`SCALE_300M`) wasn't producing good
  output, and the suspicion was the data, not the model. That suspicion was
  right. Every run up to here trained on `tiny_shakespeare` (~1.1MB of one
  author's plays) or `tiny_stories` (~22.5MB of GPT-4-generated children's
  stories, deliberately written with the vocabulary of a four-year-old).
  Neither contains the diversity a 300M-parameter model needs; TinyStories
  in particular was built to study *small* models, and its synthetic,
  narrow domain puts a hard ceiling on what any model trained on it can
  learn to say.
- **The new corpus:** `fineweb_edu` — a 400M-character slice of
  **FineWeb-Edu** (Penedo et al., 2024, odc-by), the educational subset of
  FineWeb, filtered from Common Crawl by a classifier trained to spot
  educational content. It is the best-performing open pretraining dataset
  in this size class: FineWeb-Edu-trained models beat every other public
  corpus on MMLU/ARC/OpenBookQA at matched token counts. ~18x the raw text
  of `tiny_stories`, and real prose instead of synthetic prose.
- **The detour:** the corpus is the easy half. The hard half was that this
  repo's from-scratch `ByteBPETokenizer` re-scanned the *entire* corpus on
  every merge, so training a 4096-token vocab over 400M characters
  extrapolated to **tens of hours**. That is why the earlier `SCALE_*`
  presets quietly used character-level tokenization. Rewriting the trainer
  (pre-tokenization + a frequency table with incremental pair counts)
  brought the same job on `tiny_stories` from an extrapolated **~2 hours to
  2.3 seconds**, which is what made real subword tokenization affordable
  here at all.
- **The honest scaling verdict:** 400M characters is **115.9M BPE tokens**
  across 82,999 documents. Chinchilla-optimal for `SCALE_300M` starts at
  ~2.50B tokens. So this corpus is a large step up but still **~4.6% of
  compute-optimal** — good enough to prove the data was the bottleneck, not
  big enough to finish the job. Sizing for a genuinely compute-optimal run
  is worked out at the end.

## What the old data actually was

| corpus | raw text | what it is |
| --- | --- | --- |
| `tiny_shakespeare` | 1.1MB | Every line of Shakespeare, one author, archaic English |
| `tiny_stories` | 22.5MB | GPT-4-generated stories using only words a 3-4 year old knows |
| `fineweb_edu` (new) | ~400MB | Real web pages, classifier-filtered for educational content |

TinyStories is a *good* dataset for what it was built for — Eldan & Li
(2023) used it to show that models in the 1-30M range can learn fluent
English if the text is simple enough. Using it to train a 337M-parameter
MoE is using it backwards: the model has capacity to learn things the
corpus never demonstrates. No architecture change fixes that.

## Choosing the corpus

Requirements, in order: (1) real text, broad domain; (2) quality-filtered,
because unfiltered Common Crawl is mostly junk; (3) permissively licensed;
(4) not so large the download is absurd for a laptop project.

**FineWeb-Edu** satisfies all four. Its 1.3T-token full release is
published alongside official `sample-10BT` / `sample-100BT` / `sample-350BT`
subsets; the smallest, `sample-10BT`, is 14 parquet shards of ~2.15GB each.
Exactly one shard is downloaded (`sample/10BT/000_00000.parquet`), and only
the first ~400M characters of its `text` column are kept — read row-group
by row-group so a 2.15GB file is never fully held in memory.

Runners-up: **WikiText-103** (~100M tokens, high quality, but its original
plain-text release has gone 403 on S3 and it is encyclopedic-only);
**OpenWebText** (~40GB, too big); **MiniPile** (built for exactly this
purpose, but FineWeb-Edu's filtering is newer and measurably better).

## Sizing it with the scaling law

`qwenmoe/train/scaling.py` already had the machinery, so the sizing uses
the repo's own functions rather than new arithmetic. At `vocab_size=4096`:

| preset | total params | active/token | sparsity | `20 x active` | `20 x total` |
| --- | --- | --- | --- | --- | --- |
| `SCALE_100M` | 106,729,856 | 40,669,568 | 2.62x | 813.4M tokens | 2.13B tokens |
| `SCALE_300M` | 337,458,176 | 125,121,536 | 2.70x | 2.50B tokens | 6.75B tokens |

`chinchilla_budget` is `20 x active_params` (Hoffmann et al., 2022), and
`moe_budget_note` already carries the caveat that matters here: 20x is a
**dense-model** constant, where `total == active`. For a sparse MoE, Clark
et al. (2022) put the compute-optimal budget *between* `20 x active` and
`20 x total` — the idle experts still have to be trained. So `SCALE_300M`'s
real target is somewhere in **2.50B – 6.75B tokens**, with 2.50B as a floor.

Against that, this corpus delivers **115,936,101 BPE tokens**. Reaching even
the floor means **21.6 passes** over it (`SCALE_100M`: 7.0), and
`epochs_for_budget` warns past 4 — the point where repeated-data returns
fall off sharply (Muennighoff et al., 2023). Stated plainly: this corpus is
correctly sized for the *runs this machine can actually finish* (tens to a
few hundred million tokens, per journal 037's wall-clock measurements), and
undersized for a compute-optimal `SCALE_300M`. Both things are true; the
second one is the next problem, not this one.

## Cleaning: how a "character-level" vocab hit 5,164 characters

The first prepared version of the corpus produced a character vocabulary of
**5,164** — against `tiny_stories`' 91. Real crawled text drags in a long
tail of CJK, Cyrillic, emoji, maths symbols and box-drawing characters from
incidental non-English content. Nearly all of them appear so rarely that
their embedding rows would never receive a meaningful gradient, and
`CharTokenizer.encode` *raises* on any character it didn't see in training,
so a fat tail is also a fragile tokenizer.

Dropping documents that were *mostly* non-ASCII only got it to 3,728 —
because the problem isn't a few bad documents, it's one or two stray glyphs
each across hundreds of thousands of otherwise-fine documents. What worked
was normalising instead of filtering (`_to_ascii`): map the common
typographic characters to their ASCII equivalents (curly quotes to straight,
en/em dash to hyphen, ellipsis to `...`, NBSP to space), NFKD-fold accented
letters to their base letters so `café`/`naïve` survive as `cafe`/`naive`,
and drop whatever is left character by character. A document hollowed out by
more than 20% is skipped entirely, since that means it was mostly another
script.

Result: **95 characters**, all printable ASCII — the same neighbourhood as
`tiny_shakespeare` (65) and `tiny_stories` (91).

## The tokenizer, which turned out to be the real blocker

Journal 037 chose character-level tokenization for the `SCALE_*` presets and
recorded the reason as BPE being too slow. This entry finally measured *how*
slow, and then fixed it.

### What was wrong

`ByteBPETokenizer.train` counted every adjacent pair across the whole corpus,
merged the most frequent one, and then **counted every pair again from
scratch**, once per merge. Runtime therefore scaled as `characters x merges`.
Measured on TinyStories text at a 512 target vocab (256 merges):

| characters | train time |
| --- | --- |
| 20,000 | 0.41s |
| 50,000 | 1.02s |
| 100,000 | 2.07s |

Linear in both factors, as predicted — which extrapolates to **~2 hours**
for `tiny_stories` (22.5M chars) at the default 4096 vocab, and **~35 hours**
for the new 400M-character corpus. That is the real reason the scale-up
presets were stuck on character-level tokens.

### What changed

Two standard ideas, both from the BPE literature the module already cited:

1. **Pre-tokenization** (GPT-2's regex, rewritten to use stdlib `re`
   instead of adding a `regex` dependency): cut text into word-ish chunks
   first and learn merges *within* a chunk. This replaces the module's
   earlier deliberate simplification — merges over the raw byte stream,
   free to span word boundaries — which wasted vocabulary on cross-word
   merges and, more importantly, forced every count to span the corpus.
2. **A frequency table with incremental pair counts** (Sennrich et al.'s
   own approach): collapse identical chunks into one entry with a count, so
   the working set is the *unique* chunks (millions) rather than every
   position (hundreds of millions). Each merge then revisits only the chunks
   containing that pair and patches the affected counts in place.

`encode` gained the matching treatment: it encodes chunk by chunk and
memoises the result per chunk, so ordinary repeated words become a dict hit.

### Measured after

| job | before | after |
| --- | --- | --- |
| train vocab=4096 over 22.5M chars | ~2 hours (extrapolated) | **2.3s** |
| encode 20M chars | — | **0.9s** (3.95 chars/token) |

A ~3000x improvement on the same task, and subword tokenization for the new
corpus goes from "not worth attempting" to a rounding error in data prep.

### A real bug this caught during development

Translating GPT-2's pattern from the `regex` module to stdlib `re` looks
mechanical — `\p{L}` becomes `[^\W\d_]`, `\p{N}` becomes `\d` — and the
first version got the symbol class wrong in a way that no existing test
could see. GPT-2's symbol class is `[^\s\p{L}\p{N}]`: not whitespace, not a
letter, not a number. The direct-looking translation is `[^\s\w]` — but
Python's `\w` *includes underscore*, so underscore was excluded from the
symbol class while also not being a letter or a digit. It therefore matched
**no alternative at all**, and `re.finditer` does not complain about text it
cannot match — it just skips it:

```
"file_name.py and snake_case_words"
  -> ['file', 'name', '.', 'py', ' and', ' snake', 'case', 'words']
  -> decodes to "filename.py and snakecasewords"
```

Every underscore in the corpus was being silently deleted. The existing BPE
tests all passed, because the corpus they use (`"the quick brown fox..."`)
contains no underscores — and the round-trip tests only ever round-tripped
text made of the same characters.

The fix is `(?:[^\s\w]|_)+` for the symbol class. The lesson is the test
that was missing, now added: assert that the chunks **reassemble into
exactly the input** (`test_pre_tokenization_covers_every_character`).
Pre-tokenization is a partition of the text, and any pattern gap is silent
data loss, which is the worst failure mode available — a corpus that trains
fine and is quietly wrong. Two round-trip tests over deliberately awkward
text (underscores, symbols, digits, contractions, tabs, non-Latin,
newlines) back it up.

This mattered in practice: the first full prepare run over the corpus
produced 115,904,021 tokens, the fixed one 115,936,101 — the 32,080-token
difference is the underscores coming back.

### The alternative approach, and why it wasn't taken

There is a second, entirely legitimate way to make BPE affordable on a big
corpus, and it is worth recording because it is what production tokenizers
effectively do:

> **Train the tokenizer on a sample, then encode everything with it.** Fit
> the merge table on a bounded slice of the corpus (say 20MB — the size
> `tiny_stories` already proved tractable), then apply those learned merges
> to encode the *full* 400M-character corpus. This works because training
> and encoding have very different costs: training is
> `corpus x merges` under the naive algorithm, while encoding is a single
> pass applying an already-fixed merge table, and with per-chunk memoisation
> it is close to linear in corpus size. Real tokenizers are fitted this way
> — nobody runs tokenizer training over an entire multi-trillion-token
> pretraining set.

The tradeoff is that the merge table is then fitted to a narrower slice of
text than the model actually trains on. With a 20MB sample out of 400MB, any
vocabulary, spelling or domain that happens to be absent from the sample
gets tokenized inefficiently forever after — split into more, shorter tokens
— which quietly wastes context window on exactly the material the sample
under-represented. It is a real cost, and it is paid for the entire life of
the tokenizer.

It was rejected here only because the rewrite above made it unnecessary:
once full-corpus training costs seconds, there is no reason to accept a
sampling bias to avoid it. Had the rewrite not worked, this was the fallback,
and it would have been a perfectly defensible one. It remains the right
approach if this project ever moves to a corpus large enough that even the
fast trainer struggles — at which point the sample should be drawn randomly
across the corpus rather than taken from the front, so it is at least
representative.

## What this changes in the pipeline

- `qwenmoe/data/download.py`: new `fineweb_edu` corpus — one-shard
  `hf_hub_download`, row-group-wise parquet read capped at
  `_FINEWEB_EDU_TARGET_CHARS`, `_to_ascii` normalisation, documents joined
  by the same `<|endoftext|>` separator `tiny_stories` uses (so
  `--split-documents` works identically). Needs `pyarrow` — FineWeb-Edu has
  no plain-text release to avoid it with.
- `qwenmoe/tokenizer/bpe.py`: rewritten `train`/`encode` as above.
- `qwenmoe/config.py`, `scripts/prepare_data.py`, `scripts/train.py`:
  `fineweb_edu` added to the corpus choices.
- `pyproject.toml`: `pyarrow` dependency.

## The prepared corpus, as built

```
corpus: fineweb_edu (source: .cache/data/fineweb_edu_sample.txt)
characters: 401,080,932
tokenizer: bpe, vocab_size=4097        # 4096 learned + 1 reserved <eos>
documents: 82,999, separator id (eos_id): 4096
tokens: 115,936,101
compression ratio (characters/token): 3.459
```

End-to-end prepare time (parquet read + ASCII normalisation + tokenizer
training + encoding all 400M characters): **1m46s**.

## Tests

No pre-existing test changed. The BPE suite asserts properties — target
vocab size reached, lossless round-trip on trained and unseen text,
compression versus raw bytes, save/load equivalence, `<eos>` reservation —
and the rewritten implementation satisfies all of them unchanged, which is
the useful signal: the merge table is different (word-bounded now), the
contract isn't. Three tests were **added** for the pre-tokenizer, all of
them prompted by the underscore bug above:

- `test_pre_tokenization_covers_every_character` — the chunks must
  reassemble into exactly the input (the invariant the bug violated).
- `test_round_trip_lossless_for_awkward_text` — encode/decode over
  underscores, symbols, digits, contractions, tabs, non-Latin and newlines.
- `test_round_trip_lossless_when_trained_on_awkward_text` — the same, with
  that text also used as training data.

The download path stays untested by design, as it touches the network.

Full suite: **414 passed** (411 before this entry).

## Questions that might come to mind

**Does the pre-tokenization change break tokenizers trained before this
entry?** A saved `tokenizer.json` still loads, and `decode` is unaffected
(it is a pure id-to-bytes lookup). But `encode` now splits on chunk
boundaries before applying merges, so text can tokenize differently than it
did when an older BPE checkpoint was trained. The character-level runs —
which is every `SCALE_*` checkpoint so far — are untouched, since
`CharTokenizer` was never involved in any of this.

**Why 400M characters and not the whole shard?** Memory, not principle. The
prepared token stream is materialised as a Python list and then a torch
tensor; at 400M characters that peaks comfortably inside the 24GB this
machine has, and journal 037 already documented what happens near that
ceiling (swap thrashing, 20-100x slowdowns, no clean OOM). Going bigger
needs the token stream written incrementally rather than built in memory —
see below.

**So what would a genuinely compute-optimal run need?** The downloaded
shard holds 726,000 documents; at this slice's average of 4,832 characters
per document that is ~3.51B characters, or **~1.01B tokens** at the
measured 3.459 chars/token. Against that:

| target | tokens needed | as text | vs. one shard |
| --- | --- | --- | --- |
| `SCALE_100M` floor, <=4 epochs | 203M | 0.70GB | 20% of a shard |
| `SCALE_300M` floor, <=4 epochs | 626M | 2.16GB | 62% of a shard |
| `SCALE_300M` floor, 1 epoch | 2.50B | 8.66GB | ~2.5 shards |

So the single shard already on disk is enough to train `SCALE_300M` at its
Chinchilla floor inside the 4-epoch repetition guidance — the corpus cap is
the only thing in the way, and it is one constant. What it needs first is a
streaming prepare step: encode document by document and append into a
preallocated array (or write token shards to disk) instead of building one
giant Python list, plus `int32` storage rather than `int64`. At 400M
characters the current in-memory approach peaks comfortably inside 24GB; at
3.5B it would not. That is the obvious next entry.
