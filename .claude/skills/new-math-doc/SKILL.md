---
name: new-math-doc
description: Use when writing or revising a docs/math/NNN-*.md derivation and its verify_NNN_*.py script, or when a number in a math doc or the qwenmoe code it describes changed. Trigger words: math doc, verify script, derive by hand, docs/math.
---

# Add or update a math doc

A math doc is a first-principles, solved-by-hand derivation of one component, carried through one concrete example. Each doc has a `verify_NNN_*.py` that pins the same weights into the real `qwenmoe/` module and asserts the hand-computed numbers.

The full prompt for these docs is at the bottom of `docs/math/README.md` ("Prompt to generate a new entry"). Read it and follow it. In short:

1. **Number and companion.** The doc mirrors a journal entry number. Read the companion `journey_of_a_model/NNN-*.md` first, and `004-attention-gqa.md` as the style reference.
2. **Write the verify script first.** `docs/math/verify_NNN_<slug>.py`:
   - Run it from the repo root (`uv run python docs/math/verify_NNN_<slug>.py`). Model it on `verify_005_moe_layer.py`.
   - Build the tiny config, pin the weights to the values written in the doc, and run both `forward_with_trace` (the explicit path the doc walks) and `forward` (the fused production path) when both exist.
   - Print every intermediate and `assert` the hand-computed result. Report the measured max difference between fused and explicit paths.
3. **Write the doc** with one running example from start to end, a derivation before each computation, digit-by-digit arithmetic, a shape ledger, terms defined at first use, exact-versus-empirical marking, at least one Mermaid diagram, a "Questions that might come to mind" section, a symbol table, and verified citations. Quote the script's real output, never invented values.
4. **Register it:**
   - Add a row to the Index table in `docs/math/README.md`.
   - Add the one-line pointer blockquote near the top of the companion journal entry.
   - Add the doc to `DOCS` (and new boxes/edges to `NODES`/`EDGES`) in `docs/math/build_architecture_footer.py`, then run `uv run python docs/math/build_architecture_footer.py` to regenerate the closing diagram and `site/architecture-map/`.
5. **Check:** `uv run python docs/math/run_all_verify.py` (rewrites `verify-status.md`; exits 1 if any script fails), then `uv run python site/check_site.py`.

When the *code* a doc describes changes, re-run that doc's verify script. If it fails, the doc's numbers are wrong: fix the doc and the script together. Also re-check the doc's `file.py#L<line>` links.

## Lessons from the 039 / 041 / 044 docs

These came up writing the docs for journal entries 039-045 and are not in the README prompt.

- **Bundling and skipping.** One doc may cover consecutive journal entries (039 covers 039+040, 044 covers 044+045). Key it by the first number, and add the "Full derivation" pointer blockquote to *every* journal entry it covers. An entry that is only plumbing or counting (042 CUDA device plumbing, 043 sync counts) gets no doc; say so in the README's "cuts" list instead of writing a hollow derivation.
- **Measured versus derived.** If a journal figure is a hardware measurement the script cannot re-run (tok/s on a rented GPU, profiler milliseconds, an 8-minute stall), put it in the verify script as a labelled `MEASURED` input and derive everything downstream from it with the real model (`count_params` on the real preset, not a toy). Quote measurements as measurements; never reproduce them.
- **Recompute, then report disagreements.** Re-deriving a journal's figures finds small errors (044 had a table row using the wrong sequence length, and a rounded budget). Print the discrepancy in the script, assert the corrected value, and list it in the doc (044's Part 7). Do not edit the journal yourself unless asked, and do not assert a journal number you could not reproduce.
- **Test a claim as a condition, not a slogan.** "Block-wise encoding is exactly identical" (journal 040) is true only when the cut satisfies an exact rule. The rule was found by brute force over every short string, then checked against the real corpus. Prefer an exhaustive small-case check over deriving a regex's behaviour by hand.
- **Real-data checks must degrade.** A check that needs git-ignored data (`.cache/data/...`, `runs/`) prints `SKIP` when the file is absent so `run_all_verify.py` stays green on a clean clone.
- **Use the canonical symbols** (README "Canonical notation": `N` experts, `S` sequence length, `k` top-k). If the doc needs a clashing symbol, add a `NOTATION` entry in `build_architecture_footer.py` rather than renaming mid-doc.
- **Registering in the footer builder** needs more than `DOCS`: every new node needs `NODES`, `REFS` (a regex matching a real code line) and `STEPS` (a regex matching a heading in the doc), or the builder asserts. A new *era* also needs `ERAS`, `era()`, the emoji swatch `sw`, `ERA_SHORT`, and the hard-coded `--x-fill` CSS variables, `.sw.*` / `.erachip.*` rules and the JS `classDef` list. The builder then rewrites the footer of **every** doc (a new legend entry and "next:" links); that diff is expected.
- **`site/check_site.py` says `UNTRACKED map link`** for each new doc and verify script until they are committed. That is not a defect in the doc.
