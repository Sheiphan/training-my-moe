---
name: new-journal-entry
description: Use when adding a numbered entry to journey_of_a_model/ (a finished implementation step, experiment, or plan/decision) and indexing it in that folder's README. Trigger words: journal entry, journal 046, write up this change, add to the journey.
---

# Add a journal entry

Entries live in `journey_of_a_model/NNN-slug.md`, one per step, in order.

1. **Pick the number.** Take the highest `NNN` in the index table of `journey_of_a_model/README.md` and add 1. Use a short kebab-case slug.
2. **Read the bar first.** The "Documentation standard" section of `journey_of_a_model/README.md` is the checklist. Read it, and read the most recent entry (e.g. `044-*.md`) as a style reference.
3. **Header.** `# NNN — Title`, then `**Type:** <experiment | tooling | plan / decision | ...> (follow-up to NNN) · **Date:** <today, YYYY-MM-DD>`.
4. **Sections, in this order:** `## TL;DR`, the body (problem, findings, what changed), `## What this changes` where relevant, `## Tests` (say how many tests pass and whether the count changed), `## Questions that might come to mind`.
5. **Content rules** (all from the README standard):
   - Define every technical term at first use, in plain words, never via another undefined term.
   - Include at least one Mermaid diagram per concept.
   - Give the full derivation of any formula, with a worked numeric example. Mark exact math versus empirical constants.
   - Link original papers. Find them by web search and never guess a URL.
   - **No fabricated numbers.** Label each figure **measured** (with its source: log, run, script) or **estimated** (arithmetic only). If an estimate was later wrong, say so.
   - A plan or decision entry that ships no code should say so at the top.
6. **Link code with relative paths and line anchors**, e.g. `[block.py:37](../qwenmoe/model/block.py#L37)`. Verify each anchor points at the line you mean.
7. **Index it.** Add a row to the table in `journey_of_a_model/README.md` (number, date, link, a bold type tag plus summary), matching the neighbouring rows.
8. If the entry has a math companion, use the `new-math-doc` skill and add the pointer blockquote near the top of this entry.
9. Run `uv run pytest -q` and make sure the test count in the entry is the real one.
