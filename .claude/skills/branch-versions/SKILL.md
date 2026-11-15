---
name: branch-versions
description: Use when squashing, rebasing, renaming or adding a version branch in this repo, or when asked about the branch timeline. Branches are project versions stacked in a chain; rewriting one means rebuilding the ones above it. Trigger words: squash, rebase, branch timeline, new version branch, which branch.
---

# Branches are versions

Each branch is a version of the project, not a feature branch. The history is meant to read as one commit per version. The branch scheme is not described in the README; this skill is its only record. Re-check it with `git log --all --graph --oneline --decorate` before acting, because it may have moved on.

## The chain (as of 2026-11-15)

```
40d8f9d  Initial commit: the from-scratch model + Manim (journal 001-016)
 |-- model-development       side branch: RoPE deep-dive notebook, an older Manim removal (2 commits)
 `-- moe-modernization       v2: journal 017-038 + docs/math  (27 commits)
      `-- scaling-improvements   v3: journal 039-041, real data, long runs  (+3)
           `-- cuda-h100            v4: journal 042-044, GPU port, first H100 session  (+3)
                `-- docs-and-site        presentation layer: one site/, model figure, tensor walk (+2)
```

- **`moe-modernization` -> `scaling-improvements` -> `cuda-h100` are the flowing versions.** Each starts at the previous tip.
- **`model-development` is not in the chain.** It diverged from the root. Every older stacked branch still contains Manim as history. Manim was later removed from the latest code by a fresh commit (it conflicts with `model-development`'s older removal), so do not merge `model-development` for that.
- **`docs-and-site` changes no model behaviour.** Treat it as the branch you present from, rebased onto the latest version, not as a version of the model.
- **`main` is the GitHub default branch** (personal repo `Sheiphan/training-my-moe`, remote `origin`) and follows the latest tip, `docs-and-site`. Journal 017 mentions a `main` too. Journal 017 also says `moe-modernization` was branched off `model-development`; git shows the root commit.
- Journal 042 is the only entry that explains a branch split ("Why a new branch?").

## Goal: one commit per version

Squash each version branch to a single commit so the history is root -> v2 -> v3 -> v4 -> docs. **Squashing a lower branch changes its hash, so every branch above it must be rebuilt on the new commit.** Work bottom-up.

## Before rewriting anything

This rewrites history on several branches and cannot be undone casually. Do all of these first, and confirm the plan with the user:

1. **Tag backups** of every tip: `git tag backup/<branch> <branch>` for all five. Nothing is deleted until the user says so.
2. **Deal with uncommitted work** (check `git status`). Stash it, or have the user commit it, and restore it on `docs-and-site` at the end.
3. **Check for a remote** (`git remote -v`; there was none). Without one, the backups are the only safety net.
4. **Open decisions, ask rather than assume** (unless already answered in the conversation or in git): is `docs-and-site` a version or a presentation layer; does `model-development` stay as-is, get renamed (e.g. `v1-baseline`) or go; should `moe-modernization` be one commit or a few themed ones, since 27 commits over 17 days lose the task-by-task story that the journal already records.

## Doing it

Per branch, bottom-up: `git checkout <branch>`, `git reset --soft <parent-tip-or-root>`, commit once with a message naming the version and the journal range (for example "v2 moe-modernization: journal 017-038"), then `git rebase --onto <new-parent> <old-parent> <next-branch>` for the branch above. Check `git diff backup/<branch> <branch>` is empty after each squash: the squash must not change any file.

End the commit message with the attribution lines from the session's system-reminder. Do not push, and do not delete the `backup/*` tags, unless asked.

## Adding a new version

Branch from the tip of the latest version (`cuda-h100` today), say why in the first journal entry on it (as 017 and 042 do), and extend the chain above.
