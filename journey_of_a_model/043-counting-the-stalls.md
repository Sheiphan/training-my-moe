# 043 — Counting the stalls: 362 syncs a step, and 339 of them were in the optimizer

**Type:** tooling (direct follow-up to 042) · **Date:** 2026-09-19

> Still no GPU. Everything here is a **count**, not a timing — sync counts
> are exact and reproducible on CPU, which is the point: they were the part
> of the H100 plan that could be checked without renting anything.

## TL;DR

- Entry 042's plan named ~42 host-device synchronizations per optimizer
  step and ranked them as a modest 1-2% cost. I decided to count them
  instead of estimating. The real figure at `SCALE_300M`'s shape was
  **362 per step**, and **339 of those were inside `optimizer.step()`** —
  a place neither my analysis nor the design pass had looked.
- The cause is one line in PyTorch: `_get_value(step)` calls `.item()` on
  **each parameter's** step counter. With ~339 parameters that is 339
  pipeline stalls, invisible in this project's own source.
- `torch.optim.AdamW(fused=True)` keeps the counter on-device: measured
  **0**. Enabled by a capability probe rather than a flag, so MPS keeps
  exactly the optimizer it had.
- Deferring the trainer's own `.item()`s past `backward()` handled the
  rest. **362 → 20 syncs per optimizer step**, and the 20 that remain are
  all in `dispatch_batched`, which is the next entry's job.
- Also swept every `docs/math/verify_*.py`: **two had been broken for
  entries without anyone noticing** — 007 since 039, 010 since 041 (mine).
- **529 tests (7 added).**

## Why count instead of estimate

Entry 042 shipped with an estimate: ~42 syncs per micro-batch, ~30-60µs
each, "1-2% of a 170ms step at the batch size we're planning". That number
came from reading the code and adding up the `.item()` calls I could see.

Counting is cheap and exact — monkeypatch `Tensor.item` and
`Tensor.tolist`, run one `_run_step`, print the tally:

```
n_layers=10, grad_accum=1: 352 .item() + 10 .tolist() = 362 syncs/optimizer-step
n_layers=10, grad_accum=4: 382 .item() + 40 .tolist() = 422
n_layers=10, grad_accum=8: 422 .item() + 80 .tolist() = 502
```

362, not 42. And the shape of the number is the interesting part: it barely
grows with `grad_accum`. A per-micro-batch cost would have scaled with it.
Something was charging ~340 syncs *once per optimizer step*.

Attributing each call to its nearest frame in this project:

```
 339x  .item()   train/trainer.py:501 _run_step
  10x  .item()   model/moe.py:366 dispatch_batched
  10x  .tolist() model/moe.py:369 dispatch_batched
```

Line 501 is `optimizer.step()`.

## The 339

PyTorch's `torch/optim/optimizer.py`:

```python
def _get_value(x):
    # item is significantly faster than a cpu tensor in eager mode
    if not torch.jit.is_scripting() and torch.compiler.is_compiling():
        return x
    else:
        return x.item() if isinstance(x, torch.Tensor) else x
```

Adam/AdamW keep each parameter's step count as a tensor and read it back to
Python once per parameter per step, to compute the bias-correction terms.
`SCALE_300M` has ~339 parameters, so: 339 stalls.

The comment is not wrong — `.item()` *is* faster than keeping a CPU tensor
in eager mode. It is written for a CPU-resident step counter. On an
accelerator it means the GPU drains 339 times while Python collects
numbers it then uses to compute a scalar.

Measured across the AdamW variants:

| variant | `.item()` calls (40-param model) |
| --- | ---: |
| default | 40 |
| `foreach=True` | 80 |
| `capturable=True` | CUDA/XPU only |
| **`fused=True`** | **0** |

`foreach=True` being *worse* is a nice illustration of why measuring beats
reasoning: the obvious "batch the work" flag doubles the count.

## Enabling it as a probe, not a flag

```python
def _fused_adamw_supported(params) -> bool:
    if not params:
        return False
    devices = {p.device.type for p in params}
    return len(devices) == 1 and devices.pop() in ("cuda", "cpu", "xpu")
```

A capability probe rather than a `TrainConfig` field, for two reasons.
Fused AdamW isn't available on MPS, so a flag would need a fallback anyway;
and the laptop path should keep byte-identical behaviour without anyone
having to remember not to set something. CPU gets it too — torch 2.13
supports fused on CPU, which is the only reason any of this was testable
without a GPU.

Four tests pin it, and deliberately not just the flag: one counts the
syncs, and one asserts the optimizer still changes weights, because a fused
kernel that silently did nothing would pass a sync-count test perfectly.

## The trainer's own 23

Three changes, all of the same shape — stop converting on the hot path:

**`_collect_expert_counts`** returned `[block.moe.last_expert_counts.tolist()
for block in model.blocks]`: one sync *per layer*, after *every*
micro-batch, unconditionally — and then the result is discarded on the ~90%
of steps that aren't logging steps. It now returns the stacked
`(n_layers, n_experts)` tensor and the caller converts inside the logging
branch.

**`float(grad_norm)`** moved from `_run_step`'s return to the log line.

**`_forward_backward_core`** returned Python floats, which meant
`z_loss_term.item()` and `mtp_mean.item()` ran *before* `backward()` was
enqueued. Those are the worst possible placement: the pipeline drains, then
Python walks to the backward call with the accelerator idle. Both are
inert today (`z_loss_weight=0`, `mtp_depth=0`) — but the H100 config turns
z-loss on at 1e-4, so this would have quietly added the worst sync in the
file at exactly the moment it mattered. The function now returns detached
tensors and the single conversion happens in `_average_metrics`, once per
optimizer step, after every backward is queued.

Result:

```
n_layers=10, grad_accum=1:  10 .item() + 10 .tolist() =  20 syncs/optimizer-step
n_layers=10, grad_accum=4:  40 .item() + 40 .tolist() =  80
n_layers=10, grad_accum=8:  80 .item() + 80 .tolist() = 160
```

362 → 20 at `grad_accum=1`. Everything left is `dispatch_batched`'s
`int(active_counts.max().item())` and `active.tolist()` — one pair per MoE
layer per micro-batch, which is why the count still scales with
accumulation. Removing those is the fixed-capacity rewrite, and it is the
next entry.

## The verify sweep

Changing `_collect_expert_counts`' return type broke two
`docs/math/verify_*.py` scripts, which is exactly what those scripts are
for. That prompted running **all 30**, and the result was uncomfortable:

| script | broken by | since |
| --- | --- | --- |
| `verify_007_tokenizers` | 039's BPE rewrite deleted `_count_pairs` | entry 039 |
| `verify_010_training_loop` | 041 added `total_steps` + `data_fingerprint` to the checkpoint | entry 041 |

Both are mine. The 041 one is the more embarrassing: that entry's whole
subject was checkpoint payload changes, and it asserted a key set in a
verify script it never ran.

`_count_pairs` is restored as a documented reference helper rather than
deleted from the doc. It is the textbook statement of BPE's counting step
(Sennrich et al. 2016) and the form `007-tokenizers.md` walks through by
hand; `train` since 039 does the frequency-weighted, chunk-indexed
generalization of it. The two agree on a single unweighted sequence, which
is what makes the simple one honest to teach first — and a test pins that
agreement rather than leaving it as a claim.

All 30 now pass. This should be a CI job rather than a thing I remember to
do; that is noted, not yet done.

## Tests

**7 added (529 passing, was 522):**

- **`tests/test_optim.py`** (4): the device probe; that `build_optimizer`
  actually sets `fused`; that a step performs **zero** `.item()` calls; and
  that it still changes the weights.
- **`tests/test_bpe_tokenizer.py`** (3): `_count_pairs` on a normal
  sequence and on empty/one-element inputs, plus the equivalence with the
  trainer's weighted counting.

No pre-existing test changed. The MPS path is untouched: fused AdamW isn't
available there, so the probe returns `False` and the optimizer is
identical.

## Questions that might come to mind

**Is the fused kernel numerically identical?** No, and it shouldn't be
expected to be — it fuses operations that were separate, so rounding
differs. All 529 tests and all 30 verify scripts pass, including
`verify_018_optimizer_hardening` and `verify_020_muon_optimizer`, so it
agrees within the tolerances this project already asserts. It is not
bit-identical to a pre-043 run, which is the same category of change as
bf16 and is stated rather than hidden.

**Does this actually make training faster?** Unknown, and this entry does
not claim it does. A sync is a stall only if there is queued work to stall
on — on CPU there is none, so the count is the honest metric and the
speedup is not. Entry 044 measures wall-clock on a GPU.

**Why not remove the MoE's 20 too, here?** Because they cannot be removed
without changing what the model computes: `group_size` is derived from the
data, and making it static means either a capacity limit (which can drop
tokens) or a different dispatch algorithm. That is a semantics decision, so
it gets its own entry, its own tests, and its own argument.

**Isn't `foreach=True` the standard advice?** It is, for wall-clock on
small models — it batches the elementwise work into fewer kernels. It just
doesn't help here, and measurably doubles the sync count, because the
bias-correction read happens per parameter either way.
