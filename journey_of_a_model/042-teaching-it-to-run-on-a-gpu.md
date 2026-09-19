# 042 — Teaching it to run on a GPU it has never seen

**Type:** tooling (new `cuda-h100` branch) · **Date:** 2026-09-19

> **Nothing in this entry has been measured on a GPU.** No CUDA device was
> available when it was written. Every performance figure below is labelled
> as a projection, and entry 043 will replace them with measurements. The
> *correctness* claims — what the code does, which tests cover it — are real
> and the 522-test suite backs them.

## TL;DR

- Journal 041 made a multi-day run survivable. It did not make it fast: the
  real run measured **1,339 tok/s**, so the 2.5B-token Chinchilla floor is
  **21.6 days**. Renting an H100 for a few hours is the obvious answer.
- **The project could not use a GPU at all.** `TrainConfig.device` was
  `Literal["auto","mps","cpu"]` and `select_device` never returned
  `"cuda"` — so `--device cuda` was accepted, ignored, and the run trained
  on CPU. `scripts/check_capacity.py` had been saying so in a comment since
  entry 037.
- **The worst thing found was not the missing device.** It was
  `_synchronize` in `scaling.py`, which only handled MPS. CUDA dispatches
  kernels asynchronously, so `calibrate_throughput` on a GPU would have
  timed how fast Python could *enqueue* work — reporting a plausible-looking
  and completely fictional tokens/sec, which is exactly the number the whole
  rental decision hangs on.
- Memory telemetry is now device-neutral, and **the `mps_*` keys were kept
  rather than renamed** so every log on disk and every journal figure
  quoting one stays valid.
- **522 tests (12 added, 3 changed).**

## The one-line bug that would have poisoned every measurement

```python
def _synchronize(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()
```

`calibrate_throughput` is the function the entire H100 plan depends on: it
produces the tokens/sec that decides which GPU to rent and for how long.
On CUDA this function returned immediately, because CUDA kernel launches
are asynchronous — `loss.backward()` queues work and returns long before
the GPU has done any of it. The timing loop would have measured Python's
enqueue rate.

The failure mode is what makes it dangerous: it doesn't crash, and it
doesn't return an absurd number like infinity. It returns something
*plausible but too high*, which would have been written into a journal
entry as a measured figure and used to size a run.

```python
    if device == "mps":
        torch.mps.synchronize()
    elif device.startswith("cuda"):
        torch.cuda.synchronize()
```

`startswith` rather than `==` so an explicit `"cuda:1"` from a caller still
synchronizes. Four tests now pin this — one per backend plus the indexed
case — and they run on CPU CI by monkeypatching the sync functions with
counters, so the guard doesn't depend on having a GPU to test it.

## Why `select_device` returns `"cuda"` and never `"cuda:0"`

This looks cosmetic and isn't. `_forward_backward_core` does:

```python
torch.autocast(device_type=device, dtype=torch.bfloat16)
```

`device_type` must be the bare backend name; `"cuda:0"` raises. Because the
trainer passes its own device string straight through, getting this right
means **bf16 needed no CUDA-specific plumbing at all** — the autocast path
written for MPS in entry 024 started working on CUDA the moment
`select_device` could return the string. There is still no `GradScaler`
anywhere, deliberately: bf16 shares fp32's exponent range, so the fp16-era
loss-scaling machinery is not needed.

Preference order is cuda → mps → cpu, with two deliberate refusals:

- `--device mps` on a box that has both does **not** silently take cuda.
- `--device cuda` where cuda is unavailable falls back to **cpu, not mps** —
  a cuda→mps fallback would be a surprise, and cpu is the honest answer.

The three `select_device` tests had to change, and that is worth recording
as a lesson rather than a footnote: they monkeypatched only
`torch.backends.mps.is_available`, so they passed on a Mac and would have
failed on the first CUDA box. A test that fakes one backend while leaving
another live isn't testing a preference order, it's testing the machine it
runs on.

## No CPU fallback on CUDA, on purpose

`_run_step` catches a mid-step `RuntimeError` on MPS and moves the model to
CPU for the rest of the run. That stays MPS-only. Journal 041's finding was
that silently continuing at ~1% speed is the worst available unattended
failure mode; on a rented H100 the ratio is closer to 1/1000, and it would
be burning money by the hour to do it.

So CUDA fails loudly — but diagnosably. A `torch.OutOfMemoryError` now
writes what the allocator held at the moment it died:

```json
{"type": "device_oom", "step": 8412, "device": "cuda",
 "device_allocated_bytes": 78012235776, "device_reserved_bytes": 79456894976,
 "batch_size": 64, "seq_len": 1024}
```

…then re-raises. The real safety net is the machinery 041 already built:
atomic checkpoints, `latest.pt`, and a shell retry loop that re-invokes with
`--resume-from`. `scripts/watch_run.py` alerts on the new record type.

For the same reason `torch.cuda.empty_cache()` is **not** called per step.
Entry 041 added `torch.mps.empty_cache()` because the Metal driver allocator
grew without bound (9.5 → 20.5GB over four steps, with step time going
2.8s → 177s). That reasoning does not transfer: on CUDA the caching
allocator is the point, and emptying it every step forces a synchronization
and guarantees you re-pay every allocation. The call stays MPS-gated with a
comment saying why.

## Memory telemetry: add keys, don't rename them

CUDA has an exact analogue of the MPS pair entry 041 identified:

| what it means | MPS | CUDA |
| --- | --- | --- |
| what tensors currently hold | `current_allocated_memory()` | `memory_allocated()` |
| what the allocator holds from the driver | `driver_allocated_memory()` | `memory_reserved()` |

The second row is the one that predicts an OOM, and entry 041's whole
lesson was that watching only the first row is how a run looks like a clean
fit right up until it dies.

The tempting move is to rename `mps_allocated_bytes` → `device_allocated_bytes`.
I didn't, and the reason is this project's own convention rather than
politeness about old files: **every figure in the journal is supposed to be
readable back out of a real log.** Renaming would leave entry 041's
"9.5 → 20.5GB" citing a key that no longer exists in any tool that reads
these logs. So both are emitted, `watch_run.py` prefers the neutral pair and
falls back to the MPS pair, and a test pins that a pre-042 log still renders.

On MPS the two pairs are identical by construction, which the smoke test
confirms:

```
device                     mps
mps_allocated_bytes        20775168
device_allocated_bytes     20775168
```

## Citing the hardware instead of asserting it

`run_start` now carries `cuda_device_name`, `cuda_capability`,
`cuda_total_memory_bytes`, `cuda_version`, `flash_sdp_enabled` and
`tf32_matmul` — absent entirely off CUDA. Entry 043 will quote an H100 from
this record rather than from a rental listing.

`flash_sdp_enabled` is there because attention is already written as
`F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)`
whenever positional encoding is RoPE or NoPE and document masking is off —
which is the default. So FlashAttention on an H100 is something this project
gets for free and has never verified it is getting. Now the log says.

## TF32

`TrainConfig.tf32: bool = True`, applied as
`torch.set_float32_matmul_precision("high")` when the device is CUDA.

Under bf16 autocast this only touches the ops autocast leaves in fp32, so
the effect is small. It matters for `--precision fp32`, where an H100 does
roughly 50 TFLOP/s in true fp32 against roughly 400 with TF32 — the kind of
8x that looks like "the GPU is broken" rather than like a precision setting.
It defaults on and exists mainly so a numerical-reproducibility run can turn
it off with `--no-tf32`.

## What this entry deliberately does not claim

The projected arithmetic, stated as projection:

```
FLOPs/token = 6 x 121,975,808 (active non-embedding)
            + 6 x  12,582,912 (tied lm_head, vocab 16,384)
            + 6 x 10 x 1024 x 768 (attention scores at seq_len 1024)
            = 8.545e8
```

At an *assumed* 30% MFU on an H100's 989 TFLOP/s dense bf16 peak, that is
347,345 tok/s and 2.15 hours for the 2.69B-token floor. **The 30% is a
guess.** A 125M-active model underfills an H100, so the honest range is
15-40%, which spans 1.6 to 4.3 hours. Entry 043 replaces the guess with
`calibrate_throughput` divided by 8.545e8 — which is only trustworthy
because of the `_synchronize` fix above.

## Tests

**12 added, 3 changed (522 passing, was 510):**

- **`tests/test_trainer_integration.py`** (5 added, 3 changed): cuda preferred
  over mps; the bare-string guarantee; `--device mps` not taking cuda;
  explicit cuda unavailable warns and falls back to cpu, not mps. The three
  changed tests now patch both backend probes via a `_fake_backends` helper.
- **`tests/test_scaling.py`** (4 added): `_synchronize` calls the right
  function per backend, handles `"cuda:1"`, and is a no-op on cpu.
- **`tests/test_watch_run.py`** (3 added): the CUDA memory panel reads the
  device-neutral keys; a pre-042 MPS log still renders; `device_oom` alerts
  and names `latest.pt`.

Every CUDA behaviour is device-gated, so the MPS path is unchanged — a
12-step `SMOKE_TEST` run still trains (loss 4.15 → 3.89, val ppl 54.9 → 48.6)
and emits both key families.

## Questions that might come to mind

**Why a new branch?** `scaling-improvements` is about making training on
*this laptop* reliable. This changes the execution target. Keeping them
apart means a bisect of "when did MPS training change?" doesn't walk through
a CUDA port that, by construction, changed nothing about MPS.

**Does any of this make the model different?** No. `_run_step`,
`_forward_backward_core`, and every module under `qwenmoe/model/` are
untouched. What changed is which device the tensors live on, what gets
logged, and how failures are reported.

**Why not test on CUDA in CI?** There is no GPU here. Everything CUDA-shaped
is tested by monkeypatching the probes (`torch.cuda.is_available`,
`torch.cuda.synchronize`) so the *logic* is covered without the hardware.
What that cannot catch is whether the kernels are fast, which is entry 043's
job and needs a real rental.

**Is `flash_sdp_enabled` telling us attention is actually fast?** It reports
that the flash backend is *permitted*, not that SDPA chose it for a given
call. Turning on `--document-masking` would pass an explicit float mask and
push SDPA onto a slower backend regardless of that flag — which is one
reason document masking stays off for the H100 run.
