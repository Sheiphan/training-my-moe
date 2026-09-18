"""Reproduces every hand-computed number in `009-scaling-law-utilities.md`.

Run it:  uv run python docs/math/verify_009_scaling_law_utilities.py

It carries ONE running example -- the tiny MoE model from
`tests/test_scaling.py::_tiny_config` (d_model=16, n_heads=4, n_kv_heads=2,
d_ff=32, n_layers=2, n_routed_experts=4, n_shared_experts=1, top_k=2,
vocab_size=50, tied embeddings, SwiGLU experts, RoPE) -- through every
function in `qwenmoe/train/scaling.py`, driving the REAL module code,
printing every intermediate, and `assert`ing:

  * the model's full 46-tensor parameter inventory (name, shape, numel),
  * `count_params`: the per-bucket walk -> ParamBreakdown(embedding=816,
    attention_total=1568, moe_total=15520, moe_active=9376, total=17904,
    active=11760), each field hand-derived,
  * cross-check: `total` == a raw, identity-dedup'd sum of
    `model.parameters()` (independent of count_params' own bucketing),
  * the shape rules every count rests on: a bias-free `nn.Linear(in, out)`
    stores an `(out, in)` matrix of `out * in` numbers; `nn.Embedding(V, d)`
    stores `V * d`; `RMSNorm(d)` stores `d`; GQA makes w_k / w_v exactly
    `n_kv_heads / n_heads` the size of w_q,
  * total - active == sum_layers (E - k) * P_expert == 2 * (4 - 2) * 1536
    == 6144 -- the routed-expert weight no single token ever touches,
  * the untied-embedding variant adds a separate `V * d` output head,
  * `chinchilla_budget`: 20 * 11760 == 235_200, the >= 1 guard,
  * `moe_budget_note`: sparsity 17904 / 11760 == 1.52x, and the note quotes
    both 20*active == 235_200 and 20*total == 358_080 (Clark et al. 2022),
  * `epochs_for_budget`: total / unique, with the WARNING firing at 5.0
    epochs (235_200 / 47_040) and NOT at 4.0 (235_200 / 58_800), captured
    with a real logging handler; the >= 1 guards,
  * `calibrate_throughput`: with `time.perf_counter` monkeypatched to a
    fake clock (elapsed forced to exactly 2.5 s), tokens/sec ==
    batch*seq_len*n_steps / elapsed == 4*8*10 / 2.5 == 128.0 EXACTLY; then
    a real, un-patched run on cpu (and mps if present) whose number is
    machine-dependent and only printed, not asserted; the n_steps >= 1
    guard; and that the model's weights are byte-identical afterwards (no
    optimizer.step is ever called),
  * the end-to-end planning chain: model -> count_params -> active ->
    chinchilla_budget -> epochs_for_budget, plus calibrate -> tokens/sec ->
    estimated wall-clock = total_tokens / tokens_per_sec.

Unlike the attention / MoE verifiers there is no fused-vs-explicit kernel to
reconcile: `count_params` walks `nn.Module.parameters()` and the rest is
scalar arithmetic. The one "two methods agree" check is Part 3's
bucketed-sum vs. raw-dedup'd-sum. No fabricated numbers: this script is the
source of truth the prose quotes.
"""
from __future__ import annotations

import logging

import torch

import qwenmoe.train.scaling as scaling
from qwenmoe.config import ModelConfig
from qwenmoe.model.transformer import QwenMoEModel
from qwenmoe.train.scaling import (
    ParamBreakdown,
    calibrate_throughput,
    chinchilla_budget,
    count_params,
    epochs_for_budget,
    moe_budget_note,
)

torch.set_printoptions(precision=6, sci_mode=False)


def _tiny_config(**overrides) -> ModelConfig:
    base = dict(
        vocab_size=50,
        d_model=16,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        d_ff=32,
        n_routed_experts=4,
        n_shared_experts=1,
        top_k=2,
        max_seq_len=16,
    )
    base.update(overrides)
    return ModelConfig(**base)


# ----------------------------------------------------------------------
# Part 1 -- the running example: one tiny model, its full parameter list.
# ----------------------------------------------------------------------
print("================ running example: the tiny MoE model ================")
torch.manual_seed(0)
CONFIG = _tiny_config()
model = QwenMoEModel(CONFIG)
model.eval()

HEAD_DIM = CONFIG.d_model // CONFIG.n_heads
print(
    f"d_model={CONFIG.d_model} n_heads={CONFIG.n_heads} n_kv_heads={CONFIG.n_kv_heads} "
    f"head_dim={HEAD_DIM} d_ff={CONFIG.d_ff} n_layers={CONFIG.n_layers}"
)
print(
    f"n_routed_experts={CONFIG.n_routed_experts} n_shared_experts={CONFIG.n_shared_experts} "
    f"top_k={CONFIG.top_k} vocab_size={CONFIG.vocab_size} "
    f"tie_word_embeddings={CONFIG.tie_word_embeddings} activation={CONFIG.activation!r} "
    f"positional_scheme={CONFIG.positional_scheme!r}"
)

inventory = [(name, tuple(p.shape), p.numel()) for name, p in model.named_parameters()]
print(f"\n{len(inventory)} parameter tensors (name, shape, numel):")
for name, shape, numel in inventory:
    print(f"  {name:44s} {str(shape):10s} {numel:5d}")

raw_sum = sum(numel for _, _, numel in inventory)
print(f"\nsum of every tensor in named_parameters() = {raw_sum}")
assert len(inventory) == 46
# tied lm_head does not appear in named_parameters() -- PyTorch dedups it
assert not any(name.startswith("lm_head") for name, _, _ in inventory)
assert model.lm_head.weight is model.embedding.weight
assert raw_sum == 17904


# ----------------------------------------------------------------------
# Part 2 -- count_params: the per-bucket walk.
# ----------------------------------------------------------------------
print("\n================ Part 2 -- count_params ================")


def _numel(module) -> int:
    return sum(p.numel() for p in module.parameters())


# -- embedding bucket (scaling.py:75-77) --
emb_table = _numel(model.embedding)
final_norm = _numel(model.final_norm)
embedding = emb_table + final_norm
print(f"  embedding table  50 x 16              = {emb_table}")
print(f"  final_norm (RMSNorm(16))              = {final_norm}")
print(f"  embedding bucket                      = {embedding}")
assert emb_table == 800 and final_norm == 16 and embedding == 816

# -- attention + moe buckets, walked per block (scaling.py:60-73) --
attention_total = 0
moe_total = 0
moe_active = 0
for i, block in enumerate(model.blocks):
    a_norm = _numel(block.attn_norm)
    a_attn = _numel(block.attn)
    attention_total += a_norm + a_attn

    moe = block.moe
    moe_norm_params = _numel(block.moe_norm)
    router_params = _numel(moe.router)
    shared_params = _numel(moe.shared_experts)
    routed_total = _numel(moe.routed_experts)
    routed_per_expert = routed_total // len(moe.routed_experts)

    moe_total += moe_norm_params + router_params + shared_params + routed_total
    moe_active += (
        moe_norm_params + router_params + shared_params + CONFIG.top_k * routed_per_expert
    )
    print(
        f"  block {i}: attn_norm {a_norm:2d} + attn {a_attn:3d} = {a_norm + a_attn:3d} | "
        f"moe_norm {moe_norm_params} + router {router_params} + shared {shared_params} "
        f"+ routed {routed_total} (={routed_per_expert}/expert)"
    )

print(f"  attention_total = {attention_total}")
print(f"  moe_total       = {moe_total}")
print(f"  moe_active      = {moe_active}")
assert attention_total == 1568
assert moe_total == 15520
assert moe_active == 9376

total = embedding + attention_total + moe_total
active = embedding + attention_total + moe_active
print(f"  total  = {embedding} + {attention_total} + {moe_total} = {total}")
print(f"  active = {embedding} + {attention_total} + {moe_active} = {active}")

breakdown = count_params(model)
print(f"\n  count_params(model) = {breakdown}")
assert breakdown == ParamBreakdown(
    embedding=816,
    attention_total=1568,
    moe_total=15520,
    moe_active=9376,
    total=17904,
    active=11760,
)
assert (breakdown.embedding, breakdown.attention_total, breakdown.moe_total) == (816, 1568, 15520)
assert breakdown.total == total == 17904
assert breakdown.active == active == 11760


# ----------------------------------------------------------------------
# Part 3 -- cross-check total against a raw, identity-dedup'd sum.
# ----------------------------------------------------------------------
print("\n================ Part 3 -- independent cross-check ================")
seen_ids = set()
real_total = 0
for p in model.parameters():
    if id(p) not in seen_ids:
        seen_ids.add(id(p))
        real_total += p.numel()
print(f"  dedup'd sum of model.parameters()     = {real_total}")
print(f"  count_params(...).total               = {breakdown.total}")
assert breakdown.total == real_total == 17904
# nothing counted twice, nothing missed: buckets partition the model exactly
assert breakdown.embedding + breakdown.attention_total + breakdown.moe_total == real_total


# ----------------------------------------------------------------------
# Part 4 -- the shape rules every count rests on.
# ----------------------------------------------------------------------
print("\n================ Part 4 -- shape rules ================")

# (a) a bias-free nn.Linear(in_features, out_features) stores (out, in)
lin = model.blocks[0].attn.w_q
print(f"  w_q: nn.Linear({lin.in_features}, {lin.out_features}) -> weight {tuple(lin.weight.shape)} "
      f"= {lin.out_features} x {lin.in_features} = {lin.weight.numel()} numbers, bias={lin.bias}")
assert lin.weight.shape == (lin.out_features, lin.in_features)
assert lin.weight.numel() == lin.out_features * lin.in_features
assert lin.bias is None

# (b) nn.Embedding(V, d) stores V x d
emb = model.embedding
assert emb.weight.shape == (CONFIG.vocab_size, CONFIG.d_model)
assert emb.weight.numel() == CONFIG.vocab_size * CONFIG.d_model == 800

# (c) RMSNorm(d) stores exactly d (one gain per feature), initialised to 1
rms = model.final_norm
assert rms.weight.shape == (CONFIG.d_model,)
assert rms.weight.numel() == CONFIG.d_model == 16
assert torch.equal(rms.weight, torch.ones(CONFIG.d_model))

# (d) GQA: w_k / w_v are exactly n_kv_heads / n_heads the size of w_q
w_q = model.blocks[0].attn.w_q.weight.numel()
w_k = model.blocks[0].attn.w_k.weight.numel()
w_v = model.blocks[0].attn.w_v.weight.numel()
w_o = model.blocks[0].attn.w_o.weight.numel()
print(f"  w_q={w_q}  w_k={w_k}  w_v={w_v}  w_o={w_o}   "
      f"w_k/w_q = {w_k}/{w_q} = {w_k / w_q} = n_kv_heads/n_heads = "
      f"{CONFIG.n_kv_heads}/{CONFIG.n_heads}")
assert w_k == w_v == w_q * CONFIG.n_kv_heads // CONFIG.n_heads == 128
assert w_q == w_o == CONFIG.d_model * CONFIG.n_heads * HEAD_DIM == 256

# (e) one SwiGLU expert = w_gate (d_ff x d_model) + w_up (d_ff x d_model) + w_down (d_model x d_ff)
expert = model.blocks[0].moe.routed_experts[0]
p_expert = _numel(expert)
print(f"  one SwiGLU expert = w_gate {expert.w_gate.weight.numel()} + w_up "
      f"{expert.w_up.weight.numel()} + w_down {expert.w_down.weight.numel()} = {p_expert}")
assert p_expert == 3 * CONFIG.d_ff * CONFIG.d_model == 1536
routed_total = _numel(model.blocks[0].moe.routed_experts)
assert routed_total // len(model.blocks[0].moe.routed_experts) == p_expert == 1536

# (f) total - active == sum_layers (E - k) * P_expert
never_touched = CONFIG.n_layers * (CONFIG.n_routed_experts - CONFIG.top_k) * p_expert
print(f"  total - active = {breakdown.total - breakdown.active}  ==  "
      f"n_layers*(E-k)*P_expert = {CONFIG.n_layers}*({CONFIG.n_routed_experts}-{CONFIG.top_k})"
      f"*{p_expert} = {never_touched}")
assert breakdown.total - breakdown.active == never_touched == 6144

# (g) untied embeddings add a separate V x d output head
untied = count_params(QwenMoEModel(_tiny_config(tie_word_embeddings=False)))
print(f"  untied: embedding {untied.embedding} (= 816 + 800), total {untied.total}")
assert untied.embedding == 816 + CONFIG.vocab_size * CONFIG.d_model == 1616
assert untied.total == 17904 + 800 == 18704


# ----------------------------------------------------------------------
# Part 5 -- chinchilla_budget.
# ----------------------------------------------------------------------
print("\n================ Part 5 -- chinchilla_budget ================")
budget = chinchilla_budget(breakdown.active)
print(f"  chinchilla_budget({breakdown.active}) = 20 * {breakdown.active} = {budget}")
assert budget == 20 * 11760 == 235_200
assert chinchilla_budget(1) == 20
for bad in (0, -5):
    try:
        chinchilla_budget(bad)
    except ValueError:
        pass
    else:
        raise AssertionError(f"expected ValueError for active_params={bad}")


# ----------------------------------------------------------------------
# Part 6 -- moe_budget_note (journal 033 / Requirement 12.6).
# ----------------------------------------------------------------------
print("\n================ Part 6 -- moe_budget_note ================")
sparsity = breakdown.total / breakdown.active
note = moe_budget_note(breakdown.total, breakdown.active)
print(f"  sparsity = {breakdown.total} / {breakdown.active} = {sparsity:.4f}x")
print(f"  note: {note}")
assert f"{sparsity:.1f}x" == "1.5x"
assert f"{20 * breakdown.active:,}" == "235,200"
assert f"{20 * breakdown.total:,}" == "358,080"
assert "235,200" in note and "358,080" in note
assert "Clark et al. 2022" in note and "floor" in note.lower()
for bad in ((0, 10), (100, 0), (50, 100)):
    try:
        moe_budget_note(*bad)
    except ValueError:
        pass
    else:
        raise AssertionError(f"expected ValueError for {bad}")


# ----------------------------------------------------------------------
# Part 7 -- epochs_for_budget (with a real logging handler).
# ----------------------------------------------------------------------
print("\n================ Part 7 -- epochs_for_budget ================")


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record):
        self.messages.append(record.getMessage())


cap = _Capture()
scaling.logger.addHandler(cap)
scaling.logger.setLevel(logging.WARNING)
try:
    cap.messages.clear()
    e5 = epochs_for_budget(235_200, 47_040)
    print(f"  epochs_for_budget(235_200, 47_040) = {235_200}/{47_040} = {e5}  "
          f"-> warnings: {len(cap.messages)}")
    assert e5 == 5.0 and len(cap.messages) == 1 and "epoch" in cap.messages[0].lower()

    cap.messages.clear()
    e4 = epochs_for_budget(235_200, 58_800)
    print(f"  epochs_for_budget(235_200, 58_800) = {235_200}/{58_800} = {e4}  "
          f"-> warnings: {len(cap.messages)}")
    assert e4 == 4.0 and len(cap.messages) == 0

    cap.messages.clear()
    e2 = epochs_for_budget(235_200, 117_600)
    print(f"  epochs_for_budget(235_200, 117_600) = {e2}  -> warnings: {len(cap.messages)}")
    assert e2 == 2.0 and len(cap.messages) == 0
finally:
    scaling.logger.removeHandler(cap)

for bad in ({"total_tokens": 0}, {"unique_tokens": 0}):
    kwargs = {"total_tokens": 100, "unique_tokens": 100, **bad}
    try:
        epochs_for_budget(**kwargs)
    except ValueError:
        pass
    else:
        raise AssertionError(f"expected ValueError for {bad}")


# ----------------------------------------------------------------------
# Part 8 -- calibrate_throughput.
# ----------------------------------------------------------------------
print("\n================ Part 8 -- calibrate_throughput ================")
BATCH, SEQ = 4, 8
torch.manual_seed(0)
x = torch.randint(0, CONFIG.vocab_size, (BATCH, SEQ))
y = torch.randint(0, CONFIG.vocab_size, (BATCH, SEQ))
tokens_per_step = BATCH * SEQ
print(f"  sample batch shape ({BATCH}, {SEQ}) -> tokens_per_step = {BATCH} * {SEQ} = {tokens_per_step}")

# (a) EXACT check with a fake clock: force elapsed = 2.5 s over n_steps = 10.
clock_model = QwenMoEModel(_tiny_config())
fake_clock = iter([100.0, 102.5])
real_perf_counter = scaling.time.perf_counter
scaling.time.perf_counter = lambda: next(fake_clock)
try:
    tps_fake = calibrate_throughput(clock_model, (x, y), device="cpu", n_steps=10)
finally:
    scaling.time.perf_counter = real_perf_counter
print(f"  fake clock (elapsed forced to 2.5 s): {tokens_per_step} * 10 / 2.5 = {tps_fake}")
assert tps_fake == tokens_per_step * 10 / 2.5 == 128.0

# (b) real, un-patched runs -- machine-dependent, PRINTED not asserted.
before = model.blocks[0].attn.w_q.weight.detach().clone()
tps_cpu = calibrate_throughput(model, (x, y), device="cpu", n_steps=20)
after = model.blocks[0].attn.w_q.weight.detach().clone()
print(f"  real cpu: {tps_cpu:.1f} tokens/sec  (elapsed ~= {tokens_per_step * 20 / tps_cpu:.4f} s)")
assert tps_cpu > 0 and torch.isfinite(torch.tensor(tps_cpu))
# no optimizer.step is ever called: weights are byte-identical afterwards
assert torch.equal(before, after)

if torch.backends.mps.is_available():
    tps_mps = calibrate_throughput(QwenMoEModel(_tiny_config()), (x, y), device="mps", n_steps=20)
    print(f"  real mps: {tps_mps:.1f} tokens/sec")
    assert tps_mps > 0

# (c) the n_steps >= 1 guard
try:
    calibrate_throughput(model, (x, y), device="cpu", n_steps=0)
except ValueError:
    pass
else:
    raise AssertionError("expected ValueError for n_steps=0")


# ----------------------------------------------------------------------
# Part 9 -- the end-to-end planning chain.
# ----------------------------------------------------------------------
print("\n================ Part 9 -- end to end ================")
pb = count_params(model)
d_ref = chinchilla_budget(pb.active)
UNIQUE_TOKENS = 47_040  # a toy corpus size, chosen so epochs comes out whole
planned_tokens = d_ref  # plan exactly the reference budget
ep = epochs_for_budget(planned_tokens, UNIQUE_TOKENS)
eta_seconds = planned_tokens / tps_cpu
print(f"  count_params(model).active           = {pb.active}")
print(f"  chinchilla_budget(active)            = {d_ref}")
print(f"  epochs_for_budget({d_ref}, {UNIQUE_TOKENS}) = {ep}")
print(f"  calibrate_throughput(...) [cpu]      = {tps_cpu:.1f} tokens/sec")
print(f"  estimated wall-clock = {d_ref} / {tps_cpu:.1f} = {eta_seconds:.1f} s")
assert pb.active == 11_760
assert d_ref == 235_200
assert ep == 5.0

print("\nParamBreakdown (quotable):")
print(f"  {count_params(model)}")

print("\nAll assertions passed -- every number in 009-scaling-law-utilities.md is real.")
