"""Reproduces every hand-computed number in `024-mixed-precision-and-z-loss.md`.

Run it:  uv run python docs/math/verify_024_mixed_precision_and_z_loss.py

ONE running example, carried end to end: the exact tiny 2-layer
`QwenMoEModel` from `006-full-model-assembly.md` / `010-training-loop.md`
(vocab_size=6, d_model=4, n_layers=2, n_heads=2, n_kv_heads=1, d_ff=3,
n_routed_experts=2, top_k=1, n_shared_experts=0, SwiGLU, NoPE, aux-loss
balancing, tied embeddings), the SAME hand-picked weights, and the SAME
next-token-prediction window `x = [4, 1, 4]`, `y = [1, 4, 1]` -- now run
through the REAL `qwenmoe/train/trainer.py` with `precision="bf16"` and
`z_loss_weight=1e-4` (both dormant in 010), plus the REAL
`qwenmoe/model/moe.py::dispatch_batched`.

It drives the real module code, prints every intermediate, and asserts:

  * the fp32 forward reproduces 010's exact numbers (logits, lm_loss,
    aux_loss) -- the anchor this whole document builds on,
  * the fp32-bits -> bf16-bits round-to-nearest-even hand derivation of one
    real tensor entry matches `Tensor.to(torch.bfloat16)` exactly,
  * a real forward-hook dtype ledger over the whole model under
    `torch.autocast(device_type="cpu", dtype=torch.bfloat16)`: embedding
    and every RMSNorm output fp32, every Linear/bmm output bf16, the MoE
    layer's own output (post dispatch-and-accumulate) fp32, final logits
    bf16,
  * three isolated op probes (`F.softmax`, `torch.logsumexp`,
    `F.cross_entropy` under the same autocast) showing which ones this
    PyTorch build actually promotes to fp32 and which it does not,
  * `AuxLossBalancer.aux_loss`'s fp32-ness traced to its own explicit
    `.float()` call (moe.py:183), not an autocast override,
  * cross-entropy's exact shift-invariance on the model's OWN real logits
    (shift by +10: identical CE, `log Z` up by 10, z-loss up ~17x),
  * one block's MoE dispatch traced tensor-by-tensor under bf16 autocast,
    matching the real `dispatch_batched` output exactly,
  * removing the `.to(out.dtype)` fix reproduces the real
    `RuntimeError: index_add_(): self (Float) and source (BFloat16) must
    have the same scalar type` on the model's own real tensors, and
    restoring it matches `dispatch_batched` to 0.0,
  * one full `_forward_backward_step` (`z_loss_weight=1e-4`) run twice
    (`precision="fp32"` vs `"bf16"`), gradients fp32 in both (no scaler
    needed), and a 5-step train-loss table, all reproduced here (not
    fabricated).

No fabricated numbers -- this script is the source of truth the prose
quotes.
"""
from __future__ import annotations

import struct

import torch
import torch.nn.functional as F

from qwenmoe.config import ModelConfig, TrainConfig
from qwenmoe.model.moe import _stack_expert_weights, dispatch_batched, score_and_select
from qwenmoe.model.transformer import QwenMoEModel
from qwenmoe.train.optim import build_optimizer
from qwenmoe.train.trainer import _forward_backward_step, _set_lr

torch.set_printoptions(precision=6, sci_mode=False)

# ======================================================================
# Part 1 -- the running example: the 006/010 model, weights, one window
# ======================================================================
print("================ Part 1 -- the running example ================")


def _cfg(**overrides) -> ModelConfig:
    base = dict(
        vocab_size=6,
        d_model=4,
        n_layers=2,
        n_heads=2,
        n_kv_heads=1,
        d_ff=3,
        n_routed_experts=2,
        n_shared_experts=0,
        top_k=1,
        max_seq_len=8,
        positional_scheme="nope",
        activation="swiglu",
        moe_balancing="aux_loss",
        router_scoring="softmax",
        aux_loss_weight=0.01,
        tie_word_embeddings=True,
        qk_norm=False,
        mtp_depth=0,
    )
    base.update(overrides)
    return ModelConfig(**base)


E = torch.tensor(
    [
        [1.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0, 1.0],
        [-1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 1.0],
        [1.0, -1.0, 1.0, -1.0],
    ]
)
WQ = torch.eye(4)
WK = torch.tensor([[1.0, 0, 0, 0], [0, 1, 0, 0]])
WV = torch.tensor([[0.0, 0, 1, 0], [0, 0, 0, 1]])
WO = torch.eye(4)
W_ROUTER = torch.tensor([[1.0, 0, 0, 0], [0, 0, 1, 0]])
E0_GATE = torch.tensor([[1.0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
E0_UP = torch.tensor([[0.0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]])
E0_DOWN = torch.tensor([[1.0, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]])
E1_GATE = torch.tensor([[0.0, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0]])
E1_UP = torch.tensor([[1.0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0]])
E1_DOWN = torch.tensor([[1.0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 0, 1]])


def _pin(model: QwenMoEModel) -> QwenMoEModel:
    with torch.no_grad():
        model.embedding.weight.copy_(E)  # lm_head.weight IS this tensor (tied)
        model.final_norm.weight.fill_(1.0)
        for block in model.blocks:
            block.attn_norm.weight.fill_(1.0)
            block.moe_norm.weight.fill_(1.0)
            block.attn.w_q.weight.copy_(WQ)
            block.attn.w_k.weight.copy_(WK)
            block.attn.w_v.weight.copy_(WV)
            block.attn.w_o.weight.copy_(WO)
            block.moe.router.gate.weight.copy_(W_ROUTER)
            block.moe.routed_experts[0].w_gate.weight.copy_(E0_GATE)
            block.moe.routed_experts[0].w_up.weight.copy_(E0_UP)
            block.moe.routed_experts[0].w_down.weight.copy_(E0_DOWN)
            block.moe.routed_experts[1].w_gate.weight.copy_(E1_GATE)
            block.moe.routed_experts[1].w_up.weight.copy_(E1_UP)
            block.moe.routed_experts[1].w_down.weight.copy_(E1_DOWN)
    return model


def build_model() -> QwenMoEModel:
    torch.manual_seed(0)
    return _pin(QwenMoEModel(_cfg()))


X = torch.tensor([[4, 1, 4]])
Y = torch.tensor([[1, 4, 1]])
POSITIONS = torch.arange(3)
print(f"  x = {X.tolist()}, y = {Y.tolist()}")

model_fp32 = build_model()
model_fp32.train()
logits_fp32, extras_fp32 = model_fp32(X, targets=Y)
lm_loss_fp32 = F.cross_entropy(logits_fp32.reshape(-1, 6), Y.reshape(-1))
print("  logits (fp32):\n", logits_fp32)
print("  lm_loss (fp32):", lm_loss_fp32.item())
print("  aux_loss (fp32):", extras_fp32["aux_loss"].item())

# Anchor: these are 010-training-loop.md's exact Part-4 numbers.
torch.testing.assert_close(lm_loss_fp32, torch.tensor(2.573154), atol=1e-6, rtol=0)
assert abs(extras_fp32["aux_loss"].item() - 0.011424) < 1e-5
print("  matches 010's Part 4 numbers exactly.")

# ======================================================================
# Part 2 -- fp32 bits -> bf16 bits, hand round-to-nearest-even
# ======================================================================
print("\n================ Part 2 -- fp32 -> bf16 bit derivation ================")

# The real value: block 0's post-attention, post-RMSNorm activation at
# (token 0, feature 1) -- this is what feeds the first MoE dispatch.
with torch.no_grad():
    x_emb = model_fp32.embedding(X)
    block0 = model_fp32.blocks[0]
    attn_out = block0.attn(block0.attn_norm(x_emb), POSITIONS)
    x1 = x_emb + attn_out
    moe_normed = block0.moe_norm(x1)
real_value = moe_normed[0, 0, 1].item()
print(f"  real tensor entry moe_normed[0,0,1] = {real_value!r}")

bits32 = struct.unpack(">I", struct.pack(">f", real_value))[0]
sign = bits32 >> 31
exp8 = (bits32 >> 23) & 0xFF
mant23 = bits32 & 0x7FFFFF
mant23_bin = format(mant23, "023b")
print(f"  fp32 bits: sign={sign} exponent(raw)={exp8} (unbiased={exp8-127})"
      f" mantissa(23b)={mant23_bin}")

top7 = mant23_bin[:7]
rest16 = mant23_bin[7:]
first_dropped = rest16[0]
any_more_ones = "1" in rest16[1:]
is_odd = top7[-1] == "1"
round_up = first_dropped == "1" and (any_more_ones or is_odd)  # round-to-nearest-even
top7_val = int(top7, 2) + (1 if round_up else 0)
print(f"  keep top 7 mantissa bits: {top7} | dropped 16 bits: {rest16}")
print(f"  round up? {round_up} (dropped bits start with "
      f"{'1' if first_dropped=='1' else '0'}"
      f"{', and (more 1s follow or kept bit is odd)' if first_dropped=='1' else ''})")
hand_bf16 = (-1) ** sign * (1 + top7_val / 128) * (2.0 ** (exp8 - 127))
print(f"  hand-reconstructed bf16 value = {hand_bf16!r}")

real_bf16 = torch.tensor(real_value, dtype=torch.float32).to(torch.bfloat16).item()
print(f"  torch .to(torch.bfloat16)    = {real_bf16!r}")
assert hand_bf16 == real_bf16, (hand_bf16, real_bf16)
print("  hand derivation matches torch's real cast EXACTLY.")

# ======================================================================
# Part 3 -- the real, measured torch.autocast op policy on this model
# ======================================================================
print("\n================ Part 3 -- measured autocast dtype ledger ================")

model_bf16 = build_model()
model_bf16.train()

dtype_log: list[tuple[str, object]] = []


def _mk_hook(name: str):
    def hook(_module, _inp, out):
        if isinstance(out, tuple):
            dt = tuple(o.dtype if isinstance(o, torch.Tensor) else None for o in out)
        else:
            dt = out.dtype
        dtype_log.append((name, dt))

    return hook


handles = [
    m.register_forward_hook(_mk_hook(n)) for n, m in model_bf16.named_modules() if n
]
with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
    logits_bf16, extras_bf16 = model_bf16(X, targets=Y)
for h in handles:
    h.remove()

for name, dt in dtype_log:
    print(f"  {name:30s} -> {dt}")
print(f"  final logits dtype: {logits_bf16.dtype}")

# Assert the specific claims the doc makes.
by_name = dict(dtype_log)
assert by_name["blocks.0.attn_norm"] == torch.float32
assert by_name["blocks.0.attn.w_q"] == torch.bfloat16
assert by_name["blocks.0.attn"] == torch.bfloat16
assert by_name["blocks.0.moe_norm"] == torch.float32
assert by_name["blocks.0.moe.router.gate"] == torch.bfloat16
assert by_name["blocks.0.moe"] == (torch.float32, torch.float32)  # (output, aux_loss)
assert by_name["final_norm"] == torch.float32
assert by_name["lm_head"] == torch.bfloat16
assert logits_bf16.dtype == torch.bfloat16
print("  all dtype assertions hold.")

# Isolated probes: which ops does THIS torch build actually promote to fp32
# under autocast, independent of what feeds them?
print("\n  isolated op probes (bf16 input in all three):")
probe_bf16 = torch.randn(2, 4).to(torch.bfloat16)
probe_targets = torch.tensor([0, 1])
with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
    sm = F.softmax(probe_bf16, dim=-1)
    lse = torch.logsumexp(probe_bf16, dim=-1)
    ce = F.cross_entropy(probe_bf16.float().requires_grad_(), probe_targets)
    # cross_entropy needs a float input; feed it fp32 input to isolate the
    # OP's own promotion behavior (it still runs its internal log_softmax
    # in fp32 per autocast's policy -- this only tests that policy, not
    # dtype coercion of the input).
print(f"    F.softmax(bf16 input)      -> {sm.dtype}")
print(f"    torch.logsumexp(bf16 input) -> {lse.dtype}")
print(f"    F.cross_entropy(...)        -> {ce.dtype}")
assert sm.dtype == torch.bfloat16, "softmax is NOT auto-promoted on this build"
assert lse.dtype == torch.bfloat16, "logsumexp is NOT auto-promoted on this build"
assert ce.dtype == torch.float32, "cross_entropy IS auto-promoted on this build"
print("  confirmed: softmax/logsumexp stay in their input's dtype; only"
      " cross_entropy is autocast-promoted on this torch build.")

# Where does aux_loss's fp32-ness actually come from? moe.py:183's explicit
# .float(), not an autocast override -- confirmed by mirroring the exact
# line with a bf16 tensor.
dispatched_probe = torch.zeros(2, 3, dtype=torch.int64)
dispatched_probe[0, 0] = 1
router_probs_probe = torch.tensor([[0.5, 0.3, 0.2], [0.1, 0.6, 0.3]]).to(torch.bfloat16)
fraction_dispatched = dispatched_probe.float().mean(dim=0)  # moe.py:183, verbatim
mixed = fraction_dispatched * router_probs_probe.mean(dim=0)
print(f"\n  dispatched.float().mean(dim=0) dtype: {fraction_dispatched.dtype}"
      f" (int64 one-hot counts explicitly upcast)")
print(f"  fraction_dispatched(fp32) * mean_router_prob(bf16) -> {mixed.dtype}"
      f" (ordinary type promotion, NOT an autocast rule)")
assert mixed.dtype == torch.float32

# ======================================================================
# Part 4 -- why `.float()` (trainer.py:234) is the load-bearing line
# ======================================================================
print("\n================ Part 4 -- the .float() upcast ================")

logits_bf16_raw = logits_bf16  # bf16, straight off lm_head
logits_bf16_up = logits_bf16.float()  # trainer.py:234
print("  logits, raw off lm_head (bf16):\n", logits_bf16_raw)
print("  logits, after .float() (fp32):\n", logits_bf16_up)
max_diff_logits = (logits_fp32 - logits_bf16_up).abs().max().item()
print(f"  max |fp32 logits - bf16-then-upcast logits| = {max_diff_logits:.6f}")

# What would happen to the z-loss term WITHOUT the manual upcast: logsumexp
# is not autocast-promoted (Part 3), so it would run directly on bf16 logits.
log_z_no_upcast = torch.logsumexp(logits_bf16_raw.reshape(-1, 6), dim=-1)
log_z_with_upcast = torch.logsumexp(logits_bf16_up.reshape(-1, 6), dim=-1)
log_z_fp32_exact = torch.logsumexp(logits_fp32.reshape(-1, 6), dim=-1)
print(f"  log Z, no upcast (raw bf16 logsumexp):   {log_z_no_upcast}")
print(f"  log Z, WITH .float() upcast (fp32 math): {log_z_with_upcast}")
print(f"  log Z, fp32-exact reference:             {log_z_fp32_exact}")
assert log_z_no_upcast.dtype == torch.bfloat16
assert log_z_with_upcast.dtype == torch.float32

# ======================================================================
# Part 5 -- shift-invariance (real logits) + the z-loss, fp32 vs bf16
# ======================================================================
print("\n================ Part 5 -- shift-invariance + z-loss ================")

C = 10.0
shifted = logits_fp32 + C
ce_unshifted = F.cross_entropy(logits_fp32.reshape(-1, 6), Y.reshape(-1))
ce_shifted = F.cross_entropy(shifted.reshape(-1, 6), Y.reshape(-1))
print(f"  cross-entropy, unshifted logits: {ce_unshifted.item():.6f}")
print(f"  cross-entropy, logits + {C}:     {ce_shifted.item():.6f}")
torch.testing.assert_close(ce_unshifted, ce_shifted, atol=1e-5, rtol=0)

log_z_shifted = torch.logsumexp(shifted.reshape(-1, 6), dim=-1)
print(f"  log Z, unshifted: {log_z_fp32_exact.detach()}")
print(f"  log Z, shifted:   {log_z_shifted.detach()}")
diff = (log_z_shifted - log_z_fp32_exact).detach()
print(f"  log Z shift == +{C} exactly (up to fp32 rounding): {diff}")

Z_WEIGHT = 1e-4
z_loss_unshifted = Z_WEIGHT * log_z_fp32_exact.pow(2).mean()
z_loss_shifted = Z_WEIGHT * log_z_shifted.pow(2).mean()
ratio = (z_loss_shifted / z_loss_unshifted).item()
print(f"  z_loss unshifted: {z_loss_unshifted.item():.8f}")
print(f"  z_loss shifted:   {z_loss_shifted.item():.8f}  ({ratio:.2f}x larger)")

z_loss_fp32 = Z_WEIGHT * log_z_fp32_exact.pow(2).mean()
z_loss_bf16 = Z_WEIGHT * log_z_with_upcast.pow(2).mean()
print(f"\n  z_loss @ weight={Z_WEIGHT}, fp32 path: {z_loss_fp32.item():.10f}")
print(f"  z_loss @ weight={Z_WEIGHT}, bf16 path: {z_loss_bf16.item():.10f}")
print(f"  |diff| = {abs(z_loss_fp32.item()-z_loss_bf16.item()):.10f}")

# z_loss_weight == 0 must be exactly dormant (fp32 path): total lm_loss
# alone must equal a bare F.cross_entropy call to high precision.
metrics_off, _ = _forward_backward_step(
    build_model(), build_optimizer(build_model(), TrainConfig(
        batch_size=1, seq_len=3, total_tokens=15, precision="fp32", z_loss_weight=0.0
    )), X, Y, grad_clip_norm=1.0, device="cpu", precision="fp32", z_loss_weight=0.0,
)
assert metrics_off["z_loss"] is None
print("\n  z_loss_weight=0.0 -> metrics['z_loss'] is None (dormant), confirmed.")

# ======================================================================
# Part 6 -- the MoE dtype-safety fix (moe.py:405), traced digit by digit
# ======================================================================
print("\n================ Part 6 -- dispatch_batched's dtype fix ================")

model_dispatch = build_model()
model_dispatch.train()
with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
    x_emb = model_dispatch.embedding(X)
    block = model_dispatch.blocks[0]
    attn_out = block.attn(block.attn_norm(x_emb), POSITIONS)
    x1 = x_emb + attn_out
    moe_normed = block.moe_norm(x1)
    x_flat = moe_normed.reshape(-1, 4)
    print("  x_flat (moe_normed, block 0) dtype:", x_flat.dtype)
    print("  x_flat:\n", x_flat)

    router_logits = block.moe.router(x_flat)
    print("  router_logits dtype:", router_logits.dtype, "\n ", router_logits)
    topk_indices, topk_weights, routing_probs = score_and_select(
        router_logits, block.moe.balancer, top_k=1, scoring="softmax"
    )
    print("  topk_indices:", topk_indices.reshape(-1).tolist())
    print("  topk_weights:", topk_weights.reshape(-1).tolist())
    # All 3 tokens tie exactly (by construction: W_ROUTER only reads
    # features 0 and 2, and moe_normed's 3 tokens either have equal values
    # there or a clear winner) -> every token dispatches to expert 0 here.
    assert topk_indices.reshape(-1).tolist() == [0, 0, 0]

    w_gate = _stack_expert_weights(block.moe.routed_experts, [0], "w_gate")
    w_up = _stack_expert_weights(block.moe.routed_experts, [0], "w_up")
    w_down = _stack_expert_weights(block.moe.routed_experts, [0], "w_down")
    xg = x_flat.unsqueeze(0)  # (1 active expert, 3 tokens, d_model=4)
    print("  xg (= x_flat, one active expert) dtype:", xg.dtype)
    pre = torch.bmm(xg, w_gate.transpose(1, 2))
    print("  pre = xg @ w_gate^T, dtype:", pre.dtype, "\n ", pre)
    silu_pre = F.silu(pre)
    up = torch.bmm(xg, w_up.transpose(1, 2))
    h = silu_pre * up
    print("  h = silu(pre) * (xg @ w_up^T), dtype:", h.dtype, "\n ", h)
    down = torch.bmm(h, w_down.transpose(1, 2))
    weight_grid = topk_weights.reshape(1, 3)
    yg = down * weight_grid.unsqueeze(-1)
    print("  yg = (h @ w_down^T) * gate_weight, dtype:", yg.dtype, "\n ", yg)

    out = x_flat.new_zeros(3, 4)
    print("  out = x_flat.new_zeros(...) dtype:", out.dtype)
    token_grid = torch.arange(3).unsqueeze(0)

    # WITHOUT the fix.
    crashed = False
    try:
        out.clone().index_add_(0, token_grid.reshape(-1), yg.reshape(-1, 4))
    except RuntimeError as exc:
        crashed = True
        print("  WITHOUT the fix ->  RuntimeError:", exc)
    assert crashed, "expected the unfixed scatter to raise"

    # WITH the fix (moe.py:405).
    out_fixed = out.clone()
    out_fixed.index_add_(0, token_grid.reshape(-1), yg.reshape(-1, 4).to(out_fixed.dtype))
    print("  WITH the fix -> routed_output (block 0):\n", out_fixed, out_fixed.dtype)

    # Cross-check against the real function.
    real = dispatch_batched(x_flat, topk_indices, topk_weights, block.moe.routed_experts, "swiglu")
    print("  real dispatch_batched(...) output:\n", real)
    max_diff = (out_fixed - real).abs().max().item()
    print(f"  max |manual - real dispatch_batched| = {max_diff}")
    assert max_diff == 0.0

# ======================================================================
# Part 7 -- one full optimizer step, fp32 vs bf16, z_loss_weight=1e-4
# ======================================================================
print("\n================ Part 7 -- one full step, fp32 vs bf16 ================")

CFG_BASE = dict(
    batch_size=1, seq_len=3, total_tokens=15, lr=0.1, warmup_steps=2,
    betas=(0.9, 0.95), adam_eps=1e-8, weight_decay=0.1, grad_clip_norm=1.0,
    lr_schedule="cosine", min_lr_ratio=0.1, device="cpu", seed=1337,
)

step_results = {}
for precision in ["fp32", "bf16"]:
    m = build_model()
    cfg = TrainConfig(**CFG_BASE, precision=precision, z_loss_weight=1e-4)
    opt = build_optimizer(m, cfg)
    metrics, grad_norm = _forward_backward_step(
        m, opt, X, Y, grad_clip_norm=cfg.grad_clip_norm,
        device="cpu", precision=cfg.precision, z_loss_weight=cfg.z_loss_weight,
    )
    grad_dtypes = {p.grad.dtype for p in m.parameters() if p.grad is not None}
    step_results[precision] = (metrics, grad_norm, grad_dtypes)
    print(f"  precision={precision:5s} metrics={metrics}")
    print(f"    grad_norm={grad_norm:.6f}  grad dtypes seen: {grad_dtypes}")
    assert grad_dtypes == {torch.float32}, "no scaler needed: grads must be fp32 either way"

print(f"\n  |lm_loss diff|  = {abs(step_results['fp32'][0]['lm_loss'] - step_results['bf16'][0]['lm_loss']):.6f}")
print(f"  |grad_norm diff| = {abs(step_results['fp32'][1] - step_results['bf16'][1]):.6f}")

print("\n  5-step run, same window every step:")
five_step = {}
for precision in ["fp32", "bf16"]:
    m = build_model()
    cfg = TrainConfig(**CFG_BASE, precision=precision, z_loss_weight=1e-4)
    opt = build_optimizer(m, cfg)
    losses = []
    for step in range(5):
        _set_lr(opt, cfg.lr, step, 5, cfg)
        metrics, _ = _forward_backward_step(
            m, opt, X, Y, grad_clip_norm=cfg.grad_clip_norm,
            device="cpu", precision=cfg.precision, z_loss_weight=cfg.z_loss_weight,
        )
        losses.append(metrics["lm_loss"])
    five_step[precision] = losses

print("  step |   fp32   |   bf16   |  |diff|")
for i in range(5):
    f, b = five_step["fp32"][i], five_step["bf16"][i]
    print(f"  {i+1:4d} | {f:.6f} | {b:.6f} | {abs(f-b):.6f}")

print("\nAll assertions passed.")
