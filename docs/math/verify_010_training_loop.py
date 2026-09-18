"""Reproduces every hand-computed number in `010-training-loop.md`.

Run it:  uv run python docs/math/verify_010_training_loop.py

ONE running example, carried end to end: the exact tiny 2-layer
`QwenMoEModel` from `006-full-model-assembly.md` (vocab_size=6, d_model=4,
n_layers=2, n_heads=2, n_kv_heads=1, d_ff=3, n_routed_experts=2, top_k=1,
n_shared_experts=0, SwiGLU, NoPE, aux-loss balancing, tied embeddings),
with the SAME hand-picked weights pinned in, trained by the REAL
`qwenmoe/train/trainer.py` on ONE next-token-prediction window
`x = [4, 1, 4]`, `y = [1, 4, 1]` (token stream `[4, 1, 4, 1]`) reused every
step, for 5 optimizer steps with

    lr=0.1, warmup_steps=2, total_steps=5, betas=(0.9, 0.95), adam_eps=1e-8,
    weight_decay=0.1, grad_clip_norm=1.0, lr_schedule="cosine",
    min_lr_ratio=0.1, precision="fp32".

It drives the real module code, prints every intermediate, and asserts:

  * the param-group split (decay vs no-decay) and `total_steps` from the
    token budget: 15 // (1 * 3) == 5,
  * `lr_multiplier`: the linear-warmup then cosine-to-`min_lr_ratio` curve,
    steps 0..5 -> [0.5, 1.0, 1.0, 0.775, 0.325, 0.1] EXACTLY, plus the WSD
    variant,
  * forward: logits (1,3,6) == 006's, softmax rows, per-position
    cross-entropy [3.41173, 0.99474, 3.31299], `lm_loss` 2.573154,
    `aux_loss` 0.011424, `total_loss` 2.584578,
  * backward: `dL/dlogits == (softmax - onehot) / N` by hand == autograd;
    `dL/d final_norm.weight` by hand (chain rule through the tied head +
    RMSNorm only) == autograd, to 1e-6,
  * gradient clipping: global 2-norm 3.222362 > 1.0 -> every grad scaled by
    1/3.222362 == 0.310331,
  * AdamW step 1 (t=1): bias-corrected m-hat/(sqrt(v-hat)+eps) == g/|g| ==
    sign(g) exactly, so no-decay `final_norm.weight` -> 1 - lr*sign(g);
    decay-group `w_q` additionally gets the -lr*wd*theta decoupled term,
  * steps 2..5: the clipped per-step gradient reconstructed from the real
    optimizer state's (exp_avg, exp_avg_sq) deltas reproduces every weight
    via the AdamW recurrence,
  * gradient accumulation: G=2 over the SAME window twice is byte-identical
    to G=1; G=2 over two DISTINCT windows reproduces a true batch-of-2 loss;
    `_sum_expert_counts` doubles the logged per-expert counts,
  * z-loss: `z_loss_weight * mean(logsumexp(logits)^2)`, logsumexp per pos
    == cross-entropy + logit_target,
  * `_evaluate`: mean val loss -> perplexity `exp(loss)`; the untrained
    special case `loss == ln(V)` -> `PP == V` exactly (ln 6 -> 6),
  * checkpoint/resume: 3 steps + resume 2 == an uninterrupted 5-step run,
    bit-for-bit,
  * the two "paths agree" checks report their measured max difference.

No fabricated numbers -- this script is the source of truth the prose
quotes.
"""
from __future__ import annotations

import json
import logging
import math
import pathlib
import tempfile

import torch
import torch.nn.functional as F

from qwenmoe.config import ModelConfig, TrainConfig
from qwenmoe.data.dataset import TokenDataset
from qwenmoe.model.transformer import QwenMoEModel
from qwenmoe.train.optim import build_optimizer, split_param_groups
from qwenmoe.train.trainer import (
    Trainer,
    _average_metrics,
    _forward_backward_core,
    _run_step,
    _sum_expert_counts,
    lr_multiplier,
    select_device,
)

torch.set_printoptions(precision=6, sci_mode=False)

# ======================================================================
# Part 1 -- the running example: the 006 model, the 006 weights, one window
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


# The tied embedding / LM-head matrix E, shape (vocab_size, d_model) = (6, 4).
E = torch.tensor(
    [
        [1.0, 1.0, 0.0, 0.0],    # token 0
        [1.0, 0.0, -1.0, 0.0],   # token 1
        [0.0, 0.0, 1.0, 1.0],    # token 2
        [-1.0, 0.0, 0.0, 1.0],   # token 3
        [0.0, 1.0, 0.0, 1.0],    # token 4
        [1.0, -1.0, 1.0, -1.0],  # token 5
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
        model.embedding.weight.copy_(E)  # lm_head.weight IS this tensor
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


STREAM = [4, 1, 4, 1]
SEQ_LEN = 3
train_ds = TokenDataset(STREAM, seq_len=SEQ_LEN)
val_ds = TokenDataset(STREAM, seq_len=SEQ_LEN)
X, Y = train_ds[0]
X = X.unsqueeze(0)
Y = Y.unsqueeze(0)
print(f"  token stream               : {STREAM}")
print(f"  len(train_ds)              : {len(train_ds)}  (= {len(STREAM)} - {SEQ_LEN})")
print(f"  the one window  x, y       : {X.tolist()}  {Y.tolist()}")
assert len(train_ds) == 1
assert X.tolist() == [[4, 1, 4]] and Y.tolist() == [[1, 4, 1]]

CONFIG = TrainConfig(
    batch_size=1,
    seq_len=SEQ_LEN,
    total_tokens=15,
    lr=0.1,
    warmup_steps=2,
    checkpoint_interval_steps=1,
    eval_interval_steps=1,
    log_interval_steps=1,
    max_eval_batches=10,
    device="cpu",
    seed=1337,
    betas=(0.9, 0.95),
    adam_eps=1e-8,
    weight_decay=0.1,
    grad_clip_norm=1.0,
    lr_schedule="cosine",
    min_lr_ratio=0.1,
    precision="fp32",
    z_loss_weight=0.0,
    grad_accum_steps=1,
)

print("\n================ Part 2 -- Trainer.fit setup ================")
model = build_model()
decay, no_decay = split_param_groups(model)
decay_names = {n for n, p in model.named_parameters() if any(p is d for d in decay)}
no_decay_names = {n for n, p in model.named_parameters() if any(p is d for d in no_decay)}
print(f"\n  decay group   ({len(decay)} tensors) : attention/expert/router weight matrices")
print(f"  no-decay group ({len(no_decay)} tensors): {sorted(no_decay_names)}")
assert len(decay) == 22 and len(no_decay) == 6
assert "final_norm.weight" in no_decay_names
assert "blocks.0.attn.w_q.weight" in decay_names

EFF_BATCH = CONFIG.batch_size * CONFIG.grad_accum_steps
TOTAL_STEPS = max(1, CONFIG.total_tokens // (EFF_BATCH * CONFIG.seq_len))
print(f"\n  total_steps = total_tokens // (batch*grad_accum*seq_len)")
print(f"             = {CONFIG.total_tokens} // ({CONFIG.batch_size}*{CONFIG.grad_accum_steps}*{CONFIG.seq_len}) = {TOTAL_STEPS}")
assert TOTAL_STEPS == 5

print(f"\n  select_device('cpu')  -> {select_device('cpu')!r}   (always forced)")
print(f"  select_device('auto') -> {select_device('auto')!r}   (mps if available, else cpu)")
assert select_device("cpu") == "cpu"
assert select_device("auto") in ("mps", "cpu")


# ======================================================================
# Part 3 -- the learning-rate schedule
# ======================================================================
print("\n================ Part 3 -- lr_multiplier ================")
mults = [lr_multiplier(s, 2, 5, schedule="cosine", min_lr_ratio=0.1) for s in range(6)]
for s, m in enumerate(mults):
    phase = "warmup" if s < 2 else "cosine"
    print(f"  step {s}: multiplier = {m:.6f}   lr = {0.1 * m:.6f}   ({phase})")
EXPECTED_MULTS = [0.5, 1.0, 1.0, 0.775, 0.325, 0.1]
for got, want in zip(mults, EXPECTED_MULTS):
    assert abs(got - want) < 1e-12, (got, want)
# continuity at the warmup->cosine seam: warmup's value at its last step and
# cosine's value at its first step are both 1.0
assert lr_multiplier(1, 2, 5, min_lr_ratio=0.1) == lr_multiplier(2, 2, 5, min_lr_ratio=0.1) == 1.0
# cos(pi/3) = 0.5 and cos(2pi/3) = -0.5 exactly -> the clean 0.775 / 0.325
assert abs(0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi / 3)) - 0.775) < 1e-12
assert abs(0.1 + 0.9 * 0.5 * (1 + math.cos(2 * math.pi / 3)) - 0.325) < 1e-12

print("\n  WSD variant (warmup=2, total=10, decay_fraction=0.3, min_lr_ratio=0.1):")
wsd = [lr_multiplier(s, 2, 10, schedule="wsd", min_lr_ratio=0.1, decay_fraction=0.3) for s in range(11)]
for s, m in enumerate(wsd):
    print(f"    step {s}: {m:.4f}")
assert wsd[2:8] == [1.0] * 6          # stable plateau
assert abs(wsd[8] - 0.7) < 1e-12 and abs(wsd[9] - 0.4) < 1e-12 and abs(wsd[10] - 0.1) < 1e-12


# ======================================================================
# Part 4 -- the forward half of one step
# ======================================================================
print("\n================ Part 4 -- forward: logits, loss ================")
model = build_model()
model.train()

captured: dict = {}
handle = model.final_norm.register_forward_hook(
    lambda mod, inp, out: captured.__setitem__("u", inp[0].detach().clone())
)
logits, extras = model(X, targets=Y)
handle.remove()

U = captured["u"].squeeze(0)               # pre-final-norm residual stream (3, 4)
print("  pre-final-norm residual  u (3,4):\n", U)
print("\n  logits (1,3,6):\n", logits.squeeze(0))

flat_logits = logits.reshape(-1, 6)
flat_y = Y.reshape(-1)
N = flat_y.numel()
sm = F.softmax(flat_logits, dim=-1)
print("\n  softmax(logits) rows:\n", sm)

per_pos = F.cross_entropy(flat_logits, flat_y, reduction="none")
lm_loss = F.cross_entropy(flat_logits, flat_y)
aux_loss = extras["aux_loss"]
total_loss = lm_loss + aux_loss
logsumexp = torch.logsumexp(flat_logits, dim=-1)
print(f"\n  per-position cross-entropy : {[round(v, 6) for v in per_pos.tolist()]}")
print(f"  logsumexp per position     : {[round(v, 6) for v in logsumexp.tolist()]}")
print(f"  lm_loss  (mean)            : {lm_loss.item():.6f}")
print(f"  aux_loss (mean of 2 blocks): {aux_loss.item():.6f}")
print(f"  total_loss = lm + aux      : {total_loss.item():.6f}")

# cross-entropy at a position == logsumexp - logit_target
for p in range(N):
    assert abs(per_pos[p].item() - (logsumexp[p].item() - flat_logits[p, flat_y[p]].item())) < 1e-5
torch.testing.assert_close(
    per_pos, torch.tensor([3.411731, 0.994745, 3.312987]), atol=1e-5, rtol=0
)
assert abs(lm_loss.item() - 2.573154) < 1e-5
assert abs(aux_loss.item() - 0.011424) < 1e-5
assert abs(total_loss.item() - 2.584578) < 1e-5

# logits must equal the 006 companion's hand-derived logits exactly
EXPECTED_LOGITS = torch.tensor(
    [
        [1.414213, 0.000000, 1.414213, 1.414213, 2.828427, -2.828427],
        [-0.039239, 0.622172, -0.530086, 1.682475, 1.643236, -3.855798],
        [1.185529, 0.009417, 0.999658, 1.577581, 2.763110, -3.341033],
    ]
)
torch.testing.assert_close(logits.squeeze(0), EXPECTED_LOGITS, atol=2e-5, rtol=0)


# ======================================================================
# Part 5 -- the backward half, by hand where the chain is short
# ======================================================================
print("\n================ Part 5 -- backward + gradient clipping ================")

# (a) dL/dlogits = (softmax - onehot) / N, derived from d(-log softmax_y)/dz.
onehot = F.one_hot(flat_y, 6).float()
dlogits_hand = (sm - onehot) / N
model.zero_grad(set_to_none=True)
lg, ex = model(X, targets=Y)
lg.retain_grad()
(F.cross_entropy(lg.reshape(-1, 6), flat_y) + ex["aux_loss"]).backward()
dlogits_auto = lg.grad.reshape(-1, 6)
print("  dL/dlogits by hand (3,6):\n", dlogits_hand)
print("  max |hand - autograd|:", (dlogits_hand - dlogits_auto).abs().max().item())
torch.testing.assert_close(dlogits_hand, dlogits_auto, atol=1e-6, rtol=0)

# (b) dL/d final_norm.weight -- chain rule through the tied head + RMSNorm ONLY.
#     out_{p,i} = normed_{p,i} * g_i,   normed_{p,i} = u_{p,i} * inv_rms_p,
#     logit_{p,j} = sum_i out_{p,i} * E_{j,i}
var = U.pow(2).mean(dim=-1, keepdim=True)
inv_rms = torch.rsqrt(var + 1e-6)
normed = U * inv_rms
d_out = dlogits_hand @ E                     # (3,4)  dL/d out_p
dg_hand = (d_out * normed).sum(dim=0)        # (4,)   sum over positions
model.zero_grad(set_to_none=True)
lg, ex = model(X, targets=Y)
(F.cross_entropy(lg.reshape(-1, 6), flat_y) + ex["aux_loss"]).backward()
dg_auto = model.final_norm.weight.grad
print(f"\n  inv_rms per position       : {[round(v, 6) for v in inv_rms.flatten().tolist()]}")
print("  u_tilde = u * inv_rms (3,4):\n", normed)
print("  dL/d o_p = dL/dlogits @ E (3,4):\n", d_out)
print(f"  dL/d final_norm.weight hand : {[round(v, 6) for v in dg_hand.tolist()]}")
print(f"  dL/d final_norm.weight auto : {[round(v, 6) for v in dg_auto.tolist()]}")
print(f"  max |hand - autograd|      : {(dg_hand - dg_auto).abs().max().item():.2e}")
torch.testing.assert_close(dg_hand, dg_auto, atol=1e-6, rtol=0)

# (c) global gradient 2-norm and the clip coefficient.
grad_sq = sum(p.grad.pow(2).sum() for p in model.parameters() if p.grad is not None)
global_norm = grad_sq.sqrt()
clip_coef = min(1.0, CONFIG.grad_clip_norm / global_norm.item())
print(f"\n  global grad 2-norm         : {global_norm.item():.6f}")
print(f"  grad_clip_norm             : {CONFIG.grad_clip_norm}")
print(f"  clip coefficient = 1/norm  : {clip_coef:.6f}   (norm > 1 -> clips)")
assert abs(global_norm.item() - 3.222362) < 1e-4
assert abs(clip_coef - 0.310331) < 1e-5
g_clipped = model.final_norm.weight.grad * clip_coef
print(f"  final_norm.weight.grad clipped: {[round(v, 6) for v in g_clipped.tolist()]}")
wq00_raw = model.blocks[0].attn.w_q.weight.grad[0, 0].item()
print(f"  w_q[0,0].grad raw / clipped : {wq00_raw:.6f} / {wq00_raw * clip_coef:.6f}   (stays > 0)")
assert wq00_raw > 0 and wq00_raw * clip_coef > 0
assert abs(wq00_raw - 0.120422) < 1e-5


# ======================================================================
# Part 6 -- AdamW step 1 (t = 1), by hand
# ======================================================================
print("\n================ Part 6 -- AdamW step 1 ================")
model = build_model()
opt = build_optimizer(model, CONFIG)
wq = model.blocks[0].attn.w_q.weight
metrics, gn, cur_lr, dev, model, ec = _run_step(model, opt, [(X, Y, None)], "cpu", 0, TOTAL_STEPS, CONFIG)

beta1, beta2 = CONFIG.betas
eps = CONFIG.adam_eps
lr0 = 0.1 * 0.5                              # step-0 multiplier
st_fn = opt.state[model.final_norm.weight]
m1 = st_fn["exp_avg"]
v1 = st_fn["exp_avg_sq"]
# reconstruct the clipped gradient the optimizer actually saw: m1 = (1-b1) * g'
g_prime = m1 / (1 - beta1)
mhat = m1 / (1 - beta1 ** 1)
vhat = v1 / (1 - beta2 ** 1)
update = mhat / (vhat.sqrt() + eps)
fn_after_hand = 1.0 - lr0 * update           # no-decay group: no theta term
print(f"  lr this step               : {cur_lr:.6f}  (= 0.1 * 0.5)")
print(f"  clipped g' (final_norm.w)  : {[round(x, 6) for x in g_prime.tolist()]}")
print(f"  m_hat / (sqrt(v_hat)+eps)  : {[round(x, 8) for x in update.tolist()]}   (== sign(g'))")
print(f"  final_norm.weight by hand  : {[round(x, 6) for x in fn_after_hand.tolist()]}")
print(f"  final_norm.weight real     : {[round(x, 6) for x in model.final_norm.weight.detach().tolist()]}")
torch.testing.assert_close(update, torch.sign(g_prime), atol=1e-6, rtol=0)
torch.testing.assert_close(fn_after_hand, model.final_norm.weight.detach(), atol=1e-6, rtol=0)

# decay-group weight w_q[0,0] = 1.0: extra decoupled term theta <- theta*(1 - lr*wd)
wd = CONFIG.weight_decay
g_wq00 = opt.state[wq]["exp_avg"][0, 0] / (1 - beta1)
wq00_hand = 1.0 * (1 - lr0 * wd) - lr0 * torch.sign(g_wq00)
print(f"\n  w_q[0,0] by hand           : {wq00_hand.item():.6f}  = 1*(1-{lr0}*{wd}) - {lr0}*sign(g)")
print(f"  w_q[0,0] real              : {wq[0, 0].item():.6f}")
assert abs(wq00_hand.item() - wq[0, 0].item()) < 1e-6
# w_q[0,1] = 0.0: the decay term does nothing to a zero weight
wq01_hand = 0.0 * (1 - lr0 * wd) - lr0 * torch.sign(opt.state[wq]["exp_avg"][0, 1])
print(f"  w_q[0,1] by hand           : {wq01_hand.item():.6f}  (0*(...) - lr*sign(g))")
print(f"  w_q[0,1] real              : {wq[0, 1].item():.6f}")
assert abs(wq01_hand.item() - wq[0, 1].item()) < 1e-6


# ======================================================================
# Part 7 -- steps 2..5: the AdamW recurrence reproduces every weight
# ======================================================================
print("\n================ Part 7 -- steps 2..5 (full 5-step run) ================")
model = build_model()
opt = build_optimizer(model, CONFIG)
m_prev = torch.zeros(4)
v_prev = torch.zeros(4)
theta = torch.ones(4)
dev = "cpu"
for step in range(TOTAL_STEPS):
    metrics, gn, cur_lr, dev, model, ec = _run_step(model, opt, [(X, Y, None)], dev, step, TOTAL_STEPS, CONFIG)
    st = opt.state[model.final_norm.weight]
    t = step + 1
    m_now, v_now = st["exp_avg"], st["exp_avg_sq"]
    # clipped gradient this step, from the EMA deltas
    g_t = (m_now - beta1 * m_prev) / (1 - beta1)
    mhat = m_now / (1 - beta1 ** t)
    vhat = v_now / (1 - beta2 ** t)
    theta = theta - cur_lr * (mhat / (vhat.sqrt() + eps))   # no-decay group
    clipped = "clip" if gn > CONFIG.grad_clip_norm else "no-clip"
    print(f"  step {t}: lr={cur_lr:.5f}  grad_norm(pre-clip)={gn:.5f} [{clipped}]  "
          f"loss={metrics['lm_loss']:.5f}")
    print(f"          component 0: g'={g_t[0].item():.6f}  m={m_now[0].item():.6f}  v={v_now[0].item():.7f}")
    print(f"          reconstructed final_norm.weight = {[round(x, 6) for x in theta.tolist()]}")
    print(f"          real          final_norm.weight = {[round(x, 6) for x in model.final_norm.weight.detach().tolist()]}")
    torch.testing.assert_close(theta, model.final_norm.weight.detach(), atol=1e-5, rtol=0)
    m_prev, v_prev = m_now.clone(), v_now.clone()

print(f"\n  final train_loss = {metrics['lm_loss']:.6f}")


# ======================================================================
# Part 8 -- gradient accumulation
# ======================================================================
print("\n================ Part 8 -- gradient accumulation ================")
# (a) G=2 over the SAME window twice == G=1, byte for byte.
cfg_g2 = TrainConfig(**{**CONFIG.__dict__, "grad_accum_steps": 2, "total_tokens": 30})
assert cfg_g2.total_tokens // (cfg_g2.batch_size * cfg_g2.grad_accum_steps * cfg_g2.seq_len) == 5
mA = build_model()
optA = build_optimizer(mA, cfg_g2)
mB = build_model()
optB = build_optimizer(mB, CONFIG)
for step in range(5):
    rA = _run_step(mA, optA, [(X, Y, None), (X, Y, None)], "cpu", step, 5, cfg_g2)
    rB = _run_step(mB, optB, [(X, Y, None)], "cpu", step, 5, CONFIG)
diff = (mA.final_norm.weight - mB.final_norm.weight).abs().max().item()
print(f"  G=2 (same window x2) vs G=1 : max |Δ final_norm.weight| = {diff:.2e}")
print(f"  G=2 summed expert counts    : {rA[5]}   (G=1: {rB[5]})")
assert diff == 0.0
assert rA[5] == [[c * 2 for c in layer] for layer in rB[5]]

# (b) G=2 over two DISTINCT windows reproduces a true batch-of-2 loss.
stream2 = [4, 1, 4, 1, 4]
d2 = TokenDataset(stream2, seq_len=3)
w0, w1 = d2[0], d2[1]
mC = build_model()
mC.train()
mC.zero_grad(set_to_none=True)
micro = [
    _forward_backward_core(mC, w0[0].unsqueeze(0), w0[1].unsqueeze(0), "cpu", "fp32", 0.0, None, 0.5),
    _forward_backward_core(mC, w1[0].unsqueeze(0), w1[1].unsqueeze(0), "cpu", "fp32", 0.0, None, 0.5),
]
avg = _average_metrics(micro)
mD = build_model()
mD.train()
xb = torch.stack([w0[0], w1[0]])
yb = torch.stack([w0[1], w1[1]])
lg, _ = mD(xb, targets=yb)
true_batch2 = F.cross_entropy(lg.reshape(-1, 6), yb.reshape(-1)).item()
print(f"  micro-batch lm_losses       : {[round(x['lm_loss'], 6) for x in micro]}")
print(f"  _average_metrics -> lm_loss : {avg['lm_loss']:.6f}")
print(f"  true batch-of-2 lm_loss     : {true_batch2:.6f}")
print(f"  max |avg - true batch-2|    : {abs(avg['lm_loss'] - true_batch2):.2e}")
assert abs(avg["lm_loss"] - true_batch2) < 1e-5
assert _sum_expert_counts([[[1, 2]], [[3, 0]]]) == [[4, 2]]


# ======================================================================
# Part 9 -- the output z-loss
# ======================================================================
print("\n================ Part 9 -- z-loss ================")
model = build_model()
model.train()
lg, ex = model(X, targets=Y)
fl = lg.reshape(-1, 6)
lse = torch.logsumexp(fl, dim=-1)
mean_sq = lse.pow(2).mean()
for w in (1e-4, 0.1):
    z_term = w * mean_sq
    print(f"  z_loss_weight={w:<6}: {w} * mean(logsumexp^2) = {w} * {mean_sq.item():.6f} = {z_term.item():.8f}")
assert abs(mean_sq.item() - 9.879073) < 1e-4
# logsumexp per position == cross-entropy + logit_target (same identity as Part 4)
ce = F.cross_entropy(fl, flat_y, reduction="none")
for p in range(N):
    assert abs(lse[p].item() - (ce[p].item() + fl[p, flat_y[p]].item())) < 1e-5


# ======================================================================
# Part 10 -- validation loss -> perplexity
# ======================================================================
print("\n================ Part 10 -- _evaluate + perplexity ================")
V = _cfg().vocab_size
model = build_model()
model.eval()
with torch.no_grad():
    lg, _ = model(X)
    val_loss = F.cross_entropy(lg.reshape(-1, 6), flat_y).item()
print(f"  fresh (pinned) model val_loss : {val_loss:.6f}   perplexity exp(loss) = {math.exp(val_loss):.6f}")
print(f"  untrained-uniform special case: loss = ln(V) = ln 6 = {math.log(6):.6f}")
print(f"                                  PP  = exp(ln 6) = {math.exp(math.log(6)):.6f}  == V = {V}")
assert abs(math.exp(math.log(6)) - 6.0) < 1e-9

# real Trainer.fit: val loss / perplexity trajectory
out = pathlib.Path(tempfile.mkdtemp())
logging.disable(logging.CRITICAL)
res = Trainer(out).fit(build_model(), train_ds, val_ds, CONFIG)
logging.disable(logging.NOTSET)
print(f"\n  Trainer.fit -> {res}")
assert res.final_step == 5
assert abs(math.exp(res.final_val_loss) - res.val_perplexity) < 1e-9
lines = (out / "logs" / "train.jsonl").read_text().splitlines()
step_lines = [line for line in lines if '"type": "step"' in line]
eval_lines = [line for line in lines if '"type": "eval"' in line]
print(f"  train.jsonl: {len(step_lines)} 'step' records, {len(eval_lines)} 'eval' records")
assert len(step_lines) == 5 and len(eval_lines) == 5
assert '"expert_token_counts"' in step_lines[0] and '"grad_norm"' in step_lines[0]


# ======================================================================
# Part 11 -- checkpoint / resume
# ======================================================================
print("\n================ Part 11 -- checkpoint / resume ================")
logging.disable(logging.CRITICAL)
cfg_a = TrainConfig(**{**CONFIG.__dict__, "total_tokens": 9, "checkpoint_interval_steps": 3, "eval_interval_steps": 3})
oa = pathlib.Path(tempfile.mkdtemp())
Trainer(oa).fit(build_model(), train_ds, val_ds, cfg_a)               # 3 steps
cfg_b = TrainConfig(**{**CONFIG.__dict__, "total_tokens": 15, "checkpoint_interval_steps": 3, "eval_interval_steps": 3})
ob = pathlib.Path(tempfile.mkdtemp())
mb = build_model()
Trainer(ob).fit(mb, train_ds, val_ds, cfg_b, resume_from=oa / "checkpoints" / "step_3.pt")   # +2 steps
mc = build_model()
oc = pathlib.Path(tempfile.mkdtemp())
Trainer(oc).fit(mc, train_ds, val_ds, cfg_b)                          # uninterrupted 5 steps
logging.disable(logging.NOTSET)
resume_diff = (mb.final_norm.weight - mc.final_norm.weight).abs().max().item()
ckpt = torch.load(oa / "checkpoints" / "step_3.pt", weights_only=False)
print(f"  checkpoint keys            : {sorted(ckpt.keys())}")
print(f"  saved step                 : {ckpt['step']}")
print(f"  resume(3+2) vs uninterrupted(5): max |Δ final_norm.weight| = {resume_diff:.2e}")
assert ckpt["step"] == 3
assert set(ckpt.keys()) == {
    "model_state_dict", "optimizer_state_dict", "model_config", "train_config", "step", "tokenizer_meta",
}
assert resume_diff < 1e-6


# ======================================================================
# Part 12 -- the JSON-lines log records
# ======================================================================
print("\n================ Part 12 -- structured logging ================")
print("  first 'step' record:\n   ", step_lines[0])
print("  first 'eval' record:\n   ", eval_lines[0])
rec0 = json.loads(step_lines[0])
assert rec0["type"] == "step" and rec0["step"] == 1
assert rec0["lr"] == 0.05 and rec0["z_loss"] is None
assert rec0["expert_token_counts"] == [[3, 0], [3, 0]]   # all 3 tokens -> expert 0, both layers
assert abs(rec0["grad_norm"] - 3.222362) < 1e-4
ev0 = json.loads(eval_lines[0])
assert ev0["type"] == "eval" and abs(math.exp(ev0["val_loss"]) - ev0["val_perplexity"]) < 1e-9


print("\nAll assertions passed -- every number in 010-training-loop.md is real.")
