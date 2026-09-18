"""Reproduces every hand-computed number in
`038-gradient-accumulation-fair-rematch.md`.

Run it:  uv run python docs/math/verify_038_gradient_accumulation_fair_rematch.py

ONE running example, carried end to end: the exact tiny 2-layer
`QwenMoEModel` from `006-full-model-assembly.md` / `010-training-loop.md`
(vocab_size=6, d_model=4, n_layers=2, n_heads=2, n_kv_heads=1, d_ff=3,
n_routed_experts=2, top_k=1, n_shared_experts=0, SwiGLU, NoPE, tied
embeddings), with the SAME hand-picked weights pinned in, but now pushed
through TWO DISTINCT 3-token windows -- window A = ([4,1,4],[1,4,1]) and
window B = ([1,4,1],[4,1,4]) (the token stream [4,1,4,1,4], exactly
010's Part-8.3 stream) -- run as two grad-accumulation micro-batches.

010's own Part 8 already proved gradient accumulation's LOSS_SCALE
mechanism byte-for-bit on the *cross-entropy* term alone. This script
proves (and the doc derives) the piece 010 deliberately left out: whether
the *auxiliary* MoE load-balancing loss survives being split across
micro-batches the same way -- and shows, with real numbers, that it does
NOT in general, why not, and exactly what journal 038's `aux_loss_weight
=0.0` was actually buying the "trickiest test" (`test_run_step_grad_accum_
matches_a_single_bigger_batch`).

It drives the real module code, prints every intermediate, and asserts:

  * cross-entropy's batch-mean is EXACTLY linear across equal-size splits
    (provable; matches 010 Part 8.3, restated here as the baseline case),
  * the Switch/aux-loss (`AuxLossBalancer.aux_loss`) is a BILINEAR function
    of two batch-aggregate statistics (f_i, P_i), so splitting a batch in
    two and averaging the two aux_losses does NOT reproduce the aux_loss
    of the concatenated batch -- the exact gap is derived symbolically as
    a covariance-like cross term and then matched to floating point on
    real window-A / window-B router traces,
  * consequently a real `_run_step` accumulation over windows A+B, with
    `aux_loss_weight=0.01` (the project default), produces gradients that
    measurably differ from a true concatenated batch-of-2 -- and with
    `aux_loss_weight=0.0` the two are byte-identical (to float32 rounding),
  * `LossFreeBalancer`'s bias buffer is a SECOND, independent obstacle:
    it mutates after every training-mode forward call, so two sequential
    micro-batch forwards see a bias update the concatenated batch's single
    forward never does,
  * `_sum_expert_counts` on two genuinely different windows (not the same
    window doubled, as in 010 Part 8.2) matches hand addition,
  * the real repository test's own shapes (batch of 8 vs. two micro-batches
    of 4) reproduce `test_run_step_grad_accum_matches_a_single_bigger_
    batch`'s pass at `aux_loss_weight=0.0` and its (otherwise silent)
    failure at `aux_loss_weight=0.01`,
  * the token-counting arithmetic behind journal 038's "fair rematch"
    table, read off the real presets/config fields,
  * the Chinchilla implied-budget figure (~2.4B tokens) from journal 038's
    "Questions" section, computed by the real `count_params` +
    `chinchilla_budget` on the real `SCALE_300M` preset.

No fabricated numbers -- this script is the source of truth the prose
quotes.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from qwenmoe.config import ModelConfig, TrainConfig
from qwenmoe.model.moe import AuxLossBalancer
from qwenmoe.model.transformer import QwenMoEModel
from qwenmoe.train.optim import build_optimizer
from qwenmoe.train.scaling import chinchilla_budget, count_params
from qwenmoe.train.trainer import (
    _forward_backward_core,
    _forward_backward_step,
    _run_step,
    _sum_expert_counts,
)

torch.set_printoptions(precision=6, sci_mode=False)

# ======================================================================
# Part 1 -- the running example: 010's exact model + weights, two windows
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
        model.embedding.weight.copy_(E)
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


def build_model(cfg=None) -> QwenMoEModel:
    torch.manual_seed(0)
    return _pin(QwenMoEModel(cfg or _cfg()))


# Window A: 010's canonical window -- ALL 3 tokens route to expert 0
# (f^A = [1, 0], per 006 Part "Steps 6-9"). 010's own second window, from
# the SAME 5-token stream [4,1,4,1,4], ALSO routes 100% to expert 0 (both
# windows draw only on tokens {1,4}, which this router always sends to
# expert 0) -- degenerate for what this doc needs to show, since with
# f^A == f^B the bilinear gap this section derives is identically zero
# no matter what P^A, P^B are. So window B here is deliberately NOT 010's
# stream2 window: `x=[0,2,0]` is the smallest hand-picked window (still
# built from real vocabulary ids, same model, same weights) whose routing
# is genuinely different -- f^B = [2/3, 1/3] -- which is exactly the
# "every behaviour" the running example needs to exercise.
XA = torch.tensor([[4, 1, 4]])
YA = torch.tensor([[1, 4, 1]])
XB = torch.tensor([[0, 2, 0]])
YB = torch.tensor([[2, 0, 1]])
print(f"  window A: x={XA.tolist()} y={YA.tolist()}   (010's canonical window)")
print(f"  window B: x={XB.tolist()} y={YB.tolist()}   (hand-picked: routes non-degenerately)")

# ======================================================================
# Part 2 -- token-counting arithmetic behind the "fair rematch" table
# ======================================================================
print("\n================ Part 2 -- tokens/step arithmetic ================")
from configs.presets import SCALE_100M, SCALE_300M  # noqa: E402

# journal 038's own table: SCALE_100M ran batch=16, SCALE_300M (037,
# original) batch=4, both seq_len=256 (each preset's own max_seq_len).
assert SCALE_100M.max_seq_len == 256 and SCALE_300M.max_seq_len == 256
tokens_per_step_100m = 16 * SCALE_100M.max_seq_len
tokens_per_step_300m_b4 = 4 * SCALE_300M.max_seq_len
tokens_per_step_300m_accum = 4 * 4 * SCALE_300M.max_seq_len  # batch=4, grad_accum=4
print(f"  SCALE_100M  batch=16 : tokens/step = 16 * {SCALE_100M.max_seq_len} = {tokens_per_step_100m}")
print(f"  SCALE_300M  batch=4  : tokens/step = 4 * {SCALE_300M.max_seq_len} = {tokens_per_step_300m_b4}")
print(f"  SCALE_300M  batch=4,accum=4 : tokens/step = 4*4*{SCALE_300M.max_seq_len} = {tokens_per_step_300m_accum}")
assert tokens_per_step_100m == 4096
assert tokens_per_step_300m_b4 == 1024
assert tokens_per_step_300m_accum == 4096

tokens_100m_at_300 = tokens_per_step_100m * 300
tokens_300m_b4_at_500 = tokens_per_step_300m_b4 * 500
tokens_300m_accum_at_300 = tokens_per_step_300m_accum * 300
print(f"  SCALE_100M @step 300              : {tokens_per_step_100m} * 300 = {tokens_100m_at_300:,}")
print(f"  SCALE_300M(037) @step 500          : {tokens_per_step_300m_b4} * 500 = {tokens_300m_b4_at_500:,}")
print(f"  SCALE_300M(038,accum) @step 300    : {tokens_per_step_300m_accum} * 300 = {tokens_300m_accum_at_300:,}")
assert tokens_100m_at_300 == 1_228_800
assert tokens_300m_b4_at_500 == 512_000
assert tokens_300m_accum_at_300 == 1_228_800
assert tokens_100m_at_300 == tokens_300m_accum_at_300  # the "fair" match journal 038 built

# effective_batch_size / total_steps formula (trainer.py:581-582), confirmed
# generically on a tiny config -- reused from 010 Part 2.4, restated for a
# grad_accum-driven config matching the rematch's own batch/accum numbers.
cfg_rematch_like = TrainConfig(
    batch_size=4, seq_len=256, grad_accum_steps=4, total_tokens=1_228_800, device="cpu",
)
effective_batch_size = cfg_rematch_like.batch_size * cfg_rematch_like.grad_accum_steps
total_steps = max(
    1, cfg_rematch_like.total_tokens // (effective_batch_size * cfg_rematch_like.seq_len)
)
print(f"  effective_batch_size = {cfg_rematch_like.batch_size} * {cfg_rematch_like.grad_accum_steps} = {effective_batch_size}")
print(f"  total_steps = {cfg_rematch_like.total_tokens} // ({effective_batch_size}*{cfg_rematch_like.seq_len}) = {total_steps}")
assert effective_batch_size == 16
assert total_steps == 300


# ======================================================================
# Part 3 -- cross-entropy's batch-mean IS exactly linear across equal splits
# ======================================================================
print("\n================ Part 3 -- cross-entropy decomposes ================")
mE = build_model()
mE.train()
lgA, _ = mE(XA, targets=YA)
lgB, _ = mE(XB, targets=YB)
ce_A = F.cross_entropy(lgA.reshape(-1, 6), YA.reshape(-1)).item()
ce_B = F.cross_entropy(lgB.reshape(-1, 6), YB.reshape(-1)).item()
xcat = torch.cat([XA, XB], dim=0)
ycat = torch.cat([YA, YB], dim=0)
lgcat, _ = mE(xcat, targets=ycat)
ce_cat = F.cross_entropy(lgcat.reshape(-1, 6), ycat.reshape(-1)).item()
avg_ce = (ce_A + ce_B) / 2
print(f"  lm_loss(A) = {ce_A:.6f}   lm_loss(B) = {ce_B:.6f}")
print(f"  mean(lm_loss(A), lm_loss(B)) = {avg_ce:.6f}")
print(f"  lm_loss(concat A+B)          = {ce_cat:.6f}")
print(f"  max |avg - concat|           = {abs(avg_ce - ce_cat):.2e}")
assert abs(avg_ce - ce_cat) < 1e-5


# ======================================================================
# Part 4 -- the aux_loss is BILINEAR: it does NOT decompose the same way
# ======================================================================
print("\n================ Part 4 -- the aux_loss does not decompose ================")


def block_router_trace(model: QwenMoEModel, x: torch.Tensor, block_idx: int):
    _, bundle = model.forward_with_trace(x)
    lt = bundle.layers[block_idx]
    return lt.router_logits, lt.router_probs, lt.topk_indices


def aux_loss_for(model: QwenMoEModel, x: torch.Tensor, block_idx: int):
    _, probs, topk = block_router_trace(model, x, block_idx)
    balancer: AuxLossBalancer = model.blocks[block_idx].moe.balancer
    return balancer.aux_loss(probs, topk).item(), probs, topk


mF = build_model()
mF.eval()  # eval mode: aux_loss balancer.update is a no-op regardless, but this keeps the trace calls side-effect-free either way
with torch.no_grad():
    aux_A0, probsA0, topkA0 = aux_loss_for(mF, XA, 0)
    aux_B0, probsB0, topkB0 = aux_loss_for(mF, XB, 0)

print(f"  block 0, window A: router_probs=\n{probsA0}\n  topk_indices={topkA0.flatten().tolist()}  aux_loss(A)={aux_A0:.8f}")
print(f"  block 0, window B: router_probs=\n{probsB0}\n  topk_indices={topkB0.flatten().tolist()}  aux_loss(B)={aux_B0:.8f}")
assert abs(aux_A0 - 0.01191779) < 1e-7  # matches 006's own published block-0/window-A number

# The concatenated batch's own aux_loss at block 0 -- computed the ONLY way
# the real forward pass ever computes it: router_probs/topk_indices over
# ALL 6 tokens in one call (forward_with_trace requires batch==1, so we use
# a plain forward + a hook on `score_and_select` to read block 0's tensors).
captured = {}
from qwenmoe.model import moe as moe_mod  # noqa: E402

real_score_and_select = moe_mod.score_and_select


def _capturing_score_and_select(router_logits, balancer, top_k, router_scoring):
    out = real_score_and_select(router_logits, balancer, top_k, router_scoring)
    captured.setdefault("calls", []).append((router_logits.detach().clone(), *[t.detach().clone() for t in out]))
    return out


moe_mod.score_and_select = _capturing_score_and_select
try:
    mF.eval()
    with torch.no_grad():
        mF(xcat)
finally:
    moe_mod.score_and_select = real_score_and_select

# first captured call inside this forward is block 0's router
_, topk_cat0, weights_cat0, probs_cat0 = captured["calls"][0]
balancer0: AuxLossBalancer = mF.blocks[0].moe.balancer
aux_cat0 = balancer0.aux_loss(probs_cat0, topk_cat0).item()
print(f"  block 0, concat(A,B): topk_indices={topk_cat0.flatten().tolist()}  aux_loss(concat)={aux_cat0:.8f}")

avg_aux0 = (aux_A0 + aux_B0) / 2
print(f"  mean(aux_loss(A), aux_loss(B)) = {avg_aux0:.8f}")
print(f"  aux_loss(concat A+B)           = {aux_cat0:.8f}")
print(f"  gap = concat - mean            = {aux_cat0 - avg_aux0:.8f}")

# --- the symbolic identity, checked numerically ---
# f_i^(w) = fraction of window w's N tokens routed to expert i (mean of a
# one-hot indicator over N tokens); P_i^(w) = mean routing prob of expert i
# over the same N tokens. For two EQUAL-size windows (N tokens each):
#   f_i^cat = (f_i^A + f_i^B) / 2   and   P_i^cat = (P_i^A + P_i^B) / 2
# exactly (means of equal-size groups average like that). But
#   aux_loss = alpha * n_experts * sum_i f_i * P_i
# is BILINEAR in (f, P), so
#   aux_loss(cat) - mean(aux_loss(A), aux_loss(B))
#     = -(alpha * n_experts / 4) * sum_i (f_i^A - f_i^B)(P_i^A - P_i^B)
#     =  (alpha * n_experts / 4) * sum_i (f_i^A - f_i^B)(P_i^B - P_i^A)
n_experts = 2
alpha = 0.01
N = 3


def f_and_P(topk_indices, router_probs):
    dispatched = F.one_hot(topk_indices, num_classes=n_experts).sum(dim=1)
    f = dispatched.float().mean(dim=0)
    P = router_probs.mean(dim=0)
    return f, P


fA, PA = f_and_P(topkA0, probsA0)
fB, PB = f_and_P(topkB0, probsB0)
predicted_gap = -(alpha * n_experts / 4) * torch.sum((fA - fB) * (PA - PB)).item()
print(f"  f^A={fA.tolist()}  P^A={PA.tolist()}")
print(f"  f^B={fB.tolist()}  P^B={PB.tolist()}")
print(f"  predicted gap = -(alpha*n/4) * sum[(f^A-f^B)(P^A-P^B)] = {predicted_gap:.8f}")
assert abs((aux_cat0 - avg_aux0) - predicted_gap) < 1e-6
assert abs(aux_cat0 - avg_aux0) > 5e-5  # a REAL, non-negligible gap on this example


# ======================================================================
# Part 5 -- consequence: a real _run_step over A,B diverges from a real
# concatenated batch-of-2 UNLESS aux_loss_weight == 0
# ======================================================================
print("\n================ Part 5 -- the gradient-level consequence ================")


def compare_accum_vs_true_batch(aux_loss_weight: float) -> float:
    cfg = _cfg(aux_loss_weight=aux_loss_weight)
    m_accum = build_model(cfg)
    m_true = build_model(cfg)
    opt_accum = build_optimizer(m_accum, TrainConfig(batch_size=1, seq_len=3, grad_accum_steps=2, total_tokens=6, device="cpu", warmup_steps=0, grad_clip_norm=1e6))
    opt_true = build_optimizer(m_true, TrainConfig(batch_size=2, seq_len=3, grad_accum_steps=1, total_tokens=6, device="cpu", warmup_steps=0, grad_clip_norm=1e6))

    micro_batches = [(XA, YA, None), (XB, YB, None)]
    _run_step(m_accum, opt_accum, micro_batches, "cpu", 0, 1,
              TrainConfig(batch_size=1, seq_len=3, grad_accum_steps=2, total_tokens=6, device="cpu", warmup_steps=0, grad_clip_norm=1e6))

    _forward_backward_step(m_true, opt_true, xcat, ycat, grad_clip_norm=1e6, device="cpu")

    max_diff = 0.0
    for p_accum, p_true in zip(m_accum.parameters(), m_true.parameters()):
        max_diff = max(max_diff, (p_accum - p_true).abs().max().item())
    return max_diff


diff_default = compare_accum_vs_true_batch(0.01)
diff_zeroed = compare_accum_vs_true_batch(0.0)
print(f"  aux_loss_weight=0.01 (project default): max |accum - true batch| over ALL params = {diff_default:.6e}")
print(f"  aux_loss_weight=0.0  (the fix)         : max |accum - true batch| over ALL params = {diff_zeroed:.6e}")
assert diff_default > 1e-4     # a real, structural mismatch -- not float noise
assert diff_zeroed < 1e-5      # byte-identical modulo float32 summation order


# ======================================================================
# Part 6 -- LossFreeBalancer: a SECOND, independent obstacle (state, not loss shape)
# ======================================================================
print("\n================ Part 6 -- LossFreeBalancer's bias mutation ================")
cfg_lf = _cfg(moe_balancing="loss_free")
m_lf_accum = build_model(cfg_lf)
m_lf_true = build_model(cfg_lf)

bias_before = m_lf_accum.blocks[0].moe.balancer.bias.clone()
m_lf_accum.train()
m_lf_accum(XA, targets=YA)  # micro-batch 1 of the accumulation window
bias_after_A = m_lf_accum.blocks[0].moe.balancer.bias.clone()
m_lf_accum(XB, targets=YB)  # micro-batch 2 -- sees the bias update micro-batch 1 caused
bias_after_B = m_lf_accum.blocks[0].moe.balancer.bias.clone()

m_lf_true.train()
m_lf_true(xcat, targets=ycat)  # the true concatenated batch: ONE forward, ONE update() call
bias_after_true = m_lf_true.blocks[0].moe.balancer.bias.clone()

print(f"  bias before any forward         : {bias_before.tolist()}")
print(f"  accum path, after micro-batch A : {bias_after_A.tolist()}  (mutated -- update_rate * sign(violation))")
print(f"  accum path, after micro-batch B : {bias_after_B.tolist()}  (mutated AGAIN, from the A-updated bias)")
print(f"  true-batch path, after 1 forward: {bias_after_true.tolist()}  (mutated ONCE, from aggregate counts over all 6 tokens)")
assert not torch.equal(bias_after_A, bias_before)
assert not torch.equal(bias_after_B, bias_after_true)


# ======================================================================
# Part 7 -- _sum_expert_counts on two genuinely DIFFERENT windows
# ======================================================================
print("\n================ Part 7 -- _sum_expert_counts on distinct windows ================")
mG = build_model()
mG.train()
mG.zero_grad(set_to_none=True)
_forward_backward_core(mG, XA, YA, "cpu", "fp32", 0.0, None, 0.5)
countsA = [block.moe.last_expert_counts.tolist() for block in mG.blocks]
_forward_backward_core(mG, XB, YB, "cpu", "fp32", 0.0, None, 0.5)
countsB = [block.moe.last_expert_counts.tolist() for block in mG.blocks]
summed = _sum_expert_counts([countsA, countsB])
hand_summed = [[a + b for a, b in zip(la, lb)] for la, lb in zip(countsA, countsB)]
print(f"  window A per-layer expert counts : {countsA}")
print(f"  window B per-layer expert counts : {countsB}")
print(f"  _sum_expert_counts               : {summed}")
print(f"  hand addition                    : {hand_summed}")
assert summed == hand_summed


# ======================================================================
# Part 8 -- the real repository test, at its own real shapes
# ======================================================================
print("\n================ Part 8 -- the real test's shapes (batch 8 vs. 2x4) ================")


def _tiny_model_config(**overrides):
    defaults = dict(
        vocab_size=12, d_model=16, n_layers=2, n_heads=4, n_kv_heads=2, d_ff=32,
        n_routed_experts=4, n_shared_experts=1, top_k=2, max_seq_len=32,
    )
    defaults.update(overrides)
    return ModelConfig(**defaults)


def run_repo_style_check(aux_loss_weight: float) -> float:
    torch.manual_seed(0)
    config = _tiny_model_config(aux_loss_weight=aux_loss_weight)
    reference_model = QwenMoEModel(config)
    accum_model = QwenMoEModel(config)
    accum_model.load_state_dict(reference_model.state_dict())

    torch.manual_seed(1)
    x = torch.randint(0, config.vocab_size, (8, 8))
    y = torch.randint(0, config.vocab_size, (8, 8))

    reference_config = TrainConfig(batch_size=4, seq_len=8, total_tokens=1, device="cpu", warmup_steps=0, grad_clip_norm=1e6)
    accum_config = TrainConfig(batch_size=4, seq_len=8, total_tokens=1, device="cpu", warmup_steps=0, grad_clip_norm=1e6, grad_accum_steps=2)
    reference_optimizer = build_optimizer(reference_model, reference_config)
    accum_optimizer = build_optimizer(accum_model, accum_config)

    _forward_backward_step(reference_model, reference_optimizer, x, y, grad_clip_norm=1e6, device="cpu")
    micro_batches = [(x[:4], y[:4], None), (x[4:], y[4:], None)]
    _run_step(accum_model, accum_optimizer, micro_batches, "cpu", 0, 1, accum_config)

    return max(
        (ref_p - accum_p).abs().max().item()
        for ref_p, accum_p in zip(reference_model.parameters(), accum_model.parameters())
    )


diff_repo_zero = run_repo_style_check(0.0)
diff_repo_default = run_repo_style_check(0.01)
print(f"  aux_loss_weight=0.0  : max |Δparam| = {diff_repo_zero:.6e}  (repo test's actual assertion: atol=1e-5)")
print(f"  aux_loss_weight=0.01 : max |Δparam| = {diff_repo_default:.6e}  (would FAIL atol=1e-5)")
assert diff_repo_zero < 1e-5
assert diff_repo_default > 1e-5


# ======================================================================
# Part 9 -- Chinchilla arithmetic behind the "Verdict" section's ~2.4B figure
# ======================================================================
print("\n================ Part 9 -- the implied Chinchilla budget ================")
real_300m_model = QwenMoEModel(SCALE_300M)
breakdown = count_params(real_300m_model)
implied_budget = chinchilla_budget(breakdown.active)
print(f"  SCALE_300M active params (real count_params) : {breakdown.active:,}")
print(f"  chinchilla_budget(active)  = 20 * {breakdown.active:,} = {implied_budget:,}")
print(f"  journal 038 quoted this as \"~2.4B tokens against the ~2M actually used\"")
assert 121_000_000 < breakdown.active < 123_000_000  # matches presets.py's own "~122.0M active" comment
assert 2_300_000_000 < implied_budget < 2_500_000_000

print("\nAll checks passed.")
