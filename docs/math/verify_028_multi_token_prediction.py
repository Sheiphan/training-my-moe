"""Reproduces every hand-computed number in `028-multi-token-prediction.md`.

Run it:  uv run python docs/math/verify_028_multi_token_prediction.py

Builds a real (tiny) `QwenMoEModel` with `mtp_depth=2`, pins every weight
`MTPHead.forward` (mtp.py:60-71) touches to a hand-chosen matrix, and:

  * decomposes MTP head 1's forward pass into its exact line sequence
    (RMSNorm x2 -> concat -> fuse -> pre-norm attention residual ->
    pre-norm FFN residual -> out norm), printing and asserting every
    intermediate, then confirms the decomposition matches calling the
    REAL `MTPHead.forward` directly;
  * within that, calls `Attention.forward_with_trace` (attention.py:183)
    to get the explicit q/k/v/scores/softmax the fused
    `F.scaled_dot_product_attention` path (attention.py:119, the path
    `MTPHead.forward` actually calls) computes internally, and shows the
    two agree to a measured max difference;
  * computes each depth's cross-entropy loss by hand from
    `logsumexp(logits) - logits[target]` (the formula derived in
    `010-training-loop.md`) and checks it against `F.cross_entropy`;
  * repeats the whole decomposition for depth 2, whose `hidden` input is
    depth 1's OUTPUT (not the trunk's) -- the causal chain -- and whose
    single usable position exercises the "only one key is visible"
    degenerate case of causal attention;
  * calls the REAL `QwenMoEModel._mtp_losses` (transformer.py:145) on the
    same hand-picked trunk hidden state and confirms it reproduces both
    depths' by-hand losses exactly;
  * exercises `QwenMoEModel.forward`'s MTP gating (transformer.py:137) in
    all four states (train+targets, eval+targets, train+no-targets,
    `mtp_depth=0`) and the real `_forward_backward_core`
    (trainer.py:176) that adds `mtp_lambda * mean(mtp_losses)` to the
    total loss;
  * counts real parameters for the cost section.

No fabricated numbers -- every value printed here is what the real
`qwenmoe/` modules compute; the doc only *quotes* this script's output.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from qwenmoe.config import ModelConfig
from qwenmoe.model.transformer import QwenMoEModel
from qwenmoe.train.trainer import _forward_backward_core

torch.set_printoptions(precision=6, sci_mode=False)
torch.manual_seed(0)  # only the trunk's own (otherwise-unused) weights are random


def _cfg(**overrides) -> ModelConfig:
    base = dict(
        vocab_size=4,           # A=0, B=1, C=2, D=3
        d_model=4,
        n_layers=1,             # trunk is a structural prop only -- see note below
        n_heads=2,
        n_kv_heads=1,           # < n_heads: GQA, exercised inside every MTPHead's attn
        d_ff=3,
        n_shared_experts=0,
        n_routed_experts=1,    # trivial trunk MoE (irrelevant to MTP itself)
        top_k=1,
        max_seq_len=8,
        positional_scheme="nope",  # arithmetic clarity, exactly doc004's precedent --
                                   # MTPHead is agnostic to positional_scheme, it just
                                   # forwards `positions` to whatever build_attention(config)
                                   # constructs; RoPE is unrelated to any MTP-specific logic.
        activation="swiglu",
        attention_variant="gqa",
        mtp_depth=2,
        mtp_lambda=0.3,
        aux_loss_weight=0.0,   # isolates the MTP term in the trainer's total_loss (Part 6)
    )
    base.update(overrides)
    return ModelConfig(**base)


# --- the hand-picked weights -------------------------------------------------
# Embedding table: one-hot rows. Two payoffs -- embedding-then-RMSNorm gives a
# clean rsqrt(0.25) = 2.0, and (since tie_word_embeddings=True by default) the
# SAME matrix is lm_head, so with vocab_size == d_model == 4 the lm_head
# projection is the identity: depth_logits == new_hidden, no extra matmul to
# carry through the cross-entropy step.
EMB = torch.eye(4)

# fuse: nn.Linear(2*d_model=8, d_model=4). W = [I4 | I4] makes
# fuse(cat([a, b])) = a + b -- concat-then-linear collapsing to plain
# addition for THIS weight choice (a general W_fuse would mix all 8 inputs
# into each of the 4 outputs; see the doc's derivation).
FUSE_W = torch.cat([torch.eye(4), torch.eye(4)], dim=1)

# Attention (GQA: n_heads=2, n_kv_heads=1, head_dim=2). w_q/w_o = identity;
# w_k selects dims [0,1] of its input, w_v selects dims [2,3] -- so Q's two
# heads read the SAME two slices w_k/w_v read, letting every dot product
# stay a plain sum of two products.
WQ = torch.eye(4)
WK = torch.tensor([[1.0, 0, 0, 0], [0, 1, 0, 0]])
WV = torch.tensor([[0.0, 0, 1, 0], [0, 0, 0, 1]])
WO = torch.eye(4)

# SwiGLU FFN (d_model=4, d_ff=3). w_gate picks dims [0,1,2] of its input,
# w_up picks dims [1,2,3]; w_down maps the 3 gated values back into 4
# outputs, the 4th being their sum -- one real accumulation to exercise
# nn.Linear's general row-times-matrix form, not just a selection.
WGATE = torch.tensor([[1.0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
WUP = torch.tensor([[0.0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
WDOWN = torch.tensor([[1.0, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]])


def _pin_head(head) -> None:
    """Every MTPHead in `model.mtp_heads` gets these SAME numbers (a
    deliberate economy -- each is still its own independent nn.Module and
    the doc calls that out explicitly). RMSNorm gains are left at their
    default all-ones init (rmsnorm.py:12) -- not re-pinned, just asserted.
    """
    with torch.no_grad():
        head.fuse.weight.copy_(FUSE_W)
        head.attn.w_q.weight.copy_(WQ)
        head.attn.w_k.weight.copy_(WK)
        head.attn.w_v.weight.copy_(WV)
        head.attn.w_o.weight.copy_(WO)
        head.ffn.w_gate.weight.copy_(WGATE)
        head.ffn.w_up.weight.copy_(WUP)
        head.ffn.w_down.weight.copy_(WDOWN)
    for norm in (head.hidden_norm, head.emb_norm, head.attn_norm, head.ffn_norm, head.out_norm):
        assert torch.equal(norm.weight, torch.ones(4)), "RMSNorm gains should start at 1"


cfg = _cfg()
model = QwenMoEModel(cfg)
with torch.no_grad():
    model.embedding.weight.copy_(EMB)
for head in model.mtp_heads:
    _pin_head(head)
model.eval()

assert len(model.mtp_heads) == 2

# The running example: input_ids = [A, B, C], targets = [B, C, D] (the
# standard shift-by-one target; D never appears as an input, only as a
# target -- a perfectly ordinary next-token supervision pair).
input_ids = torch.tensor([[0, 1, 2]])
targets = torch.tensor([[1, 2, 3]])
positions = torch.arange(3)

# trunk_hidden is HAND-PICKED, not produced by this toy trunk's forward --
# _mtp_losses (transformer.py:145) accepts any (batch, seq_len, d_model)
# tensor, and the trunk's own attention/MoE math is already fully derived
# in 004-attention-gqa.md / 005-moe-layer.md / 006-full-model-assembly.md,
# so this doc's "given" input starts one level in, exactly the way
# 027-batched-expert-computation.md quotes 005's routing tensors instead
# of re-deriving them.
trunk_hidden = torch.tensor([[[1.0, 0, 1, 0], [0, 1, 0, 1], [1, 1, 0, 0]]])

print("================ running example: multi-token prediction ================")
print("input_ids:", input_ids.tolist(), " targets:", targets.tolist())
print("trunk_hidden (hand-picked):\n", trunk_hidden)

# ============================================================
# Part 1 -- MTP head 1 (depth=1), decomposed line by line (mtp.py:60-71)
# ============================================================
head1 = model.mtp_heads[0]
usable1 = 3 - 1
hidden_in1 = trunk_hidden[:, :usable1, :]                       # transformer.py:172
next_emb1 = model.embedding(input_ids[:, 1 : 1 + usable1])      # transformer.py:173
pos_in1 = positions[:usable1]                                   # transformer.py:174

print("\n--- depth 1 input (transformer.py:171-174) ---")
print("hidden_in1:\n", hidden_in1)
print("next_emb1 (Emb(B), Emb(C)):\n", next_emb1)
print("pos_in1:", pos_in1.tolist())

hidden_normed1 = head1.hidden_norm(hidden_in1)                  # mtp.py:67
emb_normed1 = head1.emb_norm(next_emb1)                         # mtp.py:67
print("\n--- fusion (mtp.py:66-68) ---")
print("hidden_norm(hidden_in1):\n", hidden_normed1)
print("emb_norm(next_emb1):\n", emb_normed1)

fused1 = head1.fuse(torch.cat([hidden_normed1, emb_normed1], dim=-1))  # mtp.py:66-68
print("fused1 = fuse(cat([...])):\n", fused1)

torch.testing.assert_close(
    fused1, hidden_normed1 + emb_normed1, atol=1e-6, rtol=0
)  # W_fuse = [I|I] collapses concat+linear to addition, for THIS weight choice

attn_normed1 = head1.attn_norm(fused1)                          # mtp.py:69
print("\n--- attention sub-layer (mtp.py:69) ---")
print("attn_norm(fused1):\n", attn_normed1)

attn_out1_trace, trace1 = head1.attn.forward_with_trace(attn_normed1, pos_in1)
print("q (per head, pre-repeat, 1 kv head only shown for k/v):\n", trace1["q"])
print("k (the ONE shared kv head):\n", trace1["k"])
print("v (the ONE shared kv head):\n", trace1["v"])
print("scores, pre-mask (2 heads, 2x2):\n", trace1["attn_scores_pre_mask"])
print("scores, post-softmax:\n", trace1["attn_scores_post_softmax"])

attn_out1 = head1.attn(attn_normed1, pos_in1)                   # mtp.py:69 -- REAL fused path
print("attn_out (fused SDPA path):\n", attn_out1)
diff_attn1 = (attn_out1 - attn_out1_trace).abs().max().item()
print("max |fused - explicit trace| :", diff_attn1)
assert diff_attn1 < 1e-5

x1_depth1 = fused1 + attn_out1                                  # mtp.py:69
print("x (post-attention residual):\n", x1_depth1)

ffn_normed1 = head1.ffn_norm(x1_depth1)                         # mtp.py:70
pre_gate1 = head1.ffn.w_gate(ffn_normed1)
up1 = head1.ffn.w_up(ffn_normed1)
silu1 = F.silu(pre_gate1)
h_gated1 = silu1 * up1
ffn_out1 = head1.ffn.w_down(h_gated1)                           # moe.py:29 (SwiGLUExpert.forward)
print("\n--- FFN sub-layer (mtp.py:70) ---")
print("ffn_norm(x):\n", ffn_normed1)
print("pre_gate = w_gate(x):\n", pre_gate1)
print("up = w_up(x):\n", up1)
print("silu(pre_gate):\n", silu1)
print("h = silu(pre_gate) * up:\n", h_gated1)
print("ffn_out = w_down(h):\n", ffn_out1)

ffn_out1_direct = head1.ffn(ffn_normed1)
assert torch.equal(ffn_out1, ffn_out1_direct)

x2_depth1 = x1_depth1 + ffn_out1                                # mtp.py:70
new_hidden1 = head1.out_norm(x2_depth1)                         # mtp.py:71
print("x (post-FFN residual):\n", x2_depth1)
print("new_hidden1 = out_norm(x):\n", new_hidden1)

new_hidden1_direct = head1(hidden_in1, next_emb1, pos_in1)      # the REAL, undecomposed call
diff_decomp1 = (new_hidden1 - new_hidden1_direct).abs().max().item()
print("max |decomposed - head1.forward| :", diff_decomp1)
assert diff_decomp1 == 0.0

# --- depth 1's loss (transformer.py:176-183) ---
depth_logits1 = model.lm_head(new_hidden1).float()              # == new_hidden1 (tied, EMB=I)
depth_target1 = targets[:, 1 : 1 + usable1]
print("\n--- depth 1 loss (transformer.py:176-183) ---")
print("depth_logits1 (== new_hidden1, lm_head tied to I4):\n", depth_logits1)
print("depth_target1 (C, D):", depth_target1.tolist())

log_z1 = torch.logsumexp(depth_logits1.reshape(-1, 4), dim=-1)
picked1 = depth_logits1.reshape(-1, 4).gather(1, depth_target1.reshape(-1, 1)).squeeze(1)
ce1_manual = (log_z1 - picked1).mean()
ce1_real = F.cross_entropy(depth_logits1.reshape(-1, 4), depth_target1.reshape(-1))
print("logsumexp per position:", log_z1.tolist())
print("picked (target) logit per position:", picked1.tolist())
print("CE (by hand, logsumexp - picked, mean over 2 positions):", ce1_manual.item())
print("CE (F.cross_entropy):", ce1_real.item())
assert torch.allclose(ce1_manual, ce1_real, atol=1e-6)

# ============================================================
# Part 2 -- MTP head 2 (depth=2): hidden comes from depth 1's OUTPUT
# ============================================================
head2 = model.mtp_heads[1]
usable2 = 3 - 2
hidden_in2 = new_hidden1[:, :usable2, :]                        # the causal chain
next_emb2 = model.embedding(input_ids[:, 2 : 2 + usable2])
pos_in2 = positions[:usable2]

print("\n================ depth 2 ================")
print("hidden_in2 (= new_hidden1's first", usable2, "position(s)):\n", hidden_in2)
print("next_emb2 (Emb(C)):\n", next_emb2)
print("pos_in2:", pos_in2.tolist())

hidden_normed2 = head2.hidden_norm(hidden_in2)
emb_normed2 = head2.emb_norm(next_emb2)
fused2 = head2.fuse(torch.cat([hidden_normed2, emb_normed2], dim=-1))
attn_normed2 = head2.attn_norm(fused2)
print("hidden_norm(hidden_in2):\n", hidden_normed2)
print("emb_norm(next_emb2):\n", emb_normed2)
print("fused2:\n", fused2)
print("attn_norm(fused2):\n", attn_normed2)

attn_out2_trace, trace2 = head2.attn.forward_with_trace(attn_normed2, pos_in2)
attn_out2 = head2.attn(attn_normed2, pos_in2)
diff_attn2 = (attn_out2 - attn_out2_trace).abs().max().item()
print("q:\n", trace2["q"], "\nk:\n", trace2["k"], "\nv:\n", trace2["v"])
print("scores pre-mask:\n", trace2["attn_scores_pre_mask"])
print("scores post-softmax (only 1 key visible -> exactly 1.0):\n", trace2["attn_scores_post_softmax"])
print("attn_out2:\n", attn_out2)
print("max |fused - explicit trace| (depth 2):", diff_attn2)
assert diff_attn2 < 1e-5
# the "only one visible key" fact: softmax of a single (unmasked) score is
# always exactly 1, regardless of the score's value, so attn_out2 must be
# EXACTLY v2 (broadcast to both heads) -- an exact property, not measured.
v2 = trace2["v"][0, 0]  # the one kv head's one value vector, shape (head_dim,)
expected_attn_out2 = torch.cat([v2, v2]).view(1, 1, 4)  # broadcast to both query heads
torch.testing.assert_close(attn_out2, expected_attn_out2, atol=0, rtol=0)

x1_depth2 = fused2 + attn_out2
ffn_normed2 = head2.ffn_norm(x1_depth2)
pre_gate2 = head2.ffn.w_gate(ffn_normed2)
up2 = head2.ffn.w_up(ffn_normed2)
silu2 = F.silu(pre_gate2)
h_gated2 = silu2 * up2
ffn_out2 = head2.ffn.w_down(h_gated2)
x2_depth2 = x1_depth2 + ffn_out2
new_hidden2 = head2.out_norm(x2_depth2)
print("ffn_normed2:\n", ffn_normed2)
print("pre_gate2:\n", pre_gate2, "\nup2:\n", up2, "\nsilu2:\n", silu2, "\nh_gated2:\n", h_gated2)
print("ffn_out2:\n", ffn_out2)
print("new_hidden2:\n", new_hidden2)

new_hidden2_direct = head2(hidden_in2, next_emb2, pos_in2)
diff_decomp2 = (new_hidden2 - new_hidden2_direct).abs().max().item()
print("max |decomposed - head2.forward| :", diff_decomp2)
assert diff_decomp2 == 0.0

depth_logits2 = model.lm_head(new_hidden2).float()
depth_target2 = targets[:, 2 : 2 + usable2]
log_z2 = torch.logsumexp(depth_logits2.reshape(-1, 4), dim=-1)
picked2 = depth_logits2.reshape(-1, 4).gather(1, depth_target2.reshape(-1, 1)).squeeze(1)
ce2_manual = (log_z2 - picked2).mean()
ce2_real = F.cross_entropy(depth_logits2.reshape(-1, 4), depth_target2.reshape(-1))
print("depth_logits2:\n", depth_logits2, " depth_target2:", depth_target2.tolist())
print("CE depth 2 (by hand):", ce2_manual.item(), " (F.cross_entropy):", ce2_real.item())
assert torch.allclose(ce2_manual, ce2_real, atol=1e-6)

# ============================================================
# Part 3 -- the REAL `_mtp_losses` (transformer.py:145-185), not a
# reimplementation, called directly on the same trunk_hidden
# ============================================================
model.train()
real_losses = model._mtp_losses(trunk_hidden, input_ids, targets, positions)
print("\n================ Part 3: real _mtp_losses ================")
print("real_losses:", [l.item() for l in real_losses])
print("by-hand      :", [ce1_manual.item(), ce2_manual.item()])
assert torch.allclose(real_losses[0], ce1_manual, atol=1e-6)
assert torch.allclose(real_losses[1], ce2_manual, atol=1e-6)

# ============================================================
# Part 4 -- QwenMoEModel.forward's MTP gating (transformer.py:137-141),
# every branch
# ============================================================
print("\n================ Part 4: forward's MTP gating ================")
model.train()
logits_full, extras_full = model(input_ids, targets=targets)
print("train + targets -> extras keys:", sorted(extras_full.keys()))
assert set(extras_full.keys()) == {"aux_loss", "mtp_loss", "mtp_losses"}
assert len(extras_full["mtp_losses"]) == 2
mean_check = sum(l.item() for l in extras_full["mtp_losses"]) / 2
assert abs(extras_full["mtp_loss"].item() - mean_check) < 1e-6

model.eval()
_, extras_eval = model(input_ids, targets=targets)
print("eval + targets  -> extras keys:", sorted(extras_eval.keys()))
assert set(extras_eval.keys()) == {"aux_loss"}

model.train()
_, extras_notargets = model(input_ids)
print("train, no targets -> extras keys:", sorted(extras_notargets.keys()))
assert set(extras_notargets.keys()) == {"aux_loss"}

cfg0 = _cfg(mtp_depth=0)
model0 = QwenMoEModel(cfg0)
print("\nmtp_depth=0: len(mtp_heads) =", len(model0.mtp_heads))
assert len(model0.mtp_heads) == 0
assert not any("mtp_heads" in k for k in model0.state_dict())
model0.train()
logits0_t, extras0_t = model0(input_ids, targets=targets)
logits0_notarget, extras0_notarget = model0(input_ids)
print("mtp_depth=0, train+targets -> extras keys:", sorted(extras0_t.keys()))
assert set(extras0_t.keys()) == {"aux_loss"}
diff0 = (logits0_t - logits0_notarget).abs().max().item()
print("max |logits(x,targets=y) - logits(x)| (mtp_depth=0):", diff0)
assert diff0 == 0.0

# ============================================================
# Part 5 -- the trainer: `_forward_backward_core` (trainer.py:176-268)
# adds `mtp_lambda * mean(mtp_losses)` to the total loss
# ============================================================
print("\n================ Part 5: trainer wiring ================")
model.train()
for p in model.parameters():
    p.grad = None
metrics = _forward_backward_core(
    model, input_ids, targets, device="cpu", precision="fp32",
    z_loss_weight=0.0, segment_ids=None, loss_scale=1.0,
)
print("metrics:", metrics)
total_expected = metrics["lm_loss"] + metrics["aux_loss"] + cfg.mtp_lambda * metrics["mtp_loss"]
print("lm_loss + aux_loss + mtp_lambda * mtp_loss =", total_expected)
assert metrics["mtp_loss"] is not None
assert abs(metrics["mtp_loss"] - sum(metrics["mtp_losses"]) / 2) < 1e-6
# every pinned parameter that participates in BOTH heads' forward pass got a
# gradient from backward() (Requirement 6.3's "MTP losses train the shared
# trunk too" -- here, the shared MTP-head weights)
assert head1.fuse.weight.grad is not None
assert head2.fuse.weight.grad is not None

# ============================================================
# Part 6 -- parameter cost, on this toy model's own numbers
# ============================================================
def _nparams(m) -> int:
    return sum(p.numel() for p in m.parameters())

n_head = _nparams(model.mtp_heads[0])
n_full2 = _nparams(model)
n_full0 = _nparams(model0)
print("\n================ Part 6: parameter cost ================")
print("params in one MTPHead:", n_head)
print("full model, mtp_depth=2:", n_full2)
print("full model, mtp_depth=0:", n_full0)
print("overhead: (%d - %d) / %d = %.4f%%" % (n_full2, n_full0, n_full0, 100 * (n_full2 - n_full0) / n_full0))
assert n_full2 - n_full0 == 2 * n_head

print("\nAll assertions passed -- every number in 028-multi-token-prediction.md is real.")
