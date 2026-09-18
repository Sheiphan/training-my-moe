"""Reproduces every hand-computed number in `006-full-model-assembly.md`.

Run it:  uv run python docs/math/verify_006_full_model_assembly.py

It builds the exact tiny `QwenMoEModel` the derivation uses

    vocab_size=6, d_model=4, n_layers=2, n_heads=2, n_kv_heads=1,
    head_dim=2, positional_scheme="nope", d_ff=3, n_routed_experts=2,
    top_k=1, n_shared_experts=0, swiglu experts, aux-loss balancing,
    softmax routing, tie_word_embeddings=True, mtp_depth=0

pins the embedding table, both blocks' attention + MoE weights, and every
RMSNorm gain to the values written out by hand in the doc, then runs BOTH
model code paths

  * `forward_with_trace` -- the explicit per-sublayer path the doc walks
    (explicit attention matmul/softmax + explicit per-expert MoE loop), and
  * `forward`            -- the fused production path (SDPA + batched MoE
    dispatch),

prints every intermediate tensor (embedding lookup, positions, each
block's pre-norm / sublayer / residual, the final norm, the tied LM head,
and the per-block + combined auxiliary loss), and `assert`s:

  * the by-hand end-to-end logits `(1, 3, 6)`,
  * the by-hand combined `aux_loss` (mean, not sum, of the two blocks),
  * every RMSNorm output, recomputed from the bare formula,
  * weight tying is object identity (`lm_head.weight is embedding.weight`),
  * the two blocks own *independent* parameters (nothing is shared),
  * `forward` and `forward_with_trace` agree within 1e-5.

No fabricated numbers -- this is the source of truth the prose quotes.
"""
from __future__ import annotations

import torch

from qwenmoe.config import ModelConfig
from qwenmoe.model.transformer import QwenMoEModel

torch.set_printoptions(precision=6, sci_mode=False)
RMSNORM_EPS = 1e-6


# --- the model configuration (Part 1 of the doc) ------------------------
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


# --- the hand-picked weights (Part 1 of the doc) ------------------------
# Embedding / (tied) LM-head matrix E, shape (vocab_size, d_model) = (6, 4).
E = torch.tensor(
    [
        [1.0, 1.0, 0.0, 0.0],    # token 0
        [1.0, 0.0, -1.0, 0.0],   # token 1  -- used at position 1
        [0.0, 0.0, 1.0, 1.0],    # token 2
        [-1.0, 0.0, 0.0, 1.0],   # token 3
        [0.0, 1.0, 0.0, 1.0],    # token 4  -- used at positions 0 and 2
        [1.0, -1.0, 1.0, -1.0],  # token 5
    ]
)

INPUT_IDS = torch.tensor([[4, 1, 4]])  # (batch=1, seq_len=3)

# Attention weights, SHARED by both blocks purely to keep the weight table
# small -- nothing in the code shares them (each block owns its own
# nn.Linear objects; the script asserts this below).
WQ = torch.eye(4)                                  # q = x            (identity)
WK = torch.tensor([[1.0, 0, 0, 0], [0, 1, 0, 0]])  # k = x[:, :2]
WV = torch.tensor([[0.0, 0, 1, 0], [0, 0, 0, 1]])  # v = x[:, 2:]
WO = torch.eye(4)                                  # o = concat(heads) (identity)

# MoE weights (also shared by both blocks, same reason).
W_ROUTER = torch.tensor([[1.0, 0, 0, 0], [0, 0, 1, 0]])  # logit_e0 = x0, logit_e1 = x2

E0_GATE = torch.tensor([[1.0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])   # (d_ff=3, d_model=4)
E0_UP = torch.tensor([[0.0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]])
E0_DOWN = torch.tensor([[1.0, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]])  # (d_model=4, d_ff=3)

E1_GATE = torch.tensor([[0.0, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0]])
E1_UP = torch.tensor([[1.0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0]])
E1_DOWN = torch.tensor([[1.0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 0, 1]])


def _pin_attention(attn) -> None:
    with torch.no_grad():
        attn.w_q.weight.copy_(WQ)
        attn.w_k.weight.copy_(WK)
        attn.w_v.weight.copy_(WV)
        attn.w_o.weight.copy_(WO)


def _pin_moe(moe) -> None:
    with torch.no_grad():
        moe.router.gate.weight.copy_(W_ROUTER)
        moe.routed_experts[0].w_gate.weight.copy_(E0_GATE)
        moe.routed_experts[0].w_up.weight.copy_(E0_UP)
        moe.routed_experts[0].w_down.weight.copy_(E0_DOWN)
        moe.routed_experts[1].w_gate.weight.copy_(E1_GATE)
        moe.routed_experts[1].w_up.weight.copy_(E1_UP)
        moe.routed_experts[1].w_down.weight.copy_(E1_DOWN)


def _build() -> QwenMoEModel:
    model = QwenMoEModel(_cfg())
    with torch.no_grad():
        model.embedding.weight.copy_(E)           # lm_head.weight IS this tensor
        model.final_norm.weight.fill_(1.0)        # already ones; explicit for the doc
        for block in model.blocks:
            block.attn_norm.weight.fill_(1.0)
            block.moe_norm.weight.fill_(1.0)
            _pin_attention(block.attn)
            _pin_moe(block.moe)
    return model.eval()


def rmsnorm_by_hand(x: torch.Tensor, label: str) -> torch.Tensor:
    """RMSNorm with an all-ones gain, straight from rmsnorm.py:14-17:
        variance = mean(x**2)            (over the last axis)
        x_normed = x * rsqrt(variance + eps)
    Prints the two intermediates the doc quotes and returns the result."""
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    inv_rms = torch.rsqrt(variance + RMSNORM_EPS)
    out = x * inv_rms
    print(f"  [{label}]  mean(x^2) per row : {variance.flatten().tolist()}")
    print(f"  [{label}]  rsqrt(var+eps)    : {inv_rms.flatten().tolist()}")
    print(f"  [{label}]  x_normed          :\n{out.squeeze(0)}")
    return out


model = _build()

# ---------------------------------------------------------------------------
# Walk forward_with_trace, mirroring the doc's Step numbering.
# ---------------------------------------------------------------------------
print("================ running example: QwenMoEModel ================")
print("input_ids (batch, seq_len):", INPUT_IDS.tolist())

# Step 1 -- embedding lookup
x = model.embedding(INPUT_IDS)
print("\nStep 1  x = E[input_ids]   (1, 3, 4):\n", x.squeeze(0))

# Step 2 -- positions
positions = torch.arange(INPUT_IDS.shape[1])
print("\nStep 2  positions = arange(seq_len):", positions.tolist())

# Steps 4-15 -- the two blocks, via each block's explicit trace path
logits_trace, bundle = model.forward_with_trace(INPUT_IDS)

x = bundle.embeddings.unsqueeze(0)
aux_by_block = []
router_probs_by_block = []
for i, block in enumerate(model.blocks):
    print(f"\n---------------- Block {i} ----------------")
    normed_attn = rmsnorm_by_hand(x, f"block{i}.attn_norm")
    attn_out, attn_trace = block.attn.forward_with_trace(normed_attn, positions)
    print(f"  attn probs (post-softmax, per head):\n{attn_trace['attn_scores_post_softmax']}")
    print(f"  attn_out  (1, 3, 4):\n{attn_out.squeeze(0)}")
    x = x + attn_out
    print(f"  x <- x + attn_out:\n{x.squeeze(0)}")

    normed_moe = rmsnorm_by_hand(x, f"block{i}.moe_norm")
    moe_out, aux_loss, moe_trace = block.moe.forward_with_trace(normed_moe)
    print(f"  router_logits:\n{moe_trace['router_logits']}")
    print(f"  router_probs :\n{moe_trace['router_probs']}")
    print(f"  topk_indices : {moe_trace['topk_indices'].flatten().tolist()}"
          f"   topk_weights: {moe_trace['topk_weights'].flatten().tolist()}")
    print(f"  expert dispatch counts: {block.moe.last_expert_counts.tolist()}")
    for eid, eo in moe_trace["expert_outputs"].items():
        print(f"    expert {eid} out (pre-gate): {eo.tolist()}")
    print(f"  moe_out  (1, 3, 4):\n{moe_out.squeeze(0)}")
    print(f"  aux_loss (block {i}): {aux_loss.item():.8f}")
    x = x + moe_out
    print(f"  x <- x + moe_out:\n{x.squeeze(0)}")
    aux_by_block.append(aux_loss)
    router_probs_by_block.append(moe_trace["router_probs"])

# Step 16 -- final norm
print("\n---------------- Head ----------------")
normed_final = rmsnorm_by_hand(x, "final_norm")

# Step 17 -- tied LM head
logits_by_hand = normed_final @ model.embedding.weight.t()
print("\nStep 17  logits = final_norm(x) @ E^T   (1, 3, 6):\n", logits_by_hand.squeeze(0))

# Step 18 -- combine the per-block aux losses: MEAN, not sum
combined = torch.stack(aux_by_block).mean()
print(f"\nStep 18  aux per block = {[round(a.item(), 8) for a in aux_by_block]}")
print(f"         combined aux_loss = mean = {combined.item():.8f}"
      f"   (sum would be {torch.stack(aux_by_block).sum().item():.8f})")

# ---------------------------------------------------------------------------
# The fused production path, and the two-path agreement check.
# ---------------------------------------------------------------------------
logits_fused, extras_fused = model(INPUT_IDS)
print("\n================ fused forward() ================")
print("logits (1, 3, 6):\n", logits_fused.squeeze(0))
print("extras['aux_loss']:", extras_fused["aux_loss"].item())
print("extras keys (mtp_depth=0 -> just aux_loss):", sorted(extras_fused.keys()))
max_logit_diff = (logits_fused - logits_trace).abs().max().item()
max_logit_diff_hand = (logits_fused - logits_by_hand).abs().max().item()
print("max |fused - trace|  logits :", max_logit_diff)
print("max |fused - by_hand| logits :", max_logit_diff_hand)

# ---------------------------------------------------------------------------
# Assertions -- the doc's Parts 2-4.
# ---------------------------------------------------------------------------
# The by-hand end-to-end logits from 006-full-model-assembly.md, Step 17.
EXPECTED_LOGITS = torch.tensor(
    [
        [1.414213, 0.000000, 1.414213, 1.414213, 2.828427, -2.828427],
        [-0.039239, 0.622172, -0.530086, 1.682475, 1.643236, -3.855798],
        [1.185529, 0.009417, 0.999658, 1.577581, 2.763110, -3.341033],
    ]
)
torch.testing.assert_close(logits_fused.squeeze(0), EXPECTED_LOGITS, atol=2e-6, rtol=0)
torch.testing.assert_close(logits_trace.squeeze(0), EXPECTED_LOGITS, atol=2e-5, rtol=0)
torch.testing.assert_close(logits_by_hand.squeeze(0), logits_fused.squeeze(0), atol=2e-5, rtol=0)

# by-hand check of position 0's logit row: final_norm(x)[0] = [0, c, 0, c]
# with c = 2^-0.5 * 2 (see Step 16), so logit_j = c * (E[j,1] + E[j,3]).
c0 = normed_final.squeeze(0)[0, 1].item()
row0_by_formula = torch.tensor([c0 * (E[j, 1] + E[j, 3]).item() for j in range(6)])
torch.testing.assert_close(logits_fused.squeeze(0)[0], row0_by_formula, atol=2e-6, rtol=0)

# weight tying is object identity, not a copied value (journal 006)
assert model.lm_head.weight is model.embedding.weight, "tying must be `is`, not a copy"
assert torch.equal(model.lm_head.weight, E)

# the two blocks own independent parameters -- nothing is shared
assert model.blocks[0].attn.w_q.weight is not model.blocks[1].attn.w_q.weight
assert model.blocks[0].moe.router.gate.weight is not model.blocks[1].moe.router.gate.weight
assert model.blocks[0].attn_norm.weight is not model.blocks[1].attn_norm.weight

# combined aux loss is the MEAN of the per-block losses
torch.testing.assert_close(
    extras_fused["aux_loss"], torch.stack(aux_by_block).mean(), atol=1e-8, rtol=0
)
assert not torch.allclose(
    extras_fused["aux_loss"], torch.stack(aux_by_block).sum()
), "combined aux loss must be mean, not sum (they'd only match if n_layers==1)"

# by-hand check of each block's Switch aux loss: alpha * N * sum_i f_i * P_i
# with alpha=0.01, N=2, f = [1, 0] (every token routed to expert 0, top_k=1),
# P_i = column mean of that block's router_probs (derivation: entry 005).
for i in range(len(model.blocks)):
    P_i = router_probs_by_block[i].mean(dim=0)
    f_i = torch.tensor([1.0, 0.0])
    expected_aux = 0.01 * 2 * torch.sum(f_i * P_i)
    torch.testing.assert_close(aux_by_block[i], expected_aux, atol=1e-7, rtol=0)
    assert abs(f_i.sum().item() - 1.0) < 1e-9   # sum_i f_i == top_k == 1
    assert abs(P_i.sum().item() - 1.0) < 1e-6   # sum_i P_i == 1

# mtp_depth=0 -> forward returns exactly (logits, {'aux_loss': ...})
assert sorted(extras_fused.keys()) == ["aux_loss"]

# fused vs explicit-trace agreement (Requirement 7.2 for the sublayers)
assert max_logit_diff < 1e-5

# parameter budget for the "cost" section of the doc. `.parameters()`
# deduplicates the tied tensor, so the LM head contributes 0.
n_params = sum(p.numel() for p in model.parameters())
n_params_untied = sum(
    p.numel() for p in QwenMoEModel(_cfg(tie_word_embeddings=False)).parameters()
)
print("\nparameter count (tied)  :", n_params)
print("parameter count (untied):", n_params_untied, f"(+{n_params_untied - n_params})")
assert n_params == 300, n_params
assert n_params_untied == 300 + 6 * 4  # one extra (vocab_size, d_model) matrix

print("\nAll assertions passed -- every number in 006-full-model-assembly.md is real.")
