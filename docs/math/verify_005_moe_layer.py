"""Reproduces every hand-computed number in `005-moe-layer.md`.

Run it:  uv run python docs/math/verify_005_moe_layer.py

It builds the exact tiny `MoELayer` the derivation uses (d_model=2, d_ff=3,
n_routed_experts=4, top_k=2, n_shared_experts=1, swiglu experts, aux-loss
balancing, softmax routing -- every field on the pre-modernization path),
pins the router weight and all five expert FFNs to the values written out
by hand in the doc, runs BOTH code paths

  * `forward_with_trace` -- the explicit per-expert loop the doc walks, and
  * `forward`            -- the batched `dispatch_batched` production path,

prints every intermediate tensor, and `assert`s the by-hand end-to-end
output, the renormalized gate weights, the expert dispatch counts, and the
Switch-Transformer auxiliary loss. No fabricated numbers -- this is the
source of truth the prose quotes.
"""
from __future__ import annotations

import torch

from qwenmoe.config import ModelConfig
from qwenmoe.model.moe import MoELayer, score_and_select

torch.set_printoptions(precision=6, sci_mode=False)


# --- the model configuration (Part 1 of the doc) -------------------------
def _cfg(**overrides) -> ModelConfig:
    base = dict(
        vocab_size=10,
        d_model=2,
        d_ff=3,
        n_heads=2,          # MoE ignores these two, but ModelConfig validates
        n_kv_heads=1,       # them, so they must be internally consistent
        n_routed_experts=4,
        n_shared_experts=1,
        top_k=2,
        max_seq_len=8,
        activation="swiglu",
        moe_balancing="aux_loss",
        router_scoring="softmax",
        aux_loss_weight=0.01,
    )
    base.update(overrides)
    return ModelConfig(**base)


# --- the hand-picked weights (Part 1 of the doc) ------------------------
# router: logit for expert e at token x is  x . W_router[e]
W_ROUTER = torch.tensor([[2.0, 1.0], [1.0, -1.0], [-1.0, 2.0], [0.0, 0.0]])

# routed expert 0 (dispatched tokens 0, 1, 2)
E0_GATE = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
E0_UP = torch.tensor([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
E0_DOWN = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])

# routed expert 1 (dispatched token 0 only)
E1_GATE = torch.tensor([[-1.0, 0.0], [2.0, 0.0], [1.0, 1.0]])
E1_UP = torch.tensor([[1.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
E1_DOWN = torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])

# routed expert 2 (dispatched tokens 1, 2)
E2_GATE = torch.tensor([[0.0, 2.0], [1.0, -1.0], [0.0, 1.0]])
E2_UP = torch.tensor([[1.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
E2_DOWN = torch.tensor([[1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])

# routed expert 3 -- gets ZERO tokens this batch. Its weights are set to a
# loud, distinctive pattern precisely so the doc can assert that perturbing
# them does not move the output one bit (Requirement 2.5).
E3_GATE = torch.full((3, 2), 9.0)
E3_UP = torch.full((3, 2), 9.0)
E3_DOWN = torch.full((2, 3), 9.0)

# the one always-on shared expert (processes every token, unweighted)
ES_GATE = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 1.0]])
ES_UP = torch.tensor([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
ES_DOWN = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]])

X = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]])  # (1, 3, 2)


def _pin(expert, w_gate, w_up, w_down) -> None:
    with torch.no_grad():
        expert.w_gate.weight.copy_(w_gate)
        expert.w_up.weight.copy_(w_up)
        expert.w_down.weight.copy_(w_down)


def _build(**cfg_overrides) -> MoELayer:
    moe = MoELayer(_cfg(**cfg_overrides))
    with torch.no_grad():
        moe.router.gate.weight.copy_(W_ROUTER)
    _pin(moe.routed_experts[0], E0_GATE, E0_UP, E0_DOWN)
    _pin(moe.routed_experts[1], E1_GATE, E1_UP, E1_DOWN)
    _pin(moe.routed_experts[2], E2_GATE, E2_UP, E2_DOWN)
    _pin(moe.routed_experts[3], E3_GATE, E3_UP, E3_DOWN)
    _pin(moe.shared_experts[0], ES_GATE, ES_UP, ES_DOWN)
    return moe


def silu(z: torch.Tensor) -> torch.Tensor:
    return z * torch.sigmoid(z)


moe = _build()
moe.eval()  # AuxLossBalancer.update is a no-op anyway; eval keeps it tidy

out_trace, aux_trace, trace = moe.forward_with_trace(X)
out_fused, aux_fused = moe(X)

print("================ running example: MoELayer ================")
print("x  (batch, seq, d_model):\n", X)
print("\nrouter_logits  (n_tokens, n_routed_experts):\n", trace["router_logits"])
print("router_probs  (softmax over experts, sums to 1 per row):\n", trace["router_probs"])
print("topk_indices  (n_tokens, top_k):\n", trace["topk_indices"])
print("topk_weights  (renormalized, sum to 1 per row):\n", trace["topk_weights"])
print("last_expert_counts  (tokens dispatched to each routed expert):\n",
      moe.last_expert_counts)
print("\nper-expert dispatched outputs E_i(x) (pre-gate):")
for eid, eo in trace["expert_outputs"].items():
    print(f"  expert {eid}: {eo.tolist()}")
print("\nshared expert on every token  E_s(x):")
with torch.no_grad():
    print(" ", moe.shared_experts[0](X.reshape(-1, 2)).tolist())

# reconstruct routed_output and shared_output the way Step 6g / Step 7 do,
# straight from the trace, so the doc's per-expert scatter-add table is pinned
with torch.no_grad():
    x_flat = X.reshape(-1, 2)
    routed = torch.zeros(3, 2)
    for eid, eo in trace["expert_outputs"].items():
        tok, slot = (trace["topk_indices"] == eid).nonzero(as_tuple=True)
        gate = trace["topk_weights"][tok, slot].unsqueeze(-1)
        routed.index_add_(0, tok, gate * eo)
    shared = moe.shared_experts[0](x_flat)
print("\nrouted_output  (3, 2):\n", routed)
print("shared_output  (3, 2):\n", shared)
print("routed + shared:\n", routed + shared)

print("\noutput, explicit trace path:\n", out_trace)
print("output, fused dispatch_batched path:\n", out_fused)
print("max |trace - fused| =", (out_trace - out_fused).abs().max().item())
print("\naux_loss (Switch load-balancing), trace path :", aux_trace.item())
print("aux_loss (Switch load-balancing), fused path :", aux_fused.item())


# --- assert the by-hand results (doc Parts 2-4) --------------------------
expected_output = torch.tensor([[
    [1.666941, 0.866989],
    [1.484440, 2.018887],
    [4.808671, 3.346554],
]])
torch.testing.assert_close(out_trace, expected_output, atol=2e-6, rtol=0)
torch.testing.assert_close(out_fused, expected_output, atol=2e-6, rtol=0)

# the batched production path matches the explicit loop within 1e-5 (Req 5.2)
assert (out_trace - out_fused).abs().max().item() < 1e-5

# renormalized gate weights (Part 2, Step 4)
expected_topk_w = torch.tensor([
    [0.731059, 0.268941],
    [0.731059, 0.268941],
    [0.880797, 0.119203],
])
torch.testing.assert_close(trace["topk_weights"], expected_topk_w, atol=1e-5, rtol=0)

# routed / shared split (Step 6g, Step 7)
torch.testing.assert_close(
    routed,
    torch.tensor([[0.935883, 0.866989], [1.484440, 1.287828], [3.346554, 2.615495]]),
    atol=1e-5, rtol=0,
)
torch.testing.assert_close(
    shared,
    torch.tensor([[0.731059, 0.0], [0.0, 0.731059], [1.462117, 0.731059]]),
    atol=1e-5, rtol=0,
)

# each token picked exactly top_k=2 DISTINCT experts and the weights sum to 1
assert trace["topk_indices"].shape == (3, 2)
for row in trace["topk_indices"]:
    assert len(set(row.tolist())) == 2
torch.testing.assert_close(
    trace["topk_weights"].sum(-1), torch.ones(3), atol=1e-6, rtol=0
)

# dispatch counts: expert 0 -> 3 tokens, 1 -> 1, 2 -> 2, 3 -> 0 (idle, skipped)
assert moe.last_expert_counts.tolist() == [3, 1, 2, 0]
assert 3 not in trace["expert_outputs"]  # idle expert has no trace entry

# Requirement 2.5: the idle expert genuinely does not participate -- rebuild
# it with wildly different weights and confirm the output is byte-identical.
moe_perturbed = _build()
with torch.no_grad():
    moe_perturbed.routed_experts[3].w_gate.weight.mul_(-7.0).add_(3.0)
    moe_perturbed.routed_experts[3].w_down.weight.fill_(123.0)
moe_perturbed.eval()
out_perturbed, _ = moe_perturbed(X)
assert torch.equal(out_fused, out_perturbed), "idle expert 3 must not affect the output"

# Switch-Transformer auxiliary loss: alpha * N * sum_i f_i * P_i  (Part 3)
#   f = [1.0, 1/3, 2/3, 0.0]  (fraction of tokens with expert i in top-k)
#   P = column means of router_probs
f_i = torch.tensor([1.0, 1.0 / 3.0, 2.0 / 3.0, 0.0])
P_i = trace["router_probs"].mean(dim=0)
expected_aux = 0.01 * 4 * torch.sum(f_i * P_i)
torch.testing.assert_close(aux_trace, expected_aux, atol=1e-6, rtol=0)
assert abs(f_i.sum().item() - 2.0) < 1e-6          # sum_i f_i == top_k
assert abs(P_i.sum().item() - 1.0) < 1e-6          # sum_i P_i == 1
# the mild imbalance (expert 0 overloaded, expert 3 idle) sits above the
# perfect-balance floor, which is exactly alpha == 0.01
assert aux_trace.item() > 0.01

print("\nP_i (mean router prob per expert):", P_i.tolist())
print("aux_loss vs. perfect-balance floor (alpha=0.01):",
      f"{aux_trace.item():.6f} > 0.01")
print("\nAll assertions passed -- every number in 005-moe-layer.md is real.")
