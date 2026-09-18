"""Reproduces every hand-computed number in
`025-moe-routing-sigmoid-and-router-z-loss.md`.

Run it:  uv run python docs/math/verify_025_moe_routing_sigmoid_and_router_z_loss.py

This reuses doc 005's exact running example (same X, same router weight,
same four routed experts + one shared expert, same aux_loss_weight=0.01) --
the only thing this doc changes is which `ModelConfig` knobs are turned on:
`router_scoring` ("softmax" vs "sigmoid"), `router_z_loss_weight`, and
`aux_loss_weight`. Reusing the base example means any difference in the
printed numbers is caused *only* by the code this doc is about
(`score_and_select`'s sigmoid branch, `_add_router_z_loss`,
`build_balancer`'s config-driven weight) -- nothing else moved.

It builds several `MoELayer`s that share these pinned weights, runs BOTH
code paths (`forward_with_trace` and the batched `forward`) on each, prints
every intermediate, and `assert`s the hand-computed numbers the doc quotes.
No fabricated numbers.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from qwenmoe.config import ModelConfig
from qwenmoe.model.moe import LossFreeBalancer, MoELayer

torch.set_printoptions(precision=6, sci_mode=False)


# --- the model configuration and weights: identical to doc 005 -----------
def _cfg(**overrides) -> ModelConfig:
    base = dict(
        vocab_size=10,
        d_model=2,
        d_ff=3,
        n_heads=2,
        n_kv_heads=1,
        n_routed_experts=4,
        n_shared_experts=1,
        top_k=2,
        max_seq_len=8,
        activation="swiglu",
        moe_balancing="aux_loss",
        router_scoring="softmax",
        router_z_loss_weight=0.0,
        aux_loss_weight=0.01,
    )
    base.update(overrides)
    return ModelConfig(**base)


W_ROUTER = torch.tensor([[2.0, 1.0], [1.0, -1.0], [-1.0, 2.0], [0.0, 0.0]])

E0_GATE = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
E0_UP = torch.tensor([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
E0_DOWN = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])

E1_GATE = torch.tensor([[-1.0, 0.0], [2.0, 0.0], [1.0, 1.0]])
E1_UP = torch.tensor([[1.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
E1_DOWN = torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])

E2_GATE = torch.tensor([[0.0, 2.0], [1.0, -1.0], [0.0, 1.0]])
E2_UP = torch.tensor([[1.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
E2_DOWN = torch.tensor([[1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])

E3_GATE = torch.full((3, 2), 9.0)
E3_UP = torch.full((3, 2), 9.0)
E3_DOWN = torch.full((2, 3), 9.0)

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
    moe.eval()
    return moe


print("================ Part 1: score_and_select -- softmax vs sigmoid ================")

moe_sm = _build(router_scoring="softmax")
moe_sg = _build(router_scoring="sigmoid")

out_sm, aux_sm, trace_sm = moe_sm.forward_with_trace(X)
out_sg, aux_sg, trace_sg = moe_sg.forward_with_trace(X)
out_sm_fused, aux_sm_fused = moe_sm(X)
out_sg_fused, aux_sg_fused = moe_sg(X)

print("router_logits (identical for both -- scoring hasn't run yet):\n", trace_sm["router_logits"])
assert torch.equal(trace_sm["router_logits"], trace_sg["router_logits"])

print("\n-- softmax branch --")
print("routing_probs (softmax):\n", trace_sm["router_probs"])
print("topk_indices:\n", trace_sm["topk_indices"])
print("topk_weights:\n", trace_sm["topk_weights"])

print("\n-- sigmoid branch --")
sigmoid_affinities = torch.sigmoid(trace_sm["router_logits"])
print("raw affinities sigmoid(logits) (do NOT sum to 1):\n", sigmoid_affinities)
print("row sums:", sigmoid_affinities.sum(-1))
print("routing_probs (normalized sigmoid):\n", trace_sg["router_probs"])
print("topk_indices:\n", trace_sg["topk_indices"])
print("topk_weights:\n", trace_sg["topk_weights"])

# top-k SET is identical: both AuxLossBalancer.selection_scores branches
# return the raw logits unchanged, and unbiased selection only needs the
# logits' ranking (Requirement 4.2's monotonicity argument).
assert torch.equal(trace_sm["topk_indices"], trace_sg["topk_indices"])
expected_topk_indices = torch.tensor([[0, 1], [2, 0], [0, 2]])
torch.testing.assert_close(trace_sm["topk_indices"], expected_topk_indices)

# gate weights genuinely differ
assert not torch.allclose(trace_sm["topk_weights"], trace_sg["topk_weights"])
# but both still sum to 1 per token (Requirement 4.6)
torch.testing.assert_close(trace_sm["topk_weights"].sum(-1), torch.ones(3), atol=1e-6, rtol=0)
torch.testing.assert_close(trace_sg["topk_weights"].sum(-1), torch.ones(3), atol=1e-6, rtol=0)
# and routing_probs sums to 1 per token in both cases (the aux loss / trace contract)
torch.testing.assert_close(trace_sm["router_probs"].sum(-1), torch.ones(3), atol=1e-6, rtol=0)
torch.testing.assert_close(trace_sg["router_probs"].sum(-1), torch.ones(3), atol=1e-6, rtol=0)

expected_sg_topk_w = torch.tensor([
    [0.546449, 0.453551],
    [0.546449, 0.453551],
    [0.565785, 0.434215],
])
torch.testing.assert_close(trace_sg["topk_weights"], expected_sg_topk_w, atol=1e-5, rtol=0)

# same experts dispatched => identical per-expert raw outputs; only the gate differs
for eid in trace_sm["expert_outputs"]:
    torch.testing.assert_close(
        trace_sm["expert_outputs"][eid], trace_sg["expert_outputs"][eid], atol=1e-6, rtol=0
    )
print("\nexpert_outputs identical between the two scorings (only the gate changed):",
      all(torch.equal(trace_sm["expert_outputs"][e], trace_sg["expert_outputs"][e])
          for e in trace_sm["expert_outputs"]))

print("\noutput, softmax scoring:\n", out_sm)
print("output, sigmoid scoring:\n", out_sg)
print("max |output_softmax - output_sigmoid| =", (out_sm - out_sg).abs().max().item())

expected_out_sm = torch.tensor([[
    [1.666941, 0.866989], [1.484440, 2.018887], [4.808671, 3.346554],
]])
expected_out_sg = torch.tensor([[
    [1.807539, 1.462117], [1.294194, 1.878290], [5.133303, 3.671185],
]])
torch.testing.assert_close(out_sm, expected_out_sm, atol=2e-6, rtol=0)
torch.testing.assert_close(out_sg, expected_out_sg, atol=1e-5, rtol=0)

# each fused (dispatch_batched) path agrees with its own explicit trace path
print("\nmax |trace - fused| (softmax) =", (out_sm - out_sm_fused).abs().max().item())
print("max |trace - fused| (sigmoid) =", (out_sg - out_sg_fused).abs().max().item())
assert (out_sm - out_sm_fused).abs().max().item() < 1e-5
assert (out_sg - out_sg_fused).abs().max().item() < 1e-5

print("\naux_loss, softmax scoring:", aux_sm.item())
print("aux_loss, sigmoid scoring:", aux_sg.item())
# f_i (hard counts) is scoring-independent: same topk_indices
f_i = torch.tensor([1.0, 1.0 / 3.0, 2.0 / 3.0, 0.0])
P_i_sm = trace_sm["router_probs"].mean(dim=0)
P_i_sg = trace_sg["router_probs"].mean(dim=0)
print("P_i, softmax:", P_i_sm.tolist())
print("P_i, sigmoid:", P_i_sg.tolist())
expected_aux_sm = 0.01 * 4 * torch.sum(f_i * P_i_sm)
expected_aux_sg = 0.01 * 4 * torch.sum(f_i * P_i_sg)
torch.testing.assert_close(aux_sm, expected_aux_sm, atol=1e-6, rtol=0)
torch.testing.assert_close(aux_sg, expected_aux_sg, atol=1e-6, rtol=0)
torch.testing.assert_close(aux_sm, torch.tensor(0.030898), atol=1e-6, rtol=0)


print("\n================ Part 1b: LossFreeBalancer -- bias breaks the equivalence ================")

router_logits = trace_sm["router_logits"]  # same (3, 4) tensor throughout
balancer = LossFreeBalancer(n_routed_experts=4)
with torch.no_grad():
    balancer.bias.copy_(torch.tensor([0.0, 0.15, 0.0, 0.0]))

scores_sm_biased = balancer.selection_scores(router_logits, scoring="softmax")
scores_sg_biased = balancer.selection_scores(router_logits, scoring="sigmoid")
print("softmax(logits) + bias:\n", scores_sm_biased)
print("sigmoid(logits) + bias:\n", scores_sg_biased)

_, top2_sm_biased = scores_sm_biased.topk(2, dim=-1)
_, top2_sg_biased = scores_sg_biased.topk(2, dim=-1)
print("top-2 experts, softmax-biased selection:\n", top2_sm_biased)
print("top-2 experts, sigmoid-biased selection:\n", top2_sg_biased)

# token 2 (row index 2) genuinely disagrees on the SET of selected experts
set_sm_tok2 = set(top2_sm_biased[2].tolist())
set_sg_tok2 = set(top2_sg_biased[2].tolist())
print(f"token 2 selected set -- softmax-biased: {set_sm_tok2}, sigmoid-biased: {set_sg_tok2}")
assert set_sm_tok2 != set_sg_tok2
assert set_sm_tok2 == {0, 1}
assert set_sg_tok2 == {0, 2}
# tokens 0 and 1 are unaffected by this bias (same set either way)
for t in (0, 1):
    assert set(top2_sm_biased[t].tolist()) == set(top2_sg_biased[t].tolist())


print("\n================ Part 2: router z-loss ================")

logits = router_logits.clone()
mean_abs_logit = logits.abs().mean().item()
print("router_logits:\n", logits)
print("mean |logit| =", mean_abs_logit)
assert abs(mean_abs_logit - 1.0) < 1e-6

for scale in (1, 5, 20):
    scaled = logits * scale
    lse = torch.logsumexp(scaled.float(), dim=-1)
    z_loss_1em3 = 1e-3 * lse.pow(2).mean()
    print(f"scale x{scale:<2d}  logsumexp per row = {lse.tolist()}  "
          f"mean(logsumexp) = {lse.mean().item():.6f}  "
          f"z_loss(weight=1e-3) = {z_loss_1em3.item():.6e}")

lse_base = torch.logsumexp(logits.float(), dim=-1)
expected_lse_base = torch.tensor([2.440190, 2.440190, 3.210998])
torch.testing.assert_close(lse_base, expected_lse_base, atol=1e-5, rtol=0)
z_loss_base = 1e-3 * lse_base.pow(2).mean()
torch.testing.assert_close(z_loss_base, torch.tensor(7.406520e-03), atol=1e-6, rtol=0)

lse_5x = torch.logsumexp((logits * 5).float(), dim=-1)
z_loss_5x = 1e-3 * lse_5x.pow(2).mean()
lse_20x = torch.logsumexp((logits * 20).float(), dim=-1)
z_loss_20x = 1e-3 * lse_20x.pow(2).mean()
print("z_loss ratio 5x / base  :", (z_loss_5x / z_loss_base).item())
print("z_loss ratio 20x / base :", (z_loss_20x / z_loss_base).item())
# logsumexp is NOT linear in a uniform logit scale (it interpolates between
# "sum of N equal terms" at small scale and "the single max term" at large
# scale -- see the derivation below), so the z-loss does not simply scale by
# scale^2. It is, however, strictly increasing, and at scale=20 it has
# collapsed (in fp32) onto the exact closed form scale * max(logit):
torch.testing.assert_close(lse_20x, torch.tensor([40.0, 40.0, 60.0]), atol=1e-6, rtol=0)
assert z_loss_base.item() < z_loss_5x.item() < z_loss_20x.item()

# -- weight = 0 is a true no-op --------------------------------------------
moe_z0 = _build(router_scoring="softmax", router_z_loss_weight=0.0, aux_loss_weight=0.01)
_, aux_z0, _ = moe_z0.forward_with_trace(X)
torch.testing.assert_close(aux_z0, aux_sm, atol=0, rtol=0)
print("\nrouter_z_loss_weight=0 leaves aux_loss byte-identical:", aux_z0.item(), "==", aux_sm.item())

# -- weight > 0 adds exactly the z-loss term onto AuxLossBalancer's term ----
moe_z1 = _build(router_scoring="softmax", router_z_loss_weight=1e-3, aux_loss_weight=0.01)
_, aux_z1, trace_z1 = moe_z1.forward_with_trace(X)
z_only = 1e-3 * torch.logsumexp(trace_z1["router_logits"].float(), dim=-1).pow(2).mean()
print("aux_loss with weight=1e-3 router z-loss:", aux_z1.item())
print("aux_loss without (Part 1) + z_loss computed alone:", (aux_sm + z_only).item())
torch.testing.assert_close(aux_z1, aux_sm + z_only, atol=1e-6, rtol=0)

# -- loss-free balancer: aux_loss is otherwise None, so z-loss IS the term -
moe_lf = _build(router_scoring="sigmoid", moe_balancing="loss_free", router_z_loss_weight=1e-3)
_, aux_lf, trace_lf = moe_lf.forward_with_trace(X)
z_only_lf = 1e-3 * torch.logsumexp(trace_lf["router_logits"].float(), dim=-1).pow(2).mean()
print("loss_free balancer + router z-loss: aux_loss =", aux_lf.item(),
      " (== z-loss alone:", z_only_lf.item(), ")")
torch.testing.assert_close(aux_lf, z_only_lf, atol=1e-6, rtol=0)


print("\n================ Part 3: config-driven aux_loss_weight ================")

moe_alpha_low = _build(router_scoring="softmax", aux_loss_weight=0.01)
moe_alpha_high = _build(router_scoring="softmax", aux_loss_weight=0.05)
_, aux_low, _ = moe_alpha_low.forward_with_trace(X)
_, aux_high, _ = moe_alpha_high.forward_with_trace(X)
print("aux_loss_weight=0.01:", aux_low.item())
print("aux_loss_weight=0.05:", aux_high.item())
print("ratio:", (aux_high / aux_low).item())
torch.testing.assert_close(aux_high, aux_low * 5.0, atol=1e-6, rtol=0)
assert moe_alpha_low.balancer.loss_weight == 0.01
assert moe_alpha_high.balancer.loss_weight == 0.05

print("\nAll assertions passed -- every number in "
      "025-moe-routing-sigmoid-and-router-z-loss.md is real.")
