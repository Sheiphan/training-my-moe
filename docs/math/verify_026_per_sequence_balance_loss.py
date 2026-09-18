"""Reproduces every hand-computed number in `026-per-sequence-balance-loss.md`.

Run it:  uv run python docs/math/verify_026_per_sequence_balance_loss.py

Builds a tiny (d_model=2, n_routed_experts=2, top_k=1, n_shared_experts=0)
loss-free-balanced `MoELayer`, pins the router weight to the identity matrix
so each expert's logit *is* one input coordinate, and feeds it a batch of 2
sequences of 3 tokens each -- sequence 0's tokens all favor expert 0,
sequence 1's all favor expert 1. Summed over the whole batch that is
*perfectly* balanced (3 tokens to each expert); *within* either sequence
it is total collapse -- exactly the pathology DeepSeek-V3's complementary
per-sequence loss exists to catch.

Runs the real `MoELayer.forward_with_trace`, `LossFreeBalancer.
sequence_balance_loss`, and (for contrast, on the identical routing_probs
/topk_indices) `AuxLossBalancer.aux_loss`, then asserts:

  * every by-hand intermediate (router logits, softmax probs, top-1
    picks, f_i, P_i, the per-sequence dot product, the batch mean, the
    final weighted scalar) to float32 precision;
  * the batch-wide Switch view sits exactly at its balanced floor while
    the per-sequence view does not (the whole point of the feature);
  * the no-op gating in both directions (`seq_balance_loss_weight == 0`;
    an `aux_loss`-balanced layer ignores the weight entirely);
  * gradient reaches the router through P_i (checked against the exact
    closed-form partial derivative) and never touches the loss-free
    bias buffer;
  * a from-scratch counterexample showing the popular "always >= 1" folk
    claim about this term is not, in fact, a theorem.

No fabricated numbers -- this is the source of truth the prose quotes.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from qwenmoe.config import ModelConfig
from qwenmoe.model.moe import AuxLossBalancer, LossFreeBalancer, MoELayer

torch.set_printoptions(precision=6, sci_mode=False)
torch.manual_seed(0)


# --- the model configuration (Part 1 of the doc) -------------------------
def _cfg(**overrides) -> ModelConfig:
    base = dict(
        vocab_size=10,
        d_model=2,
        d_ff=2,
        n_heads=1,
        n_kv_heads=1,
        n_routed_experts=2,
        n_shared_experts=0,
        top_k=1,
        max_seq_len=8,
        activation="swiglu",
        moe_balancing="loss_free",
        router_scoring="softmax",
        router_z_loss_weight=0.0,
        aux_loss_weight=0.01,
        seq_balance_loss_weight=0.01,
    )
    base.update(overrides)
    return ModelConfig(**base)


# --- the hand-picked weight (Part 1 of the doc) ---------------------------
# router: logit for expert e at token x is x . W_router[e]. The identity
# matrix makes expert 0's logit *equal* the token's first coordinate and
# expert 1's logit equal its second -- no arithmetic needed to read off
# which expert a token favors.
W_ROUTER = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

# batch of 2 sequences, seq_len = 3 tokens, d_model = 2.
# sequence 0: every token's first coordinate > second -> favors expert 0,
#             by gaps 1, 2, 3 (so P_i averages three DIFFERENT probabilities).
# sequence 1: the exact mirror -> every token favors expert 1.
X = torch.tensor([
    [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],   # sequence 0
    [[0.0, 1.0], [0.0, 2.0], [0.0, 3.0]],   # sequence 1
])  # (2, 3, 2)


def _build(**cfg_overrides) -> MoELayer:
    moe = MoELayer(_cfg(**cfg_overrides))
    with torch.no_grad():
        moe.router.gate.weight.copy_(W_ROUTER)
    return moe


moe = _build()
moe.eval()  # forward_with_trace never calls balancer.update() anyway

out, aux_loss, trace = moe.forward_with_trace(X)

print("================ running example: sequence_balance_loss ================")
print("x  (batch, seq_len, d_model):\n", X)
print("\nrouter_logits  (n_tokens, n_experts):\n", trace["router_logits"])
print("router_probs  (softmax over 2 experts == sigmoid of the logit gap):\n",
      trace["router_probs"])
print("topk_indices  (n_tokens, top_k=1):\n", trace["topk_indices"].view(-1).tolist())

# --- Part 2: hand-reproduce LossFreeBalancer.sequence_balance_loss --------
# (moe.py:270-280, the exact code, just with intermediate prints)
n_experts = moe.n_routed_experts                                   # 2
seq_len = X.shape[1]                                                # 3
top_k = moe.top_k                                                   # 1
batch = trace["router_probs"].shape[0] // seq_len                   # 2

dispatched_flat = F.one_hot(trace["topk_indices"], num_classes=n_experts).sum(dim=1)
print("\ndispatched, flat (n_tokens, n_experts):\n", dispatched_flat)
dispatched = dispatched_flat.reshape(batch, seq_len, n_experts).float()
probs = trace["router_probs"].reshape(batch, seq_len, n_experts)
print("dispatched, reshaped (batch, seq_len, n_experts):\n", dispatched)
print("probs, reshaped (batch, seq_len, n_experts):\n", probs)

f_i = (n_experts / (top_k * seq_len)) * dispatched.sum(dim=1)       # (batch, n_experts)
p_i = probs.mean(dim=1)                                              # (batch, n_experts)
per_sequence = (f_i * p_i).sum(dim=-1)                                # (batch,)
seq_balance_loss = 0.01 * per_sequence.mean()

print("\nf_i  (batch, n_experts) -- realized load, sums to n_experts per row:\n", f_i)
print("p_i  (batch, n_experts) -- intended load, sums to 1 per row:\n", p_i)
print("per_sequence = sum_i f_i * p_i  (batch,):\n", per_sequence)
print("mean over batch:", per_sequence.mean().item())
print("seq_balance_loss = 0.01 * mean:", seq_balance_loss.item())

print("\naux_loss returned by forward_with_trace:", aux_loss.item())

# --- contrast: the batch-wide Switch view, on the IDENTICAL routing -------
switch = AuxLossBalancer(n_experts, loss_weight=0.01)
switch_loss = switch.aux_loss(trace["router_probs"], trace["topk_indices"])
print("\nbatch-wide Switch aux_loss on the same routing_probs/topk_indices:",
      switch_loss.item())
print("ratio (per-sequence / batch-wide):", (aux_loss / switch_loss).item())


# ============================ assertions ==================================

# router logits ARE the input coordinates (identity router weight)
torch.testing.assert_close(trace["router_logits"], X.reshape(-1, 2), atol=0, rtol=0)

# softmax over 2 classes == sigmoid of the logit gap; check both formulas agree
gaps = trace["router_logits"][:, 0] - trace["router_logits"][:, 1]
torch.testing.assert_close(trace["router_probs"][:, 0], torch.sigmoid(gaps), atol=1e-6, rtol=0)

expected_probs = torch.tensor([
    [0.731059, 0.268941],
    [0.880797, 0.119203],
    [0.952574, 0.047426],
    [0.268941, 0.731059],
    [0.119203, 0.880797],
    [0.047426, 0.952574],
])
torch.testing.assert_close(trace["router_probs"], expected_probs, atol=1e-5, rtol=0)

# every token in sequence 0 picks expert 0; every token in sequence 1 picks expert 1
assert trace["topk_indices"].view(-1).tolist() == [0, 0, 0, 1, 1, 1]

# f_i, p_i (Part 2)
torch.testing.assert_close(f_i, torch.tensor([[2.0, 0.0], [0.0, 2.0]]), atol=1e-6, rtol=0)
expected_p_i = torch.tensor([[0.854810, 0.145190], [0.145190, 0.854810]])
torch.testing.assert_close(p_i, expected_p_i, atol=1e-5, rtol=0)
assert torch.allclose(f_i.sum(dim=-1), torch.full((batch,), float(n_experts)), atol=1e-6)
assert torch.allclose(p_i.sum(dim=-1), torch.ones(batch), atol=1e-6)

# per-sequence dot product and the final scalar
torch.testing.assert_close(per_sequence, torch.tensor([1.709620, 1.709620]), atol=1e-5, rtol=0)
torch.testing.assert_close(seq_balance_loss, torch.tensor(0.0170962), atol=1e-5, rtol=0)

# the real module's LossFreeBalancer.sequence_balance_loss matches the hand version exactly
direct = moe.balancer.sequence_balance_loss(
    trace["router_probs"], trace["topk_indices"], seq_len, top_k, 0.01
)
torch.testing.assert_close(direct, seq_balance_loss, atol=0, rtol=0)

# aux_loss returned by forward_with_trace IS the seq-balance term (loss-free's own
# aux_loss() is None, router_z_loss_weight is 0 -- nothing else is in the sum)
torch.testing.assert_close(aux_loss, seq_balance_loss, atol=0, rtol=0)

# the batch-wide Switch view sits EXACTLY at its balanced floor (alpha_switch = 0.01)
# even though nothing here is actually balanced per-sequence
torch.testing.assert_close(switch_loss, torch.tensor(0.01), atol=1e-6, rtol=0)
assert (aux_loss / switch_loss).item() > 1.7  # the per-sequence term is NOT fooled


# --- no-op gating, direction 1: seq_balance_loss_weight == 0 --------------
moe_off = _build(seq_balance_loss_weight=0.0)
moe_off.eval()
_, aux_off, _ = moe_off.forward_with_trace(X)
assert aux_off is None, "weight == 0 must leave aux_loss untouched (still None)"

# --- no-op gating, direction 2: an aux_loss-balanced layer ignores the weight
moe_switch = _build(moe_balancing="aux_loss", aux_loss_weight=0.01, seq_balance_loss_weight=0.0)
with torch.no_grad():
    moe_switch.router.gate.weight.copy_(W_ROUTER)
moe_switch.eval()
_, aux_switch_lo, _ = moe_switch.forward_with_trace(X)
moe_switch.seq_balance_loss_weight = 0.05  # crank it up post-hoc -- must change nothing
_, aux_switch_hi, _ = moe_switch.forward_with_trace(X)
assert torch.equal(aux_switch_lo, aux_switch_hi), (
    "AuxLossBalancer has no sequence_balance_loss method -- the weight must be a pure no-op"
)
torch.testing.assert_close(aux_switch_lo, switch_loss, atol=1e-6, rtol=0)
assert getattr(moe_switch.balancer, "sequence_balance_loss", None) is None


# --- gradient: flows through P_i into the router, never into the bias ----
probs_leaf = trace["router_probs"].detach().clone().requires_grad_(True)
loss_from_probs = moe.balancer.sequence_balance_loss(
    probs_leaf, trace["topk_indices"], seq_len, top_k, 0.01
)
loss_from_probs.backward()
# closed form: d(loss)/d(probs[b,t,i]) = weight * f_i[b,i] / (seq_len * batch)
expected_grad = (0.01 * f_i.detach() / (seq_len * batch)).unsqueeze(1).expand(-1, seq_len, -1)
expected_grad = expected_grad.reshape(batch * seq_len, n_experts)
torch.testing.assert_close(probs_leaf.grad, expected_grad, atol=1e-7, rtol=0)
print("\nd(seq_balance_loss)/d(routing_probs), row 0:", probs_leaf.grad[0].tolist())
print("closed-form weight*f_i/(T*batch), row 0:      ", expected_grad[0].tolist())

moe.zero_grad()
out2, aux_loss2, _ = moe.forward_with_trace(X)
aux_loss2.backward()
assert moe.router.gate.weight.grad is not None
assert moe.router.gate.weight.grad.abs().sum().item() > 0
assert moe.balancer.bias.grad is None, "the loss-free bias buffer must never receive a gradient"
assert not moe.balancer.bias.requires_grad
print("\nrouter.gate.weight.grad (from the full aux_loss):\n", moe.router.gate.weight.grad)
print("balancer.bias.grad:", moe.balancer.bias.grad, " (bias.requires_grad:",
      moe.balancer.bias.requires_grad, ")")


# --- a from-scratch counterexample to "Sum f_i P_i is always >= 1" -------
# N=2, top_k=1, T=3, one sequence: two tokens where expert 0 barely edges out
# expert 1 (0.6 vs 0.4), one token where expert 1 wins by a landslide (0.99
# vs 0.01). Expert 1 is picked only once (low f) but its rare win is huge
# (pulls its mean P up); expert 0 is picked twice (high f) on modest margins
# (its mean P stays modest). That anti-correlation between f and P is
# exactly what the covariance identity below says can push the sum under 1.
cf_probs = torch.tensor([[0.6, 0.4], [0.6, 0.4], [0.01, 0.99]])
cf_topk = torch.tensor([[0], [0], [1]])
cf_balancer = LossFreeBalancer(2)
cf_value = cf_balancer.sequence_balance_loss(cf_probs, cf_topk, seq_len=3, top_k=1, weight=1.0)
print("\ncounterexample: sum_i f_i P_i =", cf_value.item(), "(< 1)")

cf_f = (2 / (1 * 3)) * F.one_hot(cf_topk, num_classes=2).sum(dim=1).sum(dim=0).float()
cf_P = cf_probs.mean(dim=0)
cf_mean_f, cf_mean_P = cf_f.mean().item(), cf_P.mean().item()  # 1.0 and 0.5 respectively
cf_cov = ((cf_f - cf_f.mean()) * (cf_P - cf_P.mean())).mean().item()
print("f =", cf_f.tolist(), " P =", cf_P.tolist())
print("covariance identity check: 1 + n_experts*cov =",
      1 + 2 * cf_cov, " vs. direct value", cf_value.item())

assert cf_value.item() < 1.0, "counterexample must actually breach the naive floor"
torch.testing.assert_close(cf_value, torch.tensor(1 + 2 * cf_cov), atol=1e-6, rtol=0)


print("\nAll assertions passed -- every number in 026-per-sequence-balance-loss.md is real.")
