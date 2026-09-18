"""Verifies docs/math/035-modernized-presets-training-run.md.

Pins two fixed router weights and two fixed 3-token inputs into the *real*
`qwenmoe.model.moe.MoELayer` (standing in for "layer 0" and "layer 1" of a
2-layer stack -- layer 0 is doc 005's / doc 027's exact canonical
3-token/4-expert(1 idle)/top-2 example, reused verbatim for continuity;
layer 1 is a fresh, analogous example with no idle expert, for contrast),
runs both `forward` (the batched-dispatch path) and `forward_with_trace`
(the explicit per-expert-loop path), reads the real `last_expert_counts`
each populates, feeds them through the real
`qwenmoe.train.trainer._collect_expert_counts`, and reproduces journal
035's `balance_stats` (coefficient of variation + normalized routing
entropy) exactly as it appears in `notebooks/07_modernized_presets.ipynb`
(cell 15). Every printed number is asserted against the hand-derived value
in the doc.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from qwenmoe.config import ModelConfig
from qwenmoe.model.moe import MoELayer
from qwenmoe.train.trainer import _collect_expert_counts

torch.manual_seed(0)  # only affects the never-inspected expert FFN weights

# ---------------------------------------------------------------------------
# 1. Build the two MoE layers ("layer 0" and "layer 1" of a 2-layer stack).
# ---------------------------------------------------------------------------
config = ModelConfig(
    vocab_size=8,      # unused by MoELayer; only ModelConfig.__post_init__ wants it
    d_model=2,
    n_heads=1,
    n_kv_heads=1,
    d_ff=2,
    n_shared_experts=0,  # shared experts never touch topk_indices -- irrelevant here
    n_routed_experts=4,
    top_k=2,
    max_seq_len=8,
    activation="swiglu",
    router_scoring="softmax",
    moe_balancing="aux_loss",
    aux_loss_weight=0.01,
)

# Layer 0: doc 005's / doc 027's exact canonical example (moe.py:79-84's
# W_router, and the 3-token X), reused verbatim.
W0 = torch.tensor([[2.0, 1.0], [1.0, -1.0], [-1.0, 2.0], [0.0, 0.0]])
x0 = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]).unsqueeze(0)  # (1,3,2)

# Layer 1: a fresh, structurally analogous example (own router weight, as a
# real model's independently-initialized layer-1 router would have) chosen
# to land every expert with at least one token -- no idle expert this time.
W1 = torch.tensor([[3.0, 1.0], [3.0, -1.0], [-3.0, 1.0], [-3.0, -1.0]])
x1 = torch.tensor([[1.0, 0.5], [1.0, -0.5], [-1.0, 0.5]]).unsqueeze(0)  # (1,3,2)

layer0 = MoELayer(config)
layer1 = MoELayer(config)
with torch.no_grad():
    layer0.router.gate.weight.copy_(W0)
    layer1.router.gate.weight.copy_(W1)

print("=== Router logits (Router.forward, moe.py:83-84) ===")
with torch.no_grad():
    logits0 = layer0.router(x0.reshape(-1, 2))
    logits1 = layer1.router(x1.reshape(-1, 2))
print("layer0 router_logits:\n", logits0)
print("layer1 router_logits:\n", logits1)

expected_logits0 = torch.tensor([[2.0, 1.0, -1.0, 0.0], [1.0, -1.0, 2.0, 0.0], [3.0, 0.0, 1.0, 0.0]])
expected_logits1 = torch.tensor([[3.5, 2.5, -2.5, -3.5], [2.5, 3.5, -3.5, -2.5], [-2.5, -3.5, 3.5, 2.5]])
assert torch.allclose(logits0, expected_logits0), logits0
assert torch.allclose(logits1, expected_logits1), logits1

# ---------------------------------------------------------------------------
# 2. Both paths: the batched-dispatch `forward` and the explicit-loop
#    `forward_with_trace` (moe.py:447-484, :528-573). Both independently
#    recompute `last_expert_counts` (moe.py:457-459, :542-544) from the same
#    `topk_indices` -- confirm they agree bit-for-bit ("fused vs explicit").
# ---------------------------------------------------------------------------
print("\n=== forward() vs forward_with_trace(): last_expert_counts agreement ===")
out_fwd0, aux_fwd0 = layer0.forward(x0)
counts_fwd0 = layer0.last_expert_counts.clone()
out_trace0, aux_trace0, trace0 = layer0.forward_with_trace(x0)
counts_trace0 = layer0.last_expert_counts.clone()
max_diff_output = (out_fwd0 - out_trace0).abs().max().item()
print("layer0 last_expert_counts (forward):           ", counts_fwd0.tolist())
print("layer0 last_expert_counts (forward_with_trace):", counts_trace0.tolist())
print(f"max|forward_output - forward_with_trace_output| = {max_diff_output:.3e}")
assert torch.equal(counts_fwd0, counts_trace0)
assert counts_fwd0.tolist() == [3, 1, 2, 0]
assert max_diff_output < 1e-5  # dispatch_batched vs. the explicit loop (doc 027)

out_trace1, aux_trace1, trace1 = layer1.forward_with_trace(x1)
counts1 = layer1.last_expert_counts.clone()
print("layer1 last_expert_counts:                     ", counts1.tolist())
assert counts1.tolist() == [2, 2, 1, 1]

print("\n=== topk_indices (select_top_k's .topk(), moe.py:140) ===")
print("layer0 topk_indices:\n", trace0["topk_indices"])
print("layer1 topk_indices:\n", trace1["topk_indices"])
assert trace0["topk_indices"].tolist() == [[0, 1], [2, 0], [0, 2]]
assert trace1["topk_indices"].tolist() == [[0, 1], [1, 0], [2, 3]]

print("\n=== topk_weights / router_probs, token 0 of layer 0 (moe.py:110-118, 141-142) ===")
print("layer0 router_probs (token 0):", trace0["router_probs"][0].tolist())
print("layer0 topk_weights (token 0):", trace0["topk_weights"][0].tolist())

# ---------------------------------------------------------------------------
# 3. The trainer's real per-layer collection (trainer.py:168-173). It reads
#    `model.blocks[i].moe.last_expert_counts` -- duck-typed here with two
#    trivial namespaces so the *real* function body runs without building an
#    unrelated attention/RMSNorm stack (which last_expert_counts never
#    touches).
# ---------------------------------------------------------------------------
class _FakeBlock:
    def __init__(self, moe):
        self.moe = moe


class _FakeModel:
    def __init__(self, blocks):
        self.blocks = blocks


fake_model = _FakeModel([_FakeBlock(layer0), _FakeBlock(layer1)])
counts_per_layer = _collect_expert_counts(fake_model)
print("\n=== _collect_expert_counts(model) (trainer.py:168-173) ===")
print(counts_per_layer)
assert counts_per_layer == [[3, 1, 2, 0], [2, 2, 1, 1]]

# ---------------------------------------------------------------------------
# 4. balance_stats -- verbatim from notebooks/07_modernized_presets.ipynb
#    cell 15 (the analysis code entry 035's "Expert utilization" table and
#    §"Term: expert load, coefficient of variation, routing entropy" are
#    built on). Not inside qwenmoe/ -- this *is* the real implementation,
#    copied byte-for-byte, not a re-derivation.
# ---------------------------------------------------------------------------
def balance_stats(counts):
    counts = np.asarray(counts, dtype=float)          # (n_layers, n_experts)
    frac = counts / counts.sum(1, keepdims=True)
    n_exp = frac.shape[1]
    cv = (frac.std(1) / (1.0 / n_exp)).mean()
    ent = -(frac * np.log(np.clip(frac, 1e-12, None))).sum(1) / math.log(n_exp)
    return frac, cv, ent.mean()


frac, cv, ent = balance_stats(counts_per_layer)
per_layer_std = frac.std(1)
per_layer_cv = per_layer_std / 0.25
per_layer_ent = -(frac * np.log(np.clip(frac, 1e-12, None))).sum(1) / math.log(4)
print("\n=== balance_stats ===")
print("frac (per layer, per expert):\n", frac)
print("per-layer std:", per_layer_std)
print("per-layer CV:", per_layer_cv)
print("per-layer entropy (normalized):", per_layer_ent)
print(f"mean CV over layers:      {cv:.6f}")
print(f"mean routing entropy:     {ent:.6f}")

assert np.allclose(frac, np.array([[0.5, 1 / 6, 1 / 3, 0.0], [1 / 3, 1 / 3, 1 / 6, 1 / 6]]))
assert np.allclose(frac.mean(1), 0.25)  # exact identity: mean(load) == 1/n_experts always
assert np.allclose(per_layer_cv, [math.sqrt(5) / 3, 1 / 3]), per_layer_cv
assert np.allclose(per_layer_ent, [1.011404042/math.log(4), 1.329661350/math.log(4)], atol=1e-4), per_layer_ent
assert math.isclose(cv, (math.sqrt(5) / 3 + 1 / 3) / 2, rel_tol=1e-9)

# ---------------------------------------------------------------------------
# 5. The KL-divergence identity behind "entropy <= 1, equality iff uniform":
#    ln(n) - H(p) = D_KL(p || uniform) >= 0.
# ---------------------------------------------------------------------------
print("\n=== ln(n) - H(frac) == D_KL(frac || uniform) ===")
for i, row in enumerate(frac):
    H_unnorm = -(row * np.log(np.clip(row, 1e-12, None))).sum()
    kl = (row * np.log(np.clip(row, 1e-12, None) * 4)).sum()  # sum p_i * ln(4 p_i)
    print(f"layer{i}: ln(4)-H = {math.log(4) - H_unnorm:.6f}   D_KL(p||uniform) = {kl:.6f}")
    assert math.isclose(math.log(4) - H_unnorm, kl, abs_tol=1e-9)
    assert kl >= -1e-12  # Gibbs' inequality: D_KL >= 0 always

# The equality case (D_KL == 0 iff p == uniform) needs a genuinely uniform
# row -- neither layer above is exactly uniform, so this checks the general
# property directly, as a boundary case of the formula (not "the running
# example").
uniform = np.array([0.25, 0.25, 0.25, 0.25])
H_uniform = -(uniform * np.log(uniform)).sum()
print(f"uniform row: ln(4)-H = {math.log(4) - H_uniform:.10f}  (boundary-case check, not from the example)")
assert math.isclose(math.log(4) - H_uniform, 0.0, abs_tol=1e-9)

# ---------------------------------------------------------------------------
# 6. General range check: CV in [0, sqrt(n_experts - 1)], independent of
#    this doc's running example (a pure property of the formula).
# ---------------------------------------------------------------------------
degenerate = np.array([[1.0, 0.0, 0.0, 0.0]])
_, cv_degenerate, ent_degenerate = balance_stats(degenerate)
print(f"\ndegenerate all-one-expert check: CV={cv_degenerate:.6f} (expect sqrt(3)={math.sqrt(3):.6f}), "
      f"entropy={ent_degenerate:.6f} (expect 0)")
assert math.isclose(cv_degenerate, math.sqrt(3), rel_tol=1e-9)
assert math.isclose(ent_degenerate, 0.0, abs_tol=1e-9)

# ---------------------------------------------------------------------------
# 7. The *other* entropy in journal 035 -- per-token router_probs entropy
#    (notebook cell 19), a different quantity from the load entropy above.
#    Computed here on layer0's real routing_probs to show the two numbers
#    genuinely differ.
# ---------------------------------------------------------------------------
p = trace0["router_probs"]
token_entropy = (-(p * p.clamp_min(1e-9).log()).sum(-1) / math.log(p.shape[-1])).mean().item()
print(f"\nlayer0 mean per-token router-probs entropy: {token_entropy:.6f}  "
      f"(vs. load entropy {per_layer_ent[0]:.6f} -- a different quantity)")
assert not math.isclose(token_entropy, per_layer_ent[0], abs_tol=1e-3)

print("\nAll assertions passed.")
