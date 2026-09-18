"""Reproduces every hand-computed number in `027-batched-expert-computation.md`.

Run it:  uv run python docs/math/verify_027_batched_expert_computation.py

Reuses 005-moe-layer.md's exact running example (d_model=2, d_ff=3,
n_routed_experts=4, top_k=2, n_shared_experts=1, swiglu experts, aux-loss
balancing, softmax routing -- same X, same router weight, same five expert
FFNs) because that config already has every property `dispatch_batched`
needs to exercise: an idle expert (3, zero tokens -- never stacked, never
gradient) and two *unequal* active group sizes (expert 0: 3 tokens,
expert 1: 1, expert 2: 2) that force real padding once they're laid into
a shared (K, G) grid.

Builds the real `MoELayer`, pins the same weights, then:

  * calls `forward_with_trace` -- the explicit per-expert loop -- as
    ground truth (identical numbers to 005-moe-layer.md);
  * re-executes `dispatch_batched`'s own line sequence (moe.py:322-406)
    directly on the same pinned tensors, printing and asserting every
    named intermediate: `pair_expert`/`pair_token`/`pair_weight`, the
    `bincount`/`active`/`group_size` grid sizing, the stacked weights
    (excluding the idle expert), the stable `argsort`, the row/slot
    index derivation, the padded grid, the two `bmm`s, both masks, and
    the `index_add_`;
  * calls the REAL `dispatch_batched` function (not a reimplementation)
    and asserts it matches both the by-hand reproduction and
    `forward_with_trace`'s routed output, well inside the project's own
    1e-5 tolerance;
  * runs a full backward pass through `MoELayer.forward` and asserts the
    idle expert's parameters get `grad is None` (Requirement 5.3), while
    every active expert's parameters do not.

No fabricated numbers -- this is the source of truth the prose quotes.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from qwenmoe.config import ModelConfig
from qwenmoe.model.moe import MoELayer, _stack_expert_weights, dispatch_batched

torch.set_printoptions(precision=6, sci_mode=False)


# --- the model configuration (identical to 005-moe-layer.md, Part 1) -----
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
        aux_loss_weight=0.01,
    )
    base.update(overrides)
    return ModelConfig(**base)


# --- the hand-picked weights (identical to 005-moe-layer.md, Part 1) -----
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

# expert 3 -- gets ZERO tokens this batch. Loud, distinctive weights, so a
# perturbation check (below) can confirm the output never moves.
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


def _build() -> MoELayer:
    moe = MoELayer(_cfg())
    with torch.no_grad():
        moe.router.gate.weight.copy_(W_ROUTER)
    _pin(moe.routed_experts[0], E0_GATE, E0_UP, E0_DOWN)
    _pin(moe.routed_experts[1], E1_GATE, E1_UP, E1_DOWN)
    _pin(moe.routed_experts[2], E2_GATE, E2_UP, E2_DOWN)
    _pin(moe.routed_experts[3], E3_GATE, E3_UP, E3_DOWN)
    _pin(moe.shared_experts[0], ES_GATE, ES_UP, ES_DOWN)
    return moe


moe = _build()
moe.eval()

# ============================================================
# Part 1 -- reuse 005's routing (Steps 1-4 of that doc)
# ============================================================
out_trace, aux_trace, trace = moe.forward_with_trace(X)
router_logits = trace["router_logits"]
routing_probs = trace["router_probs"]
topk_indices = trace["topk_indices"]
topk_weights = trace["topk_weights"]

print("================ running example: dispatch_batched ================")
print("x (batch, seq, d_model):\n", X)
print("topk_indices (n_tokens, top_k):\n", topk_indices)
print("topk_weights (n_tokens, top_k):\n", topk_weights)
print("last_expert_counts:", moe.last_expert_counts.tolist())

torch.testing.assert_close(
    topk_indices, torch.tensor([[0, 1], [2, 0], [0, 2]]), atol=0, rtol=0
)
torch.testing.assert_close(
    topk_weights,
    torch.tensor([[0.731059, 0.268941], [0.731059, 0.268941], [0.880797, 0.119203]]),
    atol=1e-5, rtol=0,
)
assert moe.last_expert_counts.tolist() == [3, 1, 2, 0]

# ============================================================
# Part 2 -- dispatch_batched's own line sequence, moe.py:349-405,
# re-executed here (same ops, same real tensors) so every intermediate
# can be printed and checked. `activation="swiglu"` -> gated=True.
# ============================================================
x_flat = X.reshape(-1, 2)
n_tokens, d_model = x_flat.shape        # 3, 2
top_k = topk_indices.shape[1]           # 2
n_experts = len(moe.routed_experts)     # 4
device = x_flat.device

pair_expert = topk_indices.reshape(-1)                                    # moe.py:355
P = pair_expert.numel()                                                   # moe.py:356
pair_token = torch.arange(n_tokens, device=device).repeat_interleave(top_k)  # moe.py:359
pair_weight = topk_weights.reshape(-1)                                    # moe.py:360

print("\n--- pairing (moe.py:355-360) ---")
print("pair_expert:", pair_expert.tolist())
print("pair_token :", pair_token.tolist())
print("pair_weight:", [round(v, 6) for v in pair_weight.tolist()])

assert pair_expert.tolist() == [0, 1, 2, 0, 0, 2]
assert pair_token.tolist() == [0, 0, 1, 1, 2, 2]
assert P == 6

counts = torch.bincount(pair_expert, minlength=n_experts)                 # moe.py:362
active = torch.nonzero(counts, as_tuple=True)[0]                          # moe.py:363
active_counts = counts[active]                                            # moe.py:364
num_active = active.numel()                                               # moe.py:365
group_size = int(active_counts.max().item())                              # moe.py:366

print("\n--- sizing the grid (moe.py:362-366) ---")
print("counts (per expert 0..3):", counts.tolist())
print("active experts (K):", active.tolist())
print("active_counts:", active_counts.tolist())
print("group_size (G):", group_size)

assert counts.tolist() == [3, 1, 2, 0]
assert active.tolist() == [0, 1, 2]
assert active_counts.tolist() == [3, 1, 2]
assert num_active == 3
assert group_size == 3
assert active_counts.sum().item() == P  # every pair belongs to exactly one active expert

expert_ids = active.tolist()                                              # moe.py:369
w_first = _stack_expert_weights(moe.routed_experts, expert_ids, "w_gate")  # moe.py:370
w_second = _stack_expert_weights(moe.routed_experts, expert_ids, "w_up")   # moe.py:371
w_out = _stack_expert_weights(moe.routed_experts, expert_ids, "w_down")    # moe.py:372

print("\n--- stacked active-expert weights (moe.py:310-319, 370-372) ---")
print("w_first (w_gate) shape:", tuple(w_first.shape), "-- experts", expert_ids)
print("w_second (w_up) shape :", tuple(w_second.shape))
print("w_out (w_down) shape  :", tuple(w_out.shape))

assert tuple(w_first.shape) == (3, 3, 2)
assert tuple(w_second.shape) == (3, 3, 2)
assert tuple(w_out.shape) == (3, 2, 3)
assert 3 not in expert_ids  # the idle expert never enters the stack at all

order = torch.argsort(pair_expert, stable=True)                           # moe.py:374
sorted_token = pair_token[order]                                          # moe.py:375
sorted_weight = pair_weight[order]                                        # moe.py:376
sorted_expert = pair_expert[order]                                        # moe.py:377

print("\n--- stable argsort by expert id (moe.py:374-377) ---")
print("order        :", order.tolist())
print("sorted_expert:", sorted_expert.tolist())
print("sorted_token :", sorted_token.tolist())
print("sorted_weight:", [round(v, 6) for v in sorted_weight.tolist()])

assert order.tolist() == [0, 3, 4, 1, 2, 5]
assert sorted_expert.tolist() == [0, 0, 0, 1, 2, 2]
assert sorted_token.tolist() == [0, 1, 2, 0, 1, 2]

expert_to_row = torch.full((n_experts,), -1, device=device, dtype=torch.long)  # moe.py:380
expert_to_row[active] = torch.arange(num_active, device=device)               # moe.py:381
row_of_pair = expert_to_row[sorted_expert]                                     # moe.py:382
row_start = active_counts.cumsum(0) - active_counts                            # moe.py:383
slot_of_pair = torch.arange(P, device=device) - row_start[row_of_pair]         # moe.py:384

print("\n--- row/slot indices (moe.py:380-384) ---")
print("expert_to_row:", expert_to_row.tolist())
print("row_of_pair  :", row_of_pair.tolist())
print("row_start    :", row_start.tolist())
print("slot_of_pair :", slot_of_pair.tolist())

assert expert_to_row.tolist() == [0, 1, 2, -1]
assert row_of_pair.tolist() == [0, 0, 0, 1, 2, 2]
assert row_start.tolist() == [0, 3, 4]
assert slot_of_pair.tolist() == [0, 1, 2, 0, 0, 1]

grid = torch.full((num_active, group_size), -1, device=device, dtype=torch.long)  # moe.py:386
grid[row_of_pair, slot_of_pair] = torch.arange(P, device=device)                  # moe.py:387
valid = grid >= 0                                                                  # moe.py:388
gathered = grid.clamp(min=0)                                                       # moe.py:389

print("\n--- the padded (K, G) grid (moe.py:386-389) ---")
print("grid:\n", grid)
print("valid:\n", valid)

assert grid.tolist() == [[0, 1, 2], [3, -1, -1], [4, 5, -1]]
assert valid.tolist() == [[True, True, True], [True, False, False], [True, True, False]]
# no two pairs ever land on the same cell: the valid cells hold 0..P-1, each once
assert sorted(grid[valid].tolist()) == list(range(P))
assert (num_active * group_size - P) == 3  # wasted grid cells this example

token_grid = sorted_token[gathered]                                       # moe.py:391
weight_grid = torch.where(valid, sorted_weight[gathered], sorted_weight.new_zeros(()))  # moe.py:392
xg = x_flat[token_grid]                                                   # moe.py:393

print("\n--- gather into the grid (moe.py:391-393) ---")
print("token_grid:\n", token_grid)
print("weight_grid:\n", weight_grid)
print("xg (K, G, d_model):\n", xg)

assert token_grid.tolist() == [[0, 1, 2], [0, 0, 0], [1, 2, 0]]
torch.testing.assert_close(
    weight_grid,
    torch.tensor([[0.731059, 0.268941, 0.880797], [0.268941, 0.0, 0.0], [0.731059, 0.119203, 0.0]]),
    atol=1e-5, rtol=0,
)

pre = torch.bmm(xg, w_first.transpose(1, 2))                              # moe.py:395
h = F.silu(pre) * torch.bmm(xg, w_second.transpose(1, 2))                 # moe.py:397 (swiglu branch)
yg_premask = torch.bmm(h, w_out.transpose(1, 2)) * weight_grid.unsqueeze(-1)  # moe.py:402
yg = torch.where(valid.unsqueeze(-1), yg_premask, yg_premask.new_zeros(()))   # moe.py:403

print("\n--- the two batched matmuls + gate + mask (moe.py:395-403) ---")
print("pre = bmm(xg, w_gate^T)  (K, G, d_ff):\n", pre)
print("h = silu(pre) * bmm(xg, w_up^T)  (K, G, d_ff):\n", h)
print("yg before the second mask (K, G, d_model):\n", yg_premask)
print("yg after the second mask:\n", yg)
print("max |yg - yg_premask| (both masks agree in THIS example):",
      (yg - yg_premask).abs().max().item())

# bmm really is K independent matmuls: row 0 alone reproduces xg[0] @ w_first[0].T
torch.testing.assert_close(pre[0], xg[0] @ w_first[0].T, atol=0, rtol=0)
torch.testing.assert_close(pre[1], xg[1] @ w_first[1].T, atol=0, rtol=0)
torch.testing.assert_close(pre[2], xg[2] @ w_first[2].T, atol=0, rtol=0)

expected_yg = torch.tensor([
    [[0.534447, 0.0], [0.196611, 0.196611], [2.839435, 2.195521]],
    [[0.401436, 0.866989], [0.0, 0.0], [0.0, 0.0]],
    [[1.287827, 1.091217], [0.507119, 0.419975], [0.0, 0.0]],
])
torch.testing.assert_close(yg, expected_yg, atol=1e-5, rtol=0)
# the one grid cell never seen in 005-moe-layer.md: expert 2 on token 0 (padding)
E2_x0 = (F.silu(x_flat[0] @ E2_GATE.T) * (x_flat[0] @ E2_UP.T)) @ E2_DOWN.T
print("\nnewly-computed padding cell E_2(x_0) (expert 2 on token 0, discarded by masking):",
      E2_x0.tolist())
torch.testing.assert_close(E2_x0, torch.tensor([0.0, 0.0]), atol=1e-6, rtol=0)

out_manual = x_flat.new_zeros(n_tokens, d_model)
out_manual.index_add_(0, token_grid.reshape(-1), yg.reshape(-1, d_model).to(out_manual.dtype))  # moe.py:405

print("\n--- scatter back to tokens (moe.py:405) ---")
print("out (routed_output), by hand:\n", out_manual)

expected_routed = torch.tensor([[0.935883, 0.866989], [1.484440, 1.287828], [3.346554, 2.615495]])
torch.testing.assert_close(out_manual, expected_routed, atol=2e-6, rtol=0)

# ============================================================
# Part 3 -- the REAL function, not a reimplementation, called directly
# ============================================================
out_real = dispatch_batched(x_flat, topk_indices, topk_weights, moe.routed_experts, "swiglu")
print("\ndispatch_batched(...) called directly:\n", out_real)
print("max |by-hand - dispatch_batched| =", (out_manual - out_real).abs().max().item())
torch.testing.assert_close(out_manual, out_real, atol=1e-6, rtol=0)

# and it must match forward_with_trace's routed_output (reconstructed the
# same way 005-moe-layer.md Step 6g does)
with torch.no_grad():
    routed_ref = torch.zeros(3, 2)
    for eid, eo in trace["expert_outputs"].items():
        tok, slot = (topk_indices == eid).nonzero(as_tuple=True)
        gate = topk_weights[tok, slot].unsqueeze(-1)
        routed_ref.index_add_(0, tok, gate * eo)
print("forward_with_trace's routed_output:\n", routed_ref)
print("max |dispatch_batched - forward_with_trace| =", (out_real - routed_ref).abs().max().item())
assert (out_real - routed_ref).abs().max().item() < 1e-5  # Requirement 5.2

# ============================================================
# Part 4 -- the full layer, both paths, end to end
# ============================================================
out_fused, aux_fused = moe(X)
print("\noutput, forward_with_trace path:\n", out_trace)
print("output, forward (dispatch_batched) path:\n", out_fused)
print("max |trace - fused| =", (out_trace - out_fused).abs().max().item())
assert (out_trace - out_fused).abs().max().item() < 1e-5

expected_output = torch.tensor([[[1.666941, 0.866989], [1.484440, 2.018887], [4.808671, 3.346554]]])
torch.testing.assert_close(out_trace, expected_output, atol=2e-6, rtol=0)
torch.testing.assert_close(out_fused, expected_output, atol=2e-6, rtol=0)

# ============================================================
# Part 5 -- Requirement 5.3: the idle expert gets NO gradient at all
# ============================================================
moe_grad = _build()
moe_grad.train()
out_g, aux_g = moe_grad(X)
(out_g.sum() + (aux_g if aux_g is not None else 0.0)).backward()

print("\n--- gradient check (Requirement 5.3) ---")
for i in range(4):
    g = moe_grad.routed_experts[i].w_gate.weight.grad
    print(f"expert {i} w_gate.grad is None: {g is None}")
assert moe_grad.routed_experts[0].w_gate.weight.grad is not None
assert moe_grad.routed_experts[1].w_gate.weight.grad is not None
assert moe_grad.routed_experts[2].w_gate.weight.grad is not None
assert moe_grad.routed_experts[3].w_gate.weight.grad is None, (
    "the idle expert must receive no gradient at all -- it was never stacked"
)

# perturbation check (005-moe-layer.md style): scrambling the idle expert's
# weights must not move the output one bit
moe_pert = _build()
with torch.no_grad():
    moe_pert.routed_experts[3].w_gate.weight.mul_(-7.0).add_(3.0)
    moe_pert.routed_experts[3].w_down.weight.fill_(123.0)
moe_pert.eval()
out_pert, _ = moe_pert(X)
assert torch.equal(out_fused, out_pert), "idle expert 3 must not affect the output"

print("\nAll assertions passed -- every number in 027-batched-expert-computation.md is real.")
