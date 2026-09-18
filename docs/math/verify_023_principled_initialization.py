"""Reproduces every hand-computed number in `023-principled-initialization.md`.

Run it:  uv run python docs/math/verify_023_principled_initialization.py

The component under derivation is `QwenMoEModel.__init__`'s principled
initialization (moe-modernization Requirement 9, journal entry 023):

  1. `self.apply(self._init_weights)` -- truncated normal at `init_std`
     for every `nn.Linear` weight and the `nn.Embedding` weight, biases
     to zero, the tied LM head skipped;
  2. a second pass re-draws the residual-path output projections
     (`*.w_o.weight`, `*.w_down.weight`, `*.w_2.weight`) at
     `init_std / sqrt(2 * n_layers)`.

Unlike the other verify scripts in this folder, the weights here are
*random by construction*, so the end-to-end assertions are on
distributional properties (realized std, the +-2 std bound, the residual
ratio, and the depth scaling of residual-stream variance) rather than on
one fixed output tensor. Every such number the prose quotes is asserted
below.

Two configs appear:

  * TINY  -- vocab_size=6, d_model=4, n_layers=2, n_heads=2, n_kv_heads=1,
             d_ff=3, 2 routed + 1 shared swiglu experts, top_k=1,
             positional_scheme="nope", tie_word_embeddings=True.
             The running example: every weight matrix fits on a page.
  * BASELINE (configs/presets.py) -- d_model=256, n_layers=6. Used only
             for the residual-stream variance ledger, where the effect is
             numerically visible (journal 023's headline table).

No fabricated numbers -- this is the source of truth the doc quotes.
"""
from __future__ import annotations

import math

import torch

from qwenmoe.config import ModelConfig
from qwenmoe.model.transformer import (
    QwenMoEModel,
    _RESIDUAL_PROJ_SUFFIXES,
    _trunc_normal_,
)

torch.set_printoptions(precision=6, sci_mode=False, linewidth=110)

SEED = 0


# --------------------------------------------------------------------------
# The running-example (TINY) configuration -- Part 1 of the doc.
# --------------------------------------------------------------------------
def tiny_cfg(**overrides) -> ModelConfig:
    base = dict(
        vocab_size=6,
        d_model=4,
        n_layers=2,
        n_heads=2,
        n_kv_heads=1,
        d_ff=3,
        n_routed_experts=2,
        n_shared_experts=1,
        top_k=1,
        max_seq_len=8,
        positional_scheme="nope",
        activation="swiglu",
        moe_balancing="aux_loss",
        router_scoring="softmax",
        tie_word_embeddings=True,
        qk_norm=False,
        mtp_depth=0,
        init_std=0.02,
    )
    base.update(overrides)
    return ModelConfig(**base)


def baseline_cfg(**overrides) -> ModelConfig:
    # configs/presets.py::BASELINE, verbatim.
    base = dict(
        vocab_size=4096,
        d_model=256,
        n_layers=6,
        n_heads=8,
        n_kv_heads=4,
        d_ff=512,
        n_shared_experts=1,
        n_routed_experts=8,
        top_k=2,
        max_seq_len=256,
        init_std=0.02,
    )
    base.update(overrides)
    return ModelConfig(**base)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def apply_order(module: torch.nn.Module, name: str = ""):
    """The exact order `nn.Module.apply` visits submodules: every child
    (depth-first) before the parent -- so `_init_weights` sees each leaf
    `nn.Linear` before it ever sees the block that owns it."""
    for child_name, child in module.named_children():
        yield from apply_order(child, f"{name}.{child_name}" if name else child_name)
    yield name or "<root>"


def classify(param_name: str, model: QwenMoEModel) -> str:
    if param_name == "embedding.weight":
        return "embedding  (trunc-normal @ init_std)"
    if param_name.endswith(_RESIDUAL_PROJ_SUFFIXES):
        return "RESIDUAL   (trunc-normal @ init_std/sqrt(2n))"
    if param_name.endswith("_norm.weight"):
        return "rmsnorm    (left as ones)"
    return "linear     (trunc-normal @ init_std)"


def force_unscaled(model: QwenMoEModel) -> None:
    """Undo the residual-scaling pass: re-draw every residual projection at
    the plain `init_std`. Used to reproduce journal 023's 'without the rule'
    column."""
    with torch.no_grad():
        for pname, p in model.named_parameters():
            if pname.endswith(_RESIDUAL_PROJ_SUFFIXES):
                _trunc_normal_(p, model.config.init_std)


def variance_trace(model: QwenMoEModel, input_ids: torch.Tensor):
    """Walk `TransformerBlock.forward`'s two residual adds per layer,
    recording the residual-stream std after the embedding and after every
    sublayer add, plus each sublayer's output-contribution variance."""
    model.eval()
    seq_len = input_ids.shape[1]
    positions = torch.arange(seq_len)
    with torch.no_grad():
        x = model.embedding(input_ids)
        var_x0 = x.var().item()
        ledger = [("after embedding", x.std().item())]
        contribs = []
        for i, block in enumerate(model.blocks):
            attn_out = block.attn(block.attn_norm(x), positions)
            contribs.append((f"block{i}.attn", attn_out.var().item()))
            x = x + attn_out
            ledger.append((f"+ block{i} attn", x.std().item()))
            moe_out, _ = block.moe(block.moe_norm(x))
            contribs.append((f"block{i}.moe", moe_out.var().item()))
            x = x + moe_out
            ledger.append((f"+ block{i} moe", x.std().item()))
        logits, _ = model(input_ids)
    return ledger, contribs, var_x0, x.var().item(), logits.std().item()


# ==========================================================================
print("=" * 78)
print("PART 1 -- the running example: TINY QwenMoEModel, seed", SEED)
print("=" * 78)
torch.manual_seed(SEED)
model = QwenMoEModel(tiny_cfg())
cfg = model.config

print("\nconfig:", dict(vocab_size=cfg.vocab_size, d_model=cfg.d_model,
                        n_layers=cfg.n_layers, n_heads=cfg.n_heads,
                        n_kv_heads=cfg.n_kv_heads, d_ff=cfg.d_ff,
                        n_routed_experts=cfg.n_routed_experts,
                        n_shared_experts=cfg.n_shared_experts,
                        top_k=cfg.top_k, init_std=cfg.init_std,
                        tie_word_embeddings=cfg.tie_word_embeddings))

print("\n--- order self.apply(self._init_weights) visits modules ---")
for nm in apply_order(model):
    print("   ", nm or "<root>")

print("\n--- every parameter tensor, shape, and which init branch it takes ---")
resid_names = []
for pname, p in model.named_parameters():
    kind = classify(pname, model)
    if pname.endswith(_RESIDUAL_PROJ_SUFFIXES):
        resid_names.append(pname)
    print(f"  {pname:48s} {str(tuple(p.shape)):9s} {p.numel():5d}  {kind}")

EXPECTED_RESID = [
    "blocks.0.attn.w_o.weight",
    "blocks.0.moe.routed_experts.0.w_down.weight",
    "blocks.0.moe.routed_experts.1.w_down.weight",
    "blocks.0.moe.shared_experts.0.w_down.weight",
    "blocks.1.attn.w_o.weight",
    "blocks.1.moe.routed_experts.0.w_down.weight",
    "blocks.1.moe.routed_experts.1.w_down.weight",
    "blocks.1.moe.shared_experts.0.w_down.weight",
]
assert resid_names == EXPECTED_RESID, resid_names
assert "blocks.0.moe.router.gate.weight" not in resid_names, "router gate is NOT a residual proj"
assert "embedding.weight" not in resid_names
print(f"\n  -> {len(resid_names)} residual projections (2 layers x [w_o + 2 routed w_down + 1 shared w_down])")

# --------------------------------------------------------------------------
print("\n" + "=" * 78)
print("PART 2 -- residual_std, and the exact truncated-normal std factor")
print("=" * 78)
two_n = 2 * cfg.n_layers
residual_std = cfg.init_std / math.sqrt(two_n)
print(f"  2 * n_layers            = {two_n}")
print(f"  sqrt(2 * n_layers)      = {math.sqrt(two_n)}")
print(f"  residual_std            = init_std / sqrt(2n) = {cfg.init_std} / {math.sqrt(two_n)} = {residual_std}")
assert residual_std == 0.01

# variance of N(0, s^2) truncated (by rejection) to [-2s, 2s]:
#   Var = s^2 * (1 - 2*alpha*phi(alpha) / (2*Phi(alpha) - 1)),  alpha = 2
alpha = 2.0
phi = math.exp(-0.5 * alpha * alpha) / math.sqrt(2.0 * math.pi)   # standard-normal pdf at 2
Phi = 0.5 * (1.0 + math.erf(alpha / math.sqrt(2.0)))              # standard-normal cdf at 2
mass = 2.0 * Phi - 1.0
var_factor = 1.0 - (2.0 * alpha * phi) / mass
std_factor = math.sqrt(var_factor)
print(f"\n  phi(2)  = {phi:.10f}")
print(f"  Phi(2)  = {Phi:.10f}   (2*Phi(2) - 1 = {mass:.10f})")
print(f"  2*alpha*phi(alpha) / (2*Phi-1)   = {(2*alpha*phi)/mass:.10f}")
print(f"  truncated variance factor       = {var_factor:.10f}")
print(f"  truncated  STD    factor        = {std_factor:.10f}   (~ 0.88)")
assert abs(std_factor - 0.8796256610342398) < 1e-12
print(f"\n  => a nominal std of {cfg.init_std}  realizes as ~ {cfg.init_std * std_factor:.6f}")
print(f"  => a nominal std of {residual_std}  realizes as ~ {residual_std * std_factor:.6f}")

# --------------------------------------------------------------------------
print("\n" + "=" * 78)
print("PART 3 -- the running example's actual drawn weights (TINY)")
print("=" * 78)
wq0 = model.blocks[0].attn.w_q.weight.detach()
wo0 = model.blocks[0].attn.w_o.weight.detach()
print("\nblocks.0.attn.w_q.weight  (4, 4)   -- plain init_std = 0.02")
print(wq0)
print(f"  flat            : {[round(v, 6) for v in wq0.flatten().tolist()]}")
print(f"  abs().max()     : {wq0.abs().max().item():.6f}   (bound 2*init_std = {2*cfg.init_std})")
print(f"  std (unbiased)  : {wq0.std().item():.6f}")
print(f"  std (population): {wq0.std(correction=0).item():.6f}")

print("\nblocks.0.attn.w_o.weight  (4, 4)   -- residual, init_std/sqrt(2n) = 0.01")
print(wo0)
print(f"  flat            : {[round(v, 6) for v in wo0.flatten().tolist()]}")
print(f"  sum             : {wo0.sum().item():.6f}")
print(f"  mean            : {wo0.mean().item():.6f}")
print(f"  sum of squares  : {wo0.pow(2).sum().item():.8f}")
print(f"  abs().max()     : {wo0.abs().max().item():.6f}   (bound 2*residual_std = {2*residual_std})")
print(f"  std (unbiased)  : {wo0.std().item():.6f}")
print(f"  std (population): {wo0.std(correction=0).item():.6f}")

assert wq0.abs().max().item() <= 2 * cfg.init_std + 1e-9
assert wo0.abs().max().item() <= 2 * residual_std + 1e-9

# RMSNorm gains untouched
for pname, p in model.named_parameters():
    if pname.endswith("_norm.weight"):
        assert torch.equal(p, torch.ones_like(p)), pname
print("\n  every *_norm.weight is exactly ones (RMSNorm skipped by _init_weights): OK")

# tied head shares the embedding tensor and is NOT re-drawn as a linear
assert model.lm_head.weight is model.embedding.weight
assert model.lm_head.weight.std().item() == model.embedding.weight.std().item()
print("  lm_head.weight IS embedding.weight (tied, _init_weights returns early): OK")

torch.manual_seed(SEED)
untied = QwenMoEModel(tiny_cfg(tie_word_embeddings=False))
assert untied.lm_head.weight is not untied.embedding.weight
assert untied.lm_head.weight.abs().max().item() <= 2 * cfg.init_std + 1e-9
print(f"  untied: lm_head is its own tensor, trunc-normal @ init_std "
      f"(abs max {untied.lm_head.weight.abs().max().item():.6f} <= {2*cfg.init_std}): OK")

# --------------------------------------------------------------------------
print("\n" + "=" * 78)
print("PART 4 -- law of large numbers: realized std at width (d_model=256)")
print("=" * 78)
torch.manual_seed(SEED)
wide = QwenMoEModel(baseline_cfg(n_layers=2, vocab_size=4096))
pred_plain = cfg.init_std * std_factor
pred_resid = (cfg.init_std / math.sqrt(2 * 2)) * std_factor   # n_layers=2 here
print(f"  predicted plain  std = 0.02  * {std_factor:.6f} = {pred_plain:.6f}")
print(f"  predicted resid  std = 0.01  * {std_factor:.6f} = {pred_resid:.6f}   (n_layers=2)")
checks = {
    "embedding.weight": ("plain", "embedding.weight"),
    "blocks.0.attn.w_q.weight": ("plain", "blocks.0.attn.w_q.weight"),
    "blocks.0.attn.w_k.weight": ("plain", "blocks.0.attn.w_k.weight"),
    "blocks.0.attn.w_v.weight": ("plain", "blocks.0.attn.w_v.weight"),
    "blocks.0.moe.router.gate.weight": ("plain", "blocks.0.moe.router.gate.weight"),
    "blocks.0.moe.routed_experts.0.w_gate.weight": ("plain", "blocks.0.moe.routed_experts.0.w_gate.weight"),
    "blocks.0.attn.w_o.weight": ("resid", "blocks.0.attn.w_o.weight"),
    "blocks.0.moe.routed_experts.0.w_down.weight": ("resid", "blocks.0.moe.routed_experts.0.w_down.weight"),
    "blocks.0.moe.shared_experts.0.w_down.weight": ("resid", "blocks.0.moe.shared_experts.0.w_down.weight"),
}
params = dict(wide.named_parameters())
wq_wide = params["blocks.0.attn.w_q.weight"].std().item()
wo_wide = params["blocks.0.attn.w_o.weight"].std().item()
for label, (kind, name) in checks.items():
    s = params[name].std().item()
    target = pred_plain if kind == "plain" else pred_resid
    rel = abs(s - target) / target
    print(f"  {name:44s} std={s:.6f}  target={target:.6f}  rel_err={rel:.3%}")
    assert rel < 0.03, (name, s, target)
ratio = wo_wide / wq_wide
print(f"\n  w_o.std() / w_q.std() = {wo_wide:.6f} / {wq_wide:.6f} = {ratio:.4f}   "
      f"(the 0.88 factor cancels -> expect residual_std/init_std = 0.5)")
assert abs(ratio - 0.5) < 0.02
# router gate is emphatically NOT residual-scaled
rg = params["blocks.0.moe.router.gate.weight"].std().item()
assert rg > 0.9 * pred_plain, "router gate must keep the plain init_std"
print(f"  router.gate.std() = {rg:.6f}  (plain init_std, ~2x the residual projections): OK")

# --------------------------------------------------------------------------
print("\n" + "=" * 78)
print("PART 5 -- residual-stream variance ledger, TINY running example")
print("=" * 78)
torch.manual_seed(SEED)
model = QwenMoEModel(tiny_cfg())
torch.manual_seed(SEED)
input_ids = torch.randint(0, cfg.vocab_size, (1, 3))
print("  input_ids:", input_ids.tolist())

led_s, con_s, vx0_s, vf_s, lstd_s = variance_trace(model, input_ids)
print("\n  scaled init (the real model):")
for name, s in led_s:
    print(f"    {name:16s} residual-stream std = {s:.6f}")
print(f"    per-sublayer contribution var: "
      f"{[(n, f'{v:.2e}') for n, v in con_s]}")
print(f"    growth (final std / embedding std) = {led_s[-1][1] / led_s[0][1]:.4f}x")
print(f"    untrained logit std = {lstd_s:.6f}")

torch.manual_seed(SEED)
model_u = QwenMoEModel(tiny_cfg())
force_unscaled(model_u)
torch.manual_seed(SEED)
input_ids = torch.randint(0, cfg.vocab_size, (1, 3))
led_u, con_u, vx0_u, vf_u, lstd_u = variance_trace(model_u, input_ids)
print("\n  unscaled init (residual projections forced back to 0.02):")
for name, s in led_u:
    print(f"    {name:16s} residual-stream std = {s:.6f}")
print(f"    growth = {led_u[-1][1] / led_u[0][1]:.4f}x    untrained logit std = {lstd_u:.6f}")
print("\n  -> at d_model=4, n_layers=2 both are ~1x: the sublayer contributions")
print("     are ~1e-7 of the stream, so depth growth is invisible (journal 023:")
print("     'not yet a stability problem here').  Requirement 9.4 still holds:")
assert 0.02 < lstd_s < 3.0, lstd_s
print(f"     0.02 < untrained logit std ({lstd_s:.4f}) < 3.0 : OK")

# --------------------------------------------------------------------------
print("\n" + "=" * 78)
print("PART 6 -- residual-stream variance ledger, BASELINE (d_model=256, n_layers=6)")
print("         (steps outside the running example to make the effect visible)")
print("=" * 78)
BSEED = 0
torch.manual_seed(BSEED)
bmodel = QwenMoEModel(baseline_cfg())
torch.manual_seed(1)
bids = torch.randint(0, 4096, (2, 32))

bled_s, bcon_s, bvx0, bvf_s, blstd_s = variance_trace(bmodel, bids)
print("\n  scaled init:")
for name, s in bled_s:
    print(f"    {name:16s} std = {s:.6f}")
growth_s = bled_s[-1][1] / bled_s[0][1]
print(f"    growth = {growth_s:.3f}x     untrained logit std = {blstd_s:.4f}")

torch.manual_seed(BSEED)
bmodel_u = QwenMoEModel(baseline_cfg())
force_unscaled(bmodel_u)
torch.manual_seed(1)
bids = torch.randint(0, 4096, (2, 32))
bled_u, bcon_u, _, bvf_u, blstd_u = variance_trace(bmodel_u, bids)
print("\n  unscaled init:")
for name, s in bled_u:
    print(f"    {name:16s} std = {s:.6f}")
growth_u = bled_u[-1][1] / bled_u[0][1]
print(f"    growth = {growth_u:.3f}x     untrained logit std = {blstd_u:.4f}")

print("\n  per-sublayer contribution variance (scaled):")
for n, v in bcon_s:
    print(f"    {n:14s} {v:.3e}")
print("  per-sublayer contribution variance (unscaled):")
for n, v in bcon_u:
    print(f"    {n:14s} {v:.3e}")

# the residual projections carry exactly 1/(2n) the variance
wo_s = bmodel.blocks[0].attn.w_o.weight.var().item()
wo_u = bmodel_u.blocks[0].attn.w_o.weight.var().item()
print(f"\n  w_o weight var: unscaled {wo_u:.3e} / scaled {wo_s:.3e} = {wo_u / wo_s:.2f}  "
      f"(= 2*n_layers = {2*bmodel.config.n_layers}, exact)")
assert abs(wo_u / wo_s - 2 * bmodel.config.n_layers) < 1.0

# linear model:  Var(final) ~= Var(x0) + sum_j v_j   (cross terms small)
sum_v_s = sum(v for _, v in bcon_s)
print(f"\n  linear-model check (scaled):  Var(x0) + sum_j v_j = "
      f"{bvx0:.3e} + {sum_v_s:.3e} = {bvx0 + sum_v_s:.3e}   vs measured Var(final) = {bvf_s:.3e}")
print(f"  relative gap (the residual-stream cross-covariance) = "
      f"{abs(bvx0 + sum_v_s - bvf_s) / bvf_s:.1%}")

# idealised constant-v prediction from block 0
v0_s = bcon_s[0][1] + bcon_s[1][1]          # attn + moe contribution, block 0
n_res = 2 * bmodel.config.n_layers
pred_final_var_s = bvx0 + n_res * (v0_s / 2)
print(f"\n  constant-v idealisation (scaled):  Var(x0) + 2n * v0_mean = "
      f"{bvx0:.3e} + {n_res} * {v0_s/2:.3e} = {pred_final_var_s:.3e}  "
      f"-> predicted growth {math.sqrt(pred_final_var_s)/bled_s[0][1]:.2f}x  (measured {growth_s:.2f}x)")

assert growth_s < growth_u
assert growth_s < 4.0
assert growth_u > 5.0
assert 0.02 < blstd_s < 3.0
print("\n  scaled growth < unscaled growth, scaled < 4x, unscaled > 5x,")
print(f"  0.02 < BASELINE untrained logit std ({blstd_s:.3f}) < 3.0 : OK")

# --------------------------------------------------------------------------
print("\n" + "=" * 78)
print("PART 7 -- what the second init pass costs (TINY)")
print("=" * 78)
torch.manual_seed(SEED)
tm = QwenMoEModel(tiny_cfg())
total_params = sum(p.numel() for p in tm.parameters())
n_tensors = sum(1 for _ in tm.named_parameters())
resid_params = sum(p.numel() for n, p in tm.named_parameters()
                   if n.endswith(_RESIDUAL_PROJ_SUFFIXES))
n_resid_tensors = sum(1 for n, _ in tm.named_parameters()
                      if n.endswith(_RESIDUAL_PROJ_SUFFIXES))
print(f"  total parameter tensors     : {n_tensors}")
print(f"  total parameters            : {total_params}")
print(f"  residual-proj tensors re-drawn by pass 2 : {n_resid_tensors}")
print(f"  residual-proj parameters                 : {resid_params}  "
      f"({resid_params / total_params:.1%} of the model)")
print("  runtime / memory cost of the whole scheme : 0 (construction-time only)")

print("\n" + "=" * 78)
print("PART 8 -- no fused path: init is a single code path")
print("=" * 78)
torch.manual_seed(SEED)
a = QwenMoEModel(tiny_cfg())
torch.manual_seed(SEED)
b = QwenMoEModel(tiny_cfg())
maxdiff = max((pa - pb).abs().max().item()
              for (_, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters()))
print(f"  two seeded builds of the same config: max |param_a - param_b| = {maxdiff}")
assert maxdiff == 0.0
print("  deterministic given the seed; there is no separate optimised init path.")

print("\n" + "=" * 78)
print("All assertions passed -- every number in 023-principled-initialization.md is real.")
print("=" * 78)
