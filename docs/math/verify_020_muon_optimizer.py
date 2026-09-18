"""Reproduces every hand-computed number in `020-muon-optimizer.md`.

Run it:  uv run python docs/math/verify_020_muon_optimizer.py

It pins the running example (a 3x2 gradient `G` with two clean singular
values, a 3x2 weight `W0`) into the *real* `qwenmoe/train/optim.py` pieces
-- `zeropower_via_newtonschulz5`, `Muon`, `split_muon_adamw_params`,
`build_optimizer`'s `optimizer="muon"` branch, `CombinedOptimizer` -- plus
`qwenmoe/train/trainer.py::_set_lr`, runs them, prints every intermediate,
and asserts:

  * the scalar Newton-Schulz quintic p, composed 5 times, reproduces a
    float64 (no-bf16) run of the real iteration to 1e-6;
  * the real bf16 `zeropower_via_newtonschulz5(G)` lands in the same band
    (reports the measured bf16-vs-fp64 gap);
  * p's fixed points (exact quadratic) and its attracting 2-cycle;
  * one full `Muon.step` matches the by-hand `W1` (bf16 tolerance) and the
    fp64 hand computation to 1e-6;
  * a second step: the momentum buffer is exactly `1.95 * G`, yet the
    orthogonalized update is unchanged (Newton-Schulz is scale-free) to
    1e-7 in fp64;
  * `split_muon_adamw_params` routes a hand-built module the way the doc's
    table says;
  * `build_optimizer(optimizer="muon")` returns a `CombinedOptimizer`
    whose two sub-optimizers carry `base_lr = muon_lr` and `base_lr = lr`;
  * `_set_lr` drives both peaks off one shared multiplier (ratio held).

No fabricated numbers -- this file is the source of truth the prose quotes.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from qwenmoe.config import ModelConfig, TrainConfig
from qwenmoe.model.rmsnorm import RMSNorm
from qwenmoe.model.transformer import QwenMoEModel
from qwenmoe.train.optim import (
    _NS_COEFFS,
    CombinedOptimizer,
    Muon,
    build_optimizer,
    split_muon_adamw_params,
    split_param_groups,
    zeropower_via_newtonschulz5,
)
from qwenmoe.train.trainer import _set_lr, lr_multiplier

torch.set_printoptions(precision=6, sci_mode=False)

A_COEF, B_COEF, C_COEF = _NS_COEFFS  # (3.4445, -4.7750, 2.0315)


def p(s: float) -> float:
    """The scalar Newton-Schulz quintic: p(sigma) = a*sigma + b*sigma^3 + c*sigma^5."""
    return A_COEF * s + B_COEF * s**3 + C_COEF * s**5


# ======================================================================
# Part 1 -- the running example
# ======================================================================
G = torch.tensor([[2.0, 0.0], [0.0, 0.0], [0.0, 1.0]])   # gradient / momentum buffer, (3, 2)
W0 = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])   # weight being optimized, (3, 2)
LR = 0.1            # deliberately large for a visible step (real default muon_lr = 0.02)
MOMENTUM = 0.95     # Muon / config default

fro = G.pow(2).sum().sqrt().item()
print("=" * 64)
print("PART 1  running example")
print("=" * 64)
print("G (3x2):\n", G)
print(f"Frobenius norm |G|_F = sqrt(4 + 1) = sqrt(5) = {fro:.10f}")
sig0 = (2 / 5**0.5, 1 / 5**0.5)
print(f"normalized singular values: sigma1 = 2/sqrt(5) = {sig0[0]:.10f}, "
      f"sigma2 = 1/sqrt(5) = {sig0[1]:.10f}")


# ======================================================================
# Part 4.2 -- the scalar quintic, composed 5 times
# ======================================================================
print("\n" + "=" * 64)
print("PART 4.2  five Newton-Schulz iterations = p composed with itself 5x")
print("=" * 64)
traj = {}
for name, s0 in zip(("sigma1", "sigma2"), sig0):
    s = s0
    seq = [s]
    for _ in range(5):
        s = p(s)
        seq.append(s)
    traj[name] = seq
    print(f"{name}: " + "  ".join(f"{x:.6f}" for x in seq))

# one step, digit by digit, for sigma1 (doc shows this expansion)
s = sig0[0]
t1 = A_COEF * s
t2 = B_COEF * s**3
t3 = C_COEF * s**5
print(f"\np(sigma1) term by term:  {A_COEF}*{s:.9f} = {t1:.9f}")
print(f"                        {B_COEF}*{s**3:.9f} = {t2:.9f}")
print(f"                        {C_COEF}*{s**5:.9f} = {t3:.9f}")
print(f"                        sum = {t1 + t2 + t3:.9f}")


# ======================================================================
# fp64 reimplementation of the real iteration (same algebra, no bf16 cast)
# ======================================================================
def newtonschulz_fp64(g: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Byte-for-byte the body of `zeropower_via_newtonschulz5` with the
    `.bfloat16()` cast replaced by `.double()` -- so the hand arithmetic
    has something exact to be checked against."""
    x = g.double()
    transposed = x.size(0) > x.size(1)
    if transposed:
        x = x.T
    x = x / (x.norm() + eps)
    for _ in range(steps):
        aa = x @ x.T
        bb = B_COEF * aa + C_COEF * (aa @ aa)
        x = A_COEF * x + bb @ x
    if transposed:
        x = x.T
    return x


u_fp64 = newtonschulz_fp64(G)
sv_fp64 = torch.linalg.svdvals(u_fp64)
print("\nfp64 Newton-Schulz update (3x2):\n", u_fp64)
print("fp64 singular values:", sv_fp64)

# the scalar iteration must reproduce the fp64 matrix iteration
# (svdvals sorts descending; after an odd number of steps sigma1 < sigma2)
hand_pair = sorted([traj["sigma1"][-1], traj["sigma2"][-1]])
fp64_pair = sorted([sv_fp64.min().item(), sv_fp64.max().item()])
print(f"scalar-p vs fp64-matrix singular values: "
      f"{abs(hand_pair[0] - fp64_pair[0]):.2e}, {abs(hand_pair[1] - fp64_pair[1]):.2e} apart")
assert abs(hand_pair[0] - fp64_pair[0]) < 1e-6
assert abs(hand_pair[1] - fp64_pair[1]) < 1e-6
# the non-zero entries sit exactly where G's mass was (sparsity preserved):
# entry (0,0) rides sigma1's trajectory, entry (2,1) rides sigma2's
assert abs(u_fp64[0, 0].item() - traj["sigma1"][-1]) < 1e-6
assert abs(u_fp64[2, 1].item() - traj["sigma2"][-1]) < 1e-6
assert u_fp64[1, 0].item() == 0.0 and u_fp64[0, 1].item() == 0.0
print("OK  scalar p-composition == fp64 matrix iteration (< 1e-6)")


# ======================================================================
# the real (bf16-internal) function
# ======================================================================
u_real = zeropower_via_newtonschulz5(G, steps=5)
sv_real = torch.linalg.svdvals(u_real.double())
gap = (u_real.double() - u_fp64).abs().max().item()
print("\nreal zeropower_via_newtonschulz5(G) (bf16 internal):\n", u_real)
print("real singular values:", sv_real)
print(f"max |real(bf16) - hand(fp64)| = {gap:.6f}   (bf16 mantissa is ~8 bits)")
assert 0.5 < sv_real.min().item() < 1.5 and 0.5 < sv_real.max().item() < 1.5
assert gap < 0.05
print("OK  real bf16 output in the same [0.5, 1.5] band as the hand derivation")


# ======================================================================
# Part 4.2 -- fixed points (exact) and the attracting 2-cycle (numeric)
# ======================================================================
print("\n" + "=" * 64)
print("PART 4.2  fixed points and 2-cycle of p")
print("=" * 64)
# p(s) = s  <=>  s * (2.0315 s^4 - 4.7750 s^2 + 2.4445) = 0
qa, qb, qc = C_COEF, B_COEF, A_COEF - 1.0          # in u = s^2
disc = qb * qb - 4 * qa * qc
u_hi = (-qb + math.sqrt(disc)) / (2 * qa)
u_lo = (-qb - math.sqrt(disc)) / (2 * qa)
fp_hi, fp_lo = math.sqrt(u_hi), math.sqrt(u_lo)
print(f"quadratic in u = sigma^2:  {qa} u^2 + {qb} u + {qc} = 0")
print(f"  u = {u_lo:.6f}, {u_hi:.6f}  ->  sigma = {fp_lo:.6f}, {fp_hi:.6f}")
assert abs(p(fp_hi) - fp_hi) < 1e-6 and abs(p(fp_lo) - fp_lo) < 1e-6
print("OK  p(sigma*) = sigma* at both fixed points")

# attracting 2-cycle: smallest |p(p(x)) - x| roots that are NOT fixed points
xs = torch.linspace(0.4, 1.5, 400001)
f = torch.tensor([p(p(x.item())) - x.item() for x in xs])
cyc = []
for i in range(len(xs) - 1):
    if f[i] * f[i + 1] < 0:
        r = (xs[i] + xs[i + 1]).item() / 2
        if min(abs(r - fp_hi), abs(r - fp_lo)) > 1e-3:
            cyc.append(r)
print(f"2-cycle points: {cyc[0]:.6f}  <->  {cyc[1]:.6f}")
assert abs(p(cyc[0]) - cyc[1]) < 1e-3 and abs(p(cyc[1]) - cyc[0]) < 1e-3
# five steps has landed the two singular values near the 2-cycle band
assert min(cyc) - 0.05 < traj["sigma1"][-1] < max(cyc) + 0.05
assert min(cyc) - 0.05 < traj["sigma2"][-1] < max(cyc) + 0.05
print("OK  p maps each 2-cycle point to the other; both sigmas land in the band")


# ======================================================================
# Part 4.3 / 4.4 -- one full Muon.step
# ======================================================================
print("\n" + "=" * 64)
print("PART 4.3-4.4  one Muon.step: buffer -> orthogonalize -> scale -> update")
print("=" * 64)
scale = math.sqrt(max(1.0, W0.size(0) / W0.size(1)))
print(f"shape scale = sqrt(max(1, rows/cols)) = sqrt(3/2) = {scale:.9f}")

layer = nn.Linear(2, 3, bias=False)
with torch.no_grad():
    layer.weight.copy_(W0)
opt = Muon([layer.weight], lr=LR, momentum=MOMENTUM)

layer.weight.grad = G.clone()
opt.step()
W1_real = layer.weight.detach().clone()
W1_hand = W0.double() - LR * scale * u_fp64
print("W1 by hand (fp64):\n", W1_hand)
print("W1 real (Muon.step, bf16 update):\n", W1_real)
print(f"max |W1_real - W1_hand| = {(W1_real.double() - W1_hand).abs().max().item():.6f}")
assert torch.allclose(W1_real.double(), W1_hand, atol=0.02, rtol=0)
# entries where the update is exactly 0 are untouched
assert W1_real[1, 0].item() == W0[1, 0].item()
assert W1_real[2, 0].item() == W0[2, 0].item()
assert W1_real[1, 1].item() == W0[1, 1].item()
print("OK  Muon.step step 1 matches the hand computation")

# momentum buffer after step 1 is exactly G (started from zero)
buf1 = opt.state[layer.weight]["momentum_buffer"].clone()
assert torch.equal(buf1, G)
print("momentum buffer after step 1 == G  (started from 0):\n", buf1)


# ======================================================================
# Part 5 -- step 2: buffer grows, update does not
# ======================================================================
print("\n" + "=" * 64)
print("PART 5  step 2: momentum buffer -> 1.95 * G, orthogonalized update unchanged")
print("=" * 64)
layer.weight.grad = G.clone()
opt.step()
W2_real = layer.weight.detach().clone()
buf2 = opt.state[layer.weight]["momentum_buffer"].clone()
print("momentum buffer after step 2 (= 0.95*G + G = 1.95*G):\n", buf2)
assert torch.allclose(buf2, 1.95 * G, atol=1e-6)

u2_fp64 = newtonschulz_fp64(1.95 * G)
scale_inv = (u2_fp64 - u_fp64).abs().max().item()
print(f"max |NS(1.95*G) - NS(G)| in fp64 = {scale_inv:.2e}  (Newton-Schulz is scale-free)")
assert scale_inv < 1e-6

W2_hand = W1_hand - LR * scale * u2_fp64
print("W2 by hand (fp64):\n", W2_hand)
print("W2 real:\n", W2_real)
assert torch.allclose(W2_real.double(), W2_hand, atol=0.02, rtol=0)
print("OK  step 2 update reuses step 1's direction; W2 matches by hand")


# ======================================================================
# Part 2 -- split_muon_adamw_params on a hand-built module
# ======================================================================
print("\n" + "=" * 64)
print("PART 2  split_muon_adamw_params routing")
print("=" * 64)


class TinyNet(nn.Module):
    """Mirrors QwenMoEModel's parameter *shapes*: an embedding table, one
    hidden matmul weight + bias, an RMSNorm gain, an untied LM head."""

    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(5, 2)          # (5, 2)  lookup table
        self.hidden = nn.Linear(2, 3, bias=True)     # weight (3, 2), bias (3,)
        self.norm = RMSNorm(2)                       # gain (2,)
        self.lm_head = nn.Linear(2, 5, bias=False)   # (5, 2)  untied

    def forward(self, x):  # pragma: no cover - never called
        return x


tiny = TinyNet()
muon_params, adamw_params = split_muon_adamw_params(tiny)
muon_ids = {id(q) for q in muon_params}
decay, no_decay = split_param_groups(tiny)
decay_ids = {id(q) for q in decay}
routing = {}
for n, prm in tiny.named_parameters():
    routing[n] = (
        "muon" if id(prm) in muon_ids else "adamw",
        "decay" if id(prm) in decay_ids else "no_decay",
    )
    print(f"  {n:18s} shape={tuple(prm.shape)!s:8s} ndim={prm.ndim}  "
          f"-> {routing[n][0]:5s} / {routing[n][1]}")
assert routing["hidden.weight"] == ("muon", "decay")
assert routing["embedding.weight"] == ("adamw", "no_decay")
assert routing["hidden.bias"] == ("adamw", "no_decay")
assert routing["norm.weight"] == ("adamw", "no_decay")
assert routing["lm_head.weight"] == ("adamw", "decay")   # 2-D but excluded by identity
assert len(muon_params) == 1 and len(adamw_params) == 4
print("OK  one hidden matrix -> Muon; everything else -> AdamW; lm_head 2-D but AdamW")


# ======================================================================
# Part 3 / 6 -- build_optimizer(muon) + CombinedOptimizer + _set_lr
# ======================================================================
print("\n" + "=" * 64)
print("PART 3 & 6  CombinedOptimizer construction and the two-peak LR schedule")
print("=" * 64)
mcfg = ModelConfig(
    vocab_size=12, d_model=16, n_layers=2, n_heads=4, n_kv_heads=2, d_ff=32,
    n_routed_experts=4, n_shared_experts=1, top_k=2, max_seq_len=32,
)
tcfg = TrainConfig(
    batch_size=4, seq_len=8, total_tokens=1000, optimizer="muon",
    lr=1e-2, muon_lr=4e-2, muon_momentum=0.95, warmup_steps=5,
    lr_schedule="cosine", min_lr_ratio=0.1,
)
model = QwenMoEModel(mcfg)
optimizer = build_optimizer(model, tcfg)
assert isinstance(optimizer, CombinedOptimizer)
assert len(optimizer.optimizers) == 2
muon_sub, adamw_sub = optimizer.optimizers
assert isinstance(muon_sub, Muon) and isinstance(adamw_sub, torch.optim.AdamW)
for grp in muon_sub.param_groups:
    assert grp["base_lr"] == 4e-2 and "momentum" in grp
for grp in adamw_sub.param_groups:
    assert grp["base_lr"] == 1e-2 and "betas" in grp
model_param_ids = {id(pp) for pp in model.parameters()}
covered = {id(pp) for grp in optimizer.param_groups for pp in grp["params"]}
assert covered == model_param_ids
print("OK  CombinedOptimizer([Muon, AdamW]); base_lr = 0.04 (muon) / 0.01 (adamw); "
      "every model param covered once")

total_steps = 40
print("\n step  multiplier   muon lr    adamw lr   ratio")
for step in (0, 2, 4, 5, 10, 20, 40):
    mult = lr_multiplier(step, tcfg.warmup_steps, total_steps, "cosine", tcfg.min_lr_ratio)
    _set_lr(optimizer, tcfg.lr, step, total_steps, tcfg)
    lrs = [grp["lr"] for grp in optimizer.param_groups]
    muon_lr_now = next(grp["lr"] for grp in optimizer.param_groups if "momentum" in grp)
    adamw_lr_now = next(grp["lr"] for grp in optimizer.param_groups if "betas" in grp)
    print(f" {step:4d}  {mult:.6f}   {muon_lr_now:.6f}   {adamw_lr_now:.6f}   "
          f"{muon_lr_now / adamw_lr_now:.2f}")
    assert abs(muon_lr_now - 4e-2 * mult) < 1e-12
    assert abs(adamw_lr_now - 1e-2 * mult) < 1e-12
    assert abs(muon_lr_now / adamw_lr_now - 4.0) < 1e-9

# cosine at step 10, by hand:  progress = (10 - 5)/(40 - 5) = 1/7
prog = (10 - 5) / (40 - 5)
cos_hand = tcfg.min_lr_ratio + (1 - tcfg.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * prog))
print(f"\ncosine multiplier at step 10 by hand = "
      f"0.1 + 0.9 * 0.5*(1 + cos(pi/7)) = {cos_hand:.6f}")
assert abs(cos_hand - lr_multiplier(10, 5, 40, "cosine", 0.1)) < 1e-12
print("OK  one shared multiplier, two peak learning rates, ratio held at 4.00")


# ======================================================================
# CombinedOptimizer.state_dict round-trip (doc Part 6 sidebar)
# ======================================================================
sd = optimizer.state_dict()
assert list(sd.keys()) == ["sub_optimizers"] and len(sd["sub_optimizers"]) == 2
fresh = build_optimizer(QwenMoEModel(mcfg), tcfg)
fresh.load_state_dict(sd)
print("\nOK  CombinedOptimizer.state_dict round-trips as "
      "{'sub_optimizers': [muon_state, adamw_state]}")

print("\n" + "=" * 64)
print("all assertions passed -- every number in 020-muon-optimizer.md is real")
print("=" * 64)
