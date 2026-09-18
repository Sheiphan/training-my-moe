"""Reproduces every hand-computed number in `018-optimizer-hardening.md`.

Run it:  uv run python docs/math/verify_018_optimizer_hardening.py

ONE running example, carried end to end: the exact tiny 2-layer
`QwenMoEModel` from `006-full-model-assembly.md` / `010-training-loop.md`
(vocab_size=6, d_model=4, n_layers=2, n_heads=2, n_kv_heads=1, d_ff=3,
n_routed_experts=2, top_k=1, n_shared_experts=0, SwiGLU, NoPE, aux-loss
balancing, tied embeddings), with the SAME hand-picked weights pinned in,
and the SAME single next-token window `x = [4, 1, 4]`, `y = [1, 4, 1]`.

`010`'s companion carries the forward pass, the loss, and the backward
pass digit by digit; `018` (this file) is the companion for the three
Task-2 decisions layered on top of that step:

  * `split_param_groups` (optim.py:43-71): the `model.modules()` walk, the
    `recurse=False` per-module parameter fetch, the `id()` dedup, and the
    "no_decay if RMSNorm/Embedding module OR ndim < 2, else decay"
    classification -- asserted parameter by parameter, plus the tensor and
    scalar counts (22 / 256 decay, 6 / 44 no_decay, 300 total), the tied
    `lm_head.weight` landing in `no_decay` via `embedding` being registered
    first, and the untied case moving `lm_head.weight` (its own tensor) to
    `decay` (23 / 280).
  * `build_optimizer` (optim.py:265-279): one `torch.optim.AdamW` over two
    param groups, `weight_decay` 0.1 / 0.0, `betas=(0.9, 0.95)`,
    `eps=1e-8` -- all read from `TrainConfig`, not PyTorch defaults.
  * gradient-norm clipping (trainer.py:297 / :432): the global 2-norm as
    the quadrature sum of the per-parameter grad norms
    (sqrt(10.383615) = 3.222362), the six zero-token expert matrices that
    receive NO gradient this step and so contribute 0, the clip coefficient
    1/3.222362 = 0.310331, and the fact the RETURNED value is the
    *pre-clip* norm (what the JSON-lines log records).

Then the two "why the constants changed" derivations, each checked against
the real `torch.optim.AdamW`:

  * beta2 0.999 -> 0.95: the effective averaging window. Exact algebra --
    the EMA weight on the squared gradient k steps back is
    (1-b2) b2^k, it sums to 1, its mean lag is b2/(1-b2) (19 vs 999), its
    1/e characteristic length is -1/ln(b2) (19.496 vs 999.500); the Taylor
    step -1/ln(b2) ~ 1/(1-b2) (20 vs 1000) is the ONLY approximation. Then
    a regime-change experiment: warm `exp_avg_sq` to 1, jump the squared
    gradient to 100, and show the real optimizer's `exp_avg_sq` matches the
    closed form v_n = 100(1-b2^n) + 1*b2^n -- b2=0.95 reaches ~92 (true
    variance) in 50 steps while b2=0.999 is still at ~5.8, a 4x-too-large
    step exactly when gradients are largest.
  * weight_decay 0.01 -> 0.1: the decoupled-decay factor (1 - lr*wd)
    compounds geometrically over a run; at lr=3e-4 the pull toward zero
    over 10k steps is 26% (wd=0.1) vs 3% (wd=0.01).

No fabricated numbers -- this script is the source of truth the prose
quotes.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from qwenmoe.config import ModelConfig, TrainConfig
from qwenmoe.data.dataset import TokenDataset
from qwenmoe.model.rmsnorm import RMSNorm
from qwenmoe.model.transformer import QwenMoEModel
from qwenmoe.train.optim import build_optimizer, split_param_groups
from qwenmoe.train.trainer import _run_step

torch.set_printoptions(precision=6, sci_mode=False)


def _section(title: str) -> None:
    print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)


# ======================================================================
# Part 1 -- the running example (006 / 010 model, weights, window)
# ======================================================================
_section("Part 1  --  the running example (the 010 model, weights, window)")


def _cfg(**overrides) -> ModelConfig:
    base = dict(
        vocab_size=6, d_model=4, n_layers=2, n_heads=2, n_kv_heads=1, d_ff=3,
        n_routed_experts=2, n_shared_experts=0, top_k=1, max_seq_len=8,
        positional_scheme="nope", activation="swiglu", moe_balancing="aux_loss",
        router_scoring="softmax", aux_loss_weight=0.01, tie_word_embeddings=True,
        qk_norm=False, mtp_depth=0,
    )
    base.update(overrides)
    return ModelConfig(**base)


# The tied embedding / LM-head matrix E, shape (vocab_size, d_model) = (6, 4).
E = torch.tensor([
    [1.0, 1.0, 0.0, 0.0],    # token 0
    [1.0, 0.0, -1.0, 0.0],   # token 1
    [0.0, 0.0, 1.0, 1.0],    # token 2
    [-1.0, 0.0, 0.0, 1.0],   # token 3
    [0.0, 1.0, 0.0, 1.0],    # token 4
    [1.0, -1.0, 1.0, -1.0],  # token 5
])
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


def _pin(model: QwenMoEModel, tie: bool = True) -> QwenMoEModel:
    with torch.no_grad():
        model.embedding.weight.copy_(E)          # lm_head.weight IS this tensor when tied
        if not tie:
            model.lm_head.weight.copy_(E)         # untied: a genuinely separate copy
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


def build_model(tie: bool = True) -> QwenMoEModel:
    torch.manual_seed(0)
    return _pin(QwenMoEModel(_cfg(tie_word_embeddings=tie)), tie=tie)


STREAM = [4, 1, 4, 1]
train_ds = TokenDataset(STREAM, seq_len=3)
val_ds = TokenDataset(STREAM, seq_len=3)
X, Y = train_ds[0]
X = X.unsqueeze(0)
Y = Y.unsqueeze(0)
print(f"  token stream          : {STREAM}")
print(f"  one window  (x, y)    : {X.tolist()}  {Y.tolist()}")
assert X.tolist() == [[4, 1, 4]] and Y.tolist() == [[1, 4, 1]]

CONFIG = TrainConfig(
    batch_size=1, seq_len=3, total_tokens=15, lr=0.1, warmup_steps=2,
    checkpoint_interval_steps=1, eval_interval_steps=1, log_interval_steps=1,
    max_eval_batches=10, device="cpu", seed=1337, betas=(0.9, 0.95),
    adam_eps=1e-8, weight_decay=0.1, grad_clip_norm=1.0, lr_schedule="cosine",
    min_lr_ratio=0.1, precision="fp32", z_loss_weight=0.0, grad_accum_steps=1,
)
print(f"  TrainConfig (optimizer knobs): optimizer={CONFIG.optimizer!r} betas={CONFIG.betas} "
      f"adam_eps={CONFIG.adam_eps} weight_decay={CONFIG.weight_decay} grad_clip_norm={CONFIG.grad_clip_norm}")
# The modernized defaults are a pure change from PyTorch's:
assert TrainConfig(batch_size=1, seq_len=1, total_tokens=1).betas == (0.9, 0.95)
assert TrainConfig(batch_size=1, seq_len=1, total_tokens=1).weight_decay == 0.1


# ======================================================================
# Part 2 -- split_param_groups: the module walk, classification, counts
# ======================================================================
_section("Part 2  --  split_param_groups: the model.modules() walk (optim.py:43-71)")

model = build_model()
_NO_DECAY_MODULE_TYPES = (RMSNorm, nn.Embedding)  # optim.py:35

# Re-walk exactly as optim.py does, but keep names for the printout.
seen: set[int] = set()
hand_decay: list[str] = []
hand_no_decay: list[str] = []
print(f"  {'#':>2}  {'parameter':40s} {'owning module':12s} {'shape':9s} {'n':>4s}  group")
i = 0
for mod_name, module in model.named_modules():
    is_no_decay_module = isinstance(module, _NO_DECAY_MODULE_TYPES)
    for pname, param in module.named_parameters(recurse=False):
        full = f"{mod_name}.{pname}" if mod_name else pname
        if not param.requires_grad:
            continue
        if id(param) in seen:
            print(f"  {i:2d}  {full:40s} {type(module).__name__:12s} {str(tuple(param.shape)):9s} "
                  f"{param.numel():4d}  SKIP  (id already seen -- tied weight)")
            i += 1
            continue
        seen.add(id(param))
        grp = "no_decay" if (is_no_decay_module or param.ndim < 2) else "decay"
        why = (f"module is {type(module).__name__}" if is_no_decay_module
               else ("ndim < 2" if param.ndim < 2 else f"ndim {param.ndim} >= 2, plain module"))
        (hand_no_decay if grp == "no_decay" else hand_decay).append(full)
        print(f"  {i:2d}  {full:40s} {type(module).__name__:12s} {str(tuple(param.shape)):9s} "
              f"{param.numel():4d}  {grp:8s} ({why})")
        i += 1

decay, no_decay = split_param_groups(model)
decay_names = sorted(n for n, p in model.named_parameters() if any(p is d for d in decay))
no_decay_names = sorted(n for n, p in model.named_parameters() if any(p is d for d in no_decay))
assert decay_names == sorted(hand_decay), (decay_names, sorted(hand_decay))
assert no_decay_names == sorted(hand_no_decay)

n_decay_scalars = sum(p.numel() for p in decay)
n_no_decay_scalars = sum(p.numel() for p in no_decay)
n_total = sum(p.numel() for p in model.parameters())
print(f"\n  decay    : {len(decay):2d} tensors, {n_decay_scalars:3d} scalars")
print(f"  no_decay : {len(no_decay):2d} tensors, {n_no_decay_scalars:3d} scalars")
print(f"  total    : {len(decay) + len(no_decay):2d} tensors, {n_total:3d} scalars   "
      f"({n_decay_scalars} + {n_no_decay_scalars} == {n_decay_scalars + n_no_decay_scalars})")
print(f"  no_decay names: {no_decay_names}")
assert len(decay) == 22 and len(no_decay) == 6
assert n_decay_scalars == 256 and n_no_decay_scalars == 44 and n_total == 300
assert no_decay_names == [
    "blocks.0.attn_norm.weight", "blocks.0.moe_norm.weight",
    "blocks.1.attn_norm.weight", "blocks.1.moe_norm.weight",
    "embedding.weight", "final_norm.weight",
]
# every parameter classified exactly once, no overlap
all_ids = {id(p) for p in decay} | {id(p) for p in no_decay}
assert all_ids == {id(p) for p in model.parameters() if p.requires_grad}
assert len(decay) + len(no_decay) == len(all_ids)

# The tied weight: embedding.weight IS lm_head.weight, and it landed in no_decay
assert model.lm_head.weight is model.embedding.weight
assert any(model.embedding.weight is p for p in no_decay)
assert not any(model.embedding.weight is p for p in decay)
occurrences = sum(1 for p in decay + no_decay if p is model.embedding.weight)
print(f"\n  tied lm_head.weight is embedding.weight : {model.lm_head.weight is model.embedding.weight}")
print(f"  that shared tensor appears in the groups exactly {occurrences} time (-> no_decay)")
assert occurrences == 1

print("\n  --- untied variant: tie_word_embeddings=False ---")
mu = build_model(tie=False)
du, ndu = split_param_groups(mu)
assert mu.lm_head.weight is not mu.embedding.weight
in_decay = any(mu.lm_head.weight is p for p in du)
in_no_decay_emb = any(mu.embedding.weight is p for p in ndu)
print(f"  lm_head.weight is embedding.weight : {mu.lm_head.weight is mu.embedding.weight}")
print(f"  decay    : {len(du):2d} tensors, {sum(p.numel() for p in du):3d} scalars")
print(f"  no_decay : {len(ndu):2d} tensors, {sum(p.numel() for p in ndu):3d} scalars")
print(f"  lm_head.weight  -> {'decay' if in_decay else '??'}   (its own 2-D Linear weight)")
print(f"  embedding.weight -> {'no_decay' if in_no_decay_emb else '??'}   (still an nn.Embedding)")
assert len(du) == 23 and sum(p.numel() for p in du) == 280
assert len(ndu) == 6 and sum(p.numel() for p in ndu) == 44
assert in_decay and in_no_decay_emb

# The ndim<2 clause: this bias-free model has no non-RMSNorm 1-D tensor,
# so demonstrate the clause on a throwaway module with a bias.
probe = nn.Sequential(nn.Linear(4, 4, bias=True))
pd, pnd = split_param_groups(probe)
assert any(probe[0].bias is p for p in pnd), "a Linear bias (ndim 1) -> no_decay via the ndim<2 clause"
assert any(probe[0].weight is p for p in pd), "a Linear weight (ndim 2) -> decay"
print(f"\n  (probe) nn.Linear(4,4,bias=True): weight -> decay, bias (ndim 1) -> no_decay  [ndim<2 clause]")


# ======================================================================
# Part 3 -- build_optimizer: one AdamW over two param groups
# ======================================================================
_section("Part 3  --  build_optimizer: torch.optim.AdamW over two groups (optim.py:271-279)")

model = build_model()
opt = build_optimizer(model, CONFIG)
print(f"  optimizer type : {type(opt).__name__}")
assert isinstance(opt, torch.optim.AdamW)
for gi, g in enumerate(opt.param_groups):
    print(f"   group {gi}: {len(g['params']):2d} tensors  weight_decay={g['weight_decay']}  "
          f"betas={g['betas']}  eps={g['eps']:g}  lr={g['lr']}  base_lr={g.get('base_lr')}")
wds = {g["weight_decay"] for g in opt.param_groups}
assert wds == {0.1, 0.0}
for g in opt.param_groups:
    assert g["betas"] == (0.9, 0.95)
    assert abs(g["eps"] - 1e-8) < 1e-20
    assert g["base_lr"] == 0.1
optimized_ids = {id(p) for g in opt.param_groups for p in g["params"]}
assert optimized_ids == {id(p) for p in model.parameters() if p.requires_grad}
print("  every model parameter is covered by exactly one group; wd in {0.1, 0.0}")


# ======================================================================
# Part 4 -- one backward: per-parameter grad norms, the global 2-norm
# ======================================================================
_section("Part 4  --  one backward: per-parameter grad norms -> global 2-norm")

model = build_model()
model.train()
model.zero_grad(set_to_none=True)
logits, extras = model(X, targets=Y)
lm_loss = F.cross_entropy(logits.reshape(-1, 6), Y.reshape(-1))
total_loss = lm_loss + extras["aux_loss"]
total_loss.backward()
print(f"  lm_loss {lm_loss.item():.6f}   aux_loss {extras['aux_loss'].item():.6f}   "
      f"total {total_loss.item():.6f}   (== 010's step-1 loss)")

sum_sq = 0.0
n_with_grad = n_without = 0
print(f"\n  {'parameter':44s} {'||g_p||':>10s} {'||g_p||^2':>12s}")
for name, p in model.named_parameters():
    if p.grad is None:
        print(f"  {name:44s} {'--':>10s} {'(no gradient this step)':>12s}")
        n_without += 1
        continue
    gp = p.grad.norm().item()
    sq = p.grad.pow(2).sum().item()
    sum_sq += sq
    n_with_grad += 1
    print(f"  {name:44s} {gp:10.6f} {sq:12.8f}")
global_norm = math.sqrt(sum_sq)
print(f"\n  tensors with a gradient this step : {n_with_grad}   without : {n_without}")
print(f"  (the {n_without} without are routed_experts.1 in both layers -- every token routed to expert 0)")
print(f"  sum of ||g_p||^2 over all parameters : {sum_sq:.6f}")
print(f"  global 2-norm  = sqrt(sum)           : {global_norm:.6f}")

# cross-check against the exact same quantity clip_grad_norm_ computes
ref = torch.nn.utils.clip_grad_norm_(
    [p for p in model.parameters() if p.grad is not None], max_norm=float("inf")
).item()
print(f"  torch.nn.utils.clip_grad_norm_(..., inf) : {ref:.6f}   (max_norm=inf -> never rescales)")
assert abs(global_norm - 3.222362) < 1e-4
assert abs(ref - global_norm) < 1e-5
assert n_with_grad == 22 and n_without == 6

# the six zero-token expert matrices contribute exactly 0 to the sum
for li in (0, 1):
    for wn in ("w_gate", "w_up", "w_down"):
        assert getattr(model.blocks[li].moe.routed_experts[1], wn).weight.grad is None

# which tensors dominate the norm
contribs = sorted(
    ((n, p.grad.pow(2).sum().item()) for n, p in model.named_parameters() if p.grad is not None),
    key=lambda t: -t[1],
)
top3 = contribs[:3]
print(f"\n  top-3 contributors to the sum of squares:")
for n, s in top3:
    print(f"    {n:40s} {s:10.6f}   ({100 * s / sum_sq:4.1f}% of the total)")
assert top3[0][0] == "embedding.weight"
assert sum(s for _, s in top3) / sum_sq > 0.70


# ======================================================================
# Part 5 -- gradient-norm clipping
# ======================================================================
_section("Part 5  --  gradient clipping: coefficient, clipped grads, the pre-clip return")

tau = CONFIG.grad_clip_norm
coef = min(1.0, tau / global_norm)
print(f"  grad_clip_norm  tau          : {tau}")
print(f"  ||g||_2                      : {global_norm:.6f}   (> tau -> clipping fires)")
print(f"  clip coefficient  tau/||g||  : {coef:.6f}")
assert abs(coef - 0.310331) < 1e-5

fn_raw = model.final_norm.weight.grad.clone()
fn_clipped = fn_raw * coef
print(f"\n  final_norm.weight.grad raw     : {[round(v, 6) for v in fn_raw.tolist()]}")
print(f"  final_norm.weight.grad clipped : {[round(v, 6) for v in fn_clipped.tolist()]}   (each entry x {coef:.6f})")
torch.testing.assert_close(
    fn_clipped, torch.tensor([0.044862, 0.161837, -0.019149, 0.215018]), atol=1e-5, rtol=0
)
# direction preserved: clipped is a positive scalar multiple of raw
ratios = (fn_clipped / fn_raw)
assert torch.allclose(ratios, torch.full((4,), coef), atol=1e-6)
# and the clipped vector's own norm is exactly tau (per-tensor slice of the global fact)
model2 = build_model()
model2.train()
model2.zero_grad(set_to_none=True)
l2, e2 = model2(X, targets=Y)
(F.cross_entropy(l2.reshape(-1, 6), Y.reshape(-1)) + e2["aux_loss"]).backward()
pre_clip = torch.nn.utils.clip_grad_norm_(model2.parameters(), tau).item()
post_clip = math.sqrt(sum(p.grad.pow(2).sum().item() for p in model2.parameters() if p.grad is not None))
print(f"\n  clip_grad_norm_ RETURN value  : {pre_clip:.6f}   (the *pre-clip* norm -- what train.jsonl logs)")
print(f"  actual ||g||_2 AFTER the call  : {post_clip:.6f}   (== tau exactly)")
assert abs(pre_clip - global_norm) < 1e-5
assert abs(post_clip - tau) < 1e-5


# ======================================================================
# Part 6 -- AdamW step 1 (t = 1): no_decay + decay, by hand
# ======================================================================
_section("Part 6  --  AdamW step 1 (t=1): the decay term is the only new arithmetic vs 010")

model = build_model()
opt = build_optimizer(model, CONFIG)
metrics, gn, cur_lr, dev, model, ec = _run_step(model, opt, [(X, Y, None)], "cpu", 0, 5, CONFIG)
b1, b2 = CONFIG.betas
eps = CONFIG.adam_eps
gamma = cur_lr  # 0.1 * lr_multiplier(step 0) = 0.1 * 0.5 = 0.05
print(f"  lr this step  gamma           : {gamma:.4f}   (0.1 x warmup multiplier 0.5)")
print(f"  pre-clip grad_norm (returned) : {gn:.6f}")
assert abs(gamma - 0.05) < 1e-12
assert abs(gn - 3.222362) < 1e-4

st = opt.state[model.final_norm.weight]
m1, v1 = st["exp_avg"], st["exp_avg_sq"]
g_prime = m1 / (1 - b1)                       # m1 = (1-b1) g'  =>  recover g'
mhat = m1 / (1 - b1 ** 1)
vhat = v1 / (1 - b2 ** 1)
ratio = mhat / (vhat.sqrt() + eps)
fn_hand = 1.0 - gamma * ratio                 # no_decay group: no (1 - gamma*lambda) factor
print(f"\n  [no_decay]  final_norm.weight  (lambda = 0)")
print(f"    clipped g'                  : {[round(x, 6) for x in g_prime.tolist()]}")
print(f"    m1 = 0.1 g'                 : {[round(x, 7) for x in m1.tolist()]}")
print(f"    v1 = 0.05 g'^2              : {[round(x, 9) for x in v1.tolist()]}")
print(f"    m_hat/(sqrt(v_hat)+eps)     : {[round(x, 8) for x in ratio.tolist()]}   (== sign(g'), the t=1 identity)")
print(f"    w1 = 1 - gamma*sign(g')     : {[round(x, 6) for x in fn_hand.tolist()]}")
print(f"    real optimizer             : {[round(x, 6) for x in model.final_norm.weight.detach().tolist()]}")
torch.testing.assert_close(ratio, torch.sign(g_prime), atol=1e-6, rtol=0)
torch.testing.assert_close(fn_hand, model.final_norm.weight.detach(), atol=1e-6, rtol=0)

# beta2 = 0.999 would give the IDENTICAL step-1 result (bias correction cancels)
vhat_999 = (0.001 * g_prime.pow(2)) / (1 - 0.999 ** 1)
assert torch.allclose(vhat_999, g_prime.pow(2), atol=1e-9), "at t=1 from a zero start, v_hat == g'^2 for any beta2"
print(f"    (beta2=0.999 gives the same v_hat = g'^2 at t=1 -> identical first step)")

wq = model.blocks[0].attn.w_q.weight
lam = CONFIG.weight_decay
g_wq00 = opt.state[wq]["exp_avg"][0, 0] / (1 - b1)
wq00_hand = 1.0 * (1 - gamma * lam) - gamma * math.copysign(1.0, g_wq00)
wq01_hand = 0.0 * (1 - gamma * lam) - gamma * math.copysign(1.0, opt.state[wq]["exp_avg"][0, 1].item())
print(f"\n  [decay]  blocks.0.attn.w_q.weight  (lambda = {lam})")
print(f"    entry [0,0] = 1.0: theta*(1 - gamma*lambda) - gamma*sign(g')")
print(f"       = 1*(1 - {gamma}*{lam}) - {gamma}*(+1) = {1 - gamma * lam} - {gamma} = {wq00_hand:.6f}")
print(f"    real : {wq[0, 0].item():.6f}")
print(f"    entry [0,1] = 0.0: 0*(1 - gamma*lambda) - gamma*sign(g') = {wq01_hand:.6f}   (decay does nothing to 0)")
print(f"    real : {wq[0, 1].item():.6f}")
assert abs(wq00_hand - wq[0, 0].item()) < 1e-6
assert abs(wq01_hand - wq[0, 1].item()) < 1e-6
# the decoupled-decay shave on this entry is exactly gamma*lambda*theta
assert abs((1.0 - wq00_hand) - (gamma * lam * 1.0 + gamma)) < 1e-6


# ======================================================================
# Part 7 -- beta2: the effective averaging window
# ======================================================================
_section("Part 7  --  beta2 0.999 -> 0.95: the effective averaging window")

print("  EMA weight on the squared gradient k steps back:  w_k = (1 - b2) * b2^k")
for b in (0.95, 0.999):
    # sum to 1 (geometric series) -- exact
    total = sum((1 - b) * b ** k for k in range(200000))
    # mean lag  sum k w_k = b2/(1-b2) -- exact
    mean_lag = sum(k * (1 - b) * b ** k for k in range(200000))
    mean_lag_closed = b / (1 - b)
    e_len = -1.0 / math.log(b)          # exact given the "drops to 1/e" definition
    taylor = 1.0 / (1 - b)              # the ONLY approximation (ln(1-u) ~ -u)
    half_life = math.log(0.5) / math.log(b)
    print(f"\n  beta2 = {b}")
    print(f"    sum_k w_k                         : {total:.8f}   (exact: 1)")
    print(f"    mean lag  sum_k k*w_k             : {mean_lag:.4f}   (exact closed form b2/(1-b2) = {mean_lag_closed:.4f})")
    print(f"    half-life  ln(0.5)/ln(b2)         : {half_life:.4f} steps")
    print(f"    1/e length  -1/ln(b2)  [exact]    : {e_len:.4f} steps")
    print(f"    Taylor   1/(1-b2)   [approx]      : {taylor:.4f} steps   (err {100 * abs(taylor - e_len) / e_len:.2f}%)")
    assert abs(total - 1.0) < 1e-3
    assert abs(mean_lag - mean_lag_closed) < 1e-2

print("\n  --- regime-change experiment: squared gradient jumps 1 -> 100 ---")
print("  closed form (v converged to s_old, then constant s_new):")
print("      v_n = s_new*(1 - b2^n) + s_old*b2^n")
s_old, s_new = 1.0, 100.0
true_factor = 1.0 / math.sqrt(s_new)
for b2v in (0.95, 0.999):
    p = nn.Parameter(torch.zeros(1))
    o = torch.optim.AdamW([p], lr=0.0, betas=(0.9, b2v), eps=1e-12, weight_decay=0.0)
    for _ in range(20000):                       # warm exp_avg_sq -> s_old = 1
        p.grad = torch.tensor([math.sqrt(s_old)])
        o.step()
    v_warm = o.state[p]["exp_avg_sq"].item()
    reals = []
    for _ in range(50):                          # regime change: g^2 = s_new = 100
        p.grad = torch.tensor([math.sqrt(s_new)])
        o.step()
        reals.append(o.state[p]["exp_avg_sq"].item())
    print(f"\n  beta2 = {b2v}   (exp_avg_sq warmed to {v_warm:.6f})")
    print(f"    {'n':>3s}  {'v_n closed':>12s}  {'v_n real':>12s}  {'sqrt(v_n)':>10s}  {'1/sqrt(v_n)':>12s}  vs true {true_factor:.3f}")
    for n in (1, 5, 10, 20, 50):
        vc = s_new * (1 - b2v ** n) + s_old * b2v ** n
        vr = reals[n - 1]
        print(f"    {n:3d}  {vc:12.5f}  {vr:12.5f}  {math.sqrt(vr):10.4f}  {1 / math.sqrt(vr):12.4f}"
              f"  ({1 / math.sqrt(vr) / true_factor:.1f}x too large)")
        assert abs(vc - vr) < 1e-3, (b2v, n, vc, vr)
    # the punchline: 50 steps after the jump, beta2=0.999 still steps far too big
    v50_999 = reals[49]

# beta2=0.95 lands within ~5% of the true step size after 50 steps; 0.999 is ~4x off
p95 = nn.Parameter(torch.zeros(1))
o95 = torch.optim.AdamW([p95], lr=0.0, betas=(0.9, 0.95), eps=1e-12)
for _ in range(20000):
    p95.grad = torch.tensor([1.0]); o95.step()
for _ in range(50):
    p95.grad = torch.tensor([10.0]); o95.step()
factor95 = 1 / math.sqrt(o95.state[p95]["exp_avg_sq"].item())
factor999 = 1 / math.sqrt(v50_999)
print(f"\n  after 50 steps in the high-variance regime, step-size factor 1/sqrt(v):")
print(f"    beta2=0.95  : {factor95:.4f}   (true {true_factor:.4f};  {factor95 / true_factor:.2f}x)")
print(f"    beta2=0.999 : {factor999:.4f}   (true {true_factor:.4f};  {factor999 / true_factor:.2f}x  <- overconfident)")
assert factor95 / true_factor < 1.15
assert factor999 / true_factor > 3.0


# ======================================================================
# Part 8 -- weight_decay 0.01 -> 0.1: the compounding pull
# ======================================================================
_section("Part 8  --  weight_decay 0.01 -> 0.1: the decoupled-decay factor compounds")

# AdamW applies the decoupled shrink (1 - gamma*lambda) on every step a
# parameter is IN the update -- i.e. p.grad is not None. A None grad is
# skipped entirely (no adaptive step AND no decay); an actual zero-tensor
# grad IS decayed.
p_none = nn.Parameter(torch.tensor([1.0]))
p_zero = nn.Parameter(torch.tensor([1.0]))
o8 = torch.optim.AdamW([p_none, p_zero], lr=0.1, weight_decay=0.5)
p_zero.grad = torch.zeros(1)      # p_none.grad stays None
o8.step()
print(f"  grad=None  -> p = {p_none.item():.6f}   (untouched: AdamW continues past a None grad)")
print(f"  grad=0.0   -> p = {p_zero.item():.6f}   (decayed by 1 - 0.1*0.5 = 0.95)")
assert p_none.item() == 1.0
assert abs(p_zero.item() - 0.95) < 1e-6

print("\n  over S steps a weak-gradient weight follows ~ theta_0 * (1 - gamma*lambda)^S")
print(f"\n  {'lambda':>7s}  {'lr':>8s}  {'steps':>7s}  {'(1-lr*wd)^steps':>16s}  {'pull toward 0':>14s}")
for lam_v in (0.01, 0.1):
    for lr_v, steps in ((0.05, 1), (0.1, 5), (3e-4, 10000)):
        factor = (1 - lr_v * lam_v) ** steps
        print(f"  {lam_v:7g}  {lr_v:8g}  {steps:7d}  {factor:16.6f}  {1 - factor:14.4f}")
# one-step shave on w_q[0,0] = 1.0 at gamma = 0.05:
assert abs((1 - 0.05 * 0.1) - 0.995) < 1e-12          # lambda = 0.1  -> 0.995
assert abs((1 - 0.05 * 0.01) - 0.9995) < 1e-12        # lambda = 0.01 -> 0.9995
# over a 10k-step run at the real default lr, 0.1 pulls ~26%, 0.01 only ~3%
assert abs((1 - 3e-4 * 0.1) ** 10000 - 0.740815) < 1e-4
assert abs((1 - 3e-4 * 0.01) ** 10000 - 0.970445) < 1e-4


print("\n" + "=" * 72)
print("All assertions passed -- every number in 018-optimizer-hardening.md is real.")
print("=" * 72)
