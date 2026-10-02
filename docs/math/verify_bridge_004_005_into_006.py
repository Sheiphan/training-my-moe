"""Bridge: run doc 004's attention math and doc 005's MoE math on the tensors
doc 006's model actually produces.

Doc 006 treats attention (004) and the MoE layer (005) as black boxes, and
004/005 were derived on their own hand-picked inputs, so their numbers never
touch 006's. This script closes that gap. It

  1. builds doc 006's exact model (imports `_build`/`INPUT_IDS` from
     `verify_006_full_model_assembly.py`, so there is one source of truth),
  2. hooks each block's `attn_norm -> attn` and `moe_norm -> moe` to capture the
     REAL tensors flowing into and out of every sublayer,
  3. re-computes each sublayer from scratch with the step-by-step algorithms
     written in docs 004 and 005 (no `nn.Module` calls), on the captured inputs,
  4. asserts they match the real module outputs, and
  5. re-assembles the whole forward pass out of ONLY those hand algorithms
     (embedding lookup, RMSNorm, 004 attention, 005 MoE, residuals, tied head)
     and asserts it equals the real model's logits and combined aux loss.

Scope: 006 uses `positional_scheme="nope"`, so 004's RoPE/ALiBi steps are not
exercised here (see 004 Step 4 for those).

Run it:  uv run python docs/math/verify_bridge_004_005_into_006.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_006_full_model_assembly import INPUT_IDS, _build  # noqa: E402

torch.set_printoptions(precision=6, sci_mode=False)
DT = torch.float64
TOL = 1e-5


def rmsnorm(x: torch.Tensor, gain: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt((x**2).mean(-1, keepdim=True) + eps) * gain


# --- doc 004, Steps 1-11 (positional_scheme = "nope") ---------------------
def attention_004(x, w_q, w_k, w_v, w_o, n_heads: int, n_kv_heads: int) -> torch.Tensor:
    """x: (S, d). Weights are nn.Linear-style (out, in). Returns (S, d)."""
    S, d = x.shape
    d_k = d // n_heads
    q = x @ w_q.T                                   # Step 1a  (S, H*d_k)
    k = x @ w_k.T                                   # Step 1b  (S, Hkv*d_k)
    v = x @ w_v.T                                   # Step 1c
    q = q.view(S, n_heads, d_k).transpose(0, 1)     # Step 2   (H, S, d_k)
    k = k.view(S, n_kv_heads, d_k).transpose(0, 1)  # (Hkv, S, d_k)
    v = v.view(S, n_kv_heads, d_k).transpose(0, 1)
    r = n_heads // n_kv_heads                       # Step 5   GQA: replicate K, V
    k = k.repeat_interleave(r, dim=0)
    v = v.repeat_interleave(r, dim=0)
    scores = q @ k.transpose(-1, -2)                # Step 6   (H, S, S)
    scores = scores / math.sqrt(d_k)                # Step 7
    causal = torch.triu(torch.ones(S, S, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(causal, float("-inf"))  # Step 9
    weights = torch.softmax(scores, dim=-1)         # Step 10
    heads = weights @ v                             # Step 11  (H, S, d_k)
    concat = heads.transpose(0, 1).reshape(S, n_heads * d_k)  # Step 12
    return concat @ w_o.T                           # Step 13


# --- doc 005, Steps 1-9 (softmax routing, aux-loss balancing, no shared) ---
def moe_005(x, w_router, experts, top_k: int, alpha: float):
    """x: (T, d). experts: list of (w_gate, w_up, w_down). Returns (out, aux_loss)."""
    T, d = x.shape
    N = w_router.shape[0]
    logits = x @ w_router.T                         # Step 2
    probs = torch.softmax(logits, dim=-1)           # Step 3
    topk_idx = logits.topk(top_k, dim=-1).indices   # Step 4b (selection scores = logits)
    topk_p = probs.gather(1, topk_idx)              # Step 4c
    gate = topk_p / topk_p.sum(dim=-1, keepdim=True)
    out = torch.zeros(T, d, dtype=x.dtype)
    for e in range(N):                              # Step 6 per-expert loop
        tok, slot = (topk_idx == e).nonzero(as_tuple=True)
        if tok.numel() == 0:
            continue
        w_gate, w_up, w_down = experts[e]
        h = F.silu(x[tok] @ w_gate.T) * (x[tok] @ w_up.T)   # 6c SwiGLU
        out.index_add_(0, tok, (h @ w_down.T) * gate[tok, slot].unsqueeze(-1))  # 6g
    dispatched = F.one_hot(topk_idx, N).sum(dim=1).to(x.dtype)  # Step 9
    f = dispatched.mean(dim=0)
    P = probs.mean(dim=0)
    aux = alpha * N * (f * P).sum()
    return out, aux


def _close(a: torch.Tensor, b: torch.Tensor, what: str) -> float:
    diff = (a.to(DT) - b.to(DT)).abs().max().item()
    print(f"  {what:<46s} max|diff| = {diff:.2e}")
    assert diff < TOL, (what, diff)
    return diff


def main() -> None:
    model = _build()
    cfg = model.config
    cap: dict[str, torch.Tensor] = {}
    hooks = []
    for i, blk in enumerate(model.blocks):
        for name, mod in (("attn", blk.attn), ("moe", blk.moe)):
            def pre(m, args, i=i, name=name):
                cap[f"b{i}.{name}.in"] = args[0].detach().squeeze(0).to(DT)

            def post(m, args, out, i=i, name=name):
                o = out[0] if isinstance(out, tuple) else out
                cap[f"b{i}.{name}.out"] = o.detach().squeeze(0).to(DT)
                if name == "moe":
                    cap[f"b{i}.moe.aux"] = out[1].detach().to(DT)

            hooks.append(mod.register_forward_pre_hook(pre))
            hooks.append(mod.register_forward_hook(post))

    with torch.no_grad():
        logits, extras = model(INPUT_IDS)
    for h in hooks:
        h.remove()

    print("== 1. 004 attention on the real tensors 006 feeds each block ==")
    for i, blk in enumerate(model.blocks):
        a = blk.attn
        got = attention_004(
            cap[f"b{i}.attn.in"], a.w_q.weight.to(DT), a.w_k.weight.to(DT),
            a.w_v.weight.to(DT), a.w_o.weight.to(DT), cfg.n_heads, cfg.n_kv_heads,
        )
        _close(got, cap[f"b{i}.attn.out"], f"block {i}: attention (004 Steps 1-13)")

    print("== 2. 005 MoE on the real tensors 006 feeds each block ==")
    for i, blk in enumerate(model.blocks):
        m = blk.moe
        experts = [
            (e.w_gate.weight.to(DT), e.w_up.weight.to(DT), e.w_down.weight.to(DT))
            for e in m.routed_experts
        ]
        out, aux = moe_005(
            cap[f"b{i}.moe.in"], m.router.gate.weight.to(DT), experts, cfg.top_k, cfg.aux_loss_weight
        )
        _close(out, cap[f"b{i}.moe.out"], f"block {i}: MoE output (005 Steps 1-8)")
        _close(aux, cap[f"b{i}.moe.aux"], f"block {i}: aux loss (005 Step 9)")

    print("== 3. whole forward re-assembled from ONLY the hand algorithms ==")
    E = model.embedding.weight.detach().to(DT)
    x = E[INPUT_IDS[0]]                              # 006 Step 1
    auxes = []
    for blk in model.blocks:
        a, m = blk.attn, blk.moe
        x = x + attention_004(
            rmsnorm(x, blk.attn_norm.weight.detach().to(DT)),
            a.w_q.weight.detach().to(DT), a.w_k.weight.detach().to(DT),
            a.w_v.weight.detach().to(DT), a.w_o.weight.detach().to(DT),
            cfg.n_heads, cfg.n_kv_heads,
        )                                            # 006 Steps 4-6
        experts = [
            (e.w_gate.weight.detach().to(DT), e.w_up.weight.detach().to(DT), e.w_down.weight.detach().to(DT))
            for e in m.routed_experts
        ]
        mo, aux = moe_005(
            rmsnorm(x, blk.moe_norm.weight.detach().to(DT)),
            m.router.gate.weight.detach().to(DT), experts, cfg.top_k, cfg.aux_loss_weight,
        )
        x = x + mo                                   # 006 Steps 7-9
        auxes.append(aux)
    my_logits = rmsnorm(x, model.final_norm.weight.detach().to(DT)) @ E.T  # 006 Steps 16-17
    my_aux = torch.stack(auxes).mean()               # 006 Step 18 (mean, not sum)
    _close(my_logits, logits.squeeze(0), "logits (1,3,6) vs QwenMoEModel")
    _close(my_aux, extras["aux_loss"], "combined aux loss vs QwenMoEModel")

    print("\nlogits (rows = positions 0,1,2; cols = tokens 0..5):")
    print(my_logits.to(torch.float32))
    print("BRIDGE OK: 004 + 005 hand algorithms reproduce 006's real forward pass.")


if __name__ == "__main__":
    main()
