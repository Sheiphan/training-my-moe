"""Reproduces every hand-computed number in `036-attention-logit-spread.md`.

Run it:  uv run python docs/math/verify_036_attention_logit_spread.py

This does NOT re-derive QK-norm itself (RMSNorm, Cauchy-Schwarz, scale
invariance) -- that is `022-qk-norm.md` / `verify_022_qk_norm.py`'s job, and
this file reuses its exact `Attention` config and weights unchanged. What is
new here is the *measurement* journal entry 036 and
`scripts/export_trace.py::_layer_stats` build on top of an unchanged forward
pass:

    attn_logit_std    = layer.attn_scores_pre_mask.std()       (export_trace.py:221)
    attn_logit_absmax = layer.attn_scores_pre_mask.abs().max()  (export_trace.py:222)
    spread(values)    = max(values) - min(values)              (site/ablation/js/deep-dive.js:86-88)

applied per `TransformerBlock` layer and then, as "range across depth",
across a whole model's layer list. This file:

  1. builds ONE tiny `Attention` module (doc 022's exact 3-token, 2-head,
     1-KV-head, head_dim=4 example) and evaluates it at TWO input scales
     (1x and 10x) to stand in for a "shallow" and a "deep" layer's residual
     stream -- with qk_norm OFF and ON -- and hand-verifies `std`/`absmax`/
     `spread` on both;
  2. loads the REAL trained `BASELINE` / `BASELINE_V2` / `FINEGRAINED`
     checkpoints journal 036 measured, runs `forward_with_trace` on the
     real prompt, recomputes `_layer_stats`' exact formula per layer, and
     checks the result against journal 036's own quoted numbers (the
     "23x smaller range" headline).

No fabricated numbers -- this file is the source of truth the prose quotes.
"""
from __future__ import annotations

import torch

from qwenmoe.config import ModelConfig
from qwenmoe.model.attention import Attention
from qwenmoe.model.transformer import QwenMoEModel
from qwenmoe.tokenizer.base import build_tokenizer_from_meta
from qwenmoe.train.trainer import load_checkpoint

torch.set_printoptions(precision=8, sci_mode=False)


def _section(title: str) -> None:
    print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)


def spread(values: list[float]) -> float:
    """`site/ablation/js/deep-dive.js:86-88`, ported to Python verbatim."""
    return max(values) - min(values)


# ======================================================================
# The running example -- doc 022's exact Attention config and weights.
# ======================================================================
D_MODEL, N_HEADS, N_KV_HEADS, HEAD_DIM, SEQ_LEN = 8, 2, 1, 4, 3

X = torch.zeros(1, SEQ_LEN, D_MODEL)
X[0, 0, 0] = 1.0
X[0, 1, 1] = 1.0
X[0, 2, 2] = 1.0
POSITIONS = torch.arange(SEQ_LEN)

W_Q = torch.zeros(N_HEADS * HEAD_DIM, D_MODEL)
W_Q[:, 0] = torch.tensor([1.0, 1, 1, 1, 2, 0, 2, 0])
W_Q[:, 1] = torch.tensor([3.0, 0, 4, 0, 1, 1, 1, 1])
W_Q[:, 2] = torch.tensor([2.0, 2, 0, 0, 0, 3, 0, 4])

W_K = torch.zeros(N_KV_HEADS * HEAD_DIM, D_MODEL)
W_K[:, 0] = torch.tensor([2.0, 0, 0, 0])
W_K[:, 1] = torch.tensor([0.0, 2, 2, 2])
W_K[:, 2] = torch.tensor([1.0, 1, 1, 1])

W_V = torch.zeros(N_KV_HEADS * HEAD_DIM, D_MODEL)
W_V[:, 0] = torch.tensor([1.0, 0, 0, 0])
W_V[:, 1] = torch.tensor([0.0, 1, 0, 0])
W_V[:, 2] = torch.tensor([1.0, 1, 1, 0])

W_O = torch.eye(N_HEADS * HEAD_DIM)


def _cfg(qk_norm: bool) -> ModelConfig:
    return ModelConfig(
        vocab_size=10,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_kv_heads=N_KV_HEADS,
        max_seq_len=8,
        positional_scheme="nope",
        qk_norm=qk_norm,
        rope_theta=10000.0,
    )


def _build(qk_norm: bool) -> Attention:
    attn = Attention(_cfg(qk_norm))
    with torch.no_grad():
        attn.w_q.weight.copy_(W_Q)
        attn.w_k.weight.copy_(W_K)
        attn.w_v.weight.copy_(W_V)
        attn.w_o.weight.copy_(W_O)
    return attn


# ======================================================================
# 1. attn_logit_std / attn_logit_absmax, by hand, qk_norm=False, scale=1x
# ======================================================================
_section("1. qk_norm=False, scale=1x ('shallow' layer): std and absmax by hand")
attn_off = _build(qk_norm=False)
_, tr_off_1x = attn_off.forward_with_trace(X * 1.0, POSITIONS)
scores_off_1x = tr_off_1x["attn_scores_pre_mask"]
flat = scores_off_1x.flatten()
n = flat.numel()
mean = flat.sum() / n
sq_dev_sum = ((flat - mean) ** 2).sum()
std_by_hand = (sq_dev_sum / (n - 1)).sqrt()
print(f"  flattened scores ({n} entries): {[round(v, 4) for v in flat.tolist()]}")
print(f"  mean = sum / {n} = {mean.item():.6f}")
print(f"  sum of squared deviations = {sq_dev_sum.item():.6f}")
print(f"  unbiased variance = sum_sq_dev / (n-1) = sum_sq_dev / {n - 1} = "
      f"{(sq_dev_sum / (n - 1)).item():.6f}")
print(f"  std (by hand)   = {std_by_hand.item():.8f}")
print(f"  std (torch)     = {scores_off_1x.std().item():.8f}")
print(f"  absmax (by hand)= {flat.abs().max().item():.8f}")
torch.testing.assert_close(std_by_hand, scores_off_1x.std(), atol=1e-5, rtol=0)
assert mean.item() == 2.5
assert sq_dev_sum.item() == 38.0
assert flat.abs().max().item() == 7.0
print("  OK -- hand-derived std/absmax match torch exactly.")


# ======================================================================
# 2. Same layer, input scaled by 10x ('deep' layer): homogeneity of std/max
# ======================================================================
_section("2. qk_norm=False, scale=10x ('deep' layer): exact c^2 homogeneity")
_, tr_off_10x = attn_off.forward_with_trace(X * 10.0, POSITIONS)
scores_off_10x = tr_off_10x["attn_scores_pre_mask"]
std_off_1x, std_off_10x = scores_off_1x.std().item(), scores_off_10x.std().item()
absmax_off_1x, absmax_off_10x = scores_off_1x.abs().max().item(), scores_off_10x.abs().max().item()
print(f"  std,    1x -> 10x : {std_off_1x:.6f} -> {std_off_10x:.6f}  "
      f"(ratio {std_off_10x / std_off_1x:.4f}, predicted 10^2=100)")
print(f"  absmax, 1x -> 10x : {absmax_off_1x:.6f} -> {absmax_off_10x:.6f}  "
      f"(ratio {absmax_off_10x / absmax_off_1x:.4f}, predicted 100)")
assert abs(std_off_10x / std_off_1x - 100.0) < 1e-3
assert abs(absmax_off_10x / absmax_off_1x - 100.0) < 1e-6

spread_off = spread([std_off_1x, std_off_10x])
print(f"  spread([std_1x, std_10x]) = {std_off_10x:.6f} - {std_off_1x:.6f} = {spread_off:.6f}")


# ======================================================================
# 3. qk_norm=True: the same two scales, near-perfect invariance
# ======================================================================
_section("3. qk_norm=True, scale=1x and 10x: std/absmax pinned (up to eps)")
attn_on = _build(qk_norm=True)
_, tr_on_1x = attn_on.forward_with_trace(X * 1.0, POSITIONS)
_, tr_on_10x = attn_on.forward_with_trace(X * 10.0, POSITIONS)
scores_on_1x = tr_on_1x["attn_scores_pre_mask"]
scores_on_10x = tr_on_10x["attn_scores_pre_mask"]
std_on_1x, std_on_10x = scores_on_1x.std().item(), scores_on_10x.std().item()
absmax_on_1x, absmax_on_10x = scores_on_1x.abs().max().item(), scores_on_10x.abs().max().item()
print(f"  std,    1x -> 10x : {std_on_1x:.8f} -> {std_on_10x:.8f}  "
      f"(diff {abs(std_on_10x - std_on_1x):.2e})")
print(f"  absmax, 1x -> 10x : {absmax_on_1x:.8f} -> {absmax_on_10x:.8f}  "
      f"(diff {abs(absmax_on_10x - absmax_on_1x):.2e})")
assert abs(std_on_10x - std_on_1x) < 1e-4
assert abs(absmax_on_10x - absmax_on_1x) < 1e-4

spread_on = spread([std_on_1x, std_on_10x])
print(f"  spread([std_1x, std_10x]) = {std_on_10x:.8f} - {std_on_1x:.8f} = {spread_on:.2e}")


# ======================================================================
# 4. The toy 'ratio' -- an idealized worst case, not journal 036's real one
# ======================================================================
_section("4. toy spread ratio (idealized 10x scalar blow-up)")
ratio_toy = spread_off / max(spread_on, 1e-12)
print(f"  spread(OFF) / spread(ON) = {spread_off:.6f} / {spread_on:.2e} = {ratio_toy:.3e}")
print("  (this toy uses an EXACT scalar rescale between the two 'layers' -- an")
print("   idealized worst case, not the real trained-checkpoint number in step 5)")


# ======================================================================
# 5. The real trained checkpoints journal 036 measured.
# ======================================================================
_section("5. real checkpoints: per-layer attn_logit_std, absmax, spread")
PROMPT = "ROMEO:\nWhat is the cause of this?\n"
CHECKPOINTS = {
    "BASELINE": "runs/modernized_presets/BASELINE/seed_1337/checkpoints/step_305.pt",
    "BASELINE_V2": "runs/modernized_presets/BASELINE_V2/seed_1337/checkpoints/step_305.pt",
    "FINEGRAINED": "runs/modernized_presets/FINEGRAINED/seed_1337/checkpoints/step_305.pt",
}
EXPECTED_RANGE = {"BASELINE": 1.615, "BASELINE_V2": 0.070, "FINEGRAINED": 0.019}

results = {}
for name, path in CHECKPOINTS.items():
    checkpoint = load_checkpoint(path)
    cfg = checkpoint["model_config"]
    model = QwenMoEModel(cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    tokenizer = build_tokenizer_from_meta(checkpoint["tokenizer_meta"])
    token_ids = torch.tensor([tokenizer.encode(PROMPT)])
    with torch.no_grad():
        _, bundle = model.forward_with_trace(token_ids)

    stds = [float(layer.attn_scores_pre_mask.std()) for layer in bundle.layers]
    absmaxes = [float(layer.attn_scores_pre_mask.abs().max()) for layer in bundle.layers]
    layer_range = spread(stds)
    results[name] = {"stds": stds, "absmaxes": absmaxes, "range": layer_range}

    print(f"\n  {name}  (qk_norm={cfg.qk_norm}, n_layers={cfg.n_layers})")
    print(f"    attn_logit_std   per layer: {[round(v, 4) for v in stds]}")
    print(f"    attn_logit_absmax per layer: {[round(v, 4) for v in absmaxes]}")
    print(f"    spread (range across depth) = {layer_range:.4f}   "
          f"(journal 036 quoted: {EXPECTED_RANGE[name]})")
    assert abs(layer_range - EXPECTED_RANGE[name]) < 0.005

ratio_real = results["BASELINE"]["range"] / results["BASELINE_V2"]["range"]
ratio_finegrained = results["BASELINE"]["range"] / results["FINEGRAINED"]["range"]
print(f"\n  spread(BASELINE) / spread(BASELINE_V2) = {ratio_real:.1f}x   "
      f"(journal 036 headline: '23x smaller range')")
print(f"  spread(BASELINE) / spread(FINEGRAINED) = {ratio_finegrained:.1f}x   "
      f"(not in the journal's headline -- an extra confirmation)")
assert 20.0 < ratio_real < 26.0

print("\nAll assertions passed -- every number in 036-attention-logit-spread.md is real.")
