"""Reproduces every hand-computed number in
`030-min-p-and-repetition-penalty.md`.

Run it:  uv run python docs/math/verify_030_min_p_and_repetition_penalty.py

The running example is ONE hand-built `(1, 6)` logits vector `Z` plus ONE
`generated_ids` tensor, carried through every part of the doc:

  Part 1 -- `apply_repetition_penalty` (the sign-aware divide/multiply,
            the no-op at penalty=1.0, the negative branch, and the
            argmax FLIP it causes: id 0 -> id 2).
  Part 2 -- `filter_min_p`, applied to the SAME post-penalty vector at two
            temperatures, showing the adaptive keep-count (confident: 3
            survivors: unsure: 5 survivors) -- no new example, just the
            one vector scaled two ways, exactly as `_sample_next` does.
  Part 3 -- the two "never masks everything" edge cases (a one-hot-ish
            peaked distribution, and an exactly-uniform one where ties
            all survive) and the four `ValueError`s.
  Part 4 -- `_sample_next`'s full composition on `Z` (repetition penalty
            -> temperature -> min_p -> softmax -> multinomial), with the
            CDF worked by hand and a seeded draw checked against the real
            function.
  Part 5 -- `generate()` in the wild: reuses the *exact* tiny `QwenMoEModel`
            006/011 already hand-derived and verified (same weights, same
            "DAD" prompt) -- NOT re-derived here -- to show the repetition
            penalty changing real greedy decoding: baseline 'DADDDD'
            (the same degeneration 011 found), `penalty=1.3` unchanged,
            `penalty=2.0` breaking the D-lock (verified against the by-hand
            argmax flip on the model's own real step-1 logits) but locking
            onto a *different* repeated token ('DADCCC') -- a real,
            unflattering result quoted verbatim, not smoothed over.

No fabricated numbers -- every value printed below is produced by the real
`qwenmoe.model.generate` functions, and every value quoted in the doc is
copied from this script's own output.
"""
from __future__ import annotations

import torch

from qwenmoe.config import ModelConfig
from qwenmoe.model.generate import (
    _sample_next,
    apply_repetition_penalty,
    filter_min_p,
    generate,
)
from qwenmoe.model.kv_cache import build_kv_cache
from qwenmoe.model.transformer import QwenMoEModel
from qwenmoe.tokenizer.char import CharTokenizer

torch.set_printoptions(precision=6, sci_mode=False)


# ===========================================================================
# Part 1 -- apply_repetition_penalty
# ===========================================================================
print("================ Part 1: apply_repetition_penalty ================")

# The running example: 6 tokens, ids 0..5. Chosen so the seen set covers
# every sign x seen/unseen combination that matters:
#   id0 = 3.0   seen, positive, AND the argmax  (the "self-reinforcing loop" case)
#   id1 = 2.5   seen, positive, not the argmax
#   id2 = 2.1   UNSEEN, positive -- close enough to id0 to flip the argmax
#               once id0 is penalized
#   id3 = -1.0  unseen, negative
#   id4 = -2.0  seen, negative
#   id5 = 0.5   unseen, positive
Z = torch.tensor([[3.0, 2.5, 2.1, -1.0, -2.0, 0.5]])   # (1, 6)
GENERATED_IDS = torch.tensor([[0, 1, 4, 0, 1]])          # (1, 5) -- a "sequence"
PENALTY = 1.5

seen = torch.unique(GENERATED_IDS)
print("z             :", Z.tolist()[0])
print("generated_ids :", GENERATED_IDS.tolist()[0])
print("seen = torch.unique(generated_ids):", seen.tolist())

z_pen = apply_repetition_penalty(Z, GENERATED_IDS, PENALTY)
print("z_pen         :", z_pen.tolist()[0])
print("argmax(z)     :", torch.argmax(Z, dim=-1).item())
print("argmax(z_pen) :", torch.argmax(z_pen, dim=-1).item())

EXPECTED_Z_PEN = torch.tensor([[2.0, 2.5 / 1.5, 2.1, -1.0, -3.0, 0.5]])
torch.testing.assert_close(z_pen, EXPECTED_Z_PEN, atol=1e-6, rtol=0)
assert torch.argmax(Z, dim=-1).item() == 0
assert torch.argmax(z_pen, dim=-1).item() == 2, "penalty must flip the argmax id0 -> id2"

# no-op at penalty == 1.0 (early return, byte-identical tensor)
z_noop = apply_repetition_penalty(Z, GENERATED_IDS, 1.0)
print("penalty=1.0 no-op, torch.equal(z_noop, z):", torch.equal(z_noop, Z))
assert torch.equal(z_noop, Z)

# The negative branch, demonstrated on the SAME vector: what if id3 (-1.0,
# currently unseen) had instead appeared in generated_ids?
z_neg_demo = apply_repetition_penalty(Z, torch.tensor([[3]]), PENALTY)
print("if id3 (-1.0) were seen -> penalized to:", z_neg_demo[0, 3].item(), "(expect -1.5)")
torch.testing.assert_close(z_neg_demo[0, 3], torch.tensor(-1.5), atol=1e-6, rtol=0)

# a naive "always divide" rule would instead RAISE a negative seen logit's
# probability -- show the counterfactual explicitly.
naive_divide = Z.clone()
naive_divide[0, 3] = Z[0, 3] / PENALTY
print("naive 'always divide' on id3:", naive_divide[0, 3].item(), "(less negative = HIGHER prob -- wrong)")
assert naive_divide[0, 3].item() > Z[0, 3].item()
assert z_neg_demo[0, 3].item() < Z[0, 3].item()


# ===========================================================================
# Part 2 -- filter_min_p, adaptivity from the SAME z_pen at two temperatures
# ===========================================================================
print("\n================ Part 2: filter_min_p (adaptivity) ================")

T_LOW, T_HIGH, MIN_P = 0.5, 2.5, 0.2

scaled_low = z_pen / T_LOW
scaled_high = z_pen / T_HIGH
print("scaled_low  (T=0.5):", scaled_low.tolist()[0])
print("scaled_high (T=2.5):", scaled_high.tolist()[0])

probs_low = torch.softmax(scaled_low, dim=-1)
probs_high = torch.softmax(scaled_high, dim=-1)
print("probs_low   :", probs_low.tolist()[0], " sum =", probs_low.sum().item())
print("probs_high  :", probs_high.tolist()[0], " sum =", probs_high.sum().item())
print("p_max_low   :", probs_low.max().item())
print("p_max_high  :", probs_high.max().item())
print("threshold_low  = min_p * p_max_low  =", (MIN_P * probs_low.max()).item())
print("threshold_high = min_p * p_max_high =", (MIN_P * probs_high.max()).item())

f_low = filter_min_p(scaled_low, MIN_P)
f_high = filter_min_p(scaled_high, MIN_P)
print("filter_min_p(scaled_low, 0.2) :", f_low.tolist()[0])
print("filter_min_p(scaled_high,0.2) :", f_high.tolist()[0])
n_low = torch.isfinite(f_low).sum().item()
n_high = torch.isfinite(f_high).sum().item()
print(f"survivors: low-T (confident) = {n_low}, high-T (unsure) = {n_high}")

EXPECTED_F_LOW = torch.tensor([[4.0, 10.0 / 3.0, 4.2, float("-inf"), float("-inf"), float("-inf")]])
EXPECTED_F_HIGH = torch.tensor([[0.8, 2.0 / 3.0, 0.84, -0.4, float("-inf"), 0.2]])
torch.testing.assert_close(f_low, EXPECTED_F_LOW, atol=1e-5, rtol=0, equal_nan=False)
torch.testing.assert_close(f_high, EXPECTED_F_HIGH, atol=1e-5, rtol=0, equal_nan=False)
assert n_low == 3 and n_high == 5
assert n_low < n_high, "min-p must keep FEWER tokens when the distribution is more confident"

# Exact closed form: token i survives filter_min_p(z_pen / T, min_p) iff
# T >= T_i* = (z_pen[i] - max(z_pen)) / ln(min_p)  (derived in the doc).
print("\nper-token survival temperature T_i* = (z_i - z_max) / ln(min_p):")
zmax = z_pen.max().item()
ln_min_p = torch.log(torch.tensor(MIN_P)).item()
t_star = [(zi - zmax) / ln_min_p for zi in z_pen.tolist()[0]]
print(" ", [round(t, 6) for t in t_star])
for i, t in enumerate(t_star):
    survives_low = t <= T_LOW + 1e-9
    survives_high = t <= T_HIGH + 1e-9
    assert bool(torch.isfinite(f_low[0, i])) == survives_low
    assert bool(torch.isfinite(f_high[0, i])) == survives_high
print("T_i* <= T_low (0.5) predicts survivors:", [i for i, t in enumerate(t_star) if t <= T_LOW + 1e-9])
print("T_i* <= T_high(2.5) predicts survivors:", [i for i, t in enumerate(t_star) if t <= T_HIGH + 1e-9])


# ===========================================================================
# Part 3 -- edge cases: never masks everything; out-of-range rejection
# ===========================================================================
print("\n================ Part 3: edge cases ================")

peaked = torch.tensor([[8.0, 1.0, 1.0, 1.0, 1.0, 1.0]])
f_peaked = filter_min_p(peaked, 1.0)  # the strictest possible threshold
n_peaked = torch.isfinite(f_peaked).sum().item()
print("filter_min_p(peaked, min_p=1.0)  :", f_peaked.tolist()[0], " survivors =", n_peaked)
assert n_peaked == 1 and torch.isfinite(f_peaked).any()

uniform = torch.zeros(1, 6)
f_uniform = filter_min_p(uniform, 1.0)
n_uniform = torch.isfinite(f_uniform).sum().item()
print("filter_min_p(uniform, min_p=1.0) :", f_uniform.tolist()[0], " survivors =", n_uniform)
assert n_uniform == 6, "exact ties at the threshold survive (strict '<' comparison)"

for bad_min_p in (0.0, -0.1, 1.5):
    try:
        filter_min_p(torch.zeros(1, 3), bad_min_p)
    except ValueError as e:
        print(f"  filter_min_p(min_p={bad_min_p}) -> ValueError: {e}")
    else:
        raise AssertionError(f"expected ValueError for min_p={bad_min_p}")

for bad_penalty in (0.0, -1.0):
    try:
        apply_repetition_penalty(torch.zeros(1, 4), torch.tensor([0]), bad_penalty)
    except ValueError as e:
        print(f"  apply_repetition_penalty(penalty={bad_penalty}) -> ValueError: {e}")
    else:
        raise AssertionError(f"expected ValueError for penalty={bad_penalty}")


# ===========================================================================
# Part 4 -- _sample_next: the full composed pipeline, on the SAME z
# ===========================================================================
print("\n================ Part 4: _sample_next composition ================")

filtered = filter_min_p(scaled_high, MIN_P)   # reuse Part 2's high-T (unsure) case
probs = torch.softmax(filtered, dim=-1)
cdf = torch.cumsum(probs, dim=-1)
print("filtered (rep-penalty -> /2.5 -> min_p) :", filtered.tolist()[0])
print("probs = softmax(filtered)               :", probs.tolist()[0], " sum =", probs.sum().item())
print("cdf = cumsum(probs)                      :", cdf.tolist()[0])

torch.manual_seed(0)
manual_draw = torch.multinomial(probs, num_samples=1)
print("manual torch.multinomial(probs, 1), seed 0 ->", manual_draw.item())

torch.manual_seed(0)
next_token = _sample_next(
    Z, T_HIGH, None, None, min_p=MIN_P, repetition_penalty=PENALTY, generated_ids=GENERATED_IDS
)
print("_sample_next(Z, T=2.5, min_p=0.2, penalty=1.5, seed 0) ->", next_token.item())
assert next_token.item() == manual_draw.item() == 2

EXPECTED_PROBS = torch.tensor([[0.265534, 0.232389, 0.276371, 0.079977, 0.0, 0.145728]])
torch.testing.assert_close(probs, EXPECTED_PROBS, atol=1e-5, rtol=0)
torch.testing.assert_close(probs.sum(), torch.tensor(1.0), atol=1e-6, rtol=0)

# 100k-draw frequency check -- ties the exact probs to the empirical distribution
torch.manual_seed(0)
draws = torch.multinomial(probs.expand(100_000, -1), num_samples=1)
freq = torch.bincount(draws.flatten(), minlength=6).float() / 100_000
print("100k-draw empirical frequency:", freq.tolist())
torch.testing.assert_close(freq, probs.flatten(), atol=0.01, rtol=0)


# ===========================================================================
# Part 5 -- generate() in the wild: the exact 006/011 tiny model, unmodified
# ===========================================================================
print("\n================ Part 5: generate() with the 006/011 model ================")


def _cfg(**overrides) -> ModelConfig:
    base = dict(
        vocab_size=6, d_model=4, n_layers=2, n_heads=2, n_kv_heads=1, d_ff=3,
        n_routed_experts=2, n_shared_experts=0, top_k=1, max_seq_len=6,
        positional_scheme="nope", activation="swiglu", moe_balancing="aux_loss",
        router_scoring="softmax", aux_loss_weight=0.01, tie_word_embeddings=True,
        qk_norm=False, mtp_depth=0,
    )
    base.update(overrides)
    return ModelConfig(**base)


# Weights copied verbatim from verify_006_full_model_assembly.py / verify_011 --
# NOT re-derived here; see those two scripts and their companion docs.
E = torch.tensor([
    [1.0, 1.0, 0.0, 0.0],
    [1.0, 0.0, -1.0, 0.0],
    [0.0, 0.0, 1.0, 1.0],
    [-1.0, 0.0, 0.0, 1.0],
    [0.0, 1.0, 0.0, 1.0],
    [1.0, -1.0, 1.0, -1.0],
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


def _pin_attention(attn) -> None:
    with torch.no_grad():
        attn.w_q.weight.copy_(WQ)
        attn.w_k.weight.copy_(WK)
        attn.w_v.weight.copy_(WV)
        attn.w_o.weight.copy_(WO)


def _pin_moe(moe) -> None:
    with torch.no_grad():
        moe.router.gate.weight.copy_(W_ROUTER)
        moe.routed_experts[0].w_gate.weight.copy_(E0_GATE)
        moe.routed_experts[0].w_up.weight.copy_(E0_UP)
        moe.routed_experts[0].w_down.weight.copy_(E0_DOWN)
        moe.routed_experts[1].w_gate.weight.copy_(E1_GATE)
        moe.routed_experts[1].w_up.weight.copy_(E1_UP)
        moe.routed_experts[1].w_down.weight.copy_(E1_DOWN)


def _build() -> QwenMoEModel:
    model = QwenMoEModel(_cfg())
    with torch.no_grad():
        model.embedding.weight.copy_(E)
        model.final_norm.weight.fill_(1.0)
        for block in model.blocks:
            block.attn_norm.weight.fill_(1.0)
            block.moe_norm.weight.fill_(1.0)
            _pin_attention(block.attn)
            _pin_moe(block.moe)
    return model.eval()


model = _build()
tok = CharTokenizer()
tok.train("DAD BED ACE CAB")
assert tok.encode("DAD") == [4, 1, 4]

baseline = generate(model, tok, "DAD", max_new_tokens=10, temperature=0.0)
print("repetition_penalty=1.0 (default) :", repr(baseline))
assert baseline == "DADDDD"  # the exact degeneration 011 found

out_13 = generate(model, tok, "DAD", max_new_tokens=10, temperature=0.0, repetition_penalty=1.3)
print("repetition_penalty=1.3           :", repr(out_13))
assert out_13 == "DADDDD"  # penalty too weak here to move the argmax (matches 011 Step 9)

out_20 = generate(model, tok, "DAD", max_new_tokens=10, temperature=0.0, repetition_penalty=2.0)
print("repetition_penalty=2.0           :", repr(out_20))
assert out_20 == "DADCCC"
assert len(set(out_20)) > len(set(baseline)), "breaks the 2-character D-lock into a 3-character run"

# Confirm the by-hand step-1 flip against the model's own REAL logits
# (the model's forward pass itself is 006/011's derivation, not re-done here).
seq = torch.tensor([tok.encode("DAD")])
logits, _ = model(seq)
z1 = logits[:, -1, :]
print("\nreal step-1 last-position logits z1:", z1.tolist()[0])
EXPECTED_Z1 = torch.tensor([[1.185529, 0.009417, 0.999658, 1.577581, 2.763110, -3.341033]])
torch.testing.assert_close(z1, EXPECTED_Z1, atol=2e-6, rtol=0)

for pen, expect_argmax in ((1.3, 4), (2.0, 3)):
    zp = apply_repetition_penalty(z1, seq, pen)
    am = torch.argmax(zp, dim=-1).item()
    print(f"apply_repetition_penalty(z1, seen={{1,4}}, penalty={pen}) -> argmax = {am}")
    assert am == expect_argmax

# The full 3-step trace at penalty=2.0, showing WHY it settles on 'C' (not
# a cure-all): once 'C' itself becomes a seen token (step 2), it too gets
# penalized, but its raw logit dominates by enough margin to survive anyway.
print("\nfull step-by-step trace, repetition_penalty=2.0:")
trace_seq = torch.tensor([tok.encode("DAD")])
PEN2 = 2.0
for step in range(3):
    logits_t, _ = model(trace_seq)
    z = logits_t[:, -1, :]
    zp = apply_repetition_penalty(z, trace_seq, PEN2)
    nxt = torch.argmax(zp, dim=-1, keepdim=True)
    print(f"  step {step + 1}: seen={torch.unique(trace_seq).tolist()}")
    print(f"    z  = {[round(v, 6) for v in z.tolist()[0]]}")
    print(f"    zp = {[round(v, 6) for v in zp.tolist()[0]]}")
    print(f"    argmax(raw)={torch.argmax(z, dim=-1).item()}  "
          f"argmax(penalized)={nxt.item()} ('{tok.decode([nxt.item()])}')")
    trace_seq = torch.cat([trace_seq, nxt], dim=1)
assert tok.decode(trace_seq[0].tolist()) == "DADCCC"

# ---------------------------------------------------------------------------
# Part 6 -- no fused/optimised kernel exists for these two functions (they
# are plain elementwise tensor ops); the one "two paths agree" check that
# applies is the KV-cache path, which computes logits differently but feeds
# them through the SAME apply_repetition_penalty / filter_min_p code.
# Requirement 7: show the two agree and report the measured max difference,
# extended here to a case 011 didn't cover -- repetition_penalty=2.0 on.
# ---------------------------------------------------------------------------
print("\n================ Part 6: KV-cache path agrees, penalty=2.0 ================")


def cached_trace_with_penalty(m, ids, steps, penalty):
    seq = torch.tensor([ids], dtype=torch.long)
    caches = build_kv_cache(m)
    rows = []
    for _ in range(steps):
        if caches[0].length == 0:
            lg, _ = m(seq, kv_caches=caches)
        else:
            lg, _ = m(seq[:, -1:], kv_caches=caches, position_offset=seq.shape[1] - 1)
        z = lg[:, -1, :]
        zp = apply_repetition_penalty(z, seq, penalty)
        nxt = int(torch.argmax(zp, dim=-1))
        rows.append(z.clone())
        seq = torch.cat([seq, torch.tensor([[nxt]])], dim=1)
    return seq, rows


def uncached_trace_with_penalty(m, ids, steps, penalty):
    seq = torch.tensor([ids], dtype=torch.long)
    rows = []
    for _ in range(steps):
        lg, _ = m(seq)
        z = lg[:, -1, :]
        zp = apply_repetition_penalty(z, seq, penalty)
        nxt = int(torch.argmax(zp, dim=-1))
        rows.append(z.clone())
        seq = torch.cat([seq, torch.tensor([[nxt]])], dim=1)
    return seq, rows


seq_cached, rows_cached = cached_trace_with_penalty(model, tok.encode("DAD"), 3, PEN2)
seq_uncached, rows_uncached = uncached_trace_with_penalty(model, tok.encode("DAD"), 3, PEN2)
max_diff = max((a - b).abs().max().item() for a, b in zip(rows_cached, rows_uncached))
print("cached ids  :", seq_cached[0].tolist())
print("uncached ids:", seq_uncached[0].tolist())
print(f"max |cached - uncached| last-position logit = {max_diff:.2e}")
assert seq_cached[0].tolist() == seq_uncached[0].tolist() == [4, 1, 4, 3, 3, 3]

out_20_cached = generate(
    model, tok, "DAD", max_new_tokens=10, temperature=0.0, repetition_penalty=2.0, use_kv_cache=True
)
print("generate(repetition_penalty=2.0, use_kv_cache=True):", repr(out_20_cached))
assert out_20_cached == out_20 == "DADCCC"

print("\nAll assertions passed.")
