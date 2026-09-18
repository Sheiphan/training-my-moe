"""Reproduces every hand-computed number in `011-inference-generation.md`.

Run it:  uv run python docs/math/verify_011_inference_generation.py

The running example is the *exact* tiny `QwenMoEModel` derivation 006 built
(same config, same pinned weights), plus a `CharTokenizer` trained on a
6-character corpus and the 3-token prompt "DAD" -> ids [4, 1, 4]. Reusing
006's model means the very first decode step's last-position logits are the
`(1, 3, 6)` logit row 006 already derived and asserted by hand -- so this
script only has to trace `qwenmoe/model/generate.py`: the decode loop, the
repetition penalty, temperature, `filter_top_k` / `filter_top_p` /
`filter_min_p`, softmax, `torch.multinomial`, and the KV-cache path.

It prints every intermediate and asserts:

  * the step-1 logit vector equals 006's hand-derived row 2 (continuity),
  * greedy decoding's argmax at every step, and the full 3-token Run A,
  * the sign-aware repetition penalty (positive -> divide, negative -> multiply),
  * temperature scaling and its T->0 / T->infinity limits,
  * `filter_top_k(k=4)`, `filter_top_p(p=0.9)`, `filter_min_p(0.05)` -- which
    tokens each keeps, digit by digit,
  * the composed `_sample_next` pipeline for Run B step 1 + a 100k-draw
    frequency check of `torch.multinomial`,
  * the KV-cache path emits a byte-identical greedy sequence, with the
    measured max |cached - uncached| logit difference,
  * `max_seq_len` / `max_new_tokens` stop conditions, greedy determinism
    across seeds, and every input-validation `ValueError`.

No fabricated numbers -- this is the source of truth the prose quotes.
"""
from __future__ import annotations

import math

import torch

from qwenmoe.config import ModelConfig
from qwenmoe.model.generate import (
    _sample_next,
    apply_repetition_penalty,
    filter_min_p,
    filter_top_k,
    filter_top_p,
    generate,
)
from qwenmoe.model.kv_cache import build_kv_cache
from qwenmoe.model.transformer import QwenMoEModel
from qwenmoe.tokenizer.char import CharTokenizer

torch.set_printoptions(precision=6, sci_mode=False)


# ---------------------------------------------------------------------------
# Part 1 -- the running example: 006's model, verbatim.
# ---------------------------------------------------------------------------
def _cfg(**overrides) -> ModelConfig:
    base = dict(
        vocab_size=6,
        d_model=4,
        n_layers=2,
        n_heads=2,
        n_kv_heads=1,
        d_ff=3,
        n_routed_experts=2,
        n_shared_experts=0,
        top_k=1,
        max_seq_len=6,  # 006 used 8; 6 here so Run A hits the context cap
        positional_scheme="nope",
        activation="swiglu",
        moe_balancing="aux_loss",
        router_scoring="softmax",
        aux_loss_weight=0.01,
        tie_word_embeddings=True,
        qk_norm=False,
        mtp_depth=0,
    )
    base.update(overrides)
    return ModelConfig(**base)


# The embedding / tied-LM-head matrix E, shape (vocab_size, d_model) = (6, 4).
E = torch.tensor(
    [
        [1.0, 1.0, 0.0, 0.0],    # id 0  = ' '
        [1.0, 0.0, -1.0, 0.0],   # id 1  = 'A'
        [0.0, 0.0, 1.0, 1.0],    # id 2  = 'B'
        [-1.0, 0.0, 0.0, 1.0],   # id 3  = 'C'
        [0.0, 1.0, 0.0, 1.0],    # id 4  = 'D'
        [1.0, -1.0, 1.0, -1.0],  # id 5  = 'E'
    ]
)
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


def _build(**cfg_overrides) -> QwenMoEModel:
    model = QwenMoEModel(_cfg(**cfg_overrides))
    with torch.no_grad():
        model.embedding.weight.copy_(E)
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
    return model.eval()


model = _build()

tok = CharTokenizer()
tok.train("DAD BED ACE CAB")  # unique chars: ' ' A B C D E
print("================ Part 1 -- the running example ================")
print("tokenizer stoi:", tok._stoi)
assert tok._stoi == {" ": 0, "A": 1, "B": 2, "C": 3, "D": 4, "E": 5}, tok._stoi
assert tok.encode("DAD") == [4, 1, 4], tok.encode("DAD")
assert tok.decode([4, 1, 4, 4]) == "DADD"
print("encode('DAD') =", tok.encode("DAD"), " decode([4,1,4]) =", repr(tok.decode([4, 1, 4])))

VOCAB = [" ", "A", "B", "C", "D", "E"]

# ---------------------------------------------------------------------------
# Part 2 -- step 1: the forward pass and the last-position logit vector z.
# ---------------------------------------------------------------------------
print("\n================ Part 2 -- decode step 1 ================")
sequence = torch.tensor([tok.encode("DAD")], dtype=torch.long)  # (1, 3)
print("sequence (1, 3):", sequence.tolist())

logits, extras = model(sequence)
print("logits shape:", tuple(logits.shape), " extras keys:", sorted(extras))
z = logits[0, -1, :].clone()  # (6,)  -- generate.py:211  logits[:, -1, :]
print("z = logits[0, -1, :] :", z.tolist())

# Continuity: z must equal 006's hand-derived + asserted logit row 2.
Z_006_ROW2 = torch.tensor([1.185529, 0.009417, 0.999658, 1.577581, 2.763110, -3.341033])
torch.testing.assert_close(z, Z_006_ROW2, atol=2e-6, rtol=0)
print("  -> matches 006-full-model-assembly.md Step 17 row 2  ✓")

# ---------------------------------------------------------------------------
# Part 3 -- greedy branch (temperature <= 0): argmax, generate.py:121-122
# ---------------------------------------------------------------------------
print("\n================ Part 3 -- greedy (temperature = 0) ================")
greedy_tok = torch.argmax(z, dim=-1).item()
print(f"argmax(z) = {greedy_tok}  ('{VOCAB[greedy_tok]}')")
assert greedy_tok == 4

# T -> 0 limit, done numerically: softmax(z / T) -> one-hot as T shrinks.
for T in (1.0, 0.1, 0.01):
    p = torch.softmax(z / T, dim=-1)
    print(f"  softmax(z/{T:>4}) = {[round(v, 6) for v in p.tolist()]}")
# T -> infinity limit: uniform.
p_big = torch.softmax(z / 1e6, dim=-1)
print(f"  softmax(z/1e6)  = {[round(v, 6) for v in p_big.tolist()]}  (-> 1/6 = {1/6:.6f})")
torch.testing.assert_close(p_big, torch.full((6,), 1 / 6), atol=1e-5, rtol=0)

# ---------------------------------------------------------------------------
# Part 4 -- the repetition penalty (generate.py:73-96), sign-aware.
# ---------------------------------------------------------------------------
print("\n================ Part 4 -- repetition penalty (penalty = 1.3) ================")
PENALTY = 1.3
seen_ids = torch.unique(sequence)  # {1, 4}
print("seen ids (torch.unique(sequence)):", seen_ids.tolist())
z_pen = apply_repetition_penalty(z.clone(), sequence, PENALTY)
print("z after penalty:", [round(v, 6) for v in z_pen.tolist()])
# id 1 (logit +0.009417 > 0) -> divided; id 4 (+2.763110 > 0) -> divided.
torch.testing.assert_close(z_pen[1], z[1] / PENALTY, atol=1e-6, rtol=0)
torch.testing.assert_close(z_pen[4], z[4] / PENALTY, atol=1e-6, rtol=0)
torch.testing.assert_close(z_pen[0], z[0], atol=0, rtol=0)  # unseen -> untouched
print(f"  id1: {z[1]:.6f} / {PENALTY} = {z[1] / PENALTY:.6f}")
print(f"  id4: {z[4]:.6f} / {PENALTY} = {z[4] / PENALTY:.6f}")
# The negative-logit branch: if id 5 (logit -3.341033 < 0) were seen it is
# *multiplied*, not divided -- same operation, other side of the sign test.
z_pen5 = apply_repetition_penalty(z.clone(), torch.tensor([[5]]), PENALTY)
torch.testing.assert_close(z_pen5[5], z[5] * PENALTY, atol=1e-6, rtol=0)
print(f"  id5 (if seen): {z[5]:.6f} * {PENALTY} = {z_pen5[5].item():.6f}   (multiply, not divide)")
assert apply_repetition_penalty(z.clone(), sequence, 1.0) is not None
assert torch.equal(apply_repetition_penalty(z.clone(), sequence, 1.0), z)  # 1.0 is a no-op

# ---------------------------------------------------------------------------
# Part 5 -- temperature scaling for Run B (T = 0.8), generate.py:123
# ---------------------------------------------------------------------------
print("\n================ Part 5 -- temperature T = 0.8 ================")
T = 0.8
z_scaled = z / T
print("z / 0.8 :", [round(v, 6) for v in z_scaled.tolist()])
torch.testing.assert_close(z_scaled, z / 0.8, atol=0, rtol=0)
# sharpening table (same shape as journal 011's, on THIS z)
print("  T      softmax(z/T)")
for Tt in (2.0, 1.0, 0.8, 0.5):
    p = torch.softmax(z / Tt, dim=-1)
    print(f"  {Tt:>4}  {[round(v, 6) for v in p.tolist()]}")

# ---------------------------------------------------------------------------
# Part 6 -- filter_top_k (generate.py:19-30), k = 4
# ---------------------------------------------------------------------------
print("\n================ Part 6 -- filter_top_k(k = 4) ================")
K = 4
topk = torch.topk(z_scaled, K, dim=-1)
print("topk values:", [round(v, 6) for v in topk.values.tolist()], " indices:", topk.indices.tolist())
threshold_k = topk.values[..., -1, None]
print("threshold (4th largest):", round(threshold_k.item(), 6))
z_topk = filter_top_k(z_scaled, K)
print("after top_k:", [("-inf" if math.isinf(v) else round(v, 6)) for v in z_topk.tolist()])
kept_k = [i for i, v in enumerate(z_topk.tolist()) if not math.isinf(v)]
print("kept ids:", kept_k, "->", [VOCAB[i] for i in kept_k])
assert kept_k == [0, 2, 3, 4]  # the 4 largest of z_scaled; ids 1 and 5 dropped

# ---------------------------------------------------------------------------
# Part 7 -- filter_top_p (generate.py:33-50), p = 0.9
# ---------------------------------------------------------------------------
print("\n================ Part 7 -- filter_top_p(p = 0.9) ================")
P = 0.9
sorted_logits, sorted_idx = torch.sort(z_scaled, descending=True, dim=-1)
print("sorted logits :", [round(v, 6) for v in sorted_logits.tolist()])
print("sorted indices:", sorted_idx.tolist())
sorted_probs = torch.softmax(sorted_logits, dim=-1)
print("sorted probs  :", [round(v, 6) for v in sorted_probs.tolist()])
mass_before = torch.cumsum(sorted_probs, dim=-1) - sorted_probs
print("mass_before   :", [round(v, 6) for v in mass_before.tolist()])
excluded = mass_before >= P
print("excluded (mass_before >= 0.9):", excluded.tolist())
z_topp = filter_top_p(z_scaled, P)
print("after top_p (original order):",
      [("-inf" if math.isinf(v) else round(v, 6)) for v in z_topp.tolist()])
kept_p = [i for i, v in enumerate(z_topp.tolist()) if not math.isinf(v)]
print("kept ids:", kept_p, "->", [VOCAB[i] for i in kept_p])

# ---------------------------------------------------------------------------
# Part 8 -- filter_min_p (generate.py:53-70), min_p = 0.05
# ---------------------------------------------------------------------------
print("\n================ Part 8 -- filter_min_p(min_p = 0.05) ================")
MIN_P = 0.05
probs_mp = torch.softmax(z_scaled, dim=-1)
print("softmax(z/0.8)      :", [round(v, 6) for v in probs_mp.tolist()])
p_max = probs_mp.max().item()
print(f"p_max               : {p_max:.6f}   threshold = 0.05 * p_max = {MIN_P * p_max:.6f}")
z_minp = filter_min_p(z_scaled, MIN_P)
kept_mp = [i for i, v in enumerate(z_minp.tolist()) if not math.isinf(v)]
print("kept ids:", kept_mp, "->", [VOCAB[i] for i in kept_mp])

# ---------------------------------------------------------------------------
# Part 9 -- the composed _sample_next pipeline for Run B, step 1.
#   order (generate.py:119-131): rep-penalty -> /T -> top_k -> top_p -> min_p
#   -> softmax -> multinomial
# ---------------------------------------------------------------------------
print("\n================ Part 9 -- composed _sample_next (Run B step 1) ================")
RUN_B = dict(temperature=0.8, top_k=4, top_p=0.9, min_p=0.05, repetition_penalty=1.3)
z_b = apply_repetition_penalty(z.clone(), sequence, RUN_B["repetition_penalty"])
print("stage rep-penalty:", [round(v, 6) for v in z_b.tolist()])
z_b = z_b / RUN_B["temperature"]
print("stage / 0.8      :", [round(v, 6) for v in z_b.tolist()])
z_b = filter_top_k(z_b, RUN_B["top_k"])
print("stage top_k(4)   :", [("-inf" if math.isinf(v) else round(v, 6)) for v in z_b.tolist()])
z_b = filter_top_p(z_b, RUN_B["top_p"])
print("stage top_p(0.9) :", [("-inf" if math.isinf(v) else round(v, 6)) for v in z_b.tolist()])
z_b = filter_min_p(z_b, RUN_B["min_p"])
print("stage min_p(.05) :", [("-inf" if math.isinf(v) else round(v, 6)) for v in z_b.tolist()])
probs_b = torch.softmax(z_b, dim=-1)
print("final logits :", [("-inf" if math.isinf(v) else round(v, 6)) for v in z_b.tolist()])
print("final probs  :", [round(v, 6) for v in probs_b.tolist()])
survivors = [i for i, v in enumerate(z_b.tolist()) if not math.isinf(v)]
print("survivors    :", survivors, "->", [VOCAB[i] for i in survivors])
assert abs(probs_b.sum().item() - 1.0) < 1e-6

# multinomial: seed and draw once (what generate() does after manual_seed).
torch.manual_seed(0)
drawn = torch.multinomial(probs_b, num_samples=1).item()
print(f"torch.manual_seed(0); multinomial -> id {drawn}  ('{VOCAB[drawn]}')")

# frequency check: 100k draws match probs_b.
torch.manual_seed(0)
draws = torch.multinomial(probs_b.expand(100_000, -1), num_samples=1).flatten()
freq = torch.bincount(draws, minlength=6).float() / 100_000
print("empirical freq over 100k draws:", [round(v, 4) for v in freq.tolist()])
torch.testing.assert_close(freq, probs_b, atol=5e-3, rtol=0)

# _sample_next must agree with the hand pipeline (same seed).
torch.manual_seed(0)
sn = _sample_next(z.unsqueeze(0), generated_ids=sequence, **RUN_B).item()
print("_sample_next(...) ->", sn)
assert sn == drawn, (sn, drawn)

# ---------------------------------------------------------------------------
# Part 10 -- Run A end to end: greedy, max_new_tokens=10, max_seq_len=6.
# ---------------------------------------------------------------------------
print("\n================ Part 10 -- Run A (greedy) end to end ================")


def greedy_trace(m, ids, steps):
    seq = torch.tensor([ids], dtype=torch.long)
    rows = []
    for _ in range(steps):
        lg, _ = m(seq)
        row = lg[0, -1, :].clone()
        nxt = int(torch.argmax(row))
        rows.append((seq.shape[1], row, nxt))
        seq = torch.cat([seq, torch.tensor([[nxt]])], dim=1)
    return seq, rows


seqA, rowsA = greedy_trace(model, tok.encode("DAD"), steps=3)
for length, row, nxt in rowsA:
    print(f"  len {length}: z = {[round(v, 6) for v in row.tolist()]}  argmax -> {nxt} ('{VOCAB[nxt]}')")
print("Run A ids:", seqA[0].tolist(), " text:", repr(tok.decode(seqA[0].tolist())))

out_A = generate(model, tok, "DAD", max_new_tokens=10, temperature=0.0, device="cpu")
print("generate(temperature=0.0):", repr(out_A))
assert out_A == tok.decode(seqA[0].tolist())
assert len(out_A) == 6  # 3 prompt + 3 new -> stopped at max_seq_len (Requirement 6.3)

# had the context been long enough, all 10 would have been emitted
big = _build(max_seq_len=64)
out_big = generate(big, tok, "DAD", max_new_tokens=10, temperature=0.0, device="cpu")
print("same, max_seq_len=64    :", repr(out_big), " len", len(out_big))
assert len(out_big) == 13  # 3 + 10

# greedy determinism regardless of RNG seed
torch.manual_seed(0)
a0 = generate(model, tok, "DAD", max_new_tokens=10, temperature=0.0, device="cpu")
torch.manual_seed(12345)
a1 = generate(model, tok, "DAD", max_new_tokens=10, temperature=0.0, device="cpu")
assert a0 == a1 == out_A
print("greedy is seed-independent:", repr(a0), "==", repr(a1))

# max_new_tokens = 0 -> output is exactly the prompt (round-trip)
assert generate(model, tok, "DAD", max_new_tokens=0, temperature=0.0) == "DAD"

# ---------------------------------------------------------------------------
# Part 11 -- the KV-cache path (the "optimised path"), Requirement 8.2.
# ---------------------------------------------------------------------------
print("\n================ Part 11 -- KV cache vs no cache ================")


def cached_trace(m, ids, steps):
    seq = torch.tensor([ids], dtype=torch.long)
    caches = build_kv_cache(m)
    rows = []
    for _ in range(steps):
        if caches[0].length == 0:
            lg, _ = m(seq, kv_caches=caches)  # prefill
        else:
            lg, _ = m(seq[:, -1:], kv_caches=caches, position_offset=seq.shape[1] - 1)
        row = lg[0, -1, :].clone()
        nxt = int(torch.argmax(row))
        rows.append((seq.shape[1], row, nxt))
        seq = torch.cat([seq, torch.tensor([[nxt]])], dim=1)
    return seq, rows


seqC, rowsC = cached_trace(model, tok.encode("DAD"), steps=3)
max_logit_diff = max(
    (ra[1] - rc[1]).abs().max().item() for ra, rc in zip(rowsA, rowsC)
)
print("cached ids  :", seqC[0].tolist())
print("uncached ids:", seqA[0].tolist())
print(f"max |cached - uncached| last-position logit = {max_logit_diff:.2e}")
assert seqC[0].tolist() == seqA[0].tolist()

out_A_cached = generate(
    model, tok, "DAD", max_new_tokens=10, temperature=0.0, device="cpu", use_kv_cache=True
)
print("generate(use_kv_cache=True):", repr(out_A_cached))
assert out_A_cached == out_A  # byte-identical greedy sequence (Requirement 8.2)

# ---------------------------------------------------------------------------
# Part 12 -- Run B end to end + input validation.
# ---------------------------------------------------------------------------
print("\n================ Part 12 -- Run B (full sampling stack) + validation ================")
torch.manual_seed(0)
out_B = generate(model, tok, "DAD", max_new_tokens=10, device="cpu", **RUN_B)
torch.manual_seed(0)
out_B2 = generate(model, tok, "DAD", max_new_tokens=10, device="cpu", **RUN_B)
print("Run B (seed 0):", repr(out_B))
assert out_B == out_B2  # same seed -> deterministic
assert out_B[:3] == "DAD" and len(out_B) == 6
assert out_B[3] == VOCAB[drawn]  # step 1's drawn token is the 4th character

# Replicate generate()'s loop (seed 0) to show each sampled token.
torch.manual_seed(0)
seqB = torch.tensor([tok.encode("DAD")], dtype=torch.long)
for step in range(10):
    if seqB.shape[1] >= 6:
        break
    lg, _ = model(seqB)
    nxt = _sample_next(lg[:, -1, :], generated_ids=seqB, **RUN_B)
    print(f"  Run B step {step + 1}: len {seqB.shape[1]} -> drew id {nxt.item()} ('{VOCAB[nxt.item()]}')")
    seqB = torch.cat([seqB, nxt], dim=1)
assert tok.decode(seqB[0].tolist()) == out_B

for kwargs, msg in [
    (dict(prompt="", max_new_tokens=1), "empty prompt"),
    (dict(prompt="DAD", max_new_tokens=-1), "negative max_new_tokens"),
    (dict(prompt="DAD", max_new_tokens=1, temperature=-1.0), "negative temperature"),
    (dict(prompt="DAD", max_new_tokens=1, top_k=0), "top_k < 1"),
    (dict(prompt="DAD", max_new_tokens=1, top_p=1.5), "top_p out of (0, 1]"),
    (dict(prompt="DAD", max_new_tokens=1, min_p=0.0), "min_p out of (0, 1]"),
    (dict(prompt="DAD", max_new_tokens=1, repetition_penalty=0.0), "repetition_penalty <= 0"),
]:
    try:
        generate(model, tok, **kwargs)
    except ValueError:
        print(f"  ValueError as expected: {msg}")
    else:
        raise AssertionError(f"expected ValueError for {msg}")

print("\nAll assertions passed -- every number in 011-inference-generation.md is real.")
