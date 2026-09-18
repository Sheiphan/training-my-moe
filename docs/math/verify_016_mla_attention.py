"""Reproduces every hand-computed number in `016-mla-attention.md`.

Run it:  uv run python docs/math/verify_016_mla_attention.py

ONE running example, carried end to end: the smallest `MLAAttention`
configuration that still exercises every behaviour of the component --

    d_model=8, n_heads=2, head_dim=4, rope_head_dim=2, mla_latent_dim=2,
    seq_len=3, batch=1, rope_theta=10000.

Two heads (so the *one* shared latent `c_kv` really is up-projected
per-head, and the *one* shared decoupled-RoPE key really is broadcast to
both heads); `mla_latent_dim=2 < d_model=8` (so the compression is real);
three tokens at positions 0, 1, 2 (so RoPE rotates by 0, 1, 2 rad and the
causal mask/softmax/weighted-sum are all non-trivial).

It pins the seven hand-picked weight matrices from the doc into a REAL
`qwenmoe.model.attention.MLAAttention`, runs BOTH code paths
(`forward_with_trace`, the explicit matmul/mask/softmax/matmul steps, and
`forward`, the fused `F.scaled_dot_product_attention` path), prints every
intermediate tensor, and asserts:

  * `c_kv` (the compressed cache entry) is exactly 2 numbers per token,
  * `w_uk` / `w_uv` reconstruct the full 4-wide per-head K and V for both
    heads from that 2-number latent,
  * the decoupled-RoPE key is ONE shared slice, identical across heads,
  * the same-position RoPE dot product is preserved exactly
    (q_rope . k_rope at (1,1) and (2,2) equals the pre-rotation dot),
  * the RoPE contribution to an off-diagonal score depends only on the
    position gap j - i,
  * the by-hand end-to-end output tensor, to 5 decimal places,
  * `kv_cache_elements_per_token == mla_latent_dim + rope_head_dim == 4`,
    smaller than the equivalent full-MHA / GQA cache,
  * the fused and explicit paths agree (prints the measured max |diff|).

No fabricated numbers -- this file is the source of truth the prose quotes.
"""
from __future__ import annotations

import math

import torch

from qwenmoe.config import ModelConfig
from qwenmoe.model.attention import Attention, MLAAttention

torch.set_printoptions(precision=8, sci_mode=False)

# --------------------------------------------------------------------------
# The running example: config
# --------------------------------------------------------------------------
CONFIG = ModelConfig(
    vocab_size=10,
    d_model=8,
    n_heads=2,
    n_kv_heads=2,          # unused by MLAAttention; just has to be a legal GQA value
    max_seq_len=8,
    attention_variant="mla",
    mla_latent_dim=2,
    positional_scheme="nope",  # ignored: MLA always uses its own decoupled RoPE
    rope_theta=10000.0,
)

# --------------------------------------------------------------------------
# The seven hand-picked weight matrices (Part 1 of the doc).
# nn.Linear stores weight as (out_features, in_features) and computes
# y = x @ W.T, so with x = e_i (a standard basis vector) the output is
# exactly column i of W. Every input token below is a basis vector, so only
# columns 0, 1, 2 of each matrix are ever read.
# --------------------------------------------------------------------------
# w_dkv: (latent_dim=2, d_model=8)   x -> c_kv
W_DKV = torch.tensor([
    [1.0, 0, 1, 0, 0, 0, 0, 0],
    [0.0, 1, 1, 0, 0, 0, 0, 0],
])
# w_uk: (n_heads*head_dim=8, latent_dim=2)   c_kv -> k_content (both heads)
W_UK = torch.tensor([
    [1.0, 0], [0.0, 1], [1.0, 0], [0.0, 1],   # head 0
    [0.0, 1], [1.0, 0], [0.0, 1], [1.0, 0],   # head 1
])
# w_uv: (n_heads*head_dim=8, latent_dim=2)   c_kv -> v (both heads)
W_UV = torch.tensor([
    [1.0, 0], [1.0, 0], [0.0, 1], [0.0, 1],   # head 0
    [1.0, 0], [0.0, 1], [0.0, 1], [1.0, 0],   # head 1
])
# w_q: (n_heads*head_dim=8, d_model=8)   x -> q_content (both heads)
W_Q = torch.tensor([
    [1.0, 0, 1, 0, 0, 0, 0, 0],
    [0.0, 1, 1, 0, 0, 0, 0, 0],
    [1.0, 0, 0, 0, 0, 0, 0, 0],
    [0.0, 1, 0, 0, 0, 0, 0, 0],
    [0.0, 1, 0, 0, 0, 0, 0, 0],
    [1.0, 0, 0, 0, 0, 0, 0, 0],
    [0.0, 1, 1, 0, 0, 0, 0, 0],
    [1.0, 0, 1, 0, 0, 0, 0, 0],
])
# w_qr: (n_heads*rope_head_dim=4, d_model=8)   x -> q_rope (both heads), pre-rotation
W_QR = torch.tensor([
    [1.0, 1, 1, 0, 0, 0, 0, 0],   # head 0, rope dim 0
    [0.0, 0, 1, 0, 0, 0, 0, 0],   # head 0, rope dim 1
    [0.0, 0, 1, 0, 0, 0, 0, 0],   # head 1, rope dim 0
    [1.0, 1, 0, 0, 0, 0, 0, 0],   # head 1, rope dim 1
])
# w_kr: (rope_head_dim=2, d_model=8)   x -> k_rope (ONE slice, shared), pre-rotation
W_KR = torch.tensor([
    [1.0, 0, 1, 0, 0, 0, 0, 0],
    [0.0, 1, 1, 0, 0, 0, 0, 0],
])
# w_o: (d_model=8, n_heads*head_dim=8)   concat(heads) -> output.  Identity.
W_O = torch.eye(8)

# Input: three tokens, each a standard basis vector (shape (1, 3, 8)).
X = torch.zeros(1, 3, 8)
X[0, 0, 0] = 1.0   # x_0 = e_0
X[0, 1, 1] = 1.0   # x_1 = e_1
X[0, 2, 2] = 1.0   # x_2 = e_2
POSITIONS = torch.arange(3)


def _build() -> MLAAttention:
    mla = MLAAttention(CONFIG)
    with torch.no_grad():
        mla.w_dkv.weight.copy_(W_DKV)
        mla.w_uk.weight.copy_(W_UK)
        mla.w_uv.weight.copy_(W_UV)
        mla.w_q.weight.copy_(W_Q)
        mla.w_qr.weight.copy_(W_QR)
        mla.w_kr.weight.copy_(W_KR)
        mla.w_o.weight.copy_(W_O)
    return mla


def _section(title: str) -> None:
    print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)


mla = _build()

# --------------------------------------------------------------------------
# Walk forward_with_trace step by step, re-deriving each intermediate.
# --------------------------------------------------------------------------
_section("Step 1  --  compress once:  c_kv = w_dkv(x)   (attention.py:379)")
c_kv = mla.w_dkv(X)
print("c_kv  shape", tuple(c_kv.shape), "  (batch, seq_len, latent_dim)")
print(c_kv)
assert c_kv.shape == (1, 3, 2)
torch.testing.assert_close(c_kv[0], torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]))

_section("Step 2  --  up-project K and V, per head, from that latent")
k_content = mla.w_uk(c_kv).view(1, 3, 2, 4).transpose(1, 2)
v = mla.w_uv(c_kv).view(1, 3, 2, 4).transpose(1, 2)
print("k_content  shape", tuple(k_content.shape), "  (batch, n_heads, seq_len, head_dim)")
print(k_content)
print("v          shape", tuple(v.shape))
print(v)
# reconstructed from just latent_dim=2 numbers per token:
torch.testing.assert_close(
    k_content[0, 0], torch.tensor([[1.0, 0, 1, 0], [0, 1, 0, 1], [1, 1, 1, 1]])
)
torch.testing.assert_close(
    k_content[0, 1], torch.tensor([[0.0, 1, 0, 1], [1, 0, 1, 0], [1, 1, 1, 1]])
)
torch.testing.assert_close(
    v[0, 0], torch.tensor([[1.0, 1, 0, 0], [0, 0, 1, 1], [1, 1, 1, 1]])
)
torch.testing.assert_close(
    v[0, 1], torch.tensor([[1.0, 0, 0, 1], [0, 1, 1, 0], [1, 1, 1, 1]])
)

_section("Step 3  --  plain query content:  q_content = w_q(x)   (attention.py:383)")
q_content = mla.w_q(X).view(1, 3, 2, 4).transpose(1, 2)
print("q_content  shape", tuple(q_content.shape))
print(q_content)
torch.testing.assert_close(
    q_content[0, 0], torch.tensor([[1.0, 0, 1, 0], [0, 1, 0, 1], [1, 1, 0, 0]])
)
torch.testing.assert_close(
    q_content[0, 1], torch.tensor([[0.0, 1, 0, 1], [1, 0, 1, 0], [0, 0, 1, 1]])
)

_section("Step 4  --  decoupled-RoPE slices, pre-rotation   (attention.py:385-386)")
q_rope_pre = mla.w_qr(X).view(1, 3, 2, 2).transpose(1, 2)
k_rope_pre = mla.w_kr(X).view(1, 3, 1, 2).transpose(1, 2)
print("q_rope (pre)  shape", tuple(q_rope_pre.shape), "  (batch, n_heads, seq_len, rope_head_dim)")
print(q_rope_pre)
print("k_rope (pre)  shape", tuple(k_rope_pre.shape), "  (batch, 1, seq_len, rope_head_dim) -- ONE shared slice")
print(k_rope_pre)
torch.testing.assert_close(
    q_rope_pre[0, 0], torch.tensor([[1.0, 0], [1, 0], [1, 1]])
)
torch.testing.assert_close(
    q_rope_pre[0, 1], torch.tensor([[0.0, 1], [0, 1], [1, 0]])
)
torch.testing.assert_close(
    k_rope_pre[0, 0], torch.tensor([[1.0, 0], [0, 1], [1, 1]])
)

_section("Step 5  --  rotate the RoPE slices   (attention.py:387, positional.py:41-50)")
inv_freq = mla.rope.inv_freq
print("inv_freq =", inv_freq, " (rope_head_dim=2 -> exactly one frequency, = 1.0 exactly)")
angles = mla.rope.positional_trace(POSITIONS)
print("angles (seq_len, rope_head_dim) = position in radians, repeated:")
print(angles)
q_rope, k_rope = mla.rope.apply_to_qk(q_rope_pre, k_rope_pre, POSITIONS)
print("q_rope (rotated)  shape", tuple(q_rope.shape))
print(q_rope)
print("k_rope (rotated)  shape", tuple(k_rope.shape))
print(k_rope)

cos1, sin1 = math.cos(1.0), math.sin(1.0)
cos2, sin2 = math.cos(2.0), math.sin(2.0)
print(f"cos1={cos1:.7f} sin1={sin1:.7f}  cos2={cos2:.7f} sin2={sin2:.7f}")
# hand-rotated values
torch.testing.assert_close(q_rope[0, 0, 1], torch.tensor([cos1, sin1]), atol=1e-6, rtol=0)
torch.testing.assert_close(q_rope[0, 1, 1], torch.tensor([-sin1, cos1]), atol=1e-6, rtol=0)
torch.testing.assert_close(
    q_rope[0, 0, 2], torch.tensor([cos2 - sin2, cos2 + sin2]), atol=1e-6, rtol=0
)
torch.testing.assert_close(q_rope[0, 1, 2], torch.tensor([cos2, sin2]), atol=1e-6, rtol=0)
torch.testing.assert_close(k_rope[0, 0, 1], torch.tensor([-sin1, cos1]), atol=1e-6, rtol=0)
torch.testing.assert_close(
    k_rope[0, 0, 2], torch.tensor([cos2 - sin2, cos2 + sin2]), atol=1e-6, rtol=0
)

_section("Step 6  --  broadcast the ONE shared key slice to all heads   (attention.py:389)")
k_rope_b = k_rope.expand(-1, 2, -1, -1)
print("k_rope broadcast  shape", tuple(k_rope_b.shape))
assert torch.equal(k_rope_b[0, 0], k_rope_b[0, 1]), "the decoupled key is identical across heads"
print("head 0 slice == head 1 slice:  True")

_section("Step 7  --  concat content + rope   (attention.py:391-392)")
q = torch.cat([q_content, q_rope], dim=-1)
k = torch.cat([k_content, k_rope_b], dim=-1)
print("q  shape", tuple(q.shape), "  (batch, n_heads, seq_len, head_dim + rope_head_dim = 6)")
print(q)
print("k  shape", tuple(k.shape))
print(k)

_section("Step 8  --  RoPE preservation / relative-position properties")
# same-position dot product preserved exactly (orthogonal transform):
for (h, i) in [(0, 1), (1, 1), (0, 2), (1, 2)]:
    rot = torch.dot(q_rope[0, h, i], k_rope[0, 0, i]).item()
    pre = torch.dot(q_rope_pre[0, h, i], k_rope_pre[0, 0, i]).item()
    print(f"  head {h}, position {i}: rotated q_rope.k_rope = {rot:+.7f}   pre-rotation = {pre:+.7f}")
    assert abs(rot - pre) < 1e-6
# off-diagonal (i=2, j=1): RoPE contribution depends only on the gap j - i = -1
rope_21_h0 = torch.dot(q_rope[0, 0, 2], k_rope[0, 0, 1]).item()
# predicted from the gap:  q_pre(2) . R(-1) k_pre(1)
Rm1 = torch.tensor([[math.cos(1.0), math.sin(1.0)], [-math.sin(1.0), math.cos(1.0)]])
pred = torch.dot(q_rope_pre[0, 0, 2], Rm1 @ k_rope_pre[0, 0, 1]).item()
print(f"  head 0 score-rope term (i=2,j=1): direct = {rope_21_h0:+.7f}   via R(j-i=-1) = {pred:+.7f}")
assert abs(rope_21_h0 - pred) < 1e-6

_section("Step 9  --  scaled scores  q.k^T / sqrt(head_dim + rope_head_dim)   (attention.py:395)")
scale = math.sqrt(4 + 2)
print(f"scale = sqrt(6) = {scale:.7f}")
scores = (q @ k.transpose(-2, -1)) / scale
print("scores  shape", tuple(scores.shape), "  (batch, n_heads, seq_len, seq_len)")
print(scores)

_section("Step 10 --  causal mask + softmax   (attention.py:397-404)")
causal = torch.triu(torch.full((3, 3), float("-inf")), diagonal=1)
probs = torch.softmax(scores + causal, dim=-1)
print("probs  shape", tuple(probs.shape))
print(probs)

_section("Step 10b --  softmax, digit by digit (the doc's by-hand exponentials)")
masked = scores + causal
for h in range(2):
    for i in range(3):
        row = masked[0, h, i]
        finite = row[torch.isfinite(row)]
        shifted = finite - finite.max()
        exps = torch.exp(shifted)
        probs_row = exps / exps.sum()
        print(f"  head {h} row {i}: shifted={shifted.tolist()}")
        print(f"             exp   ={[round(e, 8) for e in exps.tolist()]}  sum={exps.sum().item():.8f}")
        print(f"             probs ={[round(p, 8) for p in probs_row.tolist()]}")
        torch.testing.assert_close(probs_row, probs[0, h, i][torch.isfinite(row)], atol=1e-6, rtol=0)

_section("Step 11 --  weighted sum with V (content only)   (attention.py:405)")
ctx = probs @ v
print("context  shape", tuple(ctx.shape), "  (batch, n_heads, seq_len, head_dim)")
print(ctx)

_section("Step 12 --  concat heads + output projection   (attention.py:407-408)")
merged = ctx.transpose(1, 2).contiguous().view(1, 3, 8)
print("merged (concat heads)  shape", tuple(merged.shape))
print(merged)
output_manual = mla.w_o(merged)
print("output = w_o(merged)  shape", tuple(output_manual.shape))
print(output_manual)

# --------------------------------------------------------------------------
# Run BOTH real code paths and compare
# --------------------------------------------------------------------------
_section("Real module: forward_with_trace vs fused forward")
out_trace, trace = mla.forward_with_trace(X, POSITIONS)
out_fused = mla.forward(X, POSITIONS)
print("forward_with_trace output:\n", out_trace)
print("fused forward output:\n", out_fused)
max_diff = (out_trace - out_fused).abs().max().item()
print(f"max |trace - fused| = {max_diff:.3e}")

# our step-by-step reconstruction must equal the real trace path exactly
torch.testing.assert_close(output_manual, out_trace, atol=1e-6, rtol=0)

# --------------------------------------------------------------------------
# The by-hand end-to-end result (doc Part 2, final box)
# --------------------------------------------------------------------------
_section("ASSERT the by-hand end-to-end output")
EXPECTED = torch.tensor([[
    [1.000000, 1.000000, 0.000000, 0.000000, 1.000000, 0.000000, 0.000000, 1.000000],
    [0.355278, 0.355278, 0.644722, 0.644722, 0.172463, 0.827537, 0.827537, 0.172463],
    [0.693924, 0.693924, 0.898646, 0.898646, 0.687812, 0.813172, 0.813172, 0.687812],
]])
print("expected (hand-computed):\n", EXPECTED)
torch.testing.assert_close(out_trace, EXPECTED, atol=1e-5, rtol=0)
torch.testing.assert_close(out_fused, EXPECTED, atol=1e-5, rtol=0)

# --------------------------------------------------------------------------
# KV-cache size property (Requirement 12.4)
# --------------------------------------------------------------------------
_section("KV-cache size: MLA vs equivalent GQA / full MHA")
mla_cache = mla.kv_cache_elements_per_token
print(f"MLA cache/token/layer = latent_dim + rope_head_dim = {CONFIG.mla_latent_dim} + "
      f"{mla.rope_head_dim} = {mla_cache}")
assert mla_cache == CONFIG.mla_latent_dim + mla.rope_head_dim == 4

gqa = Attention(CONFIG)  # same d_model, n_heads; n_kv_heads=2
gqa_cache = 2 * CONFIG.n_kv_heads * gqa.head_dim  # K + V, each n_kv_heads * head_dim
mha_cache = 2 * CONFIG.n_heads * gqa.head_dim
print(f"GQA cache/token/layer (n_kv_heads={CONFIG.n_kv_heads}) = 2 * {CONFIG.n_kv_heads} * "
      f"{gqa.head_dim} = {gqa_cache}")
print(f"full-MHA cache/token/layer                = 2 * {CONFIG.n_heads} * {gqa.head_dim} = {mha_cache}")
print(f"MLA is {mla_cache / gqa_cache:.1%} of GQA, {mla_cache / mha_cache:.1%} of full MHA")
assert mla_cache < gqa_cache <= mha_cache

# --------------------------------------------------------------------------
# Parameter counts (doc's cost section)
# --------------------------------------------------------------------------
_section("Parameter counts")
mla_params = sum(p.numel() for p in mla.parameters())
gqa_params = sum(p.numel() for p in gqa.parameters())
for name, p in mla.named_parameters():
    print(f"  MLA {name:16s} {tuple(p.shape)}  = {p.numel()}")
print(f"MLA total attn params = {mla_params}")
print(f"GQA total attn params = {gqa_params}")

print("\nAll assertions passed -- every number in 016-mla-attention.md is real.")
