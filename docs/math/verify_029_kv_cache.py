"""Reproduces every hand-computed number in `029-kv-cache.md`.

Run it:  uv run python docs/math/verify_029_kv_cache.py

Five sections, each pinning weights into the REAL `qwenmoe` modules:

  A. `LayerKVCache` (GQA) -- reuses doc 004's exact 3-token example
     (`d_model=4, n_heads=2, n_kv_heads=1, head_dim=2, positional_scheme=
     "rope"`), extends it with a 4th token fed as one incremental decode
     step, and shows the cached step reproduces a full 4-token recompute.
  B. `LatentKVCache` (MLA) -- the same extension applied to doc 016's
     exact 3-token example (`d_model=8, n_heads=2, head_dim=4,
     mla_latent_dim=2, rope_head_dim=2`).
  C. `build_kv_cache` -- picks `LayerKVCache`/`LatentKVCache` per block
     matching `block.attn`'s real type.
  D. The wiring above `Attention`/`MLAAttention` -- `QwenMoEModel.forward`'s
     `position_offset` arithmetic and `generate()`'s branch selection,
     checked against a full uncached recompute AND against real greedy
     `generate()` output (cached vs uncached must be token-identical).
  E. Measured wall-clock: cached vs uncached `generate()` at two lengths,
     on a slightly larger model, to show the speed-up grows with length.

No fabricated numbers -- every value printed here is the source of truth
`029-kv-cache.md` quotes.
"""
from __future__ import annotations

import math
import time

import torch

from qwenmoe.config import ModelConfig
from qwenmoe.model.attention import Attention, MLAAttention, _causal_position_mask
from qwenmoe.model.generate import generate
from qwenmoe.model.kv_cache import LatentKVCache, LayerKVCache, build_kv_cache
from qwenmoe.model.transformer import QwenMoEModel
from qwenmoe.tokenizer.char import CharTokenizer

torch.set_printoptions(precision=6, sci_mode=False)


def _section(title: str) -> None:
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


# ============================================================================
# 0. The pure counting argument: O(n) vs O(n^2)/O(n^3), on this doc's own
#    prompt length (p=3, matching doc 004/016's 3-token example) and a few
#    generation lengths g. Per generated step at running length L: without a
#    cache the whole length-L sequence is reprocessed (L k/v projections,
#    an L*L score matrix); with a cache only the 1 new token is projected,
#    against a length-L row of scores.
# ============================================================================
_section("0. Pure counting: cumulative work generating g tokens from a 3-token prompt")
print(f"{'g':>4} {'uncached proj':>14} {'uncached scores':>16} "
      f"{'cached proj':>12} {'cached scores':>14}")
p = 3
for g in (1, 5, 10):
    lengths = range(p + 1, p + 1 + g)  # running length L at each of the g steps
    uncached_proj = sum(lengths)
    uncached_scores = sum(L * L for L in lengths)
    cached_proj = g
    cached_scores = sum(lengths)
    print(f"{g:>4} {uncached_proj:>14} {uncached_scores:>16} "
          f"{cached_proj:>12} {cached_scores:>14}")
    assert cached_proj <= uncached_proj and cached_scores <= uncached_scores

# ============================================================================
# A. LayerKVCache (GQA) -- doc 004's example, extended by one token
# ============================================================================
_section("A. LayerKVCache -- doc 004's GQA example + one incremental token")

GQA_CONFIG = ModelConfig(
    vocab_size=8,
    d_model=4,
    n_heads=2,
    n_kv_heads=1,
    max_seq_len=16,
    positional_scheme="rope",
    rope_theta=10000.0,
)

# Doc 004 Part 1.4, unchanged.
WQ = torch.tensor([[1.0, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1], [1, 0, 0, 1]])
WK = torch.tensor([[1.0, 0, 1, 0], [0, 1, 0, 1]])
WV = torch.tensor([[1.0, 1, 0, 0], [0, 0, 1, 1]])
WO = torch.tensor([[1.0, 0, 1, 0], [0, 1, 0, 1], [1, 0, 0, 1], [0, 1, 1, 0]])

# Doc 004's 3 tokens, plus a 4th ("token 3") this doc adds.
X3 = torch.tensor([[[1.0, 0, 1, 0], [0, 2, 0, 1], [1, 1, 0, 0]]])  # (1, 3, 4)
X_NEW = torch.tensor([[[1.0, 0, 1, 1]]])  # (1, 1, 4) -- token 3


def _build_attn() -> Attention:
    attn = Attention(GQA_CONFIG)
    with torch.no_grad():
        attn.w_q.weight.copy_(WQ)
        attn.w_k.weight.copy_(WK)
        attn.w_v.weight.copy_(WV)
        attn.w_o.weight.copy_(WO)
    return attn.eval()


attn = _build_attn()
positions3 = torch.arange(3)
positions1 = torch.arange(3, 4)  # position_offset=3, one query

# --- raw (pre-RoPE) q/k/v for all 4 tokens, computed the same way forward()
# computes them internally (same nn.Linear + view + transpose calls), so we
# can print/verify intermediates without touching the cache's private state.
X4 = torch.cat([X3, X_NEW], dim=1)  # (1, 4, 4)
positions4 = torch.arange(4)
with torch.no_grad():
    q_all = attn.w_q(X4).view(1, 4, 2, 2).transpose(1, 2)  # (1,2,4,2)
    k_all = attn.w_k(X4).view(1, 4, 1, 2).transpose(1, 2)  # (1,1,4,2)
    v_all = attn.w_v(X4).view(1, 4, 1, 2).transpose(1, 2)  # (1,1,4,2)
    q_rot_all, k_rot_all = attn.positional.apply_to_qk(q_all, k_all, positions4)

print("raw q (head0, head1) per token:")
print(q_all[0, :, :, :])
print("raw k (single kv head) per token:")
print(k_all[0, 0])
print("raw v (single kv head) per token:")
print(v_all[0, 0])
print("\nRoPE-rotated q (head0, head1) per token:")
print(q_rot_all[0])
print("RoPE-rotated k (single kv head) per token:")
print(k_rot_all[0, 0])

# Token 3's raw q/k/v in isolation, as a sanity check that slicing the batch
# call above is identical to a genuine length-1 call at position 3.
with torch.no_grad():
    q3_solo = attn.w_q(X_NEW).view(1, 1, 2, 2).transpose(1, 2)
    k3_solo = attn.w_k(X_NEW).view(1, 1, 1, 2).transpose(1, 2)
    v3_solo = attn.w_v(X_NEW).view(1, 1, 1, 2).transpose(1, 2)
    q3_rot_solo, k3_rot_solo = attn.positional.apply_to_qk(q3_solo, k3_solo, positions1)
torch.testing.assert_close(q3_rot_solo[0, :, 0, :], q_rot_all[0, :, 3, :])
torch.testing.assert_close(k3_rot_solo[0, 0, 0, :], k_rot_all[0, 0, 3, :])
print("\ntoken-3-in-isolation raw/rotated q,k == token-3-slice-of-the-4-token-batch: OK")

# --- Prefill: 3 tokens through Attention.forward with an (empty) cache.
cache = LayerKVCache()
assert cache.length == 0 and cache.k is None
out_prefill_cached = attn(X3, positions3, kv_cache=cache)
out_prefill_nocache = attn(X3, positions3)  # no cache at all, for comparison
print("\nprefill output (kv_cache=LayerKVCache()) :")
print(out_prefill_cached[0])
diff_prefill = (out_prefill_cached - out_prefill_nocache).abs().max().item()
print(f"max |prefill_with_cache - prefill_without_cache| = {diff_prefill:.3e}")
assert diff_prefill == 0.0, "passing an empty cache must not change the prefill's own output"

print(f"\ncache.length after prefill = {cache.length}")
print("cache.k shape", tuple(cache.k.shape), " cache.v shape", tuple(cache.v.shape))
print("cache.k[0,0] (the single kv head, 3 rows) =")
print(cache.k[0, 0])
torch.testing.assert_close(cache.k[0, 0], k_rot_all[0, 0, :3, :])
print(f"cache.nbytes after prefill = {cache.nbytes}  "
      f"(= 2 tensors * 1*1*3*2 floats * 4 bytes = {2 * 1 * 1 * 3 * 2 * 4})")

# --- Incremental step: feed ONLY token 3, position_offset=3.
out_incremental = attn(X_NEW, positions1, kv_cache=cache)
print(f"\ncache.length after the incremental step = {cache.length}")
print("cache.k[0,0] (now 4 rows) =")
print(cache.k[0, 0])
torch.testing.assert_close(cache.k[0, 0], k_rot_all[0, 0, :, :])
print(f"cache.nbytes after the incremental step = {cache.nbytes}  "
      f"(= {2 * 1 * 1 * 4 * 2 * 4})")

mask = _causal_position_mask(positions1, kv_len=4, device=X_NEW.device, dtype=torch.float32)
print("\n_causal_position_mask(positions=[3], kv_len=4) =")
print(mask)
assert torch.equal(mask, torch.zeros_like(mask)), "query at position 3 must see all 4 keys unmasked"

print("\nincremental-step output (1 query, 4 cached keys) =")
print(out_incremental[0])

# --- Manually replicate the masked-softmax-weighted-sum SDPA performs
# internally (same q_rot/k_rot/v tensors `forward` used -- just computed
# outside the fused kernel so every intermediate is printable, the same
# technique docs 004/021 use for their `forward_with_trace` walk).
q3_rot = q_rot_all[:, :, 3:4, :]          # (1, 2, 1, 2) -- both heads' query
k_cache_rot = k_rot_all                    # (1, 1, 4, 2) -- all 4 cached keys
v_cache = v_all                            # (1, 1, 4, 2)
scale = 1.0 / math.sqrt(2)
raw_scores = torch.einsum("bhqd,bkjd->bhqj", q3_rot, k_cache_rot).squeeze(2)  # (1,2,4)
scaled_scores = raw_scores * scale
probs = torch.softmax(scaled_scores, dim=-1)
weighted = torch.einsum("bhj,bkjd->bhd", probs, v_cache)  # (1, 2, 2)
print("\nmanually-replicated incremental step (matches SDPA's fused result):")
print("  raw scores (head0, head1) x 4 keys =", raw_scores[0].tolist())
print("  scaled scores (/sqrt(2))           =", scaled_scores[0].tolist())
print("  softmax weights                    =", probs[0].tolist())
print("  per-head weighted-sum output        =", weighted[0].tolist())
concat = weighted.unsqueeze(2).transpose(1, 2).reshape(1, 1, 4)
manual_out = attn.w_o(concat)
diff_manual = (manual_out - out_incremental).abs().max().item()
print(f"  after concat + w_o: {manual_out[0,0].tolist()}")
print(f"  max |manual - attn.forward's cached output| = {diff_manual:.3e}")
assert diff_manual < 1e-5

# --- Compare against a full, uncached recompute over all 4 tokens at once.
attn_full = _build_attn()  # fresh module, identical weights, no cache ever touched
out_full = attn_full(X4, positions4)
diff_incremental = (out_full[:, -1:, :] - out_incremental).abs().max().item()
print(f"\nfull 4-token recompute, row 3 = {out_full[0, -1].tolist()}")
print(f"max |incremental - full_recompute[row 3]| = {diff_incremental:.3e}")
assert diff_incremental < 1e-5

# --- Aside: ALiBi's incremental-step bias (attention_bias_positions on a
# genuinely non-square 1-query x 4-key case, not shown in doc 004/021).
from qwenmoe.model.positional import AlibiBias  # noqa: E402

alibi = AlibiBias(n_heads=2)
alibi_bias_incremental = alibi.attention_bias_positions(positions1, torch.arange(4))
print("\nALiBi aside: attention_bias_positions(q_positions=[3], k_positions=[0,1,2,3]) =")
print(alibi_bias_incremental)
expected_h0 = torch.tensor([[(j - 3) * 0.0625 for j in range(4)]])
torch.testing.assert_close(alibi_bias_incremental[0], expected_h0, atol=1e-6, rtol=0)

# ============================================================================
# B. LatentKVCache (MLA) -- doc 016's example, extended by one token
# ============================================================================
_section("B. LatentKVCache -- doc 016's MLA example + one incremental token")

MLA_CONFIG = ModelConfig(
    vocab_size=10,
    d_model=8,
    n_heads=2,
    n_kv_heads=2,
    max_seq_len=16,
    attention_variant="mla",
    mla_latent_dim=2,
    positional_scheme="nope",  # ignored by MLAAttention
    rope_theta=10000.0,
)

# Doc 016 Part 1.4, unchanged (all 8 columns; columns 3-7 are 0 since the
# doc's 3 basis-vector tokens never read them -- token 3 = e_3 now does).
W_DKV = torch.tensor([[1.0, 0, 1, 0, 0, 0, 0, 0], [0.0, 1, 1, 0, 0, 0, 0, 0]])
W_UK = torch.tensor([
    [1.0, 0], [0.0, 1], [1.0, 0], [0.0, 1],
    [0.0, 1], [1.0, 0], [0.0, 1], [1.0, 0],
])
W_UV = torch.tensor([
    [1.0, 0], [1.0, 0], [0.0, 1], [0.0, 1],
    [1.0, 0], [0.0, 1], [0.0, 1], [1.0, 0],
])
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
W_QR = torch.tensor([
    [1.0, 1, 1, 0, 0, 0, 0, 0],
    [0.0, 0, 1, 0, 0, 0, 0, 0],
    [0.0, 0, 1, 0, 0, 0, 0, 0],
    [1.0, 1, 0, 0, 0, 0, 0, 0],
])
W_KR = torch.tensor([[1.0, 0, 1, 0, 0, 0, 0, 0], [0.0, 1, 1, 0, 0, 0, 0, 0]])
W_O_MLA = torch.eye(8)

X3_mla = torch.zeros(1, 3, 8)
X3_mla[0, 0, 0] = 1.0  # e0
X3_mla[0, 1, 1] = 1.0  # e1
X3_mla[0, 2, 2] = 1.0  # e2
# Doc 016's 7 weight matrices only ever populate columns 0-2 (its example
# only used 3 basis-vector tokens); token 3 = e0 + e2 keeps this doc's
# extension inside those already-meaningful columns instead of inventing a
# new, never-before-used column 3 of every matrix. (A model's input is any
# real-valued vector, not necessarily a basis vector -- doc016 only chose
# basis vectors for easy-to-read-off arithmetic.)
X_NEW_MLA = torch.zeros(1, 1, 8)
X_NEW_MLA[0, 0, 0] = 1.0
X_NEW_MLA[0, 0, 2] = 1.0  # x_3 = e0 + e2 -- token 3


def _build_mla() -> MLAAttention:
    mla = MLAAttention(MLA_CONFIG)
    with torch.no_grad():
        mla.w_dkv.weight.copy_(W_DKV)
        mla.w_uk.weight.copy_(W_UK)
        mla.w_uv.weight.copy_(W_UV)
        mla.w_q.weight.copy_(W_Q)
        mla.w_qr.weight.copy_(W_QR)
        mla.w_kr.weight.copy_(W_KR)
        mla.w_o.weight.copy_(W_O_MLA)
    return mla.eval()


mla = _build_mla()
cache_mla = LatentKVCache()
assert cache_mla.length == 0

out_prefill_mla_cached = mla(X3_mla, positions3, kv_cache=cache_mla)
out_prefill_mla_nocache = mla(X3_mla, positions3)
diff_prefill_mla = (out_prefill_mla_cached - out_prefill_mla_nocache).abs().max().item()
print(f"max |prefill_with_cache - prefill_without_cache| (MLA) = {diff_prefill_mla:.3e}")
assert diff_prefill_mla == 0.0

print(f"\ncache_mla.length after prefill = {cache_mla.length}")
print("cache_mla.c_kv shape", tuple(cache_mla.c_kv.shape), " =")
print(cache_mla.c_kv[0])
print("cache_mla.k_rope shape", tuple(cache_mla.k_rope.shape), " =")
print(cache_mla.k_rope[0, 0])
print(f"cache_mla.nbytes after prefill = {cache_mla.nbytes}  "
      f"(c_kv: 1*3*2 floats + k_rope: 1*1*3*2 floats, *4 bytes = "
      f"{(1 * 3 * 2 + 1 * 1 * 3 * 2) * 4})")

out_incremental_mla = mla(X_NEW_MLA, positions1, kv_cache=cache_mla)
print(f"\ncache_mla.length after the incremental step = {cache_mla.length}")
print("cache_mla.c_kv =")
print(cache_mla.c_kv[0])
print(f"cache_mla.nbytes after the incremental step = {cache_mla.nbytes}  "
      f"(c_kv: 1*4*2 + k_rope: 1*1*4*2 floats, *4 bytes = "
      f"{(1 * 4 * 2 + 1 * 1 * 4 * 2) * 4})")

print("\nincremental-step output (1 query, 4 cached keys, MLA) =")
print(out_incremental_mla[0])

X4_mla = torch.cat([X3_mla, X_NEW_MLA], dim=1)
mla_full = _build_mla()
out_full_mla = mla_full(X4_mla, positions4)
print(f"full 4-token recompute, row 3 (MLA) = {out_full_mla[0, -1].tolist()}")
diff_incremental_mla = (out_full_mla[:, -1:, :] - out_incremental_mla).abs().max().item()
print(f"max |incremental - full_recompute[row 3]| (MLA) = {diff_incremental_mla:.3e}")
assert diff_incremental_mla < 1e-5

print("\n--- byte comparison at matched sequence length (this doc's own shapes) ---")
for n in (3, 4):
    gqa_bytes = 2 * GQA_CONFIG.n_kv_heads * (GQA_CONFIG.d_model // GQA_CONFIG.n_heads) * n * 4
    mla_bytes = (MLA_CONFIG.mla_latent_dim + (MLA_CONFIG.d_model // MLA_CONFIG.n_heads) // 2) * n * 4
    print(f"  n={n}:  GQA(n_kv_heads={GQA_CONFIG.n_kv_heads}) = {gqa_bytes:3d} B   "
          f"MLA(latent_dim={MLA_CONFIG.mla_latent_dim}) = {mla_bytes:3d} B   "
          f"ratio = {mla_bytes / gqa_bytes:.4f}")
print("  (tie: this doc's GQA config is already n_kv_heads=1 -- as compressed as GQA")
print("   gets. MLA's advantage over an UNcompressed multi-head cache, at the")
print("   journal's own BASELINE shape, independently recomputed here:)")
for n in (16, 64, 256):
    mha_bytes = 2 * 8 * 16 * n * 4  # n_kv_heads=8 (full MHA), head_dim=16
    mla_bytes = (32 + 8) * n * 4    # mla_latent_dim=32, rope_head_dim=8
    print(f"  n={n:>3}:  MHA/GQA = {mha_bytes:>7,d} B   MLA = {mla_bytes:>6,d} B   "
          f"ratio = {mla_bytes / mha_bytes:.4f}")
assert cache.nbytes > 0 and cache_mla.nbytes > 0

# ============================================================================
# C. build_kv_cache -- picks the cache class matching block.attn's real type
# ============================================================================
_section("C. build_kv_cache -- one cache per block, matching attention_variant")

gqa_model = QwenMoEModel(ModelConfig(
    vocab_size=12, d_model=8, n_layers=2, n_heads=2, n_kv_heads=1,
    d_ff=8, n_routed_experts=2, top_k=1, n_shared_experts=0,
    max_seq_len=16, attention_variant="gqa", positional_scheme="rope",
))
mla_model = QwenMoEModel(ModelConfig(
    vocab_size=12, d_model=8, n_layers=2, n_heads=2, n_kv_heads=2,
    d_ff=8, n_routed_experts=2, top_k=1, n_shared_experts=0,
    max_seq_len=16, attention_variant="mla", mla_latent_dim=2,
))
gqa_caches = build_kv_cache(gqa_model)
mla_caches = build_kv_cache(mla_model)
print("gqa_model (2 blocks) ->", [type(c).__name__ for c in gqa_caches])
print("mla_model (2 blocks) ->", [type(c).__name__ for c in mla_caches])
assert all(isinstance(c, LayerKVCache) for c in gqa_caches)
assert all(isinstance(c, LatentKVCache) for c in mla_caches)

# ============================================================================
# D. The wiring above Attention: position_offset + generate()'s branch logic
# ============================================================================
_section("D. QwenMoEModel.forward's position_offset + generate()'s two branches")

torch.manual_seed(0)
wiring_model = QwenMoEModel(ModelConfig(
    vocab_size=16, d_model=16, n_layers=2, n_heads=4, n_kv_heads=2,
    d_ff=16, n_routed_experts=2, top_k=1, n_shared_experts=1,
    max_seq_len=32, attention_variant="gqa", positional_scheme="rope",
)).eval()

prompt_ids = torch.tensor([[3, 7, 1]])  # (1, 3) -- an arbitrary 3-token prompt
caches_D = build_kv_cache(wiring_model)
assert caches_D[0].length == 0
with torch.no_grad():
    logits_prefill, _ = wiring_model(prompt_ids, kv_caches=caches_D)  # transformer.py:113-127
assert caches_D[0].length == 3
print(f"after prefill: caches_D[0].length = {caches_D[0].length}  "
      "(generate.py:203's `caches[0].length == 0` check now takes the else branch)")

next_id = torch.tensor([[9]])  # token 3, arbitrary id
with torch.no_grad():
    logits_step, _ = wiring_model(
        next_id, kv_caches=caches_D, position_offset=prompt_ids.shape[1]
    )  # generate.py:206-208
assert caches_D[0].length == 4

full_ids = torch.cat([prompt_ids, next_id], dim=1)  # (1, 4)
with torch.no_grad():
    logits_full, _ = wiring_model(full_ids)  # no cache: one ordinary forward over all 4
diff_wiring = (logits_full[:, -1, :] - logits_step[:, -1, :]).abs().max().item()
print(f"max |cached_step_logits - full_recompute_logits[row 3]| = {diff_wiring:.3e}")
assert diff_wiring < 1e-4

# Real generate(): cached vs uncached greedy must be token-identical.
tok = CharTokenizer()
tok.train("the cat sat on the mat and the dog ran")
gen_model = QwenMoEModel(ModelConfig(
    vocab_size=tok.vocab_size, d_model=16, n_layers=2, n_heads=4, n_kv_heads=2,
    d_ff=16, n_routed_experts=2, top_k=1, n_shared_experts=1,
    max_seq_len=64, attention_variant="gqa", positional_scheme="rope",
)).eval()
out_nocache = generate(gen_model, tok, "the cat", max_new_tokens=20,
                        temperature=0.0, device="cpu", use_kv_cache=False)
out_cached = generate(gen_model, tok, "the cat", max_new_tokens=20,
                       temperature=0.0, device="cpu", use_kv_cache=True)
print(f"\ngenerate(use_kv_cache=False) = {out_nocache!r}")
print(f"generate(use_kv_cache=True)  = {out_cached!r}")
assert out_nocache == out_cached, "greedy cached/uncached generation must match token-for-token"
print("greedy cached == greedy uncached: OK")

# ============================================================================
# E. Measured wall-clock: cached vs uncached generate(), two lengths
# ============================================================================
_section("E. Measured: cached vs uncached generate() wall-clock")

torch.manual_seed(0)
timing_model = QwenMoEModel(ModelConfig(
    vocab_size=tok.vocab_size, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2,
    d_ff=64, n_routed_experts=2, top_k=1, n_shared_experts=1,
    max_seq_len=300, attention_variant="gqa", positional_scheme="rope",
)).eval()


def _time_generate(n_tokens: int, use_kv_cache: bool) -> float:
    start = time.perf_counter()
    generate(timing_model, tok, "the", max_new_tokens=n_tokens,
              temperature=0.0, device="cpu", use_kv_cache=use_kv_cache)
    return time.perf_counter() - start


print(f"{'tokens':>8} {'uncached (s)':>14} {'cached (s)':>12} {'speed-up':>10}")
for n_tokens in (50, 150, 300):
    t_uncached = _time_generate(n_tokens, use_kv_cache=False)
    t_cached = _time_generate(n_tokens, use_kv_cache=True)
    print(f"{n_tokens:>8} {t_uncached:>14.3f} {t_cached:>12.3f} {t_uncached / t_cached:>9.2f}x")

print("\nAll sections completed; every assertion above passed.")
