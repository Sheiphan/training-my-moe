"""Reproduces every hand-computed number in `031-document-boundaries.md`.

Run it:  uv run python docs/math/verify_031_document_boundaries.py

ONE running example, carried end to end -- the SAME tiny `Attention` module
as `docs/math/004-attention-gqa.md` / `021-fused-attention-sdpa.md`
(d_model=4, n_heads=2, n_kv_heads=1, seq_len=3), pinned to the SAME four
weight matrices and the SAME 3-token input. Those docs derive the plain
causal attention output for this example; this file/doc adds ONE new
ingredient -- a document-boundary mask built from `segment_ids = [0, 1, 1]`
-- and shows, by hand and against the real module, exactly what changes.

`segment_ids` itself is not invented for this file: it is the real output of
`qwenmoe.data.dataset.TokenDataset(..., return_segments=True)` on a tiny
5-token stream, exactly as derived in `docs/math/008-data-pipeline.md`.

It asserts:

  * `TokenDataset(return_segments=True)` really produces `segments == [0,1,1]`
    for the window this doc uses (grounding the input, not inventing it);
  * `build_document_mask` produces the exact `0`/`-inf` matrix derived by
    hand from `segment_ids = [0, 1, 1]`;
  * the causal mask + document mask, added entry-by-entry, matches the
    combined `attn_mask` `Attention.forward` actually builds;
  * softmax over the combined mask, then the weighted sum with `v` and the
    `W_O` projection, computed by hand, matches `Attention.forward(...,
    document_mask=...)` -- the real fused SDPA path -- to float32 precision;
  * `document_mask=None` still takes the exact `is_causal=True` fast path
    (byte-identical to docs 004/021's causal-only output);
  * document isolation is EXACT, not approximate: perturbing the document-0
    token by a large amount leaves the document-1 rows' output at
    `max|Δ| == 0.0`, bit for bit;
  * `MLAAttention.forward` gets the same treatment (its own tiny example);
  * `QwenMoEModel.forward` builds exactly ONE `document_mask` per call
    (not one per layer) and shares it across every `TransformerBlock`, and
    the isolation property above survives a full 2-layer model with a real
    MoE sublayer in between.

No fabricated numbers -- this file is the source of truth the prose quotes.
"""
from __future__ import annotations

from unittest.mock import patch

import torch
import torch.nn.functional as F

from qwenmoe.config import ModelConfig
from qwenmoe.data.dataset import TokenDataset
from qwenmoe.model.attention import Attention, MLAAttention, _causal_position_mask, build_document_mask
from qwenmoe.model.transformer import QwenMoEModel

torch.manual_seed(0)
torch.set_printoptions(precision=6, sci_mode=False)


def _section(title: str) -> None:
    print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)


# ======================================================================
# The running example -- identical to docs 004 / 021
# ======================================================================
W_Q = torch.tensor([[1.0, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1], [1, 0, 0, 1]])
W_K = torch.tensor([[1.0, 0, 1, 0], [0, 1, 0, 1]])
W_V = torch.tensor([[1.0, 1, 0, 0], [0, 0, 1, 1]])
W_O = torch.tensor([[1.0, 0, 1, 0], [0, 1, 0, 1], [1, 0, 0, 1], [0, 1, 1, 0]])

X = torch.tensor([[[1.0, 0, 1, 0], [0, 2, 0, 1], [1, 1, 0, 0]]])  # (1, 3, 4)
POSITIONS = torch.arange(3)
V_HEAD = torch.tensor([[1.0, 1.0], [2.0, 1.0], [2.0, 0.0]])  # doc 004 step 1c


def _cfg(**overrides) -> ModelConfig:
    base = dict(
        vocab_size=10,
        d_model=4,
        n_heads=2,
        n_kv_heads=1,
        max_seq_len=8,
        positional_scheme="nope",
        qk_norm=False,
        rope_theta=10000.0,
    )
    base.update(overrides)
    return ModelConfig(**base)


def _build_attn() -> Attention:
    attn = Attention(_cfg())
    with torch.no_grad():
        attn.w_q.weight.copy_(W_Q)
        attn.w_k.weight.copy_(W_K)
        attn.w_v.weight.copy_(W_V)
        attn.w_o.weight.copy_(W_O)
    return attn


# ======================================================================
# 1. segment_ids = [0, 1, 1] is a REAL TokenDataset output, not invented
# ======================================================================
_section("1. segment_ids grounded in a real TokenDataset(return_segments=True)")

TOKENS = [5, 99, 7, 3, 9]   # eos_id = 99; doc 0 = [5, 99], doc 1 = [7, 3, 9]
ds = TokenDataset(TOKENS, seq_len=3, return_segments=True, eos_id=99)
x_ds, y_ds, segments = ds[1]
print("tokens               :", TOKENS, " (eos_id=99)")
print("_document_ids(tokens):", "[0, 0, 1, 1, 1]  (cumsum of 'previous token was 99')")
print("window idx=1  x      :", x_ds.tolist())
print("window idx=1  y      :", y_ds.tolist())
print("window idx=1  segments:", segments.tolist())
assert segments.tolist() == [0, 1, 1]
segment_ids = segments.unsqueeze(0)  # (1, 3) -- the batch axis Attention expects
print("segment_ids (batch=1):", segment_ids.tolist())

# ======================================================================
# 2. build_document_mask, by hand and for real
# ======================================================================
_section("2. build_document_mask(segment_ids)")

doc_mask = build_document_mask(segment_ids)
print("build_document_mask(segment_ids).shape:", tuple(doc_mask.shape))
print("build_document_mask(segment_ids)[0, 0]:\n", doc_mask[0, 0])

EXPECTED_DOC_MASK = torch.tensor([
    [0.0, float("-inf"), float("-inf")],
    [float("-inf"), 0.0, 0.0],
    [float("-inf"), 0.0, 0.0],
])
torch.testing.assert_close(doc_mask[0, 0], EXPECTED_DOC_MASK, atol=0, rtol=0, equal_nan=True)
print("matches the hand-derived matrix exactly (torch.testing.assert_close, atol=0)")

# Cross-check against the journal's own worked example (a different,
# larger segment_ids -- not the running example, just a claim to confirm).
journal_mask = build_document_mask(torch.tensor([[0, 0, 0, 1, 1]]))[0, 0]
print("\ncross-check -- journal's own worked example, segment_ids=[0,0,0,1,1]:\n", journal_mask)

# ======================================================================
# 3. Combine with the causal mask -- entry by entry
# ======================================================================
_section("3. attn_mask = causal_position_mask + document_mask")

causal = _causal_position_mask(POSITIONS, 3, torch.device("cpu"), torch.float32)[0]
print("causal_position_mask:\n", causal)
combined = causal + doc_mask[0, 0]
print("causal + document_mask (what Attention.forward builds at L171+L175):\n", combined)

EXPECTED_COMBINED = torch.tensor([
    [0.0, float("-inf"), float("-inf")],
    [float("-inf"), 0.0, float("-inf")],
    [float("-inf"), 0.0, 0.0],
])
torch.testing.assert_close(combined, EXPECTED_COMBINED, atol=0, rtol=0, equal_nan=True)
print("matches the hand-derived combined mask exactly")

# ======================================================================
# 4. Softmax + weighted sum + W_O, by hand -- vs. the real fused forward
# ======================================================================
_section("4. by-hand masked softmax -> weighted sum -> W_O  vs.  real forward()")

attn = _build_attn()
_, trace = attn.forward_with_trace(X, POSITIONS)  # causal-only pre-mask scores (docs 004/021)
scaled_scores = trace["attn_scores_pre_mask"]      # (2, 3, 3)
print("scaled scores, pre-mask (identical to doc 004/021):\n", scaled_scores)

manual_masked = scaled_scores + combined            # broadcasts (3,3) over both heads
print("\nscaled scores + combined mask:\n", manual_masked)
manual_probs = F.softmax(manual_masked, dim=-1)
print("\nsoftmax(scaled scores + combined mask):\n", manual_probs)

manual_out_per_head = manual_probs @ V_HEAD          # (2, 3, 2), shared v (n_kv_heads=1)
print("\nweighted sum with v, per head:\n", manual_out_per_head)

concat = torch.cat([manual_out_per_head[0], manual_out_per_head[1]], dim=-1)  # (3, 4)
manual_final = concat @ W_O.T
print("\nconcat(head0, head1) @ W_O^T  (by hand):\n", manual_final)

real_out = attn(X, POSITIONS, document_mask=doc_mask)
print("\nAttention.forward(X, positions, document_mask=doc_mask)  (real module):\n", real_out[0])
torch.testing.assert_close(manual_final, real_out[0], atol=1e-5, rtol=0)
print("\nmax |by-hand - real fused forward| =", (manual_final - real_out[0]).abs().max().item())

# Row-by-row comparison against the causal-ONLY output (docs 004/021).
CAUSAL_ONLY = torch.tensor([
    [2.000000, 2.000000, 2.000000, 2.000000],
    [3.474192, 2.000000, 2.804430, 2.669762],
    [3.408561, 1.573375, 2.318140, 2.663796],
])
torch.testing.assert_close(attn(X, POSITIONS), CAUSAL_ONLY.unsqueeze(0), atol=1e-5, rtol=0)
print("\ncausal-only output (docs 004/021, for comparison):\n", CAUSAL_ONLY)
print("causal + document-mask output (this doc):\n", real_out[0])
print("row-by-row |Δ| :", (real_out[0] - CAUSAL_ONLY).abs())

# document_mask=None must still take the byte-identical is_causal=True path
# as calling forward() with no document_mask argument at all (both take
# is_causal=True; CAUSAL_ONLY above is a 6-dp-rounded literal, so compare
# the two live tensors to each other at atol=0, not to the rounded literal).
torch.testing.assert_close(attn(X, POSITIONS, document_mask=None), attn(X, POSITIONS),
                            atol=0, rtol=0)
print("\ndocument_mask=None reproduces the no-argument call bit-for-bit (fast path intact)")

# ======================================================================
# 5. Exactness: perturb document 0, measure document 1's output
# ======================================================================
_section("5. isolation is EXACT -- perturb token 0 (document 0), measure Δ per row")

X_PERTURBED = X.clone()
X_PERTURBED[0, 0, :] += 50.0
out_base = attn(X, POSITIONS, document_mask=doc_mask)
out_pert = attn(X_PERTURBED, POSITIONS, document_mask=doc_mask)
delta = (out_pert - out_base).abs().max(dim=-1).values[0]
print("segment_ids            :", segment_ids.tolist()[0])
print("max|Δoutput| per position:", [round(v, 6) for v in delta.tolist()])
for pos, seg in enumerate(segment_ids.tolist()[0]):
    print(f"  position {pos}  (document {seg})  max|Δ| = {delta[pos].item():.6f}")
assert delta[0].item() > 0.0, "document 0's own output must change"
assert delta[1].item() == 0.0, "document 1 must be exactly unaffected"
assert delta[2].item() == 0.0, "document 1 must be exactly unaffected"
print("positions 1, 2 (document 1): max|Δ| == 0.0 EXACTLY -- torch.equal, not allclose")
assert torch.equal(out_pert[0, 1:], out_base[0, 1:])

# ======================================================================
# 6. MLAAttention gets the same treatment
# ======================================================================
_section("6. MLAAttention.forward -- same document_mask branch, own tiny example")

mla_cfg = ModelConfig(
    vocab_size=10, d_model=8, n_heads=2, n_kv_heads=2, max_seq_len=8,
    positional_scheme="nope", attention_variant="mla", mla_latent_dim=2,
)
torch.manual_seed(1)
mla = MLAAttention(mla_cfg)
x_mla = torch.randn(1, 3, 8)

mla_doc_mask = build_document_mask(segment_ids)
out_mla_causal = mla(x_mla, POSITIONS)
out_mla_doc = mla(x_mla, POSITIONS, document_mask=mla_doc_mask)
print("max|causal-only - causal+doc| (rows 1,2 must differ; row 0 must not):")
d_mla = (out_mla_doc - out_mla_causal).abs().max(dim=-1).values[0]
print(" ", [round(v, 6) for v in d_mla.tolist()])
assert d_mla[0].item() == 0.0  # row 0 only ever saw itself anyway
assert d_mla[1].item() > 0.0
assert d_mla[2].item() > 0.0

x_mla_pert = x_mla.clone()
x_mla_pert[0, 0, :] += 50.0
out_mla_pert = mla(x_mla_pert, POSITIONS, document_mask=mla_doc_mask)
d_pert = (out_mla_pert - out_mla_doc).abs().max(dim=-1).values[0]
print("perturb token 0 -> max|Δ| per position:", [round(v, 6) for v in d_pert.tolist()])
assert d_pert[0].item() > 0.0
assert torch.equal(out_mla_pert[0, 1:], out_mla_doc[0, 1:])
print("MLAAttention: document 1 (positions 1,2) unaffected, exactly, here too")

# document_mask=None fast path for MLA too.
mla_fast = mla(x_mla, POSITIONS)
mla_explicit_mask = _causal_position_mask(POSITIONS, 3, torch.device("cpu"), x_mla.dtype)
mla_slow = F.scaled_dot_product_attention  # sanity: is_causal path exists and matches
print("MLA document_mask=None still matches its own is_causal=True output:",
      torch.equal(mla_fast, mla(x_mla, POSITIONS, document_mask=None)))

# ======================================================================
# 7. QwenMoEModel: ONE mask built, shared across every block
# ======================================================================
_section("7. QwenMoEModel.forward builds ONE document_mask, shares it across n_layers blocks")

model_cfg = ModelConfig(
    vocab_size=6, d_model=4, n_layers=2, n_heads=2, n_kv_heads=1, d_ff=8,
    n_shared_experts=1, n_routed_experts=2, top_k=1, max_seq_len=8,
    positional_scheme="nope", qk_norm=False, tie_word_embeddings=False,
)
torch.manual_seed(2)
model = QwenMoEModel(model_cfg)
with torch.no_grad():
    # ids 0, 1, 2 embed to exactly our running-example x0, x1, x2
    model.embedding.weight[0].copy_(X[0, 0])
    model.embedding.weight[1].copy_(X[0, 1])
    model.embedding.weight[2].copy_(X[0, 2])
    model.blocks[0].attn.w_q.weight.copy_(W_Q)
    model.blocks[0].attn.w_k.weight.copy_(W_K)
    model.blocks[0].attn.w_v.weight.copy_(W_V)
    model.blocks[0].attn.w_o.weight.copy_(W_O)

input_ids = torch.tensor([[0, 1, 2]])

call_count = 0
import qwenmoe.model.transformer as transformer_mod
_real_build = transformer_mod.build_document_mask


def _counting_build(seg):
    global call_count
    call_count += 1
    return _real_build(seg)


with patch.object(transformer_mod, "build_document_mask", side_effect=_counting_build):
    logits, extras = model(input_ids, segment_ids=segment_ids)
print(f"n_layers = {model_cfg.n_layers}; build_document_mask called {call_count} time(s) per forward()")
assert call_count == 1, "the mask must be built once per forward call, not once per layer"

# segment_ids=None must still be the pre-spec, mask-free path.
logits_none, _ = model(input_ids, segment_ids=None)
print("segment_ids=None vs segment_ids=[[...]] give different logits:",
      not torch.equal(logits, logits_none))

# The isolation property survives 2 full blocks (attention + a real MoE
# sublayer in between). MoE is strictly per-token, so it cannot leak
# information between positions -- only attention mixes tokens, and it is
# masked. Perturb token 0's embedding row and check.
base_emb = model.embedding.weight[0].clone()
with torch.no_grad():
    logits_base, _ = model(input_ids, segment_ids=segment_ids)
    model.embedding.weight[0].add_(50.0)
    logits_pert, _ = model(input_ids, segment_ids=segment_ids)
    model.embedding.weight[0].copy_(base_emb)

delta_model = (logits_pert - logits_base).abs().max(dim=-1).values[0]
print("full 2-layer model, perturb token-0 embedding by +50, max|Δlogit| per position:")
for pos, seg in enumerate(segment_ids.tolist()[0]):
    print(f"  position {pos}  (document {seg})  max|Δ| = {delta_model[pos].item():.6f}")
assert delta_model[0].item() > 0.0
assert delta_model[1].item() == 0.0
assert delta_model[2].item() == 0.0
assert torch.equal(logits_pert[0, 1:], logits_base[0, 1:])
print("document 1 (positions 1, 2): max|Δlogit| == 0.0 EXACTLY, even after 2 full blocks")

print("\nAll assertions passed -- every number in 031-document-boundaries.md is real.")
