import pytest
import torch

from qwenmoe.config import ModelConfig
from qwenmoe.model.attention import Attention, build_document_mask


def _tiny_config(**overrides):
    defaults = dict(vocab_size=50, d_model=16, n_heads=4, n_kv_heads=4, max_seq_len=8)
    defaults.update(overrides)
    return ModelConfig(**defaults)


def test_attention_output_shape_and_finite_mha():
    """n_kv_heads == n_heads is the standard multi-head-attention case."""
    config = _tiny_config(n_heads=4, n_kv_heads=4)
    attn = Attention(config)
    x = torch.randn(2, 5, config.d_model)
    out = attn(x, torch.arange(5))
    assert out.shape == (2, 5, config.d_model)
    assert torch.isfinite(out).all()


def test_attention_output_shape_and_finite_gqa():
    """n_kv_heads < n_heads: 2 kv heads shared across 4 query heads."""
    config = _tiny_config(n_heads=4, n_kv_heads=2)
    attn = Attention(config)
    x = torch.randn(3, 6, config.d_model)
    out = attn(x, torch.arange(6))
    assert out.shape == (3, 6, config.d_model)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("scheme", ["rope", "nope", "alibi"])
def test_attention_works_with_each_positional_scheme(scheme):
    config = _tiny_config(n_heads=4, n_kv_heads=2, positional_scheme=scheme)
    attn = Attention(config)
    x = torch.randn(2, 5, config.d_model)
    out = attn(x, torch.arange(5))
    assert out.shape == (2, 5, config.d_model)
    assert torch.isfinite(out).all()


# -- QK-norm (moe-modernization Requirement 7.3, 7.5) ------------------------


def test_qk_norm_off_by_default_and_constructs_no_extra_parameters():
    """Requirement 7.5: qk_norm=False must be numerically unchanged from
    today -- the strongest version of that guarantee is that the
    normalizing submodules don't even exist, so there's no code path left
    that could touch q/k."""
    config = _tiny_config(n_heads=4, n_kv_heads=2)
    attn = Attention(config)
    assert attn.qk_norm is False
    assert not hasattr(attn, "q_norm")
    assert not hasattr(attn, "k_norm")


def test_qk_norm_true_constructs_per_head_rmsnorm_and_changes_output():
    config = _tiny_config(n_heads=4, n_kv_heads=2, qk_norm=True)
    attn = Attention(config)
    assert attn.q_norm.weight.shape == (attn.head_dim,)
    assert attn.k_norm.weight.shape == (attn.head_dim,)

    torch.manual_seed(0)
    attn_off = Attention(_tiny_config(n_heads=4, n_kv_heads=2, qk_norm=False))
    torch.manual_seed(0)
    attn_on = Attention(_tiny_config(n_heads=4, n_kv_heads=2, qk_norm=True))
    # Same manual_seed before construction => identical w_q/w_k/w_v/w_o
    # weights either way (RMSNorm's weight starts at all-ones, which
    # doesn't draw from the RNG) -- so any output difference below is
    # caused only by the qk_norm flag, not by different random weights.

    x = torch.randn(2, 5, config.d_model)
    positions = torch.arange(5)
    with torch.no_grad():
        out_off = attn_off(x, positions)
        out_on = attn_on(x, positions)

    assert torch.isfinite(out_on).all()
    assert not torch.allclose(out_off, out_on, atol=1e-4)


def test_qk_norm_bounds_attention_logit_scale_regardless_of_input_magnitude():
    """The problem QK-norm exists to fix: without it, attention logits'
    scale grows with the *residual stream's* scale (here simulated by
    scaling the layer's input) -- unbounded growth destabilizes the
    softmax. RMSNorm on q/k caps their scale before the dot product, so
    the resulting logits stay in a narrow band no matter how large the
    input gets."""
    torch.manual_seed(0)
    attn_off = Attention(_tiny_config(n_heads=4, n_kv_heads=2, qk_norm=False))
    torch.manual_seed(0)
    attn_on = Attention(_tiny_config(n_heads=4, n_kv_heads=2, qk_norm=True))
    positions = torch.arange(5)

    torch.manual_seed(1)
    x_unit = torch.randn(1, 5, 16)
    x_large = x_unit * 200.0  # simulate a residual stream that has grown large

    with torch.no_grad():
        _, trace_off_large = attn_off.forward_with_trace(x_large, positions)
        _, trace_on_unit = attn_on.forward_with_trace(x_unit, positions)
        _, trace_on_large = attn_on.forward_with_trace(x_large, positions)

    std_off_large = trace_off_large["attn_scores_pre_mask"].std().item()
    std_on_unit = trace_on_unit["attn_scores_pre_mask"].std().item()
    std_on_large = trace_on_large["attn_scores_pre_mask"].std().item()

    assert std_off_large > 1000.0  # unbounded: grows with the input scale
    assert std_on_unit < 5.0
    assert std_on_large < 5.0  # bounded: ~unchanged across a 200x input scale-up
    assert std_on_large / std_on_unit < 5.0  # nowhere near the off-path's blowup


# -- document masking (moe-modernization Requirement 10.2, 10.3) ------------


def test_build_document_mask_zero_within_document_neg_inf_across():
    segment_ids = torch.tensor([[0, 0, 1, 1, 1]])
    mask = build_document_mask(segment_ids)
    assert mask.shape == (1, 1, 5, 5)
    assert (mask[0, 0, :2, :2] == 0.0).all()
    assert (mask[0, 0, 2:, 2:] == 0.0).all()
    assert torch.isneginf(mask[0, 0, :2, 2:]).all()
    assert torch.isneginf(mask[0, 0, 2:, :2]).all()


def test_document_mask_none_is_numerically_identical_to_today():
    """Requirement 10.3: with no document mask the forward is byte-for-byte
    the pre-spec path (the default arg literally isn't referenced)."""
    torch.manual_seed(0)
    attn = Attention(_tiny_config(n_heads=4, n_kv_heads=2))
    x = torch.randn(2, 5, 16)
    positions = torch.arange(5)
    with torch.no_grad():
        assert torch.equal(attn(x, positions), attn(x, positions, document_mask=None))


def test_document_mask_blocks_attention_across_a_separator():
    """A token in the second packed document must be completely unaffected
    by any change to the first document's tokens (Requirement 10.2)."""
    torch.manual_seed(0)
    config = _tiny_config(n_heads=2, n_kv_heads=2, positional_scheme="nope")
    attn = Attention(config)
    positions = torch.arange(5)
    segment_ids = torch.tensor([[0, 0, 0, 1, 1]])  # docs: {0,1,2} then {3,4}
    document_mask = build_document_mask(segment_ids)

    x = torch.randn(1, 5, config.d_model)
    x_perturbed = x.clone()
    x_perturbed[:, 0, :] += 50.0  # perturb a document-0 token

    with torch.no_grad():
        out = attn(x, positions, document_mask=document_mask)
        out_perturbed = attn(x_perturbed, positions, document_mask=document_mask)

    # document-1 positions (3, 4): identical -- the separator blocked the influence
    assert torch.allclose(out[:, 3:, :], out_perturbed[:, 3:, :], atol=1e-6)
    # document-0 positions still see it (positions 1, 2 attend back to position 0)
    assert not torch.allclose(out[:, 1:3, :], out_perturbed[:, 1:3, :], atol=1e-4)


def test_causal_mask_blocks_future_influence():
    """Perturbing only the last token must not change any earlier
    position's output (Requirement 1.6, 9.3)."""
    torch.manual_seed(0)
    config = _tiny_config(n_heads=2, n_kv_heads=2, positional_scheme="nope")
    attn = Attention(config)
    x = torch.randn(1, 5, config.d_model)
    positions = torch.arange(5)

    with torch.no_grad():
        out_original = attn(x, positions)

    x_perturbed = x.clone()
    x_perturbed[:, -1, :] += 10.0
    with torch.no_grad():
        out_perturbed = attn(x_perturbed, positions)

    assert torch.allclose(out_original[:, :-1, :], out_perturbed[:, :-1, :], atol=1e-6)
    assert not torch.allclose(out_original[:, -1, :], out_perturbed[:, -1, :], atol=1e-6)
