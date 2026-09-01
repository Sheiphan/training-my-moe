import pytest
import torch

from qwenmoe.config import ModelConfig
from qwenmoe.model.attention import Attention, MLAAttention, build_attention


def _tiny_config(**overrides):
    defaults = dict(
        vocab_size=50,
        d_model=32,
        n_heads=4,
        n_kv_heads=2,
        max_seq_len=8,
        attention_variant="mla",
        mla_latent_dim=8,
    )
    defaults.update(overrides)
    return ModelConfig(**defaults)


def test_mla_output_shape_matches_attention():
    """Requirement 12.4: MLA is a drop-in alternative to GQA -- same
    (batch, seq_len, d_model) input shape produces the same output shape."""
    config = _tiny_config()
    mla = MLAAttention(config)
    gqa = Attention(config)
    x = torch.randn(2, 5, config.d_model)
    positions = torch.arange(5)

    mla_out = mla(x, positions)
    gqa_out = gqa(x, positions)

    assert mla_out.shape == gqa_out.shape == (2, 5, config.d_model)
    assert torch.isfinite(mla_out).all()


@pytest.mark.parametrize("batch, seq_len", [(1, 3), (3, 6)])
def test_mla_output_shape_for_multiple_batch_seq_len_combos(batch, seq_len):
    config = _tiny_config()
    mla = MLAAttention(config)
    x = torch.randn(batch, seq_len, config.d_model)
    out = mla(x, torch.arange(seq_len))
    assert out.shape == (batch, seq_len, config.d_model)
    assert torch.isfinite(out).all()


def test_mla_causal_mask_blocks_future_influence():
    """Mirrors test_attention.py's GQA causal-masking test (Requirement
    1.6, 9.3): perturbing only the last token must not change any earlier
    position's output."""
    torch.manual_seed(0)
    config = _tiny_config()
    mla = MLAAttention(config)
    x = torch.randn(1, 5, config.d_model)
    positions = torch.arange(5)

    with torch.no_grad():
        out_original = mla(x, positions)

    x_perturbed = x.clone()
    x_perturbed[:, -1, :] += 10.0
    with torch.no_grad():
        out_perturbed = mla(x_perturbed, positions)

    assert torch.allclose(out_original[:, :-1, :], out_perturbed[:, :-1, :], atol=1e-6)
    assert not torch.allclose(out_original[:, -1, :], out_perturbed[:, -1, :], atol=1e-6)


def test_mla_gradients_flow_to_every_projection():
    config = _tiny_config()
    mla = MLAAttention(config)
    x = torch.randn(2, 5, config.d_model, requires_grad=True)

    out = mla(x, torch.arange(5))
    out.sum().backward()

    assert x.grad is not None
    for name, param in mla.named_parameters():
        assert param.grad is not None, f"{name} got no gradient"


@pytest.mark.parametrize("latent_dim", [4, 8, 16])
def test_mla_works_across_latent_dims(latent_dim):
    config = _tiny_config(mla_latent_dim=latent_dim)
    mla = MLAAttention(config)
    x = torch.randn(2, 5, config.d_model)
    out = mla(x, torch.arange(5))
    assert out.shape == (2, 5, config.d_model)
    assert torch.isfinite(out).all()


def test_mla_rejects_head_dim_not_divisible_by_four():
    """d_model=24, n_heads=4 -> head_dim=6, not divisible by 4."""
    config = _tiny_config(d_model=24, n_heads=4, n_kv_heads=2)
    with pytest.raises(ValueError, match="divisible by 4"):
        MLAAttention(config)


def test_mla_requires_latent_dim():
    config = _tiny_config(attention_variant="gqa", mla_latent_dim=None)
    with pytest.raises(ValueError, match="mla_latent_dim"):
        MLAAttention(config)


def test_mla_kv_cache_smaller_than_equivalent_gqa():
    """Requirement 12.4: MLA's actual generation-time cache (the shared
    latent + shared decoupled-RoPE key) must be smaller than what GQA would
    need to cache (n_kv_heads copies of full-size K and V) for the same
    (n_heads, d_model, n_kv_heads)."""
    config = _tiny_config(n_kv_heads=2)
    mla = MLAAttention(config)

    mla_cache = mla.kv_cache_elements_per_token
    gqa_cache = 2 * config.n_kv_heads * mla.head_dim  # K + V, each n_kv_heads * head_dim

    assert mla_cache < gqa_cache


@pytest.mark.parametrize("n_kv_heads", [1, 2, 4])
def test_mla_kv_cache_smaller_than_gqa_across_group_sizes(n_kv_heads):
    config = _tiny_config(n_kv_heads=n_kv_heads)
    mla = MLAAttention(config)

    mla_cache = mla.kv_cache_elements_per_token
    gqa_cache = 2 * n_kv_heads * mla.head_dim

    assert mla_cache < gqa_cache


@pytest.mark.parametrize("scheme", ["rope", "nope", "alibi"])
def test_mla_ignores_positional_scheme_and_always_uses_decoupled_rope(scheme):
    """design.md's documented exception: MLA always applies its own
    decoupled RoPE regardless of `ModelConfig.positional_scheme`."""
    config = _tiny_config(positional_scheme=scheme)
    mla = MLAAttention(config)
    x = torch.randn(2, 5, config.d_model)
    positions = torch.arange(5)

    out_default = mla(x, positions)
    # Perturbing the token order (reversing positions) must change the
    # output only if positional information is actually being applied --
    # confirming MLA's own RoPE is live no matter what `scheme` says.
    out_shuffled_positions = mla(x, torch.arange(5).flip(0))

    assert out_default.shape == (2, 5, config.d_model)
    assert not torch.allclose(out_default, out_shuffled_positions, atol=1e-6)


def test_build_attention_constructs_mla():
    config = _tiny_config()
    attn = build_attention(config)
    assert isinstance(attn, MLAAttention)


def test_build_attention_still_constructs_gqa_by_default():
    config = _tiny_config(attention_variant="gqa", mla_latent_dim=None)
    attn = build_attention(config)
    assert isinstance(attn, Attention)
