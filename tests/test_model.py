import pytest
import torch

from qwenmoe.config import ModelConfig
from qwenmoe.model.block import TransformerBlock
from qwenmoe.model.transformer import QwenMoEModel


def _tiny_config(**overrides):
    defaults = dict(
        vocab_size=50,
        d_model=16,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        d_ff=32,
        n_routed_experts=4,
        n_shared_experts=1,
        top_k=2,
        max_seq_len=16,
    )
    defaults.update(overrides)
    return ModelConfig(**defaults)


def test_transformer_block_output_shape_and_aux_loss_finite():
    config = _tiny_config()
    block = TransformerBlock(config)
    x = torch.randn(2, 5, config.d_model)

    out, aux_loss = block(x, torch.arange(5))

    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    assert torch.isfinite(aux_loss)


def test_transformer_block_with_mla_attention_output_shape_and_aux_loss_finite():
    """Task 16: attention_variant="mla" is a drop-in alternative to GQA at
    the full-block level too, not just in isolation."""
    config = _tiny_config(attention_variant="mla", mla_latent_dim=8)
    block = TransformerBlock(config)
    x = torch.randn(2, 5, config.d_model)

    out, aux_loss = block(x, torch.arange(5))

    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    assert torch.isfinite(aux_loss)


@pytest.mark.parametrize("batch, seq_len", [(2, 5), (3, 8)])
def test_model_output_shape_for_multiple_batch_seq_len_combos(batch, seq_len):
    """Requirement 9.5."""
    config = _tiny_config()
    model = QwenMoEModel(config)
    input_ids = torch.randint(0, config.vocab_size, (batch, seq_len))

    logits, extras = model(input_ids)

    assert logits.shape == (batch, seq_len, config.vocab_size)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(extras["aux_loss"])


def test_tied_embeddings_share_weight_tensor():
    """Requirement 1.7."""
    config = _tiny_config(tie_word_embeddings=True)
    model = QwenMoEModel(config)
    assert model.lm_head.weight is model.embedding.weight


def test_untied_embeddings_are_independent_tensors():
    config = _tiny_config(tie_word_embeddings=False)
    model = QwenMoEModel(config)
    assert model.lm_head.weight is not model.embedding.weight
    assert model.lm_head.weight.shape == model.embedding.weight.shape


def test_gradients_flow_through_embedding_and_every_block():
    config = _tiny_config()
    model = QwenMoEModel(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 5))

    logits, extras = model(input_ids)
    (logits.sum() + extras["aux_loss"]).backward()

    assert model.embedding.weight.grad is not None
    for block in model.blocks:
        assert any(p.grad is not None for p in block.attn.parameters())
