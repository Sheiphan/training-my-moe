import math

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


# -- document masking (moe-modernization Requirement 10.2, 10.3) ------------


def test_segment_ids_none_is_identical_to_the_pre_spec_forward():
    """Requirement 10.3: `segment_ids=None` (the default) must reproduce the
    exact logits of a call that never passes the argument at all."""
    torch.manual_seed(0)
    model = QwenMoEModel(_tiny_config()).eval()
    ids = torch.randint(0, model.config.vocab_size, (2, 6))
    with torch.no_grad():
        a, _ = model(ids)
        b, _ = model(ids, segment_ids=None)
    assert torch.equal(a, b)


def test_segment_ids_block_cross_document_influence_end_to_end():
    """Requirement 10.2: changing a token in document 0 leaves every
    document-1 logit exactly unchanged when segment ids are supplied."""
    torch.manual_seed(0)
    model = QwenMoEModel(_tiny_config(positional_scheme="nope")).eval()
    ids = torch.randint(0, model.config.vocab_size, (1, 6))
    segment_ids = torch.tensor([[0, 0, 0, 1, 1, 1]])

    perturbed = ids.clone()
    perturbed[0, 1] = (perturbed[0, 1] + 3) % model.config.vocab_size

    with torch.no_grad():
        base, _ = model(ids, segment_ids=segment_ids)
        changed, _ = model(perturbed, segment_ids=segment_ids)

    assert torch.equal(base[:, 3:, :], changed[:, 3:, :])  # document 1 untouched
    assert not torch.equal(base[:, 1:3, :], changed[:, 1:3, :])  # document 0 does change


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


# -- principled initialization (moe-modernization Requirement 9) ------------


def test_init_every_linear_is_truncated_normal_at_init_std_not_kaiming():
    """Requirement 9.1: every nn.Linear starts from a truncated normal at
    `init_std`, replacing PyTorch's default Kaiming-uniform. Truncation at
    +/-2 std means the realized std is ~0.88x nominal and no entry exceeds
    2*init_std in magnitude -- both checked here."""
    config = _tiny_config(init_std=0.02)
    model = QwenMoEModel(config)

    w_q = model.blocks[0].attn.w_q.weight
    assert w_q.abs().max().item() <= 2 * config.init_std + 1e-6
    assert 0.7 * config.init_std < w_q.std().item() < config.init_std  # truncated => below nominal


def test_init_residual_projections_scaled_by_one_over_sqrt_2n():
    """Requirement 9.2: residual-path output projections (Attention.w_o,
    expert down-projections) get their std divided by sqrt(2 * n_layers)."""
    config = _tiny_config(n_layers=2, init_std=0.02)
    model = QwenMoEModel(config)

    w_q_std = model.blocks[0].attn.w_q.weight.std().item()
    w_o_std = model.blocks[0].attn.w_o.weight.std().item()
    expert = model.blocks[0].moe.routed_experts[0]
    down_proj = getattr(expert, "w_down", None) or getattr(expert, "w_2")

    expected_ratio = 1.0 / math.sqrt(2 * config.n_layers)  # = 0.5 for n_layers=2
    assert w_o_std / w_q_std == pytest.approx(expected_ratio, rel=0.2)
    assert down_proj.weight.std().item() / w_q_std == pytest.approx(expected_ratio, rel=0.2)


def test_init_untrained_logits_have_order_one_std():
    """Requirement 9.4: an untrained model's logits on a random batch have
    std within a small factor of 1.0 -- no 50x-style blow-up (the failure
    mode journal entry 010 hit with the old nn.Embedding default init)."""
    torch.manual_seed(0)
    config = _tiny_config(d_model=64, n_layers=2)
    model = QwenMoEModel(config)
    input_ids = torch.randint(0, config.vocab_size, (4, 16))

    with torch.no_grad():
        logits, _ = model(input_ids)

    assert 0.02 < logits.std().item() < 3.0


def test_init_tied_lm_head_keeps_embedding_init():
    """Requirement 9.3: with tied weights, the shared matrix keeps the
    embedding init (std = init_std), and is not re-initialized a second
    time as a linear."""
    torch.manual_seed(0)
    tied = QwenMoEModel(_tiny_config(tie_word_embeddings=True, init_std=0.02))
    assert tied.lm_head.weight is tied.embedding.weight
    # embedding init: truncated normal at init_std, so bounded by 2*init_std
    assert tied.embedding.weight.abs().max().item() <= 2 * 0.02 + 1e-6


@pytest.mark.parametrize("router_scoring", ["softmax", "sigmoid"])
def test_model_trains_a_few_steps_with_each_router_scoring(router_scoring):
    """moe-modernization Requirement 4.1/4.6: a full model with sigmoid
    routing (and a router z-loss) does a forward+backward with finite loss
    and a shrinking loss over a handful of manual steps."""
    torch.manual_seed(0)
    config = _tiny_config(
        d_model=32, router_scoring=router_scoring, router_z_loss_weight=1e-3
    )
    model = QwenMoEModel(config)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    input_ids = torch.randint(0, config.vocab_size, (4, 8))
    targets = torch.randint(0, config.vocab_size, (4, 8))

    losses = []
    for _ in range(15):
        logits, extras = model(input_ids)
        ce = torch.nn.functional.cross_entropy(
            logits.reshape(-1, config.vocab_size), targets.reshape(-1)
        )
        total = ce + extras["aux_loss"]
        opt.zero_grad()
        total.backward()
        opt.step()
        losses.append(ce.item())

    assert all(math.isfinite(v) for v in losses)
    assert losses[-1] < losses[0]


# -- Multi-Token Prediction (moe-modernization Requirement 6) ---------------


def test_mtp_depth_zero_is_a_structural_no_op():
    """Requirement 6.2: `mtp_depth == 0` (default) -> no MTP heads, nothing
    in the state dict, and `forward` returns exactly `(logits, {"aux_loss":
    ...})` even when `targets` is passed."""
    torch.manual_seed(0)
    config = _tiny_config()
    assert config.mtp_depth == 0
    model = QwenMoEModel(config)

    assert len(model.mtp_heads) == 0
    assert not any("mtp_heads" in k for k in model.state_dict())

    x = torch.randint(0, config.vocab_size, (2, 8))
    y = torch.randint(0, config.vocab_size, (2, 8))
    model.train()
    logits_no_targets, extras_no_targets = model(x)
    logits_targets, extras_targets = model(x, targets=y)

    assert set(extras_targets) == {"aux_loss"}
    assert set(extras_no_targets) == {"aux_loss"}
    assert torch.equal(logits_no_targets, logits_targets)


def test_mtp_depth_one_adds_params_and_exactly_one_loss_term():
    """Requirement 6.1, 6.3: one head -> one extra CE term; params grow."""
    torch.manual_seed(0)
    base = QwenMoEModel(_tiny_config())
    mtp = QwenMoEModel(_tiny_config(mtp_depth=1))

    n_base = sum(p.numel() for p in base.parameters())
    n_mtp = sum(p.numel() for p in mtp.parameters())
    assert n_mtp > n_base

    x = torch.randint(0, mtp.config.vocab_size, (2, 8))
    y = torch.randint(0, mtp.config.vocab_size, (2, 8))
    mtp.train()
    logits, extras = mtp(x, targets=y)

    assert logits.shape == (2, 8, mtp.config.vocab_size)  # primary head unchanged (Req 6.4)
    assert len(extras["mtp_losses"]) == 1
    assert torch.isfinite(extras["mtp_loss"]) and extras["mtp_loss"] > 0
    assert extras["mtp_loss"].requires_grad  # trains the trunk


def test_mtp_heads_are_skipped_at_inference_and_without_targets():
    """Requirement 6.4: eval mode, or no `targets`, -> no MTP loss, plain
    return signature (so `generate()` is untouched)."""
    torch.manual_seed(0)
    model = QwenMoEModel(_tiny_config(mtp_depth=2))
    x = torch.randint(0, model.config.vocab_size, (2, 8))
    y = torch.randint(0, model.config.vocab_size, (2, 8))

    model.eval()
    _, extras_eval = model(x, targets=y)
    assert set(extras_eval) == {"aux_loss"}

    model.train()
    _, extras_no_targets = model(x)  # trainer always passes targets; a stray caller may not
    assert set(extras_no_targets) == {"aux_loss"}


def test_mtp_depth_two_is_a_causal_chain_of_two_losses():
    torch.manual_seed(0)
    model = QwenMoEModel(_tiny_config(mtp_depth=2, d_model=32))
    x = torch.randint(0, model.config.vocab_size, (3, 10))
    y = torch.randint(0, model.config.vocab_size, (3, 10))
    model.train()

    _, extras = model(x, targets=y)

    assert len(extras["mtp_losses"]) == 2
    assert all(torch.isfinite(v) for v in extras["mtp_losses"])
    # head 2 consumes head 1's output -> a grad path from mtp_losses[1] to head 1
    extras["mtp_losses"][1].backward(retain_graph=True)
    assert any(p.grad is not None for p in model.mtp_heads[0].parameters())


def test_mtp_checkpoint_round_trips():
    """Requirement 6.5: the MTP modules save and load like any other
    submodule."""
    torch.manual_seed(0)
    model = QwenMoEModel(_tiny_config(mtp_depth=2))
    restored = QwenMoEModel(_tiny_config(mtp_depth=2))
    restored.load_state_dict(model.state_dict())

    x = torch.randint(0, model.config.vocab_size, (2, 8))
    y = torch.randint(0, model.config.vocab_size, (2, 8))
    model.train()
    restored.train()
    torch.manual_seed(1)
    _, a = model(x, targets=y)
    torch.manual_seed(1)
    _, b = restored(x, targets=y)
    assert torch.allclose(a["mtp_loss"], b["mtp_loss"], atol=1e-6)


def test_gradients_flow_through_embedding_and_every_block():
    config = _tiny_config()
    model = QwenMoEModel(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 5))

    logits, extras = model(input_ids)
    (logits.sum() + extras["aux_loss"]).backward()

    assert model.embedding.weight.grad is not None
    for block in model.blocks:
        assert any(p.grad is not None for p in block.attn.parameters())
