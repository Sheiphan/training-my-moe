import pytest
import torch

from qwenmoe.config import ModelConfig, TrainConfig
from qwenmoe.model.rmsnorm import RMSNorm
from qwenmoe.model.transformer import QwenMoEModel
from qwenmoe.train.optim import build_optimizer, split_param_groups


def _tiny_model_config(**overrides):
    defaults = dict(
        vocab_size=12,
        d_model=16,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        d_ff=32,
        n_routed_experts=4,
        n_shared_experts=1,
        top_k=2,
        max_seq_len=32,
    )
    defaults.update(overrides)
    return ModelConfig(**defaults)


def _tiny_train_config(**overrides):
    defaults = dict(batch_size=4, seq_len=8, total_tokens=1000)
    defaults.update(overrides)
    return TrainConfig(**defaults)


# -- split_param_groups (Requirement 1.1) ------------------------------------


def test_split_param_groups_covers_every_parameter_exactly_once():
    model = QwenMoEModel(_tiny_model_config())
    decay, no_decay = split_param_groups(model)

    all_ids = {id(p) for p in decay} | {id(p) for p in no_decay}
    model_ids = {id(p) for p in model.parameters() if p.requires_grad}
    assert all_ids == model_ids
    assert len(decay) + len(no_decay) == len(all_ids)  # no duplicate across groups


def test_split_param_groups_excludes_norms_and_embedding_from_decay():
    model = QwenMoEModel(_tiny_model_config())
    _, no_decay = split_param_groups(model)
    no_decay_ids = {id(p) for p in no_decay}

    assert id(model.embedding.weight) in no_decay_ids
    for module in model.modules():
        if isinstance(module, RMSNorm):
            assert id(module.weight) in no_decay_ids


def test_split_param_groups_puts_weight_matrices_in_decay():
    model = QwenMoEModel(_tiny_model_config())
    decay, _ = split_param_groups(model)
    decay_ids = {id(p) for p in decay}

    block = model.blocks[0]
    assert id(block.attn.w_q.weight) in decay_ids
    assert id(block.moe.router.gate.weight) in decay_ids


def test_split_param_groups_tied_embedding_counted_once():
    """tie_word_embeddings=True (default) makes embedding.weight and
    lm_head.weight the same Parameter object -- it must land in exactly one
    group, not be double-counted."""
    model = QwenMoEModel(_tiny_model_config(tie_word_embeddings=True))
    assert model.lm_head.weight is model.embedding.weight

    decay, no_decay = split_param_groups(model)
    occurrences = sum(1 for p in decay + no_decay if p is model.embedding.weight)
    assert occurrences == 1


def test_split_param_groups_untied_lm_head_is_its_own_decay_param():
    model = QwenMoEModel(_tiny_model_config(tie_word_embeddings=False))
    assert model.lm_head.weight is not model.embedding.weight

    decay, no_decay = split_param_groups(model)
    assert id(model.lm_head.weight) in {id(p) for p in decay}
    assert id(model.embedding.weight) in {id(p) for p in no_decay}


# -- build_optimizer (Requirement 1.1, 1.2, 1.4, 1.5) -------------------------


def test_build_optimizer_adamw_uses_configured_betas_wd_eps():
    model = QwenMoEModel(_tiny_model_config())
    config = _tiny_train_config(betas=(0.8, 0.9), adam_eps=1e-6, weight_decay=0.25)

    optimizer = build_optimizer(model, config)

    assert isinstance(optimizer, torch.optim.AdamW)
    weight_decays = {g["weight_decay"] for g in optimizer.param_groups}
    assert weight_decays == {0.25, 0.0}  # decay group / no-decay group
    for group in optimizer.param_groups:
        assert group["betas"] == (0.8, 0.9)
        assert group["eps"] == pytest.approx(1e-6)


def test_build_optimizer_covers_every_model_parameter():
    model = QwenMoEModel(_tiny_model_config())
    optimizer = build_optimizer(model, _tiny_train_config())

    optimized_ids = {id(p) for g in optimizer.param_groups for p in g["params"]}
    model_ids = {id(p) for p in model.parameters() if p.requires_grad}
    assert optimized_ids == model_ids


def test_build_optimizer_rejects_muon_before_it_is_implemented():
    """moe-modernization task 4 wires up Muon; until then requesting it
    fails clearly rather than silently falling back to AdamW."""
    model = QwenMoEModel(_tiny_model_config())
    config = _tiny_train_config(optimizer="muon")

    with pytest.raises(ValueError, match="muon"):
        build_optimizer(model, config)
