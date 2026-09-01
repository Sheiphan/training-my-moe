import pytest
import torch

from qwenmoe.config import ModelConfig, TrainConfig
from qwenmoe.model.rmsnorm import RMSNorm
from qwenmoe.model.transformer import QwenMoEModel
from qwenmoe.train.optim import (
    CombinedOptimizer,
    Muon,
    build_optimizer,
    split_muon_adamw_params,
    split_param_groups,
    zeropower_via_newtonschulz5,
)


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


def test_build_optimizer_rejects_unknown_optimizer_name():
    model = QwenMoEModel(_tiny_model_config())
    config = _tiny_train_config()
    config.optimizer = "sgd"  # bypass __post_init__'s own Literal check

    with pytest.raises(ValueError, match="sgd"):
        build_optimizer(model, config)


# -- split_muon_adamw_params (Requirement 1.4) --------------------------------


def test_split_muon_adamw_params_routes_hidden_matrices_to_muon():
    model = QwenMoEModel(_tiny_model_config())
    muon_params, _ = split_muon_adamw_params(model)
    muon_ids = {id(p) for p in muon_params}

    block = model.blocks[0]
    assert id(block.attn.w_q.weight) in muon_ids
    assert id(block.moe.router.gate.weight) in muon_ids


def test_split_muon_adamw_params_keeps_embedding_and_norms_on_adamw():
    model = QwenMoEModel(_tiny_model_config())
    _, adamw_params = split_muon_adamw_params(model)
    adamw_ids = {id(p) for p in adamw_params}

    assert id(model.embedding.weight) in adamw_ids
    for module in model.modules():
        if isinstance(module, RMSNorm):
            assert id(module.weight) in adamw_ids


def test_split_muon_adamw_params_untied_lm_head_stays_on_adamw():
    """Requirement 1.4 names the LM head explicitly as an AdamW parameter
    even though it's a 2-D matrix like any Muon-eligible weight -- Muon is
    only for *hidden* matmuls, not the vocab-sized output projection."""
    model = QwenMoEModel(_tiny_model_config(tie_word_embeddings=False))
    muon_params, adamw_params = split_muon_adamw_params(model)

    assert id(model.lm_head.weight) not in {id(p) for p in muon_params}
    assert id(model.lm_head.weight) in {id(p) for p in adamw_params}


def test_split_muon_adamw_params_covers_every_parameter_exactly_once():
    model = QwenMoEModel(_tiny_model_config())
    muon_params, adamw_params = split_muon_adamw_params(model)

    all_ids = {id(p) for p in muon_params} | {id(p) for p in adamw_params}
    model_ids = {id(p) for p in model.parameters() if p.requires_grad}
    assert all_ids == model_ids
    assert len(muon_params) + len(adamw_params) == len(all_ids)


# -- zeropower_via_newtonschulz5 (Requirement 1.4) -----------------------------


def test_zeropower_via_newtonschulz5_orthogonalizes_a_random_matrix():
    """The output's singular values should all land close to 1 -- that's
    the entire point of "zeroth power" (Requirement 1.4's "Newton-Schulz
    orthogonalization"). The fixed 5-step quintic iteration is a known
    *approximate* orthogonalization -- Jordan et al.'s own writeup notes the
    coefficients are chosen to converge fast for small singular values at
    the cost of only landing near (not exactly at) 1 for singular values
    already close to it, so the assertion is "singular values end up in a
    narrow band around 1, much narrower than the input's spread" rather
    than "singular values are all ~1.0 to 2 decimal places"."""
    torch.manual_seed(0)
    G = torch.randn(8, 5)
    input_condition_number = torch.linalg.svdvals(G.double()).max() / torch.linalg.svdvals(
        G.double()
    ).min()

    X = zeropower_via_newtonschulz5(G.double(), steps=5)

    singular_values = torch.linalg.svdvals(X)
    assert singular_values.min().item() > 0.5
    assert singular_values.max().item() < 1.5
    output_condition_number = singular_values.max() / singular_values.min()
    assert output_condition_number < input_condition_number


def test_zeropower_via_newtonschulz5_handles_wide_matrices_too():
    torch.manual_seed(0)
    G = torch.randn(5, 8)

    X = zeropower_via_newtonschulz5(G.double(), steps=5)

    assert X.shape == G.shape
    singular_values = torch.linalg.svdvals(X)
    assert singular_values.min().item() > 0.5
    assert singular_values.max().item() < 1.5


# -- Muon (Requirement 1.4) ----------------------------------------------------


def test_muon_rejects_a_1d_parameter():
    p = torch.nn.Parameter(torch.randn(5))
    p.grad = torch.randn(5)
    optimizer = Muon([p], lr=0.02)

    with pytest.raises(AssertionError, match="2-D"):
        optimizer.step()


def test_muon_step_reduces_loss_on_a_toy_linear_layer():
    torch.manual_seed(0)
    layer = torch.nn.Linear(6, 4, bias=False)
    target = torch.randn(4, 6)
    x = torch.randn(8, 6)
    y = x @ target.T

    optimizer = Muon([layer.weight], lr=0.05, momentum=0.9)
    losses = []
    for _ in range(20):
        optimizer.zero_grad()
        pred = layer(x)
        loss = torch.nn.functional.mse_loss(pred, y)
        loss.backward()
        losses.append(loss.item())
        optimizer.step()

    assert losses[-1] < losses[0]


# -- build_optimizer(optimizer="muon") / CombinedOptimizer (Requirement 1.4-1.6) --


def test_build_optimizer_muon_returns_combined_optimizer_covering_every_param():
    model = QwenMoEModel(_tiny_model_config())
    config = _tiny_train_config(optimizer="muon")

    optimizer = build_optimizer(model, config)

    assert isinstance(optimizer, CombinedOptimizer)
    optimized_ids = {id(p) for g in optimizer.param_groups for p in g["params"]}
    model_ids = {id(p) for p in model.parameters() if p.requires_grad}
    assert optimized_ids == model_ids


def test_build_optimizer_muon_uses_muon_lr_when_set_and_lr_otherwise():
    model = QwenMoEModel(_tiny_model_config())

    config_default = _tiny_train_config(optimizer="muon", lr=3e-3, muon_lr=None)
    optimizer_default = build_optimizer(model, config_default)
    muon_group = next(g for g in optimizer_default.param_groups if "momentum" in g)
    assert muon_group["lr"] == pytest.approx(3e-3)

    config_explicit = _tiny_train_config(optimizer="muon", lr=3e-3, muon_lr=0.02)
    optimizer_explicit = build_optimizer(model, config_explicit)
    muon_group_explicit = next(g for g in optimizer_explicit.param_groups if "momentum" in g)
    assert muon_group_explicit["lr"] == pytest.approx(0.02)


def test_build_optimizer_muon_step_and_zero_grad_update_every_group():
    model = QwenMoEModel(_tiny_model_config())
    optimizer = build_optimizer(model, _tiny_train_config(optimizer="muon"))

    x = torch.randint(0, 12, (2, 4))
    y = torch.randint(0, 12, (2, 4))
    logits, extras = model(x)
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 12), y.reshape(-1))
    (loss + (extras.get("aux_loss") or 0.0)).backward()

    before = {id(p): p.detach().clone() for p in model.parameters()}
    optimizer.step()
    optimizer.zero_grad()

    changed = sum(
        1 for p in model.parameters() if not torch.equal(p.detach(), before[id(p)])
    )
    assert changed > 0
    for p in model.parameters():
        assert p.grad is None


def test_combined_optimizer_state_dict_round_trips():
    model = QwenMoEModel(_tiny_model_config())
    config = _tiny_train_config(optimizer="muon")
    optimizer = build_optimizer(model, config)

    x = torch.randint(0, 12, (2, 4))
    y = torch.randint(0, 12, (2, 4))
    logits, extras = model(x)
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 12), y.reshape(-1))
    (loss + (extras.get("aux_loss") or 0.0)).backward()
    optimizer.step()

    state = optimizer.state_dict()

    fresh_model = QwenMoEModel(_tiny_model_config())
    fresh_optimizer = build_optimizer(fresh_model, config)
    fresh_optimizer.load_state_dict(state)  # must not raise

    assert len(fresh_optimizer.state) == len(optimizer.state)
