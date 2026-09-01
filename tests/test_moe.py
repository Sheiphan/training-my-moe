import pytest
import torch

from qwenmoe.config import ModelConfig
from qwenmoe.model.moe import AuxLossBalancer, LossFreeBalancer, MoELayer, Router, select_top_k


def _tiny_config(**overrides):
    defaults = dict(
        vocab_size=50,
        d_model=16,
        d_ff=32,
        n_routed_experts=4,
        n_shared_experts=1,
        top_k=2,
        max_seq_len=8,
    )
    defaults.update(overrides)
    return ModelConfig(**defaults)


def test_router_output_shape():
    router = Router(d_model=16, n_routed_experts=4)
    logits = router(torch.randn(3, 16))
    assert logits.shape == (3, 4)


def test_select_top_k_picks_exactly_k_and_weights_sum_to_one():
    """Requirement 9.4: exactly k experts selected per token, and their
    gate weights sum to 1."""
    torch.manual_seed(0)
    n_tokens, n_experts, k = 5, 6, 3
    logits = torch.randn(n_tokens, n_experts)
    probs = torch.softmax(logits, dim=-1)

    topk_indices, topk_weights = select_top_k(logits, probs, k)

    assert topk_indices.shape == (n_tokens, k)
    assert topk_weights.shape == (n_tokens, k)
    for row in topk_indices:
        assert len(set(row.tolist())) == k  # k distinct experts, no duplicates
    assert torch.allclose(topk_weights.sum(dim=-1), torch.ones(n_tokens), atol=1e-6)


def test_select_top_k_picks_highest_scores():
    scores = torch.tensor([[0.1, 0.9, 0.3, 0.05]])
    probs = torch.softmax(scores, dim=-1)
    topk_indices, topk_weights = select_top_k(scores, probs, top_k=2)
    assert set(topk_indices[0].tolist()) == {1, 2}  # the two highest-scoring experts
    assert torch.allclose(topk_weights.sum(dim=-1), torch.tensor([1.0]), atol=1e-6)


def test_aux_loss_is_finite_and_responds_to_imbalance():
    """Requirement 2.4/9.4: aux loss is a finite scalar that responds to
    induced imbalance."""
    balancer = AuxLossBalancer(n_routed_experts=4)
    n_tokens = 8

    balanced_probs = torch.full((n_tokens, 4), 0.25)
    balanced_indices = torch.arange(n_tokens).remainder(4).unsqueeze(-1)
    balanced_loss = balancer.aux_loss(balanced_probs, balanced_indices)

    imbalanced_probs = torch.zeros(n_tokens, 4)
    imbalanced_probs[:, 0] = 1.0
    imbalanced_indices = torch.zeros(n_tokens, 1, dtype=torch.long)
    imbalanced_loss = balancer.aux_loss(imbalanced_probs, imbalanced_indices)

    assert torch.isfinite(balanced_loss)
    assert torch.isfinite(imbalanced_loss)
    assert imbalanced_loss.item() > balanced_loss.item()


def test_moe_layer_output_shape_and_aux_loss_finite():
    config = _tiny_config()
    moe = MoELayer(config)
    x = torch.randn(2, 5, config.d_model)
    output, aux_loss = moe(x)
    assert output.shape == (2, 5, config.d_model)
    assert torch.isfinite(output).all()
    assert aux_loss is not None
    assert torch.isfinite(aux_loss)


def test_moe_layer_zero_shared_experts_does_not_error():
    config = _tiny_config(n_shared_experts=0)
    moe = MoELayer(config)
    x = torch.randn(2, 3, config.d_model)
    output, _ = moe(x)
    assert output.shape == (2, 3, config.d_model)
    assert torch.isfinite(output).all()


def test_zero_token_expert_does_not_raise():
    """1 token + top_k=1 guarantees n_routed_experts - 1 experts get zero
    tokens this batch (Requirement 2.5)."""
    config = _tiny_config(n_routed_experts=4, top_k=1)
    moe = MoELayer(config)
    x = torch.randn(1, 1, config.d_model)
    output, aux_loss = moe(x)
    assert output.shape == (1, 1, config.d_model)
    assert torch.isfinite(output).all()
    assert torch.isfinite(aux_loss)


def test_moe_layer_gradients_flow_to_dispatched_experts_only():
    """An expert with zero dispatched tokens must receive no gradient this
    batch (Requirement 2.5); the dispatched expert must."""
    config = _tiny_config(n_routed_experts=4, top_k=1, n_shared_experts=0)
    moe = MoELayer(config)
    x = torch.randn(1, 1, config.d_model)

    output, _ = moe(x)
    output.sum().backward()

    got_grad = [
        all(p.grad is not None for p in expert.parameters()) for expert in moe.routed_experts
    ]
    assert sum(got_grad) == 1


def test_moe_layer_loss_free_balancing_selects_k_and_weights_sum_to_one():
    """Requirement 9.4/12.3: LossFreeBalancer must satisfy the exact same
    top-k/renormalization contract as AuxLossBalancer, and reports no
    gradient-based aux loss."""
    config = _tiny_config(moe_balancing="loss_free")
    moe = MoELayer(config)
    x = torch.randn(2, 5, config.d_model)
    output, aux_loss = moe(x)
    assert output.shape == (2, 5, config.d_model)
    assert torch.isfinite(output).all()
    assert aux_loss is None


# -- LossFreeBalancer (Requirement 12.3) -------------------------------------


def test_loss_free_balancer_bias_is_a_buffer_not_a_parameter():
    """The bias must be excluded from the optimizer (design.md), so it has
    to be a buffer, never an nn.Parameter."""
    balancer = LossFreeBalancer(n_routed_experts=4)
    assert list(balancer.parameters()) == []
    assert "bias" in dict(balancer.named_buffers())
    assert torch.equal(balancer.bias, torch.zeros(4))


def test_loss_free_balancer_aux_loss_is_always_none():
    balancer = LossFreeBalancer(n_routed_experts=4)
    probs = torch.softmax(torch.randn(5, 4), dim=-1)
    indices = torch.randint(0, 4, (5, 2))
    assert balancer.aux_loss(probs, indices) is None


def test_loss_free_balancer_selection_scores_adds_bias_to_softmax_probs():
    """The bias is added to the *softmax* gating score (Eq. 3, Wang et al.
    2024) -- not to the raw pre-softmax logit -- see journal entry 015."""
    balancer = LossFreeBalancer(n_routed_experts=3)
    balancer.bias[:] = torch.tensor([0.1, -0.2, 0.3])
    logits = torch.tensor([[1.0, 2.0, 0.5]])

    expected = torch.softmax(logits, dim=-1) + balancer.bias
    assert torch.allclose(balancer.selection_scores(logits), expected)


def test_loss_free_balancer_selection_scores_top_k_and_weights_sum_to_one():
    """Requirement 9.4/12.3: exactly k experts selected, weights sum to 1,
    even with a nonzero bias in play."""
    balancer = LossFreeBalancer(n_routed_experts=4)
    balancer.bias[:] = torch.tensor([1.0, -1.0, 0.5, -0.5])
    torch.manual_seed(0)
    logits = torch.randn(6, 4)
    probs = torch.softmax(logits, dim=-1)

    scores = balancer.selection_scores(logits)
    topk_indices, topk_weights = select_top_k(scores, probs, top_k=2)

    assert topk_indices.shape == (6, 2)
    for row in topk_indices:
        assert len(set(row.tolist())) == 2
    assert torch.allclose(topk_weights.sum(dim=-1), torch.ones(6), atol=1e-6)


def test_loss_free_balancer_bias_can_flip_which_experts_are_selected():
    """Requirement 12.3: a large enough bias changes *which* experts get
    selected -- proving selection_scores' bias term actually participates
    in the top-k decision, not just a no-op addition."""
    balancer = LossFreeBalancer(n_routed_experts=3)
    logits = torch.tensor([[2.0, 1.9, -3.0]])
    probs = torch.softmax(logits, dim=-1)

    baseline_indices, _ = select_top_k(balancer.selection_scores(logits), probs, top_k=2)
    assert set(baseline_indices[0].tolist()) == {0, 1}  # highest two raw logits

    balancer.bias[:] = torch.tensor([-10.0, 0.0, 10.0])
    biased_indices, _ = select_top_k(balancer.selection_scores(logits), probs, top_k=2)
    assert set(biased_indices[0].tolist()) == {1, 2}  # expert 2's huge bias wins it a slot


def test_loss_free_balancer_update_nudges_overloaded_down_underloaded_up():
    """Requirement 12.3: an expert with above-average load gets its bias
    decreased; below-average load gets increased (Algorithm 1's sign
    update rule)."""
    balancer = LossFreeBalancer(n_routed_experts=4, update_rate=0.1)
    counts = torch.tensor([10.0, 2.0, 4.0, 4.0])  # mean = 5.0

    balancer.update(counts)

    assert balancer.bias.tolist() == pytest.approx([-0.1, 0.1, 0.1, 0.1])


def test_loss_free_balancer_update_leaves_exactly_average_expert_unchanged():
    balancer = LossFreeBalancer(n_routed_experts=3, update_rate=0.1)
    counts = torch.tensor([6.0, 3.0, 3.0])  # mean = 4.0 -> expert 0 over, 1&2 under

    balancer.update(counts)

    assert balancer.bias.tolist() == pytest.approx([-0.1, 0.1, 0.1])


def test_loss_free_balancer_bias_flattens_selection_frequency_over_repeated_imbalance():
    """Task 15.1's required test: repeated imbalanced batches should
    measurably flatten expert-selection frequency over steps -- a
    persistently-dominant expert's bias keeps dropping until it stops
    winning the top-1 slot."""
    n_experts = 4
    balancer = LossFreeBalancer(n_routed_experts=n_experts, update_rate=0.05)
    logits = torch.tensor([[5.0, 1.0, 1.0, -5.0]])  # expert 0 dominates by a landslide
    probs = torch.softmax(logits, dim=-1)

    winners = []
    for _ in range(40):
        scores = balancer.selection_scores(logits)
        topk_indices, _ = select_top_k(scores, probs, top_k=1)
        winners.append(topk_indices.item())
        counts = torch.bincount(topk_indices.flatten(), minlength=n_experts)
        balancer.update(counts)

    early_expert0_rate = winners[:5].count(0) / 5
    late_expert0_rate = winners[-5:].count(0) / 5
    assert early_expert0_rate == 1.0
    assert late_expert0_rate < early_expert0_rate


def test_moe_layer_updates_loss_free_bias_automatically_in_training_mode():
    """The bias update happens as a side effect of a normal training-mode
    forward call -- Trainer needs no special-casing per balancing
    strategy, mirroring how build_expert/build_balancer already hide
    variant differences from their callers."""
    torch.manual_seed(0)
    config = _tiny_config(moe_balancing="loss_free")
    moe = MoELayer(config)
    assert moe.training
    before = moe.balancer.bias.clone()

    moe(torch.randn(2, 5, config.d_model))

    assert not torch.equal(before, moe.balancer.bias)


def test_moe_layer_does_not_update_loss_free_bias_in_eval_mode():
    """Eval-mode forward passes (Trainer._evaluate) must not perturb the
    bias with validation-set load statistics."""
    torch.manual_seed(0)
    config = _tiny_config(moe_balancing="loss_free")
    moe = MoELayer(config)
    moe.eval()
    before = moe.balancer.bias.clone()

    moe(torch.randn(2, 5, config.d_model))

    assert torch.equal(before, moe.balancer.bias)


def test_moe_layer_loss_free_bias_never_receives_a_gradient():
    config = _tiny_config(moe_balancing="loss_free")
    moe = MoELayer(config)
    x = torch.randn(2, 5, config.d_model)

    output, _ = moe(x)
    output.sum().backward()

    assert moe.balancer.bias.grad is None
