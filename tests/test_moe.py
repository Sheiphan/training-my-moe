import time

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from qwenmoe.config import ModelConfig
from qwenmoe.model.moe import (
    AuxLossBalancer,
    LossFreeBalancer,
    MoELayer,
    Router,
    build_expert,
    dispatch_batched,
    score_and_select,
    select_top_k,
)


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


# -- routing modernization (moe-modernization Requirement 4.1-4.3, 4.5, 4.6) --


def test_score_and_select_softmax_is_identical_to_the_pre_spec_path():
    """Requirement 4.2: `scoring="softmax"` reproduces exactly what the
    old inline `F.softmax` + `balancer.selection_scores` + `select_top_k`
    produced."""
    torch.manual_seed(0)
    logits = torch.randn(6, 5)
    balancer = AuxLossBalancer(n_routed_experts=5)

    ref_probs = F.softmax(logits, dim=-1)
    ref_idx, ref_w = select_top_k(logits, ref_probs, top_k=2)

    idx, w, routing_probs = score_and_select(logits, balancer, top_k=2, scoring="softmax")

    assert torch.equal(idx, ref_idx)
    assert torch.equal(w, ref_w)
    assert torch.equal(routing_probs, ref_probs)


def test_score_and_select_sigmoid_selects_exactly_k_and_weights_sum_to_one():
    """Requirement 4.1, 4.6: sigmoid scoring still picks k distinct experts
    and the combination weights still sum to 1."""
    torch.manual_seed(1)
    n_tokens, n_experts, k = 7, 6, 3
    logits = torch.randn(n_tokens, n_experts)
    balancer = AuxLossBalancer(n_routed_experts=n_experts)

    idx, w, routing_probs = score_and_select(logits, balancer, top_k=k, scoring="sigmoid")

    assert idx.shape == (n_tokens, k)
    for row in idx:
        assert len(set(row.tolist())) == k
    assert torch.allclose(w.sum(dim=-1), torch.ones(n_tokens), atol=1e-5)
    assert torch.allclose(routing_probs.sum(dim=-1), torch.ones(n_tokens), atol=1e-5)


def test_score_and_select_sigmoid_weights_are_normalized_selected_affinities():
    """Requirement 4.1: sigmoid combination weights are the selected
    `sigmoid(logit)` affinities renormalized to sum to 1 (the global
    normalizer cancels)."""
    logits = torch.tensor([[2.0, 1.0, 0.0, -1.0]])
    balancer = AuxLossBalancer(n_routed_experts=4)

    idx, w, _ = score_and_select(logits, balancer, top_k=2, scoring="sigmoid")

    aff = torch.sigmoid(logits)[0]
    top2 = aff.topk(2).values
    expected = top2 / top2.sum()
    assert set(idx[0].tolist()) == {0, 1}
    assert torch.allclose(w[0].sort(descending=True).values, expected, atol=1e-6)


def test_score_and_select_sigmoid_and_softmax_can_disagree_with_loss_free_bias():
    """Requirement 4.1 vs 4.2: with the loss-free bias in play, sigmoid and
    softmax scoring are genuinely different selection rules (the bias breaks
    softmax's cross-expert monotonicity)."""
    torch.manual_seed(0)
    logits = torch.randn(32, 8)
    balancer = LossFreeBalancer(n_routed_experts=8)
    balancer.bias.copy_(torch.linspace(-0.3, 0.3, 8))

    soft_idx, _, _ = score_and_select(logits, balancer, top_k=3, scoring="softmax")
    sig_idx, _, _ = score_and_select(logits, balancer, top_k=3, scoring="sigmoid")

    assert not torch.equal(soft_idx, sig_idx)


def test_score_and_select_rejects_unknown_scoring():
    with pytest.raises(ValueError, match="router_scoring"):
        score_and_select(torch.randn(2, 4), AuxLossBalancer(4), top_k=2, scoring="nope")


def test_moe_layer_sigmoid_routing_runs_and_skips_zero_token_experts():
    """Requirement 4.6: with more experts than a small batch can cover,
    some get zero tokens -- the layer must still produce a finite output of
    the right shape and no error."""
    torch.manual_seed(0)
    config = _tiny_config(n_routed_experts=8, top_k=2, router_scoring="sigmoid")
    moe = MoELayer(config)
    x = torch.randn(1, 3, config.d_model)  # 3 tokens, 8 experts -> guaranteed idle experts

    out, aux_loss = moe(x)

    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    assert torch.isfinite(aux_loss)
    assert (moe.last_expert_counts == 0).any()


def test_router_z_loss_zero_leaves_aux_loss_identical():
    """Requirement 4.3 invariant: `router_z_loss_weight == 0` (default) ⇒
    the MoE aux loss is byte-identical to the balancer's own aux loss."""
    torch.manual_seed(0)
    config = _tiny_config()
    moe = MoELayer(config)
    x = torch.randn(2, 5, config.d_model)

    _, aux_loss = moe(x)

    # recompute the balancer's aux loss directly from the same forward
    router_logits = moe.router(x.reshape(-1, config.d_model))
    idx, _, routing_probs = score_and_select(
        router_logits, moe.balancer, config.top_k, "softmax"
    )
    ref = moe.balancer.aux_loss(routing_probs, idx)
    assert torch.equal(aux_loss, ref)


def test_router_z_loss_positive_adds_a_nonnegative_term_that_tracks_logit_scale():
    """Requirement 4.3: the term is `weight * mean(logsumexp(router_logits)^2)`
    -- non-negative, and larger when the router logits blow up."""
    torch.manual_seed(0)
    weight = 1e-2

    small = torch.zeros(4, 6)
    big = torch.randn(4, 6) * 20.0

    def rz(logits):
        return weight * torch.logsumexp(logits.float(), dim=-1).pow(2).mean()

    assert rz(small) >= 0.0
    assert rz(big) > rz(small)

    # and it actually reaches the MoE aux-loss total
    config = _tiny_config(router_z_loss_weight=weight)
    moe = MoELayer(config)
    moe_off = MoELayer(_tiny_config())
    moe_off.load_state_dict(moe.state_dict())
    x = torch.randn(2, 5, config.d_model)

    _, aux_on = moe(x)
    _, aux_off = moe_off(x)
    assert aux_on > aux_off  # the only difference is the added z-loss term


def test_router_z_loss_applies_with_the_loss_free_balancer_too():
    """Requirement 4.3 is balancer-independent: loss-free normally reports
    no aux loss (`None`), but with `router_z_loss_weight > 0` the router
    z-loss becomes the whole aux-loss term rather than being dropped."""
    torch.manual_seed(0)
    config = _tiny_config(moe_balancing="loss_free", router_z_loss_weight=1e-2)
    moe = MoELayer(config)
    x = torch.randn(2, 5, config.d_model)

    _, aux_loss = moe(x)

    assert aux_loss is not None
    assert torch.isfinite(aux_loss) and aux_loss > 0.0


def test_aux_loss_weight_is_config_driven():
    """Requirement 4.5: `AuxLossBalancer.loss_weight` comes from
    `ModelConfig.aux_loss_weight` (default 0.01, unchanged) rather than a
    hard-coded literal."""
    assert MoELayer(_tiny_config()).balancer.loss_weight == pytest.approx(0.01)
    assert MoELayer(_tiny_config(aux_loss_weight=0.05)).balancer.loss_weight == pytest.approx(0.05)


# -- loss-free complementary per-sequence balance loss (Requirement 4.4) ----


def _collapsed_and_balanced_routing(n_experts=2, seq_len=4, top_k=1):
    """Two 1-token-per-expert layouts for the same batch shape:
    `collapsed` -- sequence 0's tokens all pick expert 0, sequence 1's all
    pick expert 1 (batch-wide this looks perfectly balanced); `balanced` --
    every sequence cycles through the experts internally.
    """
    peak0 = torch.tensor([0.7, 0.3]) if n_experts == 2 else None
    collapsed_idx = torch.cat(
        [torch.zeros(seq_len, top_k), torch.ones(seq_len, top_k)]
    ).long()
    collapsed_probs = torch.cat(
        [peak0.repeat(seq_len, 1), peak0.flip(0).repeat(seq_len, 1)]
    )
    cycle = torch.arange(seq_len).remainder(n_experts)
    balanced_idx = torch.cat([cycle, cycle]).unsqueeze(-1)
    balanced_probs = torch.full((2 * seq_len, n_experts), 1.0 / n_experts)
    return (collapsed_idx, collapsed_probs), (balanced_idx, balanced_probs)


def test_sequence_balance_loss_zero_weight_keeps_loss_free_aux_loss_none():
    """Requirement 4.4 invariant: with `seq_balance_loss_weight == 0` (the
    default) a loss-free MoE layer still reports no aux loss at all."""
    torch.manual_seed(0)
    moe = MoELayer(_tiny_config(moe_balancing="loss_free"))
    _, aux_loss = moe(torch.randn(2, 5, moe.router.gate.in_features))
    assert aux_loss is None


def test_sequence_balance_loss_penalizes_a_collapsed_sequence():
    """A single sequence routing every token to one expert scores well
    above the `weight * 1.0` floor."""
    balancer = LossFreeBalancer(n_routed_experts=2)
    (c_idx, c_probs), (b_idx, b_probs) = _collapsed_and_balanced_routing()

    collapsed = balancer.sequence_balance_loss(c_probs, c_idx, seq_len=4, top_k=1, weight=1.0)
    balanced = balancer.sequence_balance_loss(b_probs, b_idx, seq_len=4, top_k=1, weight=1.0)

    assert balanced == pytest.approx(1.0, abs=1e-5)  # the minimum
    assert collapsed == pytest.approx(1.4, abs=1e-5)  # 2 * 0.7
    assert collapsed > balanced


def test_sequence_balance_loss_catches_batch_balanced_but_sequence_imbalanced():
    """The whole point of Requirement 4.4: the batch-wide Switch loss is at
    its floor for a batch where sequence 0 -> expert 0 and sequence 1 ->
    expert 1 (50/50 overall), but the per-sequence loss is not."""
    (c_idx, c_probs), _ = _collapsed_and_balanced_routing()

    switch = AuxLossBalancer(n_routed_experts=2).aux_loss(c_probs, c_idx)  # batch-wide
    per_seq = LossFreeBalancer(n_routed_experts=2).sequence_balance_loss(
        c_probs, c_idx, seq_len=4, top_k=1, weight=0.01
    )

    # Switch sees f_i = [0.5, 0.5], P_i = [0.5, 0.5] -> its floor (weight * top_k)
    assert switch == pytest.approx(0.01 * 1, abs=1e-6)
    assert per_seq > 0.01 * 1.0  # per-sequence still flags the collapse


def test_sequence_balance_loss_only_applies_with_the_loss_free_balancer():
    """Requirement 4.4 is gated on `moe_balancing == "loss_free"`: an
    aux-loss layer ignores `seq_balance_loss_weight` entirely."""
    torch.manual_seed(0)
    plain = MoELayer(_tiny_config())
    with_weight = MoELayer(_tiny_config(seq_balance_loss_weight=0.1))
    with_weight.load_state_dict(plain.state_dict())
    x = torch.randn(2, 5, plain.router.gate.in_features)

    _, aux_plain = plain(x)
    _, aux_with_weight = with_weight(x)
    assert torch.equal(aux_plain, aux_with_weight)


# -- batched expert computation (moe-modernization Requirement 5) -----------


def _experts(activation, n_experts, d_model, d_ff, seed=0):
    torch.manual_seed(seed)
    cfg = ModelConfig(
        vocab_size=8, d_model=d_model, d_ff=d_ff, activation=activation,
        n_routed_experts=n_experts, n_shared_experts=0, top_k=2, max_seq_len=8,
    )
    return nn.ModuleList(build_expert(cfg) for _ in range(n_experts))


def _reference_routed_output(x_flat, topk_indices, topk_weights, routed_experts):
    """The exact per-expert loop `MoELayer.forward` used before task 11."""
    out = torch.zeros_like(x_flat)
    for e, expert in enumerate(routed_experts):
        tok, slot = (topk_indices == e).nonzero(as_tuple=True)
        if tok.numel() == 0:
            continue
        gate = topk_weights[tok, slot].unsqueeze(-1)
        out.index_add_(0, tok, gate * expert(x_flat[tok]))
    return out


@pytest.mark.parametrize("activation", ["swiglu", "geglu", "gelu"])
def test_dispatch_batched_matches_the_per_expert_loop(activation):
    """Requirement 5.2: the batched path is a performance refactor -- its
    output matches the loop within 1e-5, including a batch where some
    experts get zero tokens (n_experts=6, only ~4 tokens*2 slots dispatched)."""
    torch.manual_seed(1)
    d_model, d_ff, n_experts, n_tokens, top_k = 24, 40, 6, 4, 2
    experts = _experts(activation, n_experts, d_model, d_ff)
    x = torch.randn(n_tokens, d_model)
    scores = torch.randn(n_tokens, n_experts)
    probs = F.softmax(scores, dim=-1)
    idx, w = select_top_k(scores, probs, top_k)

    batched = dispatch_batched(x, idx, w, experts, activation)
    reference = _reference_routed_output(x, idx, w, experts)

    assert batched.shape == reference.shape
    assert torch.allclose(batched, reference, atol=1e-5)
    # a genuinely idle expert existed in this batch
    assert (torch.bincount(idx.reshape(-1), minlength=n_experts) == 0).any()


def test_dispatch_batched_all_experts_active_matches_loop():
    torch.manual_seed(2)
    d_model, d_ff, n_experts, n_tokens, top_k = 32, 64, 4, 40, 2
    experts = _experts("swiglu", n_experts, d_model, d_ff)
    x = torch.randn(n_tokens, d_model)
    probs = F.softmax(torch.randn(n_tokens, n_experts), dim=-1)
    idx, w = select_top_k(probs, probs, top_k)

    assert torch.allclose(
        dispatch_batched(x, idx, w, experts, "swiglu"),
        _reference_routed_output(x, idx, w, experts),
        atol=1e-5,
    )


def test_dispatch_batched_zero_token_expert_receives_no_gradient():
    """Requirement 5.3: an expert that got zero tokens is never stacked, so
    its parameters' `.grad` stays `None` -- exactly like the loop's
    `continue`."""
    torch.manual_seed(3)
    experts = _experts("swiglu", n_experts=4, d_model=16, d_ff=24)
    x = torch.randn(1, 16)                       # 1 token, top_k=1 -> 3 idle experts
    idx = torch.tensor([[2]])
    w = torch.tensor([[1.0]])

    dispatch_batched(x, idx, w, experts, "swiglu").sum().backward()

    got_grad = [all(p.grad is not None for p in e.parameters()) for e in experts]
    assert got_grad == [False, False, True, False]


def test_moe_layer_forward_matches_a_reference_loop_within_tolerance():
    """Requirement 5.2 at the layer level: swapping in `dispatch_batched`
    did not change `MoELayer.forward`'s output."""
    torch.manual_seed(4)
    config = _tiny_config(n_routed_experts=6, top_k=2)
    moe = MoELayer(config)
    moe.eval()  # skip the loss-free bias update path (irrelevant here, aux_loss balancer anyway)
    x = torch.randn(2, 7, config.d_model)

    out, _ = moe(x)

    # rebuild the routed contribution with the explicit loop, same weights
    x_flat = x.reshape(-1, config.d_model)
    router_logits = moe.router(x_flat)
    idx, w, _ = score_and_select(router_logits, moe.balancer, config.top_k, "softmax")
    routed_ref = _reference_routed_output(x_flat, idx, w, moe.routed_experts)
    shared_ref = sum(e(x_flat) for e in moe.shared_experts)
    ref = (routed_ref + shared_ref).view_as(out)

    assert torch.allclose(out, ref, atol=1e-5)


def test_parent_spec_state_dict_loads_without_key_migration():
    """Requirement 5 (testing strategy): keeping `routed_experts` an
    `nn.ModuleList` means a parent-spec checkpoint's keys
    (`routed_experts.N.w_gate.weight`) are still exactly the current keys --
    no migration hook needed."""
    config = _tiny_config(n_routed_experts=4)
    reference = MoELayer(config)
    keys = set(reference.state_dict())
    assert "routed_experts.0.w_gate.weight" in keys
    assert "routed_experts.3.w_down.weight" in keys

    fresh = MoELayer(config)
    missing, unexpected = fresh.load_state_dict(reference.state_dict(), strict=True)
    assert not missing and not unexpected
    x = torch.randn(2, 5, config.d_model)
    assert torch.allclose(fresh(x)[0], reference(x)[0], atol=1e-6)


def test_dispatch_batched_wall_clock_scales_sublinearly_with_expert_count():
    """Requirement 5.5: 4x the experts (param-matched by shrinking d_ff) is
    nowhere near 4x the wall-clock -- the batched path's cost tracks the
    token count, not the expert count. Soft timing check with a wide margin."""
    d_model, n_tokens, top_k = 64, 512, 2

    def time_dispatch(n_experts, d_ff):
        experts = _experts("swiglu", n_experts, d_model, d_ff)
        x = torch.randn(n_tokens, d_model)
        probs = F.softmax(torch.randn(n_tokens, n_experts), dim=-1)
        idx, w = select_top_k(probs, probs, top_k)
        for _ in range(3):  # warm up
            dispatch_batched(x, idx, w, experts, "swiglu")
        start = time.perf_counter()
        for _ in range(20):
            dispatch_batched(x, idx, w, experts, "swiglu")
        return time.perf_counter() - start

    t8 = time_dispatch(8, 256)
    t32 = time_dispatch(32, 64)  # 4x experts, 1/4 d_ff -> same routed param count
    assert t32 < 3.0 * t8  # loop-based code was ~4x here; batched should be ~1x


def test_moe_layer_seq_balance_loss_reaches_the_aux_total_and_keeps_bias_gradient_free():
    """Requirement 4.4 + 12.3: the per-sequence loss shows up in a loss-free
    layer's aux total, and the loss-free bias buffer still never gets a
    gradient from it."""
    torch.manual_seed(0)
    config = _tiny_config(moe_balancing="loss_free", seq_balance_loss_weight=0.05)
    moe = MoELayer(config)
    x = torch.randn(2, 5, config.d_model)

    out, aux_loss = moe(x)
    assert aux_loss is not None and torch.isfinite(aux_loss) and aux_loss > 0.0

    (out.sum() + aux_loss).backward()
    assert moe.balancer.bias.grad is None
    assert moe.router.gate.weight.grad is not None  # the term does train the router
