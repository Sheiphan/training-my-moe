"""Mixture-of-Experts feed-forward layer (Requirement 2): expert activation
variants, a router, load-balancing auxiliary loss, and the `MoELayer` that
ties them together.

Every expert variant shares the same interface so `MoELayer` doesn't need
to know which one it holds: `forward(x) -> Tensor`, same shape in and out,
selected by `ModelConfig.activation` (Requirement 12.2).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from qwenmoe.config import ModelConfig


class SwiGLUExpert(nn.Module):
    """SwiGLU-gated feed-forward (Shazeer, 2020) — the default/baseline
    expert activation, as used by Qwen/Llama-style models."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class GEGLUExpert(nn.Module):
    """Same gated structure as `SwiGLUExpert`, but the gate branch uses
    GELU instead of SiLU (Shazeer, 2020; Requirement 12.2)."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.gelu(self.w_gate(x)) * self.w_up(x))


class GELUExpert(nn.Module):
    """Plain (non-gated) two-layer GELU feed-forward (Requirement 12.2) —
    unlike the two variants above, there's no separate gate/up split, so
    this has 2 weight matrices instead of 3."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff, bias=False)
        self.w_2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_2(F.gelu(self.w_1(x)))


def build_expert(config: ModelConfig) -> nn.Module:
    """Constructs the expert activation variant named by
    `config.activation` (Requirement 12.2)."""
    if config.activation == "swiglu":
        return SwiGLUExpert(config.d_model, config.d_ff)
    if config.activation == "geglu":
        return GEGLUExpert(config.d_model, config.d_ff)
    if config.activation == "gelu":
        return GELUExpert(config.d_model, config.d_ff)
    raise ValueError(f"Unknown activation: {config.activation!r}")


class Router(nn.Module):
    """Learned linear layer producing per-expert routing logits
    (Requirement 2.2). Softmax and top-k selection happen in `MoELayer`,
    not here, since the top-k selection scores can differ from the plain
    softmax probabilities depending on the balancing strategy
    (Requirement 12.3)."""

    def __init__(self, d_model: int, n_routed_experts: int):
        super().__init__()
        self.gate = nn.Linear(d_model, n_routed_experts, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gate(x)


def score_and_select(
    router_logits: torch.Tensor,
    balancer: nn.Module,
    top_k: int,
    scoring: str = "softmax",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Turns raw router logits into `(topk_indices, topk_weights,
    routing_probs)` (moe-modernization Requirement 4.1, 4.2).

    - `scoring == "softmax"` (default): exactly the pre-spec path -- a
      softmax over experts is *both* the routing distribution and the source
      of the renormalized top-k combination weights.
    - `scoring == "sigmoid"` (DeepSeek-V3): score each expert independently
      with `sigmoid(logit)`, then normalize. Selected affinities (rescaled
      to sum to 1) become the combination weights; the whole affinity
      vector (also rescaled to sum to 1) is `routing_probs`, used by the
      aux loss and the trace.

    In both cases `topk_weights` sums to 1 per token (Requirement 4.6) and
    `routing_probs` is a per-token distribution over every routed expert, so
    `AuxLossBalancer.aux_loss` and the Manim trace see the same shape and
    semantics regardless of `scoring`.
    """
    if scoring == "softmax":
        routing_probs = F.softmax(router_logits, dim=-1)
    elif scoring == "sigmoid":
        affinities = torch.sigmoid(router_logits)
        routing_probs = affinities / affinities.sum(dim=-1, keepdim=True)
    else:
        raise ValueError(
            f"Unknown router_scoring: {scoring!r} (expected 'softmax' or 'sigmoid')"
        )

    selection_scores = balancer.selection_scores(router_logits, scoring=scoring)
    topk_indices, topk_weights = select_top_k(selection_scores, routing_probs, top_k)
    return topk_indices, topk_weights, routing_probs


def select_top_k(
    selection_scores: torch.Tensor, router_probs: torch.Tensor, top_k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Picks the top-`k` experts per token by `selection_scores`, and
    returns their combination weights renormalized to sum to 1 (Requirement
    2.2, 2.3).

    `selection_scores` and `router_probs` are separate arguments because a
    balancing strategy may bias scores for *selection only*
    (Requirement 12.3) while the combination weights always come from the
    unbiased softmax `router_probs`. For `AuxLossBalancer` the two are
    equivalent (ranking by raw logits or by softmax probability gives the
    same order, since softmax doesn't change relative order within a
    token), so `selection_scores` is just the raw router logits.
    """
    _, topk_indices = selection_scores.topk(top_k, dim=-1)
    topk_weights = router_probs.gather(-1, topk_indices)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return topk_indices, topk_weights


class AuxLossBalancer(nn.Module):
    """Differentiable load-balancing auxiliary loss (Switch Transformer;
    Fedus, Zoph & Shazeer, 2021), Requirement 2.4.

    The paper's own formula is defined for top-1 routing; this generalizes
    `f_i` to "fraction of tokens for which expert i is among the top-k"
    (reducing to the paper's exact definition when `top_k == 1`) since this
    project's baseline uses `top_k=2`. See the journal entry for the full
    derivation and a worked numeric example.
    """

    def __init__(self, n_routed_experts: int, loss_weight: float = 0.01):
        super().__init__()
        self.n_routed_experts = n_routed_experts
        self.loss_weight = loss_weight

    def selection_scores(
        self, router_logits: torch.Tensor, scoring: str = "softmax"
    ) -> torch.Tensor:
        """Unbiased: the raw router logits rank the experts. Both `softmax`
        and `sigmoid` are monotonic in the logit (softmax shares one
        denominator per token; sigmoid is elementwise), so the top-k *set*
        is identical whichever `scoring` is used -- only the combination
        weights differ, and those come from `routing_probs` in
        `score_and_select`, not from here. The `scoring` argument is
        accepted for interface symmetry with `LossFreeBalancer` and ignored.
        """
        return router_logits

    def aux_loss(self, router_probs: torch.Tensor, topk_indices: torch.Tensor) -> torch.Tensor:
        """`router_probs`: (n_tokens, n_routed_experts) softmax output.
        `topk_indices`: (n_tokens, top_k) selected expert ids. Returns a
        scalar, minimized when routing is perfectly uniform across
        experts."""
        n_experts = self.n_routed_experts

        dispatched = F.one_hot(topk_indices, num_classes=n_experts).sum(dim=1)
        fraction_dispatched = dispatched.float().mean(dim=0)  # f_i
        mean_router_prob = router_probs.mean(dim=0)  # P_i

        return self.loss_weight * n_experts * torch.sum(fraction_dispatched * mean_router_prob)

    def update(self, expert_counts: torch.Tensor) -> None:
        """No-op: aux-loss balancing's load signal is entirely gradient-based
        (via `aux_loss` above), so there's no post-step bookkeeping to do."""


class LossFreeBalancer(nn.Module):
    """Auxiliary-loss-free bias-adjusted balancing (DeepSeek-V3's "Loss-Free
    Balancing"; Wang, Gao, Zhao, Sun & Dai, 2024, arXiv:2408.15664),
    Requirement 12.3.

    Holds one non-trainable bias per routed expert (`register_buffer`, so
    `model.parameters()` -- and therefore the optimizer -- never sees it,
    matching design.md exactly). The bias is added to the *softmax* routing
    score for top-k **selection only**; the combination weight a selected
    expert's output gets multiplied by still comes from the plain,
    unbiased `router_probs` passed separately into `select_top_k` --
    mirrored here by `selection_scores` and `aux_loss` (which always
    returns `None`: there's no gradient-based loss term at all) being two
    separate methods, exactly like `AuxLossBalancer`.

    See the journal entry for the full derivation (including why the bias
    is added post-softmax, not to the raw logits) and a worked numeric
    example.
    """

    def __init__(self, n_routed_experts: int, update_rate: float = 0.01):
        super().__init__()
        self.n_routed_experts = n_routed_experts
        self.update_rate = update_rate
        self.register_buffer("bias", torch.zeros(n_routed_experts))

    def selection_scores(
        self, router_logits: torch.Tensor, scoring: str = "softmax"
    ) -> torch.Tensor:
        """Biased: top-k selection uses the routing score plus this expert's
        bias (Eq. 3 of Wang et al., 2024). With `scoring == "sigmoid"` the
        base score is the per-expert `sigmoid(logit)` affinity (DeepSeek-V3's
        actual choice); with `"softmax"` (default) it is the softmax
        probability, exactly as before this spec. The bias breaks softmax's
        monotonicity, so unlike `AuxLossBalancer` the base score genuinely
        matters here. Recomputing it (rather than accepting `router_probs`)
        keeps this signature identical to `AuxLossBalancer`'s."""
        if scoring == "sigmoid":
            base = torch.sigmoid(router_logits)
        else:
            base = F.softmax(router_logits, dim=-1)
        return base + self.bias

    def aux_loss(self, router_probs: torch.Tensor, topk_indices: torch.Tensor) -> None:
        """Always `None`: the whole point of this strategy is controlling
        load balance with no gradient-based loss term (Requirement 12.3)."""
        return None

    @torch.no_grad()
    def update(self, expert_counts: torch.Tensor) -> None:
        """Nudges each expert's bias by a fixed step `update_rate`, in the
        direction that discourages an overloaded expert and encourages an
        underloaded one (the paper's "sign" update rule, Algorithm 1) --
        no gradient involved, just a plain buffer mutation, so this is
        wrapped in `torch.no_grad()` defensively even though addition's
        backward formula doesn't actually need `bias`'s saved value.
        `expert_counts`: (n_routed_experts,) tokens dispatched to each
        expert this batch (`MoELayer.last_expert_counts`).
        """
        counts = expert_counts.float()
        violation = counts.mean() - counts  # e_i = mean_load - load_i
        self.bias += self.update_rate * torch.sign(violation)


def build_balancer(config: ModelConfig) -> nn.Module:
    """Constructs the MoE load-balancing strategy named by
    `config.moe_balancing` (Requirement 12.3)."""
    if config.moe_balancing == "aux_loss":
        # loss_weight is config-driven now (moe-modernization Requirement
        # 4.5); the 0.01 default matches the previously hard-coded value.
        return AuxLossBalancer(config.n_routed_experts, loss_weight=config.aux_loss_weight)
    if config.moe_balancing == "loss_free":
        return LossFreeBalancer(config.n_routed_experts)
    raise ValueError(f"Unknown moe_balancing: {config.moe_balancing!r}")


class MoELayer(nn.Module):
    """Top-k routed experts + always-on shared experts (Requirement
    2.1-2.3): routed-expert outputs are combined with renormalized gate
    weights, shared-expert outputs are added unweighted, and a routed
    expert that receives zero tokens this batch is simply skipped —
    no error, no gradient for it this step (Requirement 2.5).
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.top_k = config.top_k
        self.n_routed_experts = config.n_routed_experts
        # moe-modernization Requirement 4: both default to the pre-spec path
        # ("softmax", 0.0), so a config written before this spec routes
        # bit-identically to today.
        self.router_scoring = config.router_scoring
        self.router_z_loss_weight = config.router_z_loss_weight

        self.router = Router(config.d_model, config.n_routed_experts)
        self.routed_experts = nn.ModuleList(
            build_expert(config) for _ in range(config.n_routed_experts)
        )
        self.shared_experts = nn.ModuleList(
            build_expert(config) for _ in range(config.n_shared_experts)
        )
        self.balancer = build_balancer(config)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch, seq_len, d_model = x.shape
        x_flat = x.reshape(-1, d_model)

        router_logits = self.router(x_flat)
        topk_indices, topk_weights, routing_probs = score_and_select(
            router_logits, self.balancer, self.top_k, self.router_scoring
        )
        # Cached for Trainer's expert-utilization logging (Requirement 8.2, task 10) --
        # not part of forward's return signature, so no existing caller is disturbed.
        self.last_expert_counts = torch.bincount(
            topk_indices.detach().flatten(), minlength=self.n_routed_experts
        )
        if self.training:
            # "After each training step" (design.md) -- gated on train/eval mode
            # exactly like BatchNorm's running-stats update, so an eval-only
            # forward pass (Trainer._evaluate) never perturbs the bias with
            # validation-set load stats. No-op for AuxLossBalancer.
            self.balancer.update(self.last_expert_counts)

        routed_output = torch.zeros_like(x_flat)
        for expert_id, expert in enumerate(self.routed_experts):
            token_idx, slot_idx = (topk_indices == expert_id).nonzero(as_tuple=True)
            if token_idx.numel() == 0:
                continue  # Requirement 2.5: zero tokens this batch, no error, no gradient
            gate = topk_weights[token_idx, slot_idx].unsqueeze(-1)
            # `.to(routed_output.dtype)`: under `torch.autocast(bf16)` the
            # expert's Linear layers emit bf16, but `routed_output` is the
            # fp32 residual stream and `index_add_` is dtype-strict. Upcasting
            # the contribution keeps the scatter-accumulate in fp32 (a plain
            # no-op in the default fp32 path -- byte-identical to before
            # moe-modernization task 8).
            contribution = (gate * expert(x_flat[token_idx])).to(routed_output.dtype)
            routed_output.index_add_(0, token_idx, contribution)

        shared_output = torch.zeros_like(x_flat)
        for expert in self.shared_experts:
            shared_output = shared_output + expert(x_flat)

        output = (routed_output + shared_output).view(batch, seq_len, d_model)
        aux_loss = self.balancer.aux_loss(routing_probs, topk_indices)
        aux_loss = self._add_router_z_loss(aux_loss, router_logits)
        return output, aux_loss

    def _add_router_z_loss(
        self, aux_loss: torch.Tensor | None, router_logits: torch.Tensor
    ) -> torch.Tensor | None:
        """Requirement 4.3: add `router_z_loss_weight * mean(logsumexp(
        router_logits)^2)` -- the ST-MoE penalty on the router's pre-softmax
        logits -- into the MoE aux-loss total. Balancer-independent: it also
        applies with the loss-free balancer, where `aux_loss` is otherwise
        `None` (then the router z-loss becomes the whole term). Off (weight
        `0`, the default) returns `aux_loss` untouched, so the pre-spec path
        is bit-identical.
        """
        if self.router_z_loss_weight <= 0:
            return aux_loss
        z_loss = self.router_z_loss_weight * torch.logsumexp(
            router_logits.float(), dim=-1
        ).pow(2).mean()
        return z_loss if aux_loss is None else aux_loss + z_loss

    def forward_with_trace(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None, dict]:
        """Same computation as `forward`, additionally capturing every
        intermediate tensor `LayerTrace` needs (Requirement 7.1). Only
        `TransformerBlock.forward_with_trace` calls this.
        """
        batch, seq_len, d_model = x.shape
        x_flat = x.reshape(-1, d_model)

        router_logits = self.router(x_flat)
        topk_indices, topk_weights, routing_probs = score_and_select(
            router_logits, self.balancer, self.top_k, self.router_scoring
        )
        self.last_expert_counts = torch.bincount(
            topk_indices.detach().flatten(), minlength=self.n_routed_experts
        )

        routed_output = torch.zeros_like(x_flat)
        expert_outputs: dict[int, torch.Tensor] = {}
        for expert_id, expert in enumerate(self.routed_experts):
            token_idx, slot_idx = (topk_indices == expert_id).nonzero(as_tuple=True)
            if token_idx.numel() == 0:
                continue  # Requirement 2.5, mirrored in the trace: no entry for a skipped expert
            expert_out = expert(x_flat[token_idx])
            expert_outputs[expert_id] = expert_out
            gate = topk_weights[token_idx, slot_idx].unsqueeze(-1)
            routed_output.index_add_(0, token_idx, gate * expert_out)

        shared_output = torch.zeros_like(x_flat)
        for expert in self.shared_experts:
            shared_output = shared_output + expert(x_flat)

        output = (routed_output + shared_output).view(batch, seq_len, d_model)
        aux_loss = self.balancer.aux_loss(routing_probs, topk_indices)
        aux_loss = self._add_router_z_loss(aux_loss, router_logits)

        trace = {
            "router_logits": router_logits,
            "router_probs": routing_probs,
            "topk_indices": topk_indices,
            "topk_weights": topk_weights,
            "expert_outputs": expert_outputs,
        }
        return output, aux_loss, trace
