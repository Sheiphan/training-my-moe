"""Optimizer construction (moe-modernization spec, Requirement 1): splits
parameters into weight-decay / no-decay groups and builds the configured
optimizer. `build_optimizer` is the only entry point `Trainer` calls;
`split_param_groups` is exposed separately so it's directly unit-testable
without constructing a full optimizer.

Two optimizers are supported (`TrainConfig.optimizer`):
- `"adamw"` (default): a single `torch.optim.AdamW` over two param groups
  (decay / no-decay).
- `"muon"` (task 4): every 2-D *hidden* weight matrix (attention
  projections, expert `w_gate`/`w_up`/`w_down`, the router gate) is
  optimized by a from-scratch `Muon` (momentum + Newton-Schulz
  orthogonalization); everything else -- embeddings, the LM head, RMSNorm
  gains, biases, any 1-D tensor -- stays on AdamW. `CombinedOptimizer` fans
  `step`/`zero_grad`/`state_dict`/`load_state_dict` out to both and
  concatenates `param_groups` so `Trainer`'s existing per-step LR-schedule
  loop (`for g in optimizer.param_groups: g["lr"] = ...`) needs no
  optimizer-specific branching.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from qwenmoe.config import TrainConfig
from qwenmoe.model.rmsnorm import RMSNorm

# Module types whose parameters never get weight decay, regardless of shape:
# an RMSNorm gain and a token-embedding table are both 2-D-or-otherwise
# "matrix-shaped" tensors that decaying-towards-zero would actively hurt
# (Requirement 1.1). Classified by *module type*, not just `ndim`, since a
# router/attention weight matrix is also 2-D and *should* decay.
_NO_DECAY_MODULE_TYPES: tuple[type[nn.Module], ...] = (RMSNorm, nn.Embedding)

# Newton-Schulz quintic coefficients (Jordan et al.'s Muon; see
# `zeropower_via_newtonschulz5`'s docstring for the derivation). Fixed,
# hand-tuned constants -- not derived from the weight being orthogonalized.
_NS_COEFFS = (3.4445, -4.7750, 2.0315)


def split_param_groups(model: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """`(decay, no_decay)`. A parameter is `no_decay` if it belongs to an
    `_NO_DECAY_MODULE_TYPES` module or has `ndim < 2` (a bias, or any other
    1-D tensor); everything else -- attention/expert/router weight
    matrices -- is `decay`. Walks `model.modules()` with `recurse=False`
    parameter fetches so every parameter is visited exactly once even
    though submodules nest; a tied weight (e.g. `tie_word_embeddings=True`
    sharing one `Parameter` between `embedding` and `lm_head`) is
    deduplicated by `id()` and lands in whichever group its *first*-
    registered owning module maps to -- `embedding` is registered before
    `lm_head` in `QwenMoEModel.__init__`, so a tied embedding is always
    `no_decay`, deterministically.
    """
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    seen: set[int] = set()

    for module in model.modules():
        is_no_decay_module = isinstance(module, _NO_DECAY_MODULE_TYPES)
        for param in module.parameters(recurse=False):
            if not param.requires_grad or id(param) in seen:
                continue
            seen.add(id(param))
            if is_no_decay_module or param.ndim < 2:
                no_decay.append(param)
            else:
                decay.append(param)

    return decay, no_decay


def split_muon_adamw_params(model: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """`(muon_params, adamw_params)` for Requirement 1.4's optimizer split
    (the standard Muon convention, e.g. Jordan et al.'s NanoGPT speedrun
    writeup): **only hidden matmul weights** -- 2-D tensors that aren't the
    embedding table or the LM head projection -- get the orthogonalized
    Muon update; everything shape-like-a-lookup-table or shape-like-a-bias
    stays on AdamW, same as `split_param_groups` excludes `RMSNorm`/
    `nn.Embedding` from weight decay by *module type*, not just `ndim`.

    The model's `lm_head` submodule (if present) is excluded by identity
    even when its weight is a plain 2-D matrix, because `ModelConfig.
    tie_word_embeddings=False` makes it a genuinely separate parameter from
    the embedding table -- Requirement 1.4 names it explicitly as an
    AdamW parameter. When tied (`lm_head.weight is embedding.weight`), the
    `id()` dedup below means the tied weight is already claimed by
    `embedding` (registered first in `QwenMoEModel.__init__`) before
    `lm_head` is ever visited, so the identity check is only load-bearing
    for the untied case -- but it's cheap and correct either way.
    """
    lm_head = getattr(model, "lm_head", None)
    muon: list[nn.Parameter] = []
    adamw: list[nn.Parameter] = []
    seen: set[int] = set()

    for module in model.modules():
        is_excluded_module = isinstance(module, nn.Embedding) or module is lm_head
        for param in module.parameters(recurse=False):
            if not param.requires_grad or id(param) in seen:
                continue
            seen.add(id(param))
            if not is_excluded_module and param.ndim == 2:
                muon.append(param)
            else:
                adamw.append(param)

    return muon, adamw


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    r"""Approximates the "zeroth power" (orthogonalization) of a 2-D matrix
    $G$: replaces it with the nearest semi-orthogonal matrix (all singular
    values pushed to $\approx 1$, directions -- the singular vectors --
    unchanged), without ever computing an SVD.

    **Derivation.** For a matrix $X$ with singular value decomposition
    $X = U\Sigma V^\top$, define $f(X) = aX + b(XX^\top)X + c(XX^\top)^2 X$.
    Since $XX^\top = U\Sigma^2 U^\top$, applying $f$ leaves the singular
    *vectors* $U, V$ untouched and maps every singular value $\sigma$
    through the same scalar quintic $p(\sigma) = a\sigma + b\sigma^3 +
    c\sigma^5$ independently (because $U^\top U = I$ lets the $\Sigma$
    terms be handled one singular value at a time). Iterating $f$ five
    times composes $p$ with itself five times; $(a, b, c) = (3.4445,
    -4.7750, 2.0315)$ is chosen (Jordan et al., empirically, via Chebyshev-
    style optimization -- not derived in closed form) so that $p^{\circ 5}$
    maps the whole interval $\sigma \in (0, 1]$ close to $1$ *faster* than
    the classical Newton-Schulz coefients $(1.5, -0.5)$ (which only converge
    once $\sigma$ is already near 1), while $\|G\|_F$-normalizing $G$ first
    guarantees every singular value starts in $(0, 1]$. The bf16 cast is a
    speed/memory choice (Muon's own reference implementation): five small
    matmuls in bf16 are cheap relative to one fp32 SVD, and the polynomial
    map is stable under bf16's reduced precision because it only ever
    pushes values *towards* $1$, never explodes them.

    Handles non-square $G$ by transposing so the iterated matrix is always
    at least as tall as it is wide (the $XX^\top$ inner product is then the
    smaller of the two possible Gram matrices, which is both cheaper and
    numerically better-conditioned).
    """
    assert G.ndim == 2, f"zeropower_via_newtonschulz5 expects a 2-D matrix, got ndim={G.ndim}"
    a, b, c = _NS_COEFFS
    X = G.bfloat16()
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.T
    X = X / (X.norm() + eps)  # guarantees every singular value starts in (0, 1]
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    """Momentum SGD whose update is orthogonalized by
    `zeropower_via_newtonschulz5` before being applied (Jordan et al.,
    2024/2025; building on Bernstein & Newhouse's "old optimizer, new
    norm" framing of orthogonalized updates as steepest descent under the
    spectral norm rather than the Euclidean norm AdamW implicitly uses).
    Only ever constructed by `build_optimizer` over the 2-D hidden-weight
    subset `split_muon_adamw_params` selects -- `step` asserts `p.ndim == 2`
    per parameter so a future wiring bug (e.g. accidentally routing a bias
    here) fails loudly instead of silently orthogonalizing a vector.

    Per parameter, each `step()`:
        buf = momentum * buf + grad
        update = zeropower_via_newtonschulz5(buf)      # ~orthogonal
        p -= lr * sqrt(max(1, rows / cols)) * update

    The `sqrt(max(1, rows/cols))` factor keeps the *update's* root-mean-
    square entry size roughly shape-invariant: an orthogonal matrix's rows
    are unit-norm, so a wide matrix (`cols > rows`) already has small
    per-entry magnitude, while a tall matrix (`rows > cols`) needs the
    boost -- this is Muon's own published scaling rule, not derived from
    first principles here.
    """

    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95, ns_steps: int = 5) -> None:
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"momentum must be in [0, 1), got {momentum}")
        defaults = dict(lr=lr, momentum=momentum, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            ns_steps = group["ns_steps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                assert p.ndim == 2, f"Muon only supports 2-D parameters, got ndim={p.ndim}"
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(p.grad)

                update = zeropower_via_newtonschulz5(buf, steps=ns_steps)
                scale = math.sqrt(max(1.0, p.size(0) / p.size(1)))
                p.add_(update, alpha=-lr * scale)

        return loss


class CombinedOptimizer:
    """Fans `step`/`zero_grad`/`state_dict`/`load_state_dict` out over
    `>= 1` sub-optimizers (here, always exactly `[Muon, AdamW]`) so
    `Trainer` can treat `TrainConfig.optimizer in {"adamw", "muon"}`
    identically after construction. Deliberately does **not** subclass
    `torch.optim.Optimizer` -- it has no unified per-parameter state of its
    own to own, just delegation.

    `param_groups` concatenates every sub-optimizer's groups so `Trainer`'s
    existing `for g in optimizer.param_groups: g["lr"] = ...` schedule loop
    needs no optimizer-specific branch; each group carries its own
    `"base_lr"` (set at construction: `config.lr` for AdamW groups,
    `config.muon_lr or config.lr` for the Muon group) so `_set_lr` can
    apply one shared multiplier to two different peak learning rates.
    """

    def __init__(self, optimizers: list[torch.optim.Optimizer]) -> None:
        assert len(optimizers) >= 1, "CombinedOptimizer needs at least one sub-optimizer"
        self.optimizers = optimizers

    @property
    def param_groups(self) -> list[dict]:
        return [g for opt in self.optimizers for g in opt.param_groups]

    @property
    def state(self) -> dict:
        merged: dict = {}
        for opt in self.optimizers:
            merged.update(opt.state)
        return merged

    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for opt in self.optimizers:
            opt.step()
        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict:
        return {"sub_optimizers": [opt.state_dict() for opt in self.optimizers]}

    def load_state_dict(self, state_dict: dict) -> None:
        for opt, sub_state in zip(self.optimizers, state_dict["sub_optimizers"]):
            opt.load_state_dict(sub_state)


_FUSED_ADAMW_DEVICES = ("cuda", "cpu", "xpu")


def _fused_adamw_supported(params: list[nn.Parameter]) -> bool:
    """Whether `torch.optim.AdamW(fused=True)` can take these parameters.

    Worth the check rather than a flag, because the default AdamW path
    costs **one host-device synchronization per parameter, per step**:
    `_get_value(step)` in `torch/optim/optimizer.py` calls `.item()` on each
    parameter's step counter. Measured on `SCALE_300M`'s shape (10 layers,
    ~339 parameters) that was **339 of the 362 syncs in an entire optimizer
    step** -- an order of magnitude more than every sync in the trainer and
    the MoE dispatch combined, and invisible in the trainer's own source
    because it happens inside `optimizer.step()`.

    The fused path keeps the step counter on-device and does the whole
    update in one kernel: measured **0** `.item()` calls. Not available on
    MPS, which is why this is a capability probe and not a config flag --
    the laptop path keeps exactly the optimizer it had.
    """
    if not params:
        return False
    devices = {p.device.type for p in params}
    return len(devices) == 1 and devices.pop() in _FUSED_ADAMW_DEVICES


def build_optimizer(model: nn.Module, config: TrainConfig) -> torch.optim.Optimizer:
    """Constructs the optimizer named by `config.optimizer` (Requirement
    1.4-1.5), with weight decay applied only to the `decay` group
    (Requirement 1.1) and `betas`/`eps`/`weight_decay` read from config
    (Requirement 1.2) instead of PyTorch's defaults.
    """
    decay, no_decay = split_param_groups(model)

    if config.optimizer == "adamw":
        groups = [
            {"params": decay, "weight_decay": config.weight_decay, "base_lr": config.lr},
            {"params": no_decay, "weight_decay": 0.0, "base_lr": config.lr},
        ]
        groups = [g for g in groups if g["params"]]  # AdamW rejects an empty param list
        return torch.optim.AdamW(
            groups,
            lr=config.lr,
            betas=config.betas,
            eps=config.adam_eps,
            fused=_fused_adamw_supported(decay + no_decay) or None,
        )

    if config.optimizer == "muon":
        muon_params, adamw_params = split_muon_adamw_params(model)
        adamw_ids = {id(p) for p in adamw_params}
        # adamw_params is the complement of muon_params; re-derive its own
        # decay/no-decay split from split_param_groups's classification
        # (Requirement 1.1 still applies to whichever parameters land on
        # AdamW, muon branch or not) rather than recomputing it by hand.
        adamw_decay = [p for p in decay if id(p) in adamw_ids]
        adamw_no_decay = [p for p in no_decay if id(p) in adamw_ids]
        adamw_groups = [
            {"params": adamw_decay, "weight_decay": config.weight_decay, "base_lr": config.lr},
            {"params": adamw_no_decay, "weight_decay": 0.0, "base_lr": config.lr},
        ]
        adamw_groups = [g for g in adamw_groups if g["params"]]
        adamw_opt = torch.optim.AdamW(
            adamw_groups,
            lr=config.lr,
            betas=config.betas,
            eps=config.adam_eps,
            fused=_fused_adamw_supported(adamw_decay + adamw_no_decay) or None,
        )

        muon_lr = config.muon_lr if config.muon_lr is not None else config.lr
        muon_opt = Muon(
            [{"params": muon_params, "base_lr": muon_lr}],
            lr=muon_lr,
            momentum=config.muon_momentum,
        )
        return CombinedOptimizer([muon_opt, adamw_opt])

    raise ValueError(f"Unknown optimizer: {config.optimizer!r} (expected 'adamw' or 'muon')")
