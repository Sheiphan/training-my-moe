"""Optimizer construction (moe-modernization spec, Requirement 1): splits
parameters into weight-decay / no-decay groups and builds the configured
optimizer. `build_optimizer` is the only entry point `Trainer` calls;
`split_param_groups` is exposed separately so it's directly unit-testable
without constructing a full optimizer.

Today this module has one live branch (`optimizer="adamw"`); a `"muon"`
branch lands in a later task of the same spec (`.kiro/specs/
moe-modernization/tasks.md`, task 4) without changing this function's
signature.
"""
from __future__ import annotations

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


def build_optimizer(model: nn.Module, config: TrainConfig) -> torch.optim.Optimizer:
    """Constructs the optimizer named by `config.optimizer` (Requirement
    1.4-1.5), with weight decay applied only to the `decay` group
    (Requirement 1.1) and `betas`/`eps`/`weight_decay` read from config
    (Requirement 1.2) instead of PyTorch's defaults.
    """
    decay, no_decay = split_param_groups(model)
    groups = [
        {"params": decay, "weight_decay": config.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    groups = [g for g in groups if g["params"]]  # AdamW rejects an empty param list

    if config.optimizer == "adamw":
        return torch.optim.AdamW(groups, lr=config.lr, betas=config.betas, eps=config.adam_eps)
    raise ValueError(
        f"Unknown optimizer: {config.optimizer!r} "
        "('muon' lands in moe-modernization task 4)"
    )
