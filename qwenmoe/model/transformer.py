"""Full decoder-only model: embedding lookup, stacked `TransformerBlock`s,
a final norm, and a (optionally tied) LM head (Requirement 1.1, 1.2, 1.7).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from qwenmoe.config import ModelConfig
from qwenmoe.model.block import TransformerBlock
from qwenmoe.model.rmsnorm import RMSNorm
from qwenmoe.model.trace import LayerTrace, TraceBundle

# Names of the residual-path *output* projections -- the linears that write
# a block's contribution back into the residual stream. Their init std is
# additionally divided by sqrt(2 * n_layers) (moe-modernization Requirement
# 9.2, the GPT-2/Llama residual-scaling rule -- see journal entry 023).
# `w_o`: Attention / MLAAttention output projection; `w_down`: SwiGLU/GEGLU
# expert down-projection; `w_2`: plain-GELU expert down-projection.
_RESIDUAL_PROJ_SUFFIXES = ("w_o.weight", "w_down.weight", "w_2.weight")


def _trunc_normal_(tensor: torch.Tensor, std: float) -> None:
    """Truncated-normal init at the given std, clipped at +/-2 std (the
    GPT-2/T5 convention). In place."""
    nn.init.trunc_normal_(tensor, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std)


class QwenMoEModel(nn.Module):
    """`forward(input_ids) -> (logits, {"aux_loss": ...})`. `aux_loss` is
    the mean (not sum) of every block's MoE auxiliary loss, so its scale
    stays comparable across configs with a different `n_layers`; it's
    `None` only if every block's balancer produced no loss to report.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layers))
        self.final_norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embedding.weight  # Requirement 1.7

        # Principled init (moe-modernization Requirement 9): truncated
        # normal at `init_std` for every linear + the embedding (replacing
        # PyTorch's default Kaiming-uniform on the linears), then a second
        # pass scaling the residual-path output projections down by
        # 1/sqrt(2 * n_layers). RMSNorm gains keep their all-ones init.
        self.apply(self._init_weights)
        residual_std = config.init_std / math.sqrt(2 * config.n_layers)
        for name, param in self.named_parameters():
            if name.endswith(_RESIDUAL_PROJ_SUFFIXES):
                _trunc_normal_(param, residual_std)

    def _init_weights(self, module: nn.Module) -> None:
        """Called on every submodule by `self.apply`. A tied LM head
        (`lm_head.weight is embedding.weight`) is skipped here so it keeps
        exactly the embedding's init, not a second linear-style
        re-init (Requirement 9.3) -- harmless in practice since both use
        the same `init_std`, but explicit so the intent is unambiguous.
        """
        if isinstance(module, nn.Linear):
            if module is self.lm_head and self.config.tie_word_embeddings:
                return
            _trunc_normal_(module.weight, self.config.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            _trunc_normal_(module.weight, self.config.init_std)

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, dict]:
        seq_len = input_ids.shape[1]
        positions = torch.arange(seq_len, device=input_ids.device)

        x = self.embedding(input_ids)
        aux_losses = []
        for block in self.blocks:
            x, aux_loss = block(x, positions)
            if aux_loss is not None:
                aux_losses.append(aux_loss)

        x = self.final_norm(x)
        logits = self.lm_head(x)
        combined_aux_loss = torch.stack(aux_losses).mean() if aux_losses else None
        return logits, {"aux_loss": combined_aux_loss}

    def forward_with_trace(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, TraceBundle]:
        """Requirement 7.1: the exact same computation as `forward`,
        additionally capturing every intermediate tensor a Manim scene
        might need into a `TraceBundle`. Only supports a single sequence
        at a time (`batch == 1`) since `TraceBundle`'s tensors are
        documented without a batch dimension -- this traces one real
        example prompt, never a training batch.
        """
        if input_ids.shape[0] != 1:
            raise ValueError(
                f"forward_with_trace only supports batch size 1, got {input_ids.shape[0]}"
            )
        seq_len = input_ids.shape[1]
        positions = torch.arange(seq_len, device=input_ids.device)

        embeddings = self.embedding(input_ids)
        x = embeddings
        layer_traces: list[LayerTrace] = []
        for block in self.blocks:
            x, _aux_loss, layer_trace = block.forward_with_trace(x, positions)
            layer_traces.append(layer_trace)

        final_norm_output = self.final_norm(x)
        logits = self.lm_head(final_norm_output)
        probs = F.softmax(logits[0, -1], dim=-1)

        bundle = TraceBundle(
            token_ids=input_ids[0].tolist(),
            embeddings=embeddings.squeeze(0),
            layers=layer_traces,
            final_norm_output=final_norm_output.squeeze(0),
            logits=logits.squeeze(0),
            probs=probs,
        )
        return logits, bundle
