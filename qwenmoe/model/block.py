"""One decoder layer: pre-norm attention + pre-norm MoE, each wrapped in a
residual connection (Requirement 1.4).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from qwenmoe.config import ModelConfig
from qwenmoe.model.attention import build_attention
from qwenmoe.model.moe import MoELayer
from qwenmoe.model.rmsnorm import RMSNorm
from qwenmoe.model.trace import LayerTrace


class TransformerBlock(nn.Module):
    """`x = x + Attn(RMSNorm(x))`, then `x = x + MoE(RMSNorm(x))`. Norm is
    applied *before* each sublayer (pre-norm) rather than after, per
    Requirement 1.4 and the Llama/Qwen convention — see the journal entry
    for why this is more stable to train than the original post-norm
    Transformer.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model)
        self.attn = build_attention(config)
        self.moe_norm = RMSNorm(config.d_model)
        self.moe = MoELayer(config)

    def forward(
        self, x: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        x = x + self.attn(self.attn_norm(x), positions)
        moe_out, aux_loss = self.moe(self.moe_norm(x))
        x = x + moe_out
        return x, aux_loss

    def forward_with_trace(
        self, x: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None, LayerTrace]:
        """Same computation as `forward`, additionally assembling a
        `LayerTrace` from its attention/MoE sub-layers (Requirement 7.1).
        """
        attn_out, attn_trace = self.attn.forward_with_trace(self.attn_norm(x), positions)
        x = x + attn_out
        moe_out, aux_loss, moe_trace = self.moe.forward_with_trace(self.moe_norm(x))
        x = x + moe_out

        layer_trace = LayerTrace(
            attn_output=attn_out.squeeze(0),
            moe_output=moe_out.squeeze(0),
            **attn_trace,
            **moe_trace,
        )
        return x, aux_loss, layer_trace
