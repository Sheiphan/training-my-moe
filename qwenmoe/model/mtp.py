"""Multi-Token Prediction heads (moe-modernization Requirement 6).

An MTP head is an *optional*, *training-only* module that predicts a token
two-or-more steps ahead, giving the trunk a denser training signal than
plain next-token prediction. Design follows DeepSeek-V3 (arXiv:2412.19437
Sec. 2.1.3) and Gloeckle et al. (arXiv:2404.19737): each head fuses the
previous depth's hidden state with the embedding of the token the model is
being taught to expect next, runs one lightweight pre-norm transformer
block, and projects through the *shared* output head. Depth `d` consumes
depth `d-1`'s hidden state -- a causal chain, so head `d` really predicts
one step further than head `d-1`.

`ModelConfig.mtp_depth == 0` (the default) means `QwenMoEModel` builds no
heads at all: zero extra parameters, no extra loss term, `forward`
byte-identical to before this task.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from qwenmoe.config import ModelConfig
from qwenmoe.model.attention import build_attention
from qwenmoe.model.moe import SwiGLUExpert
from qwenmoe.model.rmsnorm import RMSNorm


class MTPHead(nn.Module):
    """One Multi-Token-Prediction module.

    `forward(hidden, next_token_emb, positions)`:
    - `hidden`      `(B, L, d_model)` -- the previous depth's hidden state
      (the trunk's pre-final-norm residual stream for depth 1).
    - `next_token_emb` `(B, L, d_model)` -- `embedding(token[i + depth])`
      at each position `i`, i.e. the token this head is *given* so it can
      predict the one after it (teacher forcing).
    - `positions`   `(L,)` -- absolute positions, for the attention's
      positional transform.

    Returns a new `(B, L, d_model)` hidden state; the caller projects it
    through the shared `lm_head`.

    "Lightweight" = attention + a *dense* SwiGLU FFN (no MoE routing, so no
    router, no per-head auxiliary loss to juggle), each pre-normed with a
    residual, plus the DeepSeek-V3 fusion projection on the way in and an
    output norm on the way out.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.hidden_norm = RMSNorm(config.d_model)
        self.emb_norm = RMSNorm(config.d_model)
        self.fuse = nn.Linear(2 * config.d_model, config.d_model, bias=False)
        self.attn_norm = RMSNorm(config.d_model)
        self.attn = build_attention(config)
        self.ffn_norm = RMSNorm(config.d_model)
        self.ffn = SwiGLUExpert(config.d_model, config.d_ff)
        self.out_norm = RMSNorm(config.d_model)

    def forward(
        self,
        hidden: torch.Tensor,
        next_token_emb: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        fused = self.fuse(
            torch.cat([self.hidden_norm(hidden), self.emb_norm(next_token_emb)], dim=-1)
        )
        x = fused + self.attn(self.attn_norm(fused), positions)
        x = x + self.ffn(self.ffn_norm(x))
        return self.out_norm(x)
