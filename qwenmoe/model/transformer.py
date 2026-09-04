"""Full decoder-only model: embedding lookup, stacked `TransformerBlock`s,
a final norm, and a (optionally tied) LM head (Requirement 1.1, 1.2, 1.7).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from qwenmoe.config import ModelConfig
from qwenmoe.model.block import TransformerBlock
from qwenmoe.model.rmsnorm import RMSNorm
from qwenmoe.model.trace import LayerTrace, TraceBundle

# GPT-2-style embedding init (Radford et al., 2019); PyTorch's nn.Embedding
# default (N(0, 1)) is ~50x too large and, when tied to the LM head
# (Requirement 1.7's default), blows up output logits before any training
# happens -- found while validating task 10's training loop (see journal).
_EMBEDDING_INIT_STD = 0.02


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
        nn.init.normal_(self.embedding.weight, mean=0.0, std=_EMBEDDING_INIT_STD)
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layers))
        self.final_norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embedding.weight  # Requirement 1.7

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
        additionally capturing every intermediate tensor a visualization
        consumer might need into a `TraceBundle`. Only supports a single
        sequence at a time (`batch == 1`) since `TraceBundle`'s tensors are
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
