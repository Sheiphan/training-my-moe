"""Shared helper every Manim scene in `qwenmoe/viz/` builds on: capturing a
real `TraceBundle` from a real trained checkpoint (Requirement 7.1), so no
scene ever animates illustrative/fabricated numbers.
"""
from __future__ import annotations

from pathlib import Path

import torch

from qwenmoe.model.trace import TraceBundle
from qwenmoe.model.transformer import QwenMoEModel
from qwenmoe.tokenizer.base import build_tokenizer_from_meta
from qwenmoe.train.trainer import load_checkpoint


def capture_trace(checkpoint_path: str | Path, prompt: str) -> TraceBundle:
    """Loads `checkpoint_path`, encodes `prompt` with the exact tokenizer
    the checkpoint was trained with, and runs
    `QwenMoEModel.forward_with_trace` under `torch.no_grad()` (Requirement
    7.1) -- the one place every Manim scene gets its real numbers from.
    """
    checkpoint = load_checkpoint(checkpoint_path)
    model = QwenMoEModel(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    tokenizer = build_tokenizer_from_meta(checkpoint["tokenizer_meta"])
    token_ids = tokenizer.encode(prompt)
    if len(token_ids) == 0:
        raise ValueError("prompt must encode to at least one token")

    input_ids = torch.tensor([token_ids], dtype=torch.long)
    with torch.no_grad():
        _, bundle = model.forward_with_trace(input_ids)
    return bundle
