#!/usr/bin/env python
"""Loads a trained checkpoint and generates text from a prompt (Requirement 6.1).

Usage:
    uv run python scripts/generate.py --checkpoint runs/smoke_test/checkpoints/step_1000.pt \\
        --prompt "ROMEO:" --max-new-tokens 300 --temperature 0.8 --top-k 20
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from qwenmoe.model.generate import generate
from qwenmoe.model.transformer import QwenMoEModel
from qwenmoe.tokenizer.base import build_tokenizer_from_meta
from qwenmoe.train.trainer import load_checkpoint, select_device

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    checkpoint = load_checkpoint(args.checkpoint)
    model = QwenMoEModel(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    tokenizer = build_tokenizer_from_meta(checkpoint["tokenizer_meta"])
    device = select_device(args.device)

    logger.info("checkpoint: %s (trained to step %d)", args.checkpoint, checkpoint["step"])
    logger.info("device: %s, temperature: %.2f, top_k: %s, top_p: %s",
                device, args.temperature, args.top_k, args.top_p)

    text = generate(
        model,
        tokenizer,
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        device=device,
    )
    print(text)


if __name__ == "__main__":
    main()
