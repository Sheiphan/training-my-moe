"""Dataclass configs for model, training, and data.

Single source of truth for every tunable knob (Requirement 3.1, 12.1-12.5) —
no hard-coded literals scattered through the model/training code, and no
YAML/JSON schema layer: presets are plain Python (see configs/presets.py).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass
class ModelConfig:
    vocab_size: int
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    n_kv_heads: int = 4                # GQA; == n_heads means plain MHA
    d_ff: int = 512                    # per-expert SwiGLU intermediate size
    n_shared_experts: int = 1
    n_routed_experts: int = 8
    top_k: int = 2
    max_seq_len: int = 256
    positional_scheme: Literal["rope", "nope", "alibi"] = "rope"
    activation: Literal["swiglu", "geglu", "gelu"] = "swiglu"
    moe_balancing: Literal["aux_loss", "loss_free"] = "aux_loss"
    attention_variant: Literal["gqa", "mla"] = "gqa"
    mla_latent_dim: int | None = None  # required if attention_variant == "mla"
    lora_rank: int | None = None       # set only for LoRA fine-tuning runs
    tie_word_embeddings: bool = True
    rope_theta: float = 10000.0

    def __post_init__(self) -> None:
        if self.vocab_size < 1:
            raise ValueError(f"vocab_size must be >= 1, got {self.vocab_size}")
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        if self.n_kv_heads < 1 or self.n_kv_heads > self.n_heads:
            raise ValueError(
                f"n_kv_heads ({self.n_kv_heads}) must be between 1 and n_heads ({self.n_heads})"
            )
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"
            )
        if self.n_routed_experts < 1:
            raise ValueError(f"n_routed_experts must be >= 1, got {self.n_routed_experts}")
        if self.n_shared_experts < 0:
            raise ValueError(f"n_shared_experts must be >= 0, got {self.n_shared_experts}")
        if self.top_k < 1 or self.top_k > self.n_routed_experts:
            raise ValueError(
                f"top_k ({self.top_k}) must be between 1 and n_routed_experts "
                f"({self.n_routed_experts})"
            )
        if self.attention_variant == "mla" and self.mla_latent_dim is None:
            raise ValueError("mla_latent_dim is required when attention_variant == 'mla'")
        if self.mla_latent_dim is not None and self.mla_latent_dim < 1:
            raise ValueError(f"mla_latent_dim must be >= 1, got {self.mla_latent_dim}")
        if self.lora_rank is not None and self.lora_rank < 1:
            raise ValueError(f"lora_rank must be >= 1, got {self.lora_rank}")
        if self.max_seq_len < 1:
            raise ValueError(f"max_seq_len must be >= 1, got {self.max_seq_len}")


@dataclass
class TrainConfig:
    batch_size: int
    seq_len: int
    total_tokens: int          # scaling-law-derived budget (Requirement 11.2)
    lr: float = 3e-4
    warmup_steps: int = 200
    checkpoint_interval_steps: int = 500
    eval_interval_steps: int = 100
    max_eval_batches: int = 50          # cap so validation stays fast on a large val split
    log_interval_steps: int = 10       # console/JSON-lines logging cadence (Requirement 5.2)
    device: Literal["auto", "mps", "cpu"] = "auto"
    seed: int = 1337

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.seq_len < 1:
            raise ValueError(f"seq_len must be >= 1, got {self.seq_len}")
        if self.total_tokens < 1:
            raise ValueError(f"total_tokens must be >= 1, got {self.total_tokens}")
        if self.lr <= 0:
            raise ValueError(f"lr must be > 0, got {self.lr}")
        if self.warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {self.warmup_steps}")
        if self.checkpoint_interval_steps < 1:
            raise ValueError(
                f"checkpoint_interval_steps must be >= 1, got {self.checkpoint_interval_steps}"
            )
        if self.eval_interval_steps < 1:
            raise ValueError(f"eval_interval_steps must be >= 1, got {self.eval_interval_steps}")
        if self.max_eval_batches < 1:
            raise ValueError(f"max_eval_batches must be >= 1, got {self.max_eval_batches}")
        if self.log_interval_steps < 1:
            raise ValueError(f"log_interval_steps must be >= 1, got {self.log_interval_steps}")


@dataclass
class DataConfig:
    corpus: Literal["tiny_shakespeare", "tiny_stories"]
    tokenizer_type: Literal["char", "bpe"]
    bpe_vocab_size: int = 4096
    cache_dir: Path = Path(".cache/data")
    fixed_slice_tokens: int | None = None   # ablation mode: same slice for every variant
    fixed_slice_seed: int = 0

    def __post_init__(self) -> None:
        if self.bpe_vocab_size < 1:
            raise ValueError(f"bpe_vocab_size must be >= 1, got {self.bpe_vocab_size}")
        if self.fixed_slice_tokens is not None and self.fixed_slice_tokens < 1:
            raise ValueError(
                f"fixed_slice_tokens must be >= 1 if set, got {self.fixed_slice_tokens}"
            )
        if isinstance(self.cache_dir, str):
            self.cache_dir = Path(self.cache_dir)
