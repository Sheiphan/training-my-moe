#!/usr/bin/env python
"""Estimates whether a QwenMoE config can be trained or run (inference) on
this machine's memory -- or, given no config, recommends a safe target
total-param count for this machine. Standalone hardware-capacity ask (not
a moe-modernization spec task); the formulas here are calibrated from real
`torch.mps` peak-memory measurements on an M4 Pro/24GB Mac -- see
`journey_of_a_model/037-capacity-and-scale-runs.md` for the raw numbers.

Usage:
    uv run python scripts/check_capacity.py                                # recommend a safe target for this machine
    uv run python scripts/check_capacity.py --preset SCALE_300M            # check a specific preset
    uv run python scripts/check_capacity.py --preset SCALE_300M --batch-size 16 --seq-len 256
    uv run python scripts/check_capacity.py --preset SCALE_300M --verify   # actually run 2 real training steps
    uv run python scripts/check_capacity.py --d-model 768 --n-layers 10 --n-heads 12 --n-kv-heads 4 --d-ff 1536

Training memory = params + gradients + AdamW optimizer state (exact,
fp32) + a rough activation-memory heuristic (approximate -- pass
`--verify` for a real measured number instead of trusting the estimate
near a boundary). Inference memory = model weights, plus either the
exact KV-cache-size formula or the same rough per-step activation
heuristic for a from-scratch full-context forward pass (this project's
`generate()` default, `use_kv_cache=False`).
"""
from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from dataclasses import dataclass

import torch
import torch.nn.functional as F

import configs.presets as presets
from qwenmoe.config import ModelConfig, TrainConfig
from qwenmoe.model.transformer import QwenMoEModel
from qwenmoe.train.optim import build_optimizer

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

BYTES_PER_PARAM_TRAIN_FP32 = 16  # 4 (param) + 4 (grad) + 8 (AdamW exp_avg + exp_avg_sq)
BYTES_PER_PARAM_INFERENCE_FP32 = 4

# Rough, deliberately conservative activation-memory constant (bytes per
# token per layer per d_model unit). Calibrated from two real MPS
# measurements on this project's own models at batch=16/seq=256
# (SCALE_100M-shape, d_model=512/n_layers=7: ~2.09GB activation term ->
# ~142 bytes; SCALE_300M-shape, d_model=768/n_layers=10: ~1.91GB -> ~61
# bytes). The two don't imply one clean constant -- MoE routing/expert
# dispatch overhead doesn't scale as simply as a dense transformer's
# would -- so this uses the larger (safer) of the two. Treat this whole
# estimate as an order-of-magnitude guide, not a precise number; `--verify`
# gives the real one.
_ACTIVATION_BYTES_PER_TOKEN_LAYER_DMODEL = 150


def total_system_memory_bytes() -> int | None:
    """Best-effort total physical RAM in bytes -- no new dependency."""
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, AttributeError, OSError):
        pass
    if sys.platform == "darwin":
        try:
            out = subprocess.run(
                ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, check=True
            )
            return int(out.stdout.strip())
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/meminfo", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        return int(line.split()[1]) * 1024
        except (OSError, ValueError):
            pass
    return None


@dataclass
class HardwareInfo:
    device: str  # "cuda" | "mps" | "cpu"
    total_system_memory: int | None
    safe_budget_bytes: int  # the ceiling this script plans against


def detect_hardware() -> HardwareInfo:
    """Picks a device (cuda > mps > cpu -- informational only, this
    project's own `TrainConfig.device` doesn't support cuda yet) and a
    "safe budget": the framework's own reported ceiling where one exists
    (CUDA's currently-free memory, Metal's `recommended_max_memory`),
    else 70% of total system RAM (CPU-only fallback, leaving headroom for
    the OS and everything else running).
    """
    total = total_system_memory_bytes()
    if torch.cuda.is_available():
        free_bytes, _total_bytes = torch.cuda.mem_get_info()
        return HardwareInfo(device="cuda", total_system_memory=total, safe_budget_bytes=free_bytes)
    if torch.backends.mps.is_available():
        return HardwareInfo(
            device="mps", total_system_memory=total, safe_budget_bytes=int(torch.mps.recommended_max_memory())
        )
    safe_budget = int(total * 0.7) if total else 8 * 1024**3
    return HardwareInfo(device="cpu", total_system_memory=total, safe_budget_bytes=safe_budget)


def estimate_activation_bytes(batch_size: int, seq_len: int, n_layers: int, d_model: int, n_heads: int) -> int:
    """Rough heuristic (see module docstring) -- a linear per-token term
    plus the one piece with a real closed form: one attention score
    matrix (`batch * n_heads * seq_len^2`, fp32) per layer."""
    tokens = batch_size * seq_len
    linear_term = tokens * n_layers * d_model * _ACTIVATION_BYTES_PER_TOKEN_LAYER_DMODEL
    attn_term = batch_size * n_heads * seq_len * seq_len * n_layers * 4
    return linear_term + attn_term


def estimate_params(config: ModelConfig) -> tuple[int, int]:
    """Analytic (total, active) parameter count straight from `ModelConfig`
    fields -- mirrors `qwenmoe.train.scaling.count_params`'s real
    parameter-shape math exactly (verified to reproduce SCALE_100M's and
    SCALE_300M's real measured counts to the digit) without constructing
    a real `nn.Module`. This matters: an absurdly large hypothetical
    config would otherwise allocate real memory for every weight tensor
    just to be *counted*, defeating the point of a safety pre-check (this
    OOM-killed the process during development -- see journal 037).
    """
    head_dim = config.d_model // config.n_heads
    expert_params = (
        2 * config.d_model * config.d_ff if config.activation == "gelu" else 3 * config.d_model * config.d_ff
    )

    if config.attention_variant == "mla":
        rope_head_dim = head_dim // 2
        attn_params = (
            config.d_model * config.mla_latent_dim  # w_dkv
            + config.mla_latent_dim * (config.n_heads * head_dim)  # w_uk
            + config.mla_latent_dim * (config.n_heads * head_dim)  # w_uv
            + config.d_model * (config.n_heads * rope_head_dim)  # w_qr
            + config.d_model * rope_head_dim  # w_kr
            + (config.n_heads * head_dim) * config.d_model  # w_o
        )
    else:
        qk_norm_params = 2 * head_dim if config.qk_norm else 0
        attn_params = (
            config.d_model * config.d_model  # w_q
            + config.d_model * (config.n_kv_heads * head_dim)  # w_k
            + config.d_model * (config.n_kv_heads * head_dim)  # w_v
            + config.d_model * config.d_model  # w_o
            + qk_norm_params
        )
    attention_total_per_layer = config.d_model + attn_params  # + attn_norm

    router_params = config.d_model * config.n_routed_experts
    moe_total_per_layer = (
        config.d_model  # moe_norm
        + router_params
        + config.n_shared_experts * expert_params
        + config.n_routed_experts * expert_params
    )
    moe_active_per_layer = (
        config.d_model + router_params + config.n_shared_experts * expert_params + config.top_k * expert_params
    )

    embedding = config.vocab_size * config.d_model + config.d_model  # embedding table + final_norm
    if not config.tie_word_embeddings:
        embedding += config.vocab_size * config.d_model

    total = embedding + config.n_layers * (attention_total_per_layer + moe_total_per_layer)
    active = embedding + config.n_layers * (attention_total_per_layer + moe_active_per_layer)
    return total, active


def kv_cache_bytes(config: ModelConfig, batch_size: int, seq_len: int) -> int:
    """Exact formula, matching `qwenmoe/model/kv_cache.py`'s real cache
    objects: per-token-per-layer cached elements times batch*seq_len*4
    bytes (fp32)."""
    head_dim = config.d_model // config.n_heads
    if config.attention_variant == "mla":
        elements_per_token_per_layer = config.mla_latent_dim + head_dim // 2
    else:
        elements_per_token_per_layer = 2 * config.n_kv_heads * head_dim
    return elements_per_token_per_layer * config.n_layers * batch_size * seq_len * 4


@dataclass
class CapacityReport:
    total_params: int
    active_params: int
    train_static_bytes: int
    train_activation_bytes: int
    inference_bytes: int
    inference_with_kv_cache_bytes: int
    hardware: HardwareInfo

    @property
    def train_total_bytes(self) -> int:
        return self.train_static_bytes + self.train_activation_bytes

    def fits(self, needed_bytes: int) -> bool:
        return needed_bytes <= self.hardware.safe_budget_bytes


def analyze(config: ModelConfig, batch_size: int, seq_len: int, hardware: HardwareInfo | None = None) -> CapacityReport:
    total_params, active_params = estimate_params(config)
    hardware = hardware or detect_hardware()

    static_bytes = total_params * BYTES_PER_PARAM_TRAIN_FP32
    activation_bytes = estimate_activation_bytes(batch_size, seq_len, config.n_layers, config.d_model, config.n_heads)
    # inference (no cache): forward-only, no retained backward graph or
    # optimizer state -- a rough discount off the training activation term.
    inference_activation = estimate_activation_bytes(1, seq_len, config.n_layers, config.d_model, config.n_heads) // 3
    inference_bytes = total_params * BYTES_PER_PARAM_INFERENCE_FP32 + inference_activation
    inference_kv_bytes = total_params * BYTES_PER_PARAM_INFERENCE_FP32 + kv_cache_bytes(
        config, batch_size=1, seq_len=seq_len
    )

    return CapacityReport(
        total_params=total_params,
        active_params=active_params,
        train_static_bytes=static_bytes,
        train_activation_bytes=activation_bytes,
        inference_bytes=inference_bytes,
        inference_with_kv_cache_bytes=inference_kv_bytes,
        hardware=hardware,
    )


def recommend_max_total_params(
    hardware: HardwareInfo,
    batch_size: int,
    seq_len: int,
    d_model_hint: int = 768,
    n_layers_hint: int = 10,
    n_heads_hint: int = 12,
) -> int:
    """Solves `static + activation <= budget` for `total_params`, holding
    the architecture SHAPE fixed at a hint (activation depends on shape,
    not total_params -- MoE routed experts barely move it) and applying a
    70% safety margin on top of the already-conservative activation
    estimate (real headroom fluctuates with whatever else is running --
    see journal 037)."""
    activation = estimate_activation_bytes(batch_size, seq_len, n_layers_hint, d_model_hint, n_heads_hint)
    remaining = int(hardware.safe_budget_bytes * 0.7) - activation
    return max(0, remaining // BYTES_PER_PARAM_TRAIN_FP32)


def verify_empirically(config: ModelConfig, batch_size: int, seq_len: int, device: str) -> int:
    """Constructs the real model and runs 2 real training steps, returning
    REAL measured peak bytes (mirrors the manual investigation in journal
    037) -- authoritative, unlike `analyze()`'s estimate. Only call this
    once `analyze()` already looks safe: an oversized config can hang
    (real swap/compression thrashing on unified-memory devices) rather
    than raise a clean out-of-memory error."""
    model = QwenMoEModel(config).to(device)
    train_config = TrainConfig(batch_size=batch_size, seq_len=seq_len, total_tokens=1, device=device)
    optimizer = build_optimizer(model, train_config)

    def one_step() -> None:
        x = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
        y = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
        optimizer.zero_grad(set_to_none=True)
        logits, extras = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        if extras.get("aux_loss") is not None:
            loss = loss + extras["aux_loss"]
        loss.backward()

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        one_step()
        optimizer.step()
        one_step()  # steady-state peak, optimizer state already allocated
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated()

    if device == "mps":
        torch.mps.empty_cache()
        base = torch.mps.current_allocated_memory()
        one_step()
        optimizer.step()  # warmup: allocates optimizer state
        one_step()
        torch.mps.synchronize()
        peak = torch.mps.current_allocated_memory()  # steady-state peak, pre-2nd-step optimizer.step()
        return peak - base

    import resource

    one_step()
    optimizer.step()
    one_step()
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024  # ru_maxrss unit differs by platform


def _config_from_args(args: argparse.Namespace) -> ModelConfig:
    if args.preset:
        if not hasattr(presets, args.preset):
            raise ValueError(f"no preset named {args.preset!r} in configs/presets.py")
        return getattr(presets, args.preset)
    return ModelConfig(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads or args.n_heads,
        d_ff=args.d_ff,
        n_shared_experts=args.n_shared_experts,
        n_routed_experts=args.n_routed_experts,
        top_k=args.top_k,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preset", help="a ModelConfig preset name from configs/presets.py to check")
    parser.add_argument("--d-model", type=int)
    parser.add_argument("--n-layers", type=int)
    parser.add_argument("--n-heads", type=int)
    parser.add_argument("--n-kv-heads", type=int)
    parser.add_argument("--d-ff", type=int)
    parser.add_argument("--n-shared-experts", type=int, default=1)
    parser.add_argument("--n-routed-experts", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--vocab-size", type=int, default=91, help="placeholder; doesn't change the answer much")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument(
        "--verify", action="store_true", help="also run 2 real training steps to measure real peak memory"
    )
    args = parser.parse_args()

    hardware = detect_hardware()
    logger.info("device: %s", hardware.device)
    if hardware.total_system_memory:
        logger.info("total system memory: %.1f GB", hardware.total_system_memory / 1e9)
    logger.info("safe memory budget for this run: %.1f GB", hardware.safe_budget_bytes / 1e9)

    if not args.preset and args.d_model is None:
        target = recommend_max_total_params(hardware, args.batch_size, args.seq_len)
        logger.info("")
        logger.info("no --preset/--d-model given -- recommending a safe TARGET total-param count")
        logger.info("at batch_size=%d, seq_len=%d, fp32 AdamW: ~%s total params", args.batch_size, args.seq_len, f"{target:,}")
        logger.info(
            "(rough estimate, not a specific architecture -- see SCALE_100M/SCALE_300M in "
            "configs/presets.py for how this project turns a target into a real config, then "
            "pass --preset NAME or --d-model/--n-layers/... to check a real one)"
        )
        return

    config = _config_from_args(args)
    report = analyze(config, args.batch_size, args.seq_len, hardware=hardware)

    logger.info("")
    logger.info(
        "model: total=%s active=%s params (%.2fx sparsity)",
        f"{report.total_params:,}",
        f"{report.active_params:,}",
        report.total_params / report.active_params,
    )
    logger.info("")
    logger.info("-- training (fp32 AdamW, batch=%d seq=%d) --", args.batch_size, args.seq_len)
    logger.info("  static (params+grad+optimizer): %.2f GB", report.train_static_bytes / 1e9)
    logger.info("  activation (rough estimate):    %.2f GB", report.train_activation_bytes / 1e9)
    logger.info("  TOTAL ESTIMATE:                 %.2f GB", report.train_total_bytes / 1e9)
    logger.info(
        "  verdict: %s (safe budget: %.2f GB)",
        "FITS" if report.fits(report.train_total_bytes) else "DOES NOT FIT",
        report.hardware.safe_budget_bytes / 1e9,
    )
    logger.info("")
    logger.info("-- inference / generation --")
    logger.info("  without KV cache (full-context recompute each step): %.2f GB", report.inference_bytes / 1e9)
    logger.info("  with generate(use_kv_cache=True):                    %.2f GB", report.inference_with_kv_cache_bytes / 1e9)

    if args.verify:
        if not report.fits(report.train_total_bytes * 1.2):
            logger.warning(
                "estimate is already at/over budget -- skipping --verify to avoid a real hang "
                "(reduce --batch-size/--seq-len or pick a smaller config first)"
            )
            return
        logger.info("")
        logger.info("--verify: constructing the real model and running 2 real training steps...")
        real_bytes = verify_empirically(config, args.batch_size, args.seq_len, hardware.device)
        logger.info("  REAL measured peak: %.2f GB", real_bytes / 1e9)


if __name__ == "__main__":
    main()
