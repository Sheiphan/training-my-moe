import json
import math

import pytest
import torch
import torch.nn.functional as F

from qwenmoe.config import ModelConfig, TrainConfig
from qwenmoe.data.dataset import TokenDataset, train_val_split
from qwenmoe.model.transformer import QwenMoEModel
from qwenmoe.train.optim import build_optimizer
from qwenmoe.train.trainer import (
    Trainer,
    _evaluate,
    _forward_backward_step,
    _set_lr,
    lr_multiplier,
    select_device,
)


def _tiny_model_config(**overrides):
    defaults = dict(
        vocab_size=12,
        d_model=16,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        d_ff=32,
        n_routed_experts=4,
        n_shared_experts=1,
        top_k=2,
        max_seq_len=32,
    )
    defaults.update(overrides)
    return ModelConfig(**defaults)


def _repeating_datasets(seq_len: int = 16, n_repeats: int = 300):
    """A strictly periodic (period-10) token stream -- trivial for a
    trained model to predict, but random init still starts far from it."""
    tokens = list(range(10)) * n_repeats
    train_tokens, val_tokens = train_val_split(tokens, val_fraction=0.1)
    return TokenDataset(train_tokens, seq_len), TokenDataset(val_tokens, seq_len)


def _read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


# -- select_device (Requirement 3.2) -----------------------------------------


def test_select_device_explicit_cpu_always_returns_cpu(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert select_device("cpu") == "cpu"


def test_select_device_prefers_mps_when_available(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert select_device("auto") == "mps"
    assert select_device("mps") == "mps"


def test_select_device_falls_back_to_cpu_when_mps_unavailable(monkeypatch, caplog):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert select_device("auto") == "cpu"
    with caplog.at_level("WARNING"):
        assert select_device("mps") == "cpu"
    assert "mps" in caplog.text.lower()


# -- lr_multiplier (Requirement 5.1: warmup + cosine decay) ------------------


def test_lr_multiplier_ramps_linearly_during_warmup():
    assert lr_multiplier(0, warmup_steps=10, total_steps=100) == pytest.approx(0.1)
    assert lr_multiplier(4, warmup_steps=10, total_steps=100) == pytest.approx(0.5)
    assert lr_multiplier(9, warmup_steps=10, total_steps=100) == pytest.approx(1.0)


def test_lr_multiplier_cosine_decays_after_warmup():
    assert lr_multiplier(10, warmup_steps=10, total_steps=100) == pytest.approx(1.0)
    assert lr_multiplier(99, warmup_steps=10, total_steps=100) < 0.01
    mid = lr_multiplier(55, warmup_steps=10, total_steps=100)
    assert 0.0 < mid < 1.0


def test_lr_multiplier_no_warmup_starts_at_full_decay_curve():
    assert lr_multiplier(0, warmup_steps=0, total_steps=100) == pytest.approx(1.0)


# -- lr_multiplier: min_lr_ratio floor + WSD schedule (moe-modernization Req 2) --


def test_lr_multiplier_cosine_decays_to_min_lr_ratio_not_zero():
    """Requirement 2.1: with min_lr_ratio > 0, cosine's floor is that ratio
    of peak, not exactly 0 -- a pure-win default change from the original
    decay-to-zero behavior (still reproduced exactly by min_lr_ratio=0.0,
    the function's default, per the tests above)."""
    end = lr_multiplier(99, warmup_steps=10, total_steps=100, min_lr_ratio=0.1)
    assert end == pytest.approx(0.1, abs=1e-3)
    assert lr_multiplier(10, warmup_steps=10, total_steps=100, min_lr_ratio=0.1) == pytest.approx(1.0)


def test_lr_multiplier_wsd_holds_peak_through_the_stable_phase():
    """Requirement 2.2: WSD stays flat at 1.0 for everything before the
    final decay_fraction of total_steps -- unlike cosine, which starts
    decaying immediately after warmup."""
    mult = lr_multiplier(
        50, warmup_steps=10, total_steps=100, schedule="wsd", decay_fraction=0.2
    )
    assert mult == pytest.approx(1.0)  # decay only starts at step 80 (100 * (1-0.2))


def test_lr_multiplier_wsd_decays_linearly_to_min_lr_ratio_at_the_end():
    # decay_fraction=0.2 of total_steps=100 -> the decay window is steps [80, 100).
    boundary = lr_multiplier(
        80, warmup_steps=10, total_steps=100, schedule="wsd", decay_fraction=0.2, min_lr_ratio=0.1
    )
    assert boundary == pytest.approx(1.0)  # step 80 is the first decay step, still at peak

    midpoint = lr_multiplier(
        90, warmup_steps=10, total_steps=100, schedule="wsd", decay_fraction=0.2, min_lr_ratio=0.1
    )
    assert midpoint == pytest.approx(0.55, abs=1e-9)  # halfway through a linear 1.0 -> 0.1 decay

    at_total_steps = lr_multiplier(
        100, warmup_steps=10, total_steps=100, schedule="wsd", decay_fraction=0.2, min_lr_ratio=0.1
    )
    assert at_total_steps == pytest.approx(0.1)  # progress clamped to 1.0 -> exactly the floor


def test_lr_multiplier_unknown_schedule_raises():
    with pytest.raises(ValueError):
        lr_multiplier(50, warmup_steps=10, total_steps=100, schedule="not-a-schedule")


# -- _evaluate caps how much of the validation split it touches -------------


def test_evaluate_caps_at_max_eval_batches():
    """Requirement 5.3: validation shouldn't require a full pass over a
    potentially huge val split every eval interval."""
    config = _tiny_model_config()
    model = QwenMoEModel(config)
    seen_batches = []

    def batches():
        for i in range(100):
            seen_batches.append(i)
            x = torch.randint(0, config.vocab_size, (2, 4))
            y = torch.randint(0, config.vocab_size, (2, 4))
            yield x, y

    _evaluate(model, batches(), device="cpu", max_eval_batches=3)

    assert len(seen_batches) == 3


# -- Trainer.fit input validation ---------------------------------------------


def test_trainer_rejects_dataset_smaller_than_batch_size(tmp_path):
    config = _tiny_model_config()
    model = QwenMoEModel(config)
    train_ds, val_ds = _repeating_datasets()
    train_config = TrainConfig(batch_size=10_000, seq_len=16, total_tokens=1000, device="cpu")

    trainer = Trainer(tmp_path)
    with pytest.raises(ValueError):
        trainer.fit(model, train_ds, val_ds, train_config)


# -- Trainer.fit end to end (Requirements 5.1-5.4, 8.1-8.2) ------------------


def test_trainer_runs_and_loss_decreases(tmp_path):
    """Task 10.1's required integration test: ~50 steps on a tiny config,
    loss finite and lower at the end than at the start."""
    torch.manual_seed(0)
    config = _tiny_model_config()
    model = QwenMoEModel(config)
    train_ds, val_ds = _repeating_datasets()
    train_config = TrainConfig(
        batch_size=8,
        seq_len=16,
        total_tokens=50 * 8 * 16,
        lr=1e-2,
        warmup_steps=5,
        checkpoint_interval_steps=50,
        eval_interval_steps=25,
        log_interval_steps=1,
        device="cpu",
        seed=0,
    )

    trainer = Trainer(tmp_path)
    result = trainer.fit(model, train_ds, val_ds, train_config)

    assert result.final_step == 50
    assert math.isfinite(result.final_train_loss)

    steps = [r for r in _read_jsonl(result.log_path) if r["type"] == "step"]
    assert len(steps) == 50
    early_mean = sum(r["train_loss"] for r in steps[:5]) / 5
    late_mean = sum(r["train_loss"] for r in steps[-5:]) / 5
    assert math.isfinite(early_mean) and math.isfinite(late_mean)
    assert late_mean < early_mean


def test_trainer_checkpoint_and_val_metrics(tmp_path):
    torch.manual_seed(0)
    config = _tiny_model_config()
    model = QwenMoEModel(config)
    train_ds, val_ds = _repeating_datasets()
    train_config = TrainConfig(
        batch_size=8,
        seq_len=16,
        total_tokens=20 * 8 * 16,
        checkpoint_interval_steps=20,
        eval_interval_steps=10,
        log_interval_steps=5,
        device="cpu",
    )
    tokenizer_meta = {"tokenizer_type": "char", "vocab_size": config.vocab_size}

    trainer = Trainer(tmp_path, tokenizer_meta=tokenizer_meta)
    result = trainer.fit(model, train_ds, val_ds, train_config)

    assert result.checkpoint_path is not None
    assert result.checkpoint_path.exists()
    assert math.isfinite(result.final_val_loss)
    assert math.isfinite(result.val_perplexity)
    assert result.val_perplexity == pytest.approx(math.exp(result.final_val_loss))

    checkpoint = torch.load(result.checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint["step"] == 20
    assert checkpoint["model_config"].vocab_size == config.vocab_size
    assert checkpoint["tokenizer_meta"] == tokenizer_meta
    assert "model_state_dict" in checkpoint
    assert "optimizer_state_dict" in checkpoint


def test_trainer_logs_per_expert_token_counts(tmp_path):
    """Requirement 8.2: per-expert token counts, consumed later by the
    expert-utilization scene (task 14)."""
    config = _tiny_model_config()
    model = QwenMoEModel(config)
    train_ds, val_ds = _repeating_datasets()
    train_config = TrainConfig(
        batch_size=8,
        seq_len=16,
        total_tokens=6 * 8 * 16,
        checkpoint_interval_steps=6,
        eval_interval_steps=6,
        log_interval_steps=1,
        device="cpu",
    )

    trainer = Trainer(tmp_path)
    result = trainer.fit(model, train_ds, val_ds, train_config)

    steps = [r for r in _read_jsonl(result.log_path) if r["type"] == "step"]
    counts = steps[0]["expert_token_counts"]
    assert len(counts) == config.n_layers
    for layer_counts in counts:
        assert len(layer_counts) == config.n_routed_experts
        assert sum(layer_counts) == config.top_k * train_config.batch_size * train_config.seq_len


def test_trainer_logs_grad_norm(tmp_path):
    """moe-modernization Requirement 1.3: the pre-clip global gradient norm
    is recorded in every step log record."""
    config = _tiny_model_config()
    model = QwenMoEModel(config)
    train_ds, val_ds = _repeating_datasets()
    train_config = TrainConfig(
        batch_size=8,
        seq_len=16,
        total_tokens=6 * 8 * 16,
        checkpoint_interval_steps=6,
        eval_interval_steps=6,
        log_interval_steps=1,
        device="cpu",
    )

    trainer = Trainer(tmp_path)
    result = trainer.fit(model, train_ds, val_ds, train_config)

    steps = [r for r in _read_jsonl(result.log_path) if r["type"] == "step"]
    assert len(steps) == 6
    for record in steps:
        assert "grad_norm" in record
        assert math.isfinite(record["grad_norm"])
        assert record["grad_norm"] >= 0.0


# -- _set_lr applies one multiplier to two different peak rates (Req 1.6) ---


def test_set_lr_scales_muon_and_adamw_groups_by_their_own_base_lr():
    config = _tiny_model_config()
    model = QwenMoEModel(config)
    train_config = TrainConfig(
        batch_size=4,
        seq_len=8,
        total_tokens=1000,
        optimizer="muon",
        lr=1e-2,
        muon_lr=4e-2,
        warmup_steps=0,
        min_lr_ratio=0.0,
    )
    optimizer = build_optimizer(model, train_config)

    # step == warmup_steps == 0: lr_multiplier is past warmup, at the very
    # start of the decay curve (mult == 1.0), so each group's lr should
    # equal its own base_lr exactly.
    _set_lr(optimizer, train_config.lr, step=0, total_steps=100, config=train_config)

    muon_group = next(g for g in optimizer.param_groups if "momentum" in g)
    adamw_groups = [g for g in optimizer.param_groups if "momentum" not in g]
    assert muon_group["lr"] == pytest.approx(4e-2)
    for g in adamw_groups:
        assert g["lr"] == pytest.approx(1e-2)


# -- Muon end to end (moe-modernization Requirement 1.4-1.6) ----------------


def test_trainer_runs_and_loss_decreases_with_muon_optimizer(tmp_path):
    """Mirrors test_trainer_runs_and_loss_decreases but with
    optimizer='muon' -- Requirement 1.6's "a Muon run ... SHALL train"."""
    torch.manual_seed(0)
    config = _tiny_model_config()
    model = QwenMoEModel(config)
    train_ds, val_ds = _repeating_datasets()
    train_config = TrainConfig(
        batch_size=8,
        seq_len=16,
        total_tokens=50 * 8 * 16,
        optimizer="muon",
        lr=1e-2,
        muon_lr=2e-2,
        warmup_steps=5,
        checkpoint_interval_steps=50,
        eval_interval_steps=25,
        log_interval_steps=1,
        device="cpu",
        seed=0,
    )

    trainer = Trainer(tmp_path)
    result = trainer.fit(model, train_ds, val_ds, train_config)

    assert result.final_step == 50
    assert math.isfinite(result.final_train_loss)

    steps = [r for r in _read_jsonl(result.log_path) if r["type"] == "step"]
    early_mean = sum(r["train_loss"] for r in steps[:5]) / 5
    late_mean = sum(r["train_loss"] for r in steps[-5:]) / 5
    assert math.isfinite(early_mean) and math.isfinite(late_mean)
    assert late_mean < early_mean


def test_trainer_checkpoint_round_trips_for_muon_optimizer(tmp_path):
    """Requirement 1.6: "the checkpoint SHALL round-trip through
    save/resume for both optimizers" -- a resumed Muon run continues from
    the saved step with its CombinedOptimizer state (Muon momentum buffers
    + AdamW moment estimates) restored, not silently reset."""
    config = _tiny_model_config()

    torch.manual_seed(0)
    model_a = QwenMoEModel(config)
    train_ds, val_ds = _repeating_datasets()

    trainer = Trainer(tmp_path)
    config_a = TrainConfig(
        batch_size=8,
        seq_len=16,
        total_tokens=20 * 8 * 16,
        optimizer="muon",
        checkpoint_interval_steps=20,
        eval_interval_steps=20,
        log_interval_steps=1,
        device="cpu",
    )
    result_a = trainer.fit(model_a, train_ds, val_ds, config_a)
    assert result_a.final_step == 20
    assert result_a.checkpoint_path is not None

    checkpoint = torch.load(result_a.checkpoint_path, map_location="cpu", weights_only=False)
    assert "sub_optimizers" in checkpoint["optimizer_state_dict"]
    assert len(checkpoint["optimizer_state_dict"]["sub_optimizers"]) == 2

    torch.manual_seed(999)
    model_b = QwenMoEModel(config)
    config_b = TrainConfig(
        batch_size=8,
        seq_len=16,
        total_tokens=30 * 8 * 16,
        optimizer="muon",
        checkpoint_interval_steps=10,
        eval_interval_steps=10,
        log_interval_steps=1,
        device="cpu",
    )
    result_b = trainer.fit(model_b, train_ds, val_ds, config_b, resume_from=result_a.checkpoint_path)

    assert result_b.final_step == 30
    steps = [r for r in _read_jsonl(result_b.log_path) if r["type"] == "step"]
    assert len(steps) == 30  # not 20 + 30 -- resume picked up from step 20, not step 0


# -- Mixed precision + output z-loss (moe-modernization Requirement 3) ------


def _step_config(**overrides):
    defaults = dict(batch_size=2, seq_len=8, total_tokens=100, device="cpu")
    defaults.update(overrides)
    return TrainConfig(**defaults)


def test_forward_backward_step_z_loss_zero_is_plain_cross_entropy():
    """Requirement 3.3 invariant: with `z_loss_weight == 0` (and fp32) the
    reported lm-loss is exactly `F.cross_entropy` on the same logits -- no
    z-loss term is folded in, and `z_loss` comes back `None`."""
    torch.manual_seed(0)
    config = _tiny_model_config()
    model = QwenMoEModel(config)
    optimizer = build_optimizer(model, _step_config())
    x = torch.randint(0, config.vocab_size, (2, 8))
    y = torch.randint(0, config.vocab_size, (2, 8))

    model.train()
    with torch.no_grad():
        logits, _ = model(x)
        ref_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), y.reshape(-1)
        ).item()

    lm_loss, _aux, z_loss, grad_norm = _forward_backward_step(
        model, optimizer, x, y, grad_clip_norm=1.0, device="cpu",
        precision="fp32", z_loss_weight=0.0,
    )

    assert z_loss is None
    assert lm_loss == pytest.approx(ref_loss, rel=1e-6)
    assert math.isfinite(grad_norm) and grad_norm >= 0.0


def test_forward_backward_step_z_loss_positive_is_returned_and_nonnegative():
    """Requirement 3.3: with a positive weight the z-loss term is a finite,
    non-negative number (it is `weight * mean(logsumexp(logits)^2)`, a mean
    of squares scaled by a positive weight)."""
    torch.manual_seed(0)
    config = _tiny_model_config()
    model = QwenMoEModel(config)
    optimizer = build_optimizer(model, _step_config())
    x = torch.randint(0, config.vocab_size, (2, 8))
    y = torch.randint(0, config.vocab_size, (2, 8))

    _lm, _aux, z_loss, _gn = _forward_backward_step(
        model, optimizer, x, y, grad_clip_norm=1.0, device="cpu",
        precision="fp32", z_loss_weight=1e-3,
    )

    assert z_loss is not None
    assert math.isfinite(z_loss) and z_loss >= 0.0


def test_forward_backward_step_bf16_autocast_stays_finite():
    """Requirement 3.1/3.4: the bf16-autocast forward still produces a
    finite loss and gradient norm on cpu (bf16 CPU autocast is always
    available in torch; no gradient scaler is used)."""
    torch.manual_seed(0)
    config = _tiny_model_config()
    model = QwenMoEModel(config)
    optimizer = build_optimizer(model, _step_config())
    x = torch.randint(0, config.vocab_size, (2, 8))
    y = torch.randint(0, config.vocab_size, (2, 8))

    lm_loss, _aux, _z, grad_norm = _forward_backward_step(
        model, optimizer, x, y, grad_clip_norm=1.0, device="cpu",
        precision="bf16", z_loss_weight=0.0,
    )

    assert math.isfinite(lm_loss)
    assert math.isfinite(grad_norm) and grad_norm >= 0.0


def test_trainer_bf16_precision_runs_and_loss_decreases(tmp_path):
    """Requirement 3.4: a full `precision="bf16"` run over ~50 steps stays
    finite and its loss is lower at the end than at the start."""
    torch.manual_seed(0)
    config = _tiny_model_config()
    model = QwenMoEModel(config)
    train_ds, val_ds = _repeating_datasets()
    train_config = TrainConfig(
        batch_size=8,
        seq_len=16,
        total_tokens=50 * 8 * 16,
        precision="bf16",
        lr=1e-2,
        warmup_steps=5,
        checkpoint_interval_steps=50,
        eval_interval_steps=25,
        log_interval_steps=1,
        device="cpu",
        seed=0,
    )

    result = Trainer(tmp_path).fit(model, train_ds, val_ds, train_config)

    assert result.final_step == 50
    assert math.isfinite(result.final_train_loss)

    steps = [r for r in _read_jsonl(result.log_path) if r["type"] == "step"]
    early_mean = sum(r["train_loss"] for r in steps[:5]) / 5
    late_mean = sum(r["train_loss"] for r in steps[-5:]) / 5
    assert math.isfinite(early_mean) and math.isfinite(late_mean)
    assert late_mean < early_mean


def test_trainer_logs_z_loss_field(tmp_path):
    """Requirement 3.3: the z-loss appears as its own step-log field. It is
    `None` when off and a finite non-negative number when on."""
    config = _tiny_model_config()
    train_ds, val_ds = _repeating_datasets()
    common = dict(
        batch_size=8,
        seq_len=16,
        total_tokens=4 * 8 * 16,
        checkpoint_interval_steps=4,
        eval_interval_steps=4,
        log_interval_steps=1,
        device="cpu",
    )

    off = Trainer(tmp_path / "off").fit(
        QwenMoEModel(config), train_ds, val_ds, TrainConfig(**common)
    )
    for record in (r for r in _read_jsonl(off.log_path) if r["type"] == "step"):
        assert "z_loss" in record and record["z_loss"] is None

    on = Trainer(tmp_path / "on").fit(
        QwenMoEModel(config), train_ds, val_ds, TrainConfig(z_loss_weight=1e-3, **common)
    )
    on_steps = [r for r in _read_jsonl(on.log_path) if r["type"] == "step"]
    assert on_steps and all(
        r["z_loss"] is not None and math.isfinite(r["z_loss"]) and r["z_loss"] >= 0.0
        for r in on_steps
    )


def test_trainer_startup_logs_effective_precision(tmp_path, caplog):
    """Requirement 3.2 / design.md Error Handling: a startup log line states
    the effective precision."""
    config = _tiny_model_config()
    train_ds, val_ds = _repeating_datasets()
    train_config = TrainConfig(
        batch_size=8,
        seq_len=16,
        total_tokens=2 * 8 * 16,
        precision="bf16",
        checkpoint_interval_steps=2,
        eval_interval_steps=2,
        log_interval_steps=1,
        device="cpu",
    )

    with caplog.at_level("INFO", logger="qwenmoe.train.trainer"):
        Trainer(tmp_path).fit(QwenMoEModel(config), train_ds, val_ds, train_config)

    assert "precision" in caplog.text.lower()
    assert "bf16" in caplog.text.lower()


def test_trainer_resume_continues_from_saved_step(tmp_path):
    """Requirement 5.4: resuming continues from the checkpointed step
    rather than silently restarting from 0."""
    config = _tiny_model_config()

    torch.manual_seed(0)
    model_a = QwenMoEModel(config)
    train_ds, val_ds = _repeating_datasets()

    trainer = Trainer(tmp_path)
    config_a = TrainConfig(
        batch_size=8,
        seq_len=16,
        total_tokens=50 * 8 * 16,
        checkpoint_interval_steps=50,
        eval_interval_steps=50,
        log_interval_steps=1,
        device="cpu",
    )
    result_a = trainer.fit(model_a, train_ds, val_ds, config_a)
    assert result_a.final_step == 50
    assert result_a.checkpoint_path is not None

    torch.manual_seed(999)
    model_b = QwenMoEModel(config)  # fresh, differently-initialized weights
    config_b = TrainConfig(
        batch_size=8,
        seq_len=16,
        total_tokens=70 * 8 * 16,
        checkpoint_interval_steps=20,
        eval_interval_steps=20,
        log_interval_steps=1,
        device="cpu",
    )
    result_b = trainer.fit(model_b, train_ds, val_ds, config_b, resume_from=result_a.checkpoint_path)

    assert result_b.final_step == 70

    steps = [r for r in _read_jsonl(result_b.log_path) if r["type"] == "step"]
    # If resume had silently restarted from step 0, this would be 50 + 70 = 120.
    assert len(steps) == 70
