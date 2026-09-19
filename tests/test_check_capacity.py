import pytest

from qwenmoe.config import ModelConfig
from scripts.check_capacity import (
    CapacityReport,
    HardwareInfo,
    analyze,
    detect_hardware,
    estimate_activation_bytes,
    estimate_params,
    kv_cache_bytes,
    recommend_max_total_params,
    total_system_memory_bytes,
    verify_empirically,
)


def test_estimate_params_matches_scale_100m_exactly():
    """Real measured value (journal 037): 104,679,296 total / 38,619,008
    active. This is the actual `SCALE_100M` preset shape."""
    config = ModelConfig(
        vocab_size=91, d_model=512, n_layers=7, n_heads=8, n_kv_heads=4, d_ff=1024,
        n_shared_experts=1, n_routed_experts=8, top_k=2, qk_norm=True,
    )
    total, active = estimate_params(config)
    assert total == 104_679_296
    assert active == 38_619_008


def test_estimate_params_matches_scale_300m_exactly():
    """Real measured value (journal 037): 334,382,336 total / 122,045,696
    active. This is the actual `SCALE_300M` preset shape."""
    config = ModelConfig(
        vocab_size=91, d_model=768, n_layers=10, n_heads=12, n_kv_heads=4, d_ff=1536,
        n_shared_experts=1, n_routed_experts=8, top_k=2, qk_norm=True,
    )
    total, active = estimate_params(config)
    assert total == 334_382_336
    assert active == 122_045_696


def test_estimate_params_active_less_than_total_when_sparse():
    config = ModelConfig(vocab_size=91, d_model=128, n_layers=2, n_heads=4, n_kv_heads=2, d_ff=256,
                          n_routed_experts=8, top_k=2)
    total, active = estimate_params(config)
    assert 0 < active < total


def test_estimate_params_dense_when_top_k_equals_n_routed_experts():
    """top_k == n_routed_experts means every routed expert always runs --
    total and active should be identical."""
    config = ModelConfig(vocab_size=91, d_model=128, n_layers=2, n_heads=4, n_kv_heads=2, d_ff=256,
                          n_routed_experts=4, top_k=4)
    total, active = estimate_params(config)
    assert total == active


def test_estimate_params_mla_variant_constructs():
    config = ModelConfig(vocab_size=91, d_model=128, n_layers=2, n_heads=4, n_kv_heads=2, d_ff=256,
                          attention_variant="mla", mla_latent_dim=32)
    total, active = estimate_params(config)
    assert total > 0 and active > 0


def test_estimate_activation_bytes_scales_with_batch_seq_and_depth():
    base = estimate_activation_bytes(batch_size=8, seq_len=128, n_layers=4, d_model=256, n_heads=8)
    more_batch = estimate_activation_bytes(batch_size=16, seq_len=128, n_layers=4, d_model=256, n_heads=8)
    more_seq = estimate_activation_bytes(batch_size=8, seq_len=256, n_layers=4, d_model=256, n_heads=8)
    more_layers = estimate_activation_bytes(batch_size=8, seq_len=128, n_layers=8, d_model=256, n_heads=8)
    assert more_batch > base
    assert more_seq > base
    assert more_layers > base


def test_kv_cache_bytes_gqa_matches_hand_formula():
    config = ModelConfig(vocab_size=91, d_model=256, n_layers=6, n_heads=8, n_kv_heads=4, d_ff=512)
    head_dim = 256 // 8
    expected = 2 * config.n_kv_heads * head_dim * config.n_layers * 2 * 128 * 4
    assert kv_cache_bytes(config, batch_size=2, seq_len=128) == expected


def test_kv_cache_bytes_mla_smaller_than_gqa_at_matched_shape():
    """Mirrors task 16's real finding (journal/entry 016): MLA's cache is
    smaller than GQA's at a matched shape."""
    gqa = ModelConfig(vocab_size=91, d_model=256, n_layers=6, n_heads=8, n_kv_heads=4, d_ff=512)
    mla = ModelConfig(vocab_size=91, d_model=256, n_layers=6, n_heads=8, n_kv_heads=4, d_ff=512,
                       attention_variant="mla", mla_latent_dim=32)
    assert kv_cache_bytes(mla, batch_size=2, seq_len=128) < kv_cache_bytes(gqa, batch_size=2, seq_len=128)


def test_total_system_memory_bytes_returns_a_sane_positive_number():
    total = total_system_memory_bytes()
    # None only if every platform-specific method fails; on any dev/CI
    # machine this should resolve, but don't hard-fail portability.
    if total is not None:
        assert total > 1024**3  # every real machine has > 1GB RAM


def test_detect_hardware_returns_a_known_device_and_positive_budget():
    hardware = detect_hardware()
    assert hardware.device in ("cuda", "mps", "cpu")
    assert hardware.safe_budget_bytes > 0


def test_recommend_max_total_params_grows_with_budget():
    # Budgets raised from 10/40GB in journal 044: the activation estimate now
    # carries a measured 2.2x backward factor, so at batch 16 / seq 256 with
    # the 768x10 shape hint the activations alone are ~11.5GB. A 10GB budget
    # legitimately leaves room for zero parameters now, which is the
    # estimator being right rather than the test being wrong.
    small = HardwareInfo(device="cpu", total_system_memory=None, safe_budget_bytes=40 * 1024**3)
    large = HardwareInfo(device="cpu", total_system_memory=None, safe_budget_bytes=80 * 1024**3)
    small_target = recommend_max_total_params(small, batch_size=16, seq_len=256)
    large_target = recommend_max_total_params(large, batch_size=16, seq_len=256)
    assert 0 < small_target < large_target


def test_recommend_max_total_params_zero_when_budget_too_small_for_activations_alone():
    tiny = HardwareInfo(device="cpu", total_system_memory=None, safe_budget_bytes=1)
    assert recommend_max_total_params(tiny, batch_size=16, seq_len=256) == 0


def test_capacity_report_fits_boundary():
    hardware = HardwareInfo(device="cpu", total_system_memory=None, safe_budget_bytes=1000)
    report = CapacityReport(
        total_params=1, active_params=1, train_static_bytes=500, train_activation_bytes=400,
        inference_bytes=1, inference_with_kv_cache_bytes=1, hardware=hardware,
    )
    assert report.train_total_bytes == 900
    assert report.fits(900) is True
    assert report.fits(1001) is False


def test_analyze_reports_does_not_fit_for_a_tiny_injected_budget():
    """Injects a controlled `HardwareInfo` so the verdict is deterministic,
    independent of whatever machine runs the test."""
    config = ModelConfig(vocab_size=91, d_model=512, n_layers=7, n_heads=8, n_kv_heads=4, d_ff=1024)
    tiny_budget = HardwareInfo(device="cpu", total_system_memory=None, safe_budget_bytes=1024)
    report = analyze(config, batch_size=16, seq_len=256, hardware=tiny_budget)
    assert report.fits(report.train_total_bytes) is False


def test_analyze_reports_fits_for_a_huge_injected_budget():
    config = ModelConfig(vocab_size=91, d_model=512, n_layers=7, n_heads=8, n_kv_heads=4, d_ff=1024)
    huge_budget = HardwareInfo(device="cpu", total_system_memory=None, safe_budget_bytes=1024**4)
    report = analyze(config, batch_size=16, seq_len=256, hardware=huge_budget)
    assert report.fits(report.train_total_bytes) is True
    assert report.total_params == 104_678_400  # qk_norm defaults to False here (vs SCALE_100M's 104,679,296)


def test_analyze_never_constructs_a_real_model_for_an_absurd_config():
    """The regression this whole module exists to prevent: `analyze` must
    stay pure arithmetic so an unrealistic hypothetical config can't
    OOM-kill the process just to be evaluated."""
    config = ModelConfig(
        vocab_size=91, d_model=4096, n_layers=40, n_heads=32, n_kv_heads=8, d_ff=8192,
        n_routed_experts=64, top_k=4,
    )
    report = analyze(config, batch_size=16, seq_len=256)
    assert report.total_params > 0
    assert report.fits(report.train_total_bytes) is False


@pytest.mark.parametrize("overrides", [{}, {"n_shared_experts": 0}])
def test_verify_empirically_returns_positive_bytes_on_cpu(overrides):
    """Same pattern as `test_calibrate_throughput_returns_positive_finite_tokens_per_sec`
    (test_scaling.py) -- a tiny config on `device="cpu"` for speed/portability."""
    config = ModelConfig(
        vocab_size=32, d_model=16, n_layers=2, n_heads=2, n_kv_heads=1, d_ff=32,
        n_routed_experts=2, top_k=1, **overrides,
    )
    peak_bytes = verify_empirically(config, batch_size=2, seq_len=8, device="cpu")
    assert peak_bytes > 0


# -- recalibrated against real H100 measurements (journal 044) ---------------

# (batch, seq, measured peak GB on an H100 with SCALE_300M @ vocab 4097,
#  or None where the configuration actually OOMed)
_H100_MEASUREMENTS = [
    (8, 1024, 33.26),
    (16, 1024, 64.88),
    (24, 1024, None),
    (32, 512, 58.27),
    (48, 512, None),
    (64, 256, 51.09),
]
_H100_FREE_BYTES = 84.47e9


def _scale_300m_estimate_gb(batch_size: int, seq_len: int) -> float:
    import dataclasses

    import configs.presets as presets

    config = dataclasses.replace(presets.SCALE_300M, vocab_size=4097)
    activation = estimate_activation_bytes(
        batch_size, seq_len, config.n_layers, config.d_model, config.n_heads, config.vocab_size
    )
    static = 337_458_944 * 16  # fp32 weights + grads + two AdamW moments
    return (activation + static) / 1e9


@pytest.mark.parametrize("batch_size,seq_len,measured", _H100_MEASUREMENTS)
def test_estimate_is_within_tolerance_of_real_h100_peaks(batch_size, seq_len, measured):
    """Before journal 044 this estimator was ~50% under on CUDA and returned
    FITS for two configurations that then OOMed."""
    if measured is None:
        pytest.skip("this configuration OOMed; covered by the verdict test")
    estimate = _scale_300m_estimate_gb(batch_size, seq_len)
    error = abs(estimate - measured) / measured
    assert error < 0.15, f"estimate {estimate:.2f}GB vs measured {measured:.2f}GB ({error:.1%})"


@pytest.mark.parametrize("batch_size,seq_len,measured", _H100_MEASUREMENTS)
def test_verdict_matches_what_the_h100_actually_did(batch_size, seq_len, measured):
    """A FITS verdict that OOMs is worse than no estimator at all, because
    journal 037 established this failure mode is a hang, not a clean error."""
    from scripts.check_capacity import _CUDA_USABLE_FRACTION

    budget_gb = _H100_FREE_BYTES * _CUDA_USABLE_FRACTION / 1e9
    fits = _scale_300m_estimate_gb(batch_size, seq_len) < budget_gb
    assert fits is (measured is not None)


def test_estimate_accounts_for_vocabulary():
    """Logits are tokens x vocab and exist several times over; omitting them
    was invisible at vocab 4097 and several GB at 16384."""
    args = (16, 1024, 10, 768, 12)
    small = estimate_activation_bytes(*args, 4097)
    large = estimate_activation_bytes(*args, 16384)
    assert large > small
    # the delta is the extra logit copies, within rounding
    expected = 16 * 1024 * (16384 - 4097) * 4 * 3
    assert abs((large - small) - expected) < 1e6


def test_estimate_without_vocab_still_works():
    """The pre-044 signature has callers; it must not break."""
    assert estimate_activation_bytes(8, 128, 4, 256, 8) > 0
