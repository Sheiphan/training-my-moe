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
    small = HardwareInfo(device="cpu", total_system_memory=None, safe_budget_bytes=10 * 1024**3)
    large = HardwareInfo(device="cpu", total_system_memory=None, safe_budget_bytes=40 * 1024**3)
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
