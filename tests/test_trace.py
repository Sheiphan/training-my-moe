from pathlib import Path

import pytest
import torch

from qwenmoe.config import ModelConfig, TrainConfig
from qwenmoe.model.transformer import QwenMoEModel
from qwenmoe.tokenizer.char import CharTokenizer
from qwenmoe.train.trainer import save_checkpoint
from qwenmoe.viz.common import capture_trace


def _tiny_setup(positional_scheme: str = "rope", **overrides):
    tokenizer = CharTokenizer()
    tokenizer.train("abcdefghijklmnopqrstuvwxyz ")
    defaults = dict(
        vocab_size=tokenizer.vocab_size,
        d_model=16,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        d_ff=32,
        n_routed_experts=4,
        n_shared_experts=1,
        top_k=2,
        max_seq_len=32,
        positional_scheme=positional_scheme,
    )
    defaults.update(overrides)
    model = QwenMoEModel(ModelConfig(**defaults))
    model.eval()
    return model, tokenizer


def _encode(tokenizer: CharTokenizer, text: str) -> torch.Tensor:
    return torch.tensor([tokenizer.encode(text)], dtype=torch.long)


# -- forward_with_trace: every TraceBundle/LayerTrace field, expected shape --


@pytest.mark.parametrize("positional_scheme", ["rope", "nope", "alibi"])
def test_forward_with_trace_populates_every_field_with_expected_shape(positional_scheme):
    """Task 12: every TraceBundle/LayerTrace field populated with the
    expected shape, for each positional scheme (they disagree on what
    `rope_angles_or_alibi_bias` should look like)."""
    model, tokenizer = _tiny_setup(positional_scheme=positional_scheme)
    input_ids = _encode(tokenizer, "hello world")
    seq_len = input_ids.shape[1]
    config = model.config
    head_dim = config.d_model // config.n_heads

    with torch.no_grad():
        logits, bundle = model.forward_with_trace(input_ids)

    assert logits.shape == (1, seq_len, config.vocab_size)
    assert bundle.token_ids == input_ids[0].tolist()
    assert bundle.embeddings.shape == (seq_len, config.d_model)
    assert bundle.final_norm_output.shape == (seq_len, config.d_model)
    assert bundle.logits.shape == (seq_len, config.vocab_size)
    assert bundle.probs.shape == (config.vocab_size,)
    assert torch.isfinite(bundle.probs).all()
    assert bundle.probs.sum().item() == pytest.approx(1.0, abs=1e-4)
    assert len(bundle.layers) == config.n_layers

    for layer in bundle.layers:
        if positional_scheme == "nope":
            assert layer.rope_angles_or_alibi_bias is None
        elif positional_scheme == "rope":
            assert layer.rope_angles_or_alibi_bias.shape == (seq_len, head_dim)
        else:  # alibi
            assert layer.rope_angles_or_alibi_bias.shape == (config.n_heads, seq_len, seq_len)

        assert layer.q.shape == (config.n_heads, seq_len, head_dim)
        assert layer.k.shape == (config.n_kv_heads, seq_len, head_dim)
        assert layer.v.shape == (config.n_kv_heads, seq_len, head_dim)
        assert layer.attn_scores_pre_mask.shape == (config.n_heads, seq_len, seq_len)
        assert layer.attn_scores_post_softmax.shape == (config.n_heads, seq_len, seq_len)
        assert layer.attn_output.shape == (seq_len, config.d_model)
        assert layer.router_logits.shape == (seq_len, config.n_routed_experts)
        assert layer.router_probs.shape == (seq_len, config.n_routed_experts)
        assert layer.topk_indices.shape == (seq_len, config.top_k)
        assert layer.topk_weights.shape == (seq_len, config.top_k)
        assert layer.moe_output.shape == (seq_len, config.d_model)

        assert torch.isfinite(layer.attn_scores_post_softmax).all()
        assert torch.allclose(
            layer.attn_scores_post_softmax.sum(dim=-1),
            torch.ones(config.n_heads, seq_len),
            atol=1e-5,
        )
        assert torch.allclose(layer.topk_weights.sum(dim=-1), torch.ones(seq_len), atol=1e-5)

        assert isinstance(layer.expert_outputs, dict)
        assert len(layer.expert_outputs) > 0
        for expert_id, expert_output in layer.expert_outputs.items():
            assert 0 <= expert_id < config.n_routed_experts
            assert expert_output.shape[-1] == config.d_model
            assert torch.isfinite(expert_output).all()


@pytest.mark.parametrize("positional_scheme", ["rope", "nope", "alibi"])
def test_forward_with_trace_matches_forward_within_tolerance(positional_scheme):
    """Requirement 7.1-7.2: `forward_with_trace` must be the *same
    computation* as `forward` -- same weights, same math -- but since
    moe-modernization task 5, `forward` runs it through fused
    `F.scaled_dot_product_attention` while `forward_with_trace` keeps the
    explicit matmul/mask/softmax/matmul steps for teachability (design.md:
    "forward_with_trace keeps the explicit math plus nothing else
    changes"). A fused kernel is free to sum in a different order (and, on
    some backends, run a different algorithm such as flash attention)
    than the manual softmax, so the two paths are no longer required to be
    *bit-for-bit* identical -- only numerically identical within a tight
    tolerance, for every positional scheme (rope/nope/alibi exercise the
    three different `attention_bias`-vs-`None` branches `Attention.
    forward` now takes)."""
    model, tokenizer = _tiny_setup(positional_scheme=positional_scheme)
    input_ids = _encode(tokenizer, "hello")

    with torch.no_grad():
        logits_plain, _ = model(input_ids)
        logits_traced, bundle = model.forward_with_trace(input_ids)

    assert torch.allclose(logits_plain, logits_traced, atol=1e-5)
    assert torch.allclose(bundle.logits, logits_plain.squeeze(0), atol=1e-5)


def test_forward_with_trace_matches_forward_within_tolerance_with_qk_norm():
    """Requirement 7.3: qk_norm=True must be applied identically on both
    paths -- same tolerance-based agreement as the qk_norm=False case
    above, now with the extra RMSNorm-on-q/k step active on both."""
    model, tokenizer = _tiny_setup(qk_norm=True)
    input_ids = _encode(tokenizer, "hello")

    with torch.no_grad():
        logits_plain, _ = model(input_ids)
        logits_traced, bundle = model.forward_with_trace(input_ids)

    assert torch.allclose(logits_plain, logits_traced, atol=1e-5)
    assert torch.allclose(bundle.logits, logits_plain.squeeze(0), atol=1e-5)


def test_forward_with_trace_rejects_batch_size_greater_than_one():
    """TraceBundle's shapes have no batch dimension by design -- tracing
    only ever covers a single prompt."""
    model, _tokenizer = _tiny_setup()
    input_ids = torch.zeros((2, 3), dtype=torch.long)
    with pytest.raises(ValueError):
        model.forward_with_trace(input_ids)


def test_forward_with_trace_populates_every_field_for_mla():
    """MLAAttention.forward_with_trace (added alongside site/token-journey's
    per-variant deep-dive support): q/k are the FULL concat(content, rope)
    tensors actually used in the dot product, shaped
    (n_heads, seq_len, head_dim + rope_head_dim) -- not GQA's
    (n_heads/n_kv_heads, seq_len, head_dim)."""
    model, tokenizer = _tiny_setup(attention_variant="mla", mla_latent_dim=8, n_kv_heads=2)
    input_ids = _encode(tokenizer, "hello world")
    seq_len = input_ids.shape[1]
    config = model.config
    head_dim = config.d_model // config.n_heads
    rope_head_dim = head_dim // 2

    with torch.no_grad():
        logits, bundle = model.forward_with_trace(input_ids)

    assert logits.shape == (1, seq_len, config.vocab_size)
    assert len(bundle.layers) == config.n_layers
    for layer in bundle.layers:
        assert layer.rope_angles_or_alibi_bias.shape == (seq_len, rope_head_dim)
        assert layer.q.shape == (config.n_heads, seq_len, head_dim + rope_head_dim)
        assert layer.k.shape == (config.n_heads, seq_len, head_dim + rope_head_dim)
        assert layer.v.shape == (config.n_heads, seq_len, head_dim)
        assert layer.attn_scores_pre_mask.shape == (config.n_heads, seq_len, seq_len)
        assert layer.attn_scores_post_softmax.shape == (config.n_heads, seq_len, seq_len)
        assert torch.allclose(
            layer.attn_scores_post_softmax.sum(dim=-1), torch.ones(config.n_heads, seq_len), atol=1e-5
        )


def test_forward_with_trace_matches_forward_within_tolerance_for_mla():
    """Same tolerance-based guarantee as the GQA test above (moe-
    modernization task 5 -- see its docstring), for the MLA attention
    variant."""
    model, tokenizer = _tiny_setup(attention_variant="mla", mla_latent_dim=8, n_kv_heads=2)
    input_ids = _encode(tokenizer, "hello")

    with torch.no_grad():
        logits_plain, _ = model(input_ids)
        logits_traced, bundle = model.forward_with_trace(input_ids)

    assert torch.allclose(logits_plain, logits_traced, atol=1e-5)
    assert torch.allclose(bundle.logits, logits_plain.squeeze(0), atol=1e-5)


def test_expert_outputs_only_populated_for_dispatched_experts():
    """Requirement 2.5, mirrored in the trace: a single token with top_k=1
    can only be dispatched to exactly one of n_routed_experts experts, so
    expert_outputs should have exactly one entry, matching topk_indices."""
    tokenizer = CharTokenizer()
    tokenizer.train("ab")
    config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=8,
        n_layers=1,
        n_heads=2,
        n_kv_heads=2,
        d_ff=16,
        n_routed_experts=4,
        n_shared_experts=0,
        top_k=1,
        max_seq_len=8,
    )
    model = QwenMoEModel(config)
    model.eval()
    input_ids = torch.tensor([[0]], dtype=torch.long)

    with torch.no_grad():
        _, bundle = model.forward_with_trace(input_ids)

    layer = bundle.layers[0]
    assert len(layer.expert_outputs) == 1
    (dispatched_expert_id,) = layer.expert_outputs.keys()
    assert layer.topk_indices[0].tolist() == [dispatched_expert_id]
    assert layer.expert_outputs[dispatched_expert_id].shape == (1, config.d_model)


# -- capture_trace: loads a real checkpoint end to end (Requirement 7.1) -----


def _write_tiny_checkpoint(tmp_path: Path, model: QwenMoEModel, tokenizer: CharTokenizer) -> Path:
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer.save(tokenizer_path)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    train_config = TrainConfig(batch_size=2, seq_len=8, total_tokens=100, device="cpu")
    checkpoint_path = tmp_path / "checkpoints" / "step_1.pt"
    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        model.config,
        train_config,
        step=1,
        tokenizer_meta={"tokenizer_type": "char", "tokenizer_path": str(tokenizer_path)},
    )
    return checkpoint_path


def test_capture_trace_loads_checkpoint_and_runs_forward_with_trace(tmp_path):
    model, tokenizer = _tiny_setup()
    checkpoint_path = _write_tiny_checkpoint(tmp_path, model, tokenizer)

    bundle = capture_trace(checkpoint_path, "hello")

    expected_ids = tokenizer.encode("hello")
    assert bundle.token_ids == expected_ids
    assert bundle.embeddings.shape == (len(expected_ids), model.config.d_model)
    assert len(bundle.layers) == model.config.n_layers
    assert bundle.probs.shape == (model.config.vocab_size,)
    assert bundle.probs.sum().item() == pytest.approx(1.0, abs=1e-4)


def test_capture_trace_rejects_empty_prompt(tmp_path):
    model, tokenizer = _tiny_setup()
    checkpoint_path = _write_tiny_checkpoint(tmp_path, model, tokenizer)

    with pytest.raises(ValueError):
        capture_trace(checkpoint_path, "")
