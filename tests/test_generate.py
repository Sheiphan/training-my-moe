import pytest
import torch

from qwenmoe.config import ModelConfig
from qwenmoe.model.generate import filter_top_k, filter_top_p, generate
from qwenmoe.model.transformer import QwenMoEModel
from qwenmoe.tokenizer.char import CharTokenizer


def _tiny_setup(max_seq_len: int = 64, **overrides):
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
        max_seq_len=max_seq_len,
    )
    defaults.update(overrides)
    model = QwenMoEModel(ModelConfig(**defaults))
    return model, tokenizer


# -- filter_top_k --------------------------------------------------------


def test_filter_top_k_keeps_only_k_highest_logits():
    logits = torch.tensor([[2.0, 0.5, 3.0, -1.0]])

    filtered = filter_top_k(logits, k=2)

    assert filtered[0, 0] == 2.0
    assert filtered[0, 2] == 3.0
    assert torch.isinf(filtered[0, 1]) and filtered[0, 1] < 0
    assert torch.isinf(filtered[0, 3]) and filtered[0, 3] < 0


def test_filter_top_k_softmax_gives_zero_probability_to_filtered_tokens():
    logits = torch.tensor([[2.0, 0.5, 3.0, -1.0]])
    filtered = filter_top_k(logits, k=2)
    probs = torch.softmax(filtered, dim=-1)
    assert probs[0, 1] == 0.0
    assert probs[0, 3] == 0.0
    assert probs[0].sum().item() == pytest.approx(1.0)


def test_filter_top_k_larger_than_vocab_is_a_no_op():
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    filtered = filter_top_k(logits, k=10)
    assert torch.equal(filtered, logits)


def test_filter_top_k_of_1_keeps_only_the_argmax():
    logits = torch.tensor([[1.0, 5.0, 3.0, -2.0]])
    filtered = filter_top_k(logits, k=1)
    assert filtered[0, 1] == 5.0
    assert torch.isinf(filtered[0, [0, 2, 3]]).all()


def test_filter_top_k_rejects_non_positive_k():
    logits = torch.tensor([[1.0, 2.0]])
    with pytest.raises(ValueError):
        filter_top_k(logits, k=0)


# -- filter_top_p (nucleus sampling) --------------------------------------


def test_filter_top_p_keeps_smallest_prefix_summing_past_threshold():
    """Worked example: probabilities [0.5, 0.2, 0.15, 0.1, 0.05] (already
    descending, sums to 1). Using logits = ln(p) reproduces exactly these
    probabilities after softmax, since softmax is shift-invariant:
    softmax(ln p)_i = p_i / sum(p) = p_i here. With top_p=0.7 the smallest
    prefix whose cumulative probability is >= 0.7 is {0.5, 0.2}, summing
    to exactly 0.7 -- so only the first two tokens should survive.
    """
    probs = torch.tensor([[0.5, 0.2, 0.15, 0.1, 0.05]])
    logits = torch.log(probs)

    filtered = filter_top_p(logits, top_p=0.7)

    assert torch.isfinite(filtered[0, :2]).all()
    assert torch.isinf(filtered[0, 2:]).all()


def test_filter_top_p_includes_the_token_that_crosses_the_threshold():
    """Same distribution, top_p=0.71: {0.5, 0.2} sums to 0.7 < 0.71, so
    the smallest sufficient prefix must also include the third token
    (0.15), whose inclusion brings the cumulative sum to 0.85 >= 0.71."""
    probs = torch.tensor([[0.5, 0.2, 0.15, 0.1, 0.05]])
    logits = torch.log(probs)

    filtered = filter_top_p(logits, top_p=0.71)

    assert torch.isfinite(filtered[0, :3]).all()
    assert torch.isinf(filtered[0, 3:]).all()


def test_filter_top_p_of_1_keeps_everything():
    logits = torch.tensor([[5.0, 1.0, 0.1, -3.0]])
    filtered = filter_top_p(logits, top_p=1.0)
    assert torch.equal(filtered, logits)


def test_filter_top_p_always_keeps_the_single_highest_probability_token():
    logits = torch.tensor([[10.0, 1.0, 1.0, 1.0]])
    filtered = filter_top_p(logits, top_p=1e-6)
    assert torch.isfinite(filtered[0, 0])
    assert torch.isinf(filtered[0, 1:]).all()


def test_filter_top_p_rejects_out_of_range_values():
    logits = torch.tensor([[1.0, 2.0]])
    with pytest.raises(ValueError):
        filter_top_p(logits, top_p=0.0)
    with pytest.raises(ValueError):
        filter_top_p(logits, top_p=1.5)


# -- generate: input validation --------------------------------------------


def test_generate_rejects_negative_max_new_tokens():
    model, tokenizer = _tiny_setup()
    with pytest.raises(ValueError):
        generate(model, tokenizer, "hello", max_new_tokens=-1)


def test_generate_rejects_negative_temperature():
    model, tokenizer = _tiny_setup()
    with pytest.raises(ValueError):
        generate(model, tokenizer, "hello", temperature=-0.1)


def test_generate_rejects_bad_top_k_or_top_p():
    model, tokenizer = _tiny_setup()
    with pytest.raises(ValueError):
        generate(model, tokenizer, "hello", top_k=0)
    with pytest.raises(ValueError):
        generate(model, tokenizer, "hello", top_p=1.5)


def test_generate_rejects_empty_prompt():
    model, tokenizer = _tiny_setup()
    with pytest.raises(ValueError):
        generate(model, tokenizer, "")


# -- generate: greedy decoding is deterministic (Requirement 6.2) ---------


def test_greedy_generation_is_deterministic_regardless_of_random_seed():
    model, tokenizer = _tiny_setup()

    torch.manual_seed(0)
    out_a = generate(model, tokenizer, "hello ", max_new_tokens=20, temperature=0.0)
    torch.manual_seed(999)
    out_b = generate(model, tokenizer, "hello ", max_new_tokens=20, temperature=0.0)

    assert out_a == out_b


def test_generate_top_k_1_matches_greedy_output():
    """top_k=1 leaves only the argmax with nonzero probability, so
    `torch.multinomial` degenerately always draws it -- same output as
    temperature=0 greedy, regardless of the RNG seed in effect."""
    model, tokenizer = _tiny_setup()
    prompt = "hello "

    torch.manual_seed(0)
    greedy = generate(model, tokenizer, prompt, max_new_tokens=15, temperature=0.0)
    torch.manual_seed(123)
    top_k_1 = generate(model, tokenizer, prompt, max_new_tokens=15, temperature=1.0, top_k=1)

    assert greedy == top_k_1


# -- generate: length never exceeds configured limits (Requirement 6.3) ---


def test_generate_never_exceeds_max_new_tokens():
    model, tokenizer = _tiny_setup(max_seq_len=1000)
    prompt = "hello world"

    out = generate(model, tokenizer, prompt, max_new_tokens=15, temperature=1.0)

    # CharTokenizer is exactly 1 character per token, so lengths compare directly.
    assert len(out) <= len(prompt) + 15
    assert out.startswith(prompt)


def test_generate_zero_max_new_tokens_returns_prompt_unchanged():
    model, tokenizer = _tiny_setup()
    prompt = "hello"
    assert generate(model, tokenizer, prompt, max_new_tokens=0) == prompt


def test_generate_stops_at_model_max_seq_len_with_room_for_no_new_tokens():
    model, tokenizer = _tiny_setup(max_seq_len=10)
    prompt = "hello worl"  # exactly 10 characters -- already at max_seq_len
    assert len(tokenizer.encode(prompt)) == model.config.max_seq_len

    out = generate(model, tokenizer, prompt, max_new_tokens=50, temperature=1.0)

    assert len(out) == model.config.max_seq_len


def test_generate_stops_partway_between_prompt_and_max_new_tokens():
    model, tokenizer = _tiny_setup(max_seq_len=15)
    prompt = "hello worl"  # 10 tokens; room for exactly 5 more before max_seq_len=15

    out = generate(model, tokenizer, prompt, max_new_tokens=50, temperature=1.0)

    assert len(out) == 15


# -- generate: runs correctly on CPU (Requirement 6.4) ---------------------


def test_generate_runs_correctly_on_cpu():
    model, tokenizer = _tiny_setup()

    out = generate(model, tokenizer, "hello", max_new_tokens=10, temperature=1.0, device="cpu")

    assert isinstance(out, str)
    assert out.startswith("hello")


def test_generate_default_device_is_cpu():
    model, tokenizer = _tiny_setup()
    out = generate(model, tokenizer, "hello", max_new_tokens=5)
    assert isinstance(out, str)


def test_generate_use_kv_cache_composes_with_sampling_filters():
    """moe-modernization Requirement 8: the `use_kv_cache` flag is
    orthogonal to `temperature`/`top_k`/`top_p` -- it changes *how* the
    forward passes are computed, not the sampling."""
    model, tokenizer = _tiny_setup(max_seq_len=40)
    out = generate(
        model, tokenizer, "hello", max_new_tokens=15,
        temperature=0.8, top_k=5, top_p=0.9, use_kv_cache=True,
    )
    assert isinstance(out, str) and out.startswith("hello")
    assert len(out) == len("hello") + 15


def test_generate_works_unchanged_on_an_mtp_model():
    """moe-modernization Requirement 6.4: MTP heads are training-only, so
    `generate()` (which runs `model.eval()` and never passes `targets`)
    behaves exactly as on a non-MTP model."""
    torch.manual_seed(0)
    plain, tokenizer = _tiny_setup()
    torch.manual_seed(0)
    mtp, _ = _tiny_setup(mtp_depth=2)

    plain_out = generate(plain, tokenizer, "hello", max_new_tokens=8, temperature=0.0)
    mtp_out = generate(mtp, tokenizer, "hello", max_new_tokens=8, temperature=0.0)

    assert isinstance(mtp_out, str) and mtp_out.startswith("hello")
    # different weights (MTP model has extra params consuming the RNG stream),
    # but the generation path itself is identical and greedy-deterministic
    assert len(mtp_out) == len(plain_out)
