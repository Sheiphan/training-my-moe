import torch

from qwenmoe.config import ModelConfig
from qwenmoe.model.generate import generate
from qwenmoe.model.kv_cache import LatentKVCache, LayerKVCache, build_kv_cache
from qwenmoe.model.transformer import QwenMoEModel
from qwenmoe.tokenizer.char import CharTokenizer


def _model(**overrides):
    defaults = dict(
        vocab_size=40, d_model=32, n_layers=3, n_heads=4, n_kv_heads=2, d_ff=48,
        n_routed_experts=4, n_shared_experts=1, top_k=2, max_seq_len=64,
    )
    defaults.update(overrides)
    model = QwenMoEModel(ModelConfig(**defaults))
    model.eval()
    return model


def _tok():
    t = CharTokenizer()
    t.train("abcdefghijklmnopqrstuvwxyz ")
    return t


# -- the cache objects themselves -----------------------------------------


def test_layer_kv_cache_appends_and_reports_bytes():
    cache = LayerKVCache()
    assert cache.length == 0 and cache.nbytes == 0

    k1 = torch.randn(1, 2, 5, 8)
    k, v = cache.append(k1, k1.clone())
    assert k.shape == (1, 2, 5, 8) and cache.length == 5

    k, v = cache.append(torch.randn(1, 2, 1, 8), torch.randn(1, 2, 1, 8))
    assert k.shape == (1, 2, 6, 8) and cache.length == 6
    assert cache.nbytes == k.element_size() * (k.nelement() + v.nelement())


def test_latent_kv_cache_grows_latent_and_shared_key_only():
    cache = LatentKVCache()
    c1 = torch.randn(1, 4, 16)               # (batch, len, latent_dim)
    r1 = torch.randn(1, 1, 4, 6)             # (batch, 1, len, rope_head_dim)
    c, r = cache.append(c1, r1)
    assert c.shape == (1, 4, 16) and r.shape == (1, 1, 4, 6)

    c, r = cache.append(torch.randn(1, 1, 16), torch.randn(1, 1, 1, 6))
    assert c.shape == (1, 5, 16) and r.shape == (1, 1, 5, 6)
    assert cache.length == 5


def test_build_kv_cache_matches_each_block_attention_variant():
    gqa = build_kv_cache(_model())
    assert len(gqa) == 3 and all(isinstance(c, LayerKVCache) for c in gqa)

    mla = build_kv_cache(_model(attention_variant="mla", mla_latent_dim=8))
    assert all(isinstance(c, LatentKVCache) for c in mla)


# -- cached forward == uncached forward (Requirement 8.2) ------------------


def test_cached_incremental_forward_matches_a_full_recompute():
    """Feeding tokens one-at-a-time through the cache produces the same
    last-position logits as a single full forward over the whole
    sequence, within fused-kernel tolerance."""
    torch.manual_seed(0)
    model = _model()
    ids = torch.randint(0, 40, (1, 12))

    full_logits, _ = model(ids)

    caches = build_kv_cache(model)
    model(ids[:, :5], kv_caches=caches)  # prefill
    step_logits = []
    for t in range(5, 12):
        lg, _ = model(ids[:, t : t + 1], kv_caches=caches, position_offset=t)
        step_logits.append(lg[:, -1, :])

    for t, lg in zip(range(5, 12), step_logits):
        assert torch.allclose(lg, full_logits[:, t, :], atol=1e-4)


@torch.no_grad()
def _by_hand_bytes(cache_list):
    return sum(c.nbytes for c in cache_list)


def test_mla_runtime_cache_is_smaller_than_gqa_at_matched_shape():
    """Requirement 8.3: a *measured* retained-bytes comparison (replaces the
    parent spec's static `kv_cache_elements_per_token` estimate)."""
    torch.manual_seed(0)
    shape = dict(vocab_size=40, d_model=64, n_layers=3, n_heads=8, n_kv_heads=8,
                 d_ff=64, n_routed_experts=4, n_shared_experts=1, top_k=2, max_seq_len=128)
    gqa = QwenMoEModel(ModelConfig(**shape, attention_variant="gqa")).eval()
    mla = QwenMoEModel(ModelConfig(**shape, attention_variant="mla", mla_latent_dim=32)).eval()
    ids = torch.randint(0, 40, (1, 40))

    cg = build_kv_cache(gqa)
    gqa(ids, kv_caches=cg)
    cm = build_kv_cache(mla)
    mla(ids, kv_caches=cm)

    assert _by_hand_bytes(cm) < _by_hand_bytes(cg)


# -- generate() with the cache (Requirement 8.1, 8.2, 8.4) ----------------


def test_cached_and_uncached_greedy_generation_are_identical():
    tok = _tok()
    for variant, extra, scheme in [
        ("gqa", {}, "rope"),
        ("gqa", {}, "nope"),
        ("gqa", {}, "alibi"),
        ("mla", {"mla_latent_dim": 8}, "rope"),
    ]:
        torch.manual_seed(0)
        model = _model(
            vocab_size=tok.vocab_size, attention_variant=variant,
            positional_scheme=scheme, **extra,
        )
        a = generate(model, tok, "hello world", max_new_tokens=18,
                     temperature=0.0, use_kv_cache=False)
        b = generate(model, tok, "hello world", max_new_tokens=18,
                     temperature=0.0, use_kv_cache=True)
        assert a == b, f"{variant}/{scheme}: {a!r} != {b!r}"


def test_cached_generation_still_stops_at_max_seq_len():
    """Requirement 8.4 / 6.3: the context-length stop is unaffected by the
    cache."""
    tok = _tok()
    model = _model(vocab_size=tok.vocab_size, max_seq_len=12)
    out = generate(model, tok, "abcde", max_new_tokens=100, temperature=0.0, use_kv_cache=True)
    assert len(tok.encode(out)) == 12
