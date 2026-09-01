import pytest

from qwenmoe.config import DataConfig
from qwenmoe.tokenizer.base import Tokenizer, build_tokenizer
from qwenmoe.tokenizer.bpe import ByteBPETokenizer

CORPUS = "the quick brown fox jumps over the lazy dog. " * 40


def test_train_reaches_configured_target_vocab_size():
    target = 256 + 20
    tok = ByteBPETokenizer(target_vocab_size=target)
    tok.train(CORPUS)
    assert tok.vocab_size == target


def test_round_trip_lossless_for_trained_text():
    tok = ByteBPETokenizer(target_vocab_size=300)
    tok.train(CORPUS)
    ids = tok.encode(CORPUS)
    assert tok.decode(ids) == CORPUS


def test_round_trip_lossless_for_unseen_unicode_text():
    """Byte-level BPE covers *any* UTF-8 string, even characters that never
    appeared in training (unlike CharTokenizer's fixed corpus vocab)."""
    tok = ByteBPETokenizer(target_vocab_size=300)
    tok.train(CORPUS)
    unseen = "héllo wörld 你好 🎉"
    assert tok.decode(tok.encode(unseen)) == unseen


def test_encode_compresses_relative_to_raw_bytes():
    tok = ByteBPETokenizer(target_vocab_size=300)
    tok.train(CORPUS)
    ids = tok.encode(CORPUS)
    assert len(ids) < len(CORPUS.encode("utf-8"))


def test_untrained_tokenizer_falls_back_to_raw_bytes():
    tok = ByteBPETokenizer(target_vocab_size=300)
    text = "hi"
    assert tok.encode(text) == list(text.encode("utf-8"))
    assert tok.decode(tok.encode(text)) == text


def test_constructor_rejects_target_below_base_vocab():
    with pytest.raises(ValueError):
        ByteBPETokenizer(target_vocab_size=100)


def test_decode_out_of_range_id_raises_value_error():
    tok = ByteBPETokenizer(target_vocab_size=260)
    tok.train(CORPUS)
    with pytest.raises(ValueError):
        tok.decode([0, tok.vocab_size + 50])


def test_save_and_load_round_trip(tmp_path):
    tok = ByteBPETokenizer(target_vocab_size=280)
    tok.train(CORPUS)
    path = tmp_path / "bpe_tokenizer.json"
    tok.save(path)

    loaded = ByteBPETokenizer()  # default target; load() must override it
    loaded.load(path)
    assert loaded.vocab_size == tok.vocab_size
    assert loaded.encode(CORPUS) == tok.encode(CORPUS)
    assert loaded.decode(loaded.encode(CORPUS)) == CORPUS


def test_build_tokenizer_selects_byte_bpe_tokenizer():
    config = DataConfig(corpus="tiny_stories", tokenizer_type="bpe", bpe_vocab_size=280)
    tok = build_tokenizer(config)
    assert isinstance(tok, ByteBPETokenizer)
    assert isinstance(tok, Tokenizer)
    assert tok.vocab_size == 256  # untrained: base byte vocab only, not the target yet
