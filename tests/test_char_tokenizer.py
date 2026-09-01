import pytest

from qwenmoe.config import DataConfig
from qwenmoe.tokenizer.base import Tokenizer, build_tokenizer
from qwenmoe.tokenizer.char import CharTokenizer

CORPUS = "To be, or not to be: that is the question.\nWhether 'tis nobler…"


def test_vocab_is_sorted_unique_characters():
    tok = CharTokenizer()
    tok.train("cba" + "abc")
    assert tok.vocab_size == 3
    assert tok.encode("abc") == [0, 1, 2]  # sorted: 'a' < 'b' < 'c'


def test_round_trip_lossless_for_arbitrary_corpus_text():
    tok = CharTokenizer()
    tok.train(CORPUS)
    ids = tok.encode(CORPUS)
    assert tok.decode(ids) == CORPUS


def test_round_trip_lossless_for_substring_of_trained_corpus():
    tok = CharTokenizer()
    tok.train(CORPUS)
    sample = "the question"
    assert tok.decode(tok.encode(sample)) == sample


def test_encode_unknown_character_raises_value_error():
    tok = CharTokenizer()
    tok.train("abc")
    with pytest.raises(ValueError):
        tok.encode("abcd")


def test_decode_out_of_range_id_raises_value_error():
    tok = CharTokenizer()
    tok.train("abc")
    with pytest.raises(ValueError):
        tok.decode([0, 1, 99])


def test_save_and_load_round_trip(tmp_path):
    tok = CharTokenizer()
    tok.train(CORPUS)
    path = tmp_path / "char_tokenizer.json"
    tok.save(path)

    loaded = CharTokenizer()
    loaded.load(path)
    assert loaded.vocab_size == tok.vocab_size
    assert loaded.encode(CORPUS) == tok.encode(CORPUS)
    assert loaded.decode(loaded.encode(CORPUS)) == CORPUS


def test_build_tokenizer_selects_char_tokenizer():
    config = DataConfig(corpus="tiny_shakespeare", tokenizer_type="char")
    tok = build_tokenizer(config)
    assert isinstance(tok, CharTokenizer)
    assert isinstance(tok, Tokenizer)
