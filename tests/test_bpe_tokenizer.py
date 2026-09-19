import pytest

from qwenmoe.config import DataConfig
from qwenmoe.tokenizer.base import Tokenizer, build_tokenizer
from qwenmoe.tokenizer.bpe import _PRE_TOKEN_PATTERN, ByteBPETokenizer

CORPUS = "the quick brown fox jumps over the lazy dog. " * 40

# Every class of character the pre-tokenizer has to place in some chunk:
# underscores (which Python's `\w` counts as a word character, unlike
# GPT-2's `\p{L}`/`\p{N}`), symbols, digits, contractions, newlines, tabs
# and non-Latin text.
AWKWARD_TEXT = "file_name.py\tsnake_case x=1+2 don't 100% <tag> café 你好\n\nend_"


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


def test_pre_tokenization_covers_every_character():
    """`finditer` silently skips text no alternative matches, so any gap in
    the pattern deletes characters from the corpus. The chunks must
    reassemble into exactly the input."""
    chunks = [m.group() for m in _PRE_TOKEN_PATTERN.finditer(AWKWARD_TEXT)]
    assert "".join(chunks) == AWKWARD_TEXT


def test_round_trip_lossless_for_awkward_text():
    tok = ByteBPETokenizer(target_vocab_size=300)
    tok.train(CORPUS)
    assert tok.decode(tok.encode(AWKWARD_TEXT)) == AWKWARD_TEXT


def test_round_trip_lossless_when_trained_on_awkward_text():
    tok = ByteBPETokenizer(target_vocab_size=300)
    tok.train(AWKWARD_TEXT * 20)
    assert tok.decode(tok.encode(AWKWARD_TEXT)) == AWKWARD_TEXT


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


# -- reserved <eos> / document separator (moe-modernization Requirement 10.1)


def test_eos_token_off_by_default_and_leaves_vocab_size_unchanged():
    tok = ByteBPETokenizer(target_vocab_size=280)
    tok.train(CORPUS)
    assert tok.eos_id is None
    assert tok.vocab_size == 280


def test_eos_token_reserves_one_id_past_the_learned_vocab():
    tok = ByteBPETokenizer(target_vocab_size=280, eos_token=True)
    tok.train(CORPUS)
    assert tok.vocab_size == 281
    assert tok.eos_id == 280
    # encode never emits it; it is not a byte value
    assert tok.eos_id not in tok.encode(CORPUS)


def test_decode_drops_the_eos_token():
    tok = ByteBPETokenizer(target_vocab_size=280, eos_token=True)
    tok.train(CORPUS)
    ids = tok.encode("hello") + [tok.eos_id] + tok.encode("world")
    assert tok.decode(ids) == "helloworld"


def test_save_load_round_trips_the_eos_flag(tmp_path):
    tok = ByteBPETokenizer(target_vocab_size=280, eos_token=True)
    tok.train(CORPUS)
    path = tmp_path / "bpe.json"
    tok.save(path)

    loaded = ByteBPETokenizer()  # eos_token defaults False; load() must restore it
    loaded.load(path)
    assert loaded.eos_id == tok.eos_id
    assert loaded.vocab_size == tok.vocab_size


# -- _count_pairs, the reference counting step (journal 042) ------------------


def test_count_pairs_counts_adjacent_pairs():
    from qwenmoe.tokenizer.bpe import _count_pairs

    assert _count_pairs([1, 2, 1, 2, 3]) == {(1, 2): 2, (2, 1): 1, (2, 3): 1}


def test_count_pairs_on_short_sequences():
    from qwenmoe.tokenizer.bpe import _count_pairs

    assert _count_pairs([]) == {}
    assert _count_pairs([7]) == {}


def test_count_pairs_agrees_with_the_trainers_weighted_counting():
    """The reference form and the trainer's frequency-weighted form must
    agree on a single unweighted sequence -- that equivalence is what makes
    the simple one honest to teach first."""
    from collections import Counter

    from qwenmoe.tokenizer.bpe import _count_pairs

    ids = [5, 9, 5, 9, 4, 5, 9]
    weighted = Counter()
    for pair in zip(ids, ids[1:]):
        weighted[pair] += 1  # frequency 1, one chunk
    assert _count_pairs(ids) == dict(weighted)
