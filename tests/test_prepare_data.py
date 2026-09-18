"""moe-modernization Requirement 10.1: `scripts/prepare_data.py` packs a
multi-document corpus with a reserved separator id between documents, and
(journal 040) streams the corpus rather than holding it in memory.

The download path isn't exercised here (it touches the network) -- only the
pure text->ids helpers and the streaming iterators, which are fed real
files under `tmp_path`.
"""
import torch

import pytest

from qwenmoe.tokenizer.bpe import ByteBPETokenizer
from qwenmoe.tokenizer.char import CharTokenizer

from scripts.prepare_data import (
    _DOCUMENT_SEPARATOR,
    _encode_documents,
    _TokenBuffer,
    encode_corpus,
    iter_documents,
    iter_text_blocks,
    split_documents,
)


def _trained_tokenizer(eos_token: bool) -> ByteBPETokenizer:
    tok = ByteBPETokenizer(target_vocab_size=300, eos_token=eos_token)
    tok.train("once upon a time there was a small robot. " * 30)
    return tok


def test_split_documents_drops_empty_fragments():
    text = f"{_DOCUMENT_SEPARATOR}story one{_DOCUMENT_SEPARATOR}{_DOCUMENT_SEPARATOR}story two{_DOCUMENT_SEPARATOR}"
    assert split_documents(text) == ["story one", "story two"]


def test_split_documents_no_separator_is_a_single_document():
    assert split_documents("just one story") == ["just one story"]


def test_encode_corpus_non_document_aware_matches_a_plain_encode():
    tok = _trained_tokenizer(eos_token=False)
    text = "once upon a time there was a robot"
    assert encode_corpus(text, tok, document_aware=False) == tok.encode(text)


def test_encode_corpus_inserts_exactly_one_separator_after_each_document():
    tok = _trained_tokenizer(eos_token=True)
    text = f"story one{_DOCUMENT_SEPARATOR}story two{_DOCUMENT_SEPARATOR}story three"
    ids = encode_corpus(text, tok, document_aware=True)

    assert ids.count(tok.eos_id) == 3  # one per document, including the last
    assert ids[-1] == tok.eos_id
    # every non-separator id is a real vocab entry the tokenizer could emit
    assert all(i < tok.eos_id for i in ids if i != tok.eos_id)
    # the pieces between separators decode back to the original documents
    pieces, current = [], []
    for i in ids:
        if i == tok.eos_id:
            pieces.append(tok.decode(current))
            current = []
        else:
            current.append(i)
    assert pieces == ["story one", "story two", "story three"]


def test_encode_corpus_document_aware_requires_a_reserved_eos():
    tok = _trained_tokenizer(eos_token=False)
    with pytest.raises(ValueError):
        encode_corpus("a<|endoftext|>b", tok, document_aware=True)


# -- streaming the corpus instead of holding it in memory (journal 040)

# Deliberately awkward for a block-wise reader: uneven line lengths, a very
# long unbroken line, and separators sitting right next to newlines.
STREAMING_CORPUS = (
    "the quick brown fox jumps over the lazy dog\n"
    "a much longer line " * 20 + "\n"
    f"{_DOCUMENT_SEPARATOR}second document starts here\nwith two lines\n"
    f"{_DOCUMENT_SEPARATOR}third\n"
)


def _corpus_file(tmp_path, text=STREAMING_CORPUS):
    path = tmp_path / "corpus.txt"
    path.write_text(text, encoding="utf-8")
    return path


@pytest.mark.parametrize("block_chars", [1, 7, 64, 10_000])
def test_iter_text_blocks_reassembles_the_file_exactly(tmp_path, block_chars):
    """Any block size must partition the file -- a boundary that drops or
    duplicates a character would silently corrupt the corpus."""
    path = _corpus_file(tmp_path)
    assert "".join(iter_text_blocks(path, block_chars)) == STREAMING_CORPUS


@pytest.mark.parametrize("block_chars", [1, 7, 64, 10_000])
def test_iter_documents_matches_split_documents(tmp_path, block_chars):
    path = _corpus_file(tmp_path)
    assert list(iter_documents(path, block_chars)) == split_documents(STREAMING_CORPUS)


@pytest.mark.parametrize("block_chars", [7, 64, 10_000])
def test_streaming_encode_matches_whole_corpus_encode(tmp_path, block_chars):
    """The claim that makes block-wise encoding safe: because blocks are cut
    at newlines and BPE merges never span a pre-token, streaming produces
    exactly the ids one whole-corpus `encode` call would."""
    path = _corpus_file(tmp_path)
    tok = _trained_tokenizer(eos_token=False)

    streamed = []
    for block in iter_text_blocks(path, block_chars):
        streamed.extend(tok.encode(block))
    assert streamed == tok.encode(STREAMING_CORPUS)


@pytest.mark.parametrize("block_chars", [7, 64, 10_000])
def test_streaming_document_encode_matches_encode_corpus(tmp_path, block_chars):
    path = _corpus_file(tmp_path)
    tok = _trained_tokenizer(eos_token=True)

    streamed: list[int] = []
    n_documents, n_chars = _encode_documents(iter_documents(path, block_chars), tok, streamed)

    assert streamed == encode_corpus(STREAMING_CORPUS, tok, document_aware=True)
    assert n_documents == len(split_documents(STREAMING_CORPUS))
    assert n_chars == sum(len(d) for d in split_documents(STREAMING_CORPUS))


def test_token_buffer_grows_past_its_initial_capacity():
    buffer = _TokenBuffer(capacity=2)
    buffer.extend([1, 2, 3])
    buffer.append(4)
    buffer.extend(list(range(5, 5000)))

    tokens = buffer.as_tensor()
    assert tokens.dtype == torch.int32
    assert tokens.tolist() == [1, 2, 3, 4] + list(range(5, 5000))


def test_token_buffer_tensor_holds_no_unused_capacity(tmp_path):
    """`as_tensor` must shrink the array, or `torch.save` writes the whole
    over-allocated buffer to disk alongside the real tokens."""
    buffer = _TokenBuffer(capacity=1_000_000)
    buffer.extend([1, 2, 3])
    tokens = buffer.as_tensor()

    assert tokens.numel() == 3
    path = tmp_path / "tokens.pt"
    torch.save(tokens, path)
    assert path.stat().st_size < 10_000


def test_bpe_train_from_blocks_matches_train_from_one_string(tmp_path):
    path = _corpus_file(tmp_path)
    whole = ByteBPETokenizer(target_vocab_size=320)
    whole.train(STREAMING_CORPUS)
    streamed = ByteBPETokenizer(target_vocab_size=320)
    streamed.train(iter_text_blocks(path, 64))

    assert streamed.vocab_size == whole.vocab_size
    assert streamed.encode(STREAMING_CORPUS) == whole.encode(STREAMING_CORPUS)


def test_char_train_from_blocks_matches_train_from_one_string(tmp_path):
    path = _corpus_file(tmp_path)
    whole = CharTokenizer()
    whole.train(STREAMING_CORPUS)
    streamed = CharTokenizer()
    streamed.train(iter_text_blocks(path, 64))

    assert streamed.vocab_size == whole.vocab_size
    assert streamed.encode(STREAMING_CORPUS) == whole.encode(STREAMING_CORPUS)
