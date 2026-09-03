"""moe-modernization Requirement 10.1: `scripts/prepare_data.py` packs a
multi-document corpus with a reserved separator id between documents.

The download/train path isn't exercised here (it touches the network) --
only the pure text->ids helpers `split_documents` and `encode_corpus`.
"""
import pytest

from qwenmoe.tokenizer.bpe import ByteBPETokenizer

from scripts.prepare_data import _DOCUMENT_SEPARATOR, encode_corpus, split_documents


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
