import pytest
import torch
from torch.utils.data import DataLoader

from qwenmoe.data.dataset import (
    TokenDataset,
    _document_ids,
    _split_into_documents,
    train_val_split,
)


def test_token_dataset_length_and_shapes():
    ds = TokenDataset(list(range(20)), seq_len=5)
    assert len(ds) == 15
    x, y = ds[0]
    assert x.shape == (5,)
    assert y.shape == (5,)


def test_token_dataset_windows_are_shifted_by_one():
    ds = TokenDataset(list(range(20)), seq_len=5)
    x0, y0 = ds[0]
    assert x0.tolist() == [0, 1, 2, 3, 4]
    assert y0.tolist() == [1, 2, 3, 4, 5]

    x3, y3 = ds[3]
    assert x3.tolist() == [3, 4, 5, 6, 7]
    assert y3.tolist() == [4, 5, 6, 7, 8]


def test_token_dataset_dtype_is_long():
    x, y = TokenDataset(list(range(10)), seq_len=3)[0]
    assert x.dtype == torch.long
    assert y.dtype == torch.long


def test_token_dataset_rejects_too_few_tokens():
    with pytest.raises(ValueError):
        TokenDataset(list(range(3)), seq_len=5)


def test_token_dataset_rejects_invalid_seq_len():
    with pytest.raises(ValueError):
        TokenDataset(list(range(10)), seq_len=0)


def test_token_dataset_batches_via_dataloader():
    loader = DataLoader(TokenDataset(list(range(100)), seq_len=8), batch_size=4)
    x, y = next(iter(loader))
    assert x.shape == (4, 8)
    assert y.shape == (4, 8)


def test_train_val_split_proportions():
    train, val = train_val_split(list(range(100)), val_fraction=0.2)
    assert len(train) == 80
    assert len(val) == 20


def test_train_val_split_is_order_preserving_and_contiguous():
    train, val = train_val_split(list(range(100)), val_fraction=0.1)
    assert train.tolist() == list(range(90))
    assert val.tolist() == list(range(90, 100))


@pytest.mark.parametrize("val_fraction", [0.0, 1.0, -0.1, 1.1])
def test_train_val_split_rejects_invalid_fraction(val_fraction):
    with pytest.raises(ValueError):
        train_val_split(list(range(10)), val_fraction=val_fraction)


# -- document boundaries (moe-modernization Requirement 10) -----------------

EOS = 99


def _packed_stream(doc_lengths, eos_id=EOS):
    """Flat `eos`-separated stream: doc i is [10*i, 10*i+1, ...] then eos."""
    out = []
    for i, length in enumerate(doc_lengths):
        out.extend(range(10 * i, 10 * i + length))
        out.append(eos_id)
    return out


def test_document_ids_new_document_starts_after_each_separator():
    """The <eos> token belongs to the document it terminates; the next
    token opens a new document (Llama-3 convention)."""
    stream = torch.tensor([5, 6, 7, EOS, 5, 6, EOS, 8, 9])
    assert _document_ids(stream, EOS).tolist() == [0, 0, 0, 0, 1, 1, 1, 2, 2]


def test_token_dataset_return_segments_requires_eos_id():
    with pytest.raises(ValueError):
        TokenDataset(list(range(20)), seq_len=5, return_segments=True)


def test_token_dataset_return_segments_yields_rebased_document_ids():
    stream = _packed_stream([3, 4, 5])  # [0,1,2,99, 10,11,12,13,99, 20,21,22,23,24,99]
    ds = TokenDataset(stream, seq_len=5, return_segments=True, eos_id=EOS)

    x0, y0, seg0 = ds[0]
    assert x0.tolist() == [0, 1, 2, EOS, 10]
    assert seg0.tolist() == [0, 0, 0, 0, 1]  # separator ends doc 0, next token is doc 1

    # a window straddling a separator, rebased so its first token is document 0
    x, y, seg = ds[2]
    assert x.tolist() == [2, EOS, 10, 11, 12]
    assert seg.tolist() == [0, 0, 1, 1, 1]

    # a window fully inside one document is all zeros after rebasing
    x_mid, _, seg_mid = ds[4]
    assert x_mid.tolist() == [10, 11, 12, 13, EOS]
    assert seg_mid.tolist() == [0, 0, 0, 0, 0]


def test_token_dataset_without_return_segments_is_unchanged_two_tuple():
    ds = TokenDataset(list(range(20)), seq_len=5)
    item = ds[0]
    assert len(item) == 2


def test_split_into_documents_keeps_each_terminating_separator():
    stream = torch.tensor(_packed_stream([2, 3]))
    docs = _split_into_documents(stream, EOS)
    assert [d.tolist() for d in docs] == [[0, 1, EOS], [10, 11, 12, EOS]]


def test_train_val_split_tail_mode_matches_the_default():
    tokens = list(range(100))
    assert all(
        torch.equal(a, b)
        for a, b in zip(train_val_split(tokens, 0.2), train_val_split(tokens, 0.2, mode="tail"))
    )


def test_train_val_split_documents_holds_out_whole_documents_only():
    stream = _packed_stream([8] * 10)  # 10 docs, 9 tokens each (8 + separator)
    train, val = train_val_split(stream, val_fraction=0.25, mode="documents", eos_id=EOS)

    assert train.numel() + val.numel() == len(stream)
    # every document is intact: each half ends exactly on a separator
    assert train[-1].item() == EOS and val[-1].item() == EOS
    # >= val_fraction of tokens, taken as whole docs from the tail
    assert val.numel() >= 0.25 * len(stream)
    # whole docs added from the tail until the token target is met: 3 * 9 = 27
    assert (val == EOS).sum().item() == 3
    assert (train == EOS).sum().item() == 7


def test_train_val_split_documents_requires_eos_id():
    with pytest.raises(ValueError):
        train_val_split(list(range(50)), mode="documents")


def test_train_val_split_documents_rejects_single_document_stream():
    with pytest.raises(ValueError):
        train_val_split(list(range(50)), mode="documents", eos_id=EOS)


def test_train_val_split_rejects_unknown_mode():
    with pytest.raises(ValueError):
        train_val_split(list(range(50)), mode="sideways")
