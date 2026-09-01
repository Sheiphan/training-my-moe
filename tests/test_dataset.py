import pytest
import torch
from torch.utils.data import DataLoader

from qwenmoe.data.dataset import TokenDataset, train_val_split


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
