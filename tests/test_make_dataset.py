import os

import numpy as np
import pandas as pd
import pytest
import torch
import transformers

from src.data.make_dataset import CreateData, NewsDataset, get_tokenizer


def remove_data_files():
    dir_paths = ["data/raw", "data/processed"]

    raw_data_files = [
        file
        for file in os.listdir(dir_paths[0])
        if file.endswith(".csv") or file.endswith(".zip")
    ]
    processed_data_files = [
        file for file in os.listdir(dir_paths[1]) if file.endswith(".csv")
    ]
    data_files = (raw_data_files, processed_data_files)

    for i in range(len(dir_paths)):
        for file in data_files[i]:
            path = os.path.join(dir_paths[i], file)
            os.remove(path)


remove_data_files()


class TestCreateData:
    creator = CreateData()

    @pytest.mark.skipif(
        not os.path.exists("data/raw/train.csv"), reason="data not found"
    )
    def test_merge_csv(self):
        self.creator.merge_csv()
        assert os.path.exists("data/processed/merge.csv")

    @pytest.mark.skipif(
        not os.path.exists("data/processed/merge.csv"), reason="data not found"
    )
    def test_split(self):
        self.creator.split()
        assert os.path.exists("data/processed/train.csv")
        assert os.path.exists("data/processed/val.csv")
        assert os.path.exists("data/processed/test.csv")

    @pytest.mark.skipif(
        not os.path.exists("data/processed/train.csv"), reason="data not found"
    )
    def test_get_dataloader(self):
        dl_train = self.creator.get_data_loader("train", batch_size=2)
        dl_val = self.creator.get_data_loader("val", num_workers=2)
        dl_test = self.creator.get_data_loader("test", shuffle=True)

        assert isinstance(dl_train, torch.utils.data.DataLoader)
        train_sample = next(iter(dl_train))
        assert train_sample[0].shape[0] == 2


def test_get_tokenizer():
    tokenizer = get_tokenizer()
    assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)


def test_NewsDataset():
    df = pd.DataFrame(
        {"title": ["test text label 1", "test text label 0"], "label": [1, 0]}
    )
    ds = NewsDataset(df)
    sample = ds[0]
    assert len(sample) == 3
    assert "input_ids" in sample
    assert "attention_mask" in sample
    assert "label" in sample
    for value in sample.values():
        assert isinstance(value, torch.Tensor)
