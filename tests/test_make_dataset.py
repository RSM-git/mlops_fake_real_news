import os

import numpy as np
import pandas as pd
import pytest
import torch
import transformers

from src.data.make_dataset import CreateData, NewsDataset


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

    def test_load_kaggle(self):
        # get and unzip zip files
        self.creator.load_kaggle()

        assert os.path.exists("data/raw/Fake.csv")
        assert os.path.exists("data/raw/True.csv")

    pytest.mark.skipif(not os.path.exists("data/raw/train.csv"))

    def test_merge_csv(self):
        self.creator.merge_csv()
        assert os.path.exists("data/processed/merge.csv")

    pytest.mark.skipif(not os.path.exists("data/processed/merge.csv"))

    def test_split(self):
        self.creator.split()
        assert os.path.exists("data/processed/train.csv")
        assert os.path.exists("data/processed/val.csv")
        assert os.path.exists("data/processed/test.csv")

    pytest.mark.skipif(not os.path.exists("data/processed/train.csv"))

    def test_get_dataloader(self):
        dl_train = self.creator.get_data_loader("train", batch_size=2)
        dl_val = self.creator.get_data_loader("val", num_workers=2)
        dl_test = self.creator.get_data_loader("test", shuffle=True)

        assert isinstance(dl_train, torch.utils.data.DataLoader)
        train_sample = next(iter(dl_train))
        assert train_sample[0].shape[0] == 2


def test_get_tokenizer():
    df = pd.DataFrame(
        {"text": ["test text label 1", "test text label 0"], "label": [1, 0]}
    )
    dataset = NewsDataset(df)
    # OSError
    # "tokenizers" folder should include the "albert-base-v2" folder, as well as includes the 3 json files
    assert dataset.get_tokenizer(), OSError
    # delete "albert-base-v2" and run again


# TODO: add dataset testing

# def test_NewsDataset_label():
#     dataset = NewsDataset(df)

#     # label type
#     assert type(dataset[0]["label"]) == np.int64

#     # label value
#     assert dataset[0]["label"] == 1


# def test_NewsDataset_embedding():
#     df = pd.DataFrame(
#         {"text": ["test text label 1", "test text label 0"], "label": [1, 0]}
#     )
#     dataset = NewsDataset(df)

#     # embedding value types
#     assert type(dataset[0]["input_ids"]) == list
#     assert type(dataset[0]["attention_mask"]) == list

#     # embedding values
#     assert dataset[0]["input_ids"] == [2, 1289, 1854, 1899, 137, 3]
#     assert dataset[0]["attention_mask"] == [1, 1, 1, 1, 1, 1]
