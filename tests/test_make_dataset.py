from src.data.make_dataset import *
import pandas as pd
import transformers
import numpy as np


def test_get_tokenizer():

    # OSError
    # "tokenizers" folder should include the "albert-base-v2" folder, as well as includes the 3 json files
    assert get_tokenizer(), OSError
    # delete "albert-base-v2" and run again


def test_merge_csv():
    assert isinstance(merge_csv(), pd.DataFrame)


def test_split():
    assert isinstance(split(), tuple)


def test_NewsDataset_label():
    df = pd.DataFrame(
        {"text": ["test text label 1", "test text label 0"], "label": [1, 0]}
    )
    dataset = NewsDataset(df)

    # label type
    assert type(dataset[0]["label"]) == np.int64

    # label value
    assert dataset[0]["label"] == 1


def test_NewsDataset_embedding():
    df = pd.DataFrame(
        {"text": ["test text label 1", "test text label 0"], "label": [1, 0]}
    )
    dataset = NewsDataset(df)

    # embedding value types
    assert type(dataset[0]["input_ids"]) == list
    assert type(dataset[0]["attention_mask"]) == list

    # embedding values
    assert dataset[0]["input_ids"] == [2, 1289, 1854, 1899, 137, 3]
    assert dataset[0]["attention_mask"] == [1, 1, 1, 1, 1, 1]


def test_load_kaggle():
    input_filepath = ""
    zipped_filepath = input_filepath + "/foldername"
    output_filepath = ""

    # *commented for now*
    # load_kaggle(input_filepath, zipped_filepath, output_filepath)

    # kaggle API authentication
    ...

    # are folders / files genereated?
    ...
