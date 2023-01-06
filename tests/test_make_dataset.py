from src.data.make_dataset import *
import pandas as pd
import transformers
import numpy as np


def test_NewsDataset_label():
    df = pd.DataFrame(
        {"text": ["test text label 1", "test text label 0"], "label": [1, 0]}
    )
    dataset = NewsDataset(df)

    # label type
    assert type(dataset.__getitem__(0)[1]) == np.int64

    # label value
    assert dataset.__getitem__(0)[1] == 1


def test_NewsDataset_embedding():
    df = pd.DataFrame(
        {"text": ["test text label 1", "test text label 0"], "label": [1, 0]}
    )
    dataset = NewsDataset(df)

    # embedding value types
    assert type(dataset.__getitem__(0)[0]["input_ids"]) == list
    assert type(dataset.__getitem__(0)[0]["attention_mask"]) == list

    # embedding values
    assert dataset.__getitem__(0)[0]["input_ids"] == [2, 1289, 1854, 1899, 137, 3]
    assert dataset.__getitem__(0)[0]["attention_mask"] == [1, 1, 1, 1, 1, 1]


def test_get_tokenizer():
    tokenizer = get_tokenizer()

    # tokenizer type
    assert (
        type(tokenizer)
        == transformers.models.albert.tokenization_albert_fast.AlbertTokenizerFast
    )


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
