from src.data.make_dataset import *
import pandas as pd
import transformers

# def test_NewsDataset():
#     df =


def test_get_tokenizer():
    tokenizer = get_tokenizer()
    assert type(tokenizer) == transformers.AlbertTokenizer
