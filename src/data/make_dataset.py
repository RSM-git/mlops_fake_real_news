# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

from torch.utils.data import Dataset
import pandas as pd
import torch
import os
import transformers


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, output_filepath):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")


def get_tokenizer() -> transformers.AlbertTokenizerFast:
    path_tokenizers = "tokenizers"
    tokenizer_type = "albert"
    path_tokenizer = os.path.join(path_tokenizers, tokenizer_type)
    if not os.path.exists(path_tokenizer):
        tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_type)
        tokenizer.save_pretrained(path_tokenizer)
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(path_tokenizer)
    return


class NewsDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self.df = df
        self.tokenizer = transformers

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index: int) -> dict[str : torch.Tensor]:
        # load row
        row = self.df.iloc[0]
        text = row["text"]


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
