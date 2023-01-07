# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import os
import zipfile

from torch.utils.data import Dataset
import pandas as pd
import torch
import transformers
import numpy as np


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, output_filepath):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    # Check if zipped folder exists, before loading data files
    zipped_filepath = input_filepath + "/zip_folder"
    if os.path.isdir(zipped_filepath):
        load_kaggle(input_filepath, zipped_filepath, output_filepath)
    else:
        print("Files already exist")


def load_kaggle(input_filepath, zipped_filepath, output_filepath):
    # Load environment variables
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir)
    dotenv_path = os.path.join(project_dir, ".env")
    load_dotenv(dotenv_path)

    # Check that kaggle API authentication works
    try:
        import kaggle
    except OSError as e:
        print("Kaggle API error:")
        print(e)

    api = kaggle.api

    # Download zipped data
    kaggle.api.dataset_download_file(
        "clmentbisaillon/fake-and-real-news-dataset",
        "Fake.csv",
        path=zipped_filepath,
    )
    kaggle.api.dataset_download_file(
        "clmentbisaillon/fake-and-real-news-dataset",
        "True.csv",
        path=zipped_filepath,
    )

    # Unzip data
    unzipped_folder_raw = input_filepath
    with zipfile.ZipFile(os.path.join(zipped_filepath, "Fake.csv.zip"), "r") as zip_ref:
        zip_ref.extractall(unzipped_folder_raw)
    with zipfile.ZipFile(os.path.join(zipped_filepath, "True.csv.zip"), "r") as zip_ref:
        zip_ref.extractall(unzipped_folder_raw)


def get_tokenizer() -> transformers.AlbertTokenizerFast:
    path_tokenizers = "tokenizers"
    tokenizer_type = "albert-base-v2"
    path_tokenizer = os.path.join(path_tokenizers, tokenizer_type)
    if not os.path.exists(path_tokenizer):
        tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_type)
        tokenizer.save_pretrained(path_tokenizer)
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(path_tokenizer)

    return tokenizer


class NewsDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self.df = df
        self.tokenizer = get_tokenizer()  # transformers

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index: int) -> dict[str : torch.Tensor]:
        # load row
        row = self.df.iloc[0]
        text = row["text"]
        label = row["label"]
        encoding = self.tokenizer(text, return_token_type_ids=False)
        encoding["label"] = label

        return encoding


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
