import logging
import os
import zipfile
from pathlib import Path

import click
import pandas as pd
import torch
import transformers
from dotenv import find_dotenv, load_dotenv
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


@click.command()
# @click.argument("input_filepath", type=click.Path(exists=True))
# @click.argument("output_filepath", type=click.Path())
def main():
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    input_path = "data/raw/"
    output_path = "data/processed/"

    os.makedirs(os.path.join(input_path, "zip_folder"), exist_ok=True)
    load_kaggle(input_path)
    merge_csv(input_path, output_path)
    split(output_path)


def load_kaggle(input_path: str):
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
        exit()

    zipped_filepath = input_path + "/zip_folder"

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
    unzipped_folder_raw = input_path
    with zipfile.ZipFile(os.path.join(zipped_filepath, "Fake.csv.zip"), "r") as zip_ref:
        zip_ref.extractall(unzipped_folder_raw)
    with zipfile.ZipFile(os.path.join(zipped_filepath, "True.csv.zip"), "r") as zip_ref:
        zip_ref.extractall(unzipped_folder_raw)


def merge_csv(input_path: str, output_path: str) -> pd.DataFrame:
    a = pd.read_csv(input_path + "Fake.csv")  # Load csv
    b = pd.read_csv(input_path + "True.csv")

    # Add label column
    a["label"] = 0
    b["label"] = 1

    # Merge
    merged = a.merge(b, how="outer")
    merged = merged.sample(frac=1).reset_index(drop=True)
    merged.to_csv(output_path + "merge.csv")


def split(
    output_path: str,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    # load merged csv
    data = pd.read_csv(output_path + "merge.csv", index_col=0)

    # split data into train, split, val, test
    # split is split up into val and test
    y = data.label
    X = data.drop("label", axis=1)
    X_train, X_split, y_train, y_split = train_test_split(X, y, test_size=0.2)
    X_val, X_test, y_val, y_test = train_test_split(X_split, y_split, test_size=0.5)

    X_train["label"] = y_train.values
    X_val["label"] = y_val.values
    X_test["label"] = y_test.values

    X_train.to_csv(output_path + "train.csv")
    X_val.to_csv(output_path + "validation.csv")
    X_test.to_csv(output_path + "test.csv")


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
