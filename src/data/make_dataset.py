import logging
import os
from pathlib import Path

# import click
import pandas as pd
import torch
import transformers
from dotenv import find_dotenv, load_dotenv
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from src.utils import download_blob


class CreateData:
    """Does csv data handling until it's split
    in train, val, and test in the data/processed directory
    """

    def __init__(self):
        logger = logging.getLogger(__name__)
        logger.info("making final data set from raw data")

        self.input_path = "data/raw/"
        self.interim_path = "data/interim/"
        self.output_path = "data/processed/"

    def download_data(self):
        """download the raw data from a google cloud bucket"""
        download_blob("fake_real_news_bucket", "Fake.csv", "data/raw/Fake.csv")
        download_blob("fake_real_news_bucket", "True.csv", "data/raw/True.csv")

    def merge_csv(self):
        """the data is originally split into two dataframe defined by label.
        This method merges them into one and adds a label column
        """
        # Load csvs
        fake = pd.read_csv(self.input_path + "Fake.csv")
        true = pd.read_csv(self.input_path + "True.csv")

        # Add label column
        fake["label"] = 0
        true["label"] = 1

        # Merge
        merged = fake.merge(true, how="outer")

        # shuffle data
        merged = merged.sample(frac=1).reset_index(drop=True)
        path = Path(self.interim_path + "merge.csv")
        merged.to_csv(path)

    def split(self):
        """splits the data into train, val, and test"""
        # load merged csv
        data = pd.read_csv(self.interim_path + "merge.csv", index_col=0)

        # split data into train, split, val, test
        df_train, df_test = train_test_split(data, test_size=0.2)
        df_val, df_test = train_test_split(df_test, test_size=0.5)

        df_train.to_csv(self.output_path + "train.csv")
        df_val.to_csv(self.output_path + "val.csv")
        df_test.to_csv(self.output_path + "test.csv")

    def create(self):
        """combines all the data methods into a single method"""
        # if the pocessed data already exist, don't create it again
        if (
            os.path.exists(self.output_path + "train.csv")
            and os.path.exists(self.output_path + "val.csv")
            and os.path.exists(self.output_path + "test.csv")
        ):
            pass
        else:
            self.download_data()
            self.merge_csv()
            self.split()

    def get_data_loader(self, split: str, **kwargs) -> torch.utils.data.DataLoader:
        """creates a dataloader for a split. kwargs are send to the pytorch
        DataLoader class

        Args:
            split (str): {train, val, test} which dataframe to use for the dataloader

        Returns:
            torch.utils.data.DataLoader: the dataloader for the given split
        """
        if split == "train":
            df = pd.read_csv(self.output_path + "train.csv")
        elif split == "val":
            df = pd.read_csv(self.output_path + "val.csv")
        elif split == "test":
            df = pd.read_csv(self.output_path + "test.csv")
        else:
            raise "Error: Invalid split given\nPossible splits: 'train' , \
                'val' , 'test'"

        ds = NewsDataset(df)
        dl = torch.utils.data.DataLoader(ds, collate_fn=collate_fn, **kwargs)

        return dl


class NewsDataset(Dataset):
    def __init__(self, df: pd.DataFrame, max_length: int = 80):
        super().__init__()
        self.df = df
        self.tokenizer = get_tokenizer()
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index: int) -> dict[str : torch.Tensor]:
        # load row
        row = self.df.iloc[index]
        text = row["title"]
        label = row["label"]
        # tokenizes the text and returns pytorch tensors
        encoding = self.tokenizer(
            text,
            return_token_type_ids=False,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        )

        return {
            "input_ids": encoding.input_ids,
            "attention_mask": encoding.attention_mask,
            "label": torch.tensor(label, dtype=torch.long),
        }


def collate_fn(batch: list[dict[str : torch.Tensor]]):
    """a custom collate function as using the built-in collate
    messes with the dimensionality of the data
    """
    input_ids = [i["input_ids"] for i in batch]
    attention_mask = [i["attention_mask"] for i in batch]
    label = [i["label"] for i in batch]

    input_ids = torch.concat(input_ids, dim=0)
    attention_mask = torch.concat(attention_mask, dim=0)
    label = torch.stack(label, dim=0)

    return {"input_ids": input_ids, "attention_mask": attention_mask, "label": label}


def get_tokenizer() -> transformers.AlbertTokenizerFast:
    """download/load the albert base tokenizer and save it
    if it's not already saved

    Returns:
        transformers.AlbertTokenizerFast: pretrained tokenizer
    """
    path_tokenizers = "tokenizers"
    tokenizer_type = "albert-base-v2"
    path_tokenizer = os.path.join(path_tokenizers, tokenizer_type)
    if not os.path.exists(path_tokenizer):
        tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_type)
        tokenizer.save_pretrained(path_tokenizer)
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(path_tokenizer)

    return tokenizer


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    creator = CreateData()
    creator.create()
