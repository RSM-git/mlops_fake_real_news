# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import os
import zipfile


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


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
