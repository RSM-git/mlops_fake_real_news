import os
import random

import click
import torch
from dotenv import load_dotenv
from pytorch_lightning import Trainer, loggers
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

import wandb
from src.data.make_dataset import CreateData
from src.models.model import FakeNewsClassifier
from src.utils import load_yaml, upload_blob


@click.command()
@click.option("--config_file", type=str, default="train_cpu.yaml")
def main(config_file: str):
    """trains the model based on the parameters of the
    config file

    Args:
        config_file (str): yaml file with hyperparameters. Must have the following
            dl and trainer fields.
    """
    # load secrets
    load_dotenv(".env")

    # load hyparparameters from config file
    config_dir = "configs"
    config_file = config_file
    config_file_path = os.path.join(config_dir, config_file)
    nested_config_dict = load_yaml(config_file_path)

    # setup wandb
    wandb.login(key=os.environ["WANDB_API_KEY"])

    logger = loggers.WandbLogger(
        project="mlops_fake_real_news", entity="crulotest", config=nested_config_dict
    )
    config = logger.experiment.config

    # seed for repoducibility
    random.seed(config.seed)
    torch.manual_seed(config.seed)

    # Creates initial dataset files
    creator = CreateData()
    creator.create()
    dl_train = creator.get_data_loader("train", **config.dl)
    dl_val = creator.get_data_loader("val", **config.dl)

    model = FakeNewsClassifier(
        model_type="albert-base-v2",
        num_classes=2,
        lr=config.lr,
    )

    # cpu trainer
    trainer = Trainer(
        callbacks=[
            ModelCheckpoint("models/", monitor="val_loss", filename="best_model"),
            EarlyStopping(monitor="val_loss", patience=2),  # early stopping not needed
            # but good practice to avoid wasting resources
        ],
        logger=logger,
        **config.trainer
    )
    trainer.fit(
        model,
        train_dataloaders=dl_train,
        val_dataloaders=dl_val,
    )

    # upload the trained model to the bucket
    upload_blob("fake_real_news_bucket", "models/best_model.ckpt", "best_model.ckpt")


if __name__ == "__main__":
    main()
