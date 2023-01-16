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


@click.command()
@click.option("--config_file", type=str, default="train_subset_gpu.yaml")
def main(config_file: str):
    load_dotenv(".env")
    config_dir = "configs"
    config_file = config_file
    config_file_path = os.path.join(config_dir, config_file)

    wandb.login(key=os.environ["WANDB_API_KEY"])

    logger = loggers.WandbLogger(
        project="mlops_fake_real_news", entity="crulotest", config=config_file_path
    )
    config = logger.experiment.config
    random.seed(config.seed)
    torch.manual_seed(config.seed)

    # Creates initial dataset files
    creator = CreateData()
    creator.create()
    dl_train = creator.get_data_loader(
        "train", batch_size=config.batch_size, num_workers=config.num_workers
    )
    dl_val = creator.get_data_loader(
        "val", batch_size=config.batch_size, num_workers=config.num_workers
    )

    model = FakeNewsClassifier(
        model_type="albert-base-v2",
        num_classes=2,
        batch_size=config.batch_size,
        lr=config.learning_rate,
    )

    trainer = Trainer(
        callbacks=[
            ModelCheckpoint("models/", monitor="val_loss", filename="best_model"),
            EarlyStopping(monitor="val_loss", patience=2),
        ],
        logger=logger,
        accelerator=config.device,
        max_epochs=config.epochs,
        # limit_train_batches=config.limit_train_batches,
        # limit_val_batches=config.limit_val_batches,
        # log_every_n_steps=config.limit_train_batches
    )
    trainer.fit(
        model,
        train_dataloaders=dl_train,
        val_dataloaders=dl_val,
    )


if __name__ == "__main__":
    main()
