import random

import hydra
import torch
from dotenv import load_dotenv
from pytorch_lightning import Trainer, loggers
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from src.data.make_dataset import CreateData
from src.models.model import FakeNewsClassifier


@hydra.main(config_name="training_conf.yaml", config_path="configs")
def main(cfg):
    load_dotenv()

    batch_size = cfg.hyperparameters.batch_size
    lr = cfg.hyperparameters.learning_rate
    seed = cfg.hyperparameters.seed
    accelerator = cfg.hyperparameters.accelerator

    random.seed(seed)
    torch.manual_seed(seed)

    # Creates initial dataset files
    creator = CreateData()
    creator.create()
    dl_train = creator.get_data_loader("train", batch_size=batch_size)
    dl_val = creator.get_data_loader("val", batch_size=batch_size)

    model = FakeNewsClassifier(
        model_type="albert-base-v2", num_classes=2, batch_size=batch_size, lr=lr
    )

    trainer = Trainer(
        callbacks=[
            ModelCheckpoint("models/", monitor="val_loss", filename="best_model"),
            EarlyStopping(monitor="val_loss", patience=2),
        ],
        logger=loggers.WandbLogger(project="mlops_fake_real_news", entity="crulotest"),
        accelerator=accelerator,
        max_epochs=3,  # TODO: add epochs as an hparam
    )

    trainer.fit(
        model,
        train_dataloaders=dl_train,
        val_dataloaders=dl_val,
    )


if __name__ == "__main__":
    main()
