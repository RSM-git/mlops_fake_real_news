from pytorch_lightning import Trainer, loggers
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import hydra

# from mlops_fake_real_news.src.models.model import FakeNewsClassifier
from src.models.model import FakeNewsClassifier

import random
import torch


@hydra.main(config_name="training_conf.yaml", config_path="configs")
def main(cfg):

    batch_size = cfg.hyperparameters.batch_size
    lr = cfg.hyperparameters.learning_rate
    seed = cfg.hyperparameters.seed

    random.seed(seed)
    torch.manual_seed(seed)

    model = FakeNewsClassifier(
        model="albert-base-v2", num_classes=2, batch_size=batch_size, lr=lr
    )

    trainer = Trainer(
        model,
        callbacks=[
            ModelCheckpoint("models/", monitor="val_loss", filename="best_model"),
            EarlyStopping(monitor="val_loss", patience=2),
        ],
        logger=loggers.WandbLogger(project="mlops_fake_real_news"),
    )

    trainer.fit(model)


if __name__ == "__main__":
    main()
