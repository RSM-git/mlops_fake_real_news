from pytorch_lightning import Trainer, loggers
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import hydra

from mlops_fake_real_news.src.models.model import FakeNewsClassifier

# @hydra.main(config_name=)
model = FakeNewsClassifier("albert-base-v2", 2)

trainer = Trainer(
    model,
    callbacks=[
        ModelCheckpoint("models/", monitor="val_loss", filename="best_model"),
        EarlyStopping(monitor="val_loss", patience=2),
    ],
    logger=loggers.WandbLogger(project="mlops_fake_real_news"),
)

trainer.fit(model)
