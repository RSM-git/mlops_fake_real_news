import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics

from src.data.make_dataset import get_tokenizer
from src.models.utils import get_model


class FakeNewsClassifier(pl.LightningModule):
    def __init__(
        self,
        model_type: str = "albert-base-v2",
        num_classes: int = 2,
        lr: int = 2e-5,
    ):
        super().__init__()
        self.model = get_model(model_type, num_labels=num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.training_metrics = nn.ModuleDict(
            [
                [
                    "accuracy",
                    torchmetrics.Accuracy(task="binary", num_classes=num_classes),
                ]
            ]
        )
        self.validation_metrics = nn.ModuleDict(
            [
                [
                    "accuracy",
                    torchmetrics.Accuracy(task="binary", num_classes=num_classes),
                ]
            ]
        )
        self.lr = lr
        self.max_length = 80

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits

    def predict_from_str(self, text: str) -> str:
        tokenizer = get_tokenizer()
        encoding = tokenizer(
            text,
            return_token_type_ids=False,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        )
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]

        logits = self.model(input_ids, attention_mask).logits
        out = torch.argmax(logits)

        return "True" if out else "Fake"

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["label"]
        logits = self.model(input_ids, attention_mask).logits
        loss = self.criterion(logits, labels)
        predictions = logits.argmax(dim=1)
        self.training_metrics["accuracy"](predictions, labels)
        self.log("train_loss", loss)
        self.log(
            "train_accuracy",
            self.training_metrics["accuracy"],
            on_epoch=False,
            on_step=True,
        )

        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["label"]
        logits = self.model(input_ids, attention_mask).logits
        loss = self.criterion(logits, labels)
        predictions = logits.argmax(dim=1)
        self.validation_metrics["accuracy"](predictions, labels)
        self.log("val_loss", loss)
        self.log(
            "val_accuracy",
            self.validation_metrics["accuracy"],
            on_epoch=True,
            on_step=True,
        )

    def test_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["label"]
        logits = self.model(input_ids, attention_mask)
        loss = self.criterion(logits, labels)

        return loss

    def configure_optimizers(self, lr: float = 2e-5) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
