import torch
import torch.nn as nn
import torchmetrics
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification


class FakeNewsClassifier(LightningModule):
    def __init__(
        self,
        model: str = "albert-base-v2",
        num_classes: int = 2,
        batch_size: int = 32,
        lr: int = 2e-5,
    ):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model, num_labels=num_classes
        )
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
        self.batch_size = batch_size
        self.lr = lr

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        logits = self(input_ids, attention_mask)
        loss = self.criterion(logits, labels)

        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        logits = self(input_ids, attention_mask)
        loss = self.criterion(logits, labels)

        return loss

    def test_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        logits = self(input_ids, attention_mask)
        loss = self.criterion(logits, labels)

        return loss

    def configure_optimizers(self, lr: float = 2e-5) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=True)

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=True)
