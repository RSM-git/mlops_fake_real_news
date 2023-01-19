import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics

from src.data.make_dataset import get_tokenizer
from src.models.utils import get_model


class FakeNewsClassifier(pl.LightningModule):
    """
    FakeNewsClassifier is a PyTorch Lightning module that wraps a transformer model
    """

    def __init__(
        self,
        model_type: str = "albert-base-v2",
        num_classes: int = 2,
        lr: int = 2e-5,
    ):
        """

        Args:
            model_type (str): name of the transformer model to use
                see https://huggingface.co/models
            num_classes (int): number of classes the model should predict
            lr (float): learning rate for the optimizer
        """
        super().__init__()
        self.model = get_model(model_type, num_labels=num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.training_accuracy = torchmetrics.Accuracy(task="binary")
        self.validation_accuracy = torchmetrics.Accuracy(task="binary")
        self.lr = lr
        self.max_length = 80

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """

        Args:
            input_ids (torch.Tensor): encoded batch of text
            attention_mask (torch.Tensor): attention mask for the batch of text

        Returns:
            torch.Tensor: logits for each of the samples in the batch

        """
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits

    def predict_from_str(self, text: str) -> str:
        """Takes

        Args:
            text (str): example text to perform a prediction on

        Returns:
            str: prediction either Real or Fake

        """
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

        logits = self(input_ids, attention_mask)
        out = torch.argmax(logits)

        return "Real" if out else "Fake"

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """
        Args:
            batch: dictionary containing keys "input_ids", "attention_mask", "label"
            batch_idx: index of the batch

        Returns:
            model loss on the batch
        """
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["label"]

        predictions = self.model(input_ids, attention_mask).logits
        loss = self.criterion(predictions, labels)

        self.log("train_loss", loss)
        self.training_accuracy(predictions, labels)
        self.log(
            "train_accuracy",
            self.training_accuracy,
            on_epoch=False,
            on_step=True,
        )

        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """necessary for the trainer to perform a validation step

        Args:
            batch (dict): dictionary containing keys "input_ids",
                "attention_mask", "label"
            batch_idx (int): index of the batch

        Returns:
            model loss on the batch
        """
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["label"]

        predictions = self.model(input_ids, attention_mask).logits
        loss = self.criterion(predictions, labels)

        self.log("val_loss", loss)
        self.velidation_accuracy(predictions, labels)
        self.log(
            "val_accuracy",
            self.validation_accuracy,
            on_epoch=True,
            on_step=True,
        )

    def test_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """necessary for the trainer to perform a test step

        Args:
            batch (dict): dictionary containing keys "input_ids",
                "attention_mask", "label"
            batch_idx (int): index of the batch

        Returns:
            model loss on the batch
        """
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["label"]
        logits = self(input_ids, attention_mask)
        loss = self.criterion(logits, labels)

        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Initializes the optimizer for the trainer"""
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
