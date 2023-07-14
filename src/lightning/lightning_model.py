import torch
from torch import nn
import pytorch_lightning as pl
from torchmetrics.classification import Accuracy


class SentenceClassifier(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        num_classes: int,
        lr: float = 1e-4,
        l2_reg_alpha: float = 0,
        lr_scheduler_ctor: float | None = None,
    ):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.lr = lr
        self.l2_reg_alpha = l2_reg_alpha
        self.lr_scheduler_ctor = lr_scheduler_ctor
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.criterion(pred, y)
        self.log("train_loss", loss, prog_bar=True)
        self.train_acc(pred.argmax(dim=1), y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.criterion(pred, y)
        self.log("val_loss", loss, prog_bar=True)
        self.val_acc(pred.argmax(dim=1), y)

    def on_train_epoch_end(self):
        # log training accuracy at the end of each training epoch
        self.log("train_acc", self.train_acc.compute(), prog_bar=True)

    def on_val_epoch_end(self):
        # log validation accuracy at the end of each validation epoch
        self.log("val_acc", self.val_acc.compute(), prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.l2_reg_alpha
        )
        if self.lr_scheduler_ctor is not None:
            lr_scheduler = self.lr_scheduler_ctor(optimizer)
            return [optimizer], [lr_scheduler]
        else:
            return optimizer
