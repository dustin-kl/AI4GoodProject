import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torch


class Model(pl.LightningModule):
    """
    Do not call this class, it is used only for testing.
    """

    def __init__(self, parameters):
        super().__init__()
        self.params = parameters
        self.cnv = nn.Conv2d(1, 1, 1)
        self.bn = nn.BatchNorm2d(1)

    def forward(self, x):
        x = self.cnv(x)
        x = self.bn(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = nn.CrossEntropyLoss()
        self.log("loss", loss(y, y_hat))
        return {"loss": loss(y, y_hat)}

    def validation_step(self, batch, barch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = nn.CrossEntropyLoss()
        self.log("val_loss", loss(y, y_hat))

    def validation_epoch_end(self, outputs):
        print(outputs)

    def test_step(self, batch, batch_idx):
        x = batch[:, 0]
        y = batch[:, 1]
        y_hat = self.forward(x)
        loss = nn.CrossEntropyLoss()
        return loss(y, y_hat)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())
