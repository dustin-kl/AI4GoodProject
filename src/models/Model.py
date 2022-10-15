import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torch


class Model(pl.LightningModule):
    def __init__(self, parameters):
        super().__init__()
        self.params = parameters
        self.l1 = nn.Linear(28 * 28, 10)

    def forward(self, x):
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        return {"loss": loss, 'val_loss': loss}

    def validation_step(self, batch, barch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("val_loss", loss)

    def validation_epoch_end(self, outputs):
        print(outputs)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)
