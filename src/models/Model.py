from audioop import avgpp
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torch
from src.utils import Logger


class Model(pl.LightningModule):
    """
    Do not call this class, it is used only for testing.
    """

    def __init__(self, parameters):
        super().__init__()
        self.params = parameters
        self.cnv = nn.Conv2d(1, 3, 1)
        self.bn = nn.BatchNorm2d(1)

    def forward(self, x):
        x = self.cnv(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = nn.CrossEntropyLoss()
        self.log("loss", loss(y, y_hat), on_step=False, on_epoch=True, prog_bar=False)
        return {"loss": loss(y, y_hat)}

    def training_epoch_end(self, training_step_outputs):
        print('training steps', training_step_outputs)
        avg_loss = training_step_outputs[0]['loss']
        Logger.log_info(f"Training Loss of Epoch {self.trainer.current_epoch}: " + str(avg_loss))


    def validation_step(self, batch, barch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = nn.CrossEntropyLoss()
        self.log("val_loss", loss(y, y_hat), on_step=False, on_epoch=True, prog_bar=False)
        #Logger.log_info("Val Loss: " + str(loss(y, y_hat)) + "\n ")

    def validation_epoch_end(self, validation_step_outputs):
        avg_loss = sum(validation_step_outputs) / (len(validation_step_outputs) + 1)
        Logger.log_info(f"Validation Loss of Epoch {self.trainer.current_epoch}: " + str(avg_loss) + "\n ")


    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = nn.CrossEntropyLoss()
        self.log("test_loss", loss(y, y_hat), on_step=False, on_epoch=True, prog_bar=False)
        
        return loss(y, y_hat)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())
