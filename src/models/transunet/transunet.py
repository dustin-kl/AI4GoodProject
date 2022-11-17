import pytorch_lightning as pl
import torch
import numpy as np
import torch.nn as nn
from .networks.vit_seg_modeling import ResNetV2

from encoder import Encoder
from decoder import Decoder

class SegmentationHead(nn.Module):
    def __init__(self, parameters):
        super().__init__()
        self.conv2d = nn.Conv2d(
            in_channels=parameters["in_channels"],
            out_channels=parameters["out_channels"],
            kernel_size=parameters["kernel_size"],
            padding=parameters["padding"],
        )
        scale_factor = parameters["upsampling"]
        if scale_factor > 1:
            self.upsampling = nn.UpsamplingBilinear2d(
                scale_factor=scale_factor
            )
        else:
            self.upsampling = nn.Identity()

    def forward(self, x):
        x = self.conv2d(x)
        x = self.upsampling(x)
        return x

class TransUNet(pl.LightningModule):
    def __init__(self, parameters):
        super().__init__(parameters)
        self.encoder = Encoder(parameters)
        self.decoder = Decoder(parameters)
        self.logits = SegmentationHead(parameters)


    def forward(self, x):
        x, features = self.encoder(x)
        x = self.decoder(x, features)
        logits = self.logits(x)
        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = None
        return loss

    def validation_step(self, batch, barch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = None
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = None
        return loss

    def configure_optimizers(self):
        return None