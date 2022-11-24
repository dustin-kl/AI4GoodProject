import pytorch_lightning as pl
import torch
import numpy as np
import torch.nn as nn
from torch.nn.modules.loss import CrossEntropyLoss
from .utils import DiceLoss

from .encoder import Encoder
from .decoder import Decoder

class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)


class TransUNet(pl.LightningModule):
    def __init__(self, params, n_channels=1):
        super(TransUNet, self).__init__()
        self.classifier = params["classifier"]
        self.encoder = Encoder(params, 224, n_channels)
        self.decoder = Decoder(params)
        self.segmentation_head = SegmentationHead(
            in_channels=params['decoder_channels'][-1],
            out_channels=params['n_classes'],
            kernel_size=3,
        )
        self.params = params
        self.lr = 0.01

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        x, attn_weights, features = self.encoder(x)  # (B, n_patch, hidden)
        x = self.decoder(x, features)
        logits = self.segmentation_head(x)
        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        ce_loss = CrossEntropyLoss()
        dice_loss = DiceLoss(3)
        loss = 0.5 * ce_loss(y_hat, y[:].long()) + 0.5 * dice_loss(y_hat, y, softmax=True)
        return loss

    def validation_step(self, batch, barch_idx):
        x, y = batch
        y_hat = self.forward(x)
        ce_loss = CrossEntropyLoss()
        dice_loss = DiceLoss(3)
        loss = 0.5 * ce_loss(y_hat, y[:].long()) + 0.5 * dice_loss(y_hat, y, softmax=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        ce_loss = CrossEntropyLoss()
        dice_loss = DiceLoss(3)
        loss = 0.5 * ce_loss(y_hat, y[:].long()) + 0.5 * dice_loss(y_hat, y, softmax=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)