import pytorch_lightning as pl
import torch
import numpy as np
import torch.nn as nn
from .networks.vit_seg_modeling import ResNetV2

from encoder import Encoder
from decoder import Decoder

class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)


class TransUNet(nn.Module):
    def __init__(self, params, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(TransUNet, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = params.classifier
        self.encoder = Encoder(params, img_size)
        self.decoder = Decoder(params)
        self.segmentation_head = SegmentationHead(
            in_channels=params['decoder_channels'][-1],
            out_channels=params['n_classes'],
            kernel_size=3,
        )
        self.params = params

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