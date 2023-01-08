import pytorch_lightning as pl
import torch
import numpy as np
import torch.nn as nn
from torch.nn.modules.loss import CrossEntropyLoss
from .utils import DiceLoss
from src.metrics import iou, iou_loss, dice

from torchviz import make_dot

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
        self.lr = 0.0001
        self.weights = [3.0, 30.0, 10.0]

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        x, attn_weights, features = self.encoder(x)  # (B, n_patch, hidden)
        x = self.decoder(x, features)
        #print("DECODER OUTPUT SHAPE: ", x.shape)
        logits = self.segmentation_head(x)
        #print("LOGITS SHAPE: ", logits.shape)
        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        #ce_loss = CrossEntropyLoss(weight=torch.Tensor(self.weights).to(y.device))
        loss = iou_loss(y_hat, y)
        # loss = F.cross_entropy(y_hat, y)
        #dice_loss = DiceLoss(3)
        #loss = 0.5 * ce_loss(y_hat, y) + 0.5 * dice_loss(y_hat, y, softmax=True)
        #print(y_hat[:,:,:10,:10].shape)
        #print(y_hat[:,:,:10,:10])
        #print(y[:,:,:10,:10].shape)
        #print(y[:,:,:10,:10])
        #print(ce_loss(y_hat, y))
        self.log("train/loss", loss)
        return loss

    def _shared_eval_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        #loss = CrossEntropyLoss(weight=torch.Tensor(self.weights).to(y.device))(y_hat, y)
        loss = iou_loss(y_hat, y)
        dice_score = dice(y_hat, y)
        # loss = F.cross_entropy(y_hat, y)
        bg_iou, tc_iou, ar_iou = iou(y, y_hat)
        return loss, bg_iou, tc_iou, ar_iou, dice_score
    
    def validation_step(self, batch, batch_idx):
        loss, bg_iou, tc_iou, ar_iou, dice_score = self._shared_eval_step(batch, batch_idx)
        mean_iou = (bg_iou + tc_iou + ar_iou) / 3
        metrics = {
            "val/loss": loss,
            "val/bg_iou": bg_iou,
            "val/tc_iou": tc_iou,
            "val/ar_iou": ar_iou,
            "val/mean_iou": mean_iou,
            "val/dice": dice_score,
        }
        self.log_dict(metrics)
        return metrics

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)