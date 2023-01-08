import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import wandb

from src.metrics import iou, iou_loss, dice


class UNet(pl.LightningModule):
    def __init__(self, hparams):
        super(UNet, self).__init__()
        for key in hparams.keys():
            self.hparams[key] = hparams[key]

        self.n_channels = hparams["n_channels"]
        self.n_classes = hparams["n_classes"]
        self.bilinear = True
        self.lr = 0.001
        self.hyperparams = {
            "lr": self.lr,
            "weight_decay": 2e-8,
            "model": "medium",
            # "focal_loss_weight": 50,
            "class_weights": [0.35, 70, 5.9],
            "optimizer": "adamW",  # rmsprob, adamW
        }

        def double_conv(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        def down(in_channels, out_channels):
            return nn.Sequential(
                nn.MaxPool2d(2), double_conv(in_channels, out_channels)
            )

        class up(nn.Module):
            def __init__(self, in_channels, out_channels, bilinear=True):
                super().__init__()

                if bilinear:
                    self.up = nn.Upsample(
                        scale_factor=2, mode="bilinear", align_corners=True
                    )
                else:
                    self.up = nn.ConvTranpose2d(
                        in_channels // 2, in_channels // 2, kernel_size=2, stride=2
                    )

                self.conv = double_conv(in_channels, out_channels)

            def forward(self, x1, x2):
                x1 = self.up(x1)
                # [?, C, H, W]
                diffY = x2.size()[2] - x1.size()[2]
                diffX = x2.size()[3] - x1.size()[3]

                x1 = F.pad(
                    x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2]
                )
                x = torch.cat([x2, x1], dim=1)  ## why 1?
                return self.conv(x)

        if self.hyperparams["model"] == "large":
            # Large
            self.inc = double_conv(self.n_channels, 64)
            self.down1 = down(64, 128)
            self.down2 = down(128, 256)
            self.down3 = down(256, 512)
            self.down4 = down(512, 512)
            self.up1 = up(1024, 256)
            self.up2 = up(512, 128)
            self.up3 = up(256, 64)
            self.up4 = up(128, 64)
            self.out = nn.Conv2d(64, self.n_classes, kernel_size=1)
        elif self.hyperparams["model"] == "medium":
            # Medium
            self.inc = double_conv(self.n_channels, 32)
            self.down1 = down(32, 64)
            self.down2 = down(64, 128)
            self.down3 = down(128, 256)
            self.down4 = down(256, 256)
            self.up1 = up(512, 128)
            self.up2 = up(256, 64)
            self.up3 = up(128, 32)
            self.up4 = up(64, 32)
            self.out = nn.Conv2d(32, self.n_classes, kernel_size=1)
        else:
            # Small
            self.inc = double_conv(self.n_channels, 16)
            self.down1 = down(16, 32)
            self.down2 = down(32, 64)
            self.down3 = down(64, 128)
            self.down4 = down(128, 128)
            self.up1 = up(256, 64)
            self.up2 = up(128, 32)
            self.up3 = up(64, 16)
            self.up4 = up(32, 16)
            self.out = nn.Conv2d(16, self.n_classes, kernel_size=1)

    def forward(self, x):
        # x = x.float()
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.out(x)
        x = F.softmax(x, dim=1)
        return x

    def training_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)
        loss = iou_loss(y_hat, y)
        #loss = F.cross_entropy(y_hat, y)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)
        loss = iou_loss(y_hat, y)
        #loss = F.cross_entropy(y_hat, y)
        self.log("val/loss", loss)

        bg_iou, tc_iou, ar_iou = iou(y, y_hat)
        mean_iou = (bg_iou + tc_iou + ar_iou) / 3
        dice_score = dice(y_hat, y)
        self.log("val/bg_iou", bg_iou)
        self.log("val/tc_iou", tc_iou)
        self.log("val/ar_iou", ar_iou)
        self.log("val/mean_iou", mean_iou)
        self.log("val/dice", dice_score)


        # Log Images
        print_freq = 20
        rand_num = random.randint(0, print_freq)
        if rand_num == 0:
            max_idx_pred = (
                torch.argmax(y_hat, 1, keepdim=True)[0].detach().cpu().numpy() * 122
            )
            img_pred = wandb.Image(max_idx_pred, caption="Prediction")
            max_idx_true = (
                torch.argmax(y, 1, keepdim=True)[0].detach().cpu().numpy() * 122
            )
            img_true = wandb.Image(max_idx_true, caption="Groundtruth")
            wandb.log({"Validation Image: ": [img_pred, img_true]})
            # self.log_image(key="Val_image", images=[img_pred])

    """
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss}
        print("End of Validation ", avg_loss)
        self.log("val_loss_at_end", avg_loss.detach().cpu())

        return {"avg_val_loss": avg_loss, "log": tensorboard_logs}
    """

    def configure_optimizers(self):
        if self.hyperparams["optimizer"] == "rmsprob":
            return torch.optim.RMSprop(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.hyperparams["weight_decay"],
            )
        else:
            return torch.optim.AdamW(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.hyperparams["weight_decay"],
            )
