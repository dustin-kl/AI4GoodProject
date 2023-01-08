import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from src.metrics import iou, iou_loss, dice


def fixed_padding(inputs, kernel_size, rate):
    kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = F.pad(inputs, (pad_beg, pad_end, pad_beg, pad_end))
    return padded_inputs


class SeparableConv2d(pl.LightningModule):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, 3, bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)

    def forward(self, x):
        x = fixed_padding(x, 3, 1)
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class EntryBlock(pl.LightningModule):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.s_conv1 = SeparableConv2d(in_channels, out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.relu2 = nn.ReLU()
        self.s_conv2 = SeparableConv2d(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.mp3 = nn.MaxPool2d(3, stride=2, padding=1)

        self.conv = nn.Conv2d(in_channels, out_channels, 1, stride=2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x1 = self.conv(x)
        x1 = self.bn(x1)

        x2 = self.s_conv1(x)
        x2 = self.bn1(x2)
        x2 = self.relu2(x2)
        x2 = self.s_conv2(x2)
        x2 = self.bn2(x2)
        x2 = self.mp3(x2)

        return x1 + x2


class MiddleBlock(pl.LightningModule):
    def __init__(self, channels):
        super().__init__()

        self.relu1 = nn.ReLU()
        self.s_conv1 = SeparableConv2d(channels, channels)
        self.bn1 = nn.BatchNorm2d(channels)

        self.relu2 = nn.ReLU()
        self.s_conv2 = SeparableConv2d(channels, channels)
        self.bn2 = nn.BatchNorm2d(channels)

        self.relu3 = nn.ReLU()
        self.s_conv3 = SeparableConv2d(channels, channels)
        self.bn3 = nn.BatchNorm2d(channels)

    def forward(self, x):
        x1 = x

        x2 = self.relu1(x)
        x2 = self.s_conv1(x2)
        x2 = self.bn1(x2)
        x2 = self.relu2(x)
        x2 = self.s_conv2(x2)
        x2 = self.bn2(x2)
        x2 = self.relu3(x)
        x2 = self.s_conv3(x2)
        x2 = self.bn3(x2)

        return x1 + x2


class ExitBlock(pl.LightningModule):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.relu1 = nn.ReLU()
        self.s_conv1 = SeparableConv2d(in_channels, in_channels)
        self.bn1 = nn.BatchNorm2d(in_channels)

        self.relu2 = nn.ReLU()
        self.s_conv2 = SeparableConv2d(in_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.mp3 = nn.MaxPool2d(3, stride=2, padding=1)

        self.conv = nn.Conv2d(in_channels, out_channels, 1, stride=2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x1 = self.conv(x)
        x1 = self.bn(x1)

        x2 = self.relu1(x)
        x2 = self.s_conv1(x2)
        x2 = self.bn1(x2)
        x2 = self.relu2(x)
        x2 = self.s_conv2(x2)
        x2 = self.bn2(x2)
        x2 = self.mp3(x2)

        return x1 + x2


class EntryFlow(pl.LightningModule):
    def __init__(self, in_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, 32, 3, stride=2, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()

        self.block3 = EntryBlock(64, 128)
        self.block4 = EntryBlock(128, 256)
        self.block5 = EntryBlock(256, 728)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x1 = x2 = self.block3(x)
        x1 = x3 = self.block4(x1)
        x1 = self.block5(x1)

        return x1, x3, x2


class MiddleFlow(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.sequential = nn.Sequential(
            MiddleBlock(728),
            MiddleBlock(728),
            MiddleBlock(728),
            MiddleBlock(728),
            MiddleBlock(728),
            MiddleBlock(728),
            MiddleBlock(728),
            MiddleBlock(728),
            MiddleBlock(728),
            MiddleBlock(728),
            MiddleBlock(728),
            MiddleBlock(728),
            MiddleBlock(728),
            MiddleBlock(728),
            MiddleBlock(728),
            MiddleBlock(728),
        )

    def forward(self, x):
        return self.sequential(x)


class ExitFlow(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.block = ExitBlock(728, 1024)
        self.sequential = nn.Sequential(
            SeparableConv2d(1024, 1536),
            nn.BatchNorm2d(1536),
            nn.ReLU(),
            SeparableConv2d(1536, 2048),
            nn.BatchNorm2d(2048),
            nn.ReLU(),
        )

    def forward(self, x):
        x1 = x2 = self.block(x)
        x1 = self.sequential(x1)
        return x1, x2


class Xception(pl.LightningModule):
    def __init__(self, in_channels):
        super().__init__()

        self.entry = EntryFlow(in_channels)
        self.middle = MiddleFlow()
        self.exit = ExitFlow()

    def forward(self, x):
        x1, x2, x3 = self.entry(x)
        x1 = x4 = self.middle(x1)
        x1, x5 = self.exit(x1)

        return x1, x5, x4, x2, x3


class AstrousConv2d(pl.LightningModule):
    def __init__(self, in_channels, out_channels, rate):
        super().__init__()

        if rate == 1:
            padding = 0
            kernel_size = 1
        else:
            padding = rate
            kernel_size = 3

        self.sequential = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
                dilation=rate,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.sequential(x)


class ASPP(pl.LightningModule):
    def __init__(self, in_channels, out_channels, rates):
        super().__init__()

        self.astrous1 = AstrousConv2d(in_channels, out_channels, rates[0])
        self.astrous2 = AstrousConv2d(in_channels, out_channels, rates[1])
        self.astrous3 = AstrousConv2d(in_channels, out_channels, rates[2])
        self.astrous4 = AstrousConv2d(in_channels, out_channels, rates[3])
        self.ap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.sequential = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        mid_channels = (out_channels * 5) // 16
        self.se_block = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels * 5, mid_channels, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(mid_channels, out_channels * 5, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x1 = self.astrous1(x)
        x2 = self.astrous2(x)
        x3 = self.astrous3(x)
        x4 = self.astrous4(x)
        x5 = self.ap(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode="bilinear", align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        w = self.se_block(x)
        x = x * w
        x = self.sequential(x)

        return x


class DeepLabV3PlusAttention(pl.LightningModule):
    def __init__(
        self,
        out_channels,
        encoder_channels=[2048, 1024, 728, 256, 128],
        aspp_out=256,
        rates=[1, 6, 12, 18],
    ):
        super().__init__()
        self.encoder = Xception(4)

        self.aspp1 = ASPP(encoder_channels[0], aspp_out, rates)
        self.aspp2 = ASPP(encoder_channels[1], aspp_out, rates)
        self.aspp3 = ASPP(encoder_channels[2], aspp_out, rates)
        self.aspp4 = ASPP(encoder_channels[3], aspp_out, rates)

        self.conv1 = nn.Conv2d(4 * aspp_out, aspp_out, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(aspp_out)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(128, 48, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(48)
        self.relu2 = nn.ReLU()

        self.last_conv = nn.Sequential(
            nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            #nn.Dropout(0.5),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            #nn.Dropout(0.5),
            nn.Conv2d(256, out_channels, kernel_size=1, stride=1, bias=False),
        )

    def forward(self, x):
        in_size = x.size()

        x1, x2, x3, x4, x5 = self.encoder(x)
        x1 = self.aspp1(x1)
        x2 = self.aspp2(x2)
        x2 = F.interpolate(x2, size=x1.size()[2:], mode="bilinear", align_corners=True)
        x3 = self.aspp3(x3)
        x3 = F.interpolate(x3, size=x1.size()[2:], mode="bilinear", align_corners=True)
        x4 = self.aspp4(x4)
        x4 = F.interpolate(x4, size=x1.size()[2:], mode="bilinear", align_corners=True)

        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = F.interpolate(x, size=x5.size()[2:], mode="bilinear", align_corners=True)

        x5 = self.conv2(x5)
        x5 = self.bn2(x5)
        x5 = self.relu2(x5)

        x = torch.cat((x, x5), dim=1)

        x = self.last_conv(x)

        x = F.interpolate(x, size=in_size[2:], mode="bilinear", align_corners=True)

        x = F.softmax(x, dim=1)

        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = iou_loss(y_hat, y)
        # loss = F.cross_entropy(y_hat, y)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, bg_iou, tc_iou, ar_iou, dice_score = self._shared_eval_step(batch, batch_idx)
        mean_iou = (bg_iou + tc_iou + ar_iou) / 3
        metrics = {
            "val/loss": loss.item(),
            "val/bg_iou": bg_iou.item(),
            "val/tc_iou": tc_iou.item(),
            "val/ar_iou": ar_iou.item(),
            "val/mean_iou": mean_iou.item(),
            "val/dice": dice_score,
        }
        self.log_dict(metrics)
        return metrics

    def _shared_eval_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = iou_loss(y_hat, y)
        # loss = F.cross_entropy(y_hat, y)
        bg_iou, tc_iou, ar_iou = iou(y, y_hat)
        dice_score = dice(y_hat, y)
        return loss, bg_iou, tc_iou, ar_iou, dice_score

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())
