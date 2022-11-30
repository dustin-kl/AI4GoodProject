import numpy as np
import torch
import torch.nn as nn

class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None):
        print("DBF", x.shape)
        x = self.up(x)
        print("DBF post skip", x.shape)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        print("DBF1", x.shape)
        x = self.conv1(x)
        print("DBF2", x.shape)
        x = self.conv2(x)
        return x

class Decoder(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        head_channels = 512
        self.conv_more = Conv2dReLU(
            params["hidden_size"],
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        decoder_channels = params["decoder_channels"]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels

        if self.params["n_skip"] != 0:
            skip_channels = self.params["skip_channels"]
            for i in range(4-self.params["n_skip"]):  # re-select the skip channels according to n_skip
                skip_channels[3-i]=0

        else:
            skip_channels=[0,0,0,0]
        print("skip_channels", skip_channels)
        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)
        self.params = params

    def forward(self, hidden_states, features=None):
        print(hidden_states.shape)
        B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        p_h, p_w = self.params["patches"]["grid"]
        h, w = int(1152/64), int(768/48)
        print("H", h, "W", w)
        print("HS", hidden_states.shape)
        x = hidden_states.permute(0, 2, 1)
        print(x.shape)
        x = x.contiguous().view(B, hidden, h, w)
        x = self.conv_more(x)
        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < self.params["n_skip"]) else None
            else:
                skip = None
            x = decoder_block(x, skip=skip)
        return x