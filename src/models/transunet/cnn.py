import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, in_channels):
        super(CNN, self).__init__()

        self.width = 64
    
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
                nn.MaxPool2d(2),
                double_conv(in_channels, out_channels)
            )

        self.down1 = down(32, 64)
        self.down2 = down(64, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 1024)
        self.inc = double_conv(in_channels, 32)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        features = []
        features.append(x2)
        #print("FEAT X2", x2.shape)
        features.append(x3)
        #print("FEAT X3", x3.shape)
        features.append(x4)
        #print("FEAT X4", x4.shape)

        return x5, features[::-1]
