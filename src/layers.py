import torch.nn as nn


class SeparableConv2d(nn.Module):
    def __init__(
        self,
        inplanes,
        planes,
        kernel_size=3,
        stride=1,
        padding=0,
        dilation=1,
        bias=False,
    ):
        super(SeparableConv2d, self).__init__()
        self.conv1 = nn.Conv2d(
            inplanes,
            inplanes,
            kernel_size,
            stride,
            padding,
            dilation,
            groups=inplanes,
            bias=bias,
        )
        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        inplanes,
        planes,
        reps,
        stride=1,
        dilation=1,
        start_with_relu=True,
        grow_first=True,
        is_last=False,
    ):
        super(Block, self).__init__()

        if planes != inplanes or stride != 1:
            self.skip = nn.Conv2d(inplanes, planes, 1, stride=stride, bias=False)
            self.skipbn = nn.BatchNorm2d(planes)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        rep = []

        filters = inplanes
        if grow_first:
            rep.append(self.relu)
            rep.append(
                SeparableConv2d(inplanes, planes, 3, stride=1, dilation=dilation)
            )
            rep.append(nn.BatchNorm2d(planes))
            filters = planes

        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(
                SeparableConv2d(filters, filters, 3, stride=1, dilation=dilation)
            )
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(
                SeparableConv2d(inplanes, planes, 3, stride=1, dilation=dilation)
            )
            rep.append(nn.BatchNorm2d(planes))

        if not start_with_relu:
            rep = rep[1:]

        if stride != 1:
            rep.append(SeparableConv2d(planes, planes, 3, stride=2))

        if stride == 1 and is_last:
            rep.append(SeparableConv2d(planes, planes, 3, stride=1))

        self.rep = nn.Sequential(*rep)
