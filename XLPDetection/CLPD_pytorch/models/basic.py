import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
# from thop import profile, clever_format
from torchsummary import summary


# ============= from PP-LCNet =============
def swish(x):
    return x * x.sigmoid()


def hard_sigmoid(x, inplace=False):
    return nn.ReLU6(inplace=inplace)(x + 3) / 6


def hard_swish(x, inplace=False):
    return x * hard_sigmoid(x, inplace)


class HardSigmoid(nn.Module):
    def __init__(self, inplace=False):
        super(HardSigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return hard_sigmoid(x, inplace=self.inplace)


class HardSwish(nn.Module):
    def __init__(self, inplace=False):
        super(HardSwish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return hard_swish(x, inplace=self.inplace)
# ===================================================


class PointConv(nn.Module):
    def __init__(self, inp, oup, hswish=True):
        super(PointConv, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
            HardSwish() if hswish else nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.layers(x)
        return x


# DW + 反卷积
class DepthWiseConvTranspose(nn.Module):
    def __init__(self, inp, kernel_size):
        super(DepthWiseConvTranspose, self).__init__()

        self.dw = nn.Sequential(
            # dw
            nn.ConvTranspose2d(inp, inp, kernel_size, 2, padding=0, groups=inp, bias=False),
            nn.BatchNorm2d(inp),
            HardSwish(),
        )

        self.pw = PointConv(inp, inp, hswish=True)

    def forward(self, x):
        out = self.dw(x)
        out = self.pw(out)
        return out


class DepthWiseConv(nn.Module):
    def __init__(self, inp, kernel_size):
        super(DepthWiseConv, self).__init__()

        padding = (kernel_size - 1) // 2

        self.conv = nn.Sequential(
            # dw
            nn.Conv2d(inp, inp, kernel_size, 1, padding, groups=inp, bias=False),
            nn.BatchNorm2d(inp),
            HardSwish(),
        )

    def forward(self, x):
        return self.conv(x)


class DepSepConv(nn.Module):
    def __init__(self, inp, oup, kernel_size, stride):
        super(DepSepConv, self).__init__()

        assert stride in [1, 2]

        padding = (kernel_size - 1) // 2

        self.conv = nn.Sequential(
            # dw
            nn.Conv2d(inp, inp, kernel_size, stride, padding, groups=inp, bias=False),
            nn.BatchNorm2d(inp),
            HardSwish(),

            # pw-linear
            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
            HardSwish()
        )

    def forward(self, x):
        return self.conv(x)


class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1):
        super(GhostModule, self).__init__()
        self.oup = oup
        middle_channels = oup // ratio
        cheap_channels = oup - middle_channels

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, middle_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(middle_channels),
            HardSwish()
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(middle_channels, cheap_channels, dw_size, 1, dw_size//2, groups=middle_channels, bias=False),
            nn.BatchNorm2d(cheap_channels),
            HardSwish()
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out