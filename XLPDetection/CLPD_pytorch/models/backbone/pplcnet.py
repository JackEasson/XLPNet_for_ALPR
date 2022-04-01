"""
Creates a PP-LCNet Model as defined in:
C. Cui. T. Gao, S. Wei et al (2021).
PP-LCNet: A Lightweight CPU Convolutional Neural Network
https://arxiv.org/pdf/2109.15099.pdf.
import from https://github.com/ngnquan/PP-LCNet

Adjust LCNet's structure for AtomDet.
by Z.W. Edward(2021/10/30)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
from models.basic import *
# from thop import profile, clever_format
from torchsummary import summary

__all___ = [
    'PPLCNet_x0_25', 'PPLCNet_x0_35', 'PPLCNet_x0_5', 'PPLCNet_x0_75',
    'PPLCNet_x1_0', 'PPLCNet_x1_5', 'PPLCNet_x2_0', 'PPLCNet_x2_5'
]


def _make_divisible(v, divisor=8, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class SELayer(nn.Module):
    def __init__(self, inp, oup, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(oup, _make_divisible(inp // reduction), 1, 1, 0, ),
            nn.ReLU(),
            nn.Conv2d(_make_divisible(inp // reduction), oup, 1, 1, 0),
            HardSigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class DepSepConv(nn.Module):
    def __init__(self, inp, oup, kernel_size, stride, use_se):
        super(DepSepConv, self).__init__()

        assert stride in [1, 2]

        padding = (kernel_size - 1) // 2

        if use_se:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(inp, inp, kernel_size, stride, padding, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                HardSwish(),

                # SE
                SELayer(inp, inp),

                # pw-linear
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                HardSwish(),

            )
        else:
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


"""
adjust:
[1] stage split to output
[2] no se
"""
class PPLCNet(nn.Module):
    def __init__(self, scale=1.0):
        super(PPLCNet, self).__init__()
        """
        self.cfgs = [
            # k,  c,  s, SE
            [3, 32, 1, 0],

            [3, 64, 2, 0],
            [3, 64, 1, 0],

            [3, 128, 2, 0],
            [3, 128, 1, 0],

            [5, 256, 2, 0],
            [5, 256, 1, 0],
            [5, 256, 1, 0],
            [5, 256, 1, 0],
            [5, 256, 1, 0],
            [5, 256, 1, 0],

            [5, 512, 2, 1],
            [5, 512, 1, 1],
        ]"""

        self.stage1_cfgs = [
            # k,  c,  s, SE
            [3, 32, 1, 0],
        ]
        self.stage2_cfgs = [
            # k,  c,  s, SE
            [3, 64, 2, 0],
            [5, 64, 1, 0],
        ]
        self.stage3_cfgs = [
            # k,  c,  s, SE
            [5, 128, 2, 0],
            [3, 128, 1, 0],
            [5, 128, 1, 0],
        ]
        self.stage4_cfgs = [
            # k,  c,  s, SE
            [5, 196, 2, 0],
            [3, 196, 1, 0],
            [3, 196, 1, 0],
            [5, 196, 1, 0],
        ]
        self.stage5_cfgs = [
            # k,  c,  s, SE
            [5, 256, 2, 0],
            [5, 256, 1, 0],
            [5, 256, 1, 0],
        ]

        self.scale = scale
        block = DepSepConv
        self.stage_chns = []
        # ========> stage1 <=======
        input_channel = _make_divisible(16 * scale)
        layers = [nn.Conv2d(3, input_channel, 3, 2, 1, bias=False), HardSwish()]

        for k, c, s, use_se in self.stage1_cfgs:
            output_channel = _make_divisible(c * scale)
            layers.append(block(input_channel, output_channel, k, s, use_se))
            input_channel = output_channel
        self.features1 = nn.Sequential(*layers)
        self.stage_chns.append(input_channel)

        # ========> stage2 <=======
        layers = []
        for k, c, s, use_se in self.stage2_cfgs:
            output_channel = _make_divisible(c * scale)
            layers.append(block(input_channel, output_channel, k, s, use_se))
            input_channel = output_channel
        self.features2 = nn.Sequential(*layers)
        self.stage_chns.append(input_channel)

        # ========> stage3 <=======
        layers = []
        for k, c, s, use_se in self.stage3_cfgs:
            output_channel = _make_divisible(c * scale)
            layers.append(block(input_channel, output_channel, k, s, use_se))
            input_channel = output_channel
        self.features3 = nn.Sequential(*layers)
        self.stage_chns.append(input_channel)

        # ========> stage4 <=======
        layers = []
        for k, c, s, use_se in self.stage4_cfgs:
            output_channel = _make_divisible(c * scale)
            layers.append(block(input_channel, output_channel, k, s, use_se))
            input_channel = output_channel
        self.features4 = nn.Sequential(*layers)
        self.stage_chns.append(input_channel)

        # ========> stage5 <=======
        layers = []
        for k, c, s, use_se in self.stage5_cfgs:
            output_channel = _make_divisible(c * scale)
            layers.append(block(input_channel, output_channel, k, s, use_se))
            input_channel = output_channel
        self.features5 = nn.Sequential(*layers)
        self.stage_chns.append(input_channel)

        self._initialize_weights()

    def forward(self, x):
        x1 = self.features1(x)
        x2 = self.features2(x1)
        x3 = self.features3(x2)
        x4 = self.features4(x3)
        x5 = self.features5(x4)
        return x3, x4, x5

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.001)
                m.bias.data.zero_()


def PPLCNet_x0_25():
    """
    Constructs PPLCNet_x0_25 model
    """
    model = PPLCNet(scale=0.25)

    return model


def PPLCNet_x0_35(**kwargs):
    """
    Constructs PPLCNet_x0_35 model
    """
    model = PPLCNet(scale=0.35)

    return model


def PPLCNet_x0_5(**kwargs):
    """
    Constructs PPLCNet_x0_5 model
    """
    model = PPLCNet(scale=0.5)

    return model


def PPLCNet_x0_75(**kwargs):
    """
    Constructs PPLCNet_x0_75 model
    """
    model = PPLCNet(scale=0.75)

    return model


def PPLCNet_x1_0(**kwargs):
    """
    Constructs PPLCNet_x1_0 model
    """
    model = PPLCNet(scale=1.0)

    return model


def PPLCNet_x1_5(**kwargs):
    """
    Constructs PPLCNet_x1_5 model
    """
    model = PPLCNet(scale=1.5)

    return model


def PPLCNet_x2_0(**kwargs):
    """
    Constructs PPLCNet_x2_0 model
    """
    model = PPLCNet(scale=2.0)

    return model


def PPLCNet_x2_5(**kwargs):
    """
    Constructs PPLCNet_x2_5 model
    """
    model = PPLCNet(scale=2.5)

    return model


if __name__ == "__main__":
    model = PPLCNet_x1_0()
    print(model.stage_chns)
    sample = torch.rand([2, 3, 416, 416])
    out = model(sample)
    for i in out:
        print(i.size())
    # print("Number of parameters: ", sum([p.numel() for p in model.parameters() if p.requires_grad]))
    start = time.perf_counter()
    for i in range(10):
        out = model(sample)
    end = time.perf_counter()
    dur = (end - start) / 20
    print(dur)
    # summary(model, input_size=(3, 320, 320), device='cpu')
    # torch.save(model, "./pplcnet.pth")

