"""
Creates a MobileNetV2 Model as defined in:
Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen. (2018).
MobileNetV2: Inverted Residuals and Linear Bottlenecks
arXiv preprint arXiv:1801.04381.
import from https://github.com/tonylins/pytorch-mobilenet-v2
"""

import torch.nn as nn
import torch
import math
import time
# from thop import profile, clever_format
from torchsummary import summary


# __all__ = ['MobileNeXt']


def _make_divisible(v, divisor, min_value=None):
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


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

def group_conv_1x1_bn(inp, oup, expand_ratio):
    hidden_dim = oup // expand_ratio
    return nn.Sequential(
        nn.Conv2d(inp, hidden_dim, 1, 1, 0, groups=hidden_dim, bias=False),
        nn.BatchNorm2d(hidden_dim),
        nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

# sandglass
class SGBlock(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, keep_3x3=False):
        super(SGBlock, self).__init__()
        assert stride in [1, 2]

        hidden_dim = inp // expand_ratio
        if hidden_dim < oup / 6.:
            hidden_dim = math.ceil(oup / 6.)
            hidden_dim = _make_divisible(hidden_dim, 16)# + 16

        #self.relu = nn.ReLU6(inplace=True)
        self.identity = False
        self.identity_div = 1
        self.expand_ratio = expand_ratio
        if expand_ratio == 2:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(inp, inp, 3, 1, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(oup, oup, 3, stride, 1, groups=oup, bias=False),
                nn.BatchNorm2d(oup),
            )
        elif inp != oup and stride == 1 and keep_3x3 == False:
            self.conv = nn.Sequential(
                # pw-linear
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU6(inplace=True),
            )
        elif inp != oup and stride == 2 and keep_3x3 == False:
            self.conv = nn.Sequential(
                # pw-linear
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(oup, oup, 3, stride, 1, groups=oup, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            if keep_3x3 == False:
                self.identity = True
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(inp, inp, 3, 1, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU6(inplace=True),
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                #nn.ReLU6(inplace=True),
                # pw
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(oup, oup, 3, 1, 1, groups=oup, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        out = self.conv(x)
        if self.identity:
            out = out + x
        return out


class MobileNeXt(nn.Module):
    def __init__(self, width_mult=1.):
        super(MobileNeXt, self).__init__()
        # setting of inverted residual blocks
        # self.cfgs = [
        #     # t, c, n, s
        #     [2,  96, 1, 2],
        #     [6, 144, 1, 1],
        #     [6, 192, 3, 2],
        #     [6, 288, 3, 2],
        #     [6, 384, 4, 1],
        #     [6, 576, 4, 2],
        #     [6, 960, 3, 1],
        #     [6,1280, 1, 1],
        # ]
        self.cfgs = [
            # t, c, n, s
            # expand_ratio, output_channel, block repeat, stride
            [1, 48, 1, 1],
            [2, 96, 2, 2],
            [4, 160, 3, 2],
            [4, 196, 4, 2],
            [4, 256, 3, 2]
        ]

        self.features = nn.ModuleList()
        self.stage_chns = []
        # ==> [1] building first stage
        input_channel = _make_divisible(32 * width_mult, 4 if width_mult == 0.1 else 8)
        output_channel = 0
        layers = [conv_3x3_bn(3, input_channel, 2)]
        # building inverted residual blocks
        block = SGBlock
        t, c, n, s = self.cfgs[0]
        output_channel = _make_divisible(c * width_mult, 4 if width_mult == 0.1 else 8)
        if c == 1280 and width_mult < 1:
            output_channel = 1280
        layers.append(block(input_channel, output_channel, s, t, n == 1 and s == 1))
        input_channel = output_channel
        for i in range(n-1):
            layers.append(block(input_channel, output_channel, 1, t))
            input_channel = output_channel
        self.features.append(nn.Sequential(*layers))
        self.stage_chns.append(input_channel)
        # ==> [2] other stages
        for t, c, n, s in self.cfgs[1:]:
            layers = []
            output_channel = _make_divisible(c * width_mult, 4 if width_mult == 0.1 else 8)
            if c == 1280 and width_mult < 1:
                output_channel = 1280
            layers.append(block(input_channel, output_channel, s, t, n == 1 and s == 1))
            input_channel = output_channel
            for i in range(n-1):
                layers.append(block(input_channel, output_channel, 1, t))
                input_channel = output_channel
            self.features.append(nn.Sequential(*layers))
            self.stage_chns.append(input_channel)

        self._initialize_weights()

    def forward(self, x):
        x1 = self.features[0](x)
        x2 = self.features[1](x1)
        x3 = self.features[2](x2)
        x4 = self.features[3](x3)
        x5 = self.features[4](x4)
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
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


if __name__ == "__main__":
    model = MobileNeXt(width_mult=1.0)
    sample = torch.rand([2, 3, 416, 416])
    out = model(sample)
    for x in out:
        print(x.size())
    print("Number of parameters: ", sum([p.numel() for p in model.parameters() if p.requires_grad]))
    start = time.perf_counter()
    for i in range(10):
        out = model(sample)
    end = time.perf_counter()
    dur = (end - start) / 20
    print(dur)
    # summary(model, input_size=(3, 224, 224), device='cpu')
    torch.save(model, './mnext.pth')