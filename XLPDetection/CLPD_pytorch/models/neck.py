"""
FPN structure for CLPDet.
by Z.W.Edward(2021/12/16)
"""

from models.basic import *


# use nn.ConvTranspose2d for opencv
class StairStepNeck(nn.Module):
    def __init__(self, inp: [list or tuple], oup: int = 128):
        super(StairStepNeck, self).__init__()
        assert len(inp) == 3
        self.inp = inp
        self.postConv1 = PointConv(inp[2], oup)
        self.postConv2 = PointConv(inp[1], oup)
        self.postConv3 = PointConv(inp[0], oup)
        self.upSample1 = DepthWiseConvTranspose(oup, 2)
        self.upSample2 = DepthWiseConvTranspose(oup, 2)
        self.fuse = PointConv(oup, oup)

    def forward(self, x: list):
        x[2] = self.postConv1(x[2])
        x[1] = self.postConv2(x[1]) + self.upSample1(x[2])
        x[0] = self.postConv3(x[0]) + self.upSample2(x[1])
        out = self.fuse(x[0])
        return out


class TinyStairStepNeck(nn.Module):
    def __init__(self, inp: [list or tuple], oup: int = 128):
        super(TinyStairStepNeck, self).__init__()
        assert len(inp) == 2
        self.inp = inp
        self.postConv1 = PointConv(inp[1], oup)
        self.postConv2 = PointConv(inp[0], oup)
        self.upSample1 = DepthWiseConvTranspose(oup, 2)
        self.fuse = PointConv(oup, oup)

    def forward(self, x: list):
        x[1] = self.postConv1(x[1])
        x[0] = self.postConv2(x[0]) + self.upSample1(x[1])
        out = self.fuse(x[0])
        return out


if __name__ == "__main__":
    model = StairStepNeck(inp=(128, 160, 192), oup=128)
    # model = TinyStairStepNeck(inp=(128, 160), oup=128)
    x1 = torch.rand([2, 128, 52, 52])
    x2 = torch.rand([2, 160, 26, 26])
    x3 = torch.rand([2, 192, 13, 13])
    out = model([x1, x2, x3])
    for e in out:
        print(e.size())
    print("Number of parameters: ", sum([p.numel() for p in model.parameters() if p.requires_grad]))
    start = time.perf_counter()
    for i in range(10):
        out = model([x1, x2, x3])
    end = time.perf_counter()
    dur = (end - start) / 20
    print(dur)
    torch.save(model, 'neck.pth')