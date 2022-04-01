from models.basic import *


class DecoupleHead(nn.Module):
    def __init__(self, inp, reg_num=8, use_sigmoid=True):
        super(DecoupleHead, self).__init__()
        mid_chn = inp // 2
        self.score_branch = nn.Sequential(
            DepSepConv(inp, mid_chn, kernel_size=3, stride=1),
            DepthWiseConv(mid_chn, kernel_size=3),
            nn.Conv2d(mid_chn, reg_num // 2, 1, 1, padding=0, bias=False),
            nn.Sigmoid() if use_sigmoid else nn.Sequential(),
        )
        self.regression_branch = nn.Sequential(
            DepSepConv(inp, mid_chn, kernel_size=3, stride=1),
            DepthWiseConv(mid_chn, kernel_size=3),
            nn.Conv2d(mid_chn, reg_num, 1, 1, padding=0, bias=False),
        )

    def forward(self, x):
        out_score = self.score_branch(x)
        out_reg = self.regression_branch(x)
        return out_score, out_reg


# use ghost module
class UnitedHead(nn.Module):
    def __init__(self, inp, oup=12, use_sigmoid=True):
        super(UnitedHead, self).__init__()
        mid_chn = inp // 2
        self.united_branch = nn.Sequential(
            GhostModule(inp, inp, kernel_size=3, ratio=2),
            GhostModule(inp, mid_chn, kernel_size=3, ratio=2),
            GhostModule(mid_chn, mid_chn // 2, kernel_size=3, ratio=2),
            nn.Conv2d(mid_chn // 2, oup, 1, 1, padding=0, bias=False),
        )
        self.activate = nn.Sigmoid() if use_sigmoid else nn.Sequential()

    def forward(self, x):
        out_score_reg_map = self.united_branch(x)
        out_score_map, out_reg_map = out_score_reg_map[:, :4], out_score_reg_map[:, 4:]
        out_score_map = self.activate(out_score_map)
        return out_score_map, out_reg_map  # score: 4, reg: 8


if __name__ == "__main__":
    model = UnitedHead(128)
    torch.save(model, 'head.pth')