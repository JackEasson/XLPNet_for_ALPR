# basic
import time
import torch
import torch.nn as nn

# backbone
from models.backbone.efficientv2 import efficientnetv2_lite, efficientnetv2_s
from models.backbone.pplcnet import PPLCNet_x0_35, PPLCNet_x0_5, PPLCNet_x1_0, PPLCNet_x1_5
from models.backbone.mobilenext import MobileNeXt
# neck
from models.neck import StairStepNeck, TinyStairStepNeck
# head
from models.head import DecoupleHead, UnitedHead
# decoder
from strategy.encoder_decoder import corner_decode, corner_decode_onnx_version


class CornerLPNet(nn.Module):
    def __init__(self, backbone='pplcnet'):
        super(CornerLPNet, self).__init__()
        assert backbone in ('pplcnet', 'mobilenext', 'efficientnetv2_lite', 'efficientnetv2_s')
        self.encode_chn = 128
        if backbone == 'pplcnet':
            self.backbone = PPLCNet_x1_0()
        elif backbone == 'mobilenext':
            self.backbone = MobileNeXt(width_mult=1.0)
        elif backbone == 'efficientnetv2_lite':
            self.backbone = efficientnetv2_lite()
        else:
            self.backbone = efficientnetv2_s()
        self.neck = StairStepNeck(inp=self.backbone.stage_chns[2:5], oup=self.encode_chn)
        self.head = UnitedHead(inp=self.encode_chn, oup=12, use_sigmoid=True)

    # out_reg 表示(tx, ty, ...)
    def forward(self, x):
        # 40, 20 / 52, 26
        x3, x4, x5 = self.backbone(x)
        neck_out = self.neck([x3, x4, x5])
        out_score_map, out_reg_map = self.head(neck_out)
        # transpose
        out_score_map = out_score_map.permute(0, 2, 3, 1).contiguous()
        out_reg_map = out_reg_map.permute(0, 2, 3, 1).contiguous()
        return out_score_map, out_reg_map  # size(b, h, w, 4), size(b, h, w, 8)


class CornerLPDet(nn.Module):
    def __init__(self, inp_wh, backbone='pplcnet', device='cpu', export_onnx=False):
        super(CornerLPDet, self).__init__()
        self.inp_wh = inp_wh  # -> int
        self.use_level = 3
        self.map_size = [inp_wh[0] // (2 ** self.use_level), inp_wh[1] // (2 ** self.use_level)]  # [52, 52]
        self.device = device
        self.export_onnx = export_onnx
        # models
        self.lp_det = CornerLPNet(backbone)

    def __basic_generate_grids(self, h, w):
        yv, xv = torch.meshgrid([torch.arange(h), torch.arange(w)])
        grids = torch.stack((xv, yv), 2)  # last dim: (x, y)
        expand_grids = torch.cat([grids] * 4, dim=-1)
        expand_grids = expand_grids.reshape(1, h, w, 8).type(torch.float32).to(self.device)
        return expand_grids

    def __generate_grids(self):
        w, h = self.map_size
        expand_grids = self.__basic_generate_grids(h, w)
        return expand_grids

    def forward(self, x):
        out_score, out_reg = self.lp_det(x)
        expand_grids = self.__generate_grids()
        if not self.export_onnx:
            out_corner = corner_decode(out_reg, expand_grids, self.use_level)
            # reg为实际回归信息，corner为解码后真是角点位置信息
            return out_score, out_corner
        else:
            out_corner = corner_decode_onnx_version(out_reg, expand_grids, self.use_level)
            out = torch.cat([out_score, out_corner], dim=-1)  # (b, 52, 52, 12) score: 4 + corners: 8
            out_3d = out.reshape(-1, self.map_size[0] * self.map_size[1], 12)
            return out_3d



if __name__ == '__main__':
    # net = CornerLPDet(backbone='efficientnetv2').eval()
    # torch.save(net, "./weights/clpnet.pth")
    x = torch.randn(2, 3, 416, 416)
    # net = CornerLPNet(backbone='pplcnet')
    net = CornerLPDet(inp_wh=(416, 416), backbone='mobilenext', device='cpu', export_onnx=True).eval()
    print(net)
    # y = net(x)
    # # print(y.size())
    # # print(y[0].size(), y[1].size())
    # t = time.time()
    # # for i in range(10):
    # for i in range(10):
    #     y = net(x)
    # t = time.time() - t
    # print('Time: %.4f' % (t / 20))
    # # # summary(net, input_size=(3, 320, 320), device='cpu')
    # torch.save(net, "./CLPDet_s.pth")
    # # flops, params = profile(net, inputs=(x, ))
    # # flops, params = clever_format([flops, params], "%.3f")
    # # print("Flops: %s, params: %s" % (flops, params))