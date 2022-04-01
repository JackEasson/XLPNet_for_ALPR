import torch
import torch.nn as nn
from torch import Tensor
import math
import utils.general as GEL


# from <VarifocalNet: An IoU-aware Dense Object Detector>
# for 4 corners' score map
class VarifocalLoss(nn.Module):
    def __init__(self, reduction=None, eps=1e-9):
        super(VarifocalLoss, self).__init__()
        assert reduction in (None, 'mean', 'sum')  # 'none' is to keep dims
        self.reduction = reduction
        self.eps = eps
        self.alpha, self.beta = 0.25, 2

    def forward(self, out_score_map, target_score_map):
        """
        :param out_score_map: tensor(h, w, 4) or tensor(b, h, w, 4)
        :param target_score_map: size same to out_score_map
        :return:
        """
        positive_mask = (target_score_map != 0.)
        p, q = out_score_map, target_score_map
        positive_maps = -q * (q * torch.log(p + self.eps) + (1 - q) * torch.log(1 - p + self.eps))
        negative_maps = -self.alpha * p ** self.beta * torch.log(1 - p + self.eps)
        varifocal_loss = torch.where(positive_mask, positive_maps, negative_maps)
        varifocal_loss = torch.mean(varifocal_loss, dim=-1)  # size(h, w) or size(b, h, w)
        if self.reduction is None:
            return varifocal_loss
        elif self.reduction == 'mean':
            return torch.mean(varifocal_loss)  # size(1)
        else:
            return torch.sum(varifocal_loss)  # size(1)


def focal_loss_for_score(sample_target, score_out, score_target, gamma=2, alpha=0.25, delta=1e-16):
    """
    :param sample_target: positive and negative sample, size(B, H, W)
    :param score_out: score, size(B, H, W, 4), predicted iou score
    :param score_target: score, size(B, H, W, 4), iou score
    :param gamma for focal loss
    :param alpha for focal loss
    :param delta for log
    :return:
    """
    # positive
    difference_score = torch.abs(score_out - score_target)  # size(B, H, W)
    y_pos = torch.clamp(difference_score, max=1.0 - delta)
    pos_focal_loss = -alpha * y_pos ** gamma * torch.log(torch.ones_like(y_pos).float() - y_pos)
    pos_focal_loss = torch.sum(pos_focal_loss[sample_target == 1]) / torch.sum(sample_target == 1)
    # print(pos_focal_loss)
    # negative
    y_neg = torch.clamp(score_out, max=1.0 - delta)
    neg_focal_loss = -(1.0 - alpha) * y_neg ** gamma * torch.log(torch.ones_like(y_neg).float() - y_neg)
    neg_focal_loss = torch.sum(neg_focal_loss[sample_target == 0]) / torch.sum(sample_target == 0)
    # total focal loss
    return pos_focal_loss + neg_focal_loss


class ScoreFocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25, eps=1e-12):
        super(ScoreFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.eps = eps

    # delete ignore areas
    def forward(self, sample_target, score_out, score_target):
        """
        :param sample_target: positive and negative sample, size(B, H, W)
        :param score_out: score, size(B, H, W, 4), predicted iou score
        :param score_target: score, size(B, H, W, 4), iou score
        """
        # positive
        difference_score = torch.abs(score_out - score_target)  # size(B, H, W)
        y_pos = torch.clamp(difference_score, max=1.0 - self.eps)
        pos_focal_loss = -self.alpha * y_pos ** self.gamma * torch.log(1.0 - y_pos)
        pos_focal_loss = torch.mean(pos_focal_loss[sample_target == 1])
        # print(pos_focal_loss)
        # negative
        y_neg = torch.clamp(score_out, max=1.0 - self.eps)
        neg_focal_loss = -(1.0 - self.alpha) * y_neg ** self.gamma * torch.log(1.0 - y_neg)
        neg_focal_loss = torch.mean(neg_focal_loss[sample_target == 0])
        # total focal loss
        return pos_focal_loss + neg_focal_loss


# ------------------------------------------------------------------
# ------- [1] MultiConstraintsGaussDistanceLoss from SLPNet --------
# ------------------------------------------------------------------
class GaussDistanceLoss(nn.Module):
    def __init__(self):
        super(GaussDistanceLoss, self).__init__()

    @staticmethod
    # 计算四个角点距离平方和取平均作为 rho^2
    def gen_distance_map(out_corners_map, gt_corners_map):
        """
        :param out_corners_map: size(B, H, W, 8), from out_reg_map, real mapping corners
        :param gt_corners_map: size(B, H, W, 8), real mapping corners
        :return: size(B, H, W)
        """
        center_distance = (out_corners_map[..., 0::2] - gt_corners_map[..., 0::2]) ** 2 + \
                          (out_corners_map[..., 1::2] - gt_corners_map[..., 1::2]) ** 2  # size(B, H, W, 4)
        center_distance = torch.mean(center_distance, dim=-1)  # size(B, H, W)
        union_bbox = GEL.corner2bbox_tensor(torch.cat([out_corners_map, gt_corners_map], dim=-1))
        corner_distance = (union_bbox[..., 2] - union_bbox[..., 0]) ** 2 + \
                          (union_bbox[..., 3] - union_bbox[..., 1]) ** 2
        distance_maps = center_distance / corner_distance
        return distance_maps

    @staticmethod
    def gen_whwh_map(out_corners_map, eps=1e-9):
        # coordinate transform to (w1, h1, w2, h2), two bbox size
        """
        :param out_corners_map: size(B, H, W, 8)
        :param eps
        :return:
        """
        w1 = torch.abs(out_corners_map[..., 4] - out_corners_map[..., 0])
        w1 = torch.clamp(w1, min=eps)
        h1 = torch.abs(out_corners_map[..., 5] - out_corners_map[..., 1])
        h1 = torch.clamp(h1, min=eps)
        w2 = torch.abs(out_corners_map[..., 2] - out_corners_map[..., 6])
        w2 = torch.clamp(w2, min=eps)
        h2 = torch.abs(out_corners_map[..., 7] - out_corners_map[..., 3])
        h2 = torch.clamp(h2, min=eps)
        whwh_maps = torch.stack([w1, h1, w2, h2], dim=-1)
        return whwh_maps

    def forward(self, out_corners_map, gt_corners_map, real_gaussian_score_map):
        """
        :param out_corners_map: size(B, H, W, 8), from out_reg_map, real mapping corners
        :param gt_corners_map: size(B, H, W, 8), real mapping corners
        :param real_gaussian_score_map: size(B, H, W)
        :return: MG loss, size(B, H, W)
        """
        # [1] gauss loss
        real_score_map_mean = torch.mean(real_gaussian_score_map, dim=-1)
        loss_gauss = 1 - real_score_map_mean
        # loss_gauss = torch.mean(-torch.log(real_gaussian_score_map + 1e-9), dim=-1)
        # [2] distance
        loss_distance = self.gen_distance_map(out_corners_map, gt_corners_map)
        # [3] size
        # alpha * V
        whwh_out = self.gen_whwh_map(out_corners_map)
        whwh_gt = self.gen_whwh_map(gt_corners_map)
        size_target = (torch.atan(whwh_out[..., 0] / whwh_out[..., 1]) -
                       torch.atan(whwh_gt[..., 0] / whwh_gt[..., 1])) ** 2 + \
                      (torch.atan(whwh_out[..., 2] / whwh_out[..., 3]) -
                       torch.atan(whwh_gt[..., 2] / whwh_gt[..., 3])) ** 2
        loss_v = size_target * 2 / (math.pi ** 2)
        alpha = loss_v.detach() / (loss_gauss.detach() + loss_v.detach())
        loss_size = alpha * loss_v

        # united loss to output
        united_loss = loss_gauss + loss_distance

        return united_loss, (loss_gauss, loss_distance, loss_size)


def smooth_l1_loss(input, target, sigma=1.0, reduce=True, normalizer=1.0):
    beta = 1. / (sigma ** 2)
    diff = torch.abs(input - target)
    cond = diff < beta
    loss = torch.where(cond, 0.5 * diff ** 2 / beta, diff - 0.5 * beta)
    if reduce:
        return torch.sum(loss) / normalizer
    return torch.mean(loss, dim=-1) / normalizer


if __name__ == '__main__':
    x = torch.zeros((26, 26, 4), dtype=torch.float32)
    y = torch.ones((26, 26, 4), dtype=torch.float32) * 0.01
    VL = VarifocalLoss()
    z = VL(x, y)
    print(z)