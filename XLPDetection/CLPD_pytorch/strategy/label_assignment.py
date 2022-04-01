import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from strategy.loss_function import VarifocalLoss, GaussDistanceLoss
from utils.gaussian_utils import Gaussian2DWithRotatedTensor
import utils.general as GEL
from strategy.encoder_decoder import *
from utils.debug_tools import print_tensor_2d


class SimCornerOTA:
    def __init__(self, stage_lvl: list, stage_thres: list = None, device='cpu'):
        """
        :param stage_lvl: current feature map's level
        :param stage_thres: thresholds to select positive masks in Gaussian distribution
        """
        assert stage_lvl[-1] <= 5
        self.stage_lvl = stage_lvl  # sample: [3, 4] or [4]
        if stage_thres is not None:
            assert len(stage_thres) == len(stage_lvl)
            self.stage_thres = stage_thres
        else:
            whole_thres = [-1, -1, 0.8, 0.6, 0.4, 0.2]
            self.stage_thres = [whole_thres[i] for i in stage_lvl]

        # loss functions
        # simOTA loss: cls_loss + alpha * iou_loss
        self.score_loss_fun = VarifocalLoss(reduction=None)  # reduction=None -> keep dim
        self.reg_loss_fun = GaussDistanceLoss()
        self.alpha = 3.0
        self.device = device
        self.gauss_generator = Gaussian2DWithRotatedTensor(x=1., y=1., w=1., h=1., theta=0., ratio=1., device=device)

    # single image assignment
    @torch.no_grad()
    def single_assignment(self,
                          pred_score_map3d_list: [..., Tensor],
                          pred_corners_map3d_list: [..., Tensor],
                          gt_corners2d: Tensor,
                          gt_minRect2d: Tensor):
        """
        :param pred_score_map3d_list: corner score output from detnet, size(h, w, 4)
        :param pred_corners_map3d_list: regression decode output (predicted corners) from detnet, size(h, w, 8)
        :param gt_corners2d: corners label, size(n, 8), input scale
        :param gt_minRect2d: min bboxes label from corners, size(n, 5) -> (x, y, w, h, theta), input scale
        :return: target maps and mask maps of one image in a batch
        """
        device = pred_score_map3d_list[0].device
        assert len(pred_score_map3d_list) == len(self.stage_lvl)
        lvl = len(pred_score_map3d_list)
        # object number
        n = gt_corners2d.shape[0]
        tar_score_map_stage, tar_corner_map_stage, tar_reg_map_stage, tar_mask_map_list_stage = [], [], [], []

        grids_map_stage = []
        for lvl_idx in range(lvl):
            h, w, c = pred_score_map3d_list[lvl_idx].shape
            tar_score_map, tar_corner_map, tar_reg_map, tar_mask_map_list = self.target_maps_generator(h, w, c, n, device)
            tar_score_map_stage.append(tar_score_map)
            tar_corner_map_stage.append(tar_corner_map)
            tar_reg_map_stage.append(tar_reg_map)
            tar_mask_map_list_stage.append(tar_mask_map_list)

            # grids
            grids_map = self.generate_grids(h, w)
            grids_map_stage.append(grids_map + 0.5)  # 栅格中心点

        # operate each object in a image, generate primary mask
        for obj_idx in range(n):
            # temp store list
            store_score_map_stage, store_corner_map_stage, store_reg_map_stage, store_mask_map_stage, store_mean_cost_stage = [], [], [], [], []
            # gt_corner: tensor(8); gt_minRect: tensor(5), (x, y, w, h, theta)
            gt_corner, gt_minRect = gt_corners2d[obj_idx], gt_minRect2d[obj_idx]
            # 每个对象在每个预测特征图计算损失，取最小损失预测层为实际负责层
            for lvl_idx in range(lvl):
                h, w, c = pred_score_map3d_list[lvl_idx].shape
                cur_lvl = self.stage_lvl[lvl_idx]
                pred_score_map = pred_score_map3d_list[lvl_idx]
                pred_corners_map = pred_corners_map3d_list[lvl_idx]
                grids_map = grids_map_stage[lvl_idx]
                # ==> [1] corners transform to current level
                pred_corners_cur_map = GEL.corners_scale_transformer(pred_corners_map, cur_lvl)  # current level
                gt_corner_cur = GEL.corners_scale_transformer(gt_corner, cur_lvl)
                gt_minRect_cur = GEL.rotatedrect_scale_transformer(gt_minRect, cur_lvl)
                # ==> [2] primary label assignment
                primary_mask = self.primary_label_assignment(gt_minRect_cur, grids_map, self.stage_thres[lvl_idx])
                # ==== debug ====
                # print('primary_mask', primary_mask.sum().item())
                # print_tensor_2d(primary_mask, "%d_%d.txt" % (obj_idx, lvl_idx))
                # ===============
                # ==> [3] generate auxiliary maps
                aux_score_map, aux_corner_map, aux_reg_map = self.auxiliary_maps_generator(h, w, c, device)
                # aux_score_map, size(h, w, 4)
                tmp_tar_gauss_score_map = self.gaussian_score_maps_generator(gt_corner_cur, pred_corners_cur_map)
                aux_score_map[primary_mask] = tmp_tar_gauss_score_map[primary_mask]
                # aux_corner_map
                # print("Debug: ", aux_corner_map.device, primary_mask.device, gt_corner.device)
                aux_corner_map[primary_mask] = gt_corner
                # aux_reg_map
                grids_map = grids_map_stage[lvl_idx]
                expand_grids_maps = grids_map.unsqueeze(2).repeat(1, 1, 4, 1).reshape(h, w, -1)  # size(h, w, 8)
                tmp_tar_reg_map = corner_encode(gt_corner, expand_grids_maps, self.stage_lvl[lvl_idx])

                aux_reg_map[primary_mask] = tmp_tar_reg_map[primary_mask]
                # ==> [4] calculate cost
                score_cost = self.score_loss_fun(pred_score_map, aux_score_map)
                reg_cost, _ = self.reg_loss_fun(pred_corners_map, aux_corner_map, aux_score_map)
                cplex_cost = score_cost + self.alpha * reg_cost + 100000. * (~primary_mask).float()  # size(h, w)
                sparse_mask = self.dynamic_k_matching(aux_score_map.mean(-1), primary_mask, cplex_cost)
                # store to list
                mean_cost = cplex_cost[sparse_mask].mean()
                store_mean_cost_stage.append(mean_cost)
                store_score_map_stage.append(aux_score_map)
                store_corner_map_stage.append(aux_corner_map)
                store_reg_map_stage.append(aux_reg_map)
                store_mask_map_stage.append(sparse_mask)

            # select the minimum stage
            store_mean_cost_stage_numpy = np.array(store_mean_cost_stage)
            min_lvl_idx = int(np.argmin(store_mean_cost_stage_numpy))
            # store to tar_stage
            score_map = store_score_map_stage[min_lvl_idx]
            corner_map = store_corner_map_stage[min_lvl_idx]
            reg_map = store_reg_map_stage[min_lvl_idx]
            mask_map = store_mask_map_stage[min_lvl_idx]
            tar_score_map_stage[min_lvl_idx][mask_map] = score_map[mask_map]
            tar_corner_map_stage[min_lvl_idx][mask_map] = corner_map[mask_map]
            tar_reg_map_stage[min_lvl_idx][mask_map] = reg_map[mask_map]
            tar_mask_map_list_stage[min_lvl_idx][obj_idx][mask_map] = 1  # just true

        tar_mask_map_stage = []
        positive_sample_num = 0
        for lvl_idx in range(lvl):
            tar_mask_map_list = tar_mask_map_list_stage[lvl_idx]
            # list -> tensor3d
            tar_mask_map = torch.stack(tar_mask_map_list, dim=-1)
            #  remove ambiguous points
            mask_keep = tar_mask_map.int().sum(-1)  # size(h, w)
            mask_keep[mask_keep > 1] = 0
            mask_keep = mask_keep.bool()
            tar_score_map_stage[lvl_idx][~mask_keep] = 0
            tar_corner_map_stage[lvl_idx][~mask_keep] = 0
            tar_reg_map_stage[lvl_idx][~mask_keep] = 0
            tar_mask_map_stage.append(mask_keep)
            positive_sample_num += mask_keep.int().sum().item()
        # positive_sample_num / n: each image has how many positive samples
        positive_samples_mean = positive_sample_num / n
        return tar_score_map_stage, tar_corner_map_stage, tar_reg_map_stage, tar_mask_map_stage, positive_samples_mean

    def generate_grids(self, h, w):
        yv, xv = torch.meshgrid([torch.arange(h), torch.arange(w)])
        grids = torch.stack((xv, yv), 2).to(torch.float32).to(self.device)  # last dim: (x, y)
        return grids

    @staticmethod
    def mask_generator(h, w, device):
        """
        :param h: cls_src_map.shape[0]
        :param w: cls_src_map.shape[1]
        :param device
        :return:
        """
        # just 0 and 1, bool
        mask_aux_map = torch.zeros((h, w), dtype=torch.bool, device=device)
        return mask_aux_map

    @staticmethod
    def auxiliary_maps_generator(h, w, c, device):
        """
        :param h: cls_src_map.shape[0]
        :param w: cls_src_map.shape[1]
        :param c: how many corners
        :param device
        :return:
        """
        aux_score_map = torch.zeros((h, w, c), dtype=torch.float32, device=device)
        aux_corner_map = torch.zeros((h, w, c * 2), dtype=torch.float32, device=device)
        aux_reg_map = torch.zeros((h, w, c * 2), dtype=torch.float32, device=device)
        return aux_score_map, aux_corner_map, aux_reg_map

    @staticmethod
    def target_maps_generator(h, w, c, n, device):
        """
        :param h: cls_src_map.shape[0]
        :param w: cls_src_map.shape[1]
        :param c: cls_src_map.shape[2], also class_num
        :param n: object number of one image
        :param device
        :return:
        """
        tar_score_map = torch.zeros((h, w, c), dtype=torch.float32, device=device)
        tar_corner_map = torch.zeros((h, w, c * 2), dtype=torch.float32, device=device)
        tar_reg_map = torch.zeros((h, w, c * 2), dtype=torch.float32, device=device)
        # use int mask
        tar_mask_map_list = [torch.zeros((h, w), dtype=torch.bool, device=device) for i in range(n)]
        return tar_score_map, tar_corner_map, tar_reg_map, tar_mask_map_list

    # for each object
    def primary_label_assignment(self, gt_min_rect_cur: Tensor, grids_map: Tensor, assign_thres):
        """
        :param gt_min_rect_cur: tensor(5), float, (x, y, w, h, theta), current level scale
        :param grids_map: grids, size(h, w, 2) last dim: (x, y)
        :param assign_thres: current level's threshold, 0, 1, 2 ...
        :return:
        """
        x, y, w, h, theta = gt_min_rect_cur[0].item(), \
                            gt_min_rect_cur[1].item(), \
                            gt_min_rect_cur[2].item(), \
                            gt_min_rect_cur[3].item(), \
                            gt_min_rect_cur[4].item()
        self.gauss_generator.reset(x, y, w, h, theta)
        gauss_map = self.gauss_generator.call_maps(grids_map)
        primary_mask = (gauss_map > assign_thres)
        return primary_mask

    def gaussian_score_maps_generator(self, gt_corner, pred_corners_map):
        """
        :param gt_corner: size(8)
        :param pred_corners_map: size(h, w, 8)
        :return:
        """
        gt_corner = gt_corner.reshape(4, 2)
        h, w, _ = pred_corners_map.shape
        pred_corners_map = pred_corners_map.reshape(h, w, 4, 2)
        gauss_map_list = []
        for i in range(4):
            xy = gt_corner[i]
            x, y = xy[0].item(), xy[1].item()
            self.gauss_generator.reset(x, y)
            gauss_score_map = self.gauss_generator.call_maps(pred_corners_map[:, :, i, :])  # size(h, w)
            gauss_map_list.append(gauss_score_map)
        gaussian_score_map = torch.stack(gauss_map_list, dim=-1)
        return gaussian_score_map  # size(h, w, 4)


    @staticmethod
    def dynamic_k_matching(gauss_mean_score_map, pos_mask, cplex_cost):
        # => [1] get top-10 iou with gt
        # 返回一个元组 (values,indices)，其中indices是原始输入张量input中测元素下标。
        n_candidate_k = min(20, pos_mask.int().sum().item())
        topk_scores, _ = torch.topk(gauss_mean_score_map[pos_mask], n_candidate_k, largest=True, sorted=False)
        dynamic_ks = torch.clamp(topk_scores.sum().int(), min=1)
        cplex_cost_flatten = cplex_cost.reshape(-1)
        _, pos_idx = torch.topk(cplex_cost_flatten, dynamic_ks.item(), largest=False, sorted=False)
        new_mask = torch.zeros_like(pos_mask)
        new_mask_flatten = new_mask.reshape(-1)
        new_mask_flatten[pos_idx] = 1
        new_mask = new_mask_flatten.reshape(new_mask.shape)
        # for debug
        # print_tensor_2d(new_mask, './1.txt')
        return new_mask


class WeightedGaussianAssignment:
    def __init__(self, stage_lvl: int, assign_thres: float = 0.6, gauss_ratio=1.0, device='cpu'):
        """
        :param stage_lvl: current feature map's level
        :param assign_thres: thresholds to select positive masks in Gaussian distribution
        """
        self.stage_lvl = stage_lvl
        self.assign_thres = assign_thres

        # loss functions
        self.score_loss_fun = VarifocalLoss(reduction=None)  # reduction=None -> keep dim
        self.reg_loss_fun = GaussDistanceLoss()
        self.alpha = 3.0
        self.device = device
        self.gauss_generator = Gaussian2DWithRotatedTensor(x=1., y=1., w=1., h=1., theta=0.,
                                                           ratio=gauss_ratio, device=device)

    def single_assignment(self,
                          pred_corners_map: Tensor,
                          gt_corners2d: Tensor,
                          gt_minRect2d: Tensor):
        """
        :param pred_corners_map: regression decode output (predicted corners) from detnet, size(h, w, 8) -- 3d
        :param gt_corners2d: corners label, size(n, 8), input scale
        :param gt_minRect2d: min bboxes label from corners, size(n, 5) -> (x, y, w, h, theta), input scale
        :return: target maps and mask maps of one image in a batch
        """
        # ========== [1] pre-check ========
        device = pred_corners_map.device
        # ========== [2] generate initial targets ========
        h, w, _ = pred_corners_map.shape
        tar_corner_map, tar_weight_map, sample_target_map = self.target_maps_generator(h, w, 4, device)
        gauss_score_map_list = []
        # to generate grids
        grids_map = self.generate_grids(h, w) + 0.5  # 栅格中心点

        # operate each object in a image, generate primary mask
        n = gt_corners2d.shape[0]
        for obj_idx in range(n):
            # gt_corner: tensor(8); gt_minRect: tensor(5), (x, y, w, h, theta)
            gt_corner, gt_minRect = gt_corners2d[obj_idx], gt_minRect2d[obj_idx]

            cur_lvl = self.stage_lvl
            # ==> [1] corners transform to current level
            pred_corners_cur_map = GEL.corners_scale_transformer(pred_corners_map, cur_lvl)  # current level
            gt_corner_cur = GEL.corners_scale_transformer(gt_corner, cur_lvl)
            gt_minRect_cur = GEL.rotatedrect_scale_transformer(gt_minRect, cur_lvl)
            # ==> [2] primary label assignment
            primary_mask, weight_map = self.primary_label_assignment(gt_minRect_cur, grids_map, self.assign_thres)
            # mask
            sample_target_map[primary_mask] += 1  # 计数，最终计数为0的是负样本，1是正样本，2及以上为歧义样本（即忽略样本）
            # score
            tmp_gauss_score_map = self.gaussian_score_maps_generator(gt_corner_cur, pred_corners_cur_map)  # size(h, w, 4)
            gauss_score_map_list.append(tmp_gauss_score_map)

            # corners
            tar_corner_map[primary_mask] = gt_corner
            # weight -- 正样本区域的高斯权重
            tar_weight_map[primary_mask] = weight_map[primary_mask]

        tar_score_map = torch.stack(gauss_score_map_list, dim=-1)  # size(h, w, 4, n)
        tar_score_map, _ = torch.max(tar_score_map, dim=-1)  # size(h, w, 4)
        # sample_target_map: 0->negative; 1->positive; other->ignore
        return tar_score_map, tar_corner_map, tar_weight_map, sample_target_map

    @staticmethod
    def target_maps_generator(h, w, c, device):  # 张量初始化
        """
        :param h: cls_src_map.shape[0]
        :param w: cls_src_map.shape[1]
        :param c: cls_src_map.shape[2], also class_num
        :param device
        :return:
        """
        tar_corner_map = torch.zeros((h, w, c * 2), dtype=torch.float32, device=device)
        tar_weight_map = torch.ones((h, w), dtype=torch.float32, device=device)
        # use int mask
        sample_target_map = torch.zeros((h, w), dtype=torch.int8, device=device)
        return tar_corner_map, tar_weight_map, sample_target_map

    def generate_grids(self, h, w):  # 生成栅格坐标张量
        yv, xv = torch.meshgrid([torch.arange(h), torch.arange(w)])
        grids = torch.stack((xv, yv), 2).to(torch.float32).to(self.device)  # last dim: (x, y)
        return grids

    # 对每个对象GT，高斯得分大于阈值的栅格点定义为初始的正样本
    def primary_label_assignment(self, gt_min_rect_cur: Tensor, grids_map: Tensor, assign_thres):
        """
        :param gt_min_rect_cur: tensor(5), float, (x, y, w, h, theta), current level scale
        :param grids_map: grids, size(h, w, 2) last dim: (x, y) -- centers
        :param assign_thres: current level's threshold, 0, 1, 2 ...
        :return:
        """
        x, y, w, h, theta = gt_min_rect_cur[0].item(), \
                            gt_min_rect_cur[1].item(), \
                            gt_min_rect_cur[2].item(), \
                            gt_min_rect_cur[3].item(), \
                            gt_min_rect_cur[4].item()
        self.gauss_generator.reset(x, y, w, h, theta)
        gauss_map = self.gauss_generator.call_maps(grids_map)
        primary_mask = (gauss_map > assign_thres)
        # if no positive sample
        if primary_mask.int().sum().item() == 0:  # 如果没有正样本，定义GT中心点所在的栅格点为正样本
            y_round, x_round = int(y + 0.5), int(x + 0.5)
            primary_mask[y_round, x_round] = True
            gauss_map[y_round, x_round] = 1.0
        weight_map = gauss_map
        return primary_mask, weight_map

    # 以一组GT角点为基准，在4个角点均生成高斯分布，计算其他预测的角点在其中对应的高斯值
    def gaussian_score_maps_generator(self, gt_corner, pred_corners_map):
        """
        :param gt_corner: size(8)
        :param pred_corners_map: size(h, w, 8)
        :return:
        """
        gt_corner = gt_corner.reshape(4, 2)
        h, w, _ = pred_corners_map.shape
        pred_corners_map = pred_corners_map.reshape(h, w, 4, 2)
        gauss_map_list = []
        for i in range(4):
            xy = gt_corner[i]
            x, y = xy[0].item(), xy[1].item()
            self.gauss_generator.reset(x, y)
            gauss_score_map = self.gauss_generator.call_maps(pred_corners_map[:, :, i, :])  # size(h, w)
            gauss_map_list.append(gauss_score_map)
        gaussian_score_map = torch.stack(gauss_map_list, dim=-1)
        return gaussian_score_map  # size(h, w, 4)


if __name__ == '__main__':
    pass
    # see debug.py
