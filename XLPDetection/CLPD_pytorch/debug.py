import cv2
import torch
import numpy as np
from XLPDet_pipleline import CornerLPDet
from strategy.label_assignment import SimCornerOTA, WeightedGaussianAssignment
from utils.debug_tools import print_tensor_2d
from utils.general import cv2_get_min_bounding_rect
from data_loader.augmentation import *
from utils.plot_tools import plot_polygon_bbox


def debugAssigner():
    x = torch.randn(1, 3, 416, 416)
    net = CornerLPDet(inp_wh=(416, 416), backbone='efficientnetv2_lite', device='cpu', export_onnx=False)
    out_score, out_corner = net(x)
    # simota = SimCornerOTA(stage_lvl=[3, 4], stage_thres=[0.6, 0.6], device='cuda')
    assigner = WeightedGaussianAssignment(stage_lvl=3, assign_thres=0.5, device='cpu')
    # corners_np = np.array([[100, 100, 200, 100, 200, 150, 100, 150], [200, 200, 250, 200, 250, 230, 200, 230], [50, 50, 400, 50, 400, 400, 50, 400]], dtype=np.float32)
    corners_np = np.array([[147, 174, 209, 179, 209, 199, 147, 194]], dtype=np.float32)
    # corners_np_4x2 = corners_np.reshape(2, 4, 2)
    minRect_list = []
    for i in range(corners_np.shape[0]):
        corners_cur = corners_np[i].reshape(4, 2)
        minRect_list.append(cv2_get_min_bounding_rect(corners_cur))
    minRect_np = np.stack(minRect_list, axis=0)
    corners_tensor = torch.from_numpy(corners_np).to(torch.float32)
    minRect_tensor = torch.from_numpy(minRect_np).to(torch.float32)
    tar_score_map, tar_corner_map, tar_mask_map, tar_ignore_map, tar_weight_map = \
        assigner.single_assignment(out_corner[0], corners_tensor, minRect_tensor)
    print(tar_score_map.size())
    print_tensor_2d(tar_weight_map, 'mask1.txt')
    # print_tensor_2d(tar_weight_map_stage[1], 'mask2.txt')
    # mask2 = tar_mask_map_stage[1]
    # print(tar_score_map_stage[1][mask2])
    # print(tar_corner_map_stage[1][mask2])
    # print(tar_reg_map_stage[1][mask2])
    # print((tar_score_map_stage[1] != 0).sum().item())


def debugAugment():
    img = cv2.imread("E:\\images\\1.jpg")
    corners_np = np.array([[100, 100, 200, 100, 200, 150, 100, 150],
                           [250, 50, 650, 100, 630, 450, 240, 400]], dtype=np.float32)
    img2 = img.copy()
    plot_polygon_bbox(img, corners_np)
    cv2.imshow('1', img)
    fx = RandomHorizontalFlip(1.0)
    fx2 = RandomTranslate(diff=True)
    fx3 = RandomRotate(angle=20, shrink_scale=1.0)
    img2, corners_np2 = fx3(img2, corners_np)
    # print(corners_np)
    # print(corners_np2)
    plot_polygon_bbox(img2, corners_np2)
    cv2.imshow('2', img2)
    cv2.waitKey()

if __name__ == '__main__':
    debugAssigner()
    # debugAugment()

