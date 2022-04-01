import numpy as np
import cv2
import torch
import math


# opencv函数由四个角点得到最小外接矩形
# coordinates: size(4, 2)
def cv2_get_min_bounding_rect(coordinates: np.ndarray):
    rect = cv2.minAreaRect(coordinates)
    x, y = rect[0]
    w, h = rect[1]
    theta = rect[2] / 180 * math.pi  # 弧度制
    return x, y, w, h, theta


# ======================== numpy函数 ==========================
def corner2bbox_np(corners):
    """
    :param corners: size(8)
    :return: size(4)
    """
    left = np.min(corners[::2])
    top = np.min(corners[1::2])
    right = np.max(corners[::2])
    bottom = np.max(corners[1::2])
    bbox = np.array([left, top, right, bottom])
    return bbox


# ======================== pytorch函数 ========================
# ==> [1] corner to bbox
def corner2bbox_list(corners_list):
    """
    :param corners_list: [tensor, tensor ...]
    :return: list of box, float, not tensor
    """
    bbox_list = []
    for corners in corners_list:
        corners = corners.view(-1)
        left, _ = torch.min(corners[::2], dim=-1)
        top, _ = torch.min(corners[1::2], dim=-1)
        right, _ = torch.max(corners[::2], dim=-1)
        bottom, _ = torch.max(corners[1::2], dim=-1)
        bbox = torch.stack([left, top, right, bottom], dim=-1)
        bbox_list.append(bbox)
    return bbox_list


def corner2bbox_list_int(corners_list):
    """
    :param corners_list: [tensor, tensor ...]
    :return: list of box, int, not tensor
    """
    bbox_list = []
    for corners in corners_list:
        corners = corners.view(-1)
        left, _ = torch.min(corners[::2], dim=-1)
        top, _ = torch.min(corners[1::2], dim=-1)
        right, _ = torch.max(corners[::2], dim=-1)
        bottom, _ = torch.max(corners[1::2], dim=-1)
        bbox = torch.stack([left, top, right, bottom], dim=-1)
        bbox_list.append(bbox.int())
    return bbox_list


def corner2bbox_single(corners):
    """
    :param corners: size(8)
    :return: size(4)
    """
    left, _ = torch.min(corners[::2], dim=-1)
    top, _ = torch.min(corners[1::2], dim=-1)
    right, _ = torch.max(corners[::2], dim=-1)
    bottom, _ = torch.max(corners[1::2], dim=-1)
    bbox = torch.stack([left, top, right, bottom], dim=-1)
    return bbox


def corner2bbox_multi(corners):
    """
    :param corners: size(N, 8)
    :return: size(N, 4)
    """
    left, _ = torch.min(corners[:, ::2], dim=-1)
    top, _ = torch.min(corners[:, 1::2], dim=-1)
    right, _ = torch.max(corners[:, ::2], dim=-1)
    bottom, _ = torch.max(corners[:, 1::2], dim=-1)
    bbox = torch.stack([left, top, right, bottom], dim=-1)
    return bbox


def corner2bbox_tensor(corners_maps):
    """
    :param corners_maps: tensor size(B, H, W, 8)
    :return: size(B, H, W, 4)
    """
    left, _ = torch.min(corners_maps[..., ::2], dim=-1)
    top, _ = torch.min(corners_maps[..., 1::2], dim=-1)
    right, _ = torch.max(corners_maps[..., ::2], dim=-1)
    bottom, _ = torch.max(corners_maps[..., 1::2], dim=-1)
    bbox = torch.stack([left, top, right, bottom], dim=-1)
    return bbox


# ============== scale transformer ==============
def corners_scale_transformer(corners, cur_lvl):
    """
    :param corners: tensor, 3d or 2d or 1d
    :param cur_lvl
    :return:
    """
    corners_trans = corners / (2 ** cur_lvl)
    return corners_trans


def rotatedrect_scale_transformer(rotated_rect, cur_lvl):
    """
    :param rotated_rect: tensor, 3d or 2d or 1d, last dim: (x, y, w, h, theta)
    :param cur_lvl
    :return:
    """
    rotated_rect_trans = rotated_rect.clone()
    if rotated_rect.dim() == 1:
        rotated_rect_trans[:4] = rotated_rect_trans[:4] / (2 ** cur_lvl)
    else:
        rotated_rect_trans[..., :4] = rotated_rect_trans[..., :4] / (2 ** cur_lvl)
    return rotated_rect_trans


if __name__ == '__main__':
    # create a black use numpy,size is:512*512
    img = np.ones((500, 500, 3), np.uint8) * 255
    # coords = np.array([[100, 100], [200, 80], [280, 120], [120, 190]], dtype=np.int)
    coords = np.array([[200, 80], [280, 120], [120, 190], [100, 100]], dtype=np.int)
    rect = cv2.minAreaRect(coords)  # 得到最小外接矩形的（中心(x,y), (宽,高), 旋转角度）
    box = cv2.boxPoints(rect)  # 获取最小外接矩形的4个顶点坐标(ps: cv2.boxPoints(rect) for OpenCV 3.x)
    box = np.int0(box)
    print(rect)
    # 画出来
    cv2.drawContours(img, [coords], 0, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.drawContours(img, [box], 0, (0, 0, 255), 2, cv2.LINE_AA)
    # cv2.imwrite('contours.png', img)
    cv2.imshow('1', img)
    cv2.waitKey()

    # coords = np.array([[100, 100], [200, 80], [280, 120], [120, 190]], dtype=np.int)
    # coords = np.array([ 85.0917,  72.0828, 147.4417,  77.1034, 147.0833,  97.1862,  84.7333,
    #       92.1655], dtype=np.int32)
    # out = cv2_get_min_bounding_rect(coords.reshape(4, 2))
    # print(out)
    # print(type(out[0]))
    # y = float(out[0])
    # print(type(y))