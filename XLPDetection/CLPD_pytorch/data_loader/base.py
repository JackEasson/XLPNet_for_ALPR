# coding=utf-8
import cv2
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

EXTENSIONS = ('.jpg', '.png')


# operate Chinese path problem
def cv_imread(filepath: str):
    cv_img = cv2.imdecode(np.fromfile(filepath, dtype=np.uint8), -1)
    return cv_img


# ========== for ccpd ==========
def read_filenames(folder_path, extension=EXTENSIONS):
    assert type(extension) in (list, tuple)
    names_list = []
    for parent, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            # 后缀判断
            (name, exten) = os.path.splitext(filename)
            if exten in extension:
                names_list.append(os.path.join(folder_path, filename))  # 得到有后缀的名字列表
    return names_list


def read_filenames_from_txt(txt_file_path):
    assert txt_file_path.endswith('.txt')
    with open(txt_file_path, 'r', encoding='utf-8') as f:
        infos = f.readlines()
        infos = [info.strip() for info in infos]
    return infos


# transform the suffix
def trans_extension(filename, target_extension):
    """
    :param filename: just a name without extension, not full path
    :param target_extension:
    :return:
    """
    assert type(target_extension) is str
    none_extension_filename = os.path.splitext(filename)[0]
    target_filename = none_extension_filename + target_extension
    return target_filename


def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2


def clip_corners_np(corners, img_shape):
    # Clip corners to image shape (height, width)
    h, w, _ = img_shape
    corners[:, 0::2] = corners[:, 0::2].clip(0, w)  # x -- width
    corners[:, 1::2] = corners[:, 1::2].clip(0, h)


# image pad and resize, using gray or black padding are all OK.
def letterbox(img, corners, new_shape=(416, 416), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True):
    # bboxes: list[list] or list -- float
    # new_shape: h w
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    # ==========================> trans bboxes <==========================
    h, w = shape
    # new_unpad = (w, h)
    w_ratio, h_ratio = new_unpad[0] / w, new_unpad[1] / h
    # print(new_shape)
    wh_ratio = [w_ratio, h_ratio] * 4
    pad = [left, top] * 4
    if isinstance(corners[0], list) or isinstance(corners[0], np.ndarray):
        new_corners = []
        for corner in corners:
            new_corners.append([x * r + p for (x, r, p) in zip(corner, wh_ratio, pad)])
    else:
        # [print(x, r, p) for (x, r, p) in zip(bboxes, wh_ratio, pad)]
        new_corners = [x * r + p for (x, r, p) in zip(corners, wh_ratio, pad)]
    new_corners_np = np.array(new_corners)
    return img, new_corners_np


def resizebox(img, corners, new_shape=(416, 416)):
    """
    :param img: np.array
    :param corners: list2d/1d or np.ndarray2d/1d
    :param new_shape: (h, w)
    :return:
    """
    h, w, _ = img.shape
    if new_shape[0] == h and new_shape[1] == w:
        return
    new_img = cv2.resize(img, new_shape, interpolation=cv2.INTER_LINEAR)
    r_h, r_w = new_shape[0] / h, new_shape[1] / w
    wh_r = [r_w, r_h] * 4
    if isinstance(corners[0], list) or isinstance(corners[0], np.ndarray):
        new_corners = []
        for corner in corners:
            new_corners.append([x * r for (x, r) in zip(corner, wh_r)])
    else:
        # [print(x, r, p) for (x, r, p) in zip(bboxes, wh_ratio, pad)]
        new_corners = [x * r for (x, r) in zip(corners, wh_r)]
    return new_img, new_corners


# image pad and resize, using gray or black padding are all OK.
def bbox_letterbox(img, bboxes, new_shape=(416, 416), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True):
    # bboxes: list[list] or list -- float
    # new_shape: h w
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    # ==========================> trans bboxes <==========================
    h, w = shape
    # new_unpad = (w, h)
    w_ratio, h_ratio = new_unpad[0] / w, new_unpad[1] / h
    # print(new_shape)
    wh_ratio = [w_ratio, h_ratio] * 2
    pad = [left, top] * 2
    if isinstance(bboxes[0], list) or isinstance(bboxes[0], np.ndarray):
        new_bboxes = []
        for bbox in bboxes:
            new_bboxes.append([x * r + p for (x, r, p) in zip(bbox, wh_ratio, pad)])
    else:
        # [print(x, r, p) for (x, r, p) in zip(bboxes, wh_ratio, pad)]
        new_bboxes = [x * r + p for (x, r, p) in zip(bboxes, wh_ratio, pad)]
    new_bboxes_np = np.array(new_bboxes)
    return img, new_bboxes_np


def bbox_resizebox(img, bboxes, new_shape=(416, 416)):
    """
    :param img: np.array
    :param bboxes: list2d/1d or np.ndarray2d/1d
    :param new_shape: (h, w)
    :return:
    """
    h, w, _ = img.shape
    if new_shape[0] == h and new_shape[1] == w:
        return
    new_img = cv2.resize(img, new_shape, interpolation=cv2.INTER_LINEAR)
    r_h, r_w = new_shape[0] / h, new_shape[1] / w
    wh_r = [r_w, r_h] * 2
    if isinstance(bboxes[0], list) or isinstance(bboxes[0], np.ndarray):
        new_bboxes = []
        for bbox in bboxes:
            new_bboxes.append([x * r for (x, r) in zip(bbox, wh_r)])
    else:
        # [print(x, r, p) for (x, r, p) in zip(bboxes, wh_ratio, pad)]
        new_bboxes = [x * r for (x, r) in zip(bboxes, wh_r)]
    return new_img, new_bboxes


def base_image2tensor(img_mat):
    # img_mat: which has processed with letterbox
    # [1] BGR -> RGB
    img_mat = cv2.cvtColor(img_mat, cv2.COLOR_BGR2RGB).astype(np.float32)
    # [2] normalization
    img_mat = (img_mat / 255.0) * 2.0 - 1.0  # to -1.0 ~ 1.0
    img_mat = img_mat.transpose(2, 0, 1)  # (c, h, w)
    # img_tensor = transforms.ToTensor()(img_mat).float()
    img_tensor = torch.from_numpy(img_mat).float()
    return img_tensor


# return numpy
def decode_name2corner(img_name: str, pure_img_name=True):
    if not pure_img_name:
        img_name = os.path.split(img_name)[-1]
    infos = img_name.split('-')
    corner_info = infos[3]
    br, bl, tl, tr = corner_info.split('_')
    x1, y1 = tl.split('&')
    x2, y2 = tr.split('&')
    x3, y3 = br.split('&')
    x4, y4 = bl.split('&')
    corner = [x1, y1, x2, y2, x3, y3, x4, y4]
    corner = np.array([corner], dtype=np.float32)
    return corner


def decode_name2bbox(img_name: str, pure_img_name=True):
    if not pure_img_name:
        img_name = img_name.split('/')[-1]
    infos = img_name.split('-')
    coord_info, num_info = infos[2], infos[4]
    tl, br = coord_info.split('_')
    x1, y1 = tl.split('&')
    x2, y2 = br.split('&')
    bbox = [x1, y1, x2, y2]
    bbox = np.array([bbox], dtype=np.float32)
    # print(lp_trans_nums2str(nums))
    return bbox


if __name__ == '__main__':
    # x = np.array([[150, 150, 270, 150, 260, 170, 150, 170], [350, 150, 770, 150, 260, 170, 150, 170]], dtype=np.int)
    # img_shape = (250, 250, 3)
    # clip_corners_np(x, img_shape)
    # print(x)
    img_name = '02-1_10-178&467_410&539-410&539_189&535_178&467_399&471-0_0_2_33_31_33_18-80-28'
    print(decode_name2corner(img_name))
