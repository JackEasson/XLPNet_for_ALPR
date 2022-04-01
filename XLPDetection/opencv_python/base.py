import os
import cv2
import numpy as np
from gaussian_tools import Gaussian2DWithRotatedTensor, GaussianNMS


EXTENSIONS = ('.jpg', '.png')


def read_filenames(file_path, extension=EXTENSIONS):
    assert type(extension) in (list, tuple)
    names_list = []
    for parent, dirnames, filenames in os.walk(file_path):
        for filename in filenames:
            # 后缀判断
            (name, exten) = os.path.splitext(filename)
            if exten in extension:
                names_list.append(filename)  # 得到有后缀的名字列表
    return names_list


# operate Chinese path problem
def cv_imread(filepath: str):
    cv_img = cv2.imdecode(np.fromfile(filepath, dtype=np.uint8), -1)
    return cv_img


# ============================ [1] pre-process ===========================
def letterbox(img, new_shape=(416, 416), color=(114, 114, 114)):
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize, shape[::-1] -> (w, h)
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border

    return img


def pre_process(img_mat, new_shape=(416, 416), use_letterbox=True, color=(0, 0, 0)):
    if use_letterbox:
        new_img = letterbox(img=img_mat, new_shape=new_shape, color=color)
    else:
        new_img = cv2.resize(img_mat, new_shape, interpolation=cv2.INTER_LINEAR)
    return new_img


# ====================== post process =====================
class PostProcessor:
    def __init__(self, conf_thres=0.4, nms_thres=0.1, max_obj=50, use_distance=False):
        self.conf_thres = conf_thres
        self.nms_thres = nms_thres
        self.max_obj = max_obj
        self.gauss_nms = GaussianNMS(gauss_ratio=2., nms_thres=nms_thres, use_distance=use_distance)

    def __call__(self, net_out):
        score_map, corner_map = net_out[..., :4], net_out[..., 4:]
        score_map = np.mean(score_map, axis=-1)
        mask = score_map > self.conf_thres
        score_keep, corner_keep = score_map[mask], corner_map[mask]  # size(n), size(n, 8)
        order = np.argsort(score_keep)[::-1]
        score_keep = score_keep[order]
        corner_keep = corner_keep[order]
        if score_keep.shape[0] > self.max_obj:
            score_keep, corner_keep = score_keep[:self.max_obj], corner_keep[:self.max_obj]
        score_corner_keep = np.concatenate([np.expand_dims(score_keep, axis=-1), corner_keep], axis=-1)
        keep_index = self.gauss_nms(corner_keep, score_keep)
        nms_out = score_corner_keep[keep_index]  # size(n', 9) last_dim->(score_map_candidate, corner_map_candidate)
        return nms_out



