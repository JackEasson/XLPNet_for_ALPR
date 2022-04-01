import torch
import cv2
import numpy as np
import random
from config import AUGMENT


# ================= [1] basic augment ==================
class RandomContrastAndBrightness:
    def __init__(self):
        self.alpha_low, self.alpha_high = 0.25, 1.75
        self.gamma_low, self.gamma_high = -20, 20

    def __call__(self, image, corners):
        alpha = random.uniform(self.alpha_low, self.alpha_high)  # 均匀分布
        beta = 1 - alpha
        gamma = random.randint(self.gamma_low, self.gamma_high)
        blank = np.zeros(image.shape, image.dtype)
        dst = cv2.addWeighted(image, alpha, blank, beta, gamma)
        return dst, corners


class RandomMotionBlur:
    def __init__(self):
        self.degree_l, self.degree_h = 2, 5

    def __call__(self, image, corners):
        degree = random.randint(self.degree_l, self.degree_h)
        angle = random.uniform(-360, 360)
        image = np.array(image)

        # 这里生成任意角度的运动模糊kernel的矩阵， degree越大，模糊程度越高
        M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
        motion_blur_kernel = np.diag(np.ones(degree))
        motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))

        motion_blur_kernel = motion_blur_kernel / degree
        blurred = cv2.filter2D(image, -1, motion_blur_kernel)

        # convert to uint8
        cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
        blurred = np.array(blurred, dtype=np.uint8)
        return blurred, corners


class RandomHSVAugment:
    def __init__(self):
        self.hgain = 0.0138
        self.sgain = 0.678
        self.vgain = 0.36

    def __call__(self, img, corners):
        r = np.random.uniform(-1, 1, 3) * [self.hgain, self.sgain, self.vgain] + 1  # random gains
        hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
        dtype = img.dtype  # uint8

        x = np.arange(0, 256, dtype=np.int16)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
        img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)  # no return needed
        return img, corners


class RandomGaussianNoise:
    def __init__(self):
        self.mean = 0
        self.var_l, self.var_h = 0.001, 0.003

    def __call__(self, img, corners):
        var = random.uniform(self.var_l, self.var_h)
        image = np.array(img / 255, dtype=float)
        noise = np.random.normal(self.mean, var ** 0.5, image.shape)
        out = image + noise
        out = np.clip(out, 0.0, 1.0)
        out = np.uint8(out * 255)
        return out, corners


class RandomResize:
    def __init__(self):
        self.bottom_value = 0.8

    def __call__(self, img, corners):
        h, w, _ = img.shape
        rw = int(w * random.uniform(self.bottom_value, 1))
        rh = int(h * random.uniform(self.bottom_value, 1))

        img = cv2.resize(img, (rw, rh), interpolation=cv2.INTER_LINEAR)
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        return img, corners


# ================= [2] senior augment ==================
class RandomHorizontalFlip(object):

    """Randomly horizontally flips the Image with the probability *p*

    Parameters
    ----------
    p: float
        The probability with which the image is flipped


    Returns
    -------

    numpy.ndaaray
        Flipped image in the numpy format of shape `HxWxC`

    numpy.ndarray
        Tranformed bounding box co-ordinates of the format `n x 4` where n is
        number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box

    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, corners):
        # corners: np.array, size(n, 8)
        h, w, _ = img.shape
        if random.random() < self.p:
            img[:] = img[:, ::-1, :]
            corners[:, 0::2] = w - 1 - corners[:, 0::2]
        return img, corners


# 随机平移
class RandomTranslate(object):
    """Randomly Translates the image


    Bounding boxes which have an area of less than 25% in the remaining in the
    transformed image is dropped. The resolution is maintained, and the remaining
    area if any is filled by black color.

    Parameters
    ----------
    translate: float or tuple(float)
        if **float**, the image is translated by a factor drawn
        randomly from a range (1 - `translate` , 1 + `translate`). If **tuple**,
        `translate` is drawn randomly from values specified by the
        tuple

    Returns
    -------

    numpy.ndaaray
        Translated image in the numpy format of shape `HxWxC`

    numpy.ndarray
        Tranformed bounding box co-ordinates of the format `n x 4` where n is
        number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box

    """

    def __init__(self, diff=False):
        self.diff = diff

    def __call__(self, img, corners):
        # Chose a random digit to scale by
        orig_h, orig_w, c = img.shape

        # corners extreme value
        x1 = max(corners[:, 0::2].min(), 0)
        y1 = max(corners[:, 1::2].min(), 0)
        x2 = min(corners[:, 0::2].max(), orig_w - 1)
        y2 = min(corners[:, 1::2].max(), orig_h - 1)

        # print(x1, y1, x2, y2)
        # translate the image

        # percentage of the dimension of the image to translate
        translate_x = random.randint(-x1, orig_w - 1 - x2)
        translate_y = random.randint(-y1, orig_h - 1 - y2)
        # print(translate_x, translate_y)

        if not self.diff:
            translate_y = translate_x

        canvas = np.zeros(img.shape).astype(np.uint8)

        # change the origin to the top-left corner of the translated box
        orig_box_cords = [max(0, translate_y), max(translate_x, 0), min(orig_h, translate_y + orig_h),
                          min(orig_w, translate_x + orig_w)]

        mask = img[max(-translate_y, 0):min(img.shape[0], -translate_y + orig_h),
                   max(-translate_x, 0):min(img.shape[1], -translate_x + orig_w), :]
        canvas[orig_box_cords[0]:orig_box_cords[2], orig_box_cords[1]:orig_box_cords[3], :] = mask
        img = canvas

        corners[:, :8] += [translate_x, translate_y] * 4

        return img, corners


class RandomRotate(object):
    """Randomly rotates an image


    Bounding boxes which have an area of less than 25% in the remaining in the
    transformed image is dropped. The resolution is maintained, and the remaining
    area if any is filled by black color.

    Parameters
    ----------
    angle: float or tuple(float)
        if **float**, the image is rotated by a factor drawn
        randomly from a range (-`angle`, `angle`). If **tuple**,
        the `angle` is drawn randomly from values specified by the
        tuple

    Returns
    -------

    numpy.ndaaray
        Rotated image in the numpy format of shape `HxWxC`

    numpy.ndarray
        Tranformed bounding box co-ordinates of the format `n x 4` where n is
        number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box

    """

    def __init__(self, angle=10, shrink_scale=(0.5, 1.0)):
        self.angle = angle  # 角度制
        self.shrink_scale = shrink_scale

        if type(self.angle) == tuple:
            assert len(self.angle) == 2, "Invalid range"

        else:
            self.angle = (-self.angle, self.angle)

    def __call__(self, img, corners):
        # corners size(n, 8)

        angle = random.uniform(*self.angle)

        w, h = img.shape[1], img.shape[0]
        cx, cy = w // 2, h // 2

        if type(self.shrink_scale) == tuple:
            scale = random.uniform(*self.shrink_scale)
        else:
            scale = self.shrink_scale
        M = cv2.getRotationMatrix2D((cx, cy), angle, scale)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        # compute the new bounding dimensions of the image
        tar_w = int((h * sin) + (w * cos))
        tar_h = int((h * cos) + (w * sin))

        # adjust the rotation matrix to take into account translation
        M[0, 2] += (tar_w / 2) - cx
        M[1, 2] += (tar_h / 2) - cy

        img = cv2.warpAffine(img, M, (tar_w, tar_h))
        n = corners.shape[0]
        corners_reshape = corners.reshape(-1, 4, 2)
        ones_mat = np.ones([n, 4, 1])
        corners_reshape = np.concatenate([corners_reshape, ones_mat], axis=-1).transpose([0, 2, 1])
        corners = np.matmul(M, corners_reshape).transpose([0, 2, 1]).reshape(-1, 8)
        corners = (corners + 0.5).astype(np.int)
        return img, corners


class ObjectDetectionAugmentation:
    def __init__(self,
                 aug_probs=AUGMENT['aug_probs'],  # 一幅图是否进行增广的概率
                 select_probs=AUGMENT['select_probs'],  # 在多种增广策略中一种策略被选取的概率
                 refresh_probs=AUGMENT['refresh_probs'],  # 增广策略刷新概率
                 max_optor=AUGMENT['max_optor']  # 一次增广中最多使用的策略数
                 ):
        """
        :param aug_probs: if use data augmentation
        :param select_probs: if use current operator
        :param refresh_probs: if refresh aug operators
        :param max_optor: maximum number of operator
        """
        self.aug_probs = aug_probs
        self.select_probs = select_probs
        self.refresh_probs = refresh_probs
        self.max_optor = max_optor
        self.aug_cfg = AUGMENT
        self.aug_operators = self.__initial_operator_sequence()
        self.__current_operator_refresh()

    @staticmethod
    def __random_probs():
        return random.random()  # 用于生成一个0到1的随机符点数: 0.0 <= n < 1.0

    def __initial_operator_sequence(self):
        aug_operators = []
        if self.aug_cfg['contrast']:  # 对比度
            aug_operators.append(RandomContrastAndBrightness())
        if self.aug_cfg['blur']:  # 移动模糊
            aug_operators.append(RandomMotionBlur())
        if self.aug_cfg['hsv']:  # HSV调整
            aug_operators.append(RandomHSVAugment())
        if self.aug_cfg['noise']:  # 高斯噪声
            aug_operators.append(RandomGaussianNoise())
        if self.aug_cfg['flip']:  # 左右翻转
            aug_operators.append(RandomHorizontalFlip(p=0.5))
        if self.aug_cfg['translate']:  # 平移变换
            aug_operators.append(RandomTranslate(diff=True))  # diff=True: x/y移动距离独立
        if self.aug_cfg['rotate']:  # 旋转变换
            aug_operators.append(RandomRotate(angle=10))
        return aug_operators

    def __current_operator_refresh(self):
        self.cur_operators = random.choices(self.aug_operators, k=min(self.max_optor, len(self.aug_operators)))

    # bbox: (x1, y1, x2, y2, c)  => c: class_idx
    def __call__(self, image, corners):
        if self.__random_probs() < self.aug_probs:
            # print(self.cur_operators)
            for op in self.cur_operators:
                if self.__random_probs() < self.select_probs:
                    image, corners = op(image, corners)
        if self.__random_probs() < self.refresh_probs:  # 以一定概率刷新增广操作序列
            self.__current_operator_refresh()
        return image, corners


if __name__ == '__main__':
    pass
