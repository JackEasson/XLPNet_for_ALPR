import numpy as np
import math
import cv2


# opencv函数由四个角点得到最小外接矩形
# coordinates: size(4, 2)
def cv2_get_min_bounding_rect(coordinates: np.ndarray):
    rect = cv2.minAreaRect(coordinates)
    x, y = rect[0]
    # w, h = np.max(rect[1]), np.min(rect[1])
    w, h = rect[1]
    # theta = -rect[2] if rect[2] > -45.0 else -(rect[2] + 90)
    theta = -rect[2] / 180 * math.pi  # 弧度制
    return x, y, w, h, theta


# rotated 2d gauss generator
# for NMS: ratio=2.0
class Gaussian2DWithRotatedTensor:
    # all real  :  x, y, w, h, theta
    def __init__(self, x, y, w, h, theta, ratio=1.):
        self.x, self.y, self.w, self.h, self.theta, self.ratio = x, y, w, h, theta, ratio
        self.PI = math.pi
        self.sigma = self.__get_sigma()
        self.sigma_v = self.__get_sigma_value()
        self.sigma_i = self.__get_sigma_inverse()

    def __get_sigma(self):
        # ratio_w, ratio_h = self.ratio, self.ratio
        sigma11, sigma22 = self.w * self.ratio, self.h * self.ratio
        sigma12, sigma21 = 0, 0
        sigma = np.array([[sigma11, sigma12], [sigma21, sigma22]], dtype=np.float32)
        rotate_mat = np.array([[np.cos(self.theta), -np.sin(self.theta)], [np.sin(self.theta), np.cos(self.theta)]],
                              dtype=np.float32)
        sigma = np.matmul(np.matmul(rotate_mat.T, sigma), rotate_mat)
        return sigma

    def __get_sigma_value(self):
        v = self.sigma[0, 0] * self.sigma[1, 1] - self.sigma[0, 1] * self.sigma[1, 0]
        v = v if v != 0.0 else 1e-8
        return v

    def __get_sigma_inverse(self):
        return np.array([[self.sigma[1, 1] / self.sigma_v, -self.sigma[1, 0] / self.sigma_v],
                         [-self.sigma[0, 1] / self.sigma_v, self.sigma[0, 0] / self.sigma_v]],
                        dtype=np.float32)

    def reset(self, x, y, w=None, h=None, theta=None, ratio=None):
        self.x, self.y = x, y
        if w is not None:
            self.w = w
        if h is not None:
            self.h = h
        if theta is not None:
            self.theta = theta
        if ratio is not None:
            self.ratio = ratio
        if any([w, h, theta, ratio]):
            self.sigma = self.__get_sigma()
            self.sigma_v = self.__get_sigma_value()
            self.sigma_i = self.__get_sigma_inverse()

    def call_pair(self, x, y):
        reduce_mu = np.array([[x - self.x], [y - self.y]], dtype=np.float32)
        exponent = -0.5 * np.matmul(np.matmul(reduce_mu, self.sigma_i), reduce_mu.T)
        return np.exp(exponent)  # / np.sqrt(2 * self.PI * self.sigma_v)

    def call_points(self, point_xy):
        """
        :param point_xy: array, size(n, 2)  2: [x, y]
        :return: gaussian values vector, size(n)
        """
        error_x = point_xy[:, 0] - self.x  # size(n)
        error_y = point_xy[:, 1] - self.y  # size(n)
        error_sigma = np.power(error_x, 2) * self.sigma_i[0, 0] + \
                      error_x * error_y * (self.sigma_i[0, 1] + self.sigma_i[1, 0]) + \
                      np.power(error_y, 2) * self.sigma_i[1, 1]
        out = np.exp(-0.5 * error_sigma)  # size(h * w)
        return out


class GaussianNMS(Gaussian2DWithRotatedTensor):
    def __init__(self, gauss_ratio=2., nms_thres=0.2, use_distance=False):
        super(GaussianNMS, self).__init__(x=1., y=1., w=1., h=1., theta=0., ratio=gauss_ratio)
        self.nms_thres = nms_thres
        self.use_distance = use_distance

    def gaussian_score_maps_generator(self, gt_corner, pred_corners_map):
        """
        :param gt_corner: size(8)
        :param pred_corners_map: size(n, 8)
        :return:
        """
        gt_corner = gt_corner.reshape(4, 2)
        pred_corners_map = pred_corners_map.reshape(-1, 4, 2)
        gauss_map_list = []
        for i in range(4):
            xy = gt_corner[i]
            x, y = xy[0].item(), xy[1].item()
            self.reset(x, y)
            gauss_score_map = self.call_points(pred_corners_map[:, i, :])  # size(n)
            gauss_map_list.append(gauss_score_map)
        gaussian_score_map = np.stack(gauss_map_list, axis=-1)
        return gaussian_score_map  # size(n, 4)

    @staticmethod
    def corner2bbox(corners_maps):
        """
        :param corners_maps: tensor size(B, H, W, 8)
        :return: size(B, H, W, 4)
        """
        left = np.min(corners_maps[..., ::2], axis=-1)
        top = np.min(corners_maps[..., 1::2], axis=-1)
        right = np.max(corners_maps[..., ::2], axis=-1)
        bottom = np.max(corners_maps[..., 1::2], axis=-1)
        bbox = np.stack([left, top, right, bottom], axis=-1)
        return bbox

    # 计算四个角点距离平方和取平均作为 rho^2
    def gen_distance_quality(self, corners1, corners2):
        """
        :param corners1: size(1, 8), from out_reg_map, real mapping corners
        :param corners2: size(n, 8), real mapping corners
        :return: size(n)
        """
        center_distance = (corners1[..., 0::2] - corners2[..., 0::2]) ** 2 + \
                          (corners1[..., 1::2] - corners2[..., 1::2]) ** 2  # size(n, 4)
        center_distance = np.mean(center_distance, axis=-1)  # size(n)
        n = corners2.shape[0]
        expand_corners1 = np.expand_dims(corners1, axis=0).repeat(n, axis=0)
        union_bbox = self.corner2bbox(np.concatenate([expand_corners1, corners2], axis=-1))
        corner_distance = (union_bbox[..., 2] - union_bbox[..., 0]) ** 2 + \
                          (union_bbox[..., 3] - union_bbox[..., 1]) ** 2
        distance_quality = center_distance / corner_distance
        return distance_quality

    def __call__(self, corners, scores):
        """
        :param corners: size(n, 8)
        :param scores: size(n)
        :return:
        """
        assert corners.ndim == 2 and corners.shape[-1] == 8
        # 降序排列，order下标排序
        order = np.argsort(scores)[::-1]
        # print(_)
        keep = []
        while order.shape[0] > 0:  # 返回元素个数
            if order.shape[0] == 1:  # 保留框只剩一个
                i = order.item()
                keep.append(i)
                break
            else:
                i = order[0].item()  # 保留scores最大的那个框corner[i]
                keep.append(i)
            # 计算corners[i]与其余各框的Gauss Score
            x, y, w, h, theta = cv2_get_min_bounding_rect(corners[i].reshape(4, 2))
            self.reset(x, y, w, h, theta)
            gauss_pred = self.gaussian_score_maps_generator(corners[i], corners[order[1:]])  # return size(n - 1, 4)
            gauss_pred = gauss_pred.mean(-1)  # size(n - 1)
            if not self.use_distance:
                idx = (gauss_pred <= self.nms_thres).nonzero()[0]  # 注意此时idx为[N-1,] 而order为[N,]
            else:
                corners = corners.reshape(-1, 8)
                distance_pred = self.gen_distance_quality(corners[i], corners[order[1:]])  # size(n - 1)
                # print(gauss_pred, distance_pred)
                cplex_quality = gauss_pred - distance_pred
                idx = (cplex_quality <= self.nms_thres).nonzero()[0]
            if idx.shape[0] == 0:
                break
            order = order[idx + 1]  # 修补索引之间的差值
        return np.array(keep, dtype=np.long)  # Pytorch的索引值为LongTensor