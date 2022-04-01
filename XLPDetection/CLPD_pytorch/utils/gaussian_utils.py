"""
gauss相关的功能类/函数
reference: https://blog.csdn.net/lin_limin/article/details/81024228
by Z.W.Edward(2021/12/16)
"""
import torch
import math
import cv2
import numpy as np
from utils.general import cv2_get_min_bounding_rect, corner2bbox_tensor


# normalized 2d gauss
def norm_gauss_2d(x1, x2, u1, u2, d1, d2):
    m = torch.pow((x1 - u1) / d1, 2)
    n = torch.pow((x2 - u2) / d2, 2)
    return torch.exp(-0.5 * (m + n))


# rotated 2d gauss generator
class Gaussian2DWithRotatedNumpy:
    # all real  :  x, y, w, h, theta
    def __init__(self, x, y, w, h, theta, ratio=1.):
        self.x, self.y, self.w, self.h, self.theta, self.ratio = x, y, w, h, theta, ratio
        self.PI = math.pi
        self.sigma = self.__get_sigma()
        self.sigma_v = self.__get_sigma_value()
        self.sigma_i = self.__get_sigma_inverse()

    def __get_sigma(self):
        ratio_w, ratio_h = self.ratio, self.ratio
        sigma11, sigma22 = self.w * ratio_w, self.h * ratio_h
        sigma12, sigma21 = 0, 0
        sigma = np.array([[sigma11, sigma12], [sigma21, sigma22]])
        rotate_mat = np.array([[np.cos(self.theta), -np.sin(self.theta)], [np.sin(self.theta), np.cos(self.theta)]])
        sigma = np.matmul(np.matmul(rotate_mat, sigma), rotate_mat.T)
        return sigma

    def __get_sigma_value(self):
        v = self.sigma[0, 0] * self.sigma[1, 1] - self.sigma[0, 1] * self.sigma[1, 0]
        v = v if v != 0.0 else 1e-8
        return v

    def __get_sigma_inverse(self):
        return np.array([[self.sigma[1, 1] / self.sigma_v, -self.sigma[1, 0] / self.sigma_v],
                         [-self.sigma[0, 1] / self.sigma_v, self.sigma[0, 0] / self.sigma_v]])

    def __call__(self, x, y):
        X_reduce_mu = np.array([[x - self.x], [y - self.y]])
        exponent = -0.5 * np.matmul(np.matmul(X_reduce_mu.T, self.sigma_i), X_reduce_mu)
        return np.exp(exponent)  # / np.sqrt(2 * self.PI * self.sigma_v)


# rotated 2d gauss generator
class Gaussian2DWithRotatedTensor:
    # all real  :  x, y, w, h, theta
    def __init__(self, x, y, w, h, theta, ratio=1., device='cpu'):
        """
        Parameters
        ----------
        x, y, w, h, theta: GT四边形最小外接矩形的中心点横、纵坐标，宽、高、旋转弧度
        ratio: 缩放因子
        device:
        """
        self.x, self.y, self.w, self.h, self.theta, self.ratio = x, y, w, h, theta, ratio
        self.device = device
        self.PI = math.pi
        self.zeta = 25
        self.sigma = self.__get_sigma()
        self.sigma_v = self.__get_sigma_value()
        self.sigma_i = self.__get_sigma_inverse()

    def __get_sigma(self):
        # ratio_w, ratio_h = self.ratio, self.ratio
        sigma11, sigma22 = self.w * self.ratio, self.h * self.ratio
        sigma12, sigma21 = 0, 0
        sigma = torch.tensor([[sigma11, sigma12], [sigma21, sigma22]], dtype=torch.float32, device=self.device)
        rotate_mat = torch.tensor([[np.cos(self.theta), -np.sin(self.theta)], [np.sin(self.theta), np.cos(self.theta)]],
                                  dtype=torch.float32, device=self.device)
        sigma = torch.matmul(torch.matmul(rotate_mat, sigma), rotate_mat.T)
        return sigma

    def __get_sigma_value(self):
        v = self.sigma[0, 0] * self.sigma[1, 1] - self.sigma[0, 1] * self.sigma[1, 0]
        v = v if v != 0.0 else 1e-8
        return v

    def __get_sigma_inverse(self):  # 求sigma的逆矩阵
        return torch.tensor([[self.sigma[1, 1] / self.sigma_v, -self.sigma[1, 0] / self.sigma_v],
                            [-self.sigma[0, 1] / self.sigma_v, self.sigma[0, 0] / self.sigma_v]],
                            dtype=torch.float32, device=self.device)

    # 重新设置输入参数
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

    # 求一个点的高斯值
    def call_pair(self, x, y):
        reduce_mu = torch.tensor([[x - self.x], [y - self.y]], dtype=torch.float32, device=self.device)
        exponent = -0.5 * torch.matmul(torch.matmul(reduce_mu, self.sigma_i), reduce_mu.T)
        return torch.exp(exponent)  # / np.sqrt(2 * self.PI * self.sigma_v)  # 不需要系数，因为是归一化高斯

    # 求一个张量的高斯值，3维或者4维均可
    def call_maps(self, call_maps):
        """
        :param call_maps: size(B, H, W, 2)  2: [x, y]
        :return: gaussian values map, size(h, w)
        """
        error_x = call_maps[..., 0] - self.x  # size(h, w)
        error_y = call_maps[..., 1] - self.y  # size(h, w)
        # 利用Tensor的广播机制来计算
        error_sigma = torch.pow(error_x, 2) * self.sigma_i[0, 0] + error_x * error_y * (self.sigma_i[0, 1] + self.sigma_i[1, 0]) + \
                      torch.pow(error_y, 2) * self.sigma_i[1, 1]
        out = torch.exp(-0.5 * error_sigma)  # size(h * w)
        return out


class GaussianNMS(Gaussian2DWithRotatedTensor):
    def __init__(self, gauss_ratio=1., device='cpu', nms_thres=0.2, use_distance=False):
        super(GaussianNMS, self).__init__(x=1., y=1., w=1., h=1., theta=0., ratio=gauss_ratio, device=device)
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
            gauss_score_map = self.call_maps(pred_corners_map[:, i, :])  # size(n, 1)
            gauss_map_list.append(gauss_score_map)
        gaussian_score_map = torch.stack(gauss_map_list, dim=-1)
        return gaussian_score_map  # size(n, 4)

    @staticmethod
    # 计算四个角点距离平方和取平均作为 rho^2
    def gen_distance_quality(corners1, corners2):
        """
        :param corners1: size(1, 8), from out_reg_map, real mapping corners
        :param corners2: size(n, 8), real mapping corners
        :return: size(n)
        """
        center_distance = (corners1[..., 0::2] - corners2[..., 0::2]) ** 2 + \
                          (corners1[..., 1::2] - corners2[..., 1::2]) ** 2  # size(n, 4)
        center_distance = torch.mean(center_distance, dim=-1)  # size(n)
        n = corners2.shape[0]
        expand_corners1 = corners1.repeat(n, 1)
        union_bbox = corner2bbox_tensor(torch.cat([expand_corners1, corners2], dim=-1))
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
        assert corners.dim() == 2 and corners.shape[-1] == 8
        # 降序排列，order下标排序
        _, order = scores.sort(0, descending=True)
        # print(_)
        keep = []
        while order.numel() > 0:  # torch.numel()返回张量元素个数
            if order.numel() == 1:  # 保留框只剩一个
                i = order.item()
                keep.append(i)
                break
            else:
                i = order[0].item()  # 保留scores最大的那个框corner[i]
                keep.append(i)
            # 计算corners[i]与其余各框的Gauss Score
            x, y, w, h, theta = cv2_get_min_bounding_rect(corners[i].cpu().numpy().reshape(4, 2))
            self.reset(x, y, w, h, theta)
            gauss_pred = self.gaussian_score_maps_generator(corners[i], corners[order[1:]])  # return size(n - 1, 4)
            gauss_pred = gauss_pred.mean(-1)  # size(n - 1)
            if not self.use_distance:
                idx = (gauss_pred <= self.nms_thres).nonzero(as_tuple=False).squeeze()  # 注意此时idx为[N-1,] 而order为[N,]
            else:
                corners = corners.reshape(-1, 8)
                distance_pred = self.gen_distance_quality(corners[i], corners[order[1:]])  # size(n - 1)
                # print(gauss_pred, distance_pred)
                cplex_quality = gauss_pred - distance_pred
                idx = (cplex_quality <= self.nms_thres).nonzero(as_tuple=False).squeeze()
            if idx.numel() == 0:
                break
            order = order[idx + 1]  # 修补索引之间的差值
        return torch.LongTensor(keep)  # Pytorch的索引值为LongTensor


def test_rotated_gauss():
    # img = np.zeros((416, 416), np.uint8)
    # img = np.zeros((52, 52), np.uint8)
    img = cv2.imread("../data/example3.jpg")
    print(img.shape)
    # coords = np.array([[150, 100], [250, 180], [240, 240], [140, 160]], dtype=np.int)
    # coords = np.array([[150, 100], [250, 50], [240, 120], [140, 160]], dtype=np.int)
    # coords = np.array([[150, 100], [151, 250], [102, 252], [100, 100]], dtype=np.int)
    # coords = np.array([[100, 100], [250, 100], [250, 150], [100, 150]], dtype=np.int)
    # coords = np.array([[10, 10], [15, 10], [15, 13], [10, 13]], dtype=np.int)
    coords = np.array([155., 160., 372., 111., 378., 201, 158.,
                       262.], dtype=np.int).reshape(4, 2)
    x, y, w, h, theta = cv2_get_min_bounding_rect(coords)
    print(x, y, w, h, theta)
    gr = Gaussian2DWithRotatedTensor(x, y, w, h, theta, ratio=4.0, device='cpu')

    # create a black use numpy,size is:512*512

    Y, X = np.mgrid[0:img.shape[0]:1, 0:img.shape[1]:1]

    xy_grids = np.stack([X, Y], axis=-1)

    xy_grids = torch.from_numpy(xy_grids).to(torch.float32) + 0.5
    xy_grids = xy_grids  # .to('cuda')
    # print(xy_grids)
    gr.reset(372., 111.)
    out_map = gr.call_maps(xy_grids)

    print('..', out_map.size())
    # print((out_map > 0.06).sum())
    # out_map = (out_map > 0.06).int()
    img_out = np.zeros(img.shape[:2], np.float)
    out_mask = out_map > 0.001
    gauss_v_map = out_map.cpu().numpy()
    img_out[out_mask] = gauss_v_map[out_mask]
    img_out = img_out * 255
    img_out = img_out.astype(np.uint8)
    img_out = np.expand_dims(img_out, axis=-1)
    img_out = np.repeat(img_out, 3, axis=-1)
    rect = cv2.minAreaRect(coords)  # 得到最小外接矩形的（中心(x,y), (宽,高), 旋转角度）
    box = cv2.boxPoints(rect)  # 获取最小外接矩形的4个顶点坐标(ps: cv2.boxPoints(rect) for OpenCV 3.x)
    box = np.int0(box)
    # 画出来
    cv2.drawContours(img, [coords], 0, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.drawContours(img, [box], 0, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.drawContours(img_out, [coords], 0, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.drawContours(img_out, [box], 0, (0, 0, 255), 2, cv2.LINE_AA)
    # =================================================
    # gr.refresh(100, 100, 1000)
    # out_map2 = gr.call_maps(xy_grids)
    # mask = (out_map2 > 0.8)
    # mask = mask.cpu().numpy()
    # out_img[mask] = 255

    cv2.imshow('1', img_out)
    cv2.imwrite('../data/e4.jpg', img_out)
    cv2.imshow('2', img)
    # cv2.imwrite('../data/e1.jpg', img)
    cv2.waitKey(0)


def test_gauss_nms():
    corners = torch.tensor([[150, 100, 250, 180, 240, 240, 140, 160],
                            [250, 100, 330, 180, 320, 240, 240, 160]], dtype=torch.float32)
    scores = torch.tensor([0.894, 0.766], dtype=torch.float32)
    gnms = GaussianNMS(use_distance=True)
    keep = gnms(corners, scores)
    print(keep)


if __name__ == '__main__':
    # test_gauss_nms()
    test_rotated_gauss()
