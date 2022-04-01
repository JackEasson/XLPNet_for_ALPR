import torch
import torch.onnx
import numpy as np
import cv2
from XLPDet_pipleline import *


def test_pth():
    img_path = "E:\\images\\1.jpg"
    src_img = cv2.imread(img_path)
    print(src_img.shape)
    src_img = cv2.resize(src_img, (416, 416))
    src_img = src_img / 255.0 * 2.0 - 1.0  # -1 ~ 1
    # print(min(src_img.reshape(-1)), max(src_img.reshape(-1)))
    pthfile = "../weights/CLPDet_s.pth"
    # loaded_model = torch.load(pthfile, map_location='cpu')
    model = torch.load(pthfile, map_location='cpu')
    model.eval()
    src_tensor = torch.tensor(src_img).permute(2, 0, 1).unsqueeze(0).float()
    print(src_tensor.shape)
    out = model(src_tensor)
    print(out.shape)
    print(out[0, 10, 10, :])
    # np.save('../result/ghost/pth_out.npy', out.detach().numpy())


def read_npy():
    data = np.load('./result/pth_out.npy')
    print(data)


if __name__ == '__main__':
    test_pth()
    # read_npy()
