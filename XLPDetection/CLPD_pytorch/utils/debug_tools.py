import torch
from torch import Tensor
import numpy as np


def print_tensor_2d(mat: Tensor, txt_path):
    mat = mat.detach().cpu().numpy()
    if mat.dtype == 'bool':
        mat = mat.astype(np.int8)
    h, w = mat.shape
    with open(txt_path, 'w', encoding='utf-8') as f:
        for i in range(h):
            temp = ""
            for j in range(w):
                temp += str(mat[i][j]) + ' '
            f.write(temp + '\n')