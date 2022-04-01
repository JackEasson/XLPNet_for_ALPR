import cv2
import time
import numpy as np


# before np.array to tensor
def cv_image_processor(img_mat, target_size):
    """
    :param img_mat: color image mat
    :param target_size: (target_w, target_h)
    :return:
    """
    # %% 高度统一固定，长度长于设定值则resize，短于则padding %%
    # [1] 尺寸调整
    tar_wh_ratio = target_size[0] / target_size[1]
    img_h, img_w = img_mat.shape[:2]
    src_wh_ratio = img_w / img_h
    if src_wh_ratio >= tar_wh_ratio:  # just resize
        img_mat = cv2.resize(img_mat, target_size, cv2.INTER_LINEAR)  # cv2.INTER_NEAREST
    else:
        resize_h = target_size[1]
        resize_w = int(resize_h * src_wh_ratio)
        img_mat = cv2.resize(img_mat, (resize_w, resize_h), cv2.INTER_LINEAR)  # cv2.INTER_NEAREST
        pad_left = (target_size[0] - resize_w) // 2

        pad_right = target_size[0] - resize_w - pad_left
        # top, bottom, left, right：上下左右要扩展的像素数
        img_mat = cv2.copyMakeBorder(img_mat, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    # [2] BGR -> RGB
    img_mat = cv2.cvtColor(img_mat, cv2.COLOR_BGR2RGB).astype(np.float32)
    return img_mat
    # return img_mat


def opencv_load_onnx():
    # load onnx model
    onnx_path = "../weights/CLPDetNet_es.onnx"  # ../result/ghost/OCRNet_ghost.onnx
    net = cv2.dnn.readNetFromONNX(onnx_path)

    # net = cv2.dnn.readNetFromTorch("./")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    # 创建Blob
    img_path = "E:\\images\\1.jpg"
    src_img = cv2.imread(img_path)
    src_img2 = np.zeros((416, 416, 3), np.uint8)
    # src_img = cv_image_processor(src_img, (224, 224))
    # print(src_img.shape)
    # src_img = src_img / 255.0 * 2.0 - 1.0  # -1 ~ 1
    # Create a 4D blob from a frame.

    blob = cv2.dnn.blobFromImage(src_img, 2.0 / 255, (416, 416), [127.5, 127.5, 127.5], swapRB=False, crop=False)
    # blob = cv2.dnn.blobFromImage(src_img2, 1.0, (320, 320), [0, 0, 0], swapRB=False, crop=False)
    # Sets the input to the network
    net.setInput(blob)
    start = time.perf_counter()
    # Runs the forward pass to get output of the output layers
    # outs = net.forward(["output1", "output2", "output3", "output4"])
    out = net.forward(["output"])
    end = time.perf_counter()
    print(out[0].shape)
    print(out[0][0, 10, 10, :])
    t = (end - start) * 1000
    print("Time: %.3fms" % t)


if __name__ == '__main__':
    opencv_load_onnx()

