import os
import cv2
import numpy as np
import time
from base import read_filenames, cv_imread, pre_process, PostProcessor
from argparse import ArgumentParser
from plot_tools import plot_polygon_bbox


def detect(args):
    # [1] load the weight
    onnx_path = args.weight
    assert onnx_path.endswith('onnx')
    assert os.path.exists(args.savedir)
    print("=> Load Net ...")
    net = cv2.dnn.readNetFromONNX(onnx_path)
    print("   Load successful!")
    outLayerName = 'output'
    # [2] set backend and target
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    # [3] post processor initialization
    GaussianNMS = PostProcessor(conf_thres=args.conf_threshold, nms_thres=args.nms_threshold, max_obj=50,
                                use_distance=args.use_distance)
    # [4] get image paths
    print("=> Detect images ...")
    if os.path.isdir(args.image):
        names = read_filenames(args.image)
        image_paths = [os.path.join(args.image, n) for n in names]
    else:
        image_paths = [args.image]
    for idx, img_path in enumerate(image_paths):
        print("Now detect image no.%d" % idx)
        img_mat = cv_imread(img_path)
        img_mat = pre_process(img_mat)
        # [4] to blob
        blob = cv2.dnn.blobFromImage(img_mat, 2.0 / 255, args.shape, [127.5, 127.5, 127.5], swapRB=True, crop=False)
        # blob = cv2.dnn.blobFromImage(src_img2, 1.0, (320, 320), [0, 0, 0], swapRB=False, crop=False)
        # [5] Sets the input to the network
        net.setInput(blob)
        start = time.perf_counter()
        # [6] Runs the forward pass to get output of the output layers
        net_out = net.forward([outLayerName])
        end = time.perf_counter()
        net_out = net_out[0]
        t = (end - start) * 1000
        print("Forward Time: %.3fms" % t)
        start = end
        print(net_out.shape)
        nms_out = GaussianNMS(net_out)
        end = time.perf_counter()
        t = (end - start) * 1000
        print("NMS Time: %.3fms" % t)
        plot_polygon_bbox(img_mat, corners=nms_out[:, 1:], scores=nms_out[:, 0])
        cv2.imshow('Show%d' % idx, img_mat)
        val = cv2.waitKey(0)
        if val == ord('s'):
            save_path = os.path.join(args.savedir, 'lp350_%d.jpg' % idx)
            cv2.imwrite(save_path, img_mat)
        elif val == ord('q'):
            cv2.destroyAllWindows()
            break
        cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--image', '-i', type=str, default="./images/LP350",
                        help="folder or image")
    parser.add_argument('--shape', '-s', type=tuple or list, default=(416, 416),
                        help="folder or image")
    parser.add_argument('--conf_threshold', type=float, default=0.4)
    parser.add_argument('--nms_threshold', type=float, default=0.1)
    parser.add_argument('--use_distance', type=bool, default=True)
    parser.add_argument('--weight', '-w',  type=str, default="./weights/effS_25_3d.onnx")  # eff_s_92ap.onnx
    parser.add_argument('--savedir', '-t', type=str, default="./output/LP350")
    detect(parser.parse_args())