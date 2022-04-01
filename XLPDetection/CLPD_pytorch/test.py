"""
function: testing programmes for detection.
author: Soren Chopin
data: 2022/2/28
"""

import sys
import os
import time
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import math
from argparse import ArgumentParser
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader

from XLPDet_pipleline import CornerLPDet
from data_loader.datasets import CCPDDataSet2019
from utils.ap_tools import *
from utils.general import cv2_get_min_bounding_rect, corner2bbox_multi
from utils.log_tools import TxtStateWriter
from utils.test_tools import CLPShowDemo2019
from strategy.post_process import CLPPostProcessor, nms_out_corners2bboxes
from strategy.label_assignment import WeightedGaussianAssignment
from strategy.loss_function import VarifocalLoss, GaussDistanceLoss, ScoreFocalLoss
import config as cfg


def test(args):
    # ============================== Device Setting =========================
    args.device = args.device.lower()
    assert args.device == 'cpu' or isinstance(int(args.device), int)
    global_device = 'cpu'
    if args.device != 'cpu':
        assert torch.cuda.is_available(), 'gpu is not available.'
        gpu_id = int(args.device)
        print('Using gpu: %s' % torch.cuda.get_device_name(gpu_id))
        global_device = 'cuda:%s' % args.device
    print('=> Device setting: %s' % global_device)
    # ============================ model initial ==========================
    model = CornerLPDet(inp_wh=(cfg.INP_SIZE, cfg.INP_SIZE), backbone=cfg.BACKBONE, device=global_device, export_onnx=False)
    model.to(global_device)
    model.eval()
    # load weights
    print("Load weight from pretrained model ...")
    pretrained_dict = torch.load(args.weight)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    # [1] datasets
    # dataset_test = CCPDDataSet2019(folder=cfg.CCPD2019['root'], data_txt_path=cfg.CCPD2019['test_txt'],
    #                                inp_size=(cfg.INP_SIZE, cfg.INP_SIZE), use_letterbox=args.letterbox,
    #                                use_augment=False, limit=-1)  # limit=-1
    # loader_test = DataLoader(dataset_test, num_workers=args.workers, batch_size=args.batch_size,
    #                          drop_last=False, shuffle=False, collate_fn=dataset_test.base_collate_fn)
    # print("=>Test dataset total images: % d" % dataset_test.__len__())
    # =============================== Post Process =================================
    post_processor = CLPPostProcessor(conf_thres=cfg.CONF_THRES, gauss_ratio=cfg.GAUSS_RATIO_2, device=global_device,
                                      nms_thres=cfg.NMS_THRES, max_obj=cfg.MAX_OBJ, use_distance=cfg.USE_DISTANCE)
    # [3] test
    # 修改多线程的tensor方式为file_system，避免共享的tensor超过open files限制
    torch.multiprocessing.set_sharing_strategy('file_system')

    # split mode: false  whole test data
    if not args.split_mode:
        print("==> Whole Test Mode Start ...")
        # [1] datasets
        dataset_test = CCPDDataSet2019(folder=cfg.CCPD2019['root'], data_txt_path=cfg.CCPD2019['test_txt'],
                                       inp_size=(cfg.INP_SIZE, cfg.INP_SIZE), use_letterbox=cfg.USE_LETTER_BOX,
                                       use_augment=False, limit=1000)
        loader_test = DataLoader(dataset_test, num_workers=args.workers, batch_size=args.batch_size,
                                 drop_last=False, shuffle=False, collate_fn=dataset_test.base_collate_fn)
        print("=>Test dataset total images: % d" % dataset_test.__len__())

        with torch.no_grad():
            test_time = []
            pbar = tqdm(loader_test)
            total_metrics_tp, total_metrics_score = [], []
            n_gt = 0
            for step, (images_tensor, corners_tensor_list, minRect_tensor_list, img_name_list) in enumerate(pbar):
                # print(images.shape, first_labels.shape, second_labels.shape)
                start_time = time.time()
                images_tensor = images_tensor.to(global_device)
                corners_tensor_list = [corners_tensor.to(global_device) for corners_tensor in corners_tensor_list]
                # ==> [1] need grad
                # images_tensor = images_tensor.requires_grad_(False)
                # ==> [2] model feed forward
                out_score, out_corner = model(images_tensor)
                # ==> [3] calculate loss
                # corners_tensor: size(n, 8)
                score_loss_batch, coord_loss_batch = 0, 0
                score_bbox_out_list, gt_corner_list = [], []
                cur_batch_size = len(corners_tensor_list)
                for batch_idx in range(cur_batch_size):
                    out_score_single = out_score[batch_idx]
                    out_corner_single = out_corner[batch_idx]
                    corners_tensor_single = corners_tensor_list[batch_idx]
                    # NMS
                    nms_out = post_processor(out_score_map=out_score_single, out_corner_map=out_corner_single)
                    # if empty
                    if nms_out is None or nms_out.size(0) == 0:
                        score_bbox_out_list.append(None)
                    else:
                        score_bbox_out = nms_out_corners2bboxes(nms_out)
                        score_bbox_out_list.append(score_bbox_out)
                    gt_corner_list.append(corners_tensor_single)
                    n_gt += corners_tensor_single.size(0)
                gt_bbox_list = [corner2bbox_multi(gt_corner) for gt_corner in gt_corner_list]
                metrics_tp, metrics_score = get_batch_statistics(outputs=score_bbox_out_list, targets=gt_bbox_list,
                                                                 iou_threshold=cfg.AP_INDEX)
                total_metrics_tp.extend(metrics_tp)
                total_metrics_score.extend(metrics_score)


                # 进度条输出
                test_time.append(time.time() - start_time)
                mem = (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                pbar.set_description("==>Test - Step: %d, Men: %.3gG" % (step, mem))
                # ======================= for debug ===================
                # if step > 10:
                #     break

            # APs
            p, r, ap, f1 = ap_per_class(np.array(total_metrics_tp), np.array(total_metrics_score), n_gt)
            print("Test finish: P = %.4f  R = %.4f  F1 = %.4f  AP = %.4f" % (p, r, f1, ap))
            time_per_img = (sum(test_time) / len(test_time)) / args.batch_size
            print("The test time for per image is: %.3f (FPS: %.3f)" % (time_per_img, 1 / time_per_img))
            # save to txt
            assert args.savedir.endswith('.txt')
            with open(args.savedir, 'a', encoding='utf-8') as f:
                time_info = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                weight_info = args.weight
                result_info = "P = %.4f  R = %.4f  F1 = %.4f  AP = %.4f" % (p, r, f1, ap)
                f.write(time_info + "  " + weight_info + "  " + result_info + '\n')

    else:
        print("==> Split Test Mode Start ...")
        split_list = ['ccpd_blur.txt', 'ccpd_challenge.txt', 'ccpd_db.txt', 'ccpd_fn.txt', 'ccpd_rotate.txt',
                      'ccpd_tilt.txt']
        result_dict = {}
        for subset in split_list:
            subname = subset.split('.')[0]
            print("Now process subset: %s" % subname)
            # [1] datasets
            dataset_test = CCPDDataSet2019(folder=cfg.CCPD2019['root'],
                                           data_txt_path=os.path.join(cfg.CCPD2019['root'], "splits", subset),
                                           inp_size=(cfg.INP_SIZE, cfg.INP_SIZE), use_letterbox=cfg.USE_LETTER_BOX,
                                           use_augment=False, limit=1000)
            loader_test = DataLoader(dataset_test, num_workers=args.workers, batch_size=args.batch_size,
                                     drop_last=False, shuffle=False, collate_fn=dataset_test.base_collate_fn)
            print("=>Subset images: % d" % dataset_test.__len__())

            with torch.no_grad():
                test_time = []
                pbar = tqdm(loader_test)
                total_metrics_tp, total_metrics_score = [], []
                n_gt = 0
                for step, (images_tensor, corners_tensor_list, minRect_tensor_list, img_name_list) in enumerate(pbar):
                    # print(images.shape, first_labels.shape, second_labels.shape)
                    start_time = time.time()
                    images_tensor = images_tensor.to(global_device)
                    corners_tensor_list = [corners_tensor.to(global_device) for corners_tensor in corners_tensor_list]
                    # ==> [1] need grad
                    # images_tensor = images_tensor.requires_grad_(False)
                    # ==> [2] model feed forward
                    out_score, out_corner = model(images_tensor)
                    # ==> [3] calculate loss
                    # corners_tensor: size(n, 8)
                    score_loss_batch, coord_loss_batch = 0, 0
                    score_bbox_out_list, gt_corner_list = [], []
                    cur_batch_size = len(corners_tensor_list)
                    for batch_idx in range(cur_batch_size):
                        out_score_single = out_score[batch_idx]
                        out_corner_single = out_corner[batch_idx]
                        corners_tensor_single = corners_tensor_list[batch_idx]
                        # NMS
                        nms_out = post_processor(out_score_map=out_score_single, out_corner_map=out_corner_single)
                        # if empty
                        if nms_out is None or nms_out.size(0) == 0:
                            score_bbox_out_list.append(None)
                        else:
                            score_bbox_out = nms_out_corners2bboxes(nms_out)
                            score_bbox_out_list.append(score_bbox_out)
                        gt_corner_list.append(corners_tensor_single)
                        n_gt += corners_tensor_single.size(0)
                    gt_bbox_list = [corner2bbox_multi(gt_corner) for gt_corner in gt_corner_list]
                    metrics_tp, metrics_score = get_batch_statistics(outputs=score_bbox_out_list, targets=gt_bbox_list,
                                                                     iou_threshold=cfg.AP_INDEX)
                    total_metrics_tp.extend(metrics_tp)
                    total_metrics_score.extend(metrics_score)

                    # 进度条输出
                    test_time.append(time.time() - start_time)
                    mem = (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                    pbar.set_description("==>Test: %s - Step: %d, Men: %.3gG" % (subname, step, mem))
                    # ======================= for debug ===================
                    # if step > 10:
                    #     break
                # APs
                p, r, ap, f1 = ap_per_class(np.array(total_metrics_tp), np.array(total_metrics_score), n_gt)
                print("Test finish: P = %.4f  R = %.4f  F1 = %.4f  AP = %.4f" % (p, r, f1, ap))
                time_per_img = (sum(test_time) / len(test_time)) / args.batch_size
                print("The test time for per image is: %.3f (FPS: %.3f)" % (time_per_img, 1 / time_per_img))
                cur_info = "P = %.4f  R = %.4f  F1 = %.4f  AP = %.4f" % (p, r, f1, ap)
                result_dict[subname] = cur_info

        # save to txt
        assert args.savedir.endswith('.txt')
        with open(args.savedir, 'a', encoding='utf-8') as f:
            time_info = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            weight_info = args.weight
            f.write(time_info + "  " + weight_info + "  ")
            for (subset_name, sub_info) in result_dict.items():
                result_info = "%s: %s\n" % (subset_name, sub_info)
                f.write(result_info)
            f.write("\n")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--device', type=str, default='0', help='cuda device, i.e. 0/1/... or cpu')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--split_mode', type=bool, default=True)
    parser.add_argument('--weight', default="./output/test20220227_shift/model_best.pth")
    parser.add_argument('--savedir', default="./test_result/test2019.txt")
    test(parser.parse_args())