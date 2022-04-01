"""
function: training programmes for detection mode only.
author: Soren Chopin
data: 2021/4/21
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


def train(args, model, global_device):
    # ===============================【0】Argument Estimate ========================
    assert args.backward_delay is None or args.backward_delay > 0
    assert 0. < args.loss_shift < 1.0
    multi_gpu = True if isinstance(args.device, list) else False
    loss_shift_epoch = int(args.loss_shift * args.epochs)
    print("Loss shift epoch: %d" % loss_shift_epoch)
    # ===============================【1】DataSet=================================
    # 1）train dataset loader； 2）val dataset loader
    dataset_train = CCPDDataSet2019(folder=cfg.CCPD2019['root'], data_txt_path=cfg.CCPD2019['train_txt'],
                                    inp_size=(cfg.INP_SIZE, cfg.INP_SIZE), use_letterbox=args.letterbox,
                                    use_augment=cfg.USE_AUGMENT, limit=60000)
    dataset_val = CCPDDataSet2019(folder=cfg.CCPD2019['root'], data_txt_path=cfg.CCPD2019['val_txt'],
                                  inp_size=(cfg.INP_SIZE, cfg.INP_SIZE), use_letterbox=args.letterbox,
                                  use_augment=False, limit=5000)  # limit=-1
    print("=> Train dataset total images: % d" % dataset_train.__len__())
    print("=> Val dataset total images: % d" % dataset_val.__len__())
    loader_train = DataLoader(dataset_train, num_workers=args.workers, batch_size=args.batch_size,
                              drop_last=True, shuffle=True, collate_fn=dataset_train.base_collate_fn)
    loader_val = DataLoader(dataset_val, num_workers=args.workers, batch_size=args.batch_size,
                            drop_last=False, shuffle=False, collate_fn=dataset_val.base_collate_fn)
    # ===============================【2】Label Assignment =================================
    GaussAssigner = WeightedGaussianAssignment(stage_lvl=3, assign_thres=0.6, gauss_ratio=cfg.GAUSS_RATIO_1,
                                               device=global_device)
    # ===============================【3】Loss Function =================================
    # score loss
    # criterion_score = VarifocalLoss(reduction=None)  # reduction=None -> keep dim
    criterion_score = ScoreFocalLoss()
    # corner loss
    criterion_coord = GaussDistanceLoss()
    # ===============================【4】Post Process =================================
    post_processor = CLPPostProcessor(conf_thres=args.conf_thres, gauss_ratio=cfg.GAUSS_RATIO_2, device=global_device,
                                      nms_thres=args.nms_thres, max_obj=cfg.MAX_OBJ, use_distance=args.use_distance)

    # ===============================【5】save train information and models =================================
    # save dir
    savedir = os.path.join(cfg.saveParentFolder, str(args.savedir))

    # model instruction
    modeltxtpath = savedir + "/model.txt"
    # some logging txt
    with open(modeltxtpath, "w") as f:
        f.write(str(model))
    loss_writer = TxtStateWriter(file_root=savedir, file_name="loss.txt",
                                 state_list=['Epoch', 'Step', 'TotalStep', 'TotalLoss', 'ScoreLoss', 'CoordLoss'],
                                 resume=args.resume)
    epoch_info_writer = TxtStateWriter(file_root=savedir, file_name="epoch_info.txt",
                                       state_list=['Epoch', 'TrainDetLoss', 'TrainScoreLoss', 'TrainCoordLoss',
                                                   'ValDetLoss', 'ValScoreLoss', 'ValCoordLoss', 'F1', 'AP', 'Lr'],
                                       resume=args.resume)

    # image shower
    img_oup_dir = os.path.join(savedir, 'images')
    if not os.path.exists(img_oup_dir):
        os.makedirs(img_oup_dir)
    ImgDemo = CLPShowDemo2019(oup_folder=img_oup_dir, inp_size=(cfg.INP_SIZE, cfg.INP_SIZE), use_letterbox=True,
                              device=global_device)
    ImgDemo.set_inp_folder(cfg.CCPD2019['test_txt'])  # set input images
    # ===============================【6】optimizer =================================
    optimizer_type = args.optimizer.lower()
    if optimizer_type == 'adam':
        optimizer = Adam(model.parameters(), args.lr, (0.9, 0.999), eps=1e-08, weight_decay=1e-5)
    else:
        optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9)
    lr_scheduler_type = args.lr_scheduler.lower()
    scheduler = None
    if lr_scheduler_type == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5, last_epoch=-1)  # 每10 epoch乘以0.5
    elif lr_scheduler_type == 'exp':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8, last_epoch=-1)
    elif lr_scheduler_type == 'cos':
        lr_final = args.lr * 0.05
        lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - lr_final) + lr_final  # cosine
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    # ===============================【7】ckpt resume =================================
    start_epoch = 1
    step_total = 0
    best_ap = 0.0  # 仅训练检测部分时使用
    if args.resume:  # 继续训练
        # Must load weights, optimizer, epoch and best value.
        filenameCheckpoint = savedir + '/checkpoint.pth'
        assert os.path.exists(
            filenameCheckpoint), "Error: resume option was used but checkpoint was not found in folder"
        checkpoint = torch.load(filenameCheckpoint)
        start_epoch = checkpoint['epoch']
        step_total = checkpoint['step_total']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        best_ap = checkpoint['best_ap']
        print("=> Loaded checkpoint at epoch {}".format(checkpoint['epoch']))

    elif args.pretrained is not None:
        print("Load weight from pretrained model ...")
        pretrained_dict = torch.load(args.pretrained)
        # model_dict = model.state_dict()
        # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # model_dict.update(pretrained_dict)
        model.load_state_dict(pretrained_dict)
        print("=> Load weight successfully.")

    else:
        print("Start weight initialize ...")
        weights_init(model)
        print("=> Load weight successfully.")
    # ===============================【8】begin train =================================
    optimizer.zero_grad()
    lr_epoch = 0
    for epoch in range(start_epoch, args.epochs + 1):
        print("----- TRAINING - EPOCH", epoch, "-----")
        for param_group in optimizer.param_groups:
            lr_epoch = param_group['lr']
            print("LEARNING RATE: ", lr_epoch)
        epoch_det_loss, epoch_score_loss, epoch_coord_loss = [], [], []
        time_train = []
        model.train()
        # one epoch train process
        pbar = tqdm(loader_train)

        for step, (images_tensor, corners_tensor_list, minRect_tensor_list, img_name_list) in enumerate(pbar):
            # print(images_tensor.requires_grad)
            # print(images.shape, first_labels.shape, second_labels.shape)
            step_total += 1
            start_time = time.time()
            # add grad
            images_tensor = images_tensor.requires_grad_(True)
            # ==> [1] put to device
            if not multi_gpu:
                images_tensor = images_tensor.to(global_device)
                corners_tensor_list = [corners_tensor.to(global_device) for corners_tensor in corners_tensor_list]
                minRect_tensor_list = [minRect_tensor.to(global_device) for minRect_tensor in minRect_tensor_list]
            else:
                images_tensor = images_tensor.cuda(non_blocking=True)
                corners_tensor_list = [corners_tensor.cuda(non_blocking=True) for corners_tensor in corners_tensor_list]
                minRect_tensor_list = [minRect_tensor.cuda(non_blocking=True) for minRect_tensor in minRect_tensor_list]
            # ==> [2] model feed forward
            out_score, out_corner = model(images_tensor)
            # print(out_score[0][0][0], out_corner[0][0][0])
            # ==> [3] calculate loss
            # corners_tensor: size(n, 8)
            score_loss_batch, coord_loss_batch = 0, 0
            cur_batch_size = len(corners_tensor_list)
            for batch_idx in range(cur_batch_size):
                out_score_single = out_score[batch_idx]
                out_corner_single = out_corner[batch_idx]
                corners_tensor_single = corners_tensor_list[batch_idx]
                minRect_tensor_single = minRect_tensor_list[batch_idx]
                tar_score_map, tar_corner_map, tar_weight_map, sample_target_map = \
                    GaussAssigner.single_assignment(out_corner_single, corners_tensor_single, minRect_tensor_single)

                score_loss_per_img, coord_loss_per_img = 0, 0
                score_loss_tmp = criterion_score(sample_target=sample_target_map,
                                                 score_out=out_score_single,
                                                 score_target=tar_score_map)
                coord_loss_tmp, loss_tuple = criterion_coord(out_corners_map=out_corner_single,
                                                             gt_corners_map=tar_corner_map,
                                                             real_gaussian_score_map=tar_score_map)
                if epoch >= loss_shift_epoch:
                    coord_loss_tmp += loss_tuple[2]  # + loss_size

                weighted_coord_loss_tmp = coord_loss_tmp * tar_weight_map
                score_loss_per_img += score_loss_tmp
                coord_loss_per_img += weighted_coord_loss_tmp[sample_target_map == 1].mean()  # only positive

                score_loss_batch += score_loss_per_img
                coord_loss_batch += coord_loss_per_img

            # batch average
            score_loss = score_loss_batch / args.batch_size
            coord_loss = coord_loss_batch / args.batch_size
            detection_loss = score_loss + args.det_lambda * coord_loss

            # 加入累积梯度更新
            if args.backward_delay is not None:
                loss_avg = detection_loss / args.backward_delay
                loss_avg.backward()
                if step_total % args.backward_delay == 0:
                    optimizer.step()  # 实际更新梯度
                    optimizer.zero_grad()  # 梯度清零
            else:
                detection_loss.backward()
                optimizer.step()  # 实际更新梯度
                optimizer.zero_grad()  # 梯度清零
            # add to corresponding list
            detection_loss_data = detection_loss.detach().item()
            score_loss_data = score_loss.detach().item()
            coord_loss_data = coord_loss.detach().item()
            epoch_det_loss.append(detection_loss_data)
            epoch_score_loss.append(score_loss_data)
            epoch_coord_loss.append(coord_loss_data)

            # training time
            time_train.append(time.time() - start_time)
            # 进度条输出
            mem = (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
            pbar.set_description("==>Train- Epoch: %d, Step: %d, Men: %.3gG, ScoreLoss: %.4f, CornerLoss: %.4f, BatchTime: %.3fs" %
                                 (epoch, step, mem, score_loss_data, coord_loss_data, time.time() - start_time))

            if args.steps_interval > 0 and step_total % args.steps_interval == 0:
                # record to txt
                # ['Epoch', 'Step', 'TotalStep', 'TotalLoss', 'ScoreLoss', 'CoordLoss']
                loss_writer.add_info([epoch, step, step_total, detection_loss_data, score_loss_data, coord_loss_data])
            # ======================= for debug ===================
            # if step > 10:
            #     break
        optimizer.zero_grad()  # 梯度清零
        average_epoch_train_det_loss = sum(epoch_det_loss) / len(epoch_det_loss)
        average_epoch_train_score_loss = sum(epoch_score_loss) / len(epoch_score_loss)
        average_epoch_train_coord_loss = sum(epoch_coord_loss) / len(epoch_coord_loss)

        print("==>The average detection loss of epoch %d is %.4f, score loss: %.4f, coord loss: %.4f"
              % (epoch, average_epoch_train_det_loss, average_epoch_train_score_loss, average_epoch_train_coord_loss))
        print("==>The train time of each batch is: %.4fs\n" % (sum(time_train) / len(time_train)))

        # ######################################################################################################
        # ================================== val accuracy ======================================
        print("----- VALIDATING - EPOCH", epoch, "-----")
        epoch_det_loss, epoch_score_loss, epoch_coord_loss = [], [], []
        model.eval()
        pbar = tqdm(loader_val)
        total_metrics_tp, total_metrics_score = [], []
        n_gt = 0  # 总gt数
        with torch.no_grad():
            for step, (images_tensor, corners_tensor_list, minRect_tensor_list, img_name_list) in enumerate(pbar):
                # print(images.shape, first_labels.shape, second_labels.shape)
                step_total += 1
                start_time = time.time()
                if not multi_gpu:
                    images_tensor = images_tensor.to(global_device)
                    corners_tensor_list = [corners_tensor.to(global_device) for corners_tensor in corners_tensor_list]
                    minRect_tensor_list = [minRect_tensor.to(global_device) for minRect_tensor in minRect_tensor_list]
                else:
                    images_tensor = images_tensor.cuda(non_blocking=True)
                    corners_tensor_list = [corners_tensor.cuda(non_blocking=True) for corners_tensor in
                                           corners_tensor_list]
                    minRect_tensor_list = [minRect_tensor.cuda(non_blocking=True) for minRect_tensor in
                                           minRect_tensor_list]
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
                    minRect_tensor_single = minRect_tensor_list[batch_idx]
                    tar_score_map, tar_corner_map, tar_weight_map, sample_target_map = \
                        GaussAssigner.single_assignment(out_corner_single, corners_tensor_single, minRect_tensor_single)

                    score_loss_per_img, coord_loss_per_img = 0, 0
                    score_loss_tmp = criterion_score(sample_target=sample_target_map,
                                                     score_out=out_score_single,
                                                     score_target=tar_score_map)
                    coord_loss_tmp, loss_tuple = criterion_coord(out_corners_map=out_corner_single,
                                                                 gt_corners_map=tar_corner_map,
                                                                 real_gaussian_score_map=tar_score_map)
                    if epoch >= loss_shift_epoch:
                        coord_loss_tmp += loss_tuple[2]  # + loss_size

                    weighted_coord_loss_tmp = coord_loss_tmp * tar_weight_map
                    score_loss_per_img += score_loss_tmp
                    coord_loss_per_img += weighted_coord_loss_tmp[sample_target_map == 1].mean()  # only positive

                    score_loss_batch += score_loss_per_img
                    coord_loss_batch += coord_loss_per_img

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
                                                                 iou_threshold=args.ap_index)
                total_metrics_tp.extend(metrics_tp)
                total_metrics_score.extend(metrics_score)

                # batch average
                score_loss = score_loss_batch / args.batch_size
                coord_loss = coord_loss_batch / args.batch_size
                detection_loss = score_loss + args.det_lambda * coord_loss

                detection_loss_data = detection_loss.detach().item()
                score_loss_data = score_loss.detach().item()
                coord_loss_data = coord_loss.detach().item()
                epoch_det_loss.append(detection_loss_data)
                epoch_score_loss.append(score_loss_data)
                epoch_coord_loss.append(coord_loss_data)

                # 进度条输出
                pbar.set_description("==>Val- Epoch: %d  Step: %d  ScoreLoss: %.5f  CoordLoss: %.5f" %
                                     (epoch, step, score_loss_data, coord_loss_data))
                # ======================= for debug ===================
                # if step > 10:
                #     break
        # val epoch output
        average_epoch_val_det_loss = sum(epoch_det_loss) / len(epoch_det_loss)
        average_epoch_val_coord_loss = sum(epoch_coord_loss) / len(epoch_coord_loss)
        average_epoch_val_score_loss = sum(epoch_score_loss) / len(epoch_score_loss)

        # APs
        p, r, ap, f1 = ap_per_class(np.array(total_metrics_tp), np.array(total_metrics_score), n_gt)
        current_ap = ap
        print("==>Val End -- Epoch %d, average detection loss is %.4f, coord loss: %.4f, score loss: %.4f"
              % (epoch, average_epoch_val_det_loss, average_epoch_val_coord_loss, average_epoch_val_score_loss))
        print("P = %.4f  R = %.4f  F1 = %.4f  AP = %.4f" % (p, r, f1, ap))
        is_best = current_ap > best_ap
        best_ap = max(current_ap, best_ap)

        # record in txt per epoch
        # ['Epoch', 'TrainDetLoss', 'TrainScoreLoss', 'TrainCoordLoss', 'ValDetLoss', 'ValScoreLoss', 'ValCoordLoss', 'F1', 'AP', 'Lr']
        epoch_info_writer.add_info([epoch, average_epoch_train_det_loss, average_epoch_train_score_loss,
                                    average_epoch_train_coord_loss, average_epoch_val_det_loss,
                                    average_epoch_val_score_loss, average_epoch_val_coord_loss, f1,
                                    current_ap, lr_epoch])
        # AP clear, NOTICE!!

        # save detection results -- images
        print("==> Start Detection Demo ...")
        ImgDemo.set_model(model)
        ImgDemo.random_show(n=20)
        print("==> Demo Finished.")

        filenameCheckpoint = savedir + '/checkpoint.pth'
        filenameBest = savedir + '/checkpoint_best.pth'

        # ================ Here, if best, we save it as the best model ================
        save_checkpoint({
            'epoch': epoch + 1,
            'step_total': step_total,
            'arch': str(model),
            'state_dict': model.state_dict(),
            'best_ap': best_ap,
            'optimizer': optimizer.state_dict(),
        }, is_best, filenameCheckpoint, filenameBest)

        # SAVE MODEL AFTER EPOCH
        filename = '{}/model-{}.pth'.format(savedir, epoch)
        filenamebest = '{}/model_best.pth'.format(savedir)
        # ===================固定epoch保存的仅有模型weight参数====================
        if args.epochs_save > 0 and epoch % args.epochs_save == 0 and epoch > 5:
            torch.save(model.state_dict(), filename)
            print('save: {} (epoch: {})'.format(filename, epoch))
        if is_best:
            torch.save(model.state_dict(), filenamebest)
            print('save: {} (epoch: {})'.format(filenamebest, epoch))
            with open(savedir + "/best.txt", "w") as f:
                f.write("Best epoch is %d, with AP = %.4f" % (epoch, best_ap))
        # 调整学习率
        scheduler.step()
    torch.cuda.empty_cache()
    return model  # return model (convenience for encoder-decoder training)


def weights_init(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 0.0, 0.01)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.001)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def save_checkpoint(state, is_best, filenameCheckpoint, filenameBest):
    torch.save(state, filenameCheckpoint)
    if is_best:
        print("Saving model as best")
        torch.save(state, filenameBest)


def main(args):
    savedir = os.path.join(cfg.saveParentFolder, str(args.savedir))
    print("The save file path is: " + savedir)

    if not os.path.exists(savedir):
        os.makedirs(savedir)

    with open(savedir + '/opts.txt', "w") as myfile:
        myfile.write(str(args))

    # ============================== Device Setting =========================
    if isinstance(args.device, str):
        args.device = args.device.lower()
    global_device = 'cpu'
    multi_gpu = False
    gpus = None
    if args.device != 'cpu':
        global_device = 'cuda'
        assert torch.cuda.is_available(), 'gpu is not available.'
        if isinstance(args.device, str):  # '0', '1'
            gpu_id = int(args.device)
            print('Using gpu: %s' % torch.cuda.get_device_name(gpu_id))
            torch.cuda.set_device('cuda:{}'.format(gpu_id))
        elif isinstance(args.device, list):
            assert args.batch_size % len(args.device) == 0, 'batch size need to be divisible by gpu number'
            multi_gpu = True
            gpus = args.device
            for i, gpu_id in enumerate(gpus):
                print('Using gpu%d: %s' % (i, torch.cuda.get_device_name(gpu_id)))
            torch.cuda.set_device('cuda:{}'.format(gpus[0]))
        else:
            raise AttributeError('Wrong device format!')
    print('=> Device setting: %s' % global_device)
    # ============================== Load Model ===========================
    model = CornerLPDet(inp_wh=(cfg.INP_SIZE, cfg.INP_SIZE), backbone=cfg.BACKBONE, device=global_device,
                        export_onnx=False)
    if not multi_gpu:
        model.to(global_device)
    else:
        model = nn.DataParallel(model.to(global_device), device_ids=gpus, output_device=gpus[0])
    print("========== START TRAINING ===========")
    try:
        model = train(args, model, global_device)  # Train decoder
        print("========== TRAINING FINISHED ===========")
    except KeyboardInterrupt:
        torch.save(model.state_dict(), os.path.join(savedir, 'INTERRUPTED.pth'))
        sys.exit(0)
    print("========== TRAINING INTERRUPTED ===========")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--lr', type=int, default=cfg.LR)
    parser.add_argument('--lr_scheduler', type=str, default=cfg.LR_SCHEDULER)  # in (step, exp, cos)
    parser.add_argument('--optimizer', type=str, default=cfg.OPTIMIZER)
    parser.add_argument('--conf_thres', type=float, default=cfg.CONF_THRES)
    parser.add_argument('--nms_thres', type=float, default=cfg.NMS_THRES)
    parser.add_argument('--ap_index', type=float, default=cfg.AP_INDEX, help='to calculate APx -- AP0.5, AP0.7')
    parser.add_argument('--det_lambda', type=float, default=cfg.DET_LAMBDA, help='factor added to detection loss')
    parser.add_argument('--use_distance', type=bool, default=cfg.USE_DISTANCE,
                        help='if use distance to nms for outputs')
    parser.add_argument('--letterbox', type=bool, default=cfg.USE_LETTER_BOX, help='if use letterbox')
    # device setting
    parser.add_argument('--device', type=str or list, default=0, help='cpu or [0, 1] / 0 for gpu')

    parser.add_argument('--epochs', type=int, default=5, help='total epoch to train')
    parser.add_argument('--loss_shift', type=float, default=0.1, help='factor for epoch to shift detection loss')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--backward_delay', default=None, type=int,
                        help='Number of gradient accumulation steps')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--steps_interval', type=int, default=50, help='save loss every x steps')
    parser.add_argument('--epochs_save', type=int, default=1)  # You can use this value to save model every X epochs
    parser.add_argument('--savedir', default="test20220304_multi_gpus")
    parser.add_argument('--resume', action='store_true', default=False)  # Use this flag to load last checkpoint for training
    parser.add_argument('--pretrained', default=None)  # "./output/pretrained/model_75AP.pth"

    main(parser.parse_args())






