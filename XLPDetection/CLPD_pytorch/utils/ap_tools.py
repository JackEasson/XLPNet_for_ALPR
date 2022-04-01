"""
source from Yolo-FastestV2: https://github.com/dog-qiuqiu/Yolo-FastestV2
"""

import torch
import torchvision
import torch.nn.functional as F

import os
import time
import numpy as np


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = \
            box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = \
            box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


# if iou_threshold == 0.5, it means calculating AP0.5
# for one class
# def get_batch_statistics(outputs, targets, iou_threshold):
#     """ Compute true positives, predicted scores and predicted labels per sample """
#     """
#         outputs: detections with list, list(b) -> tensor: n1x5 (x1, y1, x2, y2, conf) in a batch
#         targets: list(b) -> tensor: n2x4 (x1, y1, x2, y2)
#     """
#     batch_metrics = []
#     for sample_i in range(len(outputs)):
#
#         if outputs[sample_i] is None:
#             continue
#
#         output = outputs[sample_i]
#         pred_boxes = output[:, :4]
#         pred_scores = output[:, 4]
#
#         true_positives = np.zeros(pred_boxes.shape[0])
#
#         annotations = targets[sample_i]
#         if len(annotations):
#             detected_boxes = []
#             target_boxes = annotations  # size(n2, 4)
#             # to each predicted bbox
#             for pred_i, pred_box in enumerate(pred_boxes):
#                 # pred_box size(4)
#                 # If targets are found break
#                 if len(detected_boxes) == len(annotations):
#                     break
#                 # select the max iou to match, one to one
#                 iou, box_index = bbox_iou(pred_box.unsqueeze(0), target_boxes).max(0)
#                 if iou >= iou_threshold and box_index not in detected_boxes:
#                     true_positives[pred_i] = 1
#                     detected_boxes += [box_index]
#         # all to numpy.ndarray
#         batch_metrics.append([true_positives, pred_scores.cpu().numpy()])
#     return batch_metrics

def get_batch_statistics(outputs, targets, iou_threshold):
    """ Compute true positives, predicted scores and predicted labels per sample """
    """
        outputs: detections with list, list(b) -> tensor: n1x5 (conf, x1, y1, x2, y2) in a batch
        targets: list(b) -> tensor: n2x4 (x1, y1, x2, y2)
    """
    metrics_tp, metrics_score = [], []
    for sample_i in range(len(outputs)):

        if outputs[sample_i] is None:
            continue

        output = outputs[sample_i]
        pred_scores = output[:, 0]
        pred_boxes = output[:, 1:5]

        true_positives = np.zeros(pred_boxes.shape[0])

        annotations = targets[sample_i]
        if len(annotations):
            detected_boxes = []
            target_boxes = annotations  # size(n2, 4)
            # to each predicted bbox
            for pred_i, pred_box in enumerate(pred_boxes):
                # pred_box size(4)
                # If targets are found break
                if len(detected_boxes) == len(annotations):
                    break
                # select the max iou to match, one to one
                iou, box_index = bbox_iou(pred_box.unsqueeze(0), target_boxes).max(0)
                if iou >= iou_threshold and box_index not in detected_boxes:
                    true_positives[pred_i] = 1
                    detected_boxes += [box_index]
        metrics_tp.extend(true_positives)
        metrics_score.extend(pred_scores.cpu().numpy())
    return metrics_tp, metrics_score


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    # mpre.size - 1, mpre.size - 2, ..., 2, 1
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def ap_per_class(tp, conf, n_gt):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (np.array).
        conf:  Objectness value from 0-1 (np.array).
        n_gt:  Number of ground truth objects
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf = tp[i], conf[i]

    n_p = i.sum()  # Number of predicted objects

    if n_p == 0 or n_gt == 0:
        ap, r, p = 0, 0, 0
    else:
        # Accumulate FPs and TPs
        fpc = (1 - tp).cumsum()
        tpc = tp.cumsum()

        # Recall
        recall_curve = tpc / (n_gt + 1e-16)
        r = recall_curve[-1]

        # Precision
        precision_curve = tpc / (tpc + fpc)
        p = precision_curve[-1]

        # AP from recall-precision curve
        ap = compute_ap(recall_curve, precision_curve)

    # Compute F1 score (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1

