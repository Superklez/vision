import torch
from collections import Counter

def intersection_over_union(bboxes1, bboxes2, bbox_format='corners'):
    '''
    Inputs:
    ----------
        bboxes1 (Tensor[n, 4]):
        bboxes2 (Tensor[n, 4]):
        bbox_format (str):

    Returns:
    -----------
        iou: Intersection over union
    '''

    if bbox_format in ['c', 'C', 'corner', 'corners']:
        box1_x1 = bboxes1[..., 0:1]
        box1_y1 = bboxes1[..., 1:2]
        box1_x2 = bboxes1[..., 2:3]
        box1_y2 = bboxes1[..., 3:4]

        box2_x1 = bboxes2[..., 0:1]
        box2_y1 = bboxes2[..., 1:2]
        box2_x2 = bboxes2[..., 2:3]
        box2_y2 = bboxes2[..., 3:4]

    elif bbox_format in ['m', 'M', 'midpoint', 'midpoints']:
        box1_x1 = bboxes1[..., 0:1] - bboxes1[..., 2:3] / 2
        box1_y1 = bboxes1[..., 1:2] - bboxes1[..., 3:4] / 2
        box1_x2 = bboxes1[..., 0:1] + bboxes1[..., 2:3] / 2
        box1_y2 = bboxes1[..., 1:2] + bboxes1[..., 3:4] / 2

        box2_x1 = bboxes2[..., 0:1] - bboxes2[..., 2:3] / 2
        box2_y1 = bboxes2[..., 1:2] - bboxes2[..., 3:4] / 2
        box2_x2 = bboxes2[..., 0:1] + bboxes2[..., 2:3] / 2
        box2_y2 = bboxes2[..., 1:2] + bboxes2[..., 3:4] / 2

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    box1_area = torch.abs(torch.mul((box1_x2 - box1_x1), (box1_y2 - box1_y1)))
    box2_area = torch.abs(torch.mul((box2_x2 - box2_x1), (box2_y2 - box2_y1)))

    intersection = torch.mul((x2 - x1).clamp(0), (y2 - y1).clamp(0))
    union = box1_area + box2_area - intersection + 1e-6
    iou_val = torch.div(intersection, union)

    return iou_val

def non_max_suppression(bboxes, prob_threshold=0.2, iou_threshold=0.4, bbox_format='corners'):
    '''
    Inputs:
    ----------
        bboxes (list):
            list of lists where each nested list correspond to
            parameters of a bounding box with the format: [class_pred,
            prob_score, x1, y1, x2, y2]
        iou_threshold (float):

    Returns:
    ----------
        list: Bounding boxes after applying non-max suppression.
    '''
    bboxes = [bbox for bbox in bboxes if bbox[1] > prob_threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_nms = []

    while bboxes:
        bboxes = [bbox for bbox in bboxes if bbox[0] != bbox_main[0]
                  or intersection_over_union(torch.tensor(bbox_main[2:]),
                  torch.tensor(bbox[2:]), bbox_format) < iou_threshold]

        bboxes_nms.append(bbox_main)

    return bboxes_nms

def mean_average_precision(pred_bboxes, true_bboxes, num_classes=10, iou_threshold=0.5, bbox_format='corners'):
    '''
    Calculates the mean average precision.

    Inputs:
        true_bboxes (list): List of lists with all true bboxes.
        pred_bboxes (list): Similar to true_bboxes but contains all predicted bboxes,
            each with a format: [train_idx, class_pred, prob_score, x1, y1, x2, y2]
        num_classes (int): Number of classes.
        iou_threshold (float): IoU threshold to consider bboxes as correct.
        bbox_format (str): "corners" or "midpoint" used to specifiy what x1, y1, x2,
            y2 correspond to.

    Returns:
        float: Mean average precision (mAP) for given IoU threshold.
    '''
    avg_precisions = []
    eps = 1e-6

    for c in range(num_classes):
        detections = [pred_bbox for pred_bbox in pred_bboxes if pred_bbox[1] == c]
        ground_truths = [true_bbox for true_bbox in true_bboxes if true_bbox[1] == c]
        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)

        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            true_img = [bbox for bbox in ground_truths if bbox[0] == detection[0]]
            num_true = len(true_img)
            best_iou = 0

            for idx, gt in enumerate(true_img):
                iou = intersection_over_union(torch.tensor(detection[3:]),
                                              torch.tensor(gt[3:]),
                                              bbox_format)

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1

                else:
                    FP[detection_idx] = 1

            else:
                FP[detection_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)

        precisions = torch.div(TP_cumsum, (TP_cumsum + FP_cumsum + eps))
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.div(TP_cumsum, (total_true_bboxes + eps))
        recalls = torch.cat((torch.tensor([0]), recalls))

        avg_precisions.append(torch.trapz(precisions, recalls))

    mAP = sum(avg_precisions) / len(avg_precisions)
    return mAP
