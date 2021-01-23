import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import Counter

def intersection_over_union(bboxes1, bboxes2, bbox_format='midpoint'):
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

    box1_area = torch.abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = torch.abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    union = box1_area + box2_area - intersection + 1e-6
    iou_val = intersection / union

    return iou_val

def non_max_suppression(bboxes, prob_threshold=0.2, iou_threshold=0.4, bbox_format='midpoint'):
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
        bbox_main = = bboxes.pop(0)
        bboxes = [bbox for bbox in bboxes if bbox[0] != bbox_main[0]
                  or intersection_over_union(torch.tensor(bbox_main[2:]),
                  torch.tensor(bbox[2:]), bbox_format) < iou_threshold]

        bboxes_nms.append(bbox_main)

    return bboxes_nms

def mean_average_precision(pred_bboxes, true_bboxes, iou_threshold=0.5, bbox_format='midpoint', num_classes=20):
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
                iou = intersection_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                    bbox_format
                )

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

        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + eps)
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = TP_cumsum / (total_true_bboxes + eps)
        recalls = torch.cat((torch.tensor([0]), recalls))

        avg_precisions.append(torch.trapz(precisions, recalls))

    mAP = sum(avg_precisions) / len(avg_precisions)
    return mAP

def plot_image(image, boxes):

    im = np.array(image)
    height, width, _ = im.shape

    fig, ax = plt.subplots()
    ax.imshow(im)

    for box in boxes:
        box = box[2:]
        assert len(box) == 4, 'box must only contain x, y, w, h dimensions'

        x0 = box[0] - box[2] / 2
        y0 = box[1] - box[3] / 2

        rect = patches.Rectangle(
            (x0 * width, y0 * height),
            box[2] * width,
            box[3] * height,
            linewidth = 1,
            edgecolor = 'r',
            facecolor = 'none'
        )

        ax.add_patch(rect)

    plt.show()

def get_bboxes(model, loader, prob_threshold=0.2, iou_threshold=0.4, bbox_format='midpoint'):

    all_pred_boxes = []
    all_true_boxes = []

    model.eval()
    train_idx = 0

    for b, data in enumerate(loader):
        images, labels = data
        #labels = labels.to(device)

        with torch.no_grad():
            predictions = model(images)

        batch_size = images.shape[0]
        pred_bboxes = cellboxes_to_boxes(predictions)
        true_bboxes = cellboxes_to_boxes(labels)

        for idx in range(batch_size):
            nms_boxes = non_max_suppression(
                pred_bboxes[idx], prob_threshold, iou_threshold, bbox_format
            )

            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)

            for box in true_bboxes[idx]:
                if box[1] > prob_threshold:
                    all_true_boxes.append([train_idx] + box)

            train_idx += 1

    model.train()
    return all_pred_boxes, all_true_boxes

def convert_cellboxes(predictions, S=7, B=2, C=20):

    batch_size = predictions.shape[0]
    predictions = predictions.reshape(batch_size, S, S, C + 5 * B)
    bboxes1 = predictions[..., C+1:C+5]
    bboxes2 = predictions[..., C+6:C+10]

    scores = torch.cat(
        (predictions[..., C].unsqueeze(0), predictions[..., C+5].unsqueeze(0)), dim=0
    )

    best_bbox = scores.argmax(0).unsqueeze(-1)
    best_bboxes = torch.mul(bboxes1, (1 - best_bbox)) + torch.mul(bboxes2, best_bbox)

    cell_indices = torch.arange(S).repeat(batch_size, S, 1).unsqueeze(-1)

    x = 1 / S * (best_bboxes[..., :1] + cell_indices)
    y = 1 / S * (best_bboxes[..., 1:2] + cell_indices.permute(0, 2, 1, 3))
    wy = 1 / S * best_bboxes[..., 2:4]

    converted_bboxes = torch.cat((x, y, wy), dim=-1)
    predicted_class = predictions[..., :C].argmax(-1).unsqueeze(-1)
    best_confidence = torch.max(predictions[..., C], predictions[..., C+5]).unsqueeze(-1)
    converted_preds = torch.cat((predicted_class, best_confidence, converted_bboxes), dim=-1)

    return converted_preds

def cellboxes_to_boxes(out, S=7):

    converted_pred = convert_cellboxes(out).reshape(out.shape[0], S * S, -1)
    converted_pred[..., 0] = converted_pred[..., 0].long()
    all_bboxes = []

    for ex_idx in range(out.shape[0]):
        bboxes = []

        for bbox_idx in range(S * S):
            bboxes.append([x.item() for x in converted_pred[ex_idx, bbox_idx, :]])

        all_bboxes.append(bboxes)

    return all_bboxes

def save_checkpoint(state, filename='my_checkpoint.pth.tar'):
    torch.save(state, filename)
    print('Checkpoint saved...')


def load_checkpoint(checkpoint, model, optimizer):
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    print('Checkpoint loaded...')


# IGNORE THESE
# FOR STORING ONLY
#image, labels = next(iter(train_loader))
#
#image = image[1, ...]
#labels = labels[1, ...]
#
#image = image[None, ...]
#labels = labels[None, ...]
#
#pred = model(image)
#
#batch_size = pred.shape[0]
#
#pred_bboxes = cellboxes_to_boxes(pred)
#true_bboxes = cellboxes_to_boxes(labels)
#
#all_pred_boxes = []
#all_true_boxes = []
#
#for idx in range(batch_size):
#    bboxes = [bbox for bbox in pred_bboxes[idx] if bbox[1] > 0.01]
#    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
#    bboxes_nms = []
#
#    while bboxes:
#        bbox_main = bboxes.pop(0)
#        bboxes = [bbox for bbox in bboxes if bbox[0] != bbox_main[0]
#                  or intersection_over_union(torch.tensor(bbox_main[2:]),
#                  torch.tensor(bbox[2:]), 'midpoint') < 0.4]
#
#        bboxes_nms.append(bbox_main)
#
#
#    #nms_boxes = non_max_suppression(
#    #    pred_bboxes[idx], prob_threshold=0.2, iou_threshold=0.4, bbox_format='midpoint'
#    #)
#
#    for nms_box in bboxes_nms:
#        all_pred_boxes.append([0] + nms_box)
#
#    for box in true_bboxes[idx]:
#        if box[1] > 0.2:
#            all_true_boxes.append([0] + box)
#
##image, labels = next(iter(train_loader))
#
#image = image[0, ...]
#labels = labels[0, ...]
#
##image = image[None, ...]
##labels = labels[None, ...]
#
#im = image.permute(1, 2, 0).numpy()
#
#fig, ax = plt.subplots()
#ax.imshow(im)
#height, width, _ = im.shape
#
#for box in all_pred_boxes:
##for box in subbox[..., 21:25]:
#    box = box[3:]
#    assert len(box) == 4, 'box must only contain x, y, w, h dimensions'
#
#    x0 = box[0] - box[2] / 2
#    y0 = box[1] - box[3] / 2
#
#    pred_rect = patches.Rectangle(
#        (x0 * width, y0 * height),
#        box[2] * width,
#        box[3] * height,
#        linewidth = 1,
#        edgecolor = 'r',
#        facecolor = 'none'
#    )
#
#for box in all_true_boxes:
##for box in subbox[..., 21:25]:
#    box = box[3:]
#    assert len(box) == 4, 'box must only contain x, y, w, h dimensions'
#
#    x0 = box[0] - box[2] / 2
#    y0 = box[1] - box[3] / 2
#
#    true_rect = patches.Rectangle(
#        (x0 * width, y0 * height),
#        box[2] * width,
#        box[3] * height,
#        linewidth = 1,
#        edgecolor = 'b',
#        facecolor = 'none'
#    )
#
#ax.add_patch(pred_rect)
#ax.add_patch(true_rect)
#
#plt.show()
