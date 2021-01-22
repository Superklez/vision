import torch
import torch.nn as nn
from utils import intersection_over_union

from collections import Counter

class YOLOloss(nn.Module):

    def __init__(self, S=7, B=2, C=20):
        super().__init__()
        self.mse = nn.MSELoss(reduction='sum') # the original paper does not take the mean
#        self.llloss = nn.CrossEntropyLoss()
        self.S = S
        self.B = B
        self.C = C
        self.lambda_coord = 5
        self.lambda_noobj = 0.5
        self.eps = 1e-6

    def forward(self, pred, true):
        pred = pred.reshape(-1, self.S, self.S, self.C + self.B * 5)

#        ious = []
#        C_loc = self.C
#        for b in range(B):
#            b += 1
#            ious.append(intersection_over_union(
#                pred[..., C_loc+1:C_loc+5],
#                true[..., C_loc+1:C_loc+5],
#                'midpoint')
#            )
#            ious[f'b{b}_iou'] = iou
#            ious.append(iou)
#            C_loc += 5
#        ious = torch.cat([val.unsqueeze(0) for val in ious], dim=0)

        iou_b1 = intersection_over_union(pred[..., 21:25], true[..., 21:25])
        iou_b2 = intersection_over_union(pred[..., 26:30], true[..., 21:25])
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2].unsqueeze(0), dim=0)

        _, best_bbox = torch.max(ious, dim=0)
        Iobj_i = true[..., 20].unsqueeze(3) # Iobj_i, identity of object i

        # ---------------
        # BOX COORDINATES
        # ---------------
        bbox_pred = torch.mul(
            Iobj_i, (
                torch.mul(best_bbox, pred[..., 26:30]) + torch.mul((1 - best_bbox), pred[..., 21:25])
            )
        )
        bbox_pred[..., 2:4] = torch.mul(
            torch.sign(bbox_pred[..., 2:4]) * torch.sqrt(
                torch.abs(bbox_pred[..., 2:4]) + self.eps
            )
        )

        bbox_true = torch.mul(Iobj_i, true[..., 21:25])
        bbox_true[..., 2:4] = torch.sqrt(bbox_true[..., 2:4])

        bbox_loss = self.mse(
            torch.flatten(bbox_pred, end_dim=-2),
            torch.flatten(bbox_true, end_dim=-2)
        )

        # -----------
        # OBJECT LOSS
        # -----------
        pred_bbox = (
            torch.mul(best_bbox, pred[..., 25:26])
            + torch.mul((1 - best_bbox), pred[..., 20:21])
        )
        object_loss = self.mse(
            torch.flatten(torch.mul(Iobj_i, pred_bbox)),
            torch.flatten(torch.mul(Iobj_i, true[..., 20:21]))
        )

        # NO OBJECT LOSS
        no_object_loss = self.mse(
            torch.flatten(torch.mul((1 - Iobj_i), pred[..., 20:21]), start_dim=1),
            torch.flatten(torch.mul((1 - Iobj_i), true[..., 20:21]), start_dim=1)
        )
        no_object_loss += self.mse(
            torch.flatten(torch.mul((1 - Iobj_i), pred[..., 25:26]), start_dim=1),
            torch.flatten(torch.mul((1 - Iobj_i), true[..., 20:21]), start_dim=1)
        )

        # CLASS LOSS
        class_loss = self.mse(
            torch.flatten(torch.mul(Iobj_i, pred[..., :20]), end_dim=-2),
            torch.flatten(torch.mul(Iobj_i, true[..., :20]), end_dim=-2)
        )

        # TOTAL LOSS
        loss = (
            self.lambda_coord * box_loss
            + object_loss
            + self.lambda_noobj * no_object_loss
            + class_loss
        )

        return loss
