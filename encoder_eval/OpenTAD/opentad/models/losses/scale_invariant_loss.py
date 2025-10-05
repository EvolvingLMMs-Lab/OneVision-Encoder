import torch
import torch.nn as nn
import torch.nn.functional as F
from ..builder import LOSSES
from ..utils.iou_tools import compute_iou_torch


@LOSSES.register_module()
class ScaleInvariantLoss(nn.Module):
    def __init__(self, pos_thresh=0.9, alpha=0.2):
        super().__init__()

        self.pos_thresh = pos_thresh
        self.alpha = alpha

    @torch.no_grad()
    def prepare_targets(self, proposals, gt_segments):
        proposals = proposals.to(gt_segments[0].device)  # [N,2]

        gt_ious = []
        gt_iou_weights = []
        for gt_segment in gt_segments:
            ious = compute_iou_torch(gt_segment, proposals)  # [N,M]
            gt_iou, max_index = torch.max(ious, dim=1)  # [N], [N]

            # gt iou weight
            gt_iou_weight = torch.ones_like(gt_iou)
            pos_mask = gt_iou > self.pos_thresh
            neg_mask = gt_iou <= self.pos_thresh
            pos_num = (ious > self.pos_thresh).sum(dim=0)  # [N], avoid 0

            gt_iou_weight[pos_mask] = 1 / pos_num[max_index[pos_mask]].clip(min=1)
            gt_iou_weight[neg_mask] = len(gt_segment) / torch.sum(neg_mask)

            gt_ious.append(gt_iou)
            gt_iou_weights.append(gt_iou_weight)

        gt_ious = torch.stack(gt_ious)
        gt_iou_weights = torch.stack(gt_iou_weights)
        return gt_ious, gt_iou_weights

    def forward(self, pred, proposal, gt_segments):
        gt_ious, gt_iou_weights = self.prepare_targets(proposal, gt_segments)

        # pred [B, N], this is after the valid mask
        pmask = (gt_ious > self.pos_thresh).float()
        nmask = (gt_ious <= self.pos_thresh).float()

        loss = F.binary_cross_entropy(pred, pmask, reduction="none")

        num_pos = torch.sum(pmask)
        num_neg = torch.sum(nmask)

        if (num_pos == 0) or (num_neg == 0):
            return torch.mean(loss)
        else:
            coef_pos = 0.5 * (num_pos + num_neg) / num_pos
            coef_neg = 0.5 * (num_pos + num_neg) / num_neg
            coef = self.alpha * pmask + (1 - self.alpha) * nmask
            loss = torch.sum(coef * loss * gt_iou_weights) / pred.shape[0]
        return loss, gt_ious

    def __repr__(self):
        return f"{self.__class__.__name__}(" f"pos_thresh={self.pos_thresh}, alpha={self.alpha})"
