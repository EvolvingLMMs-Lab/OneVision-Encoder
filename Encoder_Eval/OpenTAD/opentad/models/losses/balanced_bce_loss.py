import torch
import torch.nn as nn
import torch.nn.functional as F
from ..builder import LOSSES


@LOSSES.register_module()
class BalancedBCELoss(nn.Module):
    """Balanced Binary Cross-Entropy Loss. Usually used in temporal-evaluation-module."""

    def __init__(self, pos_thresh=0.5):
        super().__init__()

        self.pos_thresh = pos_thresh

    def forward(self, pred, gt):
        pmask = (gt > self.pos_thresh).float()
        nmask = (gt <= self.pos_thresh).float()

        loss = F.binary_cross_entropy(pred, pmask, reduction="none")

        num_pos = torch.sum(pmask)
        num_neg = torch.sum(nmask)

        if (num_pos == 0) or (num_neg == 0):
            return torch.mean(loss)
        else:
            coef_pos = 0.5 * (num_pos + num_neg) / num_pos
            coef_neg = 0.5 * (num_pos + num_neg) / num_neg
            loss = torch.mean((coef_pos * pmask + coef_neg * nmask) * loss)
        return loss

    def __repr__(self):
        return f"{self.__class__.__name__}(" f"pos_thresh={self.pos_thresh})"
