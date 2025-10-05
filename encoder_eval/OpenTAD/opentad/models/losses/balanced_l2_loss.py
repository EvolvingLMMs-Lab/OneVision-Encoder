import torch
import torch.nn as nn
import torch.nn.functional as F
from ..builder import LOSSES


@LOSSES.register_module()
class BalancedL2Loss(nn.Module):
    def __init__(self, high_thresh=0.7, low_thresh=0.3, weight=1.0):
        super().__init__()

        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.weight = weight

    def forward(self, pred, gt):
        h_mask = (gt > self.high_thresh).float()
        m_mask = ((gt <= self.high_thresh) & (gt > self.low_thresh)).float()
        l_mask = (gt <= self.low_thresh).float()

        num_h = torch.sum(h_mask)
        num_m = torch.sum(m_mask)
        num_l = torch.sum(l_mask)

        if num_h == 0:  # in case of nan
            loss = F.mse_loss(pred, gt, reduction="mean")
            return loss

        r_m = num_h / num_m
        m_mask_sample = torch.rand(m_mask.shape).to(gt.device) * m_mask
        m_mask_sample = (m_mask_sample > (1 - r_m)).float()

        r_l = num_h / num_l
        l_mask_sample = torch.rand(l_mask.shape).to(gt.device) * l_mask
        l_mask_sample = (l_mask_sample > (1 - r_l)).float()

        mask = h_mask + m_mask_sample + l_mask_sample
        loss = F.mse_loss(pred, gt, reduction="none")
        loss = torch.sum(loss * mask) / torch.sum(mask)

        loss *= self.weight
        return loss

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"high_thresh={self.high_thresh}, low_thresh={self.low_thresh}, weight={self.weight})"
        )
