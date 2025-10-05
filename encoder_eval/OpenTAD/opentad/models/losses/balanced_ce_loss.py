import torch
import torch.nn as nn
import torch.nn.functional as F
from ..builder import LOSSES


@LOSSES.register_module()
class BalancedCELoss(object):
    """Used in VSGN"""

    def __init__(self) -> None:
        self.loss = torch.nn.CrossEntropyLoss(reduction="none")

    def __call__(self, cls_pred, cls_labels):
        pmask = (cls_labels > 0).float()
        nmask = (cls_labels == 0).float()
        num_pos = torch.sum(pmask)
        num_neg = torch.sum(nmask)

        loss = self.loss(cls_pred, cls_labels.to(torch.long))

        pos_loss = torch.sum(loss * pmask) / num_pos
        neg_loss = torch.sum(loss * nmask) / num_neg

        total_loss = pos_loss + neg_loss
        return total_loss
