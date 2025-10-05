import torch
import torch.nn as nn
import torch.nn.functional as F
from ..builder import LOSSES


@LOSSES.register_module()
class SmoothL1Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        reduction: str = "sum",
        eps: float = 1e-8,
    ) -> torch.Tensor:
        loss = F.smooth_l1_loss(inputs, targets, reduction="none")

        if reduction == "mean":
            loss = loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
        elif reduction == "sum":
            loss = loss.sum()
        return loss
