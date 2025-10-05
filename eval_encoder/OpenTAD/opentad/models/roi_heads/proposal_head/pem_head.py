import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...builder import HEADS, build_loss
from ...utils.iou_tools import compute_iou_torch


@HEADS.register_module()
class PEMHead(nn.Module):
    r"""Proposal Evaluation Head, which is proposed in BMN.
    Input is a proposal map [B,C,D,T], then after several conv layers with kernel size 3x3.
    """

    def __init__(
        self,
        in_channels,
        feat_channels,
        num_classes,
        num_convs,
        kernel_size=3,
        loss=None,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.num_classes = num_classes
        self.num_convs = num_convs
        self.kernel_size = kernel_size

        # initialize layers
        self._init_layers()

        # loss
        self.cls_loss = build_loss(loss.cls_loss)
        self.reg_loss = build_loss(loss.reg_loss)

    def _init_layers(self):
        self.head = nn.Sequential()
        for i in range(self.num_convs):
            self.head.append(
                nn.Sequential(
                    nn.Conv2d(
                        self.in_channels if i == 0 else self.feat_channels,
                        self.feat_channels,
                        kernel_size=self.kernel_size,
                        stride=1,
                        padding=self.kernel_size // 3,
                    ),
                    nn.ReLU(inplace=True),
                )
            )
        self.head.append(
            nn.Conv2d(
                self.feat_channels,
                self.num_classes,
                kernel_size=1,
            )
        )

    def forward_train(self, proposal_feats, proposal_map, valid_mask, gt_segments):
        proposal_pred = self.head(proposal_feats)  # [B,2,D,T]

        loss = self.losses(proposal_pred, proposal_map, valid_mask, gt_segments)
        return loss

    def forward_test(self, proposal_feats):
        proposal_pred = self.head(proposal_feats)  # [B,2,D,T]
        return proposal_pred

    @torch.no_grad()
    def prepare_targets(self, proposals, gt_segments):
        proposals = proposals.to(gt_segments[0].device)

        gt_ious = []
        for gt_segment in gt_segments:
            gt_iou = compute_iou_torch(gt_segment, proposals)  # [B,N]
            gt_iou = torch.max(gt_iou, dim=1)[0]

            gt_ious.append(gt_iou)

        gt_ious = torch.stack(gt_ious)
        return gt_ious

    def losses(self, pred, proposal_map, valid_mask, gt_segments):
        # pred [B,2,D,T], proposal_map [D,T,2], valid_mask [D,T]

        pred = pred[:, :, valid_mask].sigmoid()  # [B, 2, N]

        gt_ious = self.prepare_targets(proposal_map[valid_mask, :], gt_segments)

        # classification loss - balanced BCE
        loss_cls = self.cls_loss(pred[:, 0, :], gt_ious)

        # regression loss - l2 loss
        loss_reg = self.reg_loss(pred[:, 1, :], gt_ious)

        losses = {"loss_cls": loss_cls, "loss_reg": loss_reg}
        return losses


@HEADS.register_module()
class TSIHead(PEMHead):
    def losses(self, pred, proposal_map, valid_mask, gt_segments):
        # pred [B,2,D,T], proposal_map [D,T,2], valid_mask [D,T]

        pred = pred[:, :, valid_mask].sigmoid()  # [B, 2, N]

        proposal = proposal_map[valid_mask, :]

        # classification loss - balanced BCE
        loss_cls, gt_ious = self.cls_loss(pred[:, 0, :], proposal, gt_segments)

        # regression loss - l2 loss
        loss_reg = self.reg_loss(pred[:, 1, :], gt_ious)

        losses = {"loss_cls": loss_cls, "loss_reg": loss_reg}
        return losses
