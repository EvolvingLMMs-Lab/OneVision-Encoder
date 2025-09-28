import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .convfc_head import ConvFC, FCHead
from ...builder import HEADS
from ...utils.iou_tools import compute_batched_iou_torch


@HEADS.register_module()
class ETADHead(nn.Module):
    def __init__(
        self,
        in_channels,
        roi_size,
        feat_channels,
        fcs_num,
        fcs_channels,
        loss=None,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.roi_size = roi_size
        self.feat_channels = feat_channels
        self.fcs_num = fcs_num
        self.fcs_channels = fcs_channels

        # initialize layers
        self._init_layers()

        # loss
        self.pos_iou_thresh = loss.pos_iou_thresh
        self.cls_weight = loss.cls_weight
        self.reg_weight = loss.reg_weight
        self.boundary_weight = loss.boundary_weight

    def _init_layers(self):
        """Initialize layers of the head."""
        self._init_shared()
        self._init_heads()

    def _init_shared(self):
        """Initialize shared conv fc layers"""
        self.shared_conv_fc = ConvFC(
            in_channels=self.in_channels,
            roi_size=self.roi_size,
            convs_num=0,
            fcs_num=1,
            fcs_channel=self.feat_channels,
        )

    def _init_heads(self):
        # extended features
        self.iou_head = FCHead(
            in_channels=self.feat_channels,
            out_channels=2,
            fcs_num=self.fcs_num,
            fcs_channel=self.fcs_channels,
        )
        self.se_head = FCHead(
            in_channels=self.feat_channels,
            out_channels=2,
            fcs_num=self.fcs_num,
            fcs_channel=self.fcs_channels,
        )
        self.reg_head = FCHead(
            in_channels=self.feat_channels,
            out_channels=4,
            fcs_num=self.fcs_num,
            fcs_channel=self.fcs_channels,
        )

    def forward_train(self, proposal_feats, proposal_list, gt_starts, gt_ends, batch_gt_segment):
        bs, N, _, _ = proposal_feats.shape
        proposal_feats = proposal_feats.flatten(0, 1)

        # proposal feature
        proposal_feats = self.shared_conv_fc(proposal_feats)

        # head
        iou_out = self.iou_head(proposal_feats).sigmoid()  # [bs*N, 2]
        reg_out = self.reg_head(proposal_feats)  # [bs*N, 4]
        se_out = self.se_head(proposal_feats).sigmoid()  # [bs*N, 2]

        iou_out = iou_out.unflatten(0, (bs, N))  # [bs,N,2]
        reg_out = reg_out.unflatten(0, (bs, N))  # [bs,N,4]
        se_out = se_out.unflatten(0, (bs, N))  # [bs,N,2]

        refined_proposals = self.refine_proposals(proposal_list, reg_out)
        loss = self.losses(iou_out, reg_out, se_out, proposal_list, gt_starts, gt_ends, batch_gt_segment)
        return loss, refined_proposals

    def forward_test(self, proposal_feats, proposal_list):
        bs, N, _, _ = proposal_feats.shape
        proposal_feats = proposal_feats.flatten(0, 1)

        # proposal feature
        proposal_feats = self.shared_conv_fc(proposal_feats)

        # head
        iou_out = self.iou_head(proposal_feats).sigmoid()  # [bs*N, 2]
        reg_out = self.reg_head(proposal_feats)  # [bs*N, 4]
        se_out = self.se_head(proposal_feats).sigmoid()  # [bs*N, 2]

        iou_out = iou_out.unflatten(0, (bs, N))  # [bs,N,2]
        reg_out = reg_out.unflatten(0, (bs, N))  # [bs,N,4]
        se_out = se_out.unflatten(0, (bs, N))  # [bs,N,2]

        refined_proposals = self.refine_proposals(proposal_list, reg_out)
        return refined_proposals, iou_out

    @torch.no_grad()
    def refine_proposals(self, anchors, regs):
        regs = regs.view(anchors.shape[0], -1, regs.shape[-1]).detach()  # [B,K,6]

        xmins = anchors[:, :, 0]
        xmaxs = anchors[:, :, 1]
        xlens = xmaxs - xmins
        xcens = (xmins + xmaxs) * 0.5

        # refine anchor by start end
        xlens1 = xlens + xlens * (regs[:, :, 1] - regs[:, :, 0])
        xcens1 = xcens + xlens * (regs[:, :, 0] + regs[:, :, 1]) * 0.5
        xmins1 = xcens1 - xlens1 * 0.5
        xmaxs1 = xcens1 + xlens1 * 0.5

        # refine anchor by center width
        xcens2 = xcens + regs[:, :, 2] * xlens
        xlens2 = xlens * torch.exp(regs[:, :, 3])
        xmins2 = xcens2 - xlens2 * 0.5
        xmaxs2 = xcens2 + xlens2 * 0.5

        nxmin = (xmins1 + xmins2) * 0.5
        nxmax = (xmaxs1 + xmaxs2) * 0.5
        new_anchors = torch.stack((nxmin, nxmax), dim=2)
        return new_anchors

    @torch.no_grad()
    def prepare_targets(self, proposals, gt_starts, gt_ends, batch_gt_segment):
        batch_gt_iou = compute_batched_iou_torch(batch_gt_segment, proposals)  # [B,K]
        batch_gt_reg = self._get_gt_regs(batch_gt_segment, proposals)  # [B,K,6]
        batch_gt_cls_s, batch_gt_cls_e = self._get_gt_bounary_cls(gt_starts, gt_ends, proposals)
        return batch_gt_iou, batch_gt_reg, batch_gt_cls_s, batch_gt_cls_e

    def _get_gt_regs(self, gt, anchors):
        # gt_segment [B,K,2] anchors_init[B,K,2]
        anchor_len = torch.clamp(anchors[:, :, 1] - anchors[:, :, 0], min=1e-6)
        gt_len = torch.clamp(gt[:, :, 1] - gt[:, :, 0], min=1e-6)
        delta_s = (gt[:, :, 0] - anchors[:, :, 0]) / anchor_len
        delta_e = (gt[:, :, 1] - anchors[:, :, 1]) / anchor_len
        delta_c = (delta_s + delta_e) * 0.5
        delta_w = torch.log(gt_len / anchor_len + 1e-6)
        delta = torch.stack([delta_s, delta_e, delta_c, delta_w], dim=-1)
        return delta

    def _get_gt_bounary_cls(self, gt_start, gt_end, anchors):
        # gt_start [B,200] anchors_init[B,K,2]
        anchors_start = anchors[..., 0].long().clamp(min=0, max=gt_start.shape[1] - 1)
        anchors_end = anchors[..., 1].long().clamp(min=0, max=gt_end.shape[1] - 1)
        gt_cls_s = torch.gather(gt_start, 1, anchors_start)
        gt_cls_e = torch.gather(gt_end, 1, anchors_end)
        return gt_cls_s, gt_cls_e

    def losses(self, iou, reg, se, proposal_list, gt_starts, gt_ends, batch_gt_segment):
        gt_iou, gt_reg, gt_cls_s, gt_cls_e = self.prepare_targets(proposal_list, gt_starts, gt_ends, batch_gt_segment)

        # iou
        loss_iou_cls = bl_sample_loss(iou[..., 0], gt_iou, pos_thresh=0.9)
        loss_iou_reg = l2_sample_loss(iou[..., 1], gt_iou, high_thresh=0.7, low_thresh=0.3)
        loss_iou = loss_iou_cls * self.cls_weight + loss_iou_reg * self.reg_weight

        # boundary regression
        loss_reg_start = smoothL1_regress_loss(reg[..., 0], gt_iou, gt_reg[..., 0], thresh=self.pos_iou_thresh)
        loss_reg_end = smoothL1_regress_loss(reg[..., 1], gt_iou, gt_reg[..., 1], thresh=self.pos_iou_thresh)
        loss_reg_center = smoothL1_regress_loss(reg[..., 2], gt_iou, gt_reg[..., 2], thresh=self.pos_iou_thresh)
        loss_reg_width = smoothL1_regress_loss(reg[..., 3], gt_iou, gt_reg[..., 3], thresh=self.pos_iou_thresh)
        loss_reg = self.boundary_weight * (loss_reg_start + loss_reg_end + loss_reg_center + loss_reg_width)

        # boundary classification
        loss_cls_start = bl_sample_loss(se[..., 0], gt_cls_s, pos_thresh=0.5)
        loss_cls_end = bl_sample_loss(se[..., 1], gt_cls_e, pos_thresh=0.5)
        loss_cls = 0.5 * (loss_cls_start + loss_cls_end)

        # total loss
        loss_pem = loss_iou + loss_reg + loss_cls

        losses = {"loss_pem": loss_pem}
        return losses


def bl_sample_loss(output, gt_iou, pos_thresh=0.9):
    gt_iou = gt_iou.cuda()

    pmask = (gt_iou > pos_thresh).float()
    nmask = (gt_iou <= pos_thresh).float()

    num_pos = torch.sum(pmask)
    num_neg = torch.sum(nmask)

    if num_pos == 0:  # in case of nan
        loss = -torch.mean(torch.log(1.0 - output + 1e-6))
        return loss

    r_l = num_pos / num_neg
    nmask_sample = torch.rand(nmask.shape).cuda()
    nmask_sample = nmask_sample * nmask
    nmask_sample = (nmask_sample > (1 - r_l)).float()

    loss_pos = pmask * torch.log(output + 1e-6)
    loss_neg = nmask_sample * torch.log(1.0 - output + 1e-6)
    loss = -torch.sum(loss_pos + loss_neg) / (num_pos + torch.sum(nmask_sample))
    return loss


def l2_sample_loss(output, gt_iou, high_thresh=0.7, low_thresh=0.3):
    gt_iou = gt_iou.cuda()

    u_hmask = (gt_iou > high_thresh).float()
    u_mmask = ((gt_iou <= high_thresh) & (gt_iou > low_thresh)).float()
    u_lmask = (gt_iou <= low_thresh).float()

    num_h = torch.sum(u_hmask)
    num_m = torch.sum(u_mmask)
    num_l = torch.sum(u_lmask)

    if num_h == 0:  # in case of nan
        loss = F.mse_loss(output, gt_iou, reduction="none")
        loss = torch.mean(loss)
        return loss

    r_m = num_h / num_m
    u_smmask = torch.rand(u_hmask.shape).cuda()
    u_smmask = u_smmask * u_mmask
    u_smmask = (u_smmask > (1 - r_m)).float()

    r_l = num_h / num_l
    u_slmask = torch.rand(u_hmask.shape).cuda()
    u_slmask = u_slmask * u_lmask
    u_slmask = (u_slmask > (1 - r_l)).float()

    mask = u_hmask + u_smmask + u_slmask
    loss = F.mse_loss(output, gt_iou, reduction="none")
    loss = torch.sum(loss * mask) / torch.sum(mask)
    return loss


def smoothL1_regress_loss(output, gt_iou, gt_reg, thresh=0.7):
    mask = (gt_iou > thresh).float().cuda()

    if torch.sum(mask) == 0:  # not have positive sample
        return torch.Tensor([0]).sum().cuda()

    loss = F.smooth_l1_loss(output, gt_reg, reduction="none")
    loss = torch.sum(loss * mask) / torch.sum(mask)
    return loss
