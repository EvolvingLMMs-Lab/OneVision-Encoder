import torch
import torch.nn as nn
import torch.nn.functional as F
from ..builder import HEADS
from ..bricks import ConvModule, GCNeXt
from ..losses.balanced_bce_loss import BalancedBCELoss
from ..utils.iou_tools import compute_ioa_torch


@HEADS.register_module()
class TemporalEvaluationHead(nn.Module):
    def __init__(self, in_channels, num_classes, conv_cfg=None, loss=None, shared=False):
        super().__init__()

        self.shared = shared

        if self.shared:  # shared backbone
            self.tem = nn.Sequential(
                ConvModule(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    act_cfg=dict(type="relu"),
                ),
                ConvModule(
                    in_channels,
                    num_classes,
                    kernel_size=1,
                    stride=1,
                ),
            )

        else:  # not shared backbone
            self.tem = nn.ModuleList([])
            for _ in range(num_classes):
                layer = nn.Sequential(
                    ConvModule(
                        in_channels,
                        in_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        conv_cfg=conv_cfg,
                        act_cfg=dict(type="relu"),
                    ),
                    ConvModule(
                        in_channels,
                        1,
                        kernel_size=1,
                        stride=1,
                    ),
                )
                self.tem.append(layer)

        self.gt_type = loss["gt_type"]
        for gt in self.gt_type:
            assert gt in ["startness", "endness", "actionness"]
        self.tem_loss = BalancedBCELoss(pos_thresh=loss["pos_thresh"])

    def forward_train(self, x, masks, gt_segments, **kwargs):
        if self.shared:
            x = self.tem(x)
        else:
            x = torch.cat([layer(x) for layer in self.tem], dim=1)

        losses = {"loss_tem": self.losses(x, gt_segments)}

        proposal_list = None
        return losses, proposal_list

    def forward_test(self, x, masks, **kwargs):
        if self.shared:
            x = self.tem(x)
        else:
            x = torch.cat([layer(x) for layer in self.tem], dim=1)
        return x

    @torch.no_grad()
    def prepare_targets(self, gt_segments, tscale):
        gt_starts = []
        gt_ends = []
        gt_actions = []

        temporal_anchor = torch.stack((torch.arange(0, tscale), torch.arange(1, tscale + 1)), dim=1)
        temporal_anchor = temporal_anchor.to(gt_segments[0].device)

        for gt_segment in gt_segments:
            gt_xmins = gt_segment[:, 0]
            gt_xmaxs = gt_segment[:, 1]

            gt_start_bboxs = torch.stack((gt_xmins - 3.0 / 2, gt_xmins + 3.0 / 2), dim=1)
            gt_end_bboxs = torch.stack((gt_xmaxs - 3.0 / 2, gt_xmaxs + 3.0 / 2), dim=1)

            gt_start = compute_ioa_torch(gt_start_bboxs, temporal_anchor)
            gt_start = torch.max(gt_start, dim=1)[0]

            gt_end = compute_ioa_torch(gt_end_bboxs, temporal_anchor)  # [T, N]
            gt_end = torch.max(gt_end, dim=1)[0]

            gt_action = compute_ioa_torch(gt_segment, temporal_anchor)  # [T, N]
            gt_action = torch.max(gt_action, dim=1)[0]

            gt_starts.append(gt_start)
            gt_ends.append(gt_end)
            gt_actions.append(gt_action)

        gt_starts = torch.stack(gt_starts)
        gt_ends = torch.stack(gt_ends)
        gt_actions = torch.stack(gt_actions)

        gts = []
        if "startness" in self.gt_type:
            gts.append(gt_starts)

        if "endness" in self.gt_type:
            gts.append(gt_ends)

        if "actionness" in self.gt_type:
            gts.append(gt_actions)

        return gts

    def losses(self, pred, gt_segments):
        # pred: [B,2,T]
        # gt_segment: list

        gts = self.prepare_targets(gt_segments, tscale=pred.shape[-1])
        assert len(gts) == pred.shape[1]

        pred = pred.sigmoid()  #  need to sigmoid

        loss = 0
        for i, gt in enumerate(gts):
            loss += self.tem_loss(pred[:, i, :], gt)
        return loss


@HEADS.register_module()
class GCNextTemporalEvaluationHead(TemporalEvaluationHead):
    def __init__(self, in_channels, num_classes, loss=None, shared=False):
        super().__init__(in_channels, num_classes, loss=loss, shared=shared)

        if self.shared:
            self.tem = nn.Sequential(
                GCNeXt(in_channels, in_channels, k=3, groups=32),
                ConvModule(in_channels, num_classes, kernel_size=1, stride=1),
            )

        else:  # not shared backbone
            self.tem = nn.ModuleList([])
            for _ in range(num_classes):
                layer = nn.Sequential(
                    GCNeXt(in_channels, in_channels, k=3, groups=32),
                    ConvModule(in_channels, 1, kernel_size=1, stride=1),
                )
                self.tem.append(layer)


@HEADS.register_module()
class LocalGlobalTemporalEvaluationHead(nn.Module):
    def __init__(self, in_channels, padding=0, loss=None):
        super().__init__()

        # ------ local branch  ------
        self.local_conv1d_s = nn.Conv1d(in_channels, 256, kernel_size=3, padding=1, groups=4)
        self.local_conv1d_s_out = nn.Conv1d(256, 1, kernel_size=1)

        self.local_conv1d_e = nn.Conv1d(in_channels, 256, kernel_size=3, padding=1, groups=4)
        self.local_conv1d_e_out = nn.Conv1d(256, 1, kernel_size=1)

        # ------ global branch ------
        channels = [128, 256, 512, 1024]
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        self.x1_1 = nn.Conv1d(in_channels, channels[0], kernel_size=3, stride=1, padding=1, groups=4)
        self.x2_1 = nn.Conv1d(channels[0], channels[1], kernel_size=3, stride=1, padding=1, groups=4)
        self.x3_1 = nn.Conv1d(channels[1], channels[2], kernel_size=3, stride=1, padding=1, groups=4)
        self.x4_1 = nn.Conv1d(channels[2], channels[3], kernel_size=3, stride=1, padding=1, groups=4)

        self.up41_to_32 = nn.ConvTranspose1d(
            channels[3],
            channels[2],
            kernel_size=2,
            stride=2,
            output_padding=padding,
            groups=4,
        )

        self.x3_2 = nn.Conv1d(channels[2] * 2, channels[2], kernel_size=3, stride=1, padding=1, groups=4)

        self.up31_to_22 = nn.ConvTranspose1d(channels[2], channels[1], kernel_size=2, stride=2, groups=4)
        self.x2_2 = nn.Conv1d(channels[1] * 2, channels[1], kernel_size=3, stride=1, padding=1, groups=4)

        self.up32_to_23 = nn.ConvTranspose1d(channels[2], channels[1], kernel_size=2, stride=2, groups=4)
        self.x2_3 = nn.Conv1d(channels[1] * 3, channels[1], kernel_size=3, stride=1, padding=1, groups=4)

        self.up21_to_12 = nn.ConvTranspose1d(channels[1], channels[0], kernel_size=2, stride=2, groups=4)
        self.x1_2 = nn.Conv1d(channels[0] * 2, channels[0], kernel_size=3, stride=1, padding=1, groups=4)

        self.up22_to_13 = nn.ConvTranspose1d(channels[1], channels[0], kernel_size=2, stride=2, groups=4)
        self.x1_3 = nn.Conv1d(channels[0] * 3, channels[0], kernel_size=3, stride=1, padding=1, groups=4)

        self.up23_to_14 = nn.ConvTranspose1d(channels[1], channels[0], kernel_size=2, stride=2, groups=4)
        self.x1_4 = nn.Conv1d(channels[0] * 4, channels[0], kernel_size=3, stride=1, padding=1, groups=4)

        self.global_s_out = nn.Conv1d(channels[0], 1, kernel_size=1)
        self.global_e_out = nn.Conv1d(channels[0], 1, kernel_size=1)

        self.tem_loss = BalancedBCELoss(pos_thresh=loss["pos_thresh"])

    def forward_train(self, x, masks, gt_segments, **kwargs):
        # ------ local branch  ------
        tbd_local_s = F.relu(self.local_conv1d_s(x))
        tbd_local_s_out = self.local_conv1d_s_out(tbd_local_s).squeeze(1)

        tbd_local_e = F.relu(self.local_conv1d_e(x))
        tbd_local_e_out = self.local_conv1d_e_out(tbd_local_e).squeeze(1)

        # ------ global branch ------
        x1_1 = self.x1_1(x)
        x2_1 = self.x2_1(self.pool(x1_1))
        x3_1 = self.x3_1(self.pool(x2_1))
        x4_1 = self.x4_1(self.pool(x3_1))

        # layer 3
        x3_2 = self.x3_2(torch.cat((x3_1, self.up41_to_32(x4_1)), dim=1))

        # layer 2
        x2_2 = self.x2_2(torch.cat((x2_1, self.up31_to_22(x3_1)), dim=1))
        x2_3 = self.x2_3(torch.cat((x2_1, x2_2, self.up32_to_23(x3_2)), dim=1))

        # layer 1
        x1_2 = self.x1_2(torch.cat((x1_1, self.up21_to_12(x2_1)), dim=1))
        x1_3 = self.x1_3(torch.cat((x1_1, x1_2, self.up22_to_13(x2_2)), dim=1))
        x1_4 = self.x1_4(torch.cat((x1_1, x1_2, x1_3, self.up23_to_14(x2_3)), dim=1))

        tbd_global_s_out = self.global_s_out(x1_4).squeeze(1)
        tbd_global_e_out = self.global_e_out(x1_4).squeeze(1)

        tbd_out = (tbd_local_s_out, tbd_local_e_out, tbd_global_s_out, tbd_global_e_out)

        losses = {"loss_tem": self.losses(tbd_out, gt_segments)}

        proposal_list = None
        return losses, proposal_list

    def forward_test(self, x, masks, **kwargs):
        # ------ local branch  ------
        tbd_local_s = F.relu(self.local_conv1d_s(x))
        tbd_local_s_out = self.local_conv1d_s_out(tbd_local_s).squeeze(1)

        tbd_local_e = F.relu(self.local_conv1d_e(x))
        tbd_local_e_out = self.local_conv1d_e_out(tbd_local_e).squeeze(1)

        # ------ global branch ------
        x1_1 = self.x1_1(x)
        x2_1 = self.x2_1(self.pool(x1_1))
        x3_1 = self.x3_1(self.pool(x2_1))
        x4_1 = self.x4_1(self.pool(x3_1))

        # layer 3
        x3_2 = self.x3_2(torch.cat((x3_1, self.up41_to_32(x4_1)), dim=1))

        # layer 2
        x2_2 = self.x2_2(torch.cat((x2_1, self.up31_to_22(x3_1)), dim=1))
        x2_3 = self.x2_3(torch.cat((x2_1, x2_2, self.up32_to_23(x3_2)), dim=1))

        # layer 1
        x1_2 = self.x1_2(torch.cat((x1_1, self.up21_to_12(x2_1)), dim=1))
        x1_3 = self.x1_3(torch.cat((x1_1, x1_2, self.up22_to_13(x2_2)), dim=1))
        x1_4 = self.x1_4(torch.cat((x1_1, x1_2, x1_3, self.up23_to_14(x2_3)), dim=1))

        tbd_global_s_out = self.global_s_out(x1_4).squeeze(1)
        tbd_global_e_out = self.global_e_out(x1_4).squeeze(1)

        tbd_out = (tbd_local_s_out, tbd_local_e_out, tbd_global_s_out, tbd_global_e_out)

        return tbd_out

    @torch.no_grad()
    def prepare_targets(self, gt_segments, tscale):
        gt_starts = []
        gt_ends = []

        temporal_anchor = torch.stack((torch.arange(0, tscale), torch.arange(1, tscale + 1)), dim=1)
        temporal_anchor = temporal_anchor.to(gt_segments[0].device)

        for gt_segment in gt_segments:
            gt_xmins = gt_segment[:, 0]
            gt_xmaxs = gt_segment[:, 1]

            gt_start_bboxs = torch.stack((gt_xmins - 3.0 / 2, gt_xmins + 3.0 / 2), dim=1)
            gt_end_bboxs = torch.stack((gt_xmaxs - 3.0 / 2, gt_xmaxs + 3.0 / 2), dim=1)

            gt_start = compute_ioa_torch(gt_start_bboxs, temporal_anchor)
            gt_start = torch.max(gt_start, dim=1)[0]

            gt_end = compute_ioa_torch(gt_end_bboxs, temporal_anchor)  # [T, N]
            gt_end = torch.max(gt_end, dim=1)[0]

            gt_starts.append(gt_start)
            gt_ends.append(gt_end)

        gt_starts = torch.stack(gt_starts)
        gt_ends = torch.stack(gt_ends)
        return gt_starts, gt_ends

    def losses(self, pred, gt_segments):
        tbd_ls, tbd_le, tbd_gs, tbd_ge = pred
        gt_starts, gt_ends = self.prepare_targets(gt_segments, tscale=tbd_ls.shape[-1])

        loss_tbd_ls = self.tem_loss(tbd_ls.sigmoid(), gt_starts)
        loss_tbd_le = self.tem_loss(tbd_le.sigmoid(), gt_ends)

        loss_tbd_gs = self.tem_loss(tbd_gs.sigmoid(), gt_starts)
        loss_tbd_ge = self.tem_loss(tbd_ge.sigmoid(), gt_ends)

        loss_tbd_local = loss_tbd_ls + loss_tbd_le
        loss_tbd_global = loss_tbd_gs + loss_tbd_ge

        loss = 0.5 * (loss_tbd_local + loss_tbd_global)
        return loss
