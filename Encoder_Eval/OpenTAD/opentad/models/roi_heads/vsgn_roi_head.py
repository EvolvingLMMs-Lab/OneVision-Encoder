import torch
import torch.nn as nn
from ..dense_heads.vsgn_rpn_head import prepare_targets, giou_loss
from ..builder import HEADS, build_roi_extractor


@HEADS.register_module()
class VSGNRoIHead(nn.Module):
    def __init__(
        self,
        in_channels=256,
        iou_thr=0.7,
        roi_extractor=None,
        loss_loc=None,
    ):
        super(VSGNRoIHead, self).__init__()
        self.in_channels = in_channels
        self.iou_thr = iou_thr

        self.roi_extractor = build_roi_extractor(roi_extractor)
        self.get_loc_towers()

        # maintain an EMA of #foreground to stabilize the loss normalizer
        # useful for small mini-batch training
        self.loss_normalizer_reg = 100
        self.loss_normalizer_momentum = 0.9

    def get_loc_towers(self):
        self.start_conv = nn.Sequential(
            nn.Conv1d(
                in_channels=self.in_channels,
                out_channels=self.in_channels,
                kernel_size=3,
                stride=2,
                padding=0,
                groups=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=self.in_channels, out_channels=1, kernel_size=1),
        )

        self.end_conv = nn.Sequential(
            nn.Conv1d(
                in_channels=self.in_channels,
                out_channels=self.in_channels,
                kernel_size=3,
                stride=2,
                padding=0,
                groups=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=self.in_channels, out_channels=1, kernel_size=1),
        )

    def forward_train(self, feats, roi_bounds, gt_segments=None, gt_labels=None, return_loss=True):
        # extract start, end corner features
        roi_feats = self.roi_extractor(feats[0], roi_bounds)  #  bs*num_props_v, C, temporal (6)
        start_feats, end_feats = torch.chunk(roi_feats, 2, dim=-1)

        # get offsets
        start_offsets = self.start_conv(start_feats).squeeze(-1)
        end_offsets = self.end_conv(end_feats).squeeze(-1)

        # refine proposals
        B, N = roi_bounds.shape[:2]
        loc_pred = torch.stack(
            [
                roi_bounds[:, :, 0] + start_offsets.reshape(B, N),
                roi_bounds[:, :, 1] + end_offsets.reshape(B, N),
            ],
            dim=-1,
        )

        if return_loss:
            # Special GT for VSGN:
            # 1) Labels shift by including background = 0;
            # 2) gt_segments has the third dimension as gt_labels
            gts = [torch.cat((gt_b, gt_l[:, None] + 1), dim=-1) for gt_b, gt_l in zip(gt_segments, gt_labels)]
            return self.cal_loc_loss(loc_pred, gts, roi_bounds)
        else:
            return loc_pred

    def forward_test(self, feats, roi_bounds, **kwargs):
        return self.forward_train(feats, roi_bounds, return_loss=False)

    def cal_loc_loss(self, loc_pred, gt_bbox, anchors):
        cls_labels, loc_targets = prepare_targets(anchors, gt_bbox, self.iou_thr)

        pos_inds = torch.nonzero(cls_labels > 0).squeeze(1)
        loc_pred = loc_pred.flatten(0, 1)

        # update the loss normalizer
        self.loss_normalizer_reg = self.loss_normalizer_momentum * self.loss_normalizer_reg + (
            1 - self.loss_normalizer_momentum
        ) * max(pos_inds.numel(), 1)

        loc_loss = giou_loss(loc_pred[pos_inds], loc_targets[pos_inds]) / self.loss_normalizer_reg

        return {"loss_stage2_loc": loc_loss}
