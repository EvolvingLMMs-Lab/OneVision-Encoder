import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..builder import HEADS, build_loss, build_head
from ..losses.balanced_ce_loss import BalancedCELoss


class FPN_Anchors(nn.Module):
    def __init__(self, pyramid_levels=[2, 4, 8, 16, 32], tscale=256, anchor_scale=[3, 7.5]):
        super(FPN_Anchors, self).__init__()
        self.pyramid_levels = pyramid_levels
        self.tscale = tscale
        self.scales = anchor_scale
        self.base_stride = pyramid_levels[0]

        self.base_anchors = []
        for stride in pyramid_levels:
            self.base_anchors.append(self.get_base_anchors(stride, self.scales))

        self.anchors = self.gen_anchors()

    def gen_anchors(self):
        feat_sizes = [math.ceil(self.tscale / stride) for stride in self.pyramid_levels]
        anchors = []
        for size, stride, base_anchors in zip(feat_sizes, self.pyramid_levels, self.base_anchors):
            shifts = torch.arange(0, size * stride, step=stride, dtype=torch.float32)[:, None].repeat(1, 2)
            anchors.append((shifts.view(-1, 1, 2) + base_anchors.view(1, -1, 2)).reshape(-1, 2))

        return anchors

    def get_base_anchors(self, stride, scales):
        anchors = torch.tensor([1, stride], dtype=torch.float) - 0.5
        anchors = self._scale_enum(anchors, scales)
        return anchors

    def _scale_enum(self, anchor, scales):
        """Enumerate a set of anchors for each scale wrt an anchor."""
        length, center = self._whctrs(anchor)
        ws = length * torch.tensor(scales)
        anchors = self._mkanchors(ws, center)
        return anchors

    def _mkanchors(self, ws, ctr):
        anchors = torch.stack(
            (
                ctr.unsqueeze(0) - 0.5 * (ws.to(dtype=torch.float32) - 1),
                ctr.unsqueeze(0) + 0.5 * (ws.to(dtype=torch.float32) - 1),
            )
        ).transpose(0, 1)

        return anchors

    def _whctrs(self, anchor):
        length = anchor[1] - anchor[0] + 1
        center = anchor[0] + 0.5 * (length - 1)
        return length, center


@HEADS.register_module()
class VSGNRPNHead(nn.Module):
    def __init__(
        self,
        num_layers=4,
        in_channels=256,
        num_classes=1,
        iou_thr=0.6,
        anchor_generator=None,
        tem_head=None,
        loss_cls=None,
        loss_loc=None,
    ):
        super(VSGNRPNHead, self).__init__()

        self.num_layers = num_layers
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_anchors = len(anchor_generator["anchor_scale"])

        self.get_cls_towers()
        self.get_reg_towers()

        self.tem_head = build_head(tem_head)

        self.iou_thr = iou_thr
        self.anchor_generator = anchor_generator
        self.anchors = FPN_Anchors(**anchor_generator).anchors

        # maintain an EMA of #foreground to stabilize the loss normalizer
        # useful for small mini-batch training
        self.loss_normalizer_cls = 100
        self.loss_normalizer_reg = 100
        self.loss_normalizer_momentum = 0.9
        self.loss_cls_func = build_loss(loss_cls)
        # default regression loss is GIoU

    def get_cls_towers(self):
        cls_tower = []
        for _ in range(self.num_layers):
            cls_tower.append(
                nn.Sequential(
                    nn.Conv1d(
                        self.in_channels,
                        self.in_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=False,
                    ),
                    nn.GroupNorm(32, self.in_channels),
                    nn.ReLU(),
                )
            )
        self.cls_tower = nn.Sequential(*cls_tower)

        self.cls_head = nn.Conv1d(
            self.in_channels,
            self.num_anchors * self.num_classes,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def get_reg_towers(self):
        reg_tower = []
        for _ in range(self.num_layers):
            reg_tower.append(
                nn.Sequential(
                    nn.Conv1d(
                        self.in_channels,
                        self.in_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=False,
                    ),
                    nn.GroupNorm(32, self.in_channels),
                    nn.ReLU(),
                )
            )
        self.reg_tower = nn.Sequential(*reg_tower)
        self.reg_head = nn.Conv1d(self.in_channels, self.num_anchors * 2, kernel_size=3, stride=1, padding=1)

    def forward_train(self, feat_list, mask_list, gt_segments=None, gt_labels=None, return_loss=True):
        B = feat_list[0].shape[0]

        # classification logits, start/end offsets
        logits_pred = []
        offsets_pred = []
        for feat in feat_list:
            logits_pred.append(self.cls_head(self.cls_tower(feat)))  # (B, scales*cls, positions) * levels
            offsets_pred.append(self.reg_head(self.reg_tower(feat)))  # (B, scales*2, positions) * levels

        # (B, levels*positions*scales, cls) / (B, levels*positions*scales, 2)
        logits_pred = torch.cat(logits_pred, dim=2).permute(0, 2, 1).reshape(B, -1, self.num_classes)
        offsets_pred = torch.cat(offsets_pred, dim=2).permute(0, 2, 1).reshape(B, -1, 2)

        # anchors: (B, levels*positions*scales, 2)
        anchors = torch.cat(self.anchors, dim=0).unsqueeze(0).repeat(B, 1, 1).to(device=feat_list[0].device)

        if self.num_classes == 1:
            score_pred = logits_pred.sigmoid()
        else:
            score_pred = F.softmax(logits_pred, dim=2)

        # convert offsets to proposals
        loc_pred = decode(offsets_pred.view(-1, 2), anchors.view(-1, 2)).view(B, -1, 2)

        # temporal evaluation module
        tem_feat = F.interpolate(feat_list[0], size=self.anchor_generator.tscale, mode="linear", align_corners=True)

        losses = {}
        if return_loss:
            # start / end / action ness
            losses.update(self.tem_head.forward_train(tem_feat, mask_list, gt_segments)[0])

            # Special GT for VSGN:
            # 1) Labels shift by including background = 0;
            # 2) gt_segments has the third dimension as gt_labels
            gts = [torch.cat((gt_b, gt_l[:, None] + 1), dim=-1) for gt_b, gt_l in zip(gt_segments, gt_labels)]
            losses.update(self.cal_loc_loss(loc_pred, gts, anchors))
            losses.update(self.cal_cls_loss(logits_pred, gts, loc_pred))
            return losses, loc_pred
        else:
            tem_score = self.tem_head.forward_test(tem_feat, mask_list)
            return tem_score, loc_pred, score_pred

    def forward_test(self, feat_list, mask_list, **kwargs):
        return self.forward_train(feat_list, mask_list, return_loss=False, **kwargs)

    def cal_loc_loss(self, loc_pred, gt_bbox, anchors):
        # reg_targets: corresponding gt_boxes of each anchor
        cls_labels, loc_targets = prepare_targets(anchors, gt_bbox, self.iou_thr)
        # bs, levels*positions*scales, num_cls/left-right

        loc_pred = loc_pred.view(-1, 2)
        pos_inds = torch.nonzero(cls_labels > 0).squeeze(1)

        # update the loss normalizer
        self.loss_normalizer_reg = self.loss_normalizer_momentum * self.loss_normalizer_reg + (
            1 - self.loss_normalizer_momentum
        ) * max(pos_inds.numel(), 1)

        reg_loss = giou_loss(loc_pred[pos_inds], loc_targets[pos_inds])
        reg_loss /= self.loss_normalizer_reg  # pos_inds.numel()
        return {"loss_stage1_reg": reg_loss}

    def cal_cls_loss(self, cls_pred, gt_bbox, anchors):
        num_cls = self.num_classes
        cls_labels, _ = prepare_targets(anchors, gt_bbox, self.iou_thr)
        # bs, levels*positions*scales, num_cls/left-right

        if num_cls == 1:
            cls_labels[cls_labels > 0] = 1

        cls_pred = cls_pred.view(-1, num_cls)
        # bs, levels*positions, scales*cls --> bs*levels*positions*scales, num_cls
        pos_inds = torch.nonzero(cls_labels > 0).squeeze(1)

        # update the loss normalizer
        self.loss_normalizer_cls = self.loss_normalizer_momentum * self.loss_normalizer_cls + (
            1 - self.loss_normalizer_momentum
        ) * max(pos_inds.numel(), 1)

        if isinstance(self.loss_cls_func, BalancedCELoss):
            cls_loss = self.loss_cls_func(cls_pred, cls_labels)
        else:
            cls_pred = cls_pred.squeeze(-1)
            cls_loss = self.loss_cls_func(cls_pred, cls_labels, reduction="sum") / self.loss_normalizer_cls
        return {"loss_stage1_cls": cls_loss}


@torch.no_grad()
def prepare_targets(anchors, gt_bbox, iou_thr):
    """
    Returns:
    cls_labels: B * levels*positions*scales, num_cls
    loc_targets: B * levels*positions*scales, left-right
    """
    cls_targets = []
    reg_targets = []

    for i in range(len(gt_bbox)):
        gt_cur_im = gt_bbox[i][:, :-1]
        gt_label = gt_bbox[i][:, -1]
        anchor_cur_im = anchors[i]

        if anchor_cur_im.numel() == 0:
            raise "Video with no proposals!"

        iou_matrix = iou_anchors_gts(anchor_cur_im, gt_cur_im)

        # Find the corresponding gt for each pred
        matched_idxs = Matcher(True)(iou_matrix.transpose(0, 1), iou_thr)

        # Use the label of the corresponding gt as the classification target for the pred
        cls_labels_cur_im = gt_label[matched_idxs]
        cls_labels_cur_im[matched_idxs < 0] = 0

        # Record the boundary offset as the regression target
        matched_gts = gt_cur_im[matched_idxs.clamp(min=0)]
        reg_targets_cur_im = matched_gts

        cls_targets.append(cls_labels_cur_im.to(dtype=torch.int32))
        reg_targets.append(reg_targets_cur_im)

    return torch.cat(cls_targets, dim=0), torch.cat(reg_targets, dim=0)


def iou_anchors_gts(anchor, gt):
    anchors_min = anchor[:, 0]
    anchors_max = anchor[:, 1]
    box_min = gt[:, 0]
    box_max = gt[:, 1]
    len_anchors = anchors_max - anchors_min + 1
    int_xmin = torch.max(anchors_min[:, None], box_min)
    int_xmax = torch.min(anchors_max[:, None], box_max)
    inter_len = torch.clamp(int_xmax - int_xmin, min=0)
    union_len = torch.clamp(len_anchors[:, None] + box_max - box_min - inter_len, min=0)

    jaccard = inter_len / union_len

    return jaccard


class Matcher(object):
    """
    This class assigns to each predicted "element" (e.g., a box) a ground-truth
    element. Each predicted element will have exactly zero or one matches; each
    ground-truth element may be assigned to zero or more predicted elements.

    Matching is based on the MxN match_quality_matrix, that characterizes how well
    each (ground-truth, predicted)-pair match. For example, if the elements are
    boxes, the matrix may contain box IoU overlap values.

    The matcher returns a tensor of size N containing the index of the ground-truth
    element m that matches to prediction n. If there is no match, a negative value
    is returned.
    """

    BELOW_LOW_THRESHOLD = -1
    BETWEEN_THRESHOLDS = -2

    def __init__(self, allow_low_quality_matches=False):
        """
        Args:
            allow_low_quality_matches (bool): if True, produce additional matches
                for predictions that have only low-quality match candidates. See
                set_low_quality_matches_ for more details.
        """
        self.allow_low_quality_matches = allow_low_quality_matches

    def __call__(self, match_quality_matrix, iou_thr=0.5):
        """
        Args:
            match_quality_matrix (Tensor[float]): an MxN tensor, containing the
            pairwise quality between M ground-truth elements and N predicted elements.

        Returns:
            matches (Tensor[int64]): an N tensor where N[i] is a matched gt in
            [0, M - 1] or a negative value indicating that prediction i could not
            be matched.
        """
        if match_quality_matrix.numel() == 0:
            # empty targets or proposals not supported during training
            if match_quality_matrix.shape[0] == 0:
                raise ValueError("No ground-truth boxes available for one of the images " "during training")
            else:
                raise ValueError("No proposal boxes available for one of the images " "during training")

        # match_quality_matrix is M (gt) x N (predicted)
        # Max over gt elements (dim 0) to find best gt candidate for each prediction
        matched_vals, matches = match_quality_matrix.max(dim=0)
        if self.allow_low_quality_matches:
            all_matches = matches.clone()

        # Assign candidate matches with low quality to negative (unassigned) values
        below_threshold = matched_vals < iou_thr
        matches[below_threshold] = Matcher.BELOW_LOW_THRESHOLD

        if self.allow_low_quality_matches:
            self.set_low_quality_matches_(matches, all_matches, match_quality_matrix)

        return matches

    def set_low_quality_matches_(self, matches, all_matches, match_quality_matrix):
        """
        Produce additional matches for predictions that have only low-quality matches.
        Specifically, for each ground-truth find the set of predictions that have
        maximum overlap with it (including ties); for each prediction in that set, if
        it is unmatched, then match it to the ground-truth with which it has the highest
        quality value.
        """
        # For each gt, find the prediction with which it has highest quality
        highest_quality_foreach_gt, _ = match_quality_matrix.max(dim=1)
        # Find highest quality match available, even if it is low, including ties
        gt_pred_pairs_of_highest_quality = torch.nonzero(match_quality_matrix == highest_quality_foreach_gt[:, None])
        # Example gt_pred_pairs_of_highest_quality:
        #   tensor([[    0, 39796],
        #           [    1, 32055],
        #           [    1, 32070],
        #           [    2, 39190],
        #           [    2, 40255],
        #           [    3, 40390],
        #           [    3, 41455],
        #           [    4, 45470],
        #           [    5, 45325],
        #           [    5, 46390]])
        # Each row is a (gt index, prediction index)
        # Note how gt items 1, 2, 3, and 5 each have two ties

        pred_inds_to_update = gt_pred_pairs_of_highest_quality[:, 1]
        matches[pred_inds_to_update] = all_matches[pred_inds_to_update]


def giou_loss(pred_boxes, gt_boxes, weight=None):
    pred_x1 = torch.min(pred_boxes[:, 0], pred_boxes[:, 1])
    pred_x2 = torch.max(pred_boxes[:, 0], pred_boxes[:, 1])
    pred_area = pred_x2 - pred_x1

    target_x1 = gt_boxes[:, 0]
    target_x2 = gt_boxes[:, 1]
    target_area = target_x2 - target_x1

    x1_intersect = torch.max(pred_x1, target_x1)
    x2_intersect = torch.min(pred_x2, target_x2)
    area_intersect = torch.zeros(pred_x1.size()).to(gt_boxes)
    mask = x2_intersect > x1_intersect
    area_intersect[mask] = x2_intersect[mask] - x1_intersect[mask]

    x1_enclosing = torch.min(pred_x1, target_x1)
    x2_enclosing = torch.max(pred_x2, target_x2)
    area_enclosing = (x2_enclosing - x1_enclosing) + 1e-7

    area_union = pred_area + target_area - area_intersect + 1e-7
    ious = area_intersect / area_union
    gious = ious - (area_enclosing - area_union) / area_enclosing

    losses = 1 - gious

    if weight is not None and weight.sum() > 0:
        return (losses * weight).sum()
    else:
        assert losses.numel() != 0
        return losses.sum()


def decode(preds, anchors):
    """
    Get absolute boundary values from anchors and predicted offsets
    """
    anchors = anchors.to(preds.dtype)

    TO_REMOVE = 1  # TODO remove
    ex_length = anchors[:, 1] - anchors[:, 0] + TO_REMOVE
    ex_center = (anchors[:, 1] + anchors[:, 0]) / 2

    wx, ww = (10, 5.0)
    dx = preds[:, 0] / wx
    dw = preds[:, 1] / ww

    # Prevent sending too large values into torch.exp()
    dw = torch.clamp(dw, max=math.log(1000.0 / 16))

    pred_ctr_x = dx * ex_length + ex_center
    pred_w = torch.exp(dw) * ex_length

    pred_boxes = torch.zeros_like(preds)
    pred_boxes[:, 0] = pred_ctr_x - 0.5 * (pred_w - 1)
    pred_boxes[:, 1] = pred_ctr_x + 0.5 * (pred_w - 1)
    return pred_boxes
