import torch
from torch.nn import functional as F

from .anchor_free_head import AnchorFreeHead
from ..builder import HEADS, build_loss


@HEADS.register_module()
class TemporalMaxerHead(AnchorFreeHead):
    def __init__(
        self,
        num_classes,
        in_channels,
        feat_channels,
        num_convs=3,
        prior_generator=None,
        loss=None,
        loss_normalizer=100,
        loss_normalizer_momentum=0.9,
        label_smoothing=0,
        cls_prior_prob=0.01,
        loss_weight=1.0,
        assigner=None,
    ):
        super().__init__(
            num_classes,
            in_channels,
            feat_channels,
            num_convs=num_convs,
            cls_prior_prob=cls_prior_prob,
            prior_generator=prior_generator,
            loss=loss,
            loss_normalizer=loss_normalizer,
            loss_normalizer_momentum=loss_normalizer_momentum,
            loss_weight=loss_weight,
            label_smoothing=label_smoothing,
        )
        self.assigner = build_loss(assigner)

    def losses(self, cls_pred, reg_pred, mask_list, points, gt_segments, gt_labels):
        gt_cls, gt_reg, weights = self.prepare_targets(points, mask_list, cls_pred, reg_pred, gt_segments, gt_labels)

        # positive mask
        gt_cls = torch.stack(gt_cls)
        valid_mask = torch.cat(mask_list, dim=1)
        pos_mask = torch.logical_and((gt_cls.sum(-1) > 0), valid_mask)
        pos_weights = torch.stack(weights)[pos_mask]
        num_pos = pos_mask.sum().item()

        # maintain an EMA of foreground to stabilize the loss normalizer
        # useful for small mini-batch training
        if self.training:
            self.loss_normalizer = self.loss_normalizer_momentum * self.loss_normalizer + (
                1 - self.loss_normalizer_momentum
            ) * max(num_pos, 1)
            loss_normalizer = self.loss_normalizer
        else:
            loss_normalizer = max(num_pos, 1)

        # 1. classification loss
        cls_pred = [x.permute(0, 2, 1) for x in cls_pred]
        cls_pred = torch.cat(cls_pred, dim=1)[valid_mask]
        gt_target = gt_cls[valid_mask]

        # weight for cls
        valid_cls_weights = torch.ones(gt_cls.shape[:-1], dtype=torch.float32).to(gt_cls.device)
        valid_cls_weights[pos_mask] = pos_weights
        valid_cls_weights = valid_cls_weights[valid_mask]

        # optional label smoothing
        gt_target *= 1 - self.label_smoothing
        gt_target += self.label_smoothing / (self.num_classes + 1)

        cls_loss = self.cls_loss(cls_pred, gt_target, reduction="none")
        cls_loss = (cls_loss * valid_cls_weights[:, None]).sum()
        cls_loss /= self.loss_normalizer

        # 2. regression using IoU/GIoU/DIOU loss (defined on positive samples)
        split_size = [reg.shape[-1] for reg in reg_pred]
        gt_reg = torch.stack(gt_reg).permute(0, 2, 1).split(split_size, dim=-1)  # [B,2,T]
        pred_segments = self.get_refined_proposals(points, reg_pred)[pos_mask]
        gt_segments = self.get_refined_proposals(points, gt_reg)[pos_mask]
        if num_pos == 0:
            reg_loss = pred_segments.sum() * 0
        else:
            # giou loss defined on positive samples
            reg_loss = self.reg_loss(pred_segments, gt_segments, reduction="none")
            reg_loss = (reg_loss * pos_weights).sum()
            reg_loss /= loss_normalizer

        if self.loss_weight > 0:
            loss_weight = self.loss_weight
        else:
            loss_weight = cls_loss.detach() / max(reg_loss.item(), 0.01)

        return {"cls_loss": cls_loss, "reg_loss": reg_loss * loss_weight}

    @torch.no_grad()
    def prepare_targets(self, points, mask_list, cls_preds, reg_preds, gt_segments, gt_labels):
        cls_preds = torch.cat([x.permute(0, 2, 1) for x in cls_preds], dim=1)  # [B, T, 20]
        reg_preds = torch.cat([x.permute(0, 2, 1) for x in reg_preds], dim=1)  # [B, T, 2]
        masks = torch.cat(mask_list, dim=1)
        points = torch.cat(points, dim=0)
        num_pts = points.shape[0]

        gt_cls, gt_reg, weights = [], [], []
        for mask, cls_pred, reg_pred, gt_segment, gt_label in zip(masks, cls_preds, reg_preds, gt_segments, gt_labels):
            # label assignment
            assign_matrix, min_inds, weight = self.assigner.assign(
                cls_pred, points.clone(), reg_pred, gt_segment, gt_label, mask
            )

            # get target
            num_gts = gt_segment.shape[0]

            # corner case where current sample does not have actions
            if num_gts == 0:
                gt_cls.append(gt_segment.new_full((num_pts, self.num_classes), 0))
                gt_reg.append(gt_segment.new_zeros((num_pts, 2)))
                weights.append(weight)
                continue

            # compute the distance of every point to each segment boundary
            # auto broadcasting for all reg target-> F T x N x2
            gt_segs = gt_segment[None].expand(num_pts, num_gts, 2)
            left = points[:, 0, None] - gt_segs[:, :, 0]
            right = gt_segs[:, :, 1] - points[:, 0, None]
            reg_targets = torch.stack((left, right), dim=-1)

            # cls_targets: F T x C; reg_targets F T x 2
            gt_label_one_hot = F.one_hot(gt_label.long(), self.num_classes).to(reg_targets.dtype)
            cls_targets = assign_matrix @ gt_label_one_hot
            # to prevent multiple GT actions with the same label and boundaries
            cls_targets.clamp_(min=0.0, max=1.0)

            reg_targets = reg_targets[range(num_pts), min_inds]
            # normalization based on stride
            reg_targets /= points[:, 3, None]

            gt_cls.append(cls_targets)
            gt_reg.append(reg_targets)
            weights.append(weight)
        return gt_cls, gt_reg, weights
