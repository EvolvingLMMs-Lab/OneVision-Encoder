from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from .focal_loss import sigmoid_focal_loss
from ..builder import LOSSES, build_matcher
from ..utils.bbox_tools import proposal_cw_to_se
from ..utils.iou_tools import compute_giou_torch, compute_iou_torch
from ..utils.misc import convert_gt_to_one_hot


@LOSSES.register_module()
class SetCriterion(nn.Module):
    """This class computes the loss for Conditional DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(
        self,
        num_classes,
        matcher,
        weight_dict,
        losses: List[str] = ["class", "boxes"],
        eos_coef: float = 0.1,
        loss_class_type: str = "focal_loss",
        alpha: float = 0.25,
        gamma: float = 2.0,
        use_multi_class: bool = True,  # for multi-class classification
        with_dn: bool = False,
    ):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = build_matcher(matcher)
        self.weight_dict = weight_dict
        self.losses = losses
        self.alpha = alpha
        self.gamma = gamma
        self.eos_coef = eos_coef
        self.loss_class_type = loss_class_type
        self.use_multi_class = use_multi_class
        self.with_dn = with_dn
        assert loss_class_type in [
            "ce_loss",
            "focal_loss",
        ], "only support ce loss and focal loss for computing classification loss"

        if self.loss_class_type == "ce_loss":
            empty_weight = torch.ones(self.num_classes + 1)
            empty_weight[-1] = eos_coef
            self.register_buffer("empty_weight", empty_weight)

    def loss_labels(self, outputs, gt_segments, gt_labels, indices, num_boxes):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t[J] for t, (_, J) in zip(gt_labels, indices)])

        # Computation classification loss
        if self.loss_class_type == "ce_loss":
            target_classes = torch.full(
                src_logits.shape[:2],
                self.num_classes,
                dtype=torch.int64,
                device=src_logits.device,
            )
            target_classes[idx] = target_classes_o
            loss_class = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        elif self.loss_class_type == "focal_loss":
            # src_logits: (b, num_queries, num_classes) = (2, 300, 80)
            # target_classes_one_hot = (2, 300, 80)
            if self.use_multi_class:
                target_classes_onehot = torch.zeros(src_logits.shape, dtype=torch.int64, device=src_logits.device)
                target_classes_onehot[idx] = target_classes_o  # [B,Q,num_classes]
                target_classes_onehot = target_classes_onehot.float()
            else:
                target_classes = torch.full(
                    src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
                )
                target_classes[idx] = target_classes_o

                target_classes_onehot = torch.zeros(
                    [src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                    dtype=src_logits.dtype,
                    layout=src_logits.layout,
                    device=src_logits.device,
                )
                target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
                target_classes_onehot = target_classes_onehot[:, :, :-1]

            loss_class = sigmoid_focal_loss(
                src_logits,
                target_classes_onehot,
                alpha=self.alpha,
                gamma=self.gamma,
            )
            loss_class = loss_class.mean(1).sum() / num_boxes
            loss_class *= src_logits.shape[1]

        losses = {"loss_class": loss_class}
        return losses

    def loss_boxes(self, outputs, gt_segments, gt_labels, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
        targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
        The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t[i] for t, (_, i) in zip(gt_segments, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")

        losses = {}
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(compute_giou_torch(proposal_cw_to_se(target_boxes), proposal_cw_to_se(src_boxes)))
        losses["loss_giou"] = loss_giou.sum() / num_boxes
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, gt_segments, gt_labels, indices, num_boxes, **kwargs):
        loss_map = {
            "class": self.loss_labels,
            "boxes": self.loss_boxes,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, gt_segments, gt_labels, indices, num_boxes, **kwargs)

    def get_encoder_loss(self, enc_outputs, gt_segments, gt_labels, indices, num_boxes, **kwargs):
        pass

    def forward(self, outputs, gt_segments, gt_labels, dn_metas=None, return_indices=False):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
             return_indices: used for vis. if True, the layer0-5 indices will be returned as well.
        """

        if self.loss_class_type == "focal_loss" and self.use_multi_class:
            # deal with multi class issues happened in THUMOS
            gt_segments, gt_labels = convert_gt_to_one_hot(gt_segments, gt_labels, num_classes=self.num_classes)

        # Retrieve the matching between the outputs of the last layer and the targets
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs" and k != "enc_outputs"}
        indices = self.matcher(outputs_without_aux, gt_segments, gt_labels)
        if return_indices:
            indices0_copy = indices
            indices_list = []

        # Compute the average number of target boxes across all nodes, for normalization purposes
        num_boxes = sum(t.shape[0] for t in gt_segments)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        dist.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / dist.get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, gt_segments, gt_labels, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, gt_segments, gt_labels)
                if return_indices:
                    indices_list.append(indices)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, gt_segments, gt_labels, indices, num_boxes)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # denoising loss
        if self.with_dn:
            aux_num = len(outputs["aux_outputs"]) if "aux_outputs" in outputs else 0
            losses.update(self.get_dn_loss(dn_metas, gt_segments, gt_labels, aux_num, num_boxes))

        # Compute losses for two-stage deformable-detr / DINO
        if "enc_outputs" in outputs:
            losses.update(self.get_encoder_loss(outputs["enc_outputs"], gt_segments, gt_labels, num_boxes))

        for k in losses.keys():
            for kk in self.weight_dict.keys():
                if kk in k:
                    losses[k] *= self.weight_dict[kk]

        if return_indices:
            indices_list.append(indices0_copy)
            return losses, indices_list
        else:
            return losses


@LOSSES.register_module()
class DeformableSetCriterion(SetCriterion):
    def get_encoder_loss(self, enc_outputs, gt_segments, gt_labels, num_boxes, **kwargs):
        bin_labels = []
        for gt_label in gt_labels:
            bin_label = torch.zeros_like(gt_label)
            bin_label[:, 0] = 1  # deformable-detr only uses binary classification for encoder output
            bin_labels.append(bin_label)
        indices = self.matcher(enc_outputs, gt_segments, bin_labels)
        encoder_losses = {}
        for loss in self.losses:
            l_dict = self.get_loss(loss, enc_outputs, gt_segments, bin_labels, indices, num_boxes, **kwargs)
            l_dict = {k + "_enc": v for k, v in l_dict.items()}
            encoder_losses.update(l_dict)
        return encoder_losses


@LOSSES.register_module()
class TadTRSetCriterion(SetCriterion):
    def loss_boxes(self, outputs, gt_segments, gt_labels, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the IoU loss
        targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
        The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t[i] for t, (_, i) in zip(gt_segments, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")

        losses = {}
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes

        loss_iou = 1 - torch.diag(compute_iou_torch(proposal_cw_to_se(target_boxes), proposal_cw_to_se(src_boxes)))
        losses["loss_iou"] = loss_iou.sum() / num_boxes
        return losses

    def loss_actionness(self, outputs, gt_segments, gt_labels, indices, num_boxes):
        """Compute the actionness regression loss
        targets dicts must contain the key "segments" containing a tensor of dim [nb_target_segments, 2]
        The target segments are expected in format (center, width), normalized by the video length.
        """
        assert "pred_actionness" in outputs

        # Compute GT IoU
        gt_iou = []
        for gt_segment, pred_boxes in zip(gt_segments, outputs["pred_boxes"]):
            iou = compute_iou_torch(proposal_cw_to_se(gt_segment), proposal_cw_to_se(pred_boxes))
            gt_iou.append(iou.max(dim=1)[0])
        gt_iou = torch.cat(gt_iou, dim=0)  # [bs*num_queries]

        pred_iou = outputs["pred_actionness"].view(-1)  # [bs*num_queries]
        loss_actionness = F.l1_loss(pred_iou, gt_iou.detach())

        losses = {}
        losses["loss_actionness"] = loss_actionness
        return losses

    def forward(self, outputs, gt_segments, gt_labels, return_indices=False):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
             return_indices: used for vis. if True, the layer0-5 indices will be returned as well.
        """
        if self.loss_class_type == "focal_loss" and self.use_multi_class:
            # deal with multi class issues happened in THUMOS
            gt_segments, gt_labels = convert_gt_to_one_hot(gt_segments, gt_labels, num_classes=self.num_classes)

        # Retrieve the matching between the outputs of the last layer and the targets
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs" and k != "enc_outputs"}
        indices = self.matcher(outputs_without_aux, gt_segments, gt_labels)
        if return_indices:
            indices0_copy = indices
            indices_list = []

        # Compute the average number of target boxes across all nodes, for normalization purposes
        num_boxes = sum(t.shape[0] for t in gt_segments)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        dist.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / dist.get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, gt_segments, gt_labels, indices, num_boxes))

        # Compute the actionness loss
        losses.update(self.loss_actionness(outputs, gt_segments, gt_labels, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, gt_segments, gt_labels)
                if return_indices:
                    indices_list.append(indices)
                for loss in self.losses:
                    l_dict = self.get_loss(loss, aux_outputs, gt_segments, gt_labels, indices, num_boxes)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        for k in losses.keys():
            for kk in self.weight_dict.keys():
                if kk in k:
                    losses[k] *= self.weight_dict[kk]

        if return_indices:
            indices_list.append(indices0_copy)
            return losses, indices_list
        else:
            return losses
