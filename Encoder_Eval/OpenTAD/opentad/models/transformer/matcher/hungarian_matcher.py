import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment

from ...utils.iou_tools import compute_giou_torch, compute_iou_torch
from ...utils.bbox_tools import proposal_cw_to_se
from ...builder import MATCHERS


@MATCHERS.register_module()
class HungarianMatcher(nn.Module):
    """HungarianMatcher which computes an assignment between targets and predictions.
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    Args:
        cost_class (float): The relative weight of the classification error
            in the matching cost. Default: 1.
        cost_bbox (float): The relative weight of the L1 error of the bounding box
            coordinates in the matching cost. Default: 1.
        cost_giou (float): This is the relative weight of the giou loss of
            the bounding box in the matching cost. Default: 1.
        cost_class_type (str): How the classification error is calculated.
            Choose from ``["ce_cost", "focal_loss_cost"]``. Default: "focal_loss_cost".
        alpha (float): Weighting factor in range (0, 1) to balance positive vs
            negative examples in focal loss. Default: 0.25.
        gamma (float): Exponent of modulating factor (1 - p_t) to balance easy vs
            hard examples in focal loss. Default: 2.
    """

    def __init__(
        self,
        cost_class: float = 1,
        cost_bbox: float = 1,
        cost_giou: float = 1,
        cost_class_type: str = "focal_loss_cost",  # or "ce_cost"
        alpha: float = 0.25,
        gamma: float = 2.0,
        iou_type: str = "giou",  # or "iou"
        use_multi_class: bool = True,
    ):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.cost_class_type = cost_class_type
        self.alpha = alpha
        self.gamma = gamma
        self.iou_type = iou_type
        self.use_multi_class = use_multi_class
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"
        assert cost_class_type in {
            "ce_cost",
            "focal_loss_cost",
        }, "only support ce loss or focal loss for computing class cost"

    @torch.no_grad()
    def forward(self, outputs, gt_segments, gt_labels):
        """Forward function for `HungarianMatcher` which performs the matching.
        Args:
            outputs (Dict[str, torch.Tensor]): This is a dict that contains at least these entries:
                - ``"pred_logits"``: Tensor of shape (bs, num_queries, num_classes) with the classification logits.
                - ``"pred_boxes"``: Tensor of shape (bs, num_queries, 4) with the predicted box coordinates.
            targets (List[Dict[str, torch.Tensor]]): This is a list of targets (len(targets) = batch_size),
                where each target is a dict containing:
                - ``"labels"``: Tensor of shape (num_target_boxes, ) (where num_target_boxes is the number of ground-truth objects in the target) containing the class labels.  # noqa
                - ``"boxes"``: Tensor of shape (num_target_boxes, 4) containing the target box coordinates.
        Returns:
            list[torch.Tensor]: A list of size batch_size, containing tuples of `(index_i, index_j)` where:
                - ``index_i`` is the indices of the selected predictions (in order)
                - ``index_j`` is the indices of the corresponding selected targets (in order)
            For each batch element, it holds: `len(index_i) = len(index_j) = min(num_queries, num_target_boxes)`
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        if self.cost_class_type == "ce_cost":
            out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        elif self.cost_class_type == "focal_loss_cost":
            out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()  # [batch_size * num_queries, num_classes]

        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 2]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat(gt_labels)  # [num_gt, num_classes]
        tgt_segment = torch.cat(gt_segments)  # num_gt

        # Compute the classification cost.
        if self.cost_class_type == "ce_cost":
            # Compute the classification cost. Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - prob[target class].
            # The 1 is a constant that doesn't change the matching, it can be omitted.
            cost_class = -out_prob[:, tgt_ids]
        elif self.cost_class_type == "focal_loss_cost":
            alpha = self.alpha
            gamma = self.gamma
            neg_cost_class = (1 - alpha) * (out_prob**gamma) * (-(1 - out_prob + 1e-8).log())
            pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
            if self.use_multi_class:
                cost_class = (
                    pos_cost_class - neg_cost_class
                ) @ tgt_ids.T.float()  # [batch_size * num_queries, num_gts]
            else:
                cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_segment, p=1)

        # Compute the giou cost between boxes
        if self.iou_type == "giou":
            cost_giou = -compute_giou_torch(proposal_cw_to_se(tgt_segment), proposal_cw_to_se(out_bbox))
        elif self.iou_type == "iou":
            cost_giou = -compute_iou_torch(proposal_cw_to_se(tgt_segment), proposal_cw_to_se(out_bbox))
        else:
            raise NotImplementedError(f"iou_type {self.iou_type} not implemented")

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [v.shape[0] for v in gt_segments]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

    def __repr__(self):
        rep_str = self.__class__.__name__ + "("
        rep_str += f"cost_class={str(self.cost_class)}, "
        rep_str += f"cost_bbox={str(self.cost_bbox)}, "
        rep_str += f"cost_giou={str(self.cost_giou)}, "
        rep_str += f"alpha={str(self.alpha)}, "
        rep_str += f"gamma={str(self.gamma)})"
        return rep_str
