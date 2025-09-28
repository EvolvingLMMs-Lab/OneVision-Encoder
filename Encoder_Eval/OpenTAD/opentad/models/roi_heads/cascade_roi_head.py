import copy
import torch
import torch.nn as nn
from ..builder import HEADS, build_proposal_generator, build_roi_extractor, build_head


@HEADS.register_module()
class CascadeRoIHead(nn.Module):
    def __init__(
        self,
        stages,
        proposal_roi_extractor,
        proposal_head,
        proposal_generator=None,
    ):
        super().__init__()

        self.stage_loss_weight = stages.loss_weight
        self.proposal_roi_extractor = build_roi_extractor(proposal_roi_extractor)

        self.proposal_heads = nn.ModuleList([])
        for i in range(stages.number):
            stage_cfg = copy.deepcopy(proposal_head)
            stage_cfg.loss.assigner.pos_iou_thr = stages.pos_iou_thresh[i]
            stage_cfg.loss.assigner.neg_iou_thr = stages.pos_iou_thresh[i]
            stage_cfg.loss.assigner.min_pos_iou = stages.pos_iou_thresh[i]
            self.proposal_heads.append(build_head(stage_cfg))

        if proposal_generator != None:
            self.proposal_generator = build_proposal_generator(proposal_generator)

    @property
    def with_proposal_generator(self):
        """bool: whether the roi head's proposals are initialized by proposal_generator"""
        return hasattr(self, "proposal_generator") and self.proposal_generator is not None

    def forward_train(self, x, proposal_list, gt_segments, gt_labels, **kwargs):
        # (Optional) proposals generator
        if self.with_proposal_generator:
            proposal_list = self.proposal_generator(x)

        losses = {}
        for i in range(len(self.proposal_heads)):
            # roi align to get the proposal feature
            proposal_feats = self.proposal_roi_extractor(x, proposal_list)  # [B,K,C,res]

            # head forward
            loss, proposal_list = self.proposal_heads[i].forward_train(
                proposal_feats,
                proposal_list,
                gt_segments,
                gt_labels,
            )

            for name, value in loss.items():
                if "loss" in name:
                    losses[f"s{i}.{name}"] = value * self.stage_loss_weight[i]
                else:
                    losses[f"s{i}.{name}"] = value
        return losses

    def forward_test(self, x, proposal_list=None, **kwargs):
        if self.with_proposal_generator:
            proposal_list = self.proposal_generator(x)

        proposals = []
        scores = []
        for i in range(len(self.proposal_heads)):
            # roi align to get the proposal feature
            proposal_feats = self.proposal_roi_extractor(x, proposal_list)  # [B,K,C,res]

            # head forward
            proposal_list, proposal_score = self.proposal_heads[i].forward_test(proposal_feats, proposal_list)

            proposals.append(proposal_list)
            scores.append(proposal_score)

        # get refined proposal: average of three stages,  [B,K,2]
        # proposals = torch.stack(proposals, dim=-1).mean(dim=-1)
        # get proposal score: average of three stages, [B,K,num_classes]
        # cls_score = torch.stack(scores, dim=-1).mean(dim=-1)

        refined_proposal_list = []
        proposal_score_list = []

        for i in range(len(proposals[0])):
            proposals_per_video = torch.stack([prop[i] for prop in proposals], dim=-1).mean(dim=-1)
            scores_per_video = torch.stack([score[i] for score in scores], dim=-1).mean(dim=-1)
            refined_proposal_list.append(proposals_per_video)
            proposal_score_list.append(scores_per_video)
        return refined_proposal_list, proposal_score_list
