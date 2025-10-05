import copy
import torch
import torch.nn as nn
from ..builder import HEADS, build_proposal_generator, build_roi_extractor, build_head
from ..utils.iou_tools import compute_iou_torch, compute_ioa_torch


@HEADS.register_module()
class ETADRoIHead(nn.Module):
    def __init__(
        self,
        stages,
        proposal_generator,
        proposal_roi_extractor,
        proposal_head,
    ):
        super().__init__()

        self.tscale = proposal_generator.tscale
        self.stage_loss_weight = stages.loss_weight
        self.proposal_generator = build_proposal_generator(proposal_generator)
        self.proposal_roi_extractor = build_roi_extractor(proposal_roi_extractor)

        self.proposal_heads = nn.ModuleList([])
        for i in range(stages.number):
            stage_cfg = copy.deepcopy(proposal_head)
            stage_cfg.loss.pos_iou_thresh = stages.pos_iou_thresh[i]
            self.proposal_heads.append(build_head(stage_cfg))

    def forward_train(self, x, proposal_list, gt_segments, gt_labels, **kwargs):
        proposal_list = self.proposal_generator(bs=x.shape[0], device=x.device, training=True)  # [B,K,2]
        gt_starts, gt_ends, batch_gt_segment = self.prepare_shared_gt(gt_segments, proposal_list)

        losses = {}
        for i in range(len(self.proposal_heads)):
            # roi align to get the proposal feature
            proposal_feats = self.proposal_roi_extractor(x, proposal_list)  # [B,K,C,res]

            # head forward
            loss, proposal_list = self.proposal_heads[i].forward_train(
                proposal_feats,
                proposal_list,
                gt_starts,
                gt_ends,
                batch_gt_segment,
            )

            for name, value in loss.items():
                if "loss" in name:
                    losses[f"{name}.s{i}"] = value * self.stage_loss_weight[i]
                else:
                    losses[f"{name}.s{i}"] = value
        return losses

    def prepare_shared_gt(self, gt_segments, proposals):
        # get startness/endness
        gt_starts = []
        gt_ends = []

        temporal_anchor = torch.stack((torch.arange(0, self.tscale), torch.arange(1, self.tscale + 1)), dim=1)
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

        gt_starts = torch.stack(gt_starts)  # [B,T]
        gt_ends = torch.stack(gt_ends)  # [B,T]

        # get corresponding gt_boxes
        batch_gt_segment = []
        for gt_segment, proposal in zip(gt_segments, proposals):
            ious = compute_iou_torch(gt_segment, proposal)  # [K,N]
            gt_iou_index = torch.max(ious, dim=1)[1]  # [K]
            batch_gt_segment.append(gt_segment[gt_iou_index])  # [K,2]
        batch_gt_segment = torch.stack(batch_gt_segment)  # [B,K,2]
        return gt_starts, gt_ends, batch_gt_segment

    def forward_test(self, x, proposal_list=None, **kwargs):
        proposal_list = self.proposal_generator(bs=x.shape[0], device=x.device)  # [B,

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
        refined_proposal = torch.stack(proposals, dim=-1).mean(dim=-1)

        # get proposal score: average of three stages, [B,K,num_classes]
        proposal_score = torch.stack(scores, dim=-1).mean(dim=-1)
        return refined_proposal, proposal_score
