import torch.nn as nn
from ..builder import HEADS, build_proposal_generator, build_roi_extractor, build_head


@HEADS.register_module()
class StandardRoIHead(nn.Module):
    def __init__(
        self,
        proposal_roi_extractor,
        proposal_head,
        proposal_generator=None,
    ):
        super().__init__()

        self.proposal_roi_extractor = build_roi_extractor(proposal_roi_extractor)

        self.proposal_head = build_head(proposal_head)

        if proposal_generator != None:
            self.proposal_generator = build_proposal_generator(proposal_generator)

    @property
    def with_proposal_generator(self):
        """bool: whether the roi head's proposals are initialized by proposal_generator"""
        return hasattr(self, "proposal_generator") and self.proposal_generator is not None

    def forward_train(self, x, proposal_list, gt_segments, gt_labels, **kwargs):
        # (Optional) proposals generator
        if self.with_proposal_generator:
            proposal_list = self.proposal_generator()

        # roi align to get the proposal feature
        proposal_feats = self.proposal_roi_extractor(x, proposal_list)  # [B,K,C,res]

        # head forward
        losses, _ = self.proposal_head.forward_train(
            proposal_feats,
            proposal_list,
            gt_segments,
            gt_labels,
        )
        return losses

    def forward_test(self, x, proposal_list, **kwargs):
        # (Optional) proposals generator
        if self.with_proposal_generator:
            proposal_list = self.proposal_generator()

        # proposal feature
        proposal_feats = self.proposal_roi_extractor(x, proposal_list)  # [B,K,C,res]

        # head forward
        proposals, scores = self.proposal_head.forward_test(proposal_feats, proposal_list)
        return proposals, scores
