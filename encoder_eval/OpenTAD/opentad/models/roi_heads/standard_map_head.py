import torch.nn as nn
from ..builder import HEADS, build_proposal_generator, build_roi_extractor, build_head


@HEADS.register_module()
class StandardProposalMapHead(nn.Module):
    def __init__(
        self,
        proposal_generator,
        proposal_roi_extractor,
        proposal_head=None,
    ):
        super().__init__()

        self.proposal_generator = build_proposal_generator(proposal_generator)
        self.proposal_roi_extractor = build_roi_extractor(proposal_roi_extractor)
        self.proposal_head = build_head(proposal_head)

    def forward_train(self, x, proposal_list, gt_segments, gt_labels, **kwargs):
        # proposals generator
        proposal_map, valid_mask = self.proposal_generator()

        # roi align to get the proposal feature
        proposal_feats = self.proposal_roi_extractor(x)

        # head forward
        losses = self.proposal_head.forward_train(proposal_feats, proposal_map, valid_mask, gt_segments)

        return losses

    def forward_test(self, x, **kwargs):
        # proposal feature
        proposal_feats = self.proposal_roi_extractor(x)

        # head forward
        proposal_pred = self.proposal_head.forward_test(proposal_feats)
        return proposal_pred
