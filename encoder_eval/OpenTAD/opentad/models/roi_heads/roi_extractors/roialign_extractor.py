import torch
import torch.nn as nn
import torchvision
from ...builder import ROI_EXTRACTORS
from .align1d.align import Align1DLayer


@ROI_EXTRACTORS.register_module()
class ROIAlignExtractor(nn.Module):
    def __init__(
        self,
        roi_size=8,
        extend_ratio=0.5,
        base_stride=1,
        fpn_extract=False,
        base_scale=8,
        method="torchvision",
        deterministic=False,
    ):
        super(ROIAlignExtractor, self).__init__()

        self.roi_size = roi_size
        self.extend_ratio = extend_ratio
        self.base_stride = base_stride
        self.fpn_extract = fpn_extract
        self.base_scale = base_scale
        self.method = method
        self.deterministic = deterministic

        if self.method == "align1d":
            self.align1d = Align1DLayer(self.roi_size, ratio=0)

    def forward(self, feat_list, proposals):
        # proposals: list [K,2]: 0~tscale

        if self.fpn_extract:  # todo, maybe wrong
            assert isinstance(feat_list, (list, tuple))
            num_levels = len(feat_list)

            # assign proposals to different levels
            #                 proposals_scale < base_scale,    level 0
            # base_scale   <= proposals_scale < base_scale*2 , level 1
            # base_scale*2 <= proposals_scale < base_scale*4 , level 2
            concat_proposals = torch.cat(proposals, dim=0)  # [sum(K),2]
            proposals_scale = (concat_proposals[:, 1] - concat_proposals[:, 0]).clamp(min=1e-6)  # [sum(K)]
            proposals_level = torch.ceil(torch.log2(proposals_scale / (self.base_scale * self.base_stride)))
            proposals_level = proposals_level.clamp(min=0, max=num_levels - 1).long()  # [sum(K)] long

            # add proposals batch idxs
            proposals_batch_idxs = [torch.ones(proposal.shape[0]) * i for i, proposal in enumerate(proposals)]
            proposals_batch_idxs = torch.cat(proposals_batch_idxs, dim=0).to(concat_proposals.device).long()
            proposals_batch_idxs = proposals_batch_idxs.unsqueeze(-1)  # [sum(K), 1]

            # get proposal feature
            proposals_feat = feat_list[0].new_zeros(
                concat_proposals.shape[0], feat_list[0].shape[1], self.roi_size
            )  # [sum(K),C,res]
            for i in range(num_levels):
                mask = proposals_level == i
                idxs = mask.nonzero(as_tuple=False).squeeze(1)
                if idxs.numel() > 0:
                    ext_proposals_ = self._get_extend_proposals(concat_proposals[idxs]) / 2**i  # [sum(K)',2]
                    ext_proposals_ = torch.cat((proposals_batch_idxs[idxs], ext_proposals_), dim=1)  # [sum(K)',3]
                    proposals_feat_ = self.align_tool(feat_list[i], ext_proposals_)  # directly use align_tool
                    proposals_feat[idxs] = proposals_feat_.squeeze(-1)  # [sum(K),C,res]
            # return the concat proposals_feat

        else:  # use the first level
            if isinstance(feat_list, (list, tuple)):
                feat = feat_list[0]
            elif isinstance(feat_list, torch.Tensor):
                feat = feat_list
            else:
                raise TypeError("feat_list must be list or tensor")

            ext_proposals = self._get_extend_proposals(proposals)  # [B,K,2]
            proposals_feat = self._align(feat, ext_proposals)
        return proposals_feat  # [B,K,C,res]

    def _get_extend_proposals(self, proposals):
        def extend(proposals):
            proposals_len = proposals[..., 1] - proposals[..., 0]
            proposals = torch.stack(
                (
                    proposals[..., 0] - self.extend_ratio * proposals_len,
                    proposals[..., 1] + self.extend_ratio * proposals_len,
                ),
                dim=-1,
            )
            return proposals

        if isinstance(proposals, list):  # list [K,2]
            return [extend(proposal) / self.base_stride for proposal in proposals]
        elif isinstance(proposals, torch.Tensor):  # [B,K,2]
            return extend(proposals) / self.base_stride

    def _align(self, feature, proposals):
        # add batch idx
        if isinstance(proposals, list):
            batch_proposals = []
            for batch_id, proposal in enumerate(proposals):
                proposal_id = proposal.new_full((proposal.shape[0], 1), batch_id)
                batch_proposals.append(torch.cat((proposal_id, proposal), dim=-1))
            batch_proposals = torch.cat(batch_proposals, dim=0)

            proposals_feat = self.align_tool(feature, batch_proposals)  # [sum(K),C,res,1]
            proposals_feat = proposals_feat.squeeze(-1)  # [sum(K),C,res]

        elif isinstance(proposals, torch.Tensor):
            # proposals.dim() == 3:  # [B,K,2]
            bs, K = proposals.shape[:2]
            bs_idxs = torch.arange(bs).view(bs, 1, 1).repeat(1, K, 1).to(proposals.device)
            proposals = torch.cat((bs_idxs, proposals), dim=2)  # [B,K,3]
            proposals = proposals.view(-1, 3)  # [B*K,3]

            proposals_feat = self.align_tool(feature, proposals)  # [B*K,C,res,1]
            proposals_feat = proposals_feat.unflatten(dim=0, sizes=(bs, K)).squeeze(-1)  # [B,K,C,res]
        return proposals_feat

    def align_tool(self, feature, proposals):
        if self.deterministic:
            original_dtype = feature.dtype
            feature = feature.half().double()
            proposals = proposals.half().double()

        if self.method == "torchvision":
            # use torchvision align
            pseudo_input = feature.unsqueeze(3)  # [B,C,T,1]
            pseudo_bbox = torch.stack(
                (
                    proposals[:, 0],
                    torch.zeros_like(proposals[:, 0]),
                    proposals[:, 1],
                    torch.ones_like(proposals[:, 0]),
                    proposals[:, 2],
                ),
                dim=1,
            )  # B*K, 5
            proposals_feat = torchvision.ops.roi_align(
                pseudo_input,
                pseudo_bbox,
                output_size=(self.roi_size, 1),
                aligned=True,
            )
        elif self.method == "align1d":  # use align1d align
            proposals_feat = self.align1d(feature, proposals)

        if self.deterministic:
            proposals_feat = proposals_feat.to(original_dtype)
        return proposals_feat
