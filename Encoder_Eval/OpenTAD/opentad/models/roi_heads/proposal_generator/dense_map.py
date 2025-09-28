import torch
import torch.nn as nn
from ...builder import PROPOSAL_GENERATORS


@PROPOSAL_GENERATORS.register_module()
class DenseProposalMap(nn.Module):
    def __init__(self, tscale, dscale):
        super().__init__()

        self.tscale = tscale
        self.dscale = dscale
        self.build_proposal_map()

    def forward(self, **kwargs):
        return self.proposal_map, self.valid_mask

    def build_proposal_map(self):
        proposal_map = []  # x axis is duration, y axis is start
        valid_mask = []
        for dur_idx in range(self.dscale):
            for start_idx in range(self.tscale):
                end_idx = start_idx + dur_idx + 1
                if end_idx <= self.tscale:
                    proposal_map.append([start_idx, end_idx])
                    valid_mask.append(1)
                else:
                    proposal_map.append([0, 0])
                    valid_mask.append(0)

        proposal_map = torch.Tensor(proposal_map)
        self.proposal_map = proposal_map.reshape(self.dscale, self.tscale, 2)

        valid_mask = torch.Tensor(valid_mask)
        self.valid_mask = valid_mask.reshape(self.dscale, self.tscale).bool()

    def __repr__(self):
        return f"{self.__class__.__name__}(" f"tscale={self.tscale}, dscale={self.dscale})"
