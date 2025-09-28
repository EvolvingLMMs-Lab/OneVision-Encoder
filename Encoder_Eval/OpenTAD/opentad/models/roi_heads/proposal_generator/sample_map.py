import random
import math
import numpy as np
import torch
import torch.nn as nn
from ...builder import PROPOSAL_GENERATORS


@PROPOSAL_GENERATORS.register_module()
class ProposalMapSampling(nn.Module):
    def __init__(self, tscale, dscale, sampling_ratio=0.06, strategy="random"):
        super().__init__()

        self.tscale = tscale
        self.dscale = dscale
        self.sampling_ratio = sampling_ratio
        self.strategy = strategy
        self.build_proposal_map()

    def forward(self, bs, device, training=False):
        if training:
            if self.training:  # training
                anchors_init = [self.sample_proposal_map() for _ in range(bs)]
                anchors_init = torch.stack(anchors_init, dim=0)
            else:  # training but validation
                anchors_init = self.proposal_map[self.grid_mask.bool()]
                anchors_init = anchors_init.unsqueeze(0).repeat(bs, 1, 1)
        else:  # testing
            anchors_init = self.proposal_map[self.valid_mask.bool()]
            anchors_init = anchors_init.unsqueeze(0).repeat(bs, 1, 1)
        return anchors_init.to(device)  # [B,K,2]

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

        # grid mask for validation
        step = math.sqrt(1 / self.sampling_ratio)
        mask = np.zeros((self.dscale, self.tscale))
        for idx in np.arange(1, self.dscale, step).round().astype(int):
            for jdx in np.arange(1, self.tscale, step).round().astype(int):
                if jdx + idx < self.tscale:
                    mask[idx, jdx] = 1
        self.grid_mask = torch.Tensor(mask)

        self.sample_num = int(valid_mask.sum().int() * self.sampling_ratio)

    def sample_proposal_map(self):
        if self.strategy == "random":  # random select
            indices = random.sample(range(self.valid_mask.sum().int()), self.sample_num)
            indices = torch.Tensor(indices).long()
            indices = torch.nonzero(self.valid_mask)[indices]
            select_mask = torch.zeros_like(self.valid_mask)
            select_mask[indices[:, 0], indices[:, 1]] = 1
            anchors_init = self.proposal_map[select_mask.bool()]

        elif self.strategy == "grid":  # grid select
            anchors_init = self.proposal_map[self.grid_mask.bool()]

        elif self.strategy == "block":  # block select
            block_w = int(math.sqrt(self.sample_num))
            x = random.randint(0, self.dscale - 2 * block_w)
            y = random.randint(0, self.tscale - 2 * block_w - x)
            mask = np.zeros((self.dscale, self.tscale))
            for idx in range(x, x + block_w):
                for jdx in range(y, y + block_w):
                    if jdx + idx < self.tscale:
                        mask[idx, jdx] = 1
            block_mask = torch.Tensor(mask)
            anchors_init = self.proposal_map[block_mask.bool()]
        return anchors_init
