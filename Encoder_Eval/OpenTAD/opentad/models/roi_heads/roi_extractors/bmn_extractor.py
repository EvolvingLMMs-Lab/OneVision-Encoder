import torch
import torch.nn as nn
import numpy as np
import math
from ...builder import ROI_EXTRACTORS


@ROI_EXTRACTORS.register_module()
class BMNExtractor(nn.Module):
    """BM layer, implemented in BMN"""

    def __init__(
        self,
        in_channels=256,
        hid_channels=128,
        roi_channels=512,
        out_channels=128,
        tscale=128,
        dscale=128,
        num_sample=32,
        num_sample_perbin=3,
        prop_extend_ratio=0.5,
    ):
        super(BMNExtractor, self).__init__()

        self.num_sample = num_sample
        self.num_sample_perbin = num_sample_perbin
        self.prop_extend_ratio = prop_extend_ratio

        self.reduce_dim = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=hid_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(inplace=True),
        )

        self.conv3d = nn.Sequential(
            nn.Conv2d(
                hid_channels,
                roi_channels,
                kernel_size=(self.num_sample, 1),
                stride=(self.num_sample, 1),
            ),
            nn.ReLU(inplace=True),
        )

        self.conv2d = nn.Sequential(
            nn.Conv2d(roi_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )
        self._get_interp1d_mask(tscale, dscale)

    def forward(self, x):
        x = self.reduce_dim(x)  # [B, 128, tscale]
        map_base = torch.tensordot(x, self.sample_mask.to(x.device), dims=([2], [0]))  # [B, 128, 32, dscale, tscale]
        dscale, tscale = map_base.shape[-2:]
        map_base = map_base.flatten(3, 4)  # [B, 128, 32, dscale * tscale]
        map_3d = self.conv3d(map_base)  # [B, 512, 1, dscale*tscale]
        map_2d = map_3d.squeeze(2).unflatten(2, (dscale, tscale))  # [B, 512, dscale, tscale]
        map_2d = self.conv2d(map_2d)  # [B, out_dim, dscale, tscale]
        return map_2d

    def _get_interp1d_mask(self, tscale, dscale):
        # generate sample mask for each point in Boundary-Matching Map
        mask_mat = []
        for start_index in range(tscale):
            mask_mat_vector = []
            for duration_index in range(dscale):
                if start_index + duration_index < tscale:
                    p_xmin = start_index
                    p_xmax = start_index + duration_index
                    center_len = float(p_xmax - p_xmin) + 1
                    sample_xmin = p_xmin - center_len * self.prop_extend_ratio
                    sample_xmax = p_xmax + center_len * self.prop_extend_ratio
                    p_mask = self._get_interp1d_bin_mask(
                        sample_xmin,
                        sample_xmax,
                        tscale,
                        self.num_sample,
                        self.num_sample_perbin,
                    )
                else:
                    p_mask = np.zeros([tscale, self.num_sample])
                mask_mat_vector.append(p_mask)
            mask_mat_vector = np.stack(mask_mat_vector, axis=2)
            mask_mat.append(mask_mat_vector)
        mask_mat = np.stack(mask_mat, axis=3).astype(np.float32)
        self.sample_mask = torch.Tensor(mask_mat)

    def _get_interp1d_bin_mask(self, seg_xmin, seg_xmax, tscale, num_sample, num_sample_perbin):
        # generate sample mask for a boundary-matching pair
        plen = float(seg_xmax - seg_xmin)
        plen_sample = plen / (num_sample * num_sample_perbin - 1.0)
        total_samples = [seg_xmin + plen_sample * ii for ii in range(num_sample * num_sample_perbin)]
        p_mask = []
        for idx in range(num_sample):
            bin_samples = total_samples[idx * num_sample_perbin : (idx + 1) * num_sample_perbin]
            bin_vector = np.zeros([tscale])
            for sample in bin_samples:
                sample_upper = math.ceil(sample)
                sample_decimal, sample_down = math.modf(sample)
                if int(sample_down) <= (tscale - 1) and int(sample_down) >= 0:
                    bin_vector[int(sample_down)] += 1 - sample_decimal
                if int(sample_upper) <= (tscale - 1) and int(sample_upper) >= 0:
                    bin_vector[int(sample_upper)] += sample_decimal
            bin_vector = 1.0 / num_sample_perbin * bin_vector
            p_mask.append(bin_vector)
        p_mask = np.stack(p_mask, axis=1)
        return p_mask
