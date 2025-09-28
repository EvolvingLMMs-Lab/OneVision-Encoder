import torch.nn as nn
import torch.nn.functional as F
import copy

from ..bricks import ConvModule
from ..builder import NECKS


@NECKS.register_module()
class FPN(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_levels,
        norm_cfg=None,
    ):
        super().__init__()

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        if isinstance(in_channels, int):
            in_channels = [in_channels] * num_levels

        for i in range(num_levels):
            self.lateral_convs.append(
                ConvModule(
                    in_channels[i],
                    out_channels,
                    kernel_size=1,
                    norm_cfg=norm_cfg,
                )
            )
            self.fpn_convs.append(
                ConvModule(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    norm_cfg=norm_cfg,
                )
            )

    def forward(self, input_list, mask_list):
        assert len(input_list) == len(self.lateral_convs)

        # build laterals
        laterals = [self.lateral_convs[i](input_list[i], mask_list[i])[0] for i in range(len(self.lateral_convs))]

        # build top-down path
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] += F.interpolate(laterals[i], scale_factor=2, mode="nearest")

        # build outputs
        fpn_outs = [self.fpn_convs[i](laterals[i], mask_list[i])[0] for i in range(len(laterals))]
        return fpn_outs, mask_list


@NECKS.register_module()
class FPNIdentity(nn.Module):
    def __init__(
        self,
        in_channels,  # input feature channels, len(in_channels) = #levels
        out_channels,  # output feature channel
        num_levels=0,
        scale_factor=2.0,  # downsampling rate between two fpn levels
        start_level=0,  # start fpn level
        end_level=-1,  # end fpn level
        norm_cfg=dict(type="LN"),  # if no norm, set to none
    ):
        super().__init__()

        self.in_channels = [in_channels] * num_levels
        self.out_channel = out_channels
        self.scale_factor = scale_factor

        self.start_level = start_level
        if end_level == -1:
            self.end_level = len(self.in_channels)
        else:
            self.end_level = end_level
        assert self.end_level <= len(self.in_channels)
        assert (self.start_level >= 0) and (self.start_level < self.end_level)

        if norm_cfg is not None:
            norm_cfg = copy.copy(norm_cfg)  # make a copy
            norm_type = norm_cfg["type"]
            norm_cfg.pop("type")
            self.norm_type = norm_type
        else:
            self.norm_type = None

        self.fpn_norms = nn.ModuleList()
        for i in range(self.start_level, self.end_level):
            # check feat dims
            assert self.in_channels[i] == self.out_channel

            if self.norm_type == "BN":
                fpn_norm = nn.BatchNorm1d(num_features=out_channels, **norm_cfg)
            elif self.norm_type == "GN":
                fpn_norm = nn.GroupNorm(num_channels=out_channels, **norm_cfg)
            elif self.norm_type == "LN":
                fpn_norm = nn.LayerNorm(out_channels, eps=1e-6)
            else:
                assert self.norm_type is None
                fpn_norm = nn.Identity()
            self.fpn_norms.append(fpn_norm)

    def forward(self, inputs, fpn_masks):
        # inputs must be a list / tuple
        assert len(inputs) == len(self.in_channels)
        assert len(fpn_masks) == len(self.in_channels)

        # apply norms, fpn_masks will remain the same with 1x1 convs
        fpn_feats = tuple()
        new_fpn_masks = tuple()
        for i in range(len(self.fpn_norms)):
            x = inputs[i + self.start_level]
            if self.norm_type == "LN":
                x = self.fpn_norms[i](x.permute(0, 2, 1)).permute(0, 2, 1)
            else:
                x = self.fpn_norms[i](x)
            fpn_feats += (x,)
            new_fpn_masks += (fpn_masks[i + self.start_level],)

        return fpn_feats, new_fpn_masks
