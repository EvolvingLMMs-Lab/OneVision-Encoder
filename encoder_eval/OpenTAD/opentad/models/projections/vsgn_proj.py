import torch.nn as nn
import torch.nn.functional as F
from .actionformer_proj import get_sinusoid_encoding
from ..builder import PROJECTIONS
from ..bricks import ConvModule
from ..bricks.gcn import xGN


@PROJECTIONS.register_module()
class VSGNPyramidProj(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        pyramid_levels=[2, 4, 8, 16, 32],
        conv_cfg=None,
        norm_cfg=None,  # dict(type="LN"),
        use_gcn=False,
        gcn_kwargs=dict(num_neigh=10, nfeat_mode="feat_ctr", agg_type="max", edge_weight="false"),
    ):
        super().__init__()

        # projection convs without downsampling
        init_stride = pyramid_levels[0] // 2
        self.stem_convs = ConvModule(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=init_stride,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=dict(type="relu"),
        )

        # downsampling convs
        scale_factor = pyramid_levels[1] // pyramid_levels[0]
        self.pyramid_convs = nn.ModuleList()
        for _ in range(len(pyramid_levels)):
            if use_gcn:
                block = xGN(
                    out_channels,
                    stride=scale_factor,
                    gcn_kwargs=gcn_kwargs,
                )
            else:
                block = ConvModule(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    stride=scale_factor,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=dict(type="relu"),
                )
            self.pyramid_convs.append(block)

    def forward(self, x, mask):
        # x shape [B,C,T], mask [B,T]

        # stem convs without downsampling
        x, mask = self.stem_convs(x, mask)

        # downsampling and saving to output
        out, out_mask = [], []
        for conv in self.pyramid_convs:
            x, mask = conv(x, mask)
            out.append(x)
            out_mask.append(mask)
        return out, out_mask
