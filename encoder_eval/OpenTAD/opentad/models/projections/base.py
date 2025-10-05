import torch.nn as nn
import torch.nn.functional as F
from .actionformer_proj import get_sinusoid_encoding
from ..builder import PROJECTIONS
from ..bricks import ConvModule


@PROJECTIONS.register_module()
class ConvSingleProj(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_convs,
        conv_cfg=None,
        norm_cfg=None,
        act_cfg=dict(type="relu"),
        drop_out=None,
    ):
        super().__init__()
        assert num_convs > 0
        self.drop_out = nn.Dropout1d(p=drop_out) if drop_out is not None else None

        self.convs = nn.ModuleList()
        for i in range(num_convs):
            self.convs.append(
                ConvModule(
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                )
            )

    def forward(self, x, mask):
        # x shape [B,C,T], mask [B,T]

        if self.drop_out is not None:
            x = self.drop_out(x)

        for conv in self.convs:
            x, mask = conv(x, mask)
        return x, mask


@PROJECTIONS.register_module()
class ConvPyramidProj(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        arch=(2, 5),  # (stem convs, downsample levels)
        conv_cfg=None,
        norm_cfg=dict(type="LN"),
        drop_out=0.0,
        drop_path=0.0,
        use_abs_pe=False,
        max_seq_len=-1,
    ):
        super().__init__()

        assert len(arch) == 2
        assert arch[1] > 0
        self.arch = arch
        self.use_abs_pe = use_abs_pe
        self.max_seq_len = max_seq_len

        self.drop_out = nn.Dropout1d(p=drop_out) if drop_out > 0 else None

        # projection convs without downsampling
        self.stem_convs = nn.ModuleList()
        for i in range(arch[0]):
            self.stem_convs.append(
                ConvModule(
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=dict(type="relu"),
                )
            )

        # position embedding (1, C, T), rescaled by 1/sqrt(n_embed)
        if self.use_abs_pe:
            pos_embed = get_sinusoid_encoding(self.max_seq_len, out_channels) / (out_channels**0.5)
            self.register_buffer("pos_embed", pos_embed, persistent=False)

        # downsampling for pyramid feature
        self.downsampling = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # convs between each level
        self.pyramid_convs = nn.ModuleList()
        for _ in range(arch[1] + 1):
            self.pyramid_convs.append(
                ConvModule(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=dict(type="relu"),
                )
            )

    def forward(self, x, mask):
        # x shape [B,C,T], mask [B,T]
        if self.drop_out is not None:
            x = self.drop_out(x)

        # stem convs without downsampling
        for conv in self.stem_convs:
            x, mask = conv(x, mask)

        # add position embedding
        if self.use_abs_pe:
            if self.training:
                assert x.shape[-1] <= self.max_seq_len, "Reached max length."
                pe = self.pos_embed
                # add pe to x
                x = x + pe[:, :, : x.shape[-1]] * mask.unsqueeze(1).to(x.dtype)
            else:
                if x.shape[-1] >= self.max_seq_len:
                    pe = F.interpolate(self.pos_embed, x.shape[-1], mode="linear", align_corners=False)
                else:
                    pe = self.pos_embed
                x = x + pe[:, :, : x.shape[-1]] * mask.unsqueeze(1).to(x.dtype)

        # downsampling and saving to output
        out, out_mask = [], []
        for level in range(self.arch[1] + 1):
            if level > 0:
                mask = self.downsampling(mask.float()).bool()
                x = self.downsampling(x) * mask.unsqueeze(1).to(x.dtype)

            x, mask = self.pyramid_convs[level](x, mask)
            out.append(x)
            out_mask.append(mask)
        return out, out_mask
