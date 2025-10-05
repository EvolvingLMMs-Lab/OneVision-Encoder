import torch
import torch.nn as nn

from ..bricks import ConvModule
from ..builder import PROJECTIONS


@PROJECTIONS.register_module()
class TemporalMaxerProj(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        arch=(2, 0, 5),  # (#convs, #stem, #branch)
        conv_cfg=None,  # kernel_size proj_pdrop
        norm_cfg=None,
        drop_out=0.0,
    ):
        super().__init__()
        assert len(arch) == 3

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.arch = arch
        self.kernel_size = conv_cfg["kernel_size"]
        self.scale_factor = 2  # as default
        self.with_norm = norm_cfg is not None

        self.drop_out = nn.Dropout1d(p=drop_out) if drop_out > 0 else None

        if isinstance(self.in_channels, (list, tuple)):
            assert isinstance(self.out_channels, (list, tuple)) and len(self.in_channels) == len(self.out_channels)
            self.proj = nn.ModuleList([])
            for n_in, n_out in zip(self.in_channels, self.out_channels):
                self.proj.append(
                    ConvModule(
                        n_in,
                        n_out,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                    )
                )
            in_channels = out_channels = sum(self.out_channels)
        else:
            self.proj = None

        # embedding network using convs
        self.embed = nn.ModuleList()
        for i in range(arch[0]):
            self.embed.append(
                ConvModule(
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    kernel_size=self.kernel_size,
                    stride=1,
                    padding=self.kernel_size // 2,
                    norm_cfg=norm_cfg,
                    act_cfg=dict(type="relu"),
                )
            )

        # stem network using (vanilla) transformer
        self.branch = nn.ModuleList()
        for _ in range(arch[2]):
            self.branch.append(
                TemporalMaxer(
                    kernel_size=3,
                    stride=self.scale_factor,
                    padding=1,
                )
            )

    def forward(self, x, mask):
        # x: batch size, feature channel, sequence length
        # mask: batch size, sequence length (bool)
        if self.drop_out is not None:
            x = self.drop_out(x)

        # feature projection
        if self.proj is not None:
            x = torch.cat([proj(s, mask)[0] for proj, s in zip(self.proj, x.split(self.in_channels, dim=1))], dim=1)

        # embedding network
        for idx in range(len(self.embed)):
            x, mask = self.embed[idx](x, mask)

        # prep for outputs
        out_feats = (x,)
        out_masks = (mask,)

        # main branch with downsampling
        for idx in range(len(self.branch)):
            x, mask = self.branch[idx](x, mask)
            out_feats += (x,)
            out_masks += (mask,)

        return out_feats, out_masks


class TemporalMaxer(nn.Module):
    def __init__(self, kernel_size, stride, padding):
        super().__init__()
        self.ds_pooling = nn.MaxPool1d(kernel_size, stride=stride, padding=padding)
        self.stride = stride

    def forward(self, x, mask):
        if self.stride > 1:
            out_mask = self.ds_pooling(mask.float()).bool()
        else:
            # masking out the features
            out_mask = mask

        out = self.ds_pooling(x) * out_mask.unsqueeze(1).to(x.dtype)

        return out, out_mask.bool()
