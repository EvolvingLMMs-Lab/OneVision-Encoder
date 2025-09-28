import torch
import torch.nn as nn
import torch.nn.functional as F

from ..bricks import ConvModule, SGPBlock
from ..builder import PROJECTIONS
from .actionformer_proj import get_sinusoid_encoding


@PROJECTIONS.register_module()
class TriDetProj(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        sgp_mlp_dim,  # dim in SGP
        arch=(2, 2, 5),  # (#convs, #stem transformers, #branch transformers)
        sgp_win_size=[-1] * 6,  # size of local window for mha
        downsample_type="max",  # how to downsample feature in FPN
        k=1.5,
        init_conv_vars=1,  # initialization of gaussian variance for the weight in SGP
        conv_cfg=None,  # kernel_size
        norm_cfg=None,
        path_pdrop=0.0,  # dropout rate for drop path
        use_abs_pe=False,  # use absolute position embedding
        max_seq_len=2304,
        input_noise=0.0,
    ):
        super().__init__()
        assert len(arch) == 3
        assert len(sgp_win_size) == (1 + arch[2])

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.arch = arch
        self.kernel_size = conv_cfg["kernel_size"]
        self.scale_factor = 2  # as default

        self.path_pdrop = path_pdrop
        self.with_norm = norm_cfg is not None
        self.use_abs_pe = use_abs_pe
        self.max_seq_len = max_seq_len
        self.sgp_win_size = sgp_win_size
        self.downsample_type = downsample_type
        self.input_noise = input_noise

        # position embedding (1, C, T), rescaled by 1/sqrt(n_embed)
        if self.use_abs_pe:
            pos_embed = get_sinusoid_encoding(self.max_seq_len, out_channels) / (out_channels**0.5)
            self.register_buffer("pos_embed", pos_embed, persistent=False)

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

        # stem network using SGP blocks
        self.stem = nn.ModuleList()
        for idx in range(arch[1]):
            self.stem.append(
                SGPBlock(
                    out_channels,
                    kernel_size=1,
                    n_ds_stride=1,
                    n_hidden=sgp_mlp_dim,
                    k=k,
                    init_conv_vars=init_conv_vars,
                )
            )

        # main branch using SGP blocks with pooling
        self.branch = nn.ModuleList()
        for idx in range(arch[2]):
            self.branch.append(
                SGPBlock(
                    out_channels,
                    kernel_size=self.sgp_win_size[1 + idx],
                    n_ds_stride=self.scale_factor,
                    path_pdrop=self.path_pdrop,
                    n_hidden=sgp_mlp_dim,
                    downsample_type=downsample_type,
                    k=k,
                    init_conv_vars=init_conv_vars,
                )
            )

    def forward(self, x, mask):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, sequence length (bool)

        # trick, adding noise may slightly increases the variability between input features.
        if self.training and self.input_noise > 0:
            noise = torch.randn_like(x) * self.input_noise
            x += noise

        # embedding network
        for idx in range(len(self.embed)):
            x, mask = self.embed[idx](x, mask)

        # training: using fixed length position embeddings
        if self.use_abs_pe and self.training:
            assert x.shape[-1] <= self.max_seq_len, "Reached max length."
            pe = self.pos_embed
            # add pe to x
            x = x + pe[:, :, : x.shape[-1]] * mask.unsqueeze(1).to(x.dtype)

        # inference: re-interpolate position embeddings for over-length sequences
        if self.use_abs_pe and (not self.training):
            if x.shape[-1] >= self.max_seq_len:
                pe = F.interpolate(self.pos_embed, x.shape[-1], mode="linear", align_corners=False)
            else:
                pe = self.pos_embed
            # add pe to x
            x = x + pe[:, :, : x.shape[-1]] * mask.unsqueeze(1).to(x.dtype)

        # stem transformer
        for idx in range(len(self.stem)):
            x, mask = self.stem[idx](x, mask)

        # prep for outputs
        out_feats = (x,)
        out_masks = (mask,)

        # main branch with downsampling
        for idx in range(len(self.branch)):
            x, mask = self.branch[idx](x, mask)
            out_feats += (x,)
            out_masks += (mask,)

        return out_feats, out_masks
