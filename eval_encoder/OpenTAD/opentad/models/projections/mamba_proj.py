import torch
import torch.nn as nn
import torch.nn.functional as F

from .actionformer_proj import get_sinusoid_encoding
from ..bricks import ConvModule, AffineDropPath
from ..builder import PROJECTIONS

try:
    from mamba_ssm.modules.mamba_simple import Mamba as ViM
    from mamba_ssm.modules.mamba_new import Mamba as DBM

    MAMBA_AVAILABLE = True

except ImportError:
    MAMBA_AVAILABLE = False


@PROJECTIONS.register_module()
class MambaProj(nn.Module):
    """Implementation of Video-Mamba-Suite: https://arxiv.org/abs/2403.09626"""

    def __init__(
        self,
        in_channels,
        out_channels,
        arch=(2, 2, 5),  # (#convs, #stem transformers, #branch transformers)
        conv_cfg=None,  # kernel_size proj_pdrop
        norm_cfg=None,
        use_abs_pe=False,  # use absolute position embedding
        max_seq_len=2304,
        input_pdrop=0.0,  # drop out the input feature
        mamba_cfg=dict(kernel_size=4, drop_path_rate=0.3, use_mamba_type="dbm"),  # default to DBM
    ):
        super().__init__()
        assert (
            MAMBA_AVAILABLE
        ), "Please install mamba-ssm to use this module. Check: https://github.com/OpenGVLab/video-mamba-suite"

        assert len(arch) == 3

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.arch = arch
        self.kernel_size = conv_cfg["kernel_size"]
        self.scale_factor = 2  # as default
        self.with_norm = norm_cfg is not None
        self.use_abs_pe = use_abs_pe
        self.max_seq_len = max_seq_len

        self.input_pdrop = nn.Dropout1d(p=input_pdrop) if input_pdrop > 0 else None

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

        # stem network using (vanilla) transformer
        self.stem = nn.ModuleList()
        for _ in range(arch[1]):
            self.stem.append(MaskMambaBlock(out_channels, **mamba_cfg))

        # main branch using transformer with pooling
        self.branch = nn.ModuleList()
        for _ in range(arch[2]):
            self.branch.append(MaskMambaBlock(out_channels, n_ds_stride=2, **mamba_cfg))

        # init weights
        self.apply(self.__init_weights__)

    def __init_weights__(self, module):
        # set nn.Linear bias term to 0
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                if not getattr(module.bias, "_no_reinit", False):
                    torch.nn.init.constant_(module.bias, 0.0)

    def forward(self, x, mask):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, sequence length (bool)

        # feature projection
        if self.proj is not None:
            x = torch.cat([proj(s, mask)[0] for proj, s in zip(self.proj, x.split(self.in_channels, dim=1))], dim=1)

        # drop out input if needed
        if self.input_pdrop is not None:
            x = self.input_pdrop(x)

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


class MaskMambaBlock(nn.Module):
    def __init__(
        self,
        n_embd,  # dimension of the input features
        kernel_size=4,  # conv kernel size
        n_ds_stride=1,  # downsampling stride for the current layer
        drop_path_rate=0.3,  # drop path rate
        use_mamba_type="dbm",
    ):
        super().__init__()
        if use_mamba_type == "dbm":
            self.mamba = DBM(n_embd, d_conv=kernel_size, use_fast_path=True, expand=1)
        elif use_mamba_type == "vim":
            # vim
            self.mamba = ViM(n_embd, d_conv=kernel_size, bimamba_type="v2", use_fast_path=True)
        else:
            raise NotImplementedError
        if n_ds_stride > 1:
            self.downsample = MaxPooler(kernel_size=3, stride=2, padding=1)
        else:
            self.downsample = None

        self.norm = nn.LayerNorm(n_embd, eps=1e-6)

        # drop path
        if drop_path_rate > 0.0:
            self.drop_path = AffineDropPath(n_embd, drop_prob=drop_path_rate)
        else:
            self.drop_path = nn.Identity()

    def forward(self, x, mask):
        res = x
        x_ = x.transpose(1, 2)
        x_ = self.norm(x_)
        x_ = self.mamba(x_).transpose(1, 2)
        x = x_ * mask.unsqueeze(1).to(x.dtype)

        x = res + self.drop_path(x)

        if self.downsample is not None:
            x, mask = self.downsample(x, mask)

        return x, mask


class MaxPooler(nn.Module):
    def __init__(
        self,
        kernel_size,
        stride,
        padding,
    ):
        super().__init__()
        self.ds_pooling = nn.MaxPool1d(kernel_size, stride=stride, padding=padding)

        self.stride = stride

    def forward(self, x, mask, **kwargs):
        if self.stride > 1:
            # downsample the mask using nearest neighbor
            out_mask = self.ds_pooling(mask.float()).bool()
        else:
            # masking out the features
            out_mask = mask

        out = self.ds_pooling(x) * out_mask.unsqueeze(1).to(x.dtype)

        return out, out_mask.bool()
