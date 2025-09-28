import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from .actionformer_proj import get_sinusoid_encoding
from ..bricks import ConvModule, AffineDropPath
from ..builder import PROJECTIONS

try:
    from mamba_ssm.ops.selective_scan_interface import mamba_inner_fn_no_out_proj

    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False

try:
    from flash_attn import flash_attn_qkvpacked_func

    FLASHATTN_AVAILABLE = True
except ImportError:
    FLASHATTN_AVAILABLE = False


@PROJECTIONS.register_module()
class CausalProj(nn.Module):
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
        mamba_kernel_size=4,  # kernel size of causal conv1d in mamba
        channel_expand=2,  # expand ratio for mamba
        num_head=4,  # number of heads in transformer
        drop_path_rate=0.3,
    ):
        super().__init__()
        assert (
            MAMBA_AVAILABLE
        ), "Please install mamba-ssm to use this module. Check: https://github.com/OpenGVLab/video-mamba-suite"
        assert (
            FLASHATTN_AVAILABLE
        ), "Please install flash-attention-2 to use this module. Check: https://github.com/Dao-AILab/flash-attentio"

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
            self.stem.append(
                HybridCausalBlock(
                    out_channels,
                    stride=1,
                    kernel_size=mamba_kernel_size,
                    expand=channel_expand,
                    num_head=num_head,
                    drop_path_rate=drop_path_rate,
                )
            )

        # main branch using transformer with pooling
        self.branch = nn.ModuleList()
        for _ in range(arch[2]):
            self.branch.append(
                HybridCausalBlock(
                    out_channels,
                    stride=2,
                    kernel_size=mamba_kernel_size,
                    expand=channel_expand,
                    num_head=num_head,
                    drop_path_rate=drop_path_rate,
                )
            )

        # init weights
        self.apply(self.__init_weights__)

    def __init_weights__(self, m):
        # set nn.Linear bias term to 0
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            if m.bias is not None:
                if not getattr(m.bias, "_no_reinit", False):
                    torch.nn.init.constant_(m.bias, 0.0)

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


class HybridCausalBlock(nn.Module):
    def __init__(
        self,
        n_embd,  # dimension of the input features
        stride=1,  # downsampling stride for the current layer
        kernel_size=4,  # conv kernel size
        expand=2,  # expand ratio for mamba
        num_head=4,  # number of heads in transformer
        drop_path_rate=0.3,  # drop path rate
    ):
        super().__init__()

        # normalization
        self.norm = nn.LayerNorm(n_embd, eps=1e-6)

        # hybrid block with mamba and self-attn
        self.block = MixtureCausalBlock(n_embd, d_conv=kernel_size, expand=expand, num_head=num_head)

        # downsampling
        if stride > 1:
            assert stride == 2
            self.downsample = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        else:
            self.downsample = None

        # drop path
        if drop_path_rate > 0.0:
            self.drop_path = AffineDropPath(n_embd, drop_prob=drop_path_rate, transpose=True)
        else:
            self.drop_path = nn.Identity()

    def forward(self, x, mask):
        x = x.permute(0, 2, 1)
        x = x + self.drop_path(self.block(self.norm(x)))
        x = x.permute(0, 2, 1)
        x = x * mask.unsqueeze(1).to(x.dtype)

        if self.downsample is not None:
            mask = self.downsample(mask.float()).bool()
            x = self.downsample(x) * mask.unsqueeze(1).to(x.dtype)
        return x, mask


class MixtureCausalBlock(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        num_head=4,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2 * 4, bias=bias)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)).clamp(
            min=dt_init_floor
        )
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner))  # Keep in fp32
        self.D._no_weight_decay = True

        # transformer
        self.qkv = nn.Conv1d(self.d_inner, self.d_inner * 3, kernel_size=3, padding=1, groups=self.d_inner)
        self.num_heads = num_head

        self.out_proj = nn.Linear(self.d_inner * 4, self.d_model, bias=bias)

    def forward(self, hidden_states):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        batch, seqlen, dim = hidden_states.shape

        # We do matmul and transpose BLH -> HBL at the same time
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        xz_f, xz_b = torch.chunk(xz, 2, dim=1)  # (B, D, L)
        xz = torch.cat([xz_f, xz_b.flip([-1])], dim=0)
        xz, xz_t = torch.chunk(xz, 2, dim=1)

        # causal conv1d -> ssm
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        # In the backward pass we write dx and dz next to each other to avoid torch.cat
        out = mamba_inner_fn_no_out_proj(
            xz,
            self.conv1d.weight,
            self.conv1d.bias,
            self.x_proj.weight,
            self.dt_proj.weight,
            A,
            None,  # input-dependent B
            None,  # input-dependent C
            self.D.float(),
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
        )

        # causal conv1d -> transformer
        x_t, z_t = torch.chunk(xz_t, 2, dim=1)
        B, _, L = x_t.shape
        qkv = self.qkv(x_t).transpose(1, 2).reshape(B, L, 3, self.num_heads, -1)
        x_t = flash_attn_qkvpacked_func(qkv, deterministic=True, causal=True)
        x_t = x_t.reshape(B, L, -1).transpose(1, 2)  # (B, D, L)

        out_t = x_t * F.silu(z_t)

        out = torch.cat([out, out_t], dim=1)
        out = out.chunk(2)
        out = torch.cat([out[0], out[1].flip([-1])], dim=1)
        out = F.linear(rearrange(out, "b d l -> b l d"), self.out_proj.weight, self.out_proj.bias)
        return out
