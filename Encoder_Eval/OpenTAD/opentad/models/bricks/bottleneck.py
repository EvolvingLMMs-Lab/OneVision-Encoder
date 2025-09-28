import math
import torch
import torch.nn as nn

from .conv import ConvModule
from .transformer import DropPath, AffineDropPath


class Bottleneck(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        cardinality=1,
        base_width=64,
        norm_cfg=None,
        drop_path=0,
        expansion=4,
    ):
        super(Bottleneck, self).__init__()

        planes = out_channels / expansion
        width = int(math.floor(planes * (base_width / 64)) * cardinality)

        self.conv_bn_act_1 = ConvModule(
            in_channels,
            width,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=dict(type="relu"),
        )

        self.conv_bn_act_2 = ConvModule(
            width,
            width,
            kernel_size=3,
            stride=stride,
            padding=1,
            conv_cfg=dict(groups=cardinality),
            norm_cfg=norm_cfg,
            act_cfg=dict(type="relu"),
        )

        self.conv_bn_3 = ConvModule(
            width,
            out_channels,
            kernel_size=1,
            norm_cfg=norm_cfg,
        )

        self.act3 = nn.ReLU(inplace=True)

        if stride > 1 or in_channels != out_channels:
            self.downsample = ConvModule(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                norm_cfg=norm_cfg,
            )
        else:
            self.downsample = None

        self.drop_path = AffineDropPath(out_channels, drop_prob=drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x, mask=None):
        shortcut, shortcut_mask = x, mask

        x, mask = self.conv_bn_act_1(x, mask)
        x, mask = self.conv_bn_act_2(x, mask)
        x, mask = self.conv_bn_3(x, mask)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.downsample is not None:
            shortcut, shortcut_mask = self.downsample(shortcut, shortcut_mask)
        x += shortcut
        x = self.act3(x)
        return x, mask


class ConvNeXtV1Block(nn.Module):
    # this follows the implementation of ConvNext V1 design
    def __init__(self, dim, kernel_size=3, stride=1, expansion_ratio=4, drop_path=0):
        super().__init__()

        # depthwise conv
        self.dw_conv = nn.Conv1d(
            dim,
            dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            groups=dim,
        )
        self.norm = nn.LayerNorm(dim, eps=1e-6)

        # pointwise conv
        self.pw_conv1 = nn.Linear(dim, expansion_ratio * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pw_conv2 = nn.Linear(expansion_ratio * dim, dim)

        # drop path
        self.drop_path = AffineDropPath(dim, drop_prob=drop_path) if drop_path > 0 else nn.Identity()

        # residual
        if stride > 1:
            self.shortcut = nn.MaxPool1d(kernel_size=kernel_size, stride=stride, padding=kernel_size // 2)
        else:
            self.shortcut = nn.Identity()

        # init weights
        self.apply(self.__init_weights__)

    def __init_weights__(self, module):
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            # use pytorch's default init
            nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
            # set nn.Conv1d bias term to 0
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0.0)

    def forward(self, x, mask=None):
        residual = x

        x = self.dw_conv(x)
        x = x.permute(0, 2, 1)  # (N, C, T) -> (N, T, C)
        x = self.norm(x)
        x = self.pw_conv1(x)
        x = self.act(x)
        x = self.pw_conv2(x)
        x = x.permute(0, 2, 1)  # (N, T, C) -> (N, C, T)

        x = self.shortcut(residual) + self.drop_path(x)

        if mask == None:
            return x
        else:
            mask = self.shortcut(mask.float()).bool()
            x = x * mask.unsqueeze(1).float().detach()
            return x, mask


class ConvNeXtV2Block(nn.Module):
    # this follows the implementation of ConvNext V2 design

    def __init__(self, dim, kernel_size=3, stride=1, expansion_ratio=4, drop_path=0):
        super().__init__()

        # depthwise conv
        self.dw_conv = nn.Conv1d(
            dim,
            dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            groups=dim,
        )
        self.norm = nn.LayerNorm(dim, eps=1e-6)

        # pointwise conv
        self.pw_conv1 = nn.Linear(dim, expansion_ratio * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)  # new in ConvNeXtV2
        self.pw_conv2 = nn.Linear(expansion_ratio * dim, dim)

        # drop path
        self.drop_path = AffineDropPath(dim, drop_prob=drop_path) if drop_path > 0 else nn.Identity()

        # residual
        if stride > 1:
            self.shortcut = nn.MaxPool1d(kernel_size=kernel_size, stride=stride, padding=kernel_size // 2)
        else:
            self.shortcut = nn.Identity()

        # init weights
        self.apply(self.__init_weights__)

    def __init_weights__(self, module):
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            # use pytorch's default init
            nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
            # set nn.Conv1d bias term to 0
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0.0)

    def forward(self, x, mask=None):
        residual = x

        x = self.dw_conv(x)
        x = x.permute(0, 2, 1)  # (N, C, T) -> (N, T, C)
        x = self.norm(x)
        x = self.pw_conv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pw_conv2(x)
        x = x.permute(0, 2, 1)  # (N, T, C) -> (N, C, T)

        x = self.shortcut(residual) + self.drop_path(x)

        if mask == None:
            return x
        else:
            mask = self.shortcut(mask.float()).bool()
            x = x * mask.unsqueeze(1).float().detach()
            return x, mask


class GRN(nn.Module):
    """GRN (Global Response Normalization) layer"""

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=1, keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class ConvFormerBlock(nn.Module):
    def __init__(self, dim, kernel_size=3, expansion_ratio=4, drop_path=0):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.conv = nn.Conv1d(
            dim,
            dim,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )

        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(dim, expansion_ratio * dim),
            nn.GELU(),
            nn.Linear(expansion_ratio * dim, dim),
        )

        # drop path
        self.drop_path = AffineDropPath(dim, drop_prob=drop_path) if drop_path > 0 else nn.Identity()

        # init weights
        self.apply(self.__init_weights__)

    def __init_weights__(self, module):
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            # use pytorch's default init
            nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
            # set nn.Conv1d bias term to 0
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0.0)

    def forward(self, x, mask=None):
        x = x + self.drop_path(self.conv(self.norm1(x.permute(0, 2, 1)).permute(0, 2, 1)))
        x = x + self.drop_path(self.mlp(self.norm2(x.permute(0, 2, 1))).permute(0, 2, 1))

        if mask == None:
            return x
        else:
            x = x * mask.unsqueeze(1).float().detach()
            return x, mask
