import torch
import torch.nn as nn
from ..builder import NECKS
from ..bricks import ConvModule


@NECKS.register_module()
class LSTMNeck(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        conv_cfg=dict(groups=4),
        norm_cfg=dict(type="GN", num_groups=16),
    ):
        super().__init__()

        self.mem_f = nn.LSTM(input_size=in_channels, hidden_size=in_channels, num_layers=1)
        self.mem_b = nn.LSTM(input_size=in_channels, hidden_size=in_channels, num_layers=1)

        self.layer2_f = ConvModule(
            in_channels,
            in_channels // 2,
            kernel_size=3,
            stride=1,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=dict(type="relu"),
        )

        self.layer2_b = ConvModule(
            in_channels,
            in_channels // 2,
            kernel_size=3,
            stride=1,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=dict(type="relu"),
        )

        self.layer2_d = ConvModule(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=dict(type="relu"),
        )

        self.layer3 = ConvModule(
            in_channels * 2,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=dict(type="relu"),
        )

    def forward(self, x, mask):
        # x shape [B,C,T]

        # flatten param for multi GPU training
        self.mem_f.flatten_parameters()
        self.mem_b.flatten_parameters()

        # memory
        x_f, _ = self.mem_f(x.permute(2, 0, 1))  # [T,B,C]
        x_b, _ = self.mem_b(x.permute(2, 0, 1).flip(dims=[0]))  # [T,B,C]

        # layer 2
        x_f, _ = self.layer2_f(x_f.permute(1, 2, 0), mask)
        x_b, _ = self.layer2_b(x_b.flip(dims=[0]).permute(1, 2, 0), mask)
        x, _ = self.layer2_d(x, mask)

        x = torch.cat((x_f, x, x_b), dim=1)

        # layer 3
        x, mask = self.layer3(x, mask)  # [B,C,T]
        return x, mask
