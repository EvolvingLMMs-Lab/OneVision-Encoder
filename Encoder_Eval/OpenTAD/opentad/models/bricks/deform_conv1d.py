import math
import torch
from torch import nn, Tensor
from torch.nn.parameter import Parameter
from torch.nn import init
from typing import Optional

import torch.nn as nn
from torchvision.ops.deform_conv import deform_conv2d


def deform_conv1d(
    input: Tensor,  # shape [B,in_channels,T_in]
    offset: Tensor,  # shape [B,kernel_size_t,T_out]
    weight: Tensor,  # shape [out_channels,in_channels,kernel_size_t]
    bias: Optional[Tensor] = None,  # shape [out_channels]
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    mask: Optional[Tensor] = None,  # not implemented yet
) -> Tensor:
    assert mask == None, "mask is not implemented yet"

    input2d = input.unsqueeze(-1)  # [B,in_channels,T_in,1]
    offset2d = torch.zeros_like(offset).repeat(1, 2, 1).unsqueeze(-1)  # [B,2*kernel_size_t*1,T_out,1]
    offset2d[:, 0::2, :, 0] += offset  # add offset to H dimension
    weight2d = weight.unsqueeze(-1)  # [out_channels,in_channels,kernel_size_t,1]

    out_2d = deform_conv2d(
        input=input2d,
        offset=offset2d,
        weight=weight2d,
        bias=bias,
        stride=(stride, 1),
        padding=(padding, 0),
        dilation=(dilation, 1),
        mask=mask,
    )
    out = out_2d.squeeze(-1)  # [B,out_channels,T_out]
    return out


class DeformConv1d(nn.Module):
    """We use  torchvision.ops.deform_conv2d to implement deform_conv1d."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()

        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.weight = Parameter(torch.empty(out_channels, in_channels // groups, self.kernel_size))

        if bias:
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor, offset: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            input (Tensor[batch_size, in_channels, in_length]): input tensor
            offset (Tensor[batch_size, offset_groups * kernel_length, out_length]):
                offsets to be applied for each position in the convolution kernel.
            mask (Tensor[batch_size, offset_groups * kernel_length, out_length]):
                masks to be applied for each position in the convolution kernel.
        """
        return deform_conv1d(
            input,
            offset,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            mask=mask,
        )

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"{self.in_channels}"
            f", {self.out_channels}"
            f", kernel_size={self.kernel_size}"
            f", stride={self.stride}"
        )
        s += f", padding={self.padding}" if self.padding != 0 else ""
        s += f", dilation={self.dilation}" if self.dilation != 1 else ""
        s += f", groups={self.groups}" if self.groups != 1 else ""
        s += ", bias=False" if self.bias is None else ""
        s += ")"

        return s
