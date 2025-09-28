import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from ..builder import NECKS


@NECKS.register_module()
class AFSDNeck(nn.Module):
    def __init__(self, in_channels, out_channels, frame_num, layer_num=6, e2e=False):
        super().__init__()

        self.frame_num = frame_num
        self.pyramids = nn.ModuleList()

        if e2e:
            input_conv = Unit3D(
                in_channels=in_channels,
                output_channels=out_channels,
                kernel_shape=[1, 3, 3],
                use_batch_norm=False,
                padding="spatial_valid",
                use_bias=True,
                activation_fn=None,
            )
        else:
            # feature-based model
            input_conv = Unit1D(
                in_channels=in_channels,
                output_channels=out_channels,
                kernel_shape=1,
                stride=1,
                use_bias=True,
                activation_fn=None,
            )

        self.pyramids.append(
            nn.Sequential(
                input_conv,
                nn.GroupNorm(32, out_channels),
                nn.ReLU(inplace=True),
            )
        )
        for _ in range(1, layer_num):
            self.pyramids.append(
                nn.Sequential(
                    Unit1D(
                        in_channels=out_channels,
                        output_channels=out_channels,
                        kernel_shape=3,
                        stride=2,
                        use_bias=True,
                        activation_fn=None,
                    ),
                    nn.GroupNorm(32, out_channels),
                    nn.ReLU(inplace=True),
                )
            )

        self.deconv = nn.Sequential(
            Unit1D(out_channels, out_channels, 3, activation_fn=None),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
            Unit1D(out_channels, out_channels, 3, activation_fn=None),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
            Unit1D(out_channels, out_channels, 1, activation_fn=None),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
        )

        # init the weights
        self.init_weights()

    def init_weights(self):
        def glorot_uniform_(tensor):
            fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
            scale = 1.0
            scale /= max(1.0, (fan_in + fan_out) / 2.0)
            limit = np.sqrt(3.0 * scale)
            return nn.init._no_grad_uniform_(tensor, -limit, limit)

        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv3d)):
                glorot_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, feat, mask):
        pyramid_feats = []
        for i, conv in enumerate(self.pyramids):
            if i == 0:
                x = conv(feat)
                x = x.squeeze(-1).squeeze(-1) * mask[:, None, :].detach().float()
            else:
                mask = F.max_pool1d(mask.unsqueeze(1).float(), kernel_size=2, stride=2).squeeze(1).bool()
                x = conv(x) * mask[:, None, :].detach().float()
            pyramid_feats.append(x)  # [B, C, 96/48/24/12/6/3]

        frame_level_feat = pyramid_feats[0].unsqueeze(-1)
        frame_level_feat = F.interpolate(frame_level_feat, [self.frame_num, 1]).squeeze(-1)
        frame_level_feat = self.deconv(frame_level_feat)  # [B, C, 768]
        return pyramid_feats, frame_level_feat


class Unit3D(nn.Module):
    def __init__(
        self,
        in_channels,
        output_channels,
        kernel_shape=(1, 1, 1),
        stride=(1, 1, 1),
        padding="spatial_valid",
        activation_fn=F.relu,
        use_batch_norm=False,
        use_bias=False,
    ):
        """Initializes Unit3D module."""
        super(Unit3D, self).__init__()

        self._output_channels = output_channels
        self._kernel_shape = kernel_shape
        self._stride = stride
        self._use_batch_norm = use_batch_norm
        self._activation_fn = activation_fn
        self._use_bias = use_bias
        self.padding = padding

        if self._use_batch_norm:
            self.bn = nn.BatchNorm3d(self._output_channels, eps=0.001, momentum=0.01)

        self.conv3d = nn.Conv3d(
            in_channels=in_channels,
            out_channels=self._output_channels,
            kernel_size=self._kernel_shape,
            stride=self._stride,
            padding=0,
            bias=self._use_bias,
        )

    def compute_pad(self, dim, s):
        if s % self._stride[dim] == 0:
            return max(self._kernel_shape[dim] - self._stride[dim], 0)
        else:
            return max(self._kernel_shape[dim] - (s % self._stride[dim]), 0)

    def forward(self, x):
        # compute 'same' padding
        if self.padding == "same":
            (batch, channel, t, h, w) = x.size()
            pad_t = self.compute_pad(0, t)
            pad_h = self.compute_pad(1, h)
            pad_w = self.compute_pad(2, w)

            pad_t_f = pad_t // 2
            pad_t_b = pad_t - pad_t_f
            pad_h_f = pad_h // 2
            pad_h_b = pad_h - pad_h_f
            pad_w_f = pad_w // 2
            pad_w_b = pad_w - pad_w_f

            pad = [pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b]
            x = F.pad(x, pad)

        if self.padding == "spatial_valid":
            (batch, channel, t, h, w) = x.size()
            pad_t = self.compute_pad(0, t)
            pad_t_f = pad_t // 2
            pad_t_b = pad_t - pad_t_f

            pad = [0, 0, 0, 0, pad_t_f, pad_t_b]
            x = F.pad(x, pad)

        x = self.conv3d(x)
        if self._use_batch_norm:
            x = self.bn(x)
        if self._activation_fn is not None:
            x = self._activation_fn(x)
        return x


class Unit1D(nn.Module):
    def __init__(
        self,
        in_channels,
        output_channels,
        kernel_shape=1,
        stride=1,
        padding="same",
        activation_fn=F.relu,
        use_bias=True,
    ):
        super(Unit1D, self).__init__()
        self.conv1d = nn.Conv1d(in_channels, output_channels, kernel_shape, stride, padding=0, bias=use_bias)
        self._activation_fn = activation_fn
        self._padding = padding
        self._stride = stride
        self._kernel_shape = kernel_shape

    def compute_pad(self, t):
        if t % self._stride == 0:
            return max(self._kernel_shape - self._stride, 0)
        else:
            return max(self._kernel_shape - (t % self._stride), 0)

    def forward(self, x):
        if self._padding == "same":
            batch, channel, t = x.size()
            pad_t = self.compute_pad(t)
            pad_t_f = pad_t // 2
            pad_t_b = pad_t - pad_t_f
            x = F.pad(x, [pad_t_f, pad_t_b])
        x = self.conv1d(x)
        if self._activation_fn is not None:
            x = self._activation_fn(x)
        return x
