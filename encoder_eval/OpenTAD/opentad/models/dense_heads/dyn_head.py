import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import HEADS
from .actionformer_head import ActionFormerHead


@HEADS.register_module()
class TDynHead(ActionFormerHead):
    def __init__(
        self,
        num_classes,
        in_channels,
        feat_channels,
        num_convs=3,
        prior_generator=None,
        loss=None,
        loss_normalizer=100,
        loss_normalizer_momentum=0.9,
        center_sample="radius",
        center_sample_radius=1.5,
        label_smoothing=0,
        cls_prior_prob=0.01,
        loss_weight=1.0,
        head_kernel_size=3,
        dyn_head_cfg=dict(tau=1.5, init_gate=0.1, type="GeReTanH", dyn_type="c"),
        **kwargs,
    ):
        self.head_kernel_size = head_kernel_size
        self.dyn_head_cfg = dyn_head_cfg

        super().__init__(
            num_classes,
            in_channels,
            feat_channels,
            num_convs=num_convs,
            cls_prior_prob=cls_prior_prob,
            prior_generator=prior_generator,
            loss=loss,
            loss_normalizer=loss_normalizer,
            loss_normalizer_momentum=loss_normalizer_momentum,
            loss_weight=loss_weight,
            label_smoothing=label_smoothing,
            center_sample=center_sample,
            center_sample_radius=center_sample_radius,
            **kwargs,
        )

    def _init_cls_convs(self):
        """Initialize classification conv layers of the head."""
        self.cls_convs = nn.ModuleList([])
        for i in range(self.num_convs):
            cls_subnet_conv = DTFAM(
                dim=self.in_channels if i == 0 else self.feat_channels,
                o_dim=self.feat_channels,
                ka=self.head_kernel_size,
                conv_type="others",
                gate_activation=self.dyn_head_cfg["type"],
                gate_activation_kwargs=self.dyn_head_cfg,
            )
            self.cls_convs.append(
                DynamicScale_chk(
                    self.in_channels if i == 0 else self.feat_channels,
                    kernel_size=self.head_kernel_size,
                    num_groups=1,
                    num_adjacent_scales=2,
                    depth_module=cls_subnet_conv,
                    gate_activation_kwargs=self.dyn_head_cfg,
                )
            )

    def _init_reg_convs(self):
        """Initialize bbox regression conv layers of the head."""
        self.reg_convs = nn.ModuleList([])
        for i in range(self.num_convs):
            reg_subnet_conv = DTFAM(
                dim=self.in_channels if i == 0 else self.feat_channels,
                o_dim=self.feat_channels,
                ka=self.head_kernel_size,
                conv_type="others",
                gate_activation=self.dyn_head_cfg["type"],
                gate_activation_kwargs=self.dyn_head_cfg,
            )
            self.reg_convs.append(
                DynamicScale_chk(
                    self.in_channels if i == 0 else self.feat_channels,
                    kernel_size=self.head_kernel_size,
                    num_groups=1,
                    num_adjacent_scales=2,
                    depth_module=reg_subnet_conv,
                    gate_activation_kwargs=self.dyn_head_cfg,
                )
            )

    def forward_train(self, feat_list, mask_list, gt_segments, gt_labels, **kwargs):
        cls_pred = []
        reg_pred = []

        cls_feat_list = feat_list
        reg_feat_list = feat_list
        for i in range(self.num_convs):
            cls_feat_list = self.cls_convs[i](cls_feat_list, mask_list)
            reg_feat_list = self.reg_convs[i](reg_feat_list, mask_list)

        for l, (cls_feat, reg_feat) in enumerate(zip(cls_feat_list, reg_feat_list)):
            cls_pred.append(self.cls_head(cls_feat))
            reg_pred.append(F.relu(self.scale[l](self.reg_head(reg_feat))))

        points = self.prior_generator(feat_list)

        losses = self.losses(cls_pred, reg_pred, mask_list, points, gt_segments, gt_labels)
        return losses

    def forward_test(self, feat_list, mask_list, **kwargs):
        cls_pred = []
        reg_pred = []

        cls_feat_list = feat_list
        reg_feat_list = feat_list
        for i in range(self.num_convs):
            cls_feat_list = self.cls_convs[i](cls_feat_list, mask_list)
            reg_feat_list = self.reg_convs[i](reg_feat_list, mask_list)

        for l, (cls_feat, reg_feat) in enumerate(zip(cls_feat_list, reg_feat_list)):
            cls_pred.append(self.cls_head(cls_feat))
            reg_pred.append(F.relu(self.scale[l](self.reg_head(reg_feat))))

        points = self.prior_generator(feat_list)

        # get refined proposals and scores
        proposals, scores = self.get_valid_proposals_scores(points, reg_pred, cls_pred, mask_list)  # list [T,2]
        return proposals, scores


class DynamicScale_chk(nn.Module):
    def __init__(
        self,
        in_channels: int,
        kernel_size: int = 1,
        num_groups: int = 1,
        num_adjacent_scales: int = 2,
        depth_module: nn.Module = None,
        gate_activation_kwargs: dict = None,
    ):
        super(DynamicScale_chk, self).__init__()
        self.num_groups = num_groups
        self.num_adjacent_scales = num_adjacent_scales
        self.depth_module = depth_module

        dynamic_convs = [
            DTFAM(
                dim=in_channels,
                o_dim=in_channels,
                ka=kernel_size,
                gate_activation="GeReTanH",
                gate_activation_kwargs=gate_activation_kwargs,
            )
            for _ in range(num_adjacent_scales)
        ]
        self.dynamic_convs = nn.ModuleList(dynamic_convs)
        self.resize = lambda x, s: F.interpolate(x, size=s, mode="nearest")

        self.scale_weight = nn.Parameter(torch.zeros(1))
        self.output_weight = nn.Parameter(torch.ones(1))
        self.init_parameters()

    def init_parameters(self):
        for module in self.dynamic_convs:
            module.init_parameters()

    def forward(self, inputs, fpn_masks):
        dynamic_scales = []
        for l, x in enumerate(inputs):
            dynamic_scales.append([m(x, fpn_masks[l]) for m in self.dynamic_convs])

        outputs = []
        for l, x in enumerate(inputs):
            scale_feature = []

            for s in range(self.num_adjacent_scales):
                l_source = l + s - self.num_adjacent_scales // 2
                l_source = l_source if l_source < l else l_source + 1
                if l_source >= 0 and l_source < len(inputs):
                    feature = self.resize(dynamic_scales[l_source][s], x.shape[-1:])
                    scale_feature.append(feature)

            scale_feature = sum(scale_feature) * self.scale_weight + x * self.output_weight

            if self.depth_module is not None:
                scale_feature = self.depth_module(scale_feature, fpn_masks[l])

            outputs.append(F.relu(scale_feature))

        return outputs


class DTFAM(nn.Module):
    def __init__(
        self,
        dim=512,
        o_dim=1,
        ka=3,
        stride=1,
        groups=1,
        padding_mode="zeros",
        conv_type="gate",
        gate_activation: str = "ReTanH",
        gate_activation_kwargs: dict = None,
    ):
        super().__init__()

        self.dim = dim
        self.ka = ka
        self.stride = stride

        self.shift_conv = nn.Conv1d(
            dim,
            dim * ka,
            kernel_size=self.ka,
            stride=stride,
            bias=False,
            groups=dim,
            padding=self.ka // 2,
            padding_mode=padding_mode,
        )
        self.conv = nn.Conv1d(dim * ka, o_dim, kernel_size=1, groups=groups)

        dyn_type = gate_activation_kwargs["dyn_type"]
        self.conv_type = conv_type
        if self.conv_type == "gate":
            if dyn_type == "c":
                self.kernel_conv = TemporalGate(
                    dim,
                    num_groups=dim,
                    kernel_size=ka,
                    padding=ka // 2,
                    stride=1,
                    gate_activation=gate_activation,
                    gate_activation_kwargs=gate_activation_kwargs,
                )
            elif dyn_type == "k":
                self.kernel_conv = TemporalGate(
                    dim,
                    num_groups=ka,
                    kernel_size=ka,
                    padding=ka // 2,
                    stride=1,
                    gate_activation=gate_activation,
                    gate_activation_kwargs=gate_activation_kwargs,
                )

            elif dyn_type == "ck":
                self.kernel_conv = TemporalGate(
                    dim,
                    num_groups=dim * ka,
                    kernel_size=ka,
                    padding=ka // 2,
                    stride=1,
                    gate_activation=gate_activation,
                    gate_activation_kwargs=gate_activation_kwargs,
                )
            else:
                assert 1 == 0
        else:
            self.kernel_conv = DynamicConv1D_chk(
                in_channels=dim * self.ka,
                out_channels=dim,
                kernel_size=self.ka,
                padding=self.ka // 2,
                stride=stride,
                num_groups=groups,
                gate_activation=gate_activation,
                gate_activation_kwargs=gate_activation_kwargs,
            )

        self.norm = nn.LayerNorm(o_dim, eps=1e-6)

        self.init_parameters()

    def shift(self, x):
        # Pure shift operation, we do not use this operation in this repo.
        # We use constant kernel conv for shift.
        B, C, T = x.shape

        out = torch.zeros((B, self.ka * C, T), device=x.device)
        padx = F.pad(x, (self.ka // 2, self.ka // 2))

        for i in range(self.ka):
            out[:, i * C : (i + 1) * C, :] = padx[:, :, i : i + T]

        out = out.reshape(B, self.ka, C, T)
        out = torch.transpose(out, 1, 2)
        out = out.reshape(B, self.ka * C, T)

        return out

    def init_parameters(self):
        #  shift initialization for group convolution
        kernel = torch.zeros(self.ka, 1, self.ka)
        for i in range(self.ka):
            kernel[i, 0, i] = 1.0

        kernel = kernel.repeat(self.dim, 1, 1)
        self.shift_conv.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, x, mask):
        _x = self.shift_conv(x)

        if self.conv_type == "gate":
            weight = self.kernel_conv(_x, x, mask)
        else:
            weight = self.kernel_conv(_x, mask)
            weight = weight.repeat_interleave(self.ka, dim=1)
        _x = _x * weight

        out_conv = self.conv(_x)
        out_conv = self.norm(out_conv.permute(0, 2, 1)).permute(0, 2, 1)
        if self.stride > 1:
            # downsample the mask using nearest neighbor
            out_mask = F.interpolate(mask.unsqueeze(1).to(x.dtype), size=out_conv.size(-1), mode="nearest")
        else:
            out_mask = mask.unsqueeze(1).to(x.dtype)

        out_conv = out_conv * out_mask.detach()
        return out_conv


class DynamicConv1D_chk(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        padding: int = 0,
        stride: int = 1,
        num_groups: int = 1,
        gate_activation: str = "ReTanH",
        gate_activation_kwargs: dict = None,
    ):
        super(DynamicConv1D_chk, self).__init__()

        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=num_groups,
        )
        self.norm = nn.LayerNorm(out_channels, eps=1e-6)

        self.gate = TemporalGate(
            in_channels,
            num_groups=num_groups,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            gate_activation=gate_activation,
            gate_activation_kwargs=gate_activation_kwargs,
        )

    def forward(self, input, mask):
        out_mask = mask.unsqueeze(1).to(input.dtype)
        data = self.conv(input)
        data = self.norm(data.permute(0, 2, 1)).permute(0, 2, 1)
        data = data * out_mask.detach()
        output = self.gate(data, input, mask)
        # masking the output, stop grad to mask
        output = output * out_mask.detach()
        return output


class TemporalGate(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_groups: int = 1,
        kernel_size: int = 1,
        padding: int = 0,
        stride: int = 1,
        gate_activation: str = "ReTanH",
        gate_activation_kwargs: dict = None,
    ):
        super(TemporalGate, self).__init__()
        self.num_groups = num_groups
        self.in_channels = in_channels
        self.ka = kernel_size

        if num_groups == kernel_size:
            self.gate_conv = nn.Conv1d(
                in_channels=in_channels,
                out_channels=num_groups,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
        elif num_groups == kernel_size * in_channels:
            self.gate_conv = nn.Conv1d(
                in_channels=in_channels,
                out_channels=num_groups,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=in_channels,
            )
        else:
            self.gate_conv = nn.Conv1d(
                in_channels=in_channels,
                out_channels=num_groups,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=num_groups,
            )

        self.gate_activation = gate_activation
        self.gate_activation_kwargs = gate_activation_kwargs
        if gate_activation == "ReTanH":
            self.gate_activate = lambda x: torch.tanh(x).clamp(min=0)

        elif gate_activation == "ReLU":
            self.gate_activate = lambda x: torch.relu(x)

        elif gate_activation == "Sigmoid":
            self.gate_activate = lambda x: torch.sigmoid(x)

        elif gate_activation == "GeReTanH":
            assert "tau" in gate_activation_kwargs
            tau = gate_activation_kwargs["tau"]
            ttau = math.tanh(tau)
            self.gate_activate = lambda x: ((torch.tanh(x - tau) + ttau) / (1 + ttau)).clamp(min=0)
        else:
            raise NotImplementedError()

    def encode(self, *inputs):
        if self.num_groups == self.ka * self.in_channels:
            return inputs

        if self.num_groups == self.ka:
            da, mask = inputs
            b, ck, t = inputs[0].shape
            x = inputs[0].view(b, self.in_channels, self.ka, t)
            da = x.permute(0, 2, 1, 3).contiguous().view(b, ck, t)
            inputs = (da, mask)

        outputs = [x.view(x.shape[0] * self.num_groups, -1, *x.shape[2:]) for x in inputs]

        return outputs

    def decode(self, *inputs):
        if self.num_groups == self.ka * self.in_channels:
            return inputs

        outputs = [x.view(x.shape[0] // self.num_groups, -1, *x.shape[2:]) for x in inputs]
        return outputs

    def forward(self, data_input, gate_input, mask):
        # data_input b c h w

        out_mask = mask.unsqueeze(1).to(data_input.dtype)

        data = data_input * out_mask.detach()
        gate = self.gate_conv(gate_input)
        gate = self.gate_activate(gate)
        gate = gate * out_mask

        data, gate = self.encode(data_input, gate)
        (output,) = self.decode(data * gate)
        return output
