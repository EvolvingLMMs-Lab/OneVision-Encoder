from typing import OrderedDict
import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from torch.cuda.amp import custom_fwd, custom_bwd
from torch.nn.modules.utils import _ntuple, _triple
from mmcv.cnn import ConvModule, NonLocal3d, build_activation_layer
from mmengine.model.weight_init import kaiming_init
from mmengine.runner.checkpoint import _load_checkpoint, load_checkpoint, load_state_dict
from mmengine.utils.dl_utils.parrots_wrapper import _BatchNorm
from mmaction.registry import MODELS


class BasicBlock3d(nn.Module):
    """BasicBlock 3d block for ResNet3D.

    Args:
        inplanes (int): Number of channels for the input in first conv3d layer.
        planes (int): Number of channels produced by some norm/conv3d layers.
        spatial_stride (int): Spatial stride in the conv3d layer. Default: 1.
        temporal_stride (int): Temporal stride in the conv3d layer. Default: 1.
        dilation (int): Spacing between kernel elements. Default: 1.
        downsample (nn.Module | None): Downsample layer. Default: None.
        style (str): ``pytorch`` or ``caffe``. If set to "pytorch", the
            stride-two layer is the 3x3 conv layer, otherwise the stride-two
            layer is the first 1x1 conv layer. Default: 'pytorch'.
        inflate (bool): Whether to inflate kernel. Default: True.
        non_local (bool): Determine whether to apply non-local module in this
            block. Default: False.
        non_local_cfg (dict): Config for non-local module. Default: ``dict()``.
        conv_cfg (dict): Config dict for convolution layer.
            Default: ``dict(type='Conv3d')``.
        norm_cfg (dict): Config for norm layers. required keys are ``type``,
            Default: ``dict(type='BN3d')``.
        act_cfg (dict): Config dict for activation layer.
            Default: ``dict(type='ReLU')``.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
    """

    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        spatial_stride=1,
        temporal_stride=1,
        dilation=1,
        downsample=None,
        style="pytorch",
        inflate=True,
        non_local=False,
        non_local_cfg=dict(),
        conv_cfg=dict(type="Conv3d"),
        norm_cfg=dict(type="BN3d"),
        act_cfg=dict(type="ReLU"),
        with_cp=False,
        **kwargs,
    ):
        super().__init__()
        assert style in ["pytorch", "caffe"]
        # make sure that only ``inflate_style`` is passed into kwargs
        assert set(kwargs).issubset(["inflate_style"])

        self.inplanes = inplanes
        self.planes = planes
        self.spatial_stride = spatial_stride
        self.temporal_stride = temporal_stride
        self.dilation = dilation
        self.style = style
        self.inflate = inflate
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.with_cp = with_cp
        self.non_local = non_local
        self.non_local_cfg = non_local_cfg

        self.conv1_stride_s = spatial_stride
        self.conv2_stride_s = 1
        self.conv1_stride_t = temporal_stride
        self.conv2_stride_t = 1

        if self.inflate:
            conv1_kernel_size = (3, 3, 3)
            conv1_padding = (1, dilation, dilation)
            conv2_kernel_size = (3, 3, 3)
            conv2_padding = (1, 1, 1)
        else:
            conv1_kernel_size = (1, 3, 3)
            conv1_padding = (0, dilation, dilation)
            conv2_kernel_size = (1, 3, 3)
            conv2_padding = (0, 1, 1)

        self.conv1 = ConvModule(
            inplanes,
            planes,
            conv1_kernel_size,
            stride=(self.conv1_stride_t, self.conv1_stride_s, self.conv1_stride_s),
            padding=conv1_padding,
            dilation=(1, dilation, dilation),
            bias=False,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
        )

        self.conv2 = ConvModule(
            planes,
            planes * self.expansion,
            conv2_kernel_size,
            stride=(self.conv2_stride_t, self.conv2_stride_s, self.conv2_stride_s),
            padding=conv2_padding,
            bias=False,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=None,
        )

        self.downsample = downsample
        self.relu = build_activation_layer(self.act_cfg)

        if self.non_local:
            self.non_local_block = NonLocal3d(self.conv2.norm.num_features, **self.non_local_cfg)

    def forward(self, x):
        """Defines the computation performed at every call."""

        def _inner_forward(x):
            """Forward wrapper for utilizing checkpoint."""
            identity = x

            out = self.conv1(x)
            out = self.conv2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out = out + identity
            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)
        out = self.relu(out)

        if self.non_local:
            out = self.non_local_block(out)

        return out


class Bottleneck3d_inv_F(nn.Module):
    """Invertible Bottleneck 3d block F for ResNet3D.

    Args:
        inplanes (int): Number of channels for the input in first conv3d layer.
        planes (int): Number of channels produced by some norm/conv3d layers.
        spatial_stride (int): Spatial stride in the conv3d layer. Default: 1.
        temporal_stride (int): Temporal stride in the conv3d layer. Default: 1.
        dilation (int): Spacing between kernel elements. Default: 1.
        downsample (nn.Module | None): Downsample layer. Default: None.
        style (str): ``pytorch`` or ``caffe``. If set to "pytorch", the
            stride-two layer is the 3x3 conv layer, otherwise the stride-two
            layer is the first 1x1 conv layer. Default: 'pytorch'.
        inflate (bool): Whether to inflate kernel. Default: True.
        inflate_style (str): ``3x1x1`` or ``3x3x3``. which determines the
            kernel sizes and padding strides for conv1 and conv2 in each block.
            Default: '3x1x1'.
        non_local (bool): Determine whether to apply non-local module in this
            block. Default: False.
        non_local_cfg (dict): Config for non-local module. Default: ``dict()``.
        conv_cfg (dict): Config dict for convolution layer.
            Default: ``dict(type='Conv3d')``.
        norm_cfg (dict): Config for norm layers. required keys are ``type``,
            Default: ``dict(type='BN3d')``.
        act_cfg (dict): Config dict for activation layer.
            Default: ``dict(type='ReLU')``.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
    """

    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        spatial_stride=1,
        temporal_stride=1,
        dilation=1,
        style="pytorch",
        inflate=True,
        inflate_style="3x1x1",
        non_local=False,
        non_local_cfg=dict(),
        conv_cfg=dict(type="Conv3d"),
        norm_cfg=dict(type="BN3d"),
        act_cfg=dict(type="ReLU"),
        with_cp=False,
    ):
        super().__init__()
        assert style in ["pytorch", "caffe"]
        assert inflate_style in ["3x1x1", "3x3x3"]

        self.inplanes = inplanes
        self.planes = planes
        self.spatial_stride = spatial_stride
        self.temporal_stride = temporal_stride
        self.dilation = dilation
        self.style = style
        self.inflate = inflate
        self.inflate_style = inflate_style
        self.norm_cfg = norm_cfg
        self.conv_cfg = conv_cfg
        self.act_cfg = act_cfg
        self.with_cp = with_cp
        self.non_local = non_local
        self.non_local_cfg = non_local_cfg

        if self.style == "pytorch":
            self.conv1_stride_s = 1
            self.conv2_stride_s = spatial_stride
            self.conv1_stride_t = 1
            self.conv2_stride_t = temporal_stride
        else:
            self.conv1_stride_s = spatial_stride
            self.conv2_stride_s = 1
            self.conv1_stride_t = temporal_stride
            self.conv2_stride_t = 1

        if self.inflate:
            if inflate_style == "3x1x1":
                conv1_kernel_size = (3, 1, 1)
                conv1_padding = (1, 0, 0)
                conv2_kernel_size = (1, 3, 3)
                conv2_padding = (0, dilation, dilation)
            else:
                conv1_kernel_size = (1, 1, 1)
                conv1_padding = (0, 0, 0)
                conv2_kernel_size = (3, 3, 3)
                conv2_padding = (1, dilation, dilation)
        else:
            conv1_kernel_size = (1, 1, 1)
            conv1_padding = (0, 0, 0)
            conv2_kernel_size = (1, 3, 3)
            conv2_padding = (0, dilation, dilation)

        self.conv1 = ConvModule(
            inplanes,
            planes,
            conv1_kernel_size,
            stride=(self.conv1_stride_t, self.conv1_stride_s, self.conv1_stride_s),
            padding=conv1_padding,
            bias=False,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
        )

        self.conv2 = ConvModule(
            planes,
            planes,
            conv2_kernel_size,
            stride=(self.conv2_stride_t, self.conv2_stride_s, self.conv2_stride_s),
            padding=conv2_padding,
            dilation=(1, dilation, dilation),
            bias=False,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
        )

        self.conv3 = ConvModule(
            planes,
            planes * self.expansion,
            1,
            bias=False,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            # No activation in the third ConvModule for bottleneck
            act_cfg=None,
        )

        self.relu = build_activation_layer(self.act_cfg)

    def forward(self, x):
        """Defines the computation performed at every call."""

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        out = self.relu(out)

        return out


class Bottleneck3d_inv(nn.Module):
    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        spatial_stride=1,
        temporal_stride=1,
        dilation=1,
        downsample=None,
        style="pytorch",
        inflate=True,
        inflate_style="3x1x1",
        non_local=False,
        non_local_cfg=dict(),
        conv_cfg=dict(type="Conv3d"),
        norm_cfg=dict(type="BN3d"),
        act_cfg=dict(type="ReLU"),
        with_cp=False,
    ):
        super().__init__()

        Fm = Bottleneck3d_inv_F(
            inplanes,
            planes,
            spatial_stride=spatial_stride,
            temporal_stride=temporal_stride,
            dilation=dilation,
            style=style,
            inflate=inflate,
            inflate_style=inflate_style,
            non_local=non_local,
            non_local_cfg=non_local_cfg,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            with_cp=with_cp,
        )
        self.downsample = downsample

        if self.downsample is None:
            self.inv_block = InvFuncWrapper(Fm, split_dim=1)
        else:
            self.Fm = Fm

    def forward(self, x):
        if self.downsample is None:
            out = self.inv_block(x)
        else:
            out = self.Fm(x)
            identity = self.downsample(x)
            out = out + identity

            out = torch.cat((out, out), dim=1)

        return out

    def backward_pass(self, y1, y2, dy1, dy2):
        return self.inv_block.backward_pass(y1=y1, y2=y2, dy1=dy1, dy2=dy2)


class InvFuncWrapper(nn.Module):
    def __init__(self, Fm, split_dim=-1):
        super(InvFuncWrapper, self).__init__()
        """ This uses NICE's invertible functions.
        """
        self.Fm = Fm
        self.split_dim = split_dim

    def forward(self, x):
        """
        Forward:
            y1 = x1
            y2 = x2 + Fm(x1)
        """
        x1, x2 = torch.chunk(x, 2, dim=self.split_dim)
        x1, x2 = x1.contiguous(), x2.contiguous()

        y1 = x1
        y2 = x2 + self.Fm(x1)

        y = torch.cat([y2, y1], dim=self.split_dim)
        return y

    def inverse(self, y):
        """
        Inverse:
            x1 = y1
            x2 = y2 - Fm(y1)
        """
        y2, y1 = torch.chunk(y, 2, dim=self.split_dim)
        y2, y1 = y2.contiguous(), y1.contiguous()

        x1 = y1
        x2 = y2 - self.Fm(y1)

        x = torch.cat([x1, x2], dim=self.split_dim)

        return x

    def backward_pass(self, y1, y2, dy1, dy2):
        """This uses NICE's invertible functions.
        Forward:
            y1 = x1
            y2 = x2 + Fm(y1)
        Inverse:
            x1 = y1
            x2 = y2 - Fm(y1)
        """

        with torch.enable_grad():
            y1.requires_grad = True
            Fy1 = self.Fm(y1)
            Fy1.backward(dy2, retain_graph=True)

        with torch.no_grad():
            x2 = y2 - Fy1
            del Fy1
            dx1 = dy1 + y1.grad
            y1.grad = None

        with torch.no_grad():
            x1 = y1
            dx2 = dy2
            x2 = x2.detach()

        return x1, x2, dx1, dx2


class RevBackProp(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, x, blocks):
        for blk in blocks:
            x = blk(x)

        all_tensors = [x.detach()]
        ctx.save_for_backward(*all_tensors)
        ctx.blocks = blocks

        return x

    @staticmethod
    @custom_bwd
    def backward(ctx, dy):
        # retrieve params from ctx for backward
        y = ctx.saved_tensors[0]
        blocks = ctx.blocks

        dy2, dy1 = torch.chunk(dy, 2, dim=1)
        y2, y1 = torch.chunk(y, 2, dim=1)

        for i, blk in enumerate(blocks[::-1]):
            if i > 0:
                y2, y1 = y1, y2
                dy2, dy1 = dy1, dy2

            y1, y2, dy1, dy2 = blk.backward_pass(y1=y1, y2=y2, dy1=dy1, dy2=dy2)

        dx = torch.cat([dy1, dy2], dim=1)
        del y1, y2, dy1, dy2
        return dx, None


class BasicLayer(nn.Module):
    """A basic layer in resnet. Each layer contains multiple blocks.
    Multiple (usually 4) layers of different resolutions form a resnet pathway.
    """

    def __init__(
        self,
        lateral,
        channel_ratio,
        expansion,
        block,
        inplanes,
        planes,
        blocks,
        spatial_stride=1,
        temporal_stride=1,
        dilation=1,
        style="pytorch",
        inflate=1,
        inflate_style="3x1x1",
        non_local=0,
        non_local_cfg=dict(),
        conv_cfg=None,
        norm_cfg=None,
        act_cfg=None,
        with_cp=False,
        inv_mode="vanilla",
    ):
        super().__init__()

        self.inv_mode = inv_mode

        inflate = inflate if not isinstance(inflate, int) else (inflate,) * blocks
        non_local = non_local if not isinstance(non_local, int) else (non_local,) * blocks
        assert len(inflate) == blocks and len(non_local) == blocks
        if lateral:
            lateral_inplanes = inplanes * 2 // channel_ratio
        else:
            lateral_inplanes = 0
        if spatial_stride != 1 or (inplanes + lateral_inplanes) != planes * expansion:
            downsample = ConvModule(
                inplanes + lateral_inplanes,
                planes * expansion,
                kernel_size=1,
                stride=(temporal_stride, spatial_stride, spatial_stride),
                bias=False,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=None,
            )
        else:
            downsample = None

        self.blocks = nn.ModuleList()
        self.blocks.append(
            block(
                inplanes + lateral_inplanes,
                planes,
                spatial_stride,
                temporal_stride,
                dilation,
                downsample,
                style=style,
                inflate=(inflate[0] == 1),
                inflate_style=inflate_style,
                non_local=(non_local[0] == 1),
                non_local_cfg=non_local_cfg,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                with_cp=with_cp,
            )
        )
        inplanes = planes * block.expansion

        for i in range(1, blocks):
            self.blocks.append(
                block(
                    inplanes,
                    planes,
                    1,
                    1,
                    dilation,
                    style=style,
                    inflate=(inflate[i] == 1),
                    inflate_style=inflate_style,
                    non_local=(non_local[i] == 1),
                    non_local_cfg=non_local_cfg,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    with_cp=with_cp,
                )
            )

    def forward(self, x):
        x = self.blocks[0](x)

        if self.inv_mode == "vanilla":
            for blk in self.blocks[1:]:
                x = blk(x)
        elif self.inv_mode == "custom_backprop":
            x = RevBackProp.apply(x, self.blocks[1:])

        _, C, _, _, _ = x.shape
        C = int(C / 2)
        x = (x[:, :C, :, :, :] + x[:, C:, :, :, :]) / 2

        return x


class ResNet3d_inv(nn.Module):
    """ResNet 3d backbone.

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        pretrained (str | None): Name of pretrained model.
        stage_blocks (tuple | None): Set number of stages for each res layer.
            Default: None.
        pretrained2d (bool): Whether to load pretrained 2D model.
            Default: True.
        in_channels (int): Channel num of input features. Default: 3.
        base_channels (int): Channel num of stem output features. Default: 64.
        out_indices (Sequence[int]): Indices of output feature. Default: (3, ).
        num_stages (int): Resnet stages. Default: 4.
        spatial_strides (Sequence[int]):
            Spatial strides of residual blocks of each stage.
            Default: ``(1, 2, 2, 2)``.
        temporal_strides (Sequence[int]):
            Temporal strides of residual blocks of each stage.
            Default: ``(1, 1, 1, 1)``.
        dilations (Sequence[int]): Dilation of each stage.
            Default: ``(1, 1, 1, 1)``.
        conv1_kernel (Sequence[int]): Kernel size of the first conv layer.
            Default: ``(3, 7, 7)``.
        conv1_stride_s (int): Spatial stride of the first conv layer.
            Default: 2.
        conv1_stride_t (int): Temporal stride of the first conv layer.
            Default: 1.
        pool1_stride_s (int): Spatial stride of the first pooling layer.
            Default: 2.
        pool1_stride_t (int): Temporal stride of the first pooling layer.
            Default: 1.
        with_pool2 (bool): Whether to use pool2. Default: True.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer. Default: 'pytorch'.
        frozen_stages (int): Stages to be frozen (all param fixed). -1 means
            not freezing any parameters. Default: -1.
        inflate (Sequence[int]): Inflate Dims of each block.
            Default: (1, 1, 1, 1).
        inflate_style (str): ``3x1x1`` or ``3x3x3``. which determines the
            kernel sizes and padding strides for conv1 and conv2 in each block.
            Default: '3x1x1'.
        conv_cfg (dict): Config for conv layers. required keys are ``type``
            Default: ``dict(type='Conv3d')``.
        norm_cfg (dict): Config for norm layers. required keys are ``type`` and
            ``requires_grad``.
            Default: ``dict(type='BN3d', requires_grad=True)``.
        act_cfg (dict): Config dict for activation layer.
            Default: ``dict(type='ReLU', inplace=True)``.
        norm_eval (bool): Whether to set BN layers to eval mode, namely, freeze
            running stats (mean and var). Default: False.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        non_local (Sequence[int]): Determine whether to apply non-local module
            in the corresponding block of each stages. Default: (0, 0, 0, 0).
        non_local_cfg (dict): Config for non-local module. Default: ``dict()``.
        zero_init_residual (bool):
            Whether to use zero initialization for residual block,
            Default: True.
        kwargs (dict, optional): Key arguments for "make_res_layer".
    """

    arch_settings = {
        18: (BasicBlock3d, (2, 2, 2, 2)),
        34: (BasicBlock3d, (3, 4, 6, 3)),
        50: (Bottleneck3d_inv, (3, 4, 6, 3)),
        101: (Bottleneck3d_inv, (3, 4, 23, 3)),
        152: (Bottleneck3d_inv, (3, 8, 36, 3)),
    }

    def __init__(
        self,
        depth,
        pretrained,
        stage_blocks=None,
        pretrained2d=True,
        in_channels=3,
        num_stages=4,
        base_channels=64,
        out_indices=(3,),
        spatial_strides=(1, 2, 2, 2),
        temporal_strides=(1, 1, 1, 1),
        dilations=(1, 1, 1, 1),
        conv1_kernel=(3, 7, 7),
        conv1_stride_s=2,
        conv1_stride_t=1,
        pool1_stride_s=2,
        pool1_stride_t=1,
        with_pool2=True,
        style="pytorch",
        frozen_stages=-1,
        inflate=(1, 1, 1, 1),
        inflate_style="3x1x1",
        conv_cfg=dict(type="Conv3d"),
        norm_cfg=dict(type="BN3d", requires_grad=True),
        act_cfg=dict(type="ReLU", inplace=True),
        norm_eval=False,
        with_cp=False,
        inv_mode=["custom_backprop", "custom_backprop", "custom_backprop", "custom_backprop"],
        non_local=(0, 0, 0, 0),
        non_local_cfg=dict(),
        zero_init_residual=True,
        **kwargs,
    ):
        super().__init__()
        if depth not in self.arch_settings:
            raise KeyError(f"invalid depth {depth} for resnet")
        self.depth = depth
        self.pretrained = pretrained
        self.pretrained2d = pretrained2d
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.num_stages = num_stages
        assert 1 <= num_stages <= 4
        self.stage_blocks = stage_blocks
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.spatial_strides = spatial_strides
        self.temporal_strides = temporal_strides
        self.dilations = dilations
        assert len(spatial_strides) == len(temporal_strides) == len(dilations) == num_stages
        if self.stage_blocks is not None:
            assert len(self.stage_blocks) == num_stages

        self.conv1_kernel = conv1_kernel
        self.conv1_stride_s = conv1_stride_s
        self.conv1_stride_t = conv1_stride_t
        self.pool1_stride_s = pool1_stride_s
        self.pool1_stride_t = pool1_stride_t
        self.with_pool2 = with_pool2
        self.style = style
        self.frozen_stages = frozen_stages
        self.stage_inflations = _ntuple(num_stages)(inflate)
        self.non_local_stages = _ntuple(num_stages)(non_local)
        self.inflate_style = inflate_style
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.norm_eval = norm_eval
        self.with_cp = with_cp
        self.zero_init_residual = zero_init_residual

        self.block, stage_blocks = self.arch_settings[depth]

        if self.stage_blocks is None:
            self.stage_blocks = stage_blocks[:num_stages]

        self.inplanes = self.base_channels

        self.non_local_cfg = non_local_cfg

        self._make_stem_layer()

        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            spatial_stride = spatial_strides[i]
            temporal_stride = temporal_strides[i]
            dilation = dilations[i]
            planes = self.base_channels * 2**i
            res_layer = BasicLayer(
                self.lateral,
                self.channel_ratio,
                self.block.expansion,
                self.block,
                self.inplanes,
                planes,
                num_blocks,
                spatial_stride=spatial_stride,
                temporal_stride=temporal_stride,
                dilation=dilation,
                style=self.style,
                norm_cfg=self.norm_cfg,
                conv_cfg=self.conv_cfg,
                act_cfg=self.act_cfg,
                non_local=self.non_local_stages[i],
                non_local_cfg=self.non_local_cfg,
                inflate=self.stage_inflations[i],
                inflate_style=self.inflate_style,
                with_cp=with_cp,
                inv_mode=inv_mode[i],
                **kwargs,
            )
            self.inplanes = planes * self.block.expansion
            layer_name = f"layer{i + 1}"
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self.feat_dim = self.block.expansion * self.base_channels * 2 ** (len(self.stage_blocks) - 1)

    @staticmethod
    def _inflate_conv_params(conv3d, state_dict_2d, module_name_2d, inflated_param_names):
        """Inflate a conv module from 2d to 3d.

        Args:
            conv3d (nn.Module): The destination conv3d module.
            state_dict_2d (OrderedDict): The state dict of pretrained 2d model.
            module_name_2d (str): The name of corresponding conv module in the
                2d model.
            inflated_param_names (list[str]): List of parameters that have been
                inflated.
        """
        weight_2d_name = module_name_2d + ".weight"

        conv2d_weight = state_dict_2d[weight_2d_name]
        kernel_t = conv3d.weight.data.shape[2]

        new_weight = conv2d_weight.data.unsqueeze(2).expand_as(conv3d.weight) / kernel_t
        conv3d.weight.data.copy_(new_weight)
        inflated_param_names.append(weight_2d_name)

        if getattr(conv3d, "bias") is not None:
            bias_2d_name = module_name_2d + ".bias"
            conv3d.bias.data.copy_(state_dict_2d[bias_2d_name])
            inflated_param_names.append(bias_2d_name)

    @staticmethod
    def _inflate_bn_params(bn3d, state_dict_2d, module_name_2d, inflated_param_names):
        """Inflate a norm module from 2d to 3d.

        Args:
            bn3d (nn.Module): The destination bn3d module.
            state_dict_2d (OrderedDict): The state dict of pretrained 2d model.
            module_name_2d (str): The name of corresponding bn module in the
                2d model.
            inflated_param_names (list[str]): List of parameters that have been
                inflated.
        """
        for param_name, param in bn3d.named_parameters():
            param_2d_name = f"{module_name_2d}.{param_name}"
            param_2d = state_dict_2d[param_2d_name]
            param.data.copy_(param_2d)
            inflated_param_names.append(param_2d_name)

        for param_name, param in bn3d.named_buffers():
            param_2d_name = f"{module_name_2d}.{param_name}"
            # some buffers like num_batches_tracked may not exist in old
            # checkpoints
            if param_2d_name in state_dict_2d:
                param_2d = state_dict_2d[param_2d_name]
                param.data.copy_(param_2d)
                inflated_param_names.append(param_2d_name)

    @staticmethod
    def _inflate_weights(self):
        """Inflate the resnet2d parameters to resnet3d.

        The differences between resnet3d and resnet2d mainly lie in an extra
        axis of conv kernel. To utilize the pretrained parameters in 2d model,
        the weight of conv2d models should be inflated to fit in the shapes of
        the 3d counterpart.
        """

        state_dict_r2d = _load_checkpoint(self.pretrained)
        if "state_dict" in state_dict_r2d:
            state_dict_r2d = state_dict_r2d["state_dict"]

        inflated_param_names = []
        for name, module in self.named_modules():
            if isinstance(module, ConvModule):
                # we use a ConvModule to wrap conv+bn+relu layers, thus the
                # name mapping is needed
                if "downsample" in name:
                    # layer{X}.{Y}.downsample.conv->layer{X}.{Y}.downsample.0
                    original_conv_name = name + ".0"
                    # layer{X}.{Y}.downsample.bn->layer{X}.{Y}.downsample.1
                    original_bn_name = name + ".1"
                else:
                    # layer{X}.{Y}.conv{n}.conv->layer{X}.{Y}.conv{n}
                    original_conv_name = name
                    # layer{X}.{Y}.conv{n}.bn->layer{X}.{Y}.bn{n}
                    original_bn_name = name.replace("conv", "bn")
                if original_conv_name + ".weight" not in state_dict_r2d:
                    print(f"Module not exist in the state_dict_r2d" f": {original_conv_name}")
                else:
                    shape_2d = state_dict_r2d[original_conv_name + ".weight"].shape
                    shape_3d = module.conv.weight.data.shape
                    if shape_2d != shape_3d[:2] + shape_3d[3:]:
                        print(
                            f"Weight shape mismatch for "
                            f": {original_conv_name} : "
                            f"3d weight shape: {shape_3d}; "
                            f"2d weight shape: {shape_2d}. "
                        )
                    else:
                        self._inflate_conv_params(module.conv, state_dict_r2d, original_conv_name, inflated_param_names)

                if original_bn_name + ".weight" not in state_dict_r2d:
                    print(f"Module not exist in the state_dict_r2d" f": {original_bn_name}")
                else:
                    self._inflate_bn_params(module.bn, state_dict_r2d, original_bn_name, inflated_param_names)

        # check if any parameters in the 2d checkpoint are not loaded
        remaining_names = set(state_dict_r2d.keys()) - set(inflated_param_names)
        if remaining_names:
            print(f"These parameters in the 2d checkpoint are not loaded" f": {remaining_names}")

    def inflate_weights(self):
        self._inflate_weights(self)

    def _make_stem_layer(self):
        """Construct the stem layers consists of a conv+norm+act module and a
        pooling layer."""
        self.conv1 = ConvModule(
            self.in_channels,
            self.base_channels,
            kernel_size=self.conv1_kernel,
            stride=(self.conv1_stride_t, self.conv1_stride_s, self.conv1_stride_s),
            padding=tuple([(k - 1) // 2 for k in _triple(self.conv1_kernel)]),
            bias=False,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
        )

        self.maxpool = nn.MaxPool3d(
            kernel_size=(1, 3, 3),
            stride=(self.pool1_stride_t, self.pool1_stride_s, self.pool1_stride_s),
            padding=(0, 1, 1),
        )

        self.pool2 = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1))

    def _freeze_stages(self):
        """Prevent all the parameters from being optimized before
        ``self.frozen_stages``."""
        if self.frozen_stages >= 0:
            self.conv1.eval()
            for param in self.conv1.parameters():
                param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f"layer{i}")
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    @staticmethod
    def _init_weights(self, pretrained=None):
        """Initiate the parameters either from existing checkpoint or from
        scratch.

        Args:
            pretrained (str | None): The path of the pretrained weight. Will
                override the original `pretrained` if set. The arg is added to
                be compatible with mmdet. Default: None.
        """
        if pretrained:
            self.pretrained = pretrained
        if isinstance(self.pretrained, str):
            if self.pretrained2d:
                # Inflate 2D model into 3D model.
                self.inflate_weights()

            else:
                # Directly load 3D model.
                load_checkpoint(self, self.pretrained, strict=False)

    def init_weights(self, pretrained=None):
        self._init_weights(self, pretrained)

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The feature of the input
            samples extracted by the backbone.
        """
        x = self.conv1(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i == 0 and self.with_pool2:
                x = self.pool2(x)
            if i in self.out_indices:
                outs.append(x)
        if len(outs) == 1:
            return outs[0]

        return tuple(outs)

    def train(self, mode=True):
        """Set the optimization status when training."""
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()


class ResNet3dPathway(ResNet3d_inv):
    """A pathway of Slowfast based on ResNet3d.

    Args:
        *args (arguments): Arguments same as :class:``ResNet3d``.
        lateral (bool): Determines whether to enable the lateral connection
            from another pathway. Default: False.
        speed_ratio (int): Speed ratio indicating the ratio between time
            dimension of the fast and slow pathway, corresponding to the
            ``alpha`` in the paper. Default: 8.
        channel_ratio (int): Reduce the channel number of fast pathway
            by ``channel_ratio``, corresponding to ``beta`` in the paper.
            Default: 8.
        fusion_kernel (int): The kernel size of lateral fusion.
            Default: 5.
        **kwargs (keyword arguments): Keywords arguments for ResNet3d.
    """

    def __init__(
        self,
        *args,
        lateral=False,
        speed_ratio=8,
        channel_ratio=8,
        fusion_kernel=5,
        **kwargs,
    ):
        self.lateral = lateral
        self.speed_ratio = speed_ratio
        self.channel_ratio = channel_ratio
        self.fusion_kernel = fusion_kernel
        super().__init__(*args, **kwargs)
        self.inplanes = self.base_channels
        if self.lateral:
            self.conv1_lateral = ConvModule(
                self.inplanes // self.channel_ratio,
                # https://arxiv.org/abs/1812.03982, the
                # third type of lateral connection has out_channel:
                # 2 * \beta * C
                self.inplanes * 2 // self.channel_ratio,
                kernel_size=(fusion_kernel, 1, 1),
                stride=(self.speed_ratio, 1, 1),
                padding=((fusion_kernel - 1) // 2, 0, 0),
                bias=False,
                conv_cfg=self.conv_cfg,
                norm_cfg=None,
                act_cfg=None,
            )

        self.lateral_connections = []
        for i in range(len(self.stage_blocks)):
            planes = self.base_channels * 2**i
            self.inplanes = planes * self.block.expansion

            if lateral and i != self.num_stages - 1:
                # no lateral connection needed in final stage
                lateral_name = f"layer{(i + 1)}_lateral"
                setattr(
                    self,
                    lateral_name,
                    ConvModule(
                        self.inplanes // self.channel_ratio,
                        self.inplanes * 2 // self.channel_ratio,
                        kernel_size=(fusion_kernel, 1, 1),
                        stride=(self.speed_ratio, 1, 1),
                        padding=((fusion_kernel - 1) // 2, 0, 0),
                        bias=False,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=None,
                        act_cfg=None,
                    ),
                )
                self.lateral_connections.append(lateral_name)

    def inflate_weights(self):
        """Inflate the resnet2d parameters to resnet3d pathway.

        The differences between resnet3d and resnet2d mainly lie in an extra
        axis of conv kernel. To utilize the pretrained parameters in 2d model,
        the weight of conv2d models should be inflated to fit in the shapes of
        the 3d counterpart. For pathway the ``lateral_connection`` part should
        not be inflated from 2d weights.
        """

        state_dict_r2d = _load_checkpoint(self.pretrained)
        if "state_dict" in state_dict_r2d:
            state_dict_r2d = state_dict_r2d["state_dict"]

        inflated_param_names = []
        for name, module in self.named_modules():
            if "lateral" in name:
                continue
            if isinstance(module, ConvModule):
                # we use a ConvModule to wrap conv+bn+relu layers, thus the
                # name mapping is needed
                if "downsample" in name:
                    # layer{X}.{Y}.downsample.conv->layer{X}.{Y}.downsample.0
                    original_conv_name = name + ".0"
                    # layer{X}.{Y}.downsample.bn->layer{X}.{Y}.downsample.1
                    original_bn_name = name + ".1"
                else:
                    # layer{X}.{Y}.conv{n}.conv->layer{X}.{Y}.conv{n}
                    original_conv_name = name
                    # layer{X}.{Y}.conv{n}.bn->layer{X}.{Y}.bn{n}
                    original_bn_name = name.replace("conv", "bn")
                if original_conv_name + ".weight" not in state_dict_r2d:
                    print(f"Module not exist in the state_dict_r2d" f": {original_conv_name}")
                else:
                    self._inflate_conv_params(
                        module.conv,
                        state_dict_r2d,
                        original_conv_name,
                        inflated_param_names,
                    )
                if original_bn_name + ".weight" not in state_dict_r2d:
                    print(f"Module not exist in the state_dict_r2d" f": {original_bn_name}")
                else:
                    self._inflate_bn_params(
                        module.bn,
                        state_dict_r2d,
                        original_bn_name,
                        inflated_param_names,
                    )

        # check if any parameters in the 2d checkpoint are not loaded
        remaining_names = set(state_dict_r2d.keys()) - set(inflated_param_names)
        if remaining_names:
            print(f"These parameters in the 2d checkpoint are not loaded" f": {remaining_names}")

    def _inflate_conv_params(self, conv3d, state_dict_2d, module_name_2d, inflated_param_names):
        """Inflate a conv module from 2d to 3d.

        The differences of conv modules betweene 2d and 3d in Pathway
        mainly lie in the inplanes due to lateral connections. To fit the
        shapes of the lateral connection counterpart, it will expand
        parameters by concatting conv2d parameters and extra zero paddings.

        Args:
            conv3d (nn.Module): The destination conv3d module.
            state_dict_2d (OrderedDict): The state dict of pretrained 2d model.
            module_name_2d (str): The name of corresponding conv module in the
                2d model.
            inflated_param_names (list[str]): List of parameters that have been
                inflated.
        """
        weight_2d_name = module_name_2d + ".weight"
        conv2d_weight = state_dict_2d[weight_2d_name]
        old_shape = conv2d_weight.shape
        new_shape = conv3d.weight.data.shape
        kernel_t = new_shape[2]
        if new_shape[1] != old_shape[1]:
            # Inplanes may be different due to lateral connections
            new_channels = new_shape[1] - old_shape[1]
            pad_shape = old_shape
            pad_shape = pad_shape[:1] + (new_channels,) + pad_shape[2:]
            # Expand parameters by concat extra channels
            conv2d_weight = torch.cat(
                (
                    conv2d_weight,
                    torch.zeros(pad_shape).type_as(conv2d_weight).to(conv2d_weight.device),
                ),
                dim=1,
            )
        new_weight = conv2d_weight.data.unsqueeze(2).expand_as(conv3d.weight) / kernel_t
        conv3d.weight.data.copy_(new_weight)
        inflated_param_names.append(weight_2d_name)

        if getattr(conv3d, "bias") is not None:
            bias_2d_name = module_name_2d + ".bias"
            conv3d.bias.data.copy_(state_dict_2d[bias_2d_name])
            inflated_param_names.append(bias_2d_name)

    def _freeze_stages(self):
        """Prevent all the parameters from being optimized before
        `self.frozen_stages`."""
        if self.frozen_stages >= 0:
            self.conv1.eval()
            for param in self.conv1.parameters():
                param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f"layer{i}")
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

            if i != len(self.res_layers) and self.lateral:
                # No fusion needed in the final stage
                lateral_name = self.lateral_connections[i - 1]
                conv_lateral = getattr(self, lateral_name)
                conv_lateral.eval()
                for param in conv_lateral.parameters():
                    param.requires_grad = False

    def init_weights(self, pretrained=None):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        if pretrained:
            self.pretrained = pretrained

        # Override the init_weights of i3d
        super().init_weights()
        for module_name in self.lateral_connections:
            layer = getattr(self, module_name)
            for m in layer.modules():
                if isinstance(m, (nn.Conv3d, nn.Conv2d)):
                    kaiming_init(m)


pathway_cfg = {
    "resnet3d": ResNet3dPathway,
    # TODO: BNInceptionPathway
}


def build_pathway(cfg, *args, **kwargs):
    """Build pathway.

    Args:
        cfg (None or dict): cfg should contain:
            - type (str): identify conv layer type.

    Returns:
        nn.Module: Created pathway.
    """
    if not (isinstance(cfg, dict) and "type" in cfg):
        raise TypeError('cfg must be a dict containing the key "type"')
    cfg_ = cfg.copy()

    pathway_type = cfg_.pop("type")
    if pathway_type not in pathway_cfg:
        raise KeyError(f"Unrecognized pathway type {pathway_type}")

    pathway_cls = pathway_cfg[pathway_type]
    pathway = pathway_cls(*args, **kwargs, **cfg_)

    return pathway


@MODELS.register_module()
class ResNet3dSlowFast_inv(nn.Module):
    """Slowfast backbone.

    This module is proposed in `SlowFast Networks for Video Recognition
    <https://arxiv.org/abs/1812.03982>`_

    Args:
        pretrained (str): The file path to a pretrained model.
        resample_rate (int): A large temporal stride ``resample_rate``
            on input frames. The actual resample rate is calculated by
            multipling the ``interval`` in ``SampleFrames`` in the
            pipeline with ``resample_rate``, equivalent to the :math:`\\tau`
            in the paper, i.e. it processes only one out of
            ``resample_rate * interval`` frames. Default: 8.
        speed_ratio (int): Speed ratio indicating the ratio between time
            dimension of the fast and slow pathway, corresponding to the
            :math:`\\alpha` in the paper. Default: 8.
        channel_ratio (int): Reduce the channel number of fast pathway
            by ``channel_ratio``, corresponding to :math:`\\beta` in the paper.
            Default: 8.
        slow_pathway (dict): Configuration of slow branch, should contain
            necessary arguments for building the specific type of pathway
            and:
            type (str): type of backbone the pathway bases on.
            lateral (bool): determine whether to build lateral connection
            for the pathway.Default:

            .. code-block:: Python

                dict(type='ResNetPathway',
                lateral=True, depth=50, pretrained=None,
                conv1_kernel=(1, 7, 7), dilations=(1, 1, 1, 1),
                conv1_stride_t=1, pool1_stride_t=1, inflate=(0, 0, 1, 1))

        fast_pathway (dict): Configuration of fast branch, similar to
            `slow_pathway`. Default:

            .. code-block:: Python

                dict(type='ResNetPathway',
                lateral=False, depth=50, pretrained=None, base_channels=8,
                conv1_kernel=(5, 7, 7), conv1_stride_t=1, pool1_stride_t=1)
    """

    def __init__(
        self,
        pretrained,
        pretrain_type="resnet_3d",  # "resnet_2d", 'resnet_3d', 'inv_resnet_3d'
        resample_rate=8,
        speed_ratio=8,
        channel_ratio=8,
        slow_pathway=dict(
            type="resnet3d",
            depth=50,
            pretrained=None,
            lateral=True,
            conv1_kernel=(1, 7, 7),
            dilations=(1, 1, 1, 1),
            conv1_stride_t=1,
            pool1_stride_t=1,
            inflate=(0, 0, 1, 1),
            frozen_stages=-1,
        ),
        fast_pathway=dict(
            type="resnet3d",
            depth=50,
            pretrained=None,
            lateral=False,
            base_channels=8,
            conv1_kernel=(5, 7, 7),
            conv1_stride_t=1,
            pool1_stride_t=1,
            frozen_stages=-1,
        ),
        inv_mode=["custom_backprop", "custom_backprop", "custom_backprop", "custom_backprop"],
    ):
        super().__init__()
        self.pretrained = pretrained
        self.pretrain_type = pretrain_type
        self.resample_rate = resample_rate
        self.speed_ratio = speed_ratio
        self.channel_ratio = channel_ratio

        if slow_pathway["lateral"]:
            slow_pathway["speed_ratio"] = speed_ratio
            slow_pathway["channel_ratio"] = channel_ratio

        slow_pathway["inv_mode"] = inv_mode
        fast_pathway["inv_mode"] = inv_mode
        self.slow_path = build_pathway(slow_pathway)
        self.fast_path = build_pathway(fast_pathway)

    def init_weights(self, pretrained=None):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        if pretrained:
            self.pretrained = pretrained

        if isinstance(self.pretrained, str):
            print(f"load model from: {self.pretrained}")

            if self.pretrain_type == "inv_resnet_3d":
                self.load_frm_inv_resnet3d()
            elif self.pretrain_type == "resnet_3d":
                self.load_frm_resnet3d()
            elif self.pretrain_type == "resnet_2d":
                raise NotImplemented("Loading resent2d pretrained model is not implemented")
            else:
                # Directly load 3D model.
                load_checkpoint(self, self.pretrained, strict=True)
        elif self.pretrained is None:
            # Init two branch seperately.
            self.fast_path.init_weights()
            self.slow_path.init_weights()
        else:
            raise TypeError("pretrained must be a str or None")

    def load_frm_resnet3d(self):
        checkpoint = torch.load(self.pretrained, map_location="cpu")
        state_dict = checkpoint["state_dict"]

        state_dict_new = OrderedDict()
        for k in state_dict.keys():
            k_new = k

            if "backbone" in k:
                k_new = k.split(".")[1:]
                k_new = ".".join(k_new)

            if ("layer" in k_new) and ("lateral" not in k_new):
                k_new = k_new.split(".")
                k_new.insert(2, "blocks")

                if not "downsample" in k:
                    if "0" in k_new:
                        k_new.insert(4, "Fm")
                    else:
                        k_new.insert(4, "inv_block.Fm")

                k_new = ".".join(k_new)

            state_dict_new[k_new] = state_dict[k]

        msg = load_state_dict(self, state_dict_new, False)
        print(msg)
        print(f"=> loaded successfully '{self.pretrained}'")
        del checkpoint
        del state_dict_new
        torch.cuda.empty_cache()

    def load_frm_inv_resnet3d(self):
        checkpoint = torch.load(self.pretrained, map_location="cpu")
        state_dict = checkpoint["state_dict"]

        state_dict_new = OrderedDict()
        for k in state_dict.keys():
            k_new = k

            if "backbone" in k:
                k_new = k.split(".")[1:]
                k_new = ".".join(k_new)

            state_dict_new[k_new] = state_dict[k]

        msg = load_state_dict(self, state_dict_new, False)
        print(msg)
        print(f"=> loaded successfully '{self.pretrained}'")
        del checkpoint
        del state_dict_new
        torch.cuda.empty_cache()

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            tuple[torch.Tensor]: The feature of the input samples extracted
                by the backbone.
        """
        x_slow = nn.functional.interpolate(x, mode="nearest", scale_factor=(1.0 / self.resample_rate, 1.0, 1.0))
        x_slow = self.slow_path.conv1(x_slow)
        x_slow = self.slow_path.maxpool(x_slow)

        x_fast = nn.functional.interpolate(
            x,
            mode="nearest",
            scale_factor=(1.0 / (self.resample_rate // self.speed_ratio), 1.0, 1.0),
        )
        x_fast = self.fast_path.conv1(x_fast)
        x_fast = self.fast_path.maxpool(x_fast)

        if self.slow_path.lateral:
            x_fast_lateral = self.slow_path.conv1_lateral(x_fast)
            x_slow = torch.cat((x_slow, x_fast_lateral), dim=1)

        for i, layer_name in enumerate(self.slow_path.res_layers):
            res_layer = getattr(self.slow_path, layer_name)
            x_slow = res_layer(x_slow)
            res_layer_fast = getattr(self.fast_path, layer_name)
            x_fast = res_layer_fast(x_fast)
            if i != len(self.slow_path.res_layers) - 1 and self.slow_path.lateral:
                # No fusion needed in the final stage
                lateral_name = self.slow_path.lateral_connections[i]
                conv_lateral = getattr(self.slow_path, lateral_name)
                x_fast_lateral = conv_lateral(x_fast)
                x_slow = torch.cat((x_slow, x_fast_lateral), dim=1)

        return (x_slow, x_fast)
