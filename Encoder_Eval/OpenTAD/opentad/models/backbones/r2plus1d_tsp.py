import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
from torchvision.models.video.resnet import VideoResNet, R2Plus1dStem, BasicBlock
from torch.utils import checkpoint as cp
from mmengine.registry import MODELS


@MODELS.register_module()
class ResNet2Plus1d_TSP(VideoResNet):
    """ResNet (2+1)d backbone.
    This model is proposed in `A Closer Look at Spatiotemporal Convolutions for
    Action Recognition <https://arxiv.org/abs/1711.11248>`_
    """

    def __init__(
        self,
        layers=[3, 4, 6, 3],
        pretrained=None,
        norm_eval=True,
        with_cp=False,
        frozen_stages=-1,  # depth 34
    ):
        super().__init__(
            block=BasicBlockCP if with_cp else BasicBlock,
            conv_makers=[Conv2Plus1D] * 4,
            layers=layers,
            stem=R2Plus1dStem,
        )

        # We need exact Caffe2 momentum for BatchNorm scaling
        for m in self.modules():
            if isinstance(m, torch.nn.BatchNorm3d):
                m.eps = 1e-3
                m.momentum = 0.9

        self.fc = nn.Sequential()

        if pretrained != None:
            checkpoint = torch.load(pretrained, map_location="cpu")
            backbone_dict = {k[9:]: v for k, v in checkpoint["model"].items() if "fc" not in k}
            self.load_state_dict(backbone_dict)
            print("load pretrained model from {}".format(pretrained))

        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval

    def _freeze_stages(self):
        """Prevent all the parameters from being optimized before
        ``self.frozen_stages``."""
        if self.frozen_stages >= 0:
            self.stem.eval()
            for m in self.stem.modules():
                for param in m.parameters():
                    param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f"layer{i}")
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def _norm_eval(self):
        if self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()

    def forward(self, x):
        self._freeze_stages()
        self._norm_eval()

        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class Conv2Plus1D(nn.Sequential):
    def __init__(self, in_planes, out_planes, midplanes, stride=1, padding=1):
        midplanes = (in_planes * out_planes * 3 * 3 * 3) // (in_planes * 3 * 3 + 3 * out_planes)
        super(Conv2Plus1D, self).__init__(
            nn.Conv3d(
                in_planes,
                midplanes,
                kernel_size=(1, 3, 3),
                stride=(1, stride, stride),
                padding=(0, padding, padding),
                bias=False,
            ),
            nn.BatchNorm3d(midplanes),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                midplanes,
                out_planes,
                kernel_size=(3, 1, 1),
                stride=(stride, 1, 1),
                padding=(padding, 0, 0),
                bias=False,
            ),
        )

    @staticmethod
    def get_downsample_stride(stride):
        return (stride, stride, stride)


class BasicBlockCP(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, conv_builder, stride=1, downsample=None):
        midplanes = (inplanes * planes * 3 * 3 * 3) // (inplanes * 3 * 3 + 3 * planes)

        super(BasicBlockCP, self).__init__()
        self.conv1 = nn.Sequential(
            conv_builder(inplanes, planes, midplanes, stride),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(conv_builder(planes, planes, midplanes), nn.BatchNorm3d(planes))
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        def _inner_forward(x):
            """Forward wrapper for utilizing checkpoint."""
            residual = x

            out = self.conv1(x)
            out = self.conv2(out)
            if self.downsample is not None:
                residual = self.downsample(x)

            out += residual

            return out

        if x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)
        return out
