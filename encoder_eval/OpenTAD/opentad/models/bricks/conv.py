import copy
import math
import torch.nn.functional as F
import torch.nn as nn
from ..builder import MODELS


@MODELS.register_module()
class ConvModule(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        conv_cfg=None,
        norm_cfg=None,
        act_cfg=None,  # default to none to remind, act_cfg=dict(type="relu"),
    ):
        super().__init__()
        # norm config
        assert norm_cfg is None or isinstance(norm_cfg, dict)
        self.with_norm = norm_cfg is not None

        # conv config
        conv_cfg_base = dict(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        if self.with_norm:
            conv_cfg_base["bias"] = False  # bias is not necessary with a normalization layer

        assert conv_cfg is None or isinstance(conv_cfg, dict)
        if conv_cfg is not None:  # update conv_cfg_base
            conv_cfg_base.update(conv_cfg)

        # build conv layer
        self.conv = nn.Conv1d(**conv_cfg_base)

        # build norm layer
        if self.with_norm:
            norm_cfg = copy.copy(norm_cfg)  # make a copy
            norm_type = norm_cfg["type"]
            norm_cfg.pop("type")
            self.norm_type = norm_type

            if norm_type == "BN":
                self.norm = nn.BatchNorm1d(num_features=out_channels, **norm_cfg)
            elif norm_type == "GN":
                self.norm = nn.GroupNorm(num_channels=out_channels, **norm_cfg)
            elif norm_type == "LN":
                self.norm = nn.LayerNorm(out_channels, eps=1e-6)

        # build activation layer
        assert act_cfg is None or isinstance(act_cfg, dict)
        self.with_act = act_cfg is not None

        if self.with_act:
            act_cfg = copy.copy(act_cfg)  # make a copy
            act_type = act_cfg["type"]
            act_cfg.pop("type")

            if act_type == "relu":
                self.act = nn.ReLU(inplace=True, **act_cfg)
            else:  # other type
                self.act = eval(act_type)(**act_cfg)

        # init weights
        self.apply(self.__init_weights__)

    def __init_weights__(self, module):
        if isinstance(module, nn.Conv1d):
            # use pytorch's default init
            nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
            # set nn.Conv1d bias term to 0
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, (nn.BatchNorm1d, nn.GroupNorm, nn.LayerNorm)):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0.0)

    def forward(self, x, mask=None):
        x = self.conv(x)

        if mask is not None:  # masking before the norm
            if mask.shape[-1] != x.shape[-1]:
                mask = (
                    F.interpolate(mask.unsqueeze(1).to(x.dtype), size=x.size(-1), mode="nearest")
                    .squeeze(1)
                    .to(mask.dtype)
                )
            x = x * mask.unsqueeze(1).float().detach()  # [B,C,T]

        if self.with_norm:
            if self.norm_type == "LN":
                x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)
            else:
                x = self.norm(x)

        if self.with_act:
            x = self.act(x)

        if mask is not None:  # masking the output
            x = x * mask.unsqueeze(1).float().detach()  # [B,C,T]
            return x, mask
        else:
            return x
