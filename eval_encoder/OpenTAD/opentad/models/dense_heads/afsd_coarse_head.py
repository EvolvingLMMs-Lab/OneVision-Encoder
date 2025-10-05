import torch
import torch.nn as nn

from ..builder import HEADS
from ..necks.afsd_neck import Unit1D


@HEADS.register_module()
class AFSDCoarseHead(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        frame_num,
        fpn_strides=[4, 8, 16, 32, 64, 128],
        num_classes=2,
        layer_num=6,
        feat_t=768 // 8,
    ):
        super().__init__()

        self.frame_num = frame_num
        self.fpn_strides = fpn_strides
        self.num_classes = num_classes

        loc_towers = []
        for i in range(2):
            loc_towers.append(
                nn.Sequential(
                    Unit1D(
                        in_channels=in_channels,
                        output_channels=out_channels,
                        kernel_shape=3,
                        stride=1,
                        use_bias=True,
                        activation_fn=None,
                    ),
                    nn.GroupNorm(32, out_channels),
                    nn.ReLU(inplace=True),
                )
            )
        self.loc_tower = nn.Sequential(*loc_towers)

        conf_towers = []
        for i in range(2):
            conf_towers.append(
                nn.Sequential(
                    Unit1D(
                        in_channels=in_channels,
                        output_channels=out_channels,
                        kernel_shape=3,
                        stride=1,
                        use_bias=True,
                        activation_fn=None,
                    ),
                    nn.GroupNorm(32, out_channels),
                    nn.ReLU(inplace=True),
                )
            )
        self.conf_tower = nn.Sequential(*conf_towers)

        # anchor priors
        self.priors = [
            torch.Tensor([[(c + 0.5) / (feat_t // 2**i), i] for c in range(feat_t // 2**i)]).view(-1, 2)
            for i in range(layer_num)
        ]

        # regression head
        self.loc_head = Unit1D(
            in_channels=out_channels,
            output_channels=2,
            kernel_shape=3,
            stride=1,
            use_bias=True,
            activation_fn=None,
        )
        self.scales = nn.ModuleList([ScaleExp() for _ in range(layer_num)])

        # classification head
        self.conf_head = Unit1D(
            in_channels=out_channels,
            output_channels=num_classes,
            kernel_shape=3,
            stride=1,
            use_bias=True,
            activation_fn=None,
        )

        # init weights
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @torch.no_grad()
    def generate_segments(self, loc, level):
        t = loc.shape[1]
        segments = loc / self.frame_num * t
        priors = self.priors[level][:, :1].expand(loc.size(0), t, 1).to(loc.device)
        new_priors = torch.round(priors * t - 0.5)
        plen = segments[:, :, :1] + segments[:, :, 1:]
        in_plen = torch.clamp(plen / 4.0, min=1.0)
        out_plen = torch.clamp(plen / 10.0, min=1.0)

        l_segment = new_priors - segments[:, :, :1]
        r_segment = new_priors + segments[:, :, 1:]
        segments = torch.cat(
            [
                torch.round(l_segment - out_plen),
                torch.round(l_segment + in_plen),
                torch.round(r_segment - in_plen),
                torch.round(r_segment + out_plen),
            ],
            dim=-1,
        )  # [B, T, 4], (0~T)

        decoded_segments = torch.cat(
            [
                priors[:, :, :1] * self.frame_num - loc[:, :, :1],
                priors[:, :, :1] * self.frame_num + loc[:, :, 1:],
            ],
            dim=-1,
        )
        plen = decoded_segments[:, :, 1:] - decoded_segments[:, :, :1] + 1.0
        in_plen = torch.clamp(plen / 4.0, min=1.0)
        out_plen = torch.clamp(plen / 10.0, min=1.0)
        frame_segments = torch.cat(
            [
                torch.round(decoded_segments[:, :, :1] - out_plen),
                torch.round(decoded_segments[:, :, :1] + in_plen),
                torch.round(decoded_segments[:, :, 1:] - in_plen),
                torch.round(decoded_segments[:, :, 1:] + out_plen),
            ],
            dim=-1,
        )  # [B, T, 4], (0~frame_num)
        return segments, frame_segments

    def forward_train(self, pyramid_feats, **kwargs):
        locs, confs = [], []
        loc_feat_list, conf_feat_list = [], []
        segments_list, frame_segments_list = [], []

        batch_num = pyramid_feats[0].size(0)
        for i, feat in enumerate(pyramid_feats):
            loc_feat = self.loc_tower(feat)  # [B,C,T]
            conf_feat = self.conf_tower(feat)  # [B,C,T]
            loc_feat_list.append(loc_feat)
            conf_feat_list.append(conf_feat)

            # regression head
            loc = self.scales[i](self.loc_head(loc_feat)).view(batch_num, 2, -1)
            loc = loc.permute(0, 2, 1).contiguous() * self.fpn_strides[i]  # [B,T,2]
            locs.append(loc)

            # classification head
            conf = self.conf_head(conf_feat).view(batch_num, self.num_classes, -1)
            conf = conf.permute(0, 2, 1).contiguous()  # [B,T,num_classes]
            confs.append(conf)

            # generate segments
            segments, frame_segments = self.generate_segments(loc, i)
            segments_list.append(segments)
            frame_segments_list.append(frame_segments)

        # compute the loss in the afsd.py, not here
        loc = torch.cat([o.view(batch_num, -1, 2) for o in locs], 1)  # [B,K,2]
        conf = torch.cat([o.view(batch_num, -1, self.num_classes) for o in confs], 1)  # [B,K,num_class]
        priors = torch.cat(self.priors, 0).to(loc.device).unsqueeze(0)  #  [1,K,2]
        return loc, conf, priors[0], loc_feat_list, conf_feat_list, segments_list, frame_segments_list

    def forward_test(self, pyramid_feats, **kwargs):
        return self.forward_train(pyramid_feats, **kwargs)


class ScaleExp(nn.Module):
    def __init__(self, init_value=1.0):
        super(ScaleExp, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return torch.exp(input * self.scale)
