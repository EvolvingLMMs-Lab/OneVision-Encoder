import torch
import torch.nn as nn
from ..builder import HEADS
from ..necks.afsd_neck import Unit1D
from .roi_extractors.boundary_pooling.boundary_pooling_op import BoundaryMaxPooling


@HEADS.register_module()
class AFSDRefineHead(nn.Module):
    def __init__(self, in_channels, num_classes, **kwargs):
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes

        self.loc_proposal_branch = ProposalBranch(in_channels, 512)
        self.conf_proposal_branch = ProposalBranch(in_channels, 512)

        self.prop_loc_head = Unit1D(
            in_channels=in_channels,
            output_channels=2,
            kernel_shape=1,
            activation_fn=None,
        )
        self.prop_conf_head = Unit1D(
            in_channels=in_channels,
            output_channels=num_classes,
            kernel_shape=1,
            activation_fn=None,
        )

        self.center_head = Unit1D(
            in_channels=in_channels,
            output_channels=1,
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
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward_train(
        self,
        frame_level_feat,
        loc_feats,
        conf_feats,
        segments_list,
        frame_segments_list,
        **kwargs,
    ):
        start = frame_level_feat[:, : self.in_channels // 2].permute(0, 2, 1).contiguous()  # [B,T,C]
        end = frame_level_feat[:, self.in_channels // 2 :].permute(0, 2, 1).contiguous()  # [B,T,C]

        prop_locs = []
        prop_confs = []
        centers = []
        batch_num = frame_level_feat.size(0)

        for i, (loc_feat, conf_feat, segments, frame_segments) in enumerate(
            zip(loc_feats, conf_feats, segments_list, frame_segments_list)
        ):
            # boundary pooling to extract proposal features
            loc_prop_feat, loc_prop_feat_ = self.loc_proposal_branch(
                loc_feat,
                frame_level_feat,
                segments,
                frame_segments,
            )
            conf_prop_feat, conf_prop_feat_ = self.conf_proposal_branch(
                conf_feat,
                frame_level_feat,
                segments,
                frame_segments,
            )

            # level 0 to predict start/end
            if i == 0:
                ndim = loc_prop_feat_.size(1) // 2
                start_loc_prop = loc_prop_feat_[:, :ndim].permute(0, 2, 1).contiguous()
                end_loc_prop = loc_prop_feat_[:, ndim:].permute(0, 2, 1).contiguous()
                start_conf_prop = conf_prop_feat_[:, :ndim].permute(0, 2, 1).contiguous()
                end_conf_prop = conf_prop_feat_[:, ndim:].permute(0, 2, 1).contiguous()

            # regression
            prop_loc = self.prop_loc_head(loc_prop_feat)
            prop_loc = prop_loc.view(batch_num, 2, -1).permute(0, 2, 1).contiguous()
            prop_locs.append(prop_loc)  # [B,T,2]

            # classification
            prop_conf = self.prop_conf_head(conf_prop_feat)
            prop_conf = prop_conf.view(batch_num, self.num_classes, -1).permute(0, 2, 1).contiguous()
            prop_confs.append(prop_conf)  # [B,T,num_classes]

            # center
            center = self.center_head(loc_prop_feat)
            center = center.view(batch_num, 1, -1).permute(0, 2, 1).contiguous()  # [B,T,1]
            centers.append(center)

        prop_loc = torch.cat([o.view(batch_num, -1, 2) for o in prop_locs], 1)
        prop_conf = torch.cat([o.view(batch_num, -1, self.num_classes) for o in prop_confs], 1)
        center = torch.cat([o.view(batch_num, -1, 1) for o in centers], 1)
        return start, end, prop_loc, prop_conf, center, start_loc_prop, end_loc_prop, start_conf_prop, end_conf_prop

    def forward_test(
        self,
        frame_level_feat,
        loc_feats,
        conf_feats,
        segments_list,
        frame_segments_list,
        **kwargs,
    ):
        return self.forward_train(frame_level_feat, loc_feats, conf_feats, segments_list, frame_segments_list)


class ProposalBranch(nn.Module):
    def __init__(self, in_channels, proposal_channels):
        super(ProposalBranch, self).__init__()
        self.cur_point_conv = nn.Sequential(
            Unit1D(
                in_channels=in_channels,
                output_channels=proposal_channels,
                kernel_shape=1,
                activation_fn=None,
            ),
            nn.GroupNorm(32, proposal_channels),
            nn.ReLU(inplace=True),
        )
        self.lr_conv = nn.Sequential(
            Unit1D(
                in_channels=in_channels,
                output_channels=proposal_channels * 2,
                kernel_shape=1,
                activation_fn=None,
            ),
            nn.GroupNorm(32, proposal_channels * 2),
            nn.ReLU(inplace=True),
        )

        self.boundary_max_pooling = BoundaryMaxPooling()

        self.roi_conv = nn.Sequential(
            Unit1D(
                in_channels=proposal_channels,
                output_channels=proposal_channels,
                kernel_shape=1,
                activation_fn=None,
            ),
            nn.GroupNorm(32, proposal_channels),
            nn.ReLU(inplace=True),
        )

        self.proposal_conv = nn.Sequential(
            Unit1D(
                in_channels=proposal_channels * 4,
                output_channels=in_channels,
                kernel_shape=1,
                activation_fn=None,
            ),
            nn.GroupNorm(32, in_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, feature, frame_level_feature, segments, frame_segments):
        fm_short = self.cur_point_conv(feature)
        feature = self.lr_conv(feature)
        prop_feature = self.boundary_max_pooling(feature, segments)
        prop_roi_feature = self.boundary_max_pooling(frame_level_feature, frame_segments)
        prop_roi_feature = self.roi_conv(prop_roi_feature)
        prop_feature = torch.cat([prop_roi_feature, prop_feature, fm_short], dim=1)
        prop_feature = self.proposal_conv(prop_feature)
        return prop_feature, feature
