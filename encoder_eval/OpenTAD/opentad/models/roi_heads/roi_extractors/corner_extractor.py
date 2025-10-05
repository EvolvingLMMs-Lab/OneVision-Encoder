import torch
from ...builder import ROI_EXTRACTORS


@ROI_EXTRACTORS.register_module()
class CornerExtractor(object):
    def __init__(self, beta=8.0, base_stride=2, tscale=256):
        super(CornerExtractor, self).__init__()
        self.beta = beta
        self.tscale = tscale
        self.base_stride = base_stride

    def __call__(self, feat_frmlvl, loc_boxes):
        """
        Args:
              feat_frmlvl: B, channel, temporal
              loc_box: list of B length, B*[num_props_v, 2]
        Returns:
              start_feats: B*num_props, channel, temporal (3)
              end_feats: B*num_props, channel, temporal (3)
        """
        B, C, T = feat_frmlvl.shape

        # for each video
        start_feats_center = []
        start_feats_left = []
        start_feats_right = []
        end_feats_center = []
        end_feats_left = []
        end_feats_right = []
        for i, loc_box in enumerate(loc_boxes):
            loc_box = torch.clamp(loc_box, min=0.0, max=(self.tscale) - 1)
            boundary_length = (loc_box[:, 1] - loc_box[:, 0] + 1) / self.beta

            # Starts
            starts = torch.clamp(
                (loc_box[:, 0] / self.base_stride).to(dtype=torch.long),
                min=0,
                max=T - 1,
            )
            starts_left = torch.clamp(
                ((loc_box[:, 0] - boundary_length) / self.base_stride).to(dtype=torch.long),
                min=0,
                max=T - 1,
            )
            starts_right = torch.clamp(
                ((loc_box[:, 0] + boundary_length) / self.base_stride).to(dtype=torch.long),
                min=0,
                max=T - 1,
            )

            start_feats_center.append(feat_frmlvl[i, :, starts].permute(1, 0))
            start_feats_left.append(feat_frmlvl[i, :, starts_left].permute(1, 0))
            start_feats_right.append(feat_frmlvl[i, :, starts_right].permute(1, 0))

            # Ends
            ends = torch.clamp(
                (loc_box[:, 1] / self.base_stride).to(dtype=torch.long),
                min=0,
                max=T - 1,
            )
            ends_left = torch.clamp(
                ((loc_box[:, 1] - boundary_length) / self.base_stride).to(dtype=torch.long),
                min=0,
                max=T - 1,
            )
            ends_right = torch.clamp(
                ((loc_box[:, 1] + boundary_length) / self.base_stride).to(dtype=torch.long),
                min=0,
                max=T - 1,
            )

            end_feats_center.append(feat_frmlvl[i, :, ends].permute(1, 0))
            end_feats_left.append(feat_frmlvl[i, :, ends_left].permute(1, 0))
            end_feats_right.append(feat_frmlvl[i, :, ends_right].permute(1, 0))

        start_feats_center = torch.cat(start_feats_center, dim=0)
        start_feats_left = torch.cat(start_feats_left, dim=0)
        start_feats_right = torch.cat(start_feats_right, dim=0)
        start_feats = torch.stack((start_feats_left, start_feats_center, start_feats_right), dim=-1)

        end_feats_center = torch.cat(end_feats_center, dim=0)
        end_feats_left = torch.cat(end_feats_left, dim=0)
        end_feats_right = torch.cat(end_feats_right, dim=0)
        end_feats = torch.stack((end_feats_left, end_feats_center, end_feats_right), dim=-1)

        return torch.cat((start_feats, end_feats), dim=-1)
