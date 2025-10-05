import torch
from ...builder import PRIOR_GENERATORS


@PRIOR_GENERATORS.register_module()
class AnchorGenerator:
    def __init__(self, scales=[1, 2, 4], strides=[1, 2, 4]):
        super(AnchorGenerator, self).__init__()

        scales = torch.Tensor(scales)
        self.base_anchors = torch.stack([-0.5 * scales, 0.5 * scales], dim=-1)
        self.strides = strides

    def __call__(self, feat_list):
        assert len(feat_list) == len(self.strides)

        multi_level_anchors = []
        for stride, feat in zip(self.strides, feat_list):
            length = feat.shape[-1]
            shift_center = torch.arange(0, length)
            all_anchors = self.base_anchors[None, :, :] + shift_center[:, None, None]
            all_anchors = all_anchors.view(-1, 2) * stride

            # all_anchors [T*num_anchors, 2]
            multi_level_anchors.append(all_anchors.to(feat.dtype).to(feat.device))
        return multi_level_anchors
