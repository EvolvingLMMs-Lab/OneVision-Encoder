import torch
from ...builder import PRIOR_GENERATORS


@PRIOR_GENERATORS.register_module()
class PointGenerator:
    def __init__(
        self,
        strides,  # strides of fpn levels
        regression_range,  # regression range (on feature grids)
        use_offset=False,  # if to align the points at grid centers
    ):
        super().__init__()
        self.strides = strides
        self.regression_range = regression_range
        self.use_offset = use_offset

    def __call__(self, feat_list):
        # feat_list: list[B,C,T]

        pts_list = []
        for i, feat in enumerate(feat_list):
            T = feat.shape[-1]

            points = torch.linspace(0, T - 1, T, dtype=torch.float) * self.strides[i]  # [T]
            reg_range = torch.as_tensor(self.regression_range[i], dtype=torch.float)
            stride = torch.as_tensor(self.strides[i], dtype=torch.float)

            if self.use_offset:
                points += 0.5 * stride

            points = points[:, None]  # [T,1]
            reg_range = reg_range[None].repeat(T, 1)  # [T,2]
            stride = stride[None].repeat(T, 1)  # [T,1]
            pts_list.append(torch.cat((points, reg_range, stride), dim=1).to(feat.device))  # [T,4]
        return pts_list
