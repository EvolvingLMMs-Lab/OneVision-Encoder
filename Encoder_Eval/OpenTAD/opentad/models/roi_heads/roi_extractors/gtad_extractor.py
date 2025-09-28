import torch
import torch.nn as nn
from ...builder import ROI_EXTRACTORS
from ...bricks import GCNeXt
from ...bricks.gcnext import get_graph_feature
from .align1d.align import Align1DLayer


@ROI_EXTRACTORS.register_module()
class GTADExtractor(nn.Module):
    """GraphAlign layer, implemented in GTAD"""

    def __init__(
        self,
        in_channels=256,
        out_channels=128,
        tscale=128,
        dscale=128,
        k=3,
        roi_size=32,
        context_size=4,
        prop_extend_ratio=0.5,
    ):
        super(GTADExtractor, self).__init__()

        self.k = k
        self.prop_extend_ratio = prop_extend_ratio

        self.gcnext = GCNeXt(in_channels, in_channels, k=3, groups=32)

        self.align_inner = Align1DLayer(roi_size, ratio=0)
        self.align_context = Align1DLayer(context_size)

        self.conv2d = nn.Sequential(
            nn.Conv2d(
                in_channels * (roi_size + context_size),
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.ReLU(inplace=True),
        )
        self._get_anchors(tscale, dscale)

    def forward(self, x):
        bs = x.shape[0]
        device = x.device
        dscale, tscale = self.anchors.shape[:2]

        x = self.gcnext(x)

        # add batch idx
        anchors = self.anchors.reshape(-1, 2).unsqueeze(0).repeat(bs, 1, 1)
        bs_idxs = torch.arange(bs).view(bs, 1, 1).float()
        bs_idxs = bs_idxs.repeat(1, anchors.shape[1], 1)
        anchors = torch.cat((bs_idxs, anchors), dim=2).to(device)  # [B,K,3]
        anchors = anchors.view(-1, 3)  # [B*dscale*tscale,3]

        feat_inner = self.align_inner(x, anchors)  # (bs*dscale*tscal, ch, roi_size)

        feat = get_graph_feature(x, k=self.k, style=2)  # (bs,ch,100) -> (bs, ch, 100, k)
        feat = feat.mean(dim=-1, keepdim=False)  # (bs. 2*ch, 100)
        feat_context = self.align_context(feat, anchors)  # (bs*dscale*tscal, ch, roi_size//2)
        feat = torch.cat((feat_inner, feat_context), dim=2).view(bs, dscale, tscale, -1)
        feat = feat.permute(0, 3, 1, 2).contiguous()  # (bs,2*ch*(-1),t,t)

        map_2d = self.conv2d(feat)  # [B, out_dim, dscale, tscale]
        return map_2d

    def _get_anchors(self, tscale, dscale):
        anchors = []
        for dur_idx in range(dscale):
            for start_idx in range(tscale):
                end_idx = start_idx + dur_idx + 1
                if end_idx <= tscale:
                    center_len = dur_idx + 1
                    sample_xmin = start_idx - center_len * self.prop_extend_ratio
                    sample_xmax = end_idx + center_len * self.prop_extend_ratio
                    anchors.append([sample_xmin, sample_xmax])
                else:
                    anchors.append([0, 0])
        self.anchors = torch.Tensor(anchors).reshape(dscale, tscale, 2)
