import torch
import torch.nn as nn
from ..builder import MODELS


# dynamic graph from knn
def knn(x, y=None, k=10):
    if y is None:
        y = x
    dif = torch.sum((x.unsqueeze(2) - y.unsqueeze(3)) ** 2, dim=1)
    idx_org = dif.topk(k=k, dim=-1, largest=False)[1]

    return idx_org


def get_neigh_idx_semantic(x, n_neigh):
    B, _, num_prop_v = x.shape
    neigh_idx = knn(x, k=n_neigh).to(dtype=torch.float32)
    shift = torch.tensor(range(B), dtype=torch.float32, device=x.device) * num_prop_v
    shift = shift[:, None, None].repeat(1, num_prop_v, n_neigh)
    neigh_idx = (neigh_idx + shift).view(-1)
    return neigh_idx


class NeighConv(nn.Module):
    def __init__(self, feat_channels, num_neigh, nfeat_mode, agg_type, edge_weight):
        super(NeighConv, self).__init__()
        self.num_neigh = num_neigh
        self.nfeat_mode = nfeat_mode
        self.agg_type = agg_type
        self.edge_weight = edge_weight

        self.mlp = nn.Linear(feat_channels * 2, feat_channels)

    def forward(self, x):
        bs, C, num_frm = x.shape
        neigh_idx = get_neigh_idx_semantic(x, self.num_neigh)

        feat_prop = x.permute(0, 2, 1).reshape(-1, C)
        feat_neigh = feat_prop[neigh_idx.to(torch.long)]
        f_neigh_temp = feat_neigh.view(-1, self.num_neigh, feat_neigh.shape[-1])

        if self.nfeat_mode == "feat_ctr":
            feat_neigh = torch.cat(
                (
                    feat_neigh.view(-1, self.num_neigh, feat_prop.size(-1)),
                    feat_prop.view(-1, 1, feat_prop.size(-1)).repeat(1, self.num_neigh, 1),
                ),
                dim=-1,
            )
        elif self.nfeat_mode == "dif_ctr":
            feat_prop = feat_prop.view(-1, 1, feat_prop.size(-1)).repeat(1, self.num_neigh, 1)
            diff = feat_neigh.view(-1, self.num_neigh, feat_prop.size(-1)) - feat_prop
            feat_neigh = torch.cat((diff, feat_prop), dim=-1)
        elif self.nfeat_mode == "feat":
            feat_neigh = feat_neigh.view(-1, self.num_neigh, feat_prop.size(-1))

        feat_neigh_out = self.mlp(feat_neigh)
        if self.edge_weight == "true":
            weight = torch.matmul(f_neigh_temp, feat_prop.unsqueeze(2))
            weight_denom1 = torch.sqrt(torch.sum(f_neigh_temp * f_neigh_temp, dim=2, keepdim=True))
            weight_denom2 = torch.sqrt(torch.sum(feat_prop.unsqueeze(2) * feat_prop.unsqueeze(2), dim=1, keepdim=True))
            weight = (weight / torch.matmul(weight_denom1, weight_denom2)).squeeze(2)
            feat_neigh_out = feat_neigh_out * weight.unsqueeze(2)

        if self.agg_type == "max":
            feat_neigh_out = feat_neigh_out.max(dim=1, keepdim=False)[0]
        elif self.agg_type == "mean":
            feat_neigh_out = feat_neigh_out.mean(dim=1, keepdim=False)
        return feat_neigh_out.view(bs, num_frm, -1).permute(0, 2, 1)


class xGN(nn.Module):
    def __init__(
        self,
        feat_channels,
        stride,
        gcn_kwargs=dict(num_neigh=10, nfeat_mode="feat_ctr", agg_type="max", edge_weight="false"),
    ):
        super(xGN, self).__init__()

        self.tconv1 = nn.Conv1d(
            in_channels=feat_channels,
            out_channels=feat_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=1,
        )

        self.nconv1 = NeighConv(feat_channels, **gcn_kwargs)

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=stride, padding=1)

    def forward(self, x, mask=None):
        """
        Args:
            x: Feature from previous layer, tensor size (B, C, T)
        Returns:
            output: tensor size (B, C, T)
        """

        # CNN
        c_out = self.tconv1(x)

        # GCN
        g_out = self.nconv1(x)

        out = self.relu(c_out + g_out)
        out = self.maxpool(out)

        if mask is not None:
            mask = self.maxpool(mask.float()).bool()
            out = out * mask.unsqueeze(1).to(out.dtype).detach()
            return out, mask
        else:
            return out
