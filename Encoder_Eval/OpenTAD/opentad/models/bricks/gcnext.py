# copy from https://github.dev/frostinassiky/gtad/blob/3f145d5d3a8ce7ac8d2b985934dbb575ca1b7981/gtad_lib/models.py
import torch
import torch.nn as nn
import numpy as np
from ..builder import MODELS


@MODELS.register_module()
class GCNeXt(nn.Module):
    def __init__(self, in_channels, out_channels, k=3, groups=32):
        super().__init__()

        width_group = 4
        self.k = k
        self.groups = groups

        width = width_group * groups
        # temporal graph
        self.tconvs = nn.Sequential(
            nn.Conv1d(in_channels, width, kernel_size=1),
            nn.ReLU(True),
            nn.Conv1d(width, width, kernel_size=3, groups=groups, padding=1),
            nn.ReLU(True),
            nn.Conv1d(width, out_channels, kernel_size=1),
        )
        # semantic graph
        self.sconvs = nn.Sequential(
            nn.Conv2d(in_channels * 2, width, kernel_size=1),
            nn.ReLU(True),
            nn.Conv2d(width, width, kernel_size=1, groups=groups),
            nn.ReLU(True),
            nn.Conv2d(width, out_channels, kernel_size=1),
        )

        self.relu = nn.ReLU(True)

    def forward(self, x, masks=None):
        identity = x  # residual
        tout = self.tconvs(x)  # conv on temporal graph

        x_f = get_graph_feature(x, k=self.k, style=1)  # (bs,ch,100) -> (bs, 2ch, 100, k)
        sout = self.sconvs(x_f)  # conv on semantic graph
        sout = sout.max(dim=-1, keepdim=False)[0]  # (bs, ch, 100, k) -> (bs, ch, 100)

        out = tout + identity + sout  # fusion
        if masks != None:
            return self.relu(out) * masks.unsqueeze(1).float().detach(), masks
        else:
            return self.relu(out)


# dynamic graph from knn
def knn(x, y=None, k=10):
    """
    :param x: BxCxN
    :param y: BxCxM
    :param k: scalar
    :return: BxMxk
    """
    if y is None:
        y = x
    inner = -2 * torch.matmul(y.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    yy = torch.sum(y**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - yy.transpose(2, 1)
    _, idx = pairwise_distance.topk(k=k, dim=-1)  # (batch_size, num_points, k)
    return idx


# get graph feature
def get_graph_feature(x, prev_x=None, k=20, idx_knn=None, r=-1, style=0):
    """
    :param x:
    :param prev_x:
    :param k:
    :param idx:
    :param r: output downsampling factor (-1 for no downsampling)
    :param style: method to get graph feature
    :return:
    """
    batch_size = x.size(0)
    num_points = x.size(2)  # if prev_x is None else prev_x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx_knn is None:
        idx_knn = knn(x=x, y=prev_x, k=k)  # (batch_size, num_points, k)
    else:
        k = idx_knn.shape[-1]
    # print(idx_knn.shape)
    device = x.device  # torch.device('cuda')
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx_knn + idx_base
    idx = idx.view(-1)
    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    if style == 0:  # use offset as feature
        feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2)
    elif style == 1:  # use feature as feature
        feature = torch.cat((feature, x), dim=3).permute(0, 3, 1, 2)
    else:  # style == 2:
        feature = feature.permute(0, 3, 1, 2)
    # downsample if needed
    if r != -1:
        select_idx = torch.from_numpy(np.random.choice(feature.size(2), feature.size(2) // r, replace=False))
        feature = feature[:, :, select_idx.to(device=device), :]
    return feature.contiguous()
