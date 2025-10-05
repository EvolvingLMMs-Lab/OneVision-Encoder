# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
import Align1D as _align_1d


class _Align1D(Function):
    @staticmethod
    def forward(ctx, input, roi, feature_dim, ratio):
        ctx.save_for_backward(roi)
        ctx.feature_dim = feature_dim
        ctx.input_shape = input.size()
        ctx.sampling_ratio = ratio
        output = _align_1d.forward(input, roi, feature_dim, ratio)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        (rois,) = ctx.saved_tensors
        feature_dim = ctx.feature_dim
        bs, ch, t = ctx.input_shape
        ratio = ctx.sampling_ratio
        grad_input = _align_1d.backward(grad_output, rois, feature_dim, bs, ch, t, ratio)
        return grad_input, None, None, None, None


align1d = _Align1D.apply


class Align1DLayer(nn.Module):
    def __init__(self, feature_dim, ratio=0):
        super(Align1DLayer, self).__init__()
        self.feature_dim = feature_dim
        self.ratio = ratio

    def forward(self, input, rois):
        # print('- input shape is', input.shape)
        # print('- input mean is', input.mean())
        # print('- rois shape is', rois.shape)
        # print('- rois is on', rois.get_device())
        assert input.device == rois.device, (
            "Align operation requires "
            + "both feature and roi are on the same device! "
            + "Get feature on {} but roi on {}".format(input.device, rois.device)
        )

        # out = align1d(input, rois, self.feature_dim, self.ratio)

        # this is a trick for deterministic implementation
        # see https://github.com/open-mmlab/mmdetection/issues/9831
        out = align1d(
            input.half().double(),
            rois.half().double(),
            self.feature_dim,
            self.ratio,
        ).to(input.dtype)

        # print('- output shape is', out.shape)
        # print('- output mean is', out.mean())
        return out

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += f"feature_dim={str(self.feature_dim)}, "
        tmpstr += f"sampling_ratio={str(self.ratio)})"
        return tmpstr


if __name__ == "__main__":
    layer = Align1DLayer(3, ratio=6)
    # layer = torch.nn.DataParallel(layer, device_ids=[0,1])
    # input = torch.tensor([[[1.,2,3,4,5,6,7,8,9,10],[11,12,13,14,15,16,17,18,19,20]]]).cuda()
    input = torch.tensor(
        [
            [
                [0.0, 1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
            ]
        ]
    ).cuda()
    # proposal = torch.tensor([[0,-0.5,9.5],[0,0.1,0.9]]).cuda()
    proposal = torch.tensor([[0, 1.5, 19.5]]).cuda()
    output = layer(input, proposal)
    # print("output has shape {}, with mean {}".format(output.shape, torch.mean(output)))
    torch.set_printoptions(linewidth=180)
    print(output)
