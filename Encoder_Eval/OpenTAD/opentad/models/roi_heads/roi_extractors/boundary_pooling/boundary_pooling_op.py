import torch.nn as nn
from torch.autograd import Function

import boundary_max_pooling_cuda


class BoundaryMaxPoolingFunction(Function):
    @staticmethod
    def forward(ctx, input, segments):
        output = boundary_max_pooling_cuda.forward(input, segments)
        ctx.save_for_backward(input, segments)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()
        input, segments = ctx.saved_tensors
        grad_input = boundary_max_pooling_cuda.backward(grad_output, input, segments)
        return grad_input, None


class BoundaryMaxPooling(nn.Module):
    def __init__(self):
        super(BoundaryMaxPooling, self).__init__()

    def forward(self, input, segments):
        return BoundaryMaxPoolingFunction.apply(input, segments)
