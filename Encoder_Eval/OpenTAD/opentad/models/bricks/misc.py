import torch
import torch.nn as nn


class Scale(nn.Module):
    """
    Multiply the output regression range by a learnable constant value
    """

    def __init__(self, init_value=1.0):
        """
        init_value : initial value for the scalar
        """
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(init_value, dtype=torch.float32), requires_grad=True)

    def forward(self, x):
        """
        input -> scale * input
        """
        return x * self.scale
