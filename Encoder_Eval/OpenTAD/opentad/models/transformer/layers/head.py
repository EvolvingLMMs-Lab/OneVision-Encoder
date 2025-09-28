import math
import torch
import torch.nn as nn
from .mlp import MLP
from ...builder import TRANSFORMERS


@TRANSFORMERS.register_module()
class SharedHead(nn.Module):
    def __init__(self, embed_dim, num_classes, num_layers=3):
        super().__init__()

        # define classification head and box head
        self.class_embed = nn.Linear(
            embed_dim, num_classes
        )  # here we do not use +1, since we are using sigmoid focal loss
        self.bbox_embed = MLP(input_dim=embed_dim, hidden_dim=embed_dim, output_dim=2, num_layers=num_layers)

    def forward(self, hidden_states):
        outputs_class = self.class_embed(hidden_states)
        outputs_coord = self.bbox_embed(hidden_states).sigmoid()
        return outputs_class, outputs_coord
