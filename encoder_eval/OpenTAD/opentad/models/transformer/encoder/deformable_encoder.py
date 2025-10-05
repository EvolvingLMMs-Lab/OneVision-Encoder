import copy
import torch.nn as nn
from ...builder import TRANSFORMERS
from ..layers import BaseTransformerLayer, MultiScaleDeformableAttention, FFN


@TRANSFORMERS.register_module()
class DeformableDETREncoder(nn.Module):
    def __init__(
        self,
        embed_dim=256,
        num_heads=8,
        attn_dropout=0.1,
        ffn_dim=2048,
        ffn_dropout=0.1,
        num_layers=6,
        num_points=4,
        num_feature_levels=4,
        post_norm=True,
        batch_first=True,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.ffn_dim = ffn_dim
        self.ffn_dropout = ffn_dropout
        self.num_layers = num_layers
        self.num_points = num_points
        self.num_feature_levels = num_feature_levels
        self.batch_first = batch_first

        self._init_transformer_layers()

        if post_norm:
            self.post_norm_layer = nn.LayerNorm(self.embed_dim)
        else:
            self.post_norm_layer = None

    def _init_transformer_layers(self):
        transformer_layers = BaseTransformerLayer(
            attn=MultiScaleDeformableAttention(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                dropout=self.attn_dropout,
                batch_first=self.batch_first,
                num_points=self.num_points,
                num_levels=self.num_feature_levels,
            ),
            ffn=FFN(
                embed_dim=self.embed_dim,
                ffn_dim=self.ffn_dim,
                ffn_drop=self.ffn_dropout,
            ),
            norm=nn.LayerNorm(
                normalized_shape=self.embed_dim,
            ),
            operation_order=("self_attn", "norm", "ffn", "norm"),
        )
        self.layers = nn.ModuleList([copy.deepcopy(transformer_layers) for _ in range(self.num_layers)])

    def forward(
        self,
        query,
        key,
        value,
        query_pos=None,
        key_pos=None,
        attn_masks=None,
        query_key_padding_mask=None,
        key_padding_mask=None,
        **kwargs,
    ):
        for layer in self.layers:
            query = layer(
                query,
                key,
                value,
                query_pos=query_pos,
                attn_masks=attn_masks,
                query_key_padding_mask=query_key_padding_mask,
                key_padding_mask=key_padding_mask,
                **kwargs,
            )

        if self.post_norm_layer is not None:
            query = self.post_norm_layer(query)
        return query
