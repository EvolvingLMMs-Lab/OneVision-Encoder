import copy
import torch
import torch.nn as nn
from ...builder import TRANSFORMERS
from ..layers import BaseTransformerLayer, MultiheadAttention, FFN


@TRANSFORMERS.register_module()
class DETRDecoder(nn.Module):
    def __init__(
        self,
        embed_dim=256,
        num_heads=8,
        attn_dropout=0.1,
        ffn_dim=2048,
        ffn_dropout=0.1,
        num_layers=6,
        post_norm=True,
        batch_first=True,
        return_intermediate=True,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.ffn_dim = ffn_dim
        self.ffn_dropout = ffn_dropout
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.return_intermediate = return_intermediate

        self._init_transformer_layers()

        if post_norm:
            self.post_norm_layer = nn.LayerNorm(self.embed_dim)
        else:
            self.post_norm_layer = None

    def _init_transformer_layers(self):
        transformer_layers = BaseTransformerLayer(
            attn=MultiheadAttention(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                attn_drop=self.attn_dropout,
                batch_first=self.batch_first,
            ),
            ffn=FFN(
                embed_dim=self.embed_dim,
                ffn_dim=self.ffn_dim,
                ffn_drop=self.ffn_dropout,
            ),
            norm=nn.LayerNorm(
                normalized_shape=self.embed_dim,
            ),
            operation_order=("self_attn", "norm", "cross_attn", "norm", "ffn", "norm"),
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
        if not self.return_intermediate:
            for layer in self.layers:
                query = layer(
                    query,
                    key,
                    value,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    attn_masks=attn_masks,
                    query_key_padding_mask=query_key_padding_mask,
                    key_padding_mask=key_padding_mask,
                    **kwargs,
                )

            if self.post_norm_layer is not None:
                query = self.post_norm_layer(query)[None]
            return query

        # return intermediate
        intermediate = []
        for layer in self.layers:
            query = layer(
                query,
                key,
                value,
                query_pos=query_pos,
                key_pos=key_pos,
                attn_masks=attn_masks,
                query_key_padding_mask=query_key_padding_mask,
                key_padding_mask=key_padding_mask,
                **kwargs,
            )

            if self.return_intermediate:
                if self.post_norm_layer is not None:
                    intermediate.append(self.post_norm_layer(query))
                else:
                    intermediate.append(query)

        return torch.stack(intermediate)
