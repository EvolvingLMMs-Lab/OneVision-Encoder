# coding=utf-8
# Copyright 2025 Apple Inc. and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
AIMv2 packing implementation using FlashAttention (varlen) following the SigLIP2 and
Llava ViT preview packing patterns.

This module consumes packed patches `[total_patches, patch_dim]` and grid_thw metadata
and runs the AIMv2 transformer with FlashAttention varlen kernels (no per-image loops).
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Aimv2VisionModel

try:
    from flash_attn import flash_attn_varlen_func

    _flash_attn_available = True
except ImportError:
    flash_attn_varlen_func = None
    _flash_attn_available = False


def _get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: torch.Tensor) -> torch.Tensor:
    omega = torch.arange(embed_dim // 2, device=pos.device, dtype=pos.dtype)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    pos = pos.reshape(-1)
    out = pos[:, None] * omega[None, :]
    emb_sin, emb_cos = torch.sin(out), torch.cos(out)
    return torch.cat([emb_sin, emb_cos], dim=1)


def get_sincos_pos_embed(h: int, w: int, embed_dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    assert embed_dim % 2 == 0
    grid_h = torch.arange(h, device=device, dtype=dtype)
    grid_w = torch.arange(w, device=device, dtype=dtype)
    grid = torch.meshgrid(grid_w, grid_h, indexing="xy")
    grid = torch.stack(grid, dim=0).reshape(2, 1, h, w)
    x_grid, y_grid = grid  # x_grid: width axis, y_grid: height axis (xy indexing)
    emb_x = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, x_grid)
    emb_y = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, y_grid)
    return torch.cat([emb_x, emb_y], dim=1)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class AIMv2PatchEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        patch_dim = config.num_channels * config.patch_size * config.patch_size
        self.proj = nn.Linear(patch_dim, config.hidden_size, bias=True)
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = self.norm(x)
        return x


class AIMv2SwiGLUFFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_features = config.intermediate_size
        in_features = config.hidden_size
        bias = config.use_bias
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.fc2 = nn.Linear(hidden_features, in_features, bias=bias)
        self.fc3 = nn.Linear(in_features, hidden_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.silu(self.fc1(x)) * self.fc3(x))


class AIMv2PackingAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout
        self.qkv = nn.Linear(self.embed_dim, self.embed_dim * 3, bias=config.qkv_bias)
        self.proj = nn.Linear(self.embed_dim, self.embed_dim, bias=config.use_bias)

    def forward(self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        qkv = self.qkv(hidden_states)
        qkv = qkv.reshape(seq_length, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(1, 0, 2, 3)
        query_states, key_states, value_states = qkv.unbind(0)

        if flash_attn_varlen_func is None:
            raise ImportError(
                "FlashAttention 2 is required for AIMv2Packing. Please install flash-attn: "
                "pip install flash-attn --no-build-isolation"
            )

        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        attn_output = flash_attn_varlen_func(
            query_states,
            key_states,
            value_states,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            dropout_p=self.dropout if self.training else 0.0,
            softmax_scale=self.scale,
            causal=False,
        )

        attn_output = attn_output.reshape(seq_length, self.embed_dim)
        return self.proj(attn_output)


class AIMv2PackingEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn = AIMv2PackingAttention(config)
        self.norm2 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = AIMv2SwiGLUFFN(config)

    def forward(self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states), cu_seqlens)
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class AIMv2PackingEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([AIMv2PackingEncoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states, cu_seqlens)
        return hidden_states


class AIMv2Packing(nn.Module):
    DEFAULT_PATCH_SIZE = 14

    def __init__(
        self,
        ckpt: str = "apple/aimv2-large-patch14-224",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        revision: Optional[str] = None,
    ):
        super().__init__()

        if not _flash_attn_available:
            raise ImportError(
                "FlashAttention 2 is required for AIMv2Packing. Please install flash-attn: "
                "pip install flash-attn --no-build-isolation"
            )

        from_kwargs = {"trust_remote_code": True}
        if revision is not None:
            from_kwargs["revision"] = revision
        base_model = Aimv2VisionModel.from_pretrained(ckpt, **from_kwargs)
        self.config = base_model.config
        self.device = torch.device(device)
        self.patch_size = getattr(self.config, "patch_size", self.DEFAULT_PATCH_SIZE)

        self.patch_embed = AIMv2PatchEmbedding(self.config)
        self.encoder = AIMv2PackingEncoder(self.config)
        self.post_norm = RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)

        # Load weights from the pretrained model
        if not hasattr(base_model, "vision_model"):
            raise AttributeError("Unexpected AIMv2 model structure; missing vision_model on base_model.")
        if not hasattr(base_model.vision_model, "trunk"):
            raise AttributeError(
                f"Unexpected AIMv2 model structure; missing trunk on vision_model (type={type(base_model.vision_model).__name__})."
            )

        patchifier = base_model.vision_model.preprocessor.patchifier
        if (
            patchifier.proj.weight.dim() != 4
            or patchifier.proj.weight.shape[2] != self.patch_size
            or patchifier.proj.weight.shape[3] != self.patch_size
        ):
            raise ValueError(
                f"Unexpected patchifier.proj weight shape for AIMv2 packing conversion. "
                f"Expected (*, *, {self.patch_size}, {self.patch_size}), got {tuple(patchifier.proj.weight.shape)}."
            )
        patch_dim = self.patch_embed.proj.in_features
        self.patch_embed.proj.weight.data.copy_(patchifier.proj.weight.reshape(self.config.hidden_size, patch_dim))
        if patchifier.proj.bias is not None:
            self.patch_embed.proj.bias.data.copy_(patchifier.proj.bias)
        else:
            self.patch_embed.proj.bias.data.zero_()
        self.patch_embed.norm.weight.data.copy_(patchifier.norm.weight)

        standard_layers = base_model.vision_model.trunk.blocks
        if len(self.encoder.layers) != len(standard_layers):
            raise ValueError("Layer count mismatch between packing encoder and base AIMv2 model.")

        # Map standard layer names (norm_1/norm_2) to packing equivalents (norm1/norm2)
        for packing_layer, standard_layer in zip(self.encoder.layers, standard_layers):
            packing_layer.norm1.load_state_dict(standard_layer.norm_1.state_dict())
            packing_layer.norm2.load_state_dict(standard_layer.norm_2.state_dict())
            packing_layer.attn.qkv.load_state_dict(standard_layer.attn.qkv.state_dict())
            packing_layer.attn.proj.load_state_dict(standard_layer.attn.proj.state_dict())
            packing_layer.mlp.load_state_dict(standard_layer.mlp.state_dict())

        self.post_norm.load_state_dict(base_model.vision_model.trunk.post_trunk_norm.state_dict())
        self.to(self.device).eval()

    def _build_position_embeddings(self, grid_thw: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        pos_embeds = []
        for t, h, w in grid_thw:
            t_int, h_int, w_int = int(t.item()), int(h.item()), int(w.item())
            pos = get_sincos_pos_embed(h_int, w_int, self.config.hidden_size, device=self.device, dtype=dtype)
            if t_int > 1:
                pos = pos.unsqueeze(0).expand(t_int, -1, -1).reshape(-1, self.config.hidden_size)
            pos_embeds.append(pos)
        return torch.cat(pos_embeds, dim=0)

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        target_dtype = self.patch_embed.proj.weight.dtype
        hidden_states = hidden_states.to(device=self.device, dtype=target_dtype)
        grid_thw = grid_thw.to(device=self.device)

        t, h, w = grid_thw.unbind(dim=1)
        seq_lengths = (t * h * w).to(torch.int32)
        cu_seqlens = F.pad(seq_lengths.cumsum(dim=0), (1, 0), value=0)

        embeddings = self.patch_embed(hidden_states)
        pos_embeddings = self._build_position_embeddings(grid_thw, embeddings.dtype)
        embeddings = embeddings + pos_embeddings

        hidden_states = self.encoder(embeddings, cu_seqlens)
        hidden_states = self.post_norm(hidden_states)
        return hidden_states


__all__ = ["AIMv2Packing"]
