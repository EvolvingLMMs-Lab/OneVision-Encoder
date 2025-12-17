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
    """
    AIMv2 Packing variant for efficient variable-length sequence processing using FlashAttention.
    
    This model accepts pre-patchified input in packing format:
    - hidden_states: torch.Tensor of shape [total_num_patches, patch_dim]
      where patch_dim = patch_size * patch_size * num_channels
    - grid_thw: torch.Tensor of shape [num_images, 3] containing [t, h, w] for each image
    
    This is optimized for batch processing where all images are concatenated into a single sequence.
    Uses FlashAttention varlen with cu_seqlens for efficient processing without explicit attention masks.
    
    Similar to Siglip2NaflexPacking, this implementation:
    1. Accepts packed patches directly
    2. Uses custom packing layers with FlashAttention varlen
    3. Controls attention through cu_seqlens (no explicit masks)
    """
    
    DEFAULT_PATCH_SIZE = 14

    def __init__(
        self,
        ckpt: str = "/video_vit/pretrain_models/apple/aimv2-large-patch14-native",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        revision: Optional[str] = None,
    ):
        """
        Initialize the AIMv2 Packing model with FlashAttention.
        
        Args:
            ckpt (str): HuggingFace checkpoint path or local path for the pre-trained model.
            device (str): Device to map the model for inference.
            revision (Optional[str]): Model revision to use (only for HuggingFace Hub).
        """
        super().__init__()

        if not _flash_attn_available:
            raise ImportError(
                "FlashAttention 2 is required for AIMv2Packing. Please install flash-attn: "
                "pip install flash-attn --no-build-isolation"
            )

        self.device = torch.device(device)
        
        # Load the pretrained model to get config and weights
        # Note: AIMv2 requires trust_remote_code=True
        from_kwargs = {
            "trust_remote_code": True,
            "dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32
        }
        if revision is not None:
            from_kwargs["revision"] = revision
            
        pretrained_model = Aimv2VisionModel.from_pretrained(ckpt, **from_kwargs)
        self.config = pretrained_model.config
        
        # Get patch size from config
        if hasattr(self.config, 'patch_size'):
            self.patch_size = self.config.patch_size
        else:
            self.patch_size = self.DEFAULT_PATCH_SIZE
        
        # Build custom packing components
        self.embeddings = AIMv2PatchEmbedding(self.config)
        self.encoder = AIMv2PackingEncoder(self.config)
        self.norm = RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
        
        # Load weights from pretrained model
        self._load_pretrained_weights(pretrained_model)
        
        # Move to device and set to eval mode
        self.to(self.device)
        self.eval()
        
        # Clean up pretrained model
        del pretrained_model
    
    def _load_pretrained_weights(self, pretrained_model):
        """
        Load weights from pretrained Aimv2VisionModel into packing layers.
        
        Args:
            pretrained_model: Pretrained Aimv2VisionModel instance
        """
        # Load patch embedding weights
        # The pretrained model might use Conv2d, but we use Linear for packing
        # We need to convert Conv2d weights to Linear weights
        
        # Try to find the patch embedding layer - different models use different attribute names
        patch_emb = None
        possible_names = ['patch_embedding', 'patch_embed', 'projection', 'proj', 'conv']
        
        for name in possible_names:
            if hasattr(pretrained_model.embeddings, name):
                patch_emb = getattr(pretrained_model.embeddings, name)
                break
        
        if patch_emb is None:
            # List available attributes for debugging
            available_attrs = [attr for attr in dir(pretrained_model.embeddings) if not attr.startswith('_')]
            raise AttributeError(
                f"Could not find patch embedding layer in pretrained model embeddings. "
                f"Tried: {possible_names}. "
                f"Available attributes: {available_attrs}"
            )
        
        # If it's a Conv2d, convert to Linear
        if isinstance(patch_emb, nn.Conv2d):
            # Conv2d: [out_channels, in_channels, kernel_h, kernel_w]
            # Our patches are flattened as [patch_h, patch_w, channels] (channels last),
            # so we need to permute Conv2d weights to match this order.
            # Permute [out_ch, in_ch, kh, kw] -> [out_ch, kh, kw, in_ch]
            # Then flatten to [out_ch, kh*kw*in_ch] for Linear layer
            conv_weight = patch_emb.weight.data  # [out_ch, in_ch, kh, kw]
            # Permute to [out_ch, kh, kw, in_ch] to match channel-last flattening
            weight_permuted = conv_weight.permute(0, 2, 3, 1)
            # Flatten to [out_ch, kh*kw*in_ch]
            linear_weight = weight_permuted.reshape(conv_weight.shape[0], -1)
            self.embeddings.proj.weight.data = linear_weight
            if patch_emb.bias is not None:
                self.embeddings.proj.bias.data = patch_emb.bias.data
        elif isinstance(patch_emb, nn.Linear):
            # If it's already Linear, load directly
            self.embeddings.proj.load_state_dict(patch_emb.state_dict())
        else:
            raise TypeError(
                f"Unexpected patch_embedding type: {type(patch_emb)}. "
                f"Expected nn.Conv2d or nn.Linear."
            )
        
        # Load norm weights if present
        if hasattr(pretrained_model.embeddings, 'norm'):
            self.embeddings.norm.load_state_dict(pretrained_model.embeddings.norm.state_dict())
        
        # Load encoder layer weights
        for packing_layer, standard_layer in zip(self.encoder.layers, pretrained_model.encoder.layers):
            # Load norm1
            packing_layer.norm1.load_state_dict(standard_layer.norm1.state_dict())
            
            # Load attention weights (QKV and projection)
            packing_layer.attn.qkv.load_state_dict(standard_layer.attn.qkv.state_dict())
            packing_layer.attn.proj.load_state_dict(standard_layer.attn.proj.state_dict())
            
            # Load norm2
            packing_layer.norm2.load_state_dict(standard_layer.norm2.state_dict())
            
            # Load MLP
            packing_layer.mlp.load_state_dict(standard_layer.mlp.state_dict())
        
        # Load final norm
        self.norm.load_state_dict(pretrained_model.norm.state_dict())

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with pre-patchified input using FlashAttention varlen approach.
        
        Args:
            hidden_states (torch.Tensor): Pre-patchified input of shape 
                [total_num_patches, patch_dim] where 
                patch_dim = patch_size * patch_size * num_channels
            grid_thw (torch.Tensor): Grid dimensions of shape [num_images, 3]
                containing [t, h, w] for each image, where:
                - t: temporal dimension (usually 1 for single images)
                - h: height in patches
                - w: width in patches
        
        Returns:
            torch.Tensor: Last hidden state of shape [total_num_patches, hidden_size]
        """
        # Get target dtype from patch embedding
        target_dtype = self.embeddings.proj.weight.dtype
        
        # Move inputs to device
        hidden_states = hidden_states.to(device=self.device, dtype=target_dtype)
        grid_thw = grid_thw.to(device=self.device)
        
        # Calculate sequence lengths from grid_thw
        # For each image: num_patches = t * h * w
        seq_lengths = grid_thw.prod(dim=1)
        
        # Apply patch embeddings directly to packed patches
        # Note: Unlike Siglip2, we process the packed patches directly without intermediate batching
        # since our Linear layer can handle the [total_patches, patch_dim] input format
        embeddings = self.embeddings(hidden_states)
        
        # Compute cumulative sequence lengths for FlashAttention
        cu_seqlens = F.pad(seq_lengths.cumsum(dim=0), (1, 0), value=0).to(torch.int32)
        
        # Encoder with FlashAttention varlen (no attention mask needed)
        encoder_output = self.encoder(embeddings, cu_seqlens)
        
        # Apply final norm
        last_hidden_state = self.norm(encoder_output)
        
        return last_hidden_state


__all__ = ["AIMv2Packing"]
