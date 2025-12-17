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
    Uses FlashAttention for efficient processing without explicit attention masks.
    
    Note: Similar to DINOv3, AIMv2 uses Conv2d for patch embeddings, so this packing implementation
    reconstructs the images from patches before processing.
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
        
        # Load the full model with FlashAttention enabled
        # Note: AIMv2 requires trust_remote_code=True
        from_kwargs = {
            "trust_remote_code": True,
            "attn_implementation": "flash_attention_2",
            "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32
        }
        if revision is not None:
            from_kwargs["revision"] = revision
            
        self.model = Aimv2VisionModel.from_pretrained(ckpt, **from_kwargs).to(self.device).eval()
        self.config = self.model.config
        
        # Get patch size from config
        if hasattr(self.config, 'patch_size'):
            self.patch_size = self.config.patch_size
        else:
            self.patch_size = self.DEFAULT_PATCH_SIZE

    def _reconstruct_images_from_patches(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor):
        """
        Reconstruct images from packed patches.
        
        Args:
            hidden_states (torch.Tensor): Packed patches of shape [total_num_patches, patch_dim]
            grid_thw (torch.Tensor): Grid dimensions of shape [num_images, 3]
        
        Returns:
            torch.Tensor: Reconstructed images of shape [num_images, channels, height, width]
        """
        num_images = grid_thw.shape[0]
        patch_dim = hidden_states.shape[1]
        
        # Infer number of channels from patch_dim
        # patch_dim = patch_size * patch_size * num_channels
        num_channels = patch_dim // (self.patch_size * self.patch_size)
        
        images = []
        start_idx = 0
        
        for i in range(num_images):
            t, h, w = grid_thw[i][0].item(), grid_thw[i][1].item(), grid_thw[i][2].item()
            num_patches = int(t * h * w)
            
            # Extract patches for this image
            image_patches = hidden_states[start_idx:start_idx + num_patches]
            start_idx += num_patches
            
            # Reshape patches to [num_patches_h, num_patches_w, patch_size, patch_size, channels]
            image_patches = image_patches.reshape(
                int(h), int(w), self.patch_size, self.patch_size, num_channels
            )
            
            # Rearrange to [channels, num_patches_h, patch_size, num_patches_w, patch_size]
            image_patches = image_patches.permute(4, 0, 2, 1, 3)
            
            # Reshape to [channels, height, width]
            image = image_patches.reshape(
                num_channels,
                int(h) * self.patch_size,
                int(w) * self.patch_size
            )
            
            images.append(image)
        
        # Stack images: [num_images, channels, height, width]
        return torch.stack(images, dim=0)

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with pre-patchified input using FlashAttention.
        
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
        with torch.no_grad():
            # Get target dtype from model parameters
            try:
                target_dtype = next(self.model.parameters()).dtype
            except (StopIteration, AttributeError):
                # Fallback to bfloat16 or float32 if parameters not accessible
                target_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            
            # Move inputs to device
            hidden_states = hidden_states.to(device=self.device, dtype=target_dtype)
            grid_thw = grid_thw.to(device=self.device)
            
            # Calculate number of patches per image from grid_thw
            num_images = grid_thw.shape[0]
            patches_per_image = []
            image_sizes = []
            for i in range(num_images):
                t, h, w = grid_thw[i][0].item(), grid_thw[i][1].item(), grid_thw[i][2].item()
                num_patches = int(t * h * w)
                patches_per_image.append(num_patches)
                image_sizes.append((int(h) * self.patch_size, int(w) * self.patch_size))
            
            # Check if all images have the same size
            all_same_size = len(set(image_sizes)) == 1
            
            if all_same_size:
                # Optimized path: batch process all images together
                # FlashAttention handles this efficiently without needing explicit masks
                pixel_values = self._reconstruct_images_from_patches(hidden_states, grid_thw)
                
                # Process through model - no attention mask needed with FlashAttention
                outputs = self.model(
                    pixel_values=pixel_values,
                    output_hidden_states=True
                )
                
                # Get the last layer's hidden state: [batch_size, seq_len, hidden_size]
                # Note: Aimv2VisionModel.last_hidden_state already excludes CLS token
                last_hidden_state = outputs.last_hidden_state
                
                # Convert back to packing format: [total_num_patches, hidden_size]
                output_list = []
                for i in range(num_images):
                    num_patches = patches_per_image[i]
                    # AIMv2 last_hidden_state already excludes CLS, so we can use it directly
                    patch_tokens = last_hidden_state[i, :num_patches]
                    output_list.append(patch_tokens)
                
                packed_output = torch.cat(output_list, dim=0)
            else:
                # Variable size path: process each image separately
                # FlashAttention automatically handles variable-length sequences efficiently
                output_list = []
                start_idx = 0
                
                for i in range(num_images):
                    num_patches = patches_per_image[i]
                    
                    # Extract patches for this image
                    image_hidden_states = hidden_states[start_idx:start_idx + num_patches].unsqueeze(0)
                    image_grid_thw = grid_thw[i:i+1]
                    
                    # Reconstruct and process this image
                    pixel_values = self._reconstruct_images_from_patches(
                        image_hidden_states.squeeze(0), image_grid_thw
                    )
                    
                    # Process through model - no attention mask needed
                    outputs = self.model(
                        pixel_values=pixel_values,
                        output_hidden_states=True
                    )
                    
                    # Get the last layer's hidden state
                    # AIMv2 last_hidden_state already excludes CLS token
                    last_hidden_state = outputs.last_hidden_state
                    
                    # Extract patch tokens
                    patch_tokens = last_hidden_state[0, :num_patches]
                    output_list.append(patch_tokens)
                    
                    start_idx += num_patches
                
                packed_output = torch.cat(output_list, dim=0)
        
        return packed_output


__all__ = ["AIMv2Packing"]
