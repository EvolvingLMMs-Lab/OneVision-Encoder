# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team.
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
AIMv2 Packing Implementation

This module contains the core components for AIMv2Packing model,
which efficiently processes variable-length sequences using FlashAttention.

Requirements:
1. Must use flash_attn_varlen_func
2. Must use transformers absolute addresses
3. Input must be hidden_states: torch.Tensor, grid_thw: torch.Tensor
4. Position encoding can use for loops, but encoder forward cannot use for loops, must use cu_seqlens
5. Loading model weights must be same as original non-packing model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.aimv2.configuration_aimv2 import Aimv2VisionConfig
from transformers.models.aimv2.modeling_aimv2 import Aimv2VisionModel

# Check if FlashAttention 2 is available for packing model
try:
    from flash_attn import flash_attn_varlen_func
    _flash_attn_available = True
except ImportError:
    _flash_attn_available = False


class Aimv2RMSNorm(nn.Module):
    """
    Aimv2RMSNorm is equivalent to T5LayerNorm
    """
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class Aimv2MLP(nn.Module):
    """
    MLP layer used in Aimv2 transformer blocks.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class Aimv2VisionEmbeddings(nn.Module):
    """
    Vision embeddings for Aimv2 with support for variable image sizes.
    Handles patch embedding and positional encoding.
    """
    def __init__(self, config: Aimv2VisionConfig):
        super().__init__()
        self.config = config
        self.patch_size = config.patch_size
        
        # Patch embedding using Conv2d (for compatibility with pretrained weights)
        self.patch_embed = nn.Conv2d(
            config.num_channels,
            config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size
        )
        self.rms_norm = Aimv2RMSNorm(config.hidden_size, config.rms_norm_eps)
        
        # Position embeddings
        num_patches = (config.image_size // config.patch_size) ** 2
        if not config.is_native:
            self.position_embedding = nn.Embedding(num_patches, config.hidden_size)
            self.register_buffer("position_ids", torch.arange(num_patches).expand((1, -1)), persistent=False)

    @staticmethod
    def build_2d_sincos_position_embedding(
        height, width, embed_dim=256, temperature=10000.0, device="cpu", dtype=torch.float32
    ) -> torch.Tensor:
        """Build 2D sincos position embedding for native models."""
        grid_w = torch.arange(int(width), dtype=dtype, device=device)
        grid_h = torch.arange(int(height), dtype=dtype, device=device)
        grid_h, grid_w = torch.meshgrid(grid_w, grid_h, indexing="xy")

        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=dtype, device=device) / pos_dim
        omega = 1.0 / (temperature**omega)

        out_h = grid_h.flatten()[..., None] @ omega[None, :]
        out_w = grid_w.flatten()[..., None] @ omega[None, :]

        return torch.concat([out_h.sin(), out_h.cos(), out_w.sin(), out_w.cos()], dim=1)[None, :, :]

    def forward(self, hidden_states: torch.Tensor, spatial_shapes: torch.LongTensor) -> torch.Tensor:
        """
        Forward with packing format.
        
        Args:
            hidden_states: Pre-patchified input of shape (batch_size, max_num_patches, patch_dim)
                where patch_dim = patch_size * patch_size * num_channels
            spatial_shapes: Spatial shapes of shape (batch_size, 2) containing [h, w] in patches
        
        Returns:
            Embeddings of shape (batch_size, max_num_patches, hidden_size)
        """
        batch_size, max_num_patches, patch_dim = hidden_states.shape
        
        # Reshape to apply Conv2d: (batch_size * max_num_patches, in_channels, patch_size, patch_size)
        hidden_states = hidden_states.view(
            batch_size * max_num_patches,
            self.config.num_channels,
            self.patch_size,
            self.patch_size
        )
        
        # Apply patch embedding
        target_dtype = self.patch_embed.weight.dtype
        hidden_states = self.patch_embed(hidden_states.to(dtype=target_dtype))
        # Shape: (batch_size * max_num_patches, hidden_size, 1, 1)
        
        # Flatten and reshape back
        hidden_states = hidden_states.flatten(2).transpose(1, 2)  # (batch_size * max_num_patches, 1, hidden_size)
        hidden_states = hidden_states.view(batch_size, max_num_patches, -1)  # (batch_size, max_num_patches, hidden_size)
        
        # Apply RMS norm
        hidden_states = self.rms_norm(hidden_states)
        
        # Add position embeddings (can use for loops as per requirement #4)
        embed_dim = self.config.hidden_size
        source_dtype = hidden_states.dtype
        
        # Position embeddings need to be computed for each image in the batch
        for i in range(batch_size):
            height, width = spatial_shapes[i]
            num_patches = height * width
            
            if self.config.is_native:
                # Build sincos position embedding
                pos_embed = self.build_2d_sincos_position_embedding(
                    height,
                    width,
                    embed_dim=embed_dim,
                    device=hidden_states.device,
                    dtype=source_dtype,
                )
                pos_embed = pos_embed.squeeze(0)  # (num_patches, embed_dim)
            else:
                # Use learned position embeddings
                pos_ids = torch.arange(num_patches, device=hidden_states.device)
                pos_embed = self.position_embedding(pos_ids)  # (num_patches, embed_dim)
            
            # Add position embeddings to this image's patches
            hidden_states[i, :num_patches] = hidden_states[i, :num_patches] + pos_embed
            # Padding positions get the same embedding as the last valid position
            if num_patches < max_num_patches:
                hidden_states[i, num_patches:] = hidden_states[i, num_patches:] + pos_embed[0]
        
        return hidden_states


class Aimv2PackingAttention(nn.Module):
    """
    Multi-headed attention with FlashAttention varlen support for packing.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=config.qkv_bias)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=config.qkv_bias)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=config.qkv_bias)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=config.qkv_bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass with variable-length attention using FlashAttention.

        Args:
            hidden_states: Input of shape (total_seq_len, hidden_size)
            cu_seqlens: Cumulative sequence lengths for flash attention

        Returns:
            Output of shape (total_seq_len, hidden_size)
        """
        seq_length = hidden_states.shape[0]

        queries = self.q_proj(hidden_states)
        keys = self.k_proj(hidden_states)
        values = self.v_proj(hidden_states)

        queries = queries.view(seq_length, self.num_heads, self.head_dim)
        keys = keys.view(seq_length, self.num_heads, self.head_dim)
        values = values.view(seq_length, self.num_heads, self.head_dim)

        # Use FlashAttention with variable lengths (Requirement #1)
        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        attn_output = flash_attn_varlen_func(
            queries,
            keys,
            values,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            dropout_p=self.dropout if self.training else 0.0,
            softmax_scale=self.scale,
            causal=False,
        )

        attn_output = attn_output.reshape(seq_length, self.embed_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output


class Aimv2PackingEncoderLayer(nn.Module):
    """
    Single transformer encoder layer with packing support.
    """
    def __init__(self, config: Aimv2VisionConfig):
        super().__init__()
        self.attention = Aimv2PackingAttention(config)
        self.ffn = Aimv2MLP(config)
        self.rms_norm1 = Aimv2RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rms_norm2 = Aimv2RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> torch.FloatTensor:
        # Pre-norm attention
        norm_hidden_states = self.rms_norm1(hidden_states)
        attn_output = self.attention(
            hidden_states=norm_hidden_states,
            cu_seqlens=cu_seqlens,
        )
        hidden_states = hidden_states + attn_output

        # Pre-norm FFN
        norm_hidden_states = self.rms_norm2(hidden_states)
        mlp_output = self.ffn(norm_hidden_states)
        hidden_states = hidden_states + mlp_output

        return hidden_states


class Aimv2PackingEncoder(nn.Module):
    """
    Transformer encoder with packing support using FlashAttention varlen.
    Requirement #4: No for loops in encoder forward, must use cu_seqlens.
    """
    def __init__(self, config: Aimv2VisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([Aimv2PackingEncoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        inputs_embeds,
        cu_seqlens: torch.Tensor,
    ) -> BaseModelOutput:
        """
        Forward pass using cu_seqlens for efficient variable-length processing.
        No for loops over images - all processing is done in packed format.
        """
        hidden_states = inputs_embeds
        
        # Process all layers without for loops over samples
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(
                hidden_states,
                cu_seqlens,
            )

        return BaseModelOutput(last_hidden_state=hidden_states)


class AIMv2Packing(nn.Module):
    """
    AIMv2 Packing variant for efficient variable-length sequence processing.

    This model accepts pre-patchified input in packing format and uses FlashAttention
    varlen for efficient variable-length processing without attention masks.

    Input format (Requirement #3):
    - hidden_states: torch.Tensor of shape [total_num_patches, patch_dim]
      where patch_dim = patch_size * patch_size * num_channels
    - grid_thw: torch.Tensor of shape [num_images, 3] containing [t, h, w] for each image

    This is optimized for batch processing where all images are concatenated into a single sequence.
    Only vision components are included (no text model).
    """

    DEFAULT_PATCH_SIZE = 14

    def __init__(self, ckpt: str = "apple/aimv2-large-patch14-224", device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the AIMv2 Packing model.

        Args:
            ckpt (str): HuggingFace checkpoint for the pre-trained model.
            device (str): Device to map the model for inference.
        """
        super(AIMv2Packing, self).__init__()

        if not _flash_attn_available:
            raise ImportError(
                "FlashAttention 2 is required for AIMv2Packing. "
                "Please install flash-attn: pip install flash-attn --no-build-isolation"
            )

        self.device = torch.device(device)

        # Requirement #5: Load the vision model from pretrained checkpoint to get config and weights
        # Using absolute import from transformers as per Requirement #2
        vision_model = Aimv2VisionModel.from_pretrained(ckpt, trust_remote_code=True)
        self.config = vision_model.config

        # Get patch size from config
        if hasattr(self.config, 'patch_size'):
            self.patch_size = self.config.patch_size
        else:
            self.patch_size = self.DEFAULT_PATCH_SIZE

        # Build the model components with packing support
        self.embeddings = Aimv2VisionEmbeddings(self.config)
        self.encoder = Aimv2PackingEncoder(self.config)
        self.rms_norm = Aimv2RMSNorm(self.config.hidden_size, self.config.rms_norm_eps)

        # Requirement #5: Load the weights from the pretrained model
        # Copy embeddings weights (patch_embed and position_embedding)
        self.embeddings.patch_embed.load_state_dict(vision_model.embeddings.patch_embed.state_dict())
        self.embeddings.rms_norm.load_state_dict(vision_model.embeddings.rms_norm.state_dict())
        if not self.config.is_native:
            self.embeddings.position_embedding.load_state_dict(vision_model.embeddings.position_embedding.state_dict())

        # Copy encoder weights (need to map standard attention to packing attention)
        for packing_layer, standard_layer in zip(self.encoder.layers, vision_model.encoder.layers):
            # Copy RMS norms
            packing_layer.rms_norm1.load_state_dict(standard_layer.rms_norm1.state_dict())
            packing_layer.rms_norm2.load_state_dict(standard_layer.rms_norm2.state_dict())

            # Copy attention projections
            packing_layer.attention.q_proj.load_state_dict(standard_layer.attention.q_proj.state_dict())
            packing_layer.attention.k_proj.load_state_dict(standard_layer.attention.k_proj.state_dict())
            packing_layer.attention.v_proj.load_state_dict(standard_layer.attention.v_proj.state_dict())
            packing_layer.attention.out_proj.load_state_dict(standard_layer.attention.out_proj.state_dict())

            # Copy MLP (FFN)
            packing_layer.ffn.load_state_dict(standard_layer.ffn.state_dict())

        # Copy post RMS norm
        self.rms_norm.load_state_dict(vision_model.rms_norm.state_dict())

        # Move to device and set to eval mode
        self.to(self.device)
        self.eval()

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor):
        """
        Forward pass with pre-patchified input using FlashAttention varlen approach.
        
        Requirement #3: Input must be hidden_states: torch.Tensor, grid_thw: torch.Tensor

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
        target_dtype = self.embeddings.patch_embed.weight.dtype

        # Move inputs to device
        hidden_states = hidden_states.to(device=self.device, dtype=target_dtype)
        grid_thw = grid_thw.to(device=self.device)

        # Calculate spatial_shapes from grid_thw
        # For Aimv2, spatial_shapes is [num_images, 2] containing [h, w]
        spatial_shapes = grid_thw[:, 1:].long()  # Extract [h, w] from [t, h, w]

        # Reshape hidden_states from [total_patches, patch_dim] to [batch_size, max_patches, patch_dim]
        # This is needed because embeddings.forward expects 3D input
        batch_size = grid_thw.shape[0]
        seq_lengths = grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]
        max_num_patches = seq_lengths.max().item()
        patch_dim = hidden_states.shape[1]

        # Create padded batch tensor
        batched_hidden_states = torch.zeros(
            (batch_size, max_num_patches, patch_dim),
            device=hidden_states.device,
            dtype=hidden_states.dtype
        )

        # Fill in the actual patches for each image
        start_idx = 0
        for i in range(batch_size):
            num_patches = seq_lengths[i].item()
            batched_hidden_states[i, :num_patches] = hidden_states[start_idx:start_idx + num_patches]
            start_idx += num_patches

        # Apply patch embeddings with position encoding
        # Requirement #4: Position encoding can use for loops (done inside embeddings.forward)
        embeddings = self.embeddings(batched_hidden_states, spatial_shapes)

        # Convert back to packed format by removing padding
        # embeddings shape: [batch_size, max_num_patches, hidden_size]
        packed_embeddings = []
        for i in range(batch_size):
            num_patches = seq_lengths[i].item()
            packed_embeddings.append(embeddings[i, :num_patches])
        embeddings = torch.cat(packed_embeddings, dim=0)

        # Requirement #4: Compute cumulative sequence lengths for FlashAttention
        # Encoder forward must use cu_seqlens, no for loops
        cu_seqlens = F.pad(seq_lengths.cumsum(dim=0), (1, 0), value=0).to(torch.int32)

        # Encoder with FlashAttention varlen (no attention mask needed, no for loops)
        encoder_outputs = self.encoder(
            inputs_embeds=embeddings,
            cu_seqlens=cu_seqlens,
        )

        # Post RMS norm
        last_hidden_state = encoder_outputs.last_hidden_state
        last_hidden_state = self.rms_norm(last_hidden_state)

        return last_hidden_state


__all__ = [
    "Aimv2VisionEmbeddings",
    "Aimv2MLP",
    "Aimv2PackingAttention",
    "Aimv2PackingEncoderLayer",
    "Aimv2PackingEncoder",
    "AIMv2Packing",
]
