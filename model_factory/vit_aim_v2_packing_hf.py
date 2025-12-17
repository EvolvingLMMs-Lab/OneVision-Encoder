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

This module provides a packing implementation for AIMv2 models that:
1. Uses flash_attn_varlen_func for efficient variable-length processing
2. Uses transformers absolute addresses
3. Accepts input as (hidden_states: torch.Tensor, grid_thw: torch.Tensor)
4. Processes all images in ONE forward pass (no single-image processing)
5. Loads weights from original non-packing Aimv2VisionModel

Following the Siglip2 packing implementation pattern with FlashAttention varlen.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.aimv2.configuration_aimv2 import Aimv2VisionConfig
from transformers.models.aimv2.modeling_aimv2 import Aimv2VisionModel

# Check if FlashAttention 2 is available
try:
    from flash_attn import flash_attn_varlen_func
    _flash_attn_available = True
except ImportError:
    _flash_attn_available = False


class Aimv2VisionEmbeddings(nn.Module):
    """
    Vision embeddings for AIMv2 with support for variable image sizes.
    Handles patch embedding using Conv2d projection with position embeddings.
    """

    def __init__(self, config: Aimv2VisionConfig):
        super().__init__()
        self.config = config
        self.patch_size = config.patch_size
        
        # AIMv2 uses Conv2d for patch embedding
        self.patch_embed = nn.Conv2d(
            config.num_channels, config.hidden_size, kernel_size=config.patch_size, stride=config.patch_size
        )
        self.rms_norm = Aimv2RMSNorm(config.hidden_size, config.rms_norm_eps)

        num_patches = (config.image_size // config.patch_size) ** 2
        if not self.config.is_native:
            self.position_embedding = nn.Embedding(num_patches, config.hidden_size)
        self.register_buffer("position_ids", torch.arange(num_patches).expand((1, -1)), persistent=False)

    @staticmethod
    def build_2d_sincos_position_embedding(
        height, width, embed_dim=256, temperature=10000.0, device="cpu", dtype=torch.float32
    ) -> torch.Tensor:
        """
        Build 2D sinusoidal position embeddings.
        
        Args:
            height: Height in patches
            width: Width in patches
            embed_dim: Embedding dimension
            temperature: Temperature for sinusoidal encoding
            device: Device for tensor creation
            dtype: Data type for tensor creation
            
        Returns:
            Position embeddings of shape [1, height*width, embed_dim]
        """
        grid_w = torch.arange(int(width), dtype=dtype, device=device)
        grid_h = torch.arange(int(height), dtype=dtype, device=device)
        grid_h, grid_w = torch.meshgrid(grid_w, grid_h, indexing="xy")

        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=dtype, device=device) / pos_dim
        omega = 1.0 / (temperature**omega)

        out_h = grid_h.flatten()[..., None] @ omega[None, :]
        out_w = grid_w.flatten()[..., None] @ omega[None, :]

        return torch.concat([out_h.sin(), out_h.cos(), out_w.sin(), out_w.cos()], dim=1)[None, :, :]

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        """
        Args:
            pixel_values (`torch.FloatTensor`):
                Pixel values of shape (batch_size, num_channels, height, width)
        """
        _, _, height, width = pixel_values.size()
        
        # Apply patch embeddings via Conv2d
        hidden_states = self.patch_embed(pixel_values).flatten(2).transpose(1, 2)
        hidden_states = self.rms_norm(hidden_states)

        # Add position embeddings
        if self.config.is_native:
            pos_embed = self.build_2d_sincos_position_embedding(
                height // self.patch_size,
                width // self.patch_size,
                embed_dim=self.config.hidden_size,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
        else:
            pos_embed = self.position_embedding(self.position_ids)

        hidden_states = hidden_states + pos_embed
        return hidden_states


class Aimv2RMSNorm(nn.Module):
    """
    RMSNorm normalization layer used in AIMv2.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class Aimv2MLP(nn.Module):
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
        self.dropout = getattr(config, 'attention_dropout', 0.0)

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

        # Use FlashAttention with variable lengths
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
    Matches the structure of Aimv2EncoderLayer with RMSNorm.
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
        # Pre-norm architecture with residual connections
        norm_hidden_states = self.rms_norm1(hidden_states)
        attn_output = self.attention(
            hidden_states=norm_hidden_states,
            cu_seqlens=cu_seqlens,
        )
        hidden_states = hidden_states + attn_output

        norm_hidden_states = self.rms_norm2(hidden_states)
        mlp_output = self.ffn(norm_hidden_states)
        hidden_states = hidden_states + mlp_output

        return hidden_states


class Aimv2PackingEncoder(nn.Module):
    """
    Transformer encoder with packing support using FlashAttention varlen.
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
        hidden_states = inputs_embeds
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

    - hidden_states: torch.Tensor of shape [total_num_patches, patch_dim]
      where patch_dim = patch_size * patch_size * num_channels
    - grid_thw: torch.Tensor of shape [num_images, 3] containing [t, h, w] for each image

    This is optimized for batch processing where all images are concatenated into a single sequence.
    Processes ALL images in ONE forward pass (no single-image processing).
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

        # Load the vision model from pretrained checkpoint to get config and weights
        try:
            base_model = Aimv2VisionModel.from_pretrained(ckpt, trust_remote_code=True)
        except Exception as e:
            raise RuntimeError(f"Failed to load Aimv2VisionModel from {ckpt}: {e}")
        
        self.config = base_model.config

        # Get patch size from config
        if hasattr(self.config, 'patch_size'):
            self.patch_size = self.config.patch_size
        else:
            self.patch_size = self.DEFAULT_PATCH_SIZE

        # Build the model components with packing support
        self.embeddings = Aimv2VisionEmbeddings(self.config)
        self.encoder = Aimv2PackingEncoder(self.config)
        # AIMv2 uses RMSNorm for the final normalization
        self.layernorm = Aimv2RMSNorm(self.config.hidden_size, self.config.rms_norm_eps)

        # Load the weights from the pretrained model
        # Aimv2VisionModel structure: the model itself has embeddings, encoder, layernorm
        try:
            # Copy embeddings weights - use the entire embeddings state dict
            # This includes patch_embed, rms_norm, and position_embedding (if not native)
            self.embeddings.load_state_dict(base_model.embeddings.state_dict())
        except AttributeError as e:
            raise RuntimeError(
                f"Failed to copy embeddings weights. "
                f"base_model attributes: {dir(base_model)}. "
                f"Error: {e}"
            )

        # Copy encoder weights (need to map standard attention to packing attention)
        try:
            for packing_layer, standard_layer in zip(self.encoder.layers, base_model.encoder.layers):
                # Copy RMS norms (rms_norm1, rms_norm2)
                packing_layer.rms_norm1.load_state_dict(standard_layer.rms_norm1.state_dict())
                packing_layer.rms_norm2.load_state_dict(standard_layer.rms_norm2.state_dict())

                # Copy attention projections (standard uses 'attention', we use 'attention' too now)
                packing_layer.attention.q_proj.load_state_dict(standard_layer.attention.q_proj.state_dict())
                packing_layer.attention.k_proj.load_state_dict(standard_layer.attention.k_proj.state_dict())
                packing_layer.attention.v_proj.load_state_dict(standard_layer.attention.v_proj.state_dict())
                packing_layer.attention.out_proj.load_state_dict(standard_layer.attention.out_proj.state_dict())

                # Copy MLP (standard uses 'ffn', we use 'ffn' too now)
                packing_layer.ffn.load_state_dict(standard_layer.ffn.state_dict())
        except AttributeError as e:
            raise RuntimeError(
                f"Failed to copy encoder weights. "
                f"base_model.encoder layer attributes: {dir(base_model.encoder.layers[0]) if hasattr(base_model, 'encoder') else 'NO ENCODER'}. "
                f"Error: {e}"
            )

        # Copy final layernorm
        try:
            self.layernorm.load_state_dict(base_model.layernorm.state_dict())
        except AttributeError as e:
            raise RuntimeError(
                f"Failed to copy layernorm weights. "
                f"base_model attributes: {dir(base_model)}. "
                f"Error: {e}"
            )

        # Move to device and set to eval mode
        self.to(self.device)
        self.eval()

    def _reconstruct_images_from_patches(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor):
        """
        Reconstruct images from packed patches for Conv2d patch embedding.
        
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

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor):
        """
        Forward pass with pre-patchified input using FlashAttention varlen approach.
        Processes ALL images in ONE forward pass.

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

        # Reconstruct images from patches for Conv2d embedding
        # This is necessary because AIMv2 uses Conv2d for patch projection
        pixel_values = self._reconstruct_images_from_patches(hidden_states, grid_thw)

        # Process embeddings for each image separately to handle variable position embeddings
        # Each image may have different dimensions, requiring different position embeddings
        batch_size = grid_thw.shape[0]
        seq_lengths = grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]
        
        packed_embeddings = []
        for i in range(batch_size):
            # Get single image
            single_pixel_values = pixel_values[i:i+1]
            _, _, height, width = single_pixel_values.size()
            
            # Apply patch embedding and RMS norm
            patch_embeds = self.embeddings.patch_embed(single_pixel_values).flatten(2).transpose(1, 2)
            patch_embeds = self.embeddings.rms_norm(patch_embeds)
            
            # Add position embeddings based on this image's specific dimensions
            if self.config.is_native:
                # Build sincos position embeddings for this specific image size
                pos_embed = self.embeddings.build_2d_sincos_position_embedding(
                    height // self.patch_size,
                    width // self.patch_size,
                    embed_dim=self.config.hidden_size,
                    device=patch_embeds.device,
                    dtype=patch_embeds.dtype,
                )
            else:
                # Use learned position embeddings
                pos_embed = self.embeddings.position_embedding(self.embeddings.position_ids)
            
            # Add position embeddings
            embeddings_with_pos = patch_embeds + pos_embed
            
            # Extract only the actual patches (seq_lengths[i])
            num_patches = seq_lengths[i].item()
            packed_embeddings.append(embeddings_with_pos[0, :num_patches])
        
        # Concatenate all embeddings into packed format
        embeddings = torch.cat(packed_embeddings, dim=0)

        # Compute cumulative sequence lengths for FlashAttention
        cu_seqlens = F.pad(seq_lengths.cumsum(dim=0), (1, 0), value=0).to(torch.int32)

        # Encoder with FlashAttention varlen (no attention mask needed)
        encoder_outputs = self.encoder(
            inputs_embeds=embeddings,
            cu_seqlens=cu_seqlens,
        )

        # Final layernorm
        last_hidden_state = encoder_outputs.last_hidden_state
        last_hidden_state = self.layernorm(last_hidden_state)

        return last_hidden_state


__all__ = [
    "Aimv2RMSNorm",
    "Aimv2VisionEmbeddings",
    "Aimv2MLP",
    "Aimv2PackingAttention",
    "Aimv2PackingEncoderLayer",
    "Aimv2PackingEncoder",
    "AIMv2Packing",
]
