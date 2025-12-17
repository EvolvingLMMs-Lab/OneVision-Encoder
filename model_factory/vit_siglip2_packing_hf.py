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
Siglip2 Naflex Packing Implementation

This module contains the core components for Siglip2NaflexPacking model,
which efficiently processes variable-length sequences using FlashAttention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.siglip2.configuration_siglip2 import Siglip2VisionConfig
from transformers.models.siglip2.modeling_siglip2 import Siglip2PreTrainedModel

# Check if FlashAttention 2 is available for packing model
try:
    from flash_attn import flash_attn_varlen_func
    _flash_attn_available = True
except ImportError:
    _flash_attn_available = False


class Siglip2VisionEmbeddings(nn.Module):
    """
    Vision embeddings for Siglip2 with support for variable image sizes.
    Handles patch embedding and positional encoding.
    """

    def __init__(self, config: Siglip2VisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Linear(
            in_features=config.num_channels * self.patch_size * self.patch_size,
            out_features=self.embed_dim,
        )

        self.num_patches = config.num_patches
        self.position_embedding_size = int(self.num_patches**0.5)
        self.position_embedding = nn.Embedding(self.num_patches, self.embed_dim)

    @staticmethod
    def resize_positional_embeddings(
        positional_embeddings: torch.Tensor,
        spatial_shapes: torch.LongTensor,
        max_length: int,
    ) -> torch.Tensor:
        """
        Resize positional embeddings to image-specific size and pad to a fixed size.

        Args:
            positional_embeddings (`torch.Tensor`):
                Position embeddings of shape (height, width, embed_dim)
            spatial_shapes (`torch.LongTensor`):
                Spatial shapes of shape (batch_size, 2) to resize the positional embeddings to
            max_length (`int`):
                Maximum length of the positional embeddings to pad resized positional embeddings to

        Returns:
            `torch.Tensor`: Embeddings of shape (batch_size, max_length, embed_dim)
        """
        batch_size = spatial_shapes.shape[0]
        embed_dim = positional_embeddings.shape[-1]
        source_dtype = positional_embeddings.dtype

        resulted_positional_embeddings = torch.empty(
            (batch_size, max_length, embed_dim),
            device=positional_embeddings.device,
            dtype=source_dtype,
        )

        # (height, width, embed_dim) -> (1, embed_dim, height, width) for interpolation
        positional_embeddings = positional_embeddings.permute(2, 0, 1).unsqueeze(0)

        # Upcast to float32 on CPU because antialias is not supported for bfloat16/float16 on CPU
        if positional_embeddings.device.type == "cpu":
            positional_embeddings = positional_embeddings.to(torch.float32)

        for i in range(batch_size):
            # (1, dim, height, width) -> (1, dim, target_height, target_width)
            height, width = spatial_shapes[i]
            resized_embeddings = F.interpolate(
                positional_embeddings,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )

            # (1, dim, target_height, target_width) -> (target_height * target_width, dim)
            resized_embeddings = resized_embeddings.reshape(embed_dim, height * width).transpose(0, 1)

            # Cast to original dtype
            resized_embeddings = resized_embeddings.to(source_dtype)

            resulted_positional_embeddings[i, : height * width] = resized_embeddings
            resulted_positional_embeddings[i, height * width :] = resized_embeddings[0]

        return resulted_positional_embeddings

    def forward(self, pixel_values: torch.FloatTensor, spatial_shapes: torch.LongTensor) -> torch.Tensor:
        """
        Args:
            pixel_values (`torch.FloatTensor`):
                Pixel values of shape (batch_size, max_num_patches, num_channels * patch_size * patch_size)
            spatial_shapes (`list[tuple[int, int]]`):
                Spatial shapes of shape (batch_size, 2) to resize the positional embeddings to
        """

        # Apply patch embeddings to already patchified pixel values
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))

        # Get positional resized and padded positional embeddings
        positional_embeddings = self.position_embedding.weight.reshape(
            self.position_embedding_size, self.position_embedding_size, -1
        )
        resized_positional_embeddings = self.resize_positional_embeddings(
            positional_embeddings, spatial_shapes, max_length=pixel_values.shape[1]
        )

        # Add positional embeddings to patch embeddings
        embeddings = patch_embeds + resized_positional_embeddings
        return embeddings


class Siglip2MLP(nn.Module):
    """
    MLP layer used in Siglip2 transformer blocks.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class Siglip2PackingAttention(nn.Module):
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

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

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


class Siglip2PackingEncoderLayer(nn.Module):
    """
    Single transformer encoder layer with packing support.
    """

    def __init__(self, config: Siglip2VisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.self_attn = Siglip2PackingAttention(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = Siglip2MLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> torch.FloatTensor:
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            cu_seqlens=cu_seqlens,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class Siglip2PackingEncoder(nn.Module):
    """
    Transformer encoder with packing support using FlashAttention varlen.
    """

    def __init__(self, config: Siglip2VisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([Siglip2PackingEncoderLayer(config) for _ in range(config.num_hidden_layers)])

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


class Siglip2PackingVisionModel(nn.Module):
    """
    Vision transformer with packing support. Structured to mirror Siglip2VisionModel
    so that pretrained weights can be loaded directly via matching state_dict keys.
    Expects packed embeddings together with cumulative sequence lengths for FlashAttention varlen.
    """

    def __init__(self, config: Siglip2VisionConfig):
        super().__init__()
        self.config = config
        self.embeddings = Siglip2VisionEmbeddings(config)
        self.encoder = Siglip2PackingEncoder(config)
        self.post_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, inputs_embeds: torch.Tensor, cu_seqlens: torch.Tensor) -> BaseModelOutput:
        encoder_outputs = self.encoder(inputs_embeds=inputs_embeds, cu_seqlens=cu_seqlens)
        hidden_states = self.post_layernorm(encoder_outputs.last_hidden_state)
        return BaseModelOutput(last_hidden_state=hidden_states)

    @property
    def patch_embed_weight(self) -> torch.nn.Parameter:
        return self.embeddings.patch_embedding.weight

    def embed(self, hidden_states: torch.Tensor, spatial_shapes: torch.Tensor) -> torch.Tensor:
        return self.embeddings(hidden_states, spatial_shapes)


class Siglip2NaflexPacking(Siglip2PreTrainedModel):
    """
    Siglip2 Naflex Packing variant for efficient variable-length sequence processing.

    This model accepts pre-patchified input in packing format and uses FlashAttention
    varlen for efficient variable-length processing without attention masks.

    - hidden_states: torch.Tensor of shape [total_num_patches, patch_dim]
      where patch_dim = patch_size * patch_size * num_channels
    - grid_thw: torch.Tensor of shape [num_images, 3] containing [t, h, w] for each image

    This is optimized for batch processing where all images are concatenated into a single sequence.
    Only vision components are included (no text model).
    Inherits from Siglip2PreTrainedModel and can be loaded directly from Siglip2 checkpoints via `from_pretrained`.
    """

    def __init__(self, config: Siglip2VisionConfig):
        """
        Initialize the Siglip2 Naflex Packing model.

        Args:
            config: Siglip2VisionConfig configuration object.
        """
        super().__init__(config)

        if not _flash_attn_available:
            raise ImportError(
                "FlashAttention 2 is required for Siglip2NaflexPacking. "
                "Please install flash-attn: pip install flash-attn --no-build-isolation"
            )

        self.vision_model = Siglip2PackingVisionModel(config)

        # Initialize weights and apply final processing
        self.post_init()

    @classmethod
    def from_checkpoint(cls, ckpt: str, **kwargs):
        """
        Backward-compatible helper to load from an existing checkpoint string using
        the standard ``from_pretrained`` mechanism.

        Args:
            ckpt: Pretrained Siglip2 checkpoint identifier or path.
            **kwargs: Additional keyword arguments forwarded to ``from_pretrained``.

        Returns:
            Siglip2NaflexPacking: Model initialized with weights from the checkpoint.
        """
        return cls.from_pretrained(ckpt, **kwargs)

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor):
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
        patch_weight = self.vision_model.patch_embed_weight
        target_dtype = patch_weight.dtype
        device = patch_weight.device

        # Move inputs to device
        hidden_states = hidden_states.to(device=device, dtype=target_dtype)
        grid_thw = grid_thw.to(device=device)

        # Calculate spatial_shapes from grid_thw
        # For Siglip2, spatial_shapes is [num_images, 2] containing [h, w]
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

        # Apply patch embeddings
        embeddings = self.vision_model.embed(batched_hidden_states, spatial_shapes)

        # Convert back to packed format by removing padding
        # embeddings shape: [batch_size, max_num_patches, hidden_size]
        packed_embeddings = []
        for i in range(batch_size):
            num_patches = seq_lengths[i].item()
            packed_embeddings.append(embeddings[i, :num_patches])
        embeddings = torch.cat(packed_embeddings, dim=0)

        # Compute cumulative sequence lengths for FlashAttention
        cu_seqlens = F.pad(seq_lengths.cumsum(dim=0), (1, 0), value=0).to(torch.int32)

        # Encoder with FlashAttention varlen (no attention mask needed)
        encoder_outputs = self.vision_model(embeddings, cu_seqlens)

        return encoder_outputs.last_hidden_state


__all__ = [
    "Siglip2VisionEmbeddings",
    "Siglip2MLP",
    "Siglip2PackingAttention",
    "Siglip2PackingEncoderLayer",
    "Siglip2PackingEncoder",
    "Siglip2PackingVisionModel",
    "Siglip2NaflexPacking",
]
