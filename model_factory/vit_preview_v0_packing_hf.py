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
"""PyTorch Llava ViT Packing model with grid_thw support (similar to Qwen2VL).

This model requires FlashAttention 2 and accepts input in [seq_len, patch_dim] format
where patch_dim = patch_size * patch_size * in_channels,
like Qwen2VL for efficient variable-length sequence processing.
"""

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.registry import register_model
from transformers.activations import ACT2FN
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from transformers.modeling_utils import PreTrainedModel
from transformers.models.siglip.modeling_siglip import SiglipMLP
from transformers.utils import logging, is_flash_attn_2_available

# FlashAttention is mandatory for this model
if not is_flash_attn_2_available():
    raise ImportError(
        "FlashAttention 2 is required for LlavaViTPackingModel. "
        "Please install flash-attn: pip install flash-attn --no-build-isolation"
    )

from flash_attn import flash_attn_varlen_func

logger = logging.get_logger(__name__)


# ---------------------------------------------------------------------------
# Configuration Class
# ---------------------------------------------------------------------------


class LlavaViTPackingConfig(PretrainedConfig):
    r"""
    Configuration class for LlavaViTPacking model with grid_thw (packing) support.

    This configuration supports variable-length sequences through packing, similar to Qwen2VL.
    Instead of using visible_indices, we use grid_thw to specify the temporal, height, and width
    dimensions for each image/video in the batch.

    Args:
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        image_size (`int`, *optional*, defaults to 224):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to 16):
            The size (resolution) of each patch.
        temporal_patch_size (`int`, *optional*, defaults to 1):
            The temporal patch size for video inputs.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler.
        layer_norm_eps (`float`, *optional*, defaults to 1e-6):
            The epsilon used by the layer normalization layers.
        layer_norm_type (`str`, *optional*, defaults to `"layer_norm"`):
            The type of layer normalization to use. Supported values: `"layer_norm"`, `"rms_norm"`.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        use_head (`bool`, *optional*, defaults to `True`):
            Whether to use the pooling head.
        spatial_merge_size (`int`, *optional*, defaults to 1):
            The spatial merge size for the patch merger (1 means no merging).
    """

    model_type = "llava_vit_packing"

    def __init__(
        self,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_channels=3,
        image_size=448,
        patch_size=16,
        temporal_patch_size=1,  # Kept for config compatibility, not used in Conv2d embeddings
        hidden_act="gelu",
        layer_norm_eps=1e-6,
        layer_norm_type="layer_norm",
        attention_dropout=0.0,
        initializer_range=0.02,
        rope_theta=10000.0,
        use_head=True,
        spatial_merge_size=1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size  # Kept for config compatibility
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.layer_norm_type = layer_norm_type
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.rope_theta = rope_theta
        self.use_head = use_head
        self.spatial_merge_size = spatial_merge_size


# ---------------------------------------------------------------------------
# Helper Functions & Layers
# ---------------------------------------------------------------------------


def get_norm_layer(config):
    if config.layer_norm_type == "rms_norm":
        return nn.RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
    else:
        return nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)


def rotate_half(x):
    """
    Interleaved rotation to match Source model's implementation.
    (x1, x2, x3, x4) -> (-x2, x1, -x4, x3)
    """
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)


def apply_rotary_pos_emb_vision(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embedding to query and key tensors for vision.

    Args:
        q: Query tensor of shape (seq_len, num_heads, head_dim)
        k: Key tensor of shape (seq_len, num_heads, head_dim)
        cos: Cosine part of rotary embedding (seq_len, head_dim)
        sin: Sine part of rotary embedding (seq_len, head_dim)

    Returns:
        Tuple of rotated query and key tensors
    """
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q, k = q.float(), k.float()
    cos = cos.unsqueeze(-2).float()  # (seq_len, 1, head_dim)
    sin = sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.to(orig_q_dtype), k_embed.to(orig_k_dtype)


class VisionRotaryEmbedding(nn.Module):
    """Rotary position embedding for vision with 4:6:6 split for T:H:W."""

    def __init__(self, config: LlavaViTPackingConfig):
        super().__init__()
        head_dim = config.hidden_size // config.num_attention_heads
        base = config.rope_theta

        assert head_dim % 2 == 0, "head_dim must be even for rotary."
        assert head_dim % 16 == 0, "head_dim must be divisible by 16."
        half = head_dim // 2
        assert half % 16 == 0, "head_dim//2 must be divisible by 16 to split into 4:6:6."

        self.head_dim = head_dim
        self.half = half

        unit = half // 16
        self.t_size = 4 * unit
        self.h_size = 6 * unit
        self.w_size = 6 * unit

        self.register_buffer(
            "inv_freq_t",
            1.0 / (base ** (torch.arange(self.t_size, dtype=torch.float32) / self.t_size)),
            persistent=False,
        )
        self.register_buffer(
            "inv_freq_h",
            1.0 / (base ** (torch.arange(self.h_size, dtype=torch.float32) / self.h_size)),
            persistent=False,
        )
        self.register_buffer(
            "inv_freq_w",
            1.0 / (base ** (torch.arange(self.w_size, dtype=torch.float32) / self.w_size)),
            persistent=False,
        )

    def forward(self, grid_thw: torch.Tensor) -> torch.Tensor:
        """Compute rotary position embeddings based on grid_thw.

        Args:
            grid_thw: Tensor of shape (num_images, 3) with [t, h, w] for each image

        Returns:
            Rotary position embeddings of shape (total_seq_len, head_dim)
        """
        device = grid_thw.device

        inv_t = self.inv_freq_t.to(device=device)
        inv_h = self.inv_freq_h.to(device=device)
        inv_w = self.inv_freq_w.to(device=device)

        # Vectorized implementation to avoid .item() calls and CUDA synchronization
        # Extract t, h, w as tensors and move to CPU once to avoid repeated sync
        t_vals = grid_thw[:, 0].cpu()  # [num_images] on CPU
        h_vals = grid_thw[:, 1].cpu()  # [num_images] on CPU
        w_vals = grid_thw[:, 2].cpu()  # [num_images] on CPU
        
        # Process all images at once using vectorized operations
        pos_ids = []
        for i in range(grid_thw.shape[0]):
            t = t_vals[i].item()
            h = h_vals[i].item()
            w = w_vals[i].item()
            patches_per_frame = h * w

            # Compute position ids for each axis
            # Temporal position IDs are the frame index (0, 1, 2, ..., t-1) repeated
            # for each patch in that frame, matching the source model's behavior
            t_ids = torch.arange(t, device=device).repeat_interleave(patches_per_frame)
            h_base = torch.arange(h, device=device).repeat_interleave(w)
            h_ids = h_base.repeat(t)
            w_base = torch.arange(w, device=device).repeat(h)
            w_ids = w_base.repeat(t)

            # Compute frequencies for each axis
            ft = torch.outer(t_ids.float(), inv_t)
            fh = torch.outer(h_ids.float(), inv_h)
            fw = torch.outer(w_ids.float(), inv_w)

            # Concatenate frequencies
            freqs = torch.cat([ft, fh, fw], dim=-1)
            pos_ids.append(freqs)

        pos_ids = torch.cat(pos_ids, dim=0)
        # Duplicate for full head_dim
        emb = torch.cat([pos_ids, pos_ids], dim=-1)
        return emb

    def forward_from_positions(self, patch_positions: torch.Tensor) -> torch.Tensor:
        """Compute rotary position embeddings from explicit patch positions.

        This method computes RoPE frequencies directly from patch positions,
        which allows for patches from different spatial-temporal locations
        to be processed together (e.g., for packing multiple images/videos).

        Note: This method expects pre-scaled temporal positions if you want
        consistency with the default grid_thw-based forward() method. Use
        `compute_patch_positions_from_grid_thw()` to generate positions that
        match the default scaling behavior.

        Args:
            patch_positions: Tensor of shape (num_patches, 3) containing [t, h, w]
                positions for each patch in the sequence. For consistency with
                the default forward() behavior, temporal positions should be
                pre-scaled (e.g., [0, 8, 16, 24, ...] for 8 frames instead of
                [0, 1, 2, 3, ...]).

        Returns:
            Rotary position embeddings of shape (num_patches, head_dim)
        """
        device = patch_positions.device

        inv_t = self.inv_freq_t.to(device=device)
        inv_h = self.inv_freq_h.to(device=device)
        inv_w = self.inv_freq_w.to(device=device)

        # Extract positions for each axis
        t_pos = patch_positions[:, 0].float()
        h_pos = patch_positions[:, 1].float()
        w_pos = patch_positions[:, 2].float()

        # Compute frequencies for each axis
        ft = torch.outer(t_pos, inv_t)
        fh = torch.outer(h_pos, inv_h)
        fw = torch.outer(w_pos, inv_w)

        # Concatenate frequencies
        freqs = torch.cat([ft, fh, fw], dim=-1)
        # Duplicate for full head_dim
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb


class Siglip2MultiheadAttentionPoolingHead(nn.Module):
    """Multi-Head Attention Pooling with a learned probe (PMA-style)."""

    def __init__(self, config: LlavaViTPackingConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.probe = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.attention = nn.MultiheadAttention(
            config.hidden_size, config.num_attention_heads, batch_first=True
        )
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)

    def forward(self, hidden_states):
        batch_size = hidden_states.shape[0]
        probe = self.probe.repeat(batch_size, 1, 1)
        attn_output, _ = self.attention(probe, hidden_states, hidden_states)
        residual = attn_output
        attn_output = self.norm(attn_output)
        attn_output = residual + self.mlp(attn_output)
        return attn_output[:, 0]


# ---------------------------------------------------------------------------
# Modeling Components
# ---------------------------------------------------------------------------


class LlavaViTPackingPatchEmbed(nn.Module):
    """Patch embedding layer for packing model (Qwen2VL style).

    Input: (seq_len, patch_size * patch_size * in_channels)
    Output: (seq_len, hidden_size)

    Note: Uses Conv2d for compatibility with vit_preview_v0_hf weights.
    """

    def __init__(self, config: LlavaViTPackingConfig):
        super().__init__()
        self.patch_size = config.patch_size
        self.in_channels = config.num_channels
        self.embed_dim = config.hidden_size

        # Use Conv2d for compatibility with vit_preview_v0_hf weights
        self.proj = nn.Conv2d(
            config.num_channels,
            config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=False,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Embed patches from flattened pixel values.

        Args:
            hidden_states: Flattened pixel values of shape
                (seq_len, patch_size * patch_size * in_channels)

        Returns:
            Patch embeddings of shape (seq_len, hidden_size)
        """
        target_dtype = self.proj.weight.dtype
        # Reshape to (seq_len, in_channels, patch_size, patch_size)
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.patch_size, self.patch_size
        )
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)
        return hidden_states


class LlavaViTPackingAttention(nn.Module):
    """Multi-headed attention with RoPE support for packing."""

    def __init__(self, config: LlavaViTPackingConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} "
                f"and `num_heads`: {self.num_heads})."
            )

        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.qkv = nn.Linear(self.embed_dim, self.embed_dim * 3, bias=True)
        self.proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Forward pass with variable-length attention using FlashAttention.

        Args:
            hidden_states: Input of shape (total_seq_len, hidden_size)
            cu_seqlens: Cumulative sequence lengths for flash attention
            position_embeddings: Tuple of (cos, sin) for rotary embeddings

        Returns:
            Output of shape (total_seq_len, hidden_size)
        """
        seq_length = hidden_states.shape[0]

        # QKV projection
        qkv = self.qkv(hidden_states)
        qkv = qkv.reshape(seq_length, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(1, 0, 2, 3)  # (3, seq_len, num_heads, head_dim)
        query_states, key_states, value_states = qkv.unbind(0)

        # Apply rotary position embeddings
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

        # Use FlashAttention with variable lengths (mandatory)
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

        attn_output = attn_output.reshape(seq_length, -1)
        attn_output = self.proj(attn_output)
        return attn_output


class LlavaViTPackingEncoderLayer(nn.Module):
    """Encoder layer with pre-norm and packing support."""

    def __init__(self, config: LlavaViTPackingConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.layer_norm1 = get_norm_layer(config)
        self.layer_norm2 = get_norm_layer(config)
        self.self_attn = LlavaViTPackingAttention(config)
        self.mlp = SiglipMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(
            hidden_states, cu_seqlens=cu_seqlens, position_embeddings=position_embeddings
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class LlavaViTPackingEncoder(nn.Module):
    """Stack of encoder layers."""

    def __init__(self, config: LlavaViTPackingConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [LlavaViTPackingEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        output_hidden_states: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        all_hidden_states = () if output_hidden_states else None

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            hidden_states = layer(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
            )

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return hidden_states, all_hidden_states


# ---------------------------------------------------------------------------
# Main Models
# ---------------------------------------------------------------------------


class LlavaViTPackingPreTrainedModel(PreTrainedModel):
    config_class = LlavaViTPackingConfig
    base_model_prefix = "llava_vit_packing"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlavaViTPackingEncoderLayer"]
    _supports_flash_attn_2 = True

    def _init_weights(self, module):
        """Initialize the weights."""
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv3d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, (nn.LayerNorm, nn.RMSNorm)):
            module.weight.data.fill_(1.0)
            if hasattr(module, "bias") and module.bias is not None:
                module.bias.data.zero_()


class LlavaViTPackingModel(LlavaViTPackingPreTrainedModel):
    """Llava ViT Model with packing support using grid_thw (Qwen2VL style).

    This model requires FlashAttention and accepts input in
    [seq_len, patch_dim] format where patch_dim = patch_size * patch_size * in_channels,
    similar to Qwen2VL approach.
    """

    def __init__(self, config: LlavaViTPackingConfig):
        super().__init__(config)
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size

        # Patch embeddings layer (Qwen2VL style)
        self.patch_embed = LlavaViTPackingPatchEmbed(config)
        self.layernorm_pre = get_norm_layer(config)
        self.encoder = LlavaViTPackingEncoder(config)
        self.rotary_emb = VisionRotaryEmbedding(config)

        if config.use_head:
            self.layernorm_post = get_norm_layer(config)
            self.head = Siglip2MultiheadAttentionPoolingHead(config)
        else:
            self.layernorm_post = None
            self.head = None

        self.post_init()

    def forward(
        self,
        hidden_states: torch.Tensor,
        grid_thw: torch.Tensor,
        patch_positions: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        """Forward pass with packing support (Qwen2VL style input).

        Args:
            hidden_states: Flattened pixel values of shape (seq_len, patch_dim)
                where seq_len = sum(t*h*w for all images in batch)
                and patch_dim = patch_size * patch_size * in_channels.
            grid_thw: Tensor of shape (num_images, 3) with [t, h, w] for each image,
                where h and w are the number of patches (not pixels).
            patch_positions: Optional tensor of shape (seq_len, 3) containing [t, h, w]
                positions for each patch in the sequence. When provided, this overrides
                the default position calculation based on grid_thw for RoPE computation.
                This is useful when patches come from different images/videos with
                varying spatial-temporal positions.

        Returns:
            BaseModelOutputWithPooling with last_hidden_state and optionally pooler_output
        """
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
            if hasattr(self.config, "output_hidden_states")
            else False
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # hidden_states should be (seq_len, patch_dim) - flattened pixel values
        if hidden_states.dim() != 2:
            raise ValueError(
                f"Expected hidden_states to have 2 dimensions (seq_len, patch_dim), "
                f"got {hidden_states.dim()} dimensions with shape {hidden_states.shape}. "
                f"Input should be flattened pixel values."
            )

        # Patch embedding: (seq_len, patch_dim) -> (seq_len, hidden_size)
        hidden_states = self.patch_embed(hidden_states)

        # Compute rotary position embeddings
        if patch_positions is not None:
            # Use explicit patch positions for RoPE calculation
            if patch_positions.ndim != 2 or patch_positions.shape[1] != 3:
                raise ValueError(
                    f"patch_positions must have shape (seq_len, 3), got {patch_positions.shape}"
                )
            if patch_positions.shape[0] != hidden_states.shape[0]:
                raise ValueError(
                    f"patch_positions seq_len ({patch_positions.shape[0]}) must match "
                    f"hidden_states seq_len ({hidden_states.shape[0]})"
                )
            rotary_pos_emb = self.rotary_emb.forward_from_positions(patch_positions)
        else:
            # Compute positions from grid_thw (default behavior)
            rotary_pos_emb = self.rotary_emb(grid_thw)
        position_embeddings = (rotary_pos_emb.cos(), rotary_pos_emb.sin())

        # Compute cumulative sequence lengths for FlashAttention
        seq_lengths = grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]
        cu_seqlens = F.pad(seq_lengths.cumsum(dim=0), (1, 0), value=0).to(torch.int32)

        # Pre-norm
        hidden_states = self.layernorm_pre(hidden_states)

        # Encoder
        hidden_states, all_hidden_states = self.encoder(
            hidden_states,
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
            output_hidden_states=output_hidden_states,
        )

        # Post-norm and pooling head
        if self.layernorm_post is not None:
            hidden_states = self.layernorm_post(hidden_states)

        # Handle pooling for each sample separately
        pooled_output = None
        if self.head is not None:
            # Split hidden states by sample and apply pooling
            pooled_outputs = []
            for i in range(len(cu_seqlens) - 1):
                start = cu_seqlens[i].item()
                end = cu_seqlens[i + 1].item()
                sample_hidden = hidden_states[start:end].unsqueeze(0)
                pooled = self.head(sample_hidden)
                pooled_outputs.append(pooled)
            pooled_output = torch.cat(pooled_outputs, dim=0)

        if not return_dict:
            return (hidden_states, pooled_output) + (all_hidden_states,)

        return BaseModelOutputWithPooling(
            last_hidden_state=hidden_states,
            pooler_output=pooled_output,
            hidden_states=all_hidden_states,
            attentions=None,
        )


# ---------------------------------------------------------------------------
# TIMM Registry Functions
# ---------------------------------------------------------------------------


@register_model
def hf_llava_vit_packing_small_ln(pretrained: bool = False, ckpt_path=None, **kwargs):
    config = LlavaViTPackingConfig(
        patch_size=16,
        hidden_size=384,
        num_attention_heads=384 // 64,
        num_hidden_layers=6,
        intermediate_size=1536,
        hidden_act="gelu",
        layer_norm_type="layer_norm",
        use_head=True,
    )
    # FlashAttention is mandatory for this model
    config._attn_implementation = "flash_attention_2"
    model = LlavaViTPackingModel(config)
    return model


@register_model
def hf_llava_vit_packing_base_ln(pretrained: bool = False, ckpt_path=None, **kwargs):
    config = LlavaViTPackingConfig(
        patch_size=16,
        hidden_size=768,
        num_attention_heads=768 // 64,
        num_hidden_layers=12,
        intermediate_size=3072,
        hidden_act="gelu",
        layer_norm_type="layer_norm",
        use_head=True,
    )
    # FlashAttention is mandatory for this model
    config._attn_implementation = "flash_attention_2"
    model = LlavaViTPackingModel(config)
    return model


@register_model
def hf_llava_vit_packing_large_ln(pretrained: bool = False, ckpt_path=None, **kwargs):
    config = LlavaViTPackingConfig(
        patch_size=14,
        hidden_size=1024,
        num_attention_heads=1024 // 64,
        num_hidden_layers=24,
        intermediate_size=4096,
        hidden_act="gelu",
        layer_norm_type="layer_norm",
        use_head=True,
    )
    # FlashAttention is mandatory for this model
    config._attn_implementation = "flash_attention_2"
    model = LlavaViTPackingModel(config)
    return model


@register_model
def hf_llava_vit_packing_huge_ln(pretrained: bool = False, ckpt_path=None, **kwargs):
    config = LlavaViTPackingConfig(
        patch_size=14,
        hidden_size=1280,
        num_attention_heads=1280 // 64,
        num_hidden_layers=32,
        intermediate_size=5120,
        hidden_act="gelu",
        layer_norm_type="layer_norm",
        use_head=True,
    )
    # FlashAttention is mandatory for this model
    config._attn_implementation = "flash_attention_2"
    model = LlavaViTPackingModel(config)
    return model


@register_model
def hf_llava_vit_packing_giant_ln(pretrained: bool = False, ckpt_path=None, **kwargs):
    config = LlavaViTPackingConfig(
        patch_size=14,
        hidden_size=1536,
        num_attention_heads=1536 // 96,
        num_hidden_layers=40,
        intermediate_size=6144,
        hidden_act="gelu",
        layer_norm_type="layer_norm",
        use_head=True,
    )
    # FlashAttention is mandatory for this model
    config._attn_implementation = "flash_attention_2"
    model = LlavaViTPackingModel(config)
    return model


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------


def compute_patch_positions_from_grid_thw(grid_thw: torch.Tensor) -> torch.Tensor:
    """Compute patch positions from grid_thw tensor.

    This utility function generates patch_positions tensor from grid_thw,
    which can be used to explicitly specify the RoPE positions for each patch.

    Note: The temporal positions are the frame index (0, 1, 2, ..., t-1), repeated
    for each patch in that frame, matching the source model's behavior.

    Args:
        grid_thw: Tensor of shape (num_images, 3) with [t, h, w] for each image

    Returns:
        patch_positions: Tensor of shape (total_seq_len, 3) with [t, h, w] positions
            for each patch in the sequence.
    """
    device = grid_thw.device
    positions = []

    # Vectorized implementation to avoid .item() calls and CUDA synchronization
    # Extract t, h, w as tensors and move to CPU once to avoid repeated sync
    t_vals = grid_thw[:, 0].cpu()  # [num_images] on CPU
    h_vals = grid_thw[:, 1].cpu()  # [num_images] on CPU
    w_vals = grid_thw[:, 2].cpu()  # [num_images] on CPU

    for i in range(grid_thw.shape[0]):
        t = t_vals[i].item()
        h = h_vals[i].item()
        w = w_vals[i].item()
        patches_per_frame = h * w

        # Compute position for each axis
        # Temporal positions are the frame index, matching the source model
        t_ids = torch.arange(t, device=device).repeat_interleave(patches_per_frame)
        h_base = torch.arange(h, device=device).repeat_interleave(w)
        h_ids = h_base.repeat(t)
        w_base = torch.arange(w, device=device).repeat(h)
        w_ids = w_base.repeat(t)

        # Stack positions as (L, 3) tensor
        pos = torch.stack([t_ids, h_ids, w_ids], dim=-1)
        positions.append(pos)

    return torch.cat(positions, dim=0)


if __name__ == "__main__":
    import timm

    # Test with a simple example
    # Note: This requires FlashAttention and CUDA
    print("Creating model (requires FlashAttention)...")
    model = timm.create_model("hf_llava_vit_packing_base_ln", pretrained=False)
    print(f"Model created: {type(model)}")

    # Create test input in [seq_len, patch_dim] format (Qwen2VL style)
    # Simulating 1 image with 14x14 = 196 patches
    # patch_dim = patch_size * patch_size * in_channels
    # For base model: patch_size=16, in_channels=3
    patch_size = 16
    in_channels = 3
    patch_dim = patch_size * patch_size * in_channels  # 16*16*3 = 768
    seq_len = 14 * 14  # 196 patches for a single image

    grid_thw = torch.tensor([[1, 14, 14]], dtype=torch.long)
    hidden_states = torch.randn(seq_len, patch_dim)  # (seq_len, patch_dim)

    print(f"Input shape: {hidden_states.shape}")
    print(f"patch_dim = {patch_size}*{patch_size}*{in_channels} = {patch_dim}")
    print(f"grid_thw: {grid_thw}")

    # Forward pass (requires CUDA for FlashAttention)
    if torch.cuda.is_available():
        model = model.cuda()
        hidden_states = hidden_states.cuda()
        grid_thw = grid_thw.cuda()

        with torch.no_grad():
            outputs = model(hidden_states=hidden_states, grid_thw=grid_thw)

        print(f"Last hidden state shape: {outputs.last_hidden_state.shape}")
        if outputs.pooler_output is not None:
            print(f"Pooler output shape: {outputs.pooler_output.shape}")
    else:
        print("CUDA not available - skipping forward pass test (FlashAttention requires CUDA)")
