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
""" PyTorch Llava ViT model."""

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from timm.models.registry import register_model
from torch.nn import functional as F
from transformers.activations import ACT2FN
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import (BaseModelOutput,
                                           BaseModelOutputWithPooling)
from transformers.modeling_utils import PreTrainedModel
from transformers.models.siglip.modeling_siglip import SiglipMLP
from transformers.utils import (add_start_docstrings,
                                add_start_docstrings_to_model_forward, logging,
                                replace_return_docstrings, is_flash_attn_2_available)


from flash_attn import flash_attn_func

logger = logging.get_logger(__name__)


# ---------------------------------------------------------------------------
# Configuration Class
# ---------------------------------------------------------------------------

class LlavaViTConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`LlavaViTModel`]. It is used to instantiate a
    Llava ViT model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the Llava ViT architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

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

    Example:

    ```python
    >>> from transformers import LlavaViTConfig, LlavaViTModel

    >>> # Initializing a LlavaViT configuration
    >>> configuration = LlavaViTConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = LlavaViTModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "llava_vit"

    def __init__(
        self,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_channels=3,
        image_size=448,
        patch_size=16,
        hidden_act="gelu",
        layer_norm_eps=1e-6,
        layer_norm_type="layer_norm",
        attention_dropout=0.0,
        initializer_range=0.02,
        rope_theta=10000.0,
        use_head=True,
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
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.layer_norm_type = layer_norm_type
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.rope_theta = rope_theta
        self.use_head = use_head


# ---------------------------------------------------------------------------
# Model Docstrings
# ---------------------------------------------------------------------------

LLAVA_VIT_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`LlavaViTConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

LLAVA_VIT_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` or `(batch_size, num_channels, num_frames, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`].
        visible_indices (`torch.Tensor`, *optional*):
            Indices of visible patches for masking. Used in MAE-style pretraining or inference.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


# ---------------------------------------------------------------------------
# Helper Functions & Layers
# ---------------------------------------------------------------------------

def get_norm_layer(config):
    if config.layer_norm_type == "rms_norm":
        return nn.RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
    else:
        return nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)


# def rotate_half(x):
#     x1 = x[..., : x.shape[-1] // 2]
#     x2 = x[..., x.shape[-1] // 2 :]
#     return torch.cat((-x2, x1), dim=-1)


def rotate_half(x):
    """
    Interleaved rotation to match Source model's implementation.
    (x1, x2, x3, x4) -> (-x2, x1, -x4, x3)
    """
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)


def apply_rotary_pos_emb(q, k, freqs):
    # q, k: (B, H, L, D)
    # freqs: (B, L, D)

    # We need to broadcast freqs to match heads
    # (B, L, D) -> (B, 1, L, D)

    # !!! CRITICAL FIX: Cast cos/sin to q.dtype (bf16/fp16) immediately
    # freqs are typically float32, so cos() returns float32.
    # Without this cast, (q * cos) upcasts q to float32, causing FlashAttention to fail.
    cos = freqs.cos().unsqueeze(1).to(q.dtype)
    sin = freqs.sin().unsqueeze(1).to(q.dtype)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class VideoRotaryEmbeddingSplit466(nn.Module):
    """
    3D (T,H,W) Rotary frequency constructor with 4:6:6 split.
    """
    def __init__(self, config: LlavaViTConfig):
        super().__init__()
        head_dim = config.hidden_size // config.num_attention_heads
        base = config.rope_theta

        assert head_dim % 2 == 0, "head_dim must be even for rotary."
        assert head_dim % 16 == 0, "head_dim must be divisible by 16."
        half = head_dim // 2
        assert half % 16 == 0, "head_dim//2 must also be divisible by 16 to split into 4:6:6."

        self.head_dim = head_dim
        self.half = half

        unit = half // 16
        self.t_size = 4 * unit
        self.h_size = 6 * unit
        self.w_size = 6 * unit

        self.register_buffer("inv_freq_t", 1.0 / (base ** (torch.arange(self.t_size, dtype=torch.float32) / self.t_size)), persistent=False)
        self.register_buffer("inv_freq_h", 1.0 / (base ** (torch.arange(self.h_size, dtype=torch.float32) / self.h_size)), persistent=False)
        self.register_buffer("inv_freq_w", 1.0 / (base ** (torch.arange(self.w_size, dtype=torch.float32) / self.w_size)), persistent=False)

    def forward(self, t: int, h: int, w: int, device=None):
        if device is None: device = self.inv_freq_t.device

        inv_t = self.inv_freq_t.to(device=device)
        inv_h = self.inv_freq_h.to(device=device)
        inv_w = self.inv_freq_w.to(device=device)

        ft = torch.outer(torch.arange(t, device=device, dtype=torch.float32), inv_t)
        fh = torch.outer(torch.arange(h, device=device, dtype=torch.float32), inv_h)
        fw = torch.outer(torch.arange(w, device=device, dtype=torch.float32), inv_w)

        t_ids = torch.arange(t, device=device).repeat_interleave(h * w)
        h_ids = torch.arange(h, device=device).repeat_interleave(w).repeat(t)
        w_ids = torch.arange(w, device=device).repeat(h).repeat(t)

        freqs = torch.cat([ft[t_ids], fh[h_ids], fw[w_ids]], dim=-1)
        return freqs


class Siglip2MultiheadAttentionPoolingHead(nn.Module):
    """
    Multi-Head Attention Pooling with a learned probe (PMA-style).
    """
    def __init__(self, config: LlavaViTConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.probe = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.attention = nn.MultiheadAttention(config.hidden_size, config.num_attention_heads, batch_first=True)
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

class LlavaViTEmbeddings(nn.Module):
    def __init__(self, config: LlavaViTConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        # Handle 4D (B, C, H, W) or 5D (B, C, T, H, W) inputs
        if pixel_values.dim() == 4:
             pixel_values = pixel_values.unsqueeze(2) # (B, C, 1, H, W)

        batch_size, channels, t_frames, height, width = pixel_values.shape

        # Merge time into batch for Conv2d
        x_2d = pixel_values.permute(0, 2, 1, 3, 4).reshape(batch_size * t_frames, channels, height, width)

        # Patch Embed
        embeddings = self.patch_embedding(x_2d)  # (B*T, C, Hp, Wp)
        embeddings = embeddings.flatten(2).transpose(1, 2) # (B*T, L_frame, C)

        # Flatten all patches
        total_patches = t_frames * (height // self.patch_size) * (width // self.patch_size)
        embeddings = embeddings.reshape(batch_size, total_patches, self.embed_dim)

        return embeddings


class LlavaViTAttention(nn.Module):
    """Multi-headed attention with RoPE support"""
    def __init__(self, config: LlavaViTConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
             raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
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
        attention_mask: Optional[torch.Tensor] = None,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        batch_size, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # (B, L, H, D) -> Transpose to (B, H, L, D)
        query_states = query_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        if rotary_pos_emb is not None:
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, rotary_pos_emb)

        # Calculate attention scores
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale
        # print("attention_mask", attention_mask.shape)

        if attention_mask is not None:
            if attention_mask.size() != (batch_size, 1, q_len, q_len):
                if attention_mask.dim() == 3:
                     attention_mask = attention_mask.unsqueeze(1)
            attn_weights = attn_weights + attention_mask
            
        # FIX: Remove dtype=torch.float32 to stay in original dtype (bf16/fp16)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, q_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights if output_attentions else None


class LlavaViTFlashAttention2(nn.Module):
    """
    Multi-headed attention with RoPE support using Flash Attention 2.
    This module implements the same attention mechanism as LlavaViTAttention but uses
    Flash Attention for improved performance and memory efficiency.
    """
    def __init__(self, config: LlavaViTConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
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
        attention_mask: Optional[torch.Tensor] = None,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass using Flash Attention 2.
        """
        batch_size, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Flash Attention requires (B, L, H, D) format
        query_states = query_states.view(batch_size, q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(batch_size, q_len, self.num_heads, self.head_dim)
        value_states = value_states.view(batch_size, q_len, self.num_heads, self.head_dim)

        # Apply RoPE if provided
        if rotary_pos_emb is not None:
            # Transpose for RoPE application: (B, L, H, D) -> (B, H, L, D)
            query_states = query_states.transpose(1, 2)
            key_states = key_states.transpose(1, 2)
            # NOTE: apply_rotary_pos_emb now ensures NO float32 cast happens
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, rotary_pos_emb)
            # Transpose back: (B, H, L, D) -> (B, L, H, D)
            query_states = query_states.transpose(1, 2)
            key_states = key_states.transpose(1, 2)

        # FIX: Removed the explicit float32 check and downcast.
        # We assume input is already correct (bf16/fp16) thanks to RoPE fix.

        # Flash Attention forward pass
        attn_output = flash_attn_func(
            query_states,
            key_states,
            value_states,
            dropout_p=self.dropout if self.training else 0.0,
            softmax_scale=self.scale,
            causal=False,
        )

        # Reshape to (B, L, embed_dim)
        attn_output = attn_output.reshape(batch_size, q_len, self.embed_dim)

        # No extra casting here.
        attn_output = self.out_proj(attn_output)

        return attn_output, None


LLAVA_VIT_ATTENTION_CLASSES = {
    # "eager": LlavaViTAttention,
    "flash_attention_2": LlavaViTFlashAttention2,
}


class LlavaViTEncoderLayer(nn.Module):
    def __init__(self, config: LlavaViTConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        # Get attention implementation from config, default to "flash_attention_2"
        # attn_implementation = getattr(config, "_attn_implementation", "flash_attention_2")
        # if attn_implementation not in LLAVA_VIT_ATTENTION_CLASSES:
        #     raise ValueError(
        #         f"Unknown attention implementation: {attn_implementation}. "
        #         f"Available implementations: {list(LLAVA_VIT_ATTENTION_CLASSES.keys())}"
        #     )
        self.self_attn = LLAVA_VIT_ATTENTION_CLASSES["flash_attention_2"](config)
        self.layer_norm1 = get_norm_layer(config)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = get_norm_layer(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)

        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            rotary_pos_emb=rotary_pos_emb,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states, attn_weights) if output_attentions else (hidden_states,)
        return outputs


class LlavaViTEncoder(nn.Module):
    def __init__(self, config: LlavaViTConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([LlavaViTEncoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[Tuple, BaseModelOutput]:

        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                rotary_pos_emb=rotary_pos_emb,
                output_attentions=output_attentions,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


# ---------------------------------------------------------------------------
# Main Models
# ---------------------------------------------------------------------------

@add_start_docstrings(
    "The bare Llava ViT Model outputting raw hidden-states without any specific head on top.",
    LLAVA_VIT_START_DOCSTRING,
)
class LlavaViTPreTrainedModel(PreTrainedModel):
    config_class = LlavaViTConfig
    base_model_prefix = "llava_vit"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlavaViTEncoderLayer"]
    _supports_flash_attn_2 = True

    def _init_weights(self, module):
        """Initialize the weights"""
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, (nn.LayerNorm, nn.RMSNorm)):
            # 修复：RMSNorm 没有 bias，必须先检查 hasattr
            module.weight.data.fill_(1.0)
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.zero_()


@add_start_docstrings(
    "Llava ViT Model with a vision transformer encoder.",
    LLAVA_VIT_START_DOCSTRING,
)
class LlavaViTModel(LlavaViTPreTrainedModel):
    def __init__(self, config: LlavaViTConfig):
        super().__init__(config)
        self.config = config

        self.embeddings = LlavaViTEmbeddings(config)
        self.layernorm_pre = get_norm_layer(config)
        self.encoder = LlavaViTEncoder(config)
        self.video_rope = VideoRotaryEmbeddingSplit466(config)

        if config.use_head:
             self.layernorm_post = get_norm_layer(config)
             self.head = Siglip2MultiheadAttentionPoolingHead(config)
        else:
             self.layernorm_post = None
             self.head = None

        self.post_init()


    def mask_by_residual_topk(self, res: torch.Tensor, k_keep: int):
        """
        基于残差 res 的 Top-K 掩码。
        选择 |res| 在 patch 内求和后的得分最高的 K 个 patch 作为可见，其余为 mask。

        Args:
            res:  (B, 1, T, H, W)  —— I 帧建议事先置 0，这样自然会优先选到 P 帧。
            k_keep: int            —— 每个样本保留的可见块数量（Top-K 超参）

        Returns:
            visible_indices: LongTensor (B, K)   —— 选中的线性 patch 索引（按升序）
            visible_mask:    BoolTensor (B, L)   —— L = T * (H/Ph) * (W/Pw)
            ids_restore:     LongTensor (B, L)   —— MAE 风格的还原下标
        """
        assert res.dim() == 5 and res.size(1) == 1, "res 需为 (B,1,T,H,W)"
        B, _, T, H, W = res.shape

        patch_size = self.config.patch_size
        if isinstance(patch_size, int):
            ph = pw = patch_size
        else:
            ph, pw = patch_size
            
        assert H % ph == 0 and W % pw == 0, "H/W 必须能被 patch 大小整除"

        hb, wb = H // ph, W // pw        # 每帧的 patch 网格
        L = T * hb * wb                  # 总 patch 数

        # K 边界
        K = int(max(0, min(k_keep, L)))

        # 计算每个 patch 的残差得分（|.| 在 patch 内求和） -> (B, T, hb, wb)
        # 参考：res_c = res[:hb*ph, :wb*pw].reshape(hb, ph, wb, pw); s = |res_c|.sum(axis=(1,3))
        res_abs = res.abs().squeeze(1)                                 # (B,T,H,W)
        scores = res_abs.reshape(B, T, hb, ph, wb, pw).sum(dim=(3, 5)) # (B,T,hb,wb)
        scores = scores.reshape(B, L)                                  # (B, L)

        # 选 Top-K（按 batch 独立进行）
        if K > 0:
            topk_idx = torch.topk(scores, k=K, dim=1, largest=True, sorted=False).indices  # (B, K)
            visible_indices = torch.sort(topk_idx, dim=1).values                           # (B, K) 升序，便于后续索引
        else:
            visible_indices = torch.empty(B, 0, dtype=torch.long, device=res.device)

        # 构造可见 mask
        visible_mask = torch.zeros(B, L, dtype=torch.bool, device=res.device)
        if K > 0:
            visible_mask.scatter_(1, visible_indices, True)

        # MAE 风格 ids_restore：把可见块排在前面、遮挡块排在后面
        vis_int   = visible_mask.long()
        mask_int  = 1 - vis_int
        vis_rank  = torch.cumsum(vis_int, dim=1) - 1
        mask_rank = torch.cumsum(mask_int, dim=1) - 1
        n_visible_col = vis_int.sum(dim=1, keepdim=True)
        ids_restore = torch.where(visible_mask, vis_rank, n_visible_col + mask_rank).long()

        return visible_indices, visible_mask, ids_restore
    
    @add_start_docstrings_to_model_forward(LLAVA_VIT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=LlavaViTConfig)
    def forward(
        self,
        pixel_values: torch.Tensor,
        visible_indices: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from transformers import LlavaViTModel
        >>> import torch

        >>> model = LlavaViTModel.from_pretrained("DeepGlint-AI/llava-vit")
        >>> pixel_values = torch.randn(1, 3, 224, 224) # Video input can be 5D (B, C, T, H, W) or 4D
        >>> outputs = model(pixel_values)
        >>> last_hidden_states = outputs.last_hidden_state
        ```
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Determine video dimensions for RoPE
        # Note: pixel_values passed to embeddings can be 4D or 5D
        if pixel_values.dim() == 5:
             t_frames = pixel_values.shape[2]
             height = pixel_values.shape[3]
             width = pixel_values.shape[4]
        else:
             t_frames = 1
             height = pixel_values.shape[2]
             width = pixel_values.shape[3]

        # 1. Embeddings
        hidden_states = self.embeddings(pixel_values)
        batch_size, total_patches, _ = hidden_states.shape

        if visible_indices.ndim == 5:
            visible_indices, _, _ = self.mask_by_residual_topk(
                res=visible_indices,
                k_keep=2048
            )

        # 2. Visible Indices Handling
        if visible_indices is None:
             visible_indices = torch.arange(total_patches, device=pixel_values.device).unsqueeze(0).expand(batch_size, -1)


        gather_index = visible_indices.unsqueeze(-1).expand(-1, -1, self.config.hidden_size)
        hidden_states = torch.gather(hidden_states, 1, gather_index)

        # 3. RoPE Construction
        freqs_full = self.video_rope(
            t=t_frames,
            h=height // self.config.patch_size,
            w=width // self.config.patch_size,
            device=pixel_values.device
        )
        freqs_visible = freqs_full[visible_indices]

        # Concatenate D/2 + D/2 -> D for applying rope
        freqs_visible = torch.cat([freqs_visible, freqs_visible], dim=-1)

        # 4. Pre-Norm & Encoder
        hidden_states = self.layernorm_pre(hidden_states)

        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=None,
            rotary_pos_emb=freqs_visible,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = encoder_outputs[0]

        # === 修改开始 ===
        # Source Logic: out = self.ln_post(out); head_output = self.head(out)
        # 为了对齐，我们在 encoder 输出后立即应用 Post-  Norm
        if self.layernorm_post is not None:
            sequence_output = self.layernorm_post(sequence_output)

        # 5. Pooling Head
        pooled_output = None
        if self.head is not None:
            # 因为 sequence_output 已经 Norm 过了，这里直接传给 head
            pooled_output = self.head(sequence_output)
        # === 修改结束 ===

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output, # 现在这包含了 ln_post 的效果
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


# ---------------------------------------------------------------------------
# TIMM Registry Functions
# ---------------------------------------------------------------------------
@register_model
def hf_llava_vit_small_ln(pretrained: bool = False, ckpt_path=None, **kwargs):
    config = LlavaViTConfig(
        patch_size=16,
        hidden_size=384,
        num_attention_heads=384 // 64,
        num_hidden_layers=6,
        intermediate_size=1536,
        hidden_act="gelu",
        layer_norm_type="layer_norm",
        use_head=True
    )
    config._attn_implementation = "flash_attention_2"
    model = LlavaViTModel(config)
    return model

@register_model
def hf_llava_vit_base_ln(pretrained: bool = False, ckpt_path=None, **kwargs):
    config = LlavaViTConfig(
        patch_size=16,
        hidden_size=768,
        num_attention_heads=768 // 64,
        num_hidden_layers=12,
        intermediate_size=3072,
        hidden_act="gelu",
        layer_norm_type="layer_norm",
        use_head=True
    )
    config._attn_implementation = "flash_attention_2"
    model = LlavaViTModel(config)
    return model

@register_model
def hf_llava_vit_large_ln(pretrained: bool = False, ckpt_path=None, **kwargs):
    config = LlavaViTConfig(
        patch_size=14,
        hidden_size=1024,
        num_attention_heads=1024 // 64,
        num_hidden_layers=24,
        intermediate_size=4096,
        hidden_act="gelu",
        layer_norm_type="layer_norm",
        use_head=True
    )
    config._attn_implementation = "flash_attention_2"
    model = LlavaViTModel(config)
    return model

@register_model
def hf_llava_vit_huge_ln(pretrained: bool = False, ckpt_path=None, **kwargs):
    config = LlavaViTConfig(
        patch_size=14,
        hidden_size=1536,
        num_attention_heads=16,
        num_hidden_layers=27,
        intermediate_size=4608,
        hidden_act="gelu",
        layer_norm_type="layer_norm",
        use_head=True
    )
    config._attn_implementation = "flash_attention_2"
    model = LlavaViTModel(config)
    return model

@register_model
def hf_llava_vit_giant_ln(pretrained: bool = False, ckpt_path=None, **kwargs):
    config = LlavaViTConfig(
        patch_size=14,
        hidden_size=1536,
        num_attention_heads=1536 // 96,
        num_hidden_layers=40,
        intermediate_size=6144,
        hidden_act="gelu",
        layer_norm_type="layer_norm",
        use_head=True
    )
    config._attn_implementation = "flash_attention_2"
    model = LlavaViTModel(config)
    return model

if __name__ == "__main__":
    import timm

    model = timm.create_model("hf_llava_vit_huge_ln", pretrained=False).cuda()

    bs = 4
    test_input = torch.randn(bs, 3, 224, 224, device=model.device)
    # model, test_input = model.bfloat16(), test_input.bfloat16()
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        last_hidden_state = model(test_input).last_hidden_state

    print(f"Input shape: {test_input.shape}")
    print(f"Last hidden state shape: {last_hidden_state.shape}")
