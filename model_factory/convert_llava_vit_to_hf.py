import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from typing import Callable, Optional, Tuple, Union
import numpy as np
import math

# ==============================================================================
# 1. Source Model Definitions (Original Implementation)
# ==============================================================================

from timm.models.layers import to_2tuple, trunc_normal_
from timm.models.registry import register_model

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

class VideoRotaryEmbeddingSplit466(nn.Module):
    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        assert head_dim % 2 == 0
        assert head_dim % 16 == 0
        half = head_dim // 2
        assert half % 16 == 0

        self.head_dim = head_dim
        self.half = half
        self.base = base

        unit = half // 16
        self.t_size = 4 * unit
        self.h_size = 6 * unit
        self.w_size = 6 * unit

        self.register_buffer(
            "inv_freq_t",
            1.0 / (base ** (torch.arange(self.t_size, dtype=torch.float32) / self.t_size)),
            persistent=False
        )
        self.register_buffer(
            "inv_freq_h",
            1.0 / (base ** (torch.arange(self.h_size, dtype=torch.float32) / self.h_size)),
            persistent=False
        )
        self.register_buffer(
            "inv_freq_w",
            1.0 / (base ** (torch.arange(self.w_size, dtype=torch.float32) / self.w_size)),
            persistent=False
        )

    @torch.no_grad()
    def forward(self, t: int, h: int, w: int, device=None, dtype=torch.float32):
        if device is None:
            device = self.inv_freq_t.device

        inv_t = self.inv_freq_t.to(device=device, dtype=dtype)
        inv_h = self.inv_freq_h.to(device=device, dtype=dtype)
        inv_w = self.inv_freq_w.to(device=device, dtype=dtype)

        ft = torch.outer(torch.arange(t, device=device, dtype=dtype), inv_t)
        fh = torch.outer(torch.arange(h, device=device, dtype=dtype), inv_h)
        fw = torch.outer(torch.arange(w, device=device, dtype=dtype), inv_w)

        t_ids = torch.arange(t, device=device).repeat_interleave(h * w)
        h_base = torch.arange(h, device=device).repeat_interleave(w)
        h_ids = h_base.repeat(t)
        w_base = torch.arange(w, device=device).repeat(h)
        w_ids = w_base.repeat(t)

        freqs = torch.cat([ft[t_ids], fh[h_ids], fw[w_ids]], dim=-1)
        return freqs

class VisionSdpaAttentionCausal(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attn_dropout=0.0):
        super().__init__()
        assert hidden_size % num_attention_heads == 0
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.in_proj = nn.Linear(hidden_size, hidden_size * 3)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.attn_dropout = attn_dropout

    def forward(self, hidden_states, rotary_pos_emb=None, attention_mask=None):
        L, B, C = hidden_states.shape
        qkv = self.in_proj(hidden_states).view(L, B, 3, self.num_attention_heads, self.head_dim).permute(2,1,0,3,4)
        q, k, v = qkv.unbind(0)

        if rotary_pos_emb is not None:
            if rotary_pos_emb.dim() == 2:
                rotary_pos_emb = rotary_pos_emb.unsqueeze(0)
            cos = rotary_pos_emb.cos()
            sin = rotary_pos_emb.sin()
            cos = torch.cat([cos, cos], dim=-1).unsqueeze(2)
            sin = torch.cat([sin, sin], dim=-1).unsqueeze(2)

            def rotate_half_local(x):
                x_even = x[..., ::2]
                x_odd  = x[..., 1::2]
                return torch.stack((-x_odd, x_even), dim=-1).reshape_as(x)

            q = (q * cos) + (rotate_half_local(q) * sin)
            k = (k * cos) + (rotate_half_local(k) * sin)

        q = q.permute(0,2,1,3)
        k = k.permute(0,2,1,3)
        v = v.permute(0,2,1,3)

        if attention_mask is not None:
            attn_mask = attention_mask.unsqueeze(1)
        else:
            attn_mask = None

        attn_out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout if self.training else 0.0
        )
        attn_out = attn_out.permute(2,0,1,3).contiguous().view(L,B,C)
        return self.out_proj(attn_out)

class ResidualAttentionBlockCausal(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, intermediate_size,
                 act_layer=nn.GELU, attn_dropout=0.0, norm_cls: Callable=nn.LayerNorm):
        super().__init__()
        self.ln_1 = norm_cls(hidden_size)
        self.attn = VisionSdpaAttentionCausal(hidden_size, num_attention_heads, attn_dropout=attn_dropout)
        self.ln_2 = norm_cls(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            act_layer(),
            nn.Linear(intermediate_size, hidden_size),
        )

    def forward(self, x, rotary_pos_emb=None, attention_mask=None):
        x = x + self.attn(self.ln_1(x), rotary_pos_emb=rotary_pos_emb, attention_mask=attention_mask)
        x = x + self.mlp(self.ln_2(x))
        return x

class TransformerCausal(nn.Module):
    def __init__(self,
                 hidden_size,
                 num_hidden_layers,
                 num_attention_heads,
                 intermediate_size,
                 act_layer=nn.GELU,
                 gradient_checkpointing=False,
                 attn_dropout=0.0,
                 norm_cls: Callable=nn.LayerNorm):
        super().__init__()
        self.layers = nn.ModuleList([
            ResidualAttentionBlockCausal(
                hidden_size,
                num_attention_heads,
                intermediate_size,
                act_layer=act_layer,
                attn_dropout=attn_dropout,
                norm_cls=norm_cls
            ) for _ in range(num_hidden_layers)
        ])
        self.grad_checkpointing = gradient_checkpointing

    def forward(self, x, rotary_pos_emb=None, attention_mask=None):
        for blk in self.layers:
            blk: ResidualAttentionBlockCausal
            if self.grad_checkpointing and not torch.jit.is_scripting():
                def cf(t):
                    return blk(t, rotary_pos_emb=rotary_pos_emb, attention_mask=attention_mask)
                x = torch.utils.checkpoint.checkpoint(cf, x)
            else:
                x = blk(x, rotary_pos_emb=rotary_pos_emb, attention_mask=attention_mask)
        return x

class Siglip2MLP_Src(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.activation_fn = F.gelu
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states

class Siglip2MultiheadAttentionPoolingHead_Src(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, intermediate_size, norm_cls=nn.RMSNorm):
        super().__init__()
        self.probe = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.attention = torch.nn.MultiheadAttention(hidden_size, num_attention_heads, batch_first=True)
        self.norm = norm_cls(hidden_size)
        self.mlp = Siglip2MLP_Src(hidden_size, intermediate_size)
        self.num_heads = num_attention_heads

    def forward(self, hidden_state: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = hidden_state.shape[0]
        probe = self.probe.repeat(batch_size, 1, 1)
        hidden_state = self.attention(probe, hidden_state, hidden_state)[0]
        residual = hidden_state
        hidden_state = self.norm(hidden_state)
        hidden_state = residual + self.mlp(hidden_state)
        return hidden_state[:, 0]

class LlavaViTEncoder(nn.Module):
    def __init__(
        self,
        patch_size=16,
        hidden_size=384,
        head_dim=64,
        num_hidden_layers=12,
        intermediate_size=1536,
        act_layer=nn.GELU,
        num_key_value_heads=None,
        use_gradient_checkpointing=False,
        attn_dropout=0.0,
        use_causal_temporal=False,
        norm_cls=nn.RMSNorm,
        mask_ratio=0.5,
        use_head=False,
    ):
        super().__init__()
        assert hidden_size % head_dim == 0
        num_attention_heads = hidden_size // head_dim
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.num_attention_heads = num_attention_heads
        self.patch_size = to_2tuple(patch_size)

        self.conv1 = nn.Conv2d(3, hidden_size, kernel_size=patch_size, stride=patch_size, bias=False)
        self.ln_pre = norm_cls(hidden_size)

        self.transformer = TransformerCausal(
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            act_layer=act_layer,
            gradient_checkpointing=use_gradient_checkpointing,
            attn_dropout=attn_dropout,
            norm_cls=norm_cls,
        )

        self.video_rope = VideoRotaryEmbeddingSplit466(head_dim)
        self.ln_post = norm_cls(hidden_size)
        self.use_head = use_head
        if use_head:
            self.head = Siglip2MultiheadAttentionPoolingHead_Src(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                intermediate_size=intermediate_size,
                norm_cls=norm_cls
            )

    def forward(self, x: torch.Tensor, visible_indices = None, mask_ratio=0.5):
        if x.dim() == 4:
            x = x.unsqueeze(2)
        batch, channels, t_frames, height, width = x.shape
        patch_h, patch_w = self.patch_size
        h_patches = height // patch_h
        w_patches = width // patch_w
        total_patches = t_frames * h_patches * w_patches
        device = x.device

        x_2d = x.permute(0,2,1,3,4).reshape(batch * t_frames, channels, height, width)
        feats = self.conv1(x_2d)
        feats = feats.reshape(batch, t_frames, self.hidden_size, h_patches, w_patches).permute(0,1,3,4,2)
        tokens = feats.reshape(batch, total_patches, self.hidden_size)

        if visible_indices is None:
            visible_indices = torch.arange(total_patches, device=device).unsqueeze(0).expand(batch, -1)

        gather_index = visible_indices.unsqueeze(-1).expand(-1, -1, self.hidden_size)
        visible_tokens = torch.gather(tokens, 1, gather_index)

        freqs_full = self.video_rope(t=t_frames, h=h_patches, w=w_patches, device=device, dtype=tokens.dtype)
        freqs_visible = freqs_full[visible_indices]

        x_in = self.ln_pre(visible_tokens).permute(1, 0, 2)
        out = self.transformer(x_in, rotary_pos_emb=freqs_visible)
        out = out.permute(1, 0, 2)

        head_output = None
        if self.use_head:
            out = self.ln_post(out)
            head_output = self.head(out)

        return {"visible_embeddings": out, "head_output": head_output}


# ==============================================================================
# 2. Target HF Model Definitions
# ==============================================================================

from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from transformers.modeling_utils import PreTrainedModel
from transformers.models.siglip.modeling_siglip import SiglipMLP
from transformers.utils import logging

logger = logging.get_logger(__name__)

class HFLlavaViTConfig(PretrainedConfig):
    model_type = "hf_llava_vit"
    def __init__(
        self,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_channels=3,
        image_size=224,
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

def get_norm_layer(config):
    if config.layer_norm_type == "rms_norm":
        return nn.RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
    else:
        return nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

def apply_rotary_pos_emb_hf(q, k, freqs):
    # q, k: (B, H, L, D)
    cos = freqs.cos().unsqueeze(1)
    sin = freqs.sin().unsqueeze(1)

    def rotate_half_hf(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    q_embed = (q * cos) + (rotate_half_hf(q) * sin)
    k_embed = (k * cos) + (rotate_half_hf(k) * sin)
    return q_embed, k_embed

class VideoRotaryEmbeddingSplit466HF(nn.Module):
    def __init__(self, config: HFLlavaViTConfig):
        super().__init__()
        head_dim = config.hidden_size // config.num_attention_heads
        base = config.rope_theta
        self.head_dim = head_dim
        self.half = head_dim // 2

        unit = self.half // 16
        self.t_size = 4 * unit
        self.h_size = 6 * unit
        self.w_size = 6 * unit

        self.register_buffer("inv_freq_t", 1.0 / (base ** (torch.arange(self.t_size, dtype=torch.float32) / self.t_size)), persistent=False)
        self.register_buffer("inv_freq_h", 1.0 / (base ** (torch.arange(self.h_size, dtype=torch.float32) / self.h_size)), persistent=False)
        self.register_buffer("inv_freq_w", 1.0 / (base ** (torch.arange(self.w_size, dtype=torch.float32) / self.w_size)), persistent=False)

    def forward(self, t, h, w, device=None):
        if device is None: device = self.inv_freq_t.device
        inv_t = self.inv_freq_t.to(device)
        inv_h = self.inv_freq_h.to(device)
        inv_w = self.inv_freq_w.to(device)

        ft = torch.outer(torch.arange(t, device=device, dtype=torch.float32), inv_t)
        fh = torch.outer(torch.arange(h, device=device, dtype=torch.float32), inv_h)
        fw = torch.outer(torch.arange(w, device=device, dtype=torch.float32), inv_w)

        t_ids = torch.arange(t, device=device).repeat_interleave(h * w)
        h_ids = torch.arange(h, device=device).repeat_interleave(w).repeat(t)
        w_ids = torch.arange(w, device=device).repeat(h).repeat(t)
        return torch.cat([ft[t_ids], fh[h_ids], fw[w_ids]], dim=-1)

class HFLlavaViTEmbeddings(nn.Module):
    def __init__(self, config: HFLlavaViTConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=False,
        )

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        if pixel_values.dim() == 4: pixel_values = pixel_values.unsqueeze(2)
        batch_size, channels, t_frames, height, width = pixel_values.shape
        x_2d = pixel_values.permute(0, 2, 1, 3, 4).reshape(batch_size * t_frames, channels, height, width)
        embeddings = self.patch_embedding(x_2d)
        embeddings = embeddings.flatten(2).transpose(1, 2)
        total_patches = t_frames * (height // self.config.patch_size) * (width // self.config.patch_size)
        return embeddings.reshape(batch_size, total_patches, self.embed_dim)

class HFLlavaViTAttention(nn.Module):
    def __init__(self, config: HFLlavaViTConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, hidden_states, rotary_pos_emb=None):
        batch_size, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states).view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        if rotary_pos_emb is not None:
            query_states, key_states = apply_rotary_pos_emb_hf(query_states, key_states, rotary_pos_emb)

        attn_weights = (query_states @ key_states.transpose(-2, -1)) * self.scale
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = (attn_weights @ value_states)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(batch_size, q_len, self.embed_dim)
        return self.out_proj(attn_output)

class HFLlavaViTEncoderLayer(nn.Module):
    def __init__(self, config: HFLlavaViTConfig):
        super().__init__()
        self.self_attn = HFLlavaViTAttention(config)
        self.layer_norm1 = get_norm_layer(config)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = get_norm_layer(config)

    def forward(self, hidden_states, rotary_pos_emb=None):
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states, rotary_pos_emb=rotary_pos_emb)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

class HFLlavaViTEncoder(nn.Module):
    def __init__(self, config: HFLlavaViTConfig):
        super().__init__()
        self.layers = nn.ModuleList([HFLlavaViTEncoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, rotary_pos_emb=None):
        for layer in self.layers:
            hidden_states = layer(hidden_states, rotary_pos_emb=rotary_pos_emb)
        return hidden_states

class HFMultiheadAttentionPoolingHead(nn.Module):
    def __init__(self, config: HFLlavaViTConfig):
        super().__init__()
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

class HFLlavaViTModel(PreTrainedModel):
    config_class = HFLlavaViTConfig
    base_model_prefix = "llava_vit"

    def __init__(self, config: HFLlavaViTConfig):
        super().__init__(config)
        self.config = config
        self.embeddings = HFLlavaViTEmbeddings(config)
        self.layernorm_pre = get_norm_layer(config)
        self.encoder = HFLlavaViTEncoder(config)
        self.video_rope = VideoRotaryEmbeddingSplit466HF(config)

        if config.use_head:
             self.layernorm_post = get_norm_layer(config)
             self.head = HFMultiheadAttentionPoolingHead(config)
        else:
             self.layernorm_post = None
             self.head = None

        self.post_init()

    def forward(self, pixel_values: torch.Tensor, visible_indices: Optional[torch.Tensor] = None):
        if pixel_values.dim() == 5:
             t_frames, height, width = pixel_values.shape[2], pixel_values.shape[3], pixel_values.shape[4]
        else:
             t_frames, height, width = 1, pixel_values.shape[2], pixel_values.shape[3]

        hidden_states = self.embeddings(pixel_values)

        if visible_indices is None:
             visible_indices = torch.arange(hidden_states.shape[1], device=pixel_values.device).unsqueeze(0).expand(pixel_values.shape[0], -1)

        gather_index = visible_indices.unsqueeze(-1).expand(-1, -1, self.config.hidden_size)
        hidden_states = torch.gather(hidden_states, 1, gather_index)

        freqs_full = self.video_rope(t_frames, height // self.config.patch_size, width // self.config.patch_size, device=pixel_values.device)
        freqs_visible = freqs_full[visible_indices]
        freqs_visible = torch.cat([freqs_visible, freqs_visible], dim=-1)

        hidden_states = self.layernorm_pre(hidden_states)
        hidden_states = self.encoder(hidden_states, rotary_pos_emb=freqs_visible)

        pooled_output = None
        if self.head is not None:
            head_input = self.layernorm_post(hidden_states)
            pooled_output = self.head(head_input)

        return BaseModelOutputWithPooling(last_hidden_state=hidden_states, pooler_output=pooled_output)


# ==============================================================================
# 3. Conversion and Verification Logic
# ==============================================================================

def convert_llava_vit_checkpoint(source_model, target_model):
    source_state_dict = source_model.state_dict()
    target_state_dict = target_model.state_dict()
    converted_dict = {}
    print(f"Start converting weights...")

    # Embeddings
    converted_dict['embeddings.patch_embedding.weight'] = source_state_dict['conv1.weight']
    converted_dict['layernorm_pre.weight'] = source_state_dict['ln_pre.weight']
    if 'ln_pre.bias' in source_state_dict: converted_dict['layernorm_pre.bias'] = source_state_dict['ln_pre.bias']

    # Encoder Layers
    for i in range(len(source_model.transformer.layers)):
        prefix_src = f'transformer.layers.{i}'
        prefix_tgt = f'encoder.layers.{i}'

        converted_dict[f'{prefix_tgt}.layer_norm1.weight'] = source_state_dict[f'{prefix_src}.ln_1.weight']
        if f'{prefix_src}.ln_1.bias' in source_state_dict: converted_dict[f'{prefix_tgt}.layer_norm1.bias'] = source_state_dict[f'{prefix_src}.ln_1.bias']

        qkv_weight = source_state_dict[f'{prefix_src}.attn.in_proj.weight']
        q_w, k_w, v_w = qkv_weight.chunk(3, dim=0)
        converted_dict[f'{prefix_tgt}.self_attn.q_proj.weight'] = q_w
        converted_dict[f'{prefix_tgt}.self_attn.k_proj.weight'] = k_w
        converted_dict[f'{prefix_tgt}.self_attn.v_proj.weight'] = v_w

        if f'{prefix_src}.attn.in_proj.bias' in source_state_dict:
            qkv_bias = source_state_dict[f'{prefix_src}.attn.in_proj.bias']
            q_b, k_b, v_b = qkv_bias.chunk(3, dim=0)
            converted_dict[f'{prefix_tgt}.self_attn.q_proj.bias'] = q_b
            converted_dict[f'{prefix_tgt}.self_attn.k_proj.bias'] = k_b
            converted_dict[f'{prefix_tgt}.self_attn.v_proj.bias'] = v_b

        converted_dict[f'{prefix_tgt}.self_attn.out_proj.weight'] = source_state_dict[f'{prefix_src}.attn.out_proj.weight']
        if f'{prefix_src}.attn.out_proj.bias' in source_state_dict:
            converted_dict[f'{prefix_tgt}.self_attn.out_proj.bias'] = source_state_dict[f'{prefix_src}.attn.out_proj.bias']

        converted_dict[f'{prefix_tgt}.layer_norm2.weight'] = source_state_dict[f'{prefix_src}.ln_2.weight']
        if f'{prefix_src}.ln_2.bias' in source_state_dict: converted_dict[f'{prefix_tgt}.layer_norm2.bias'] = source_state_dict[f'{prefix_src}.ln_2.bias']

        converted_dict[f'{prefix_tgt}.mlp.fc1.weight'] = source_state_dict[f'{prefix_src}.mlp.0.weight']
        converted_dict[f'{prefix_tgt}.mlp.fc1.bias']   = source_state_dict[f'{prefix_src}.mlp.0.bias']
        converted_dict[f'{prefix_tgt}.mlp.fc2.weight'] = source_state_dict[f'{prefix_src}.mlp.2.weight']
        converted_dict[f'{prefix_tgt}.mlp.fc2.bias']   = source_state_dict[f'{prefix_src}.mlp.2.bias']

    # Head
    if source_model.use_head and target_model.head is not None:
        converted_dict['layernorm_post.weight'] = source_state_dict['ln_post.weight']
        if 'ln_post.bias' in source_state_dict: converted_dict['layernorm_post.bias'] = source_state_dict['ln_post.bias']
        converted_dict['head.probe'] = source_state_dict['head.probe']
        converted_dict['head.attention.in_proj_weight'] = source_state_dict['head.attention.in_proj_weight']
        converted_dict['head.attention.in_proj_bias']   = source_state_dict['head.attention.in_proj_bias']
        converted_dict['head.attention.out_proj.weight'] = source_state_dict['head.attention.out_proj.weight']
        converted_dict['head.attention.out_proj.bias']   = source_state_dict['head.attention.out_proj.bias']
        converted_dict['head.norm.weight'] = source_state_dict['head.norm.weight']
        if 'head.norm.bias' in source_state_dict: converted_dict['head.norm.bias'] = source_state_dict['head.norm.bias']
        converted_dict['head.mlp.fc1.weight'] = source_state_dict['head.mlp.fc1.weight']
        converted_dict['head.mlp.fc1.bias']   = source_state_dict['head.mlp.fc1.bias']
        converted_dict['head.mlp.fc2.weight'] = source_state_dict['head.mlp.fc2.weight']
        converted_dict['head.mlp.fc2.bias']   = source_state_dict['head.mlp.fc2.bias']

    missing, unexpected = target_model.load_state_dict(converted_dict, strict=False)
    real_missing = [k for k in missing if 'inv_freq' not in k] # Ignore RoPE buffers
    if len(real_missing) > 0: print(f"WARNING: Missing: {real_missing}")
    print("Conversion loaded.")
    return target_model

def verify_consistency():
    torch.manual_seed(42)
    np.random.seed(42)

    # 1. Initialize Source Model
    print("Initializing Source Model...")
    src_model = LlavaViTEncoder(
        patch_size=16, hidden_size=768, head_dim=64, num_hidden_layers=2, # Reduced layers for quick check
        intermediate_size=3072, act_layer=nn.GELU, norm_cls=nn.RMSNorm, use_head=True
    )
    src_model.eval()

    # 2. Initialize Target Model
    print("Initializing Target Model...")
    config = HFLlavaViTConfig(
        patch_size=16, hidden_size=768, num_attention_heads=12, num_hidden_layers=2,
        intermediate_size=3072, hidden_act="gelu", layer_norm_type="rms_norm", use_head=True
    )
    tgt_model = HFLlavaViTModel(config)
    tgt_model.eval()

    # 3. Convert
    convert_llava_vit_checkpoint(src_model, tgt_model)

    # 4. Verify
    B, C, T, H, W = 2, 3, 8, 224, 224
    input_video = torch.randn(B, C, T, H, W)

    with torch.no_grad():
        src_out = src_model(input_video)
        tgt_out = tgt_model(input_video)

    feat_diff = (src_out['visible_embeddings'] - tgt_out.last_hidden_state).abs().max()
    head_diff = (src_out['head_output'] - tgt_out.pooler_output).abs().max()

    print(f"Feature Max Diff: {feat_diff.item()}")
    print(f"Head Max Diff: {head_diff.item()}")

    if feat_diff < 1e-4 and head_diff < 1e-4:
        print("✅ PASS: Models match.")
    else:
        print("❌ FAIL: Models do not match.")

if __name__ == "__main__":
    verify_consistency()