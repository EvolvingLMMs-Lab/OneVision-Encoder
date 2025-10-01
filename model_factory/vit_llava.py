import collections.abc
from collections import OrderedDict
from functools import partial
from itertools import repeat
from typing import Callable, Optional
from multiprocessing import Value

import random
import numpy as np
import math
import torch
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torch import nn
from torch.utils.checkpoint import checkpoint

from .registry import MODEL_REGISTRY

# --------------------------------------------------------
# Utility functions for position embeddings
# --------------------------------------------------------

def rotate_half(x):
    """
    Rotate half of the hidden dimensions of the input tensor.
    
    Args:
        x (torch.Tensor): Input tensor
        
    Returns:
        torch.Tensor: Tensor with the second half negated and swapped with the first half
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_vision(tensor: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary positional embeddings to a tensor for vision tasks.
    
    Args:
        tensor (torch.Tensor): Input tensor requiring positional embeddings
        freqs (torch.Tensor): Frequencies for computing cosine and sine values
        
    Returns:
        torch.Tensor: Tensor with rotary positional embeddings applied
    """
    orig_dtype = tensor.dtype
    tensor = tensor.float()
    cos = freqs.cos()
    sin = freqs.sin()
    cos = cos.unsqueeze(1).repeat(1, 1, 2).unsqueeze(0).float()
    sin = sin.unsqueeze(1).repeat(1, 1, 2).unsqueeze(0).float()
    output = (tensor * cos) + (rotate_half(tensor) * sin)
    output = output.to(orig_dtype)
    return output



# --------------------------------------------------------
# Position embedding modules
# --------------------------------------------------------

class VisionRotaryEmbedding(nn.Module):
    """
    Rotary embeddings for vision models.
    
    Computes embeddings through a combination of sequence position and frequency information.
    """
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        """
        Initialize the VisionRotaryEmbedding.
        
        Args:
            dim (int): Embedding dimension size
            theta (float): Parameter controlling frequency range, default is 10000.0
        """
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        """
        Compute rotary embeddings for the given sequence length.
        
        Args:
            seqlen (int): Length of the input sequence
            
        Returns:
            torch.Tensor: Frequency matrix of shape (seqlen, dim/2)
        """
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs


# --------------------------------------------------------
# LayerNorm modules
# --------------------------------------------------------

class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.to(orig_type)

# --------------------------------------------------------
# AttentionPoolingBlock modules
# --------------------------------------------------------

class CrossAttention(nn.Module):
    """
    Cross-attention module for attending between different representations.
    """
    def __init__(
        self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
        proj_drop=0., attn_head_dim=None, out_dim=None):
        super().__init__()
        if out_dim is None:
            out_dim = dim

        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5
        assert all_head_dim == dim

        self.q = nn.Linear(dim, all_head_dim, bias=False)
        self.k = nn.Linear(dim, all_head_dim, bias=False)
        self.v = nn.Linear(dim, all_head_dim, bias=False)

        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.k_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.k_bias = None
            self.v_bias = None
            
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, out_dim)
        self.proj_drop = nn.Dropout(proj_drop) 

    def forward(self, x, k=None, v=None):
        B, N, C = x.shape
        assert k.shape[1] == v.shape[1]

        # Q: 来自 x
        q = self.q(x).view(B, N, self.num_heads, -1).transpose(1, 2)   # [B, h, N, d]

        # K, V: 来自外部输入
        k = self.k(k).view(B, k.shape[1], self.num_heads, -1).transpose(1, 2)  # [B, h, N_k, d]
        v = self.v(v).view(B, v.shape[1], self.num_heads, -1).transpose(1, 2)  # [B, h, N_v, d]

        # 调用 PyTorch 2.0+ 的高效注意力
        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop.p if self.training else 0.0
        )  # [B, h, N, d]

        x = x.transpose(1, 2).reshape(B, N, -1)  # [B, N, h*d]
        return self.proj_drop(self.proj(x))
        # B, N, C = x.shape
        # N_K = k.shape[1]
        # N_V = v.shape[1]
        # assert Nk == Nv

        
        # q_bias, k_bias, v_bias = None, None, None
        # if self.q_bias is not None:
        #     q_bias = self.q_bias
        #     k_bias = self.k_bias
        #     v_bias = self.v_bias

        # q = F.linear(input=x, weight=self.q.weight, bias=q_bias)
        # q = q.reshape(B, N, 1, self.num_heads, -1).permute(2, 0, 3, 1, 4).squeeze(0)  # (B, N_head, N_q, dim)
        
        # k = F.linear(input=k, weight=self.k.weight, bias=k_bias)
        # k = k.reshape(B, N_k, 1, self.num_heads, -1).permute(2, 0, 3, 1, 4).squeeze(0)
        
        # v = F.linear(input=v, weight=self.v.weight, bias=v_bias)
        # v = v.reshape(B, N_v, 1, self.num_heads, -1).permute(2, 0, 3, 1, 4).squeeze(0)

        
        # q = q * self.scale
        # attn = (q @ k.transpose(-2, -1))  # (B, N_head, N_q, N_k)
        
        # attn = attn.softmax(dim=-1)
        # attn = self.attn_drop(attn)
        
        # x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        # x = self.proj(x)
        # x = self.proj_drop(x)

        # return x
    # def __init__(
    #         self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
    #         proj_drop=0., attn_head_dim=None, out_dim=None):
    #     super().__init__()
    #     if out_dim is None:
    #         out_dim = dim
    #     self.num_heads = num_heads
    #     head_dim = dim // num_heads
    #     if attn_head_dim is not None:
    #         head_dim = attn_head_dim
    #     all_head_dim = head_dim * self.num_heads
    #     self.scale = qk_scale or head_dim ** -0.5
    #     assert all_head_dim == dim
        
    #     self.q = nn.Linear(dim, all_head_dim, bias=False)
    #     self.k = nn.Linear(dim, all_head_dim, bias=False)
    #     self.v = nn.Linear(dim, all_head_dim, bias=False)
        
    #     if qkv_bias:
    #         self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
    #         self.k_bias = nn.Parameter(torch.zeros(all_head_dim))
    #         self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
    #     else:
    #         self.q_bias = None
    #         self.k_bias = None
    #         self.v_bias = None
        
    #     self.attn_drop = nn.Dropout(attn_drop)
    #     self.proj = nn.Linear(all_head_dim, out_dim)
    #     self.proj_drop = nn.Dropout(proj_drop)
    
    # def forward(self, x, k=None, v=None):
    #     B, N, C = x.shape
    #     N_k = k.shape[1]
    #     N_v = v.shape[1]
        
    #     q_bias, k_bias, v_bias = None, None, None
    #     if self.q_bias is not None:
    #         q_bias = self.q_bias
    #         k_bias = self.k_bias
    #         v_bias = self.v_bias
        
    #     q = F.linear(input=x, weight=self.q.weight, bias=q_bias)
    #     q = q.reshape(B, N, 1, self.num_heads, -1).permute(2, 0, 3, 1, 4).squeeze(0)  # (B, N_head, N_q, dim)
        
    #     k = F.linear(input=k, weight=self.k.weight, bias=k_bias)
    #     k = k.reshape(B, N_k, 1, self.num_heads, -1).permute(2, 0, 3, 1, 4).squeeze(0)
        
    #     v = F.linear(input=v, weight=self.v.weight, bias=v_bias)
    #     v = v.reshape(B, N_v, 1, self.num_heads, -1).permute(2, 0, 3, 1, 4).squeeze(0)
        
    #     q = q * self.scale
    #     attn = (q @ k.transpose(-2, -1))  # (B, N_head, N_q, N_k)
        
    #     attn = attn.softmax(dim=-1)
    #     attn = self.attn_drop(attn)
        
    #     x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    #     x = self.proj(x)
    #     x = self.proj_drop(x)
        
    #     return x


class AttentiveBlock(nn.Module):
    """Base attention block used for cross-attention operations."""
    def __init__(self, dim, num_heads, qkv_bias=False, qk_scale=None, drop=0, attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, attn_head_dim=None, out_dim=None):
        super().__init__()

        self.norm1_q = norm_layer(dim)
        self.norm1_k = norm_layer(dim)
        self.norm1_v = norm_layer(dim)
        self.cross_attn = CrossAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
            proj_drop=drop, attn_head_dim=attn_head_dim, out_dim=out_dim
        )

        if drop_path > 0.:
            print(f"Use DropPath in projector: {drop_path}")
        self.drop__path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    
    def forward(self, x_q, x_kv, pos_q, pos_k, bool_masked_pos, rel_pos_bias=None):
        x_q = self.norm1_q(x_q + pos_q)
        x_k = self.norm1_k(x_kv + pos_k)
        x_v = self.norm1_v(x_kv)
        x = self.cross_attn(x_q, k=x_k, v=x_v)

        return x



class AttentionPoolingBlock(AttentiveBlock):
    """Attention pooling for aggregating information across tokens."""
    def forward(self, x):
        x_q = x.mean(1, keepdim=True)
        x_kv, pos_q, pos_k = x, 0, 0
        x = super().forward(x_q, x_kv, pos_q, pos_k, bool_masked_pos=None, rel_pos_bias=None)
        x = x.squeeze(1)
        return x 










# --------------------------------------------------------
# VisionSdpaAttention blocks modules
# --------------------------------------------------------

class VisionSdpaAttention(nn.Module):
    """
    Vision-specific scaled dot product attention using PyTorch's native implementation.
    """
    def __init__(self, dim: int, num_heads: int = 16) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.in_proj = nn.Linear(dim, dim * 3, bias=True)
        self.out_proj = nn.Linear(dim, dim)
    
    def forward(
        self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor = None, rotary_pos_emb: torch.Tensor = None
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        batch_size = hidden_states.shape[1]

        # print("hidden_states shape:", hidden_states.shape)
        # Compute Q, K, V matrices
        qkv = self.in_proj(hidden_states)
        # print("qkv", qkv.shape)
        qkv = qkv.view(seq_length, batch_size, 3, self.num_heads, -1)
        # print("qkv after view", qkv.shape)
        qkv = qkv.permute(2, 1, 0, 3, 4)
        q, k, v = qkv.unbind(0)     # batch, seq, numhead, dim

        # print("q shape:", q.shape)
        # print("k shape:", k.shape)
        # print("v shape:", v.shape)

        # Applu rotary postion embeddings
        q = apply_rotary_pos_emb_vision(q, rotary_pos_emb)
        k = apply_rotary_pos_emb_vision(k, rotary_pos_emb)

        # Prepare for scaled dot product attention
        attention_mask = None
        q = q.permute(0, 2, 1, 3).contiguous()
        k = k.permute(0, 2, 1, 3).contiguous()
        v = v.permute(0, 2, 1, 3).contiguous()

        # Compute attention
        # print(q.shape)
        # print("k shape:", k.shape)
        # print("v shape:", v.shape)
        # print("attention_mask shape:", attention_mask.shape if attention_mask is not None else None)
        attn_output = F.scaled_dot_product_attention(q, k ,v, attention_mask, dropout_p=0.0)
        attn_output = attn_output.permute(2, 0, 1, 3).contiguous()
        attn_output = attn_output.view(seq_length, batch_size, -1)
        attn_output = self.out_proj(attn_output)

        return attn_output




# --------------------------------------------------------
# attention blocks modules
# --------------------------------------------------------

class ResidualAttentionBlock(nn.Module):
    """
    Residual attention block with different attention types support.
    """
    def __init__(
            self,
            d_model: int,
            n_head: int,
            mlp_ratio: float = 4.0,
            act_layer: Callable = nn.GELU,
            scale_cosine_attn: bool = False,
            scale_heads: bool = False,
            scale_attn: bool = False,
            scale_fc: bool = False,
            attn_type = 'vision',
            drop_path = 0,
    ):
        super().__init__()
        self.attn_type = attn_type
        self.ln_1 = LayerNorm(d_model)

        # Select attention type
        if attn_type == 'vision':
            self.attn = VisionSdpaAttention(d_model, n_head)
        else:
            self.attn = nn.MultiheadAttention(d_model, n_head)

        self.ln_attn = LayerNorm(d_model) if scale_attn else nn.Identity()
        self.ln_2 = LayerNorm(d_model)
        
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, mlp_width)),
            ('ln', LayerNorm(mlp_width) if scale_fc else nn.Identity()),
            ("gelu", act_layer()),
            ("c_proj", nn.Linear(mlp_width, d_model))
        ]))

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def attention(self, x: torch.Tensor, rotary_pos_emb=None):
        if self.attn_type == 'vision':
            assert rotary_pos_emb is not None
            return self.attn(x, rotary_pos_emb=rotary_pos_emb)
        else:
            return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]


    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, rotary_pos_emb: Optional[torch.Tensor] = None,):
        if rotary_pos_emb is not None:
            x = x + self.drop_path(self.ln_attn(self.attention(self.ln_1(x), rotary_pos_emb=rotary_pos_emb)))
            x = x + self.drop_path(self.mlp(self.ln_2(x)))
        else:
            x = x + self.drop_path(self.ln_attn(self.attention(self.ln_1(x))))
            x = x + self.drop_path(self.mlp(self.ln_2(x)))
        return x




# --------------------------------------------------------
# Transformers modules
# --------------------------------------------------------

class Transformer(nn.Module):
    """
    Standard transformer encoder with support for vision and video attention types.
    """
    def __init__(self, width: int, layers: int, heads: int,  mlp_ratio: float = 4.0, act_layer: Callable = nn.GELU, attn_type='text', drop_path=0):
        super().__init__()
        self.width = width
        self.layers = layers
        self.grad_checkpointing = False
        self.attn_type = attn_type
        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock(
                width, heads, mlp_ratio,
                act_layer=act_layer, attn_type=attn_type,
                drop_path=drop_path
            )
            for _ in range(layers)
        ])
    
    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, rotary_pos_emb: Optional[torch.Tensor] = None):
        if self.attn_type == "vision" or self.attn_type == 'video':
            for r in self.resblocks:
                if self.grad_checkpointing and not torch.jit.is_scripting():
                    x = checkpoint(r, x, attn_mask, rotary_pos_emb)
                else:
                    x = r(x, attn_mask=attn_mask, rotary_pos_emb=rotary_pos_emb)
            return x
        else:
            for r in self.resblocks:
                if self.grad_checkpointing and not torch.jit.is_scripting():
                    x = checkpoint(r, x, attn_mask)
                else:
                    x = r(x, attn_mask=attn_mask)
            return x



# --------------------------------------------------------
# Encoder 
# --------------------------------------------------------

class PretrainEncoder(nn.Module):
    """
    Visual transformer model that supports both image and video processing with temporal awareness.
    """
    def __init__(
            self,
            in_chans: int = 3,
            patch_size: int = 16,
            img_size: int = 224,
            qkv_bias: bool = False,
            drop_path_rate: float = 0.25,
            embed_dim: int = 1408,
            layer_norm: Callable = nn.Identity,
            head_drop_path_rate: float = 0.,
            num_heads: int = 16,
            mlp_ratio: float = 4.3637,
            init_values: float = 1e-5,
            qk_normalization: bool = True,
            depth: int = 40,
            use_flash_attn: bool = False,
            use_fused_rmsnorm: bool = True,
            use_fused_mlp: bool = True,
            fused_mlp_heuristic: int = 1,
            attn_pool_num_heads: int = 16,
            clip_embed_dim: int = 512,
            layerscale_no_force_fp32: bool = False,
            num_frames: int = 16,
            tubelet_size: int = 1,
            sep_pos_embed: bool = False,
            use_checkpoint: bool = False,
            checkpoint_num: int = 0,
            fc_drop_rate: float = 0., 
            num_classes: int = 1000, 
            init_scale: float = 0.001,
            act_layer: Callable = nn.GELU,
            drop_path=0,

            spatial_pred_mask_scale=(0.2, 0.8),
            temporal_pred_mask_scale=(1.0, 1.0),
            aspect_ratio=(0.3, 3.0),
            npred=1,
            max_context_frames_ratio=1.0,
            max_keep=None,
    ):
        super().__init__()


        self.embed_dim = embed_dim
        self.patch_size = to_2tuple(patch_size)
        self.grid_size = (
            num_frames // tubelet_size, 
            img_size // patch_size, 
            img_size // patch_size
        )

        # Patch embedding layers
        self.conv1 = nn.Conv2d(
            in_channels=3, 
            out_channels=embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size, 
            bias=False
        )

        # self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]

        # Core components
        scale = embed_dim ** -0.5
        self.norm = nn.Identity()
        self.class_embedding = nn.Parameter(scale * torch.randn(embed_dim))
        self.ln_pre = LayerNorm(embed_dim)

        # Transformer layers
        # To Do Need To Fix
        self.transformer = Transformer(
            embed_dim, depth, num_heads, mlp_ratio, 
            act_layer=act_layer, attn_type='vision', drop_path=drop_path
        )

        # Check RMSNorm or LayerNorm 
        # Output layers
        self.ln_post = LayerNorm(embed_dim)

        # Position embeddings
        # Need To Do
        self.spatial_merge_size = 1
        self.half_head_dim = embed_dim // num_heads // 2
        self.vision_rotary_embedding = VisionRotaryEmbedding(embed_dim // num_heads)

        # Initialize position embeddings with sinusoidal values
        # trunc_normal_(self.class_pos_emb1, std=.02)
        # trunc_normal_(self.class_pos_emb2, std=.02)
        # trunc_normal_(self.class_pos_emb3, std=.02)
        self.class_pos_emb = nn.Parameter(torch.randn(1, embed_dim // num_heads // 2))
        trunc_normal_(self.class_pos_emb, std=.02)

        self.tubelet_size = tubelet_size
        # self.mask_generator = VjepaMaskGenerator(img_size, num_frames, patch_size, tubelet_size, spatial_pred_mask_scale=(0.2, 0.8), temporal_pred_mask_scale=(1.0, 1.0),
        # aspect_ratio=(0.3, 3.0), npred=1, max_context_frames_ratio=1.0, max_keep=None)

        
        spatial_scale =[(0.15, 0.15), (0.7, 0.7)]
        temporal_scale = [(1.0, 1.0), (1.0, 1.0)]
        aspect_ratio = [(0.75, 1.5), (0.75, 1.5)]
        num_blocks = [8, 2]
        max_temporal_keep = [1.0, 1.0]
        max_keep = [None, None]
        full_complement = [False, False]
        pred_full_complement = [False, False]
        
        self.mask_generator_list = []
        for i in range(len(spatial_scale)):
            mask_generator = VjepaMaskGenerator(
                crop_size=img_size,
                num_frames=num_frames,
                spatial_patch_size=patch_size,
                temporal_patch_size=tubelet_size,
                spatial_pred_mask_scale=spatial_scale[i],
                npred=num_blocks[i],
                max_context_frames_ratio=max_temporal_keep[i],
                max_keep=max_keep[i],
                full_complement=full_complement[i],
                pred_full_complement=pred_full_complement[i],
                # inv_block=inv_block[i]
            )
            self.mask_generator_list.append(mask_generator)
    # 按照多模态大模型的RoPE写的，而不是3D RoPE
    def rot_pos_emb(self, grid_thw, half_head_dim=None):
        t, w, h = grid_thw[0]

        tpos_ids = torch.arange(t)
        tpos_ids = tpos_ids.view(-1, 1).expand(-1, h * w).flatten()

        hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
        hpos_ids = hpos_ids.reshape(
            h // self.spatial_merge_size,
            self.spatial_merge_size,
            w // self.spatial_merge_size,
            self.spatial_merge_size,
        )
        hpos_ids = hpos_ids.permute(0, 2, 1, 3)
        hpos_ids = hpos_ids.flatten()

        wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
        wpos_ids = wpos_ids.reshape(
            h // self.spatial_merge_size,
            self.spatial_merge_size,
            w // self.spatial_merge_size,
            self.spatial_merge_size,
        )
        wpos_ids = wpos_ids.permute(0, 2, 1, 3)
        wpos_ids = wpos_ids.flatten()

        # import pdb; pdb.set_trace()
        pos_ids = torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1)
        max_grid_size = grid_thw[:,:].max()
        rotary_pos_emb_full = self.vision_rotary_embedding(max_grid_size)
        temporal_pos_emb = rotary_pos_emb_full[tpos_ids][:, half_head_dim*3//4:]
        spatial_pos_emb = rotary_pos_emb_full[pos_ids]
        hpos_emb = spatial_pos_emb[:,0,:half_head_dim*3//8]
        wpos_emb = spatial_pos_emb[:,1,half_head_dim*3//8:half_head_dim*3//4]
        rotary_pos_emb = torch.cat([wpos_emb, hpos_emb, temporal_pos_emb], dim=-1)
        return rotary_pos_emb



    def apply_mask_with_i_p_frames(
        self, x, list_I, list_P, mask_generators, patch_size, tubelet_size
    ):
        """
        Apply VjepaMaskGenerator only to P frames while keeping I frames intact.

        Args:
            x: Video tensor of shape (b, c, t, h, w)
            list_I: I-frame indices of shape (b, num_i_frames)
            list_P: P-frame indices of shape (b, num_p_frames)
            mask_generators: list of VjepaMaskGenerator (e.g., [short_mask_gen, long_mask_gen])
            patch_size: Spatial patch size (height, width)
            tubelet_size: Temporal patch size (number of frames per tube)

        Returns:
            final_masks: Binary mask tensor of shape (b, num_patches) where 1 means keep, 0 means masked
            ids_restore: Tensor of shape (b, num_restore_indices)
        """
        b, c, t, h, w = x.shape

        if not isinstance(patch_size, tuple):
            patch_size = (patch_size, patch_size)

        h_patches = h // patch_size[0]
        w_patches = w // patch_size[1]
        t_patches = t // tubelet_size
        total_patches = t_patches * h_patches * w_patches

        final_masks = torch.ones((b, total_patches), dtype=torch.float32, device=x.device)

        # === Step 1: 统一确定 P 帧的 temporal patches 数量 ===
        p_t_patches_list = [list_P[i].shape[0] // tubelet_size for i in range(b)]
        min_p_t_patches = min(p_t_patches_list)

        if min_p_t_patches == 0:
            # 没有有效 P 帧，直接返回全保留
            ids_restore = torch.arange(total_patches, device=x.device).unsqueeze(0).repeat(b, 1)
            return final_masks, ids_restore

        # === Step 2: 随机选择一个 mask 策略，并对整个 batch 生成 mask ===
        mask_gen = random.choice(mask_generators)
        original_duration = mask_gen.duration
        mask_gen.duration = min_p_t_patches
        enc_masks, pred_masks = mask_gen(b)  # 一次性为整个 batch 生成
        mask_gen.duration = original_duration

        # enc_masks: (b, keep_len)
        # pred_masks: (b, pred_len)

        # === Step 3: 映射到 P 帧对应的位置 ===
        for batch_idx in range(b):
            p_frames = list_P[batch_idx]
            p_patch_indices = sorted(list(set([p // tubelet_size for p in p_frames])))
            p_patch_indices = p_patch_indices[:min_p_t_patches]

            # 每个样本的 enc mask → binary mask
            p_frame_mask = torch.zeros((min_p_t_patches * h_patches * w_patches,),
                                    dtype=torch.float32, device=x.device)
            p_frame_mask[enc_masks[batch_idx]] = 1

            for i, p_patch_idx in enumerate(p_patch_indices):
                start_idx = p_patch_idx * h_patches * w_patches
                end_idx = start_idx + h_patches * w_patches
                p_mask_start = i * h_patches * w_patches
                p_mask_end = p_mask_start + h_patches * w_patches
                final_masks[batch_idx, start_idx:end_idx] = p_frame_mask[p_mask_start:p_mask_end]

            # I 帧保持可见
            i_frames = list_I[batch_idx]
            i_patch_indices = [i // tubelet_size for i in i_frames if i // tubelet_size < t_patches]
            for i_patch_idx in set(i_patch_indices):
                start_idx = i_patch_idx * h_patches * w_patches
                end_idx = start_idx + h_patches * w_patches
                final_masks[batch_idx, start_idx:end_idx] = 1

        # === Step 4: 构造 ids_restore（保证 batch 内对齐） ===
        masked_indices, kept_indices, ids_restore = [], [], []
        num_keep_per_sample = final_masks.sum(dim=1).to(torch.int64)
        # min_keep = int(num_keep_per_sample.min().item())

        for b0 in range(b):
            kept_idx = (final_masks[b0] == 1).nonzero(as_tuple=False).squeeze(-1)
            masked_idx = (final_masks[b0] == 0).nonzero(as_tuple=False).squeeze(-1)

            kept_idx = kept_idx[:]  # 截断到最小
            id_restore = torch.cat([masked_idx, kept_idx])
            id_restore = id_restore - 256
            id_restore = id_restore[id_restore >= 0]
            ids_restore.append(id_restore)

        ids_restore = torch.stack(ids_restore, dim=0)

        return final_masks, ids_restore



    # def apply_masks(self, x, masks):
    #     # x: (b, n, d)
    #     # masks: (b, n), 1 is keep, 0 is remove
    #     all_x = []
    #     for m in masks:
    #         mask_keep = m.unsqueeze(-1, repeat(1, 1, x.size(-1)))
    #         all_x += [torch.gather(x, dim=1, index=mask_keep)]
    #     return torch.cat(all_x, dim=0)

    def apply_masks(self, x, masks):
        # x: (b, n, d)
        # masks: (b, n), 1=keep, 0=remove
        all_x = []
        for i in range(x.size(0)):  # 遍历 batch
            m = masks[i].bool()     # (n,) 转为布尔 mask
            x_keep = x[i][m]        # 选出保留的 token, shape (k, d), k = 保留数
            all_x.append(x_keep)
        return torch.cat(all_x, dim=0)
    
    def apply_masks_batch(self, x, masks):
        # x: (b, n, d)
        # masks: (b, n), 1=keep, 0=remove
        # 假设每个样本保留数量相同
        b = x.size(0)
        masks = masks.bool()
        n1 = masks.sum(dim=1)[0]  # 取第一个样本的保留数
        assert masks.sum(dim=1).eq(n1).all(), "每个 batch 保留的 token 数量必须相同！"
        return x[masks].view(b, n1.item(), -1)  # (b, n1, d)

    def apply_mask_to_rotary_pos_emb(self, rotary_pos_emb, masks):
        """
        rotary_pos_emb: (n, d)
        masks: (b, n) —— 所有样本 mask 相同
        输出: (n1, d) —— 只选一次！
        """
        # 只取第一个样本的 mask（因为都一样）
        m = masks[0].bool()  # (n,)
        pos_keep = rotary_pos_emb[m]  # (n1, d)
        return pos_keep

    def forward(self, x: torch.Tensor, list_I = None, list_P = None):
        """
        Forward pass for the visual transformer.

        Args:
            x: Input tensor (image or video)
            twh: Optional tuple of (time, width, height) dimensions
            list_I: Optional list of I frames (b, num_i_frames)
            list_P: Optional list of P frames (b, num_p_frames)

        Returns:
            Output tensor from the model and ids_restore
        """
        # import pdb; pdb.set_trace()
        # print("list-I", list_I)
        # print("list-I", list_I)

        # Determine the spatial dimensions based on inputs
        if len(x.shape) == 4: # Image Input
            twh = (1, x.size(3) // self.patch_size[0], x.size(2) // self.patch_size[1])
        else:
            twh = (x.size(2), x.size(4) // self.patch_size[0], x.size(3) // self.patch_size[1])
        t, _, _ = twh
        rotary_pos_emb = self.rot_pos_emb(torch.tensor([twh], device=x.device), self.half_head_dim)
        # print("rotary_pos_emb before mask", rotary_pos_emb.shape)
        mask, ids_restore = self.apply_mask_with_i_p_frames(x, list_I, list_P, self.mask_generator_list, self.patch_size, self.tubelet_size)

        # Patch embedding for images and videos
        if len(x.shape) == 4:      # x (b, c, h, w)
            x = self.conv1(x)
            x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
            x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        elif len(x.shape) == 5:    # x (b, c, t, h, w)
            b, c, t, h, w = x.shape
            x = x.reshape(b*t, c, h, w)
            x = self.conv1(x)
            x = x.reshape(b, -1, x.shape[1]) # x (b, num_patches, embed_dim)

        h_patches = h // self.patch_size[0]
        w_patches = w // self.patch_size[1]
        t_patches = t // self.tubelet_size

        patches_per_frame = h_patches * w_patches
        total_patches = patches_per_frame * t_patches
        # mask_frame = torch.zeros(b, total_patches, dtype=torch.bool, device=x.device)

        # if list_I is not None:
        #     for batch_idx in range(b):
        #         for i_frame in list_I[batch_idx]:
        #             start_idx = i_frame * patches_per_frame
        #             end_idx = start_idx + patches_per_frame
        #             if end_idx <= total_patches:
        #                 mask_frame[batch_idx, start_idx:end_idx] = True


        # Create masks based on I and P frame indices
        if mask is not None:
            mask = mask[0].unsqueeze(0).repeat(mask.size(0), 1)
            ids_restore = ids_restore[0].unsqueeze(0).repeat(ids_restore.size(0), 1)
            x = self.apply_masks_batch(x, mask)
            rotary_pos_emb = self.apply_mask_to_rotary_pos_emb(rotary_pos_emb, mask)
            # mask_frame = self.apply_masks_batch(mask_frame, mask)

        # import pdb; pdb.set_trace()
        # 加上cls token
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        rotary_pos_emb = torch.cat(([self.class_pos_emb, rotary_pos_emb]), dim=0)


        # Apply pre-normalization
        x = self.ln_pre(x)

        # print("x", x.shape)
        # print("rotary_pos_emb", rotary_pos_emb.shape)
        # run through transformers
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, rotary_pos_emb=rotary_pos_emb)
        x = x.permute(1, 0, 2)  # LND -> NLD

        B, N, D = x.shape
        # 取值
        # i_mask = mask_frame.squeeze(-1).bool()    
        # # i_mask = mask_frame.bool()    # (B, N)，True 表示 I 帧
        # p_mask = ~i_mask              # (B, N)，True 表示 P 帧

        # 直接用布尔 mask 取
        cls_token = x[:, :1, :]
        x = x[:, 1:, :]
        x_unmask   = x[:,:256].reshape(B, -1, D)   # I帧 (B, Ni, D)
        # print("x_unmask", x_unmask.shape)
        x_mask = x[:,256:].reshape(B, -1, D) 
        # print("x_mask", x_mask.shape)
        x_mask = torch.cat([cls_token, x_mask], dim=1)
        return {
                "unmasked_embeddings": x_unmask,
                "masked_embeddings": x_mask,
            }, ids_restore

# --------------------------------------------------------
# Decoder
# --------------------------------------------------------

class PretrainDecoder(nn.Module):
    """
    Visual transformer model that supports both image and video processing with temporal awareness.
    """
    def __init__(
            self,
            in_chans: int = 3,
            patch_size: int = 16,
            img_size: int = 224,
            qkv_bias: bool = True,
            drop_path_rate: float = 0.25,
            pre_embed_dim: int = 384,
            embed_dim: int = 384,
            out_embed_dim: int = 384,
            layer_norm: Callable = nn.Identity,
            head_drop_path_rate: float = 0.,
            num_heads: int = 16,
            mlp_ratio: float = 4.3637,
            init_values: float = 1e-5,
            qk_normalization: bool = True,
            depth: int = 40,
            use_flash_attn: bool = False,
            use_fused_rmsnorm: bool = True,
            use_fused_mlp: bool = True,
            fused_mlp_heuristic: int = 1,
            attn_pool_num_heads: int = 16,
            clip_embed_dim: int = 512,
            layerscale_no_force_fp32: bool = False,
            num_frames: int = 16,
            tubelet_size: int = 1,
            sep_pos_embed: bool = False,
            use_checkpoint: bool = False,
            checkpoint_num: int = 0,
            fc_drop_rate: float = 0., 
            num_classes: int = 1000, 
            init_scale: float = 0.001,
            act_layer: Callable = nn.GELU,
            drop_path=0,
            predictor_embed_dim=384,
    ):
        super().__init__()


        self.embed_dim = embed_dim
        self.patch_size = to_2tuple(patch_size)
        self.grid_size = (
            num_frames // tubelet_size, 
            img_size // patch_size, 
            img_size // patch_size
        )

        self.token_per_frame = img_size // patch_size * img_size // patch_size
        # Patch embedding layers
        self.conv1 = nn.Conv2d(
            in_channels=3, 
            out_channels=embed_dim, 
            kernel_size=patch_size,  
            stride=patch_size, 
            bias=False
        )

        self.proj_embed = nn.Linear(out_embed_dim, embed_dim, bias=True)

        # self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]

        # Core components
        scale = embed_dim ** -0.5
        self.norm = nn.Identity()
        # self.class_embedding = nn.Parameter(scale * torch.randn(embed_dim))
        self.ln_pre = LayerNorm(embed_dim)

        # Transformer layers
        # To Do Need To Fix
        self.transformer = Transformer(
            embed_dim, depth, num_heads, mlp_ratio, 
            act_layer=act_layer, attn_type='vision', drop_path=drop_path
        )

        # Check RMSNorm or LayerNorm 
        # Output layers
        self.ln_post = LayerNorm(embed_dim)

        self.predictor_proj = nn.Linear(embed_dim, out_embed_dim, bias=True)


        # Position embeddings
        # Need To Do
        self.spatial_merge_size = 1
        self.half_head_dim = embed_dim // num_heads // 2
        self.vision_rotary_embedding = VisionRotaryEmbedding(embed_dim // num_heads)

        # Initialize position embeddings with sinusoidal values
        # trunc_normal_(self.class_pos_emb1, std=.02)
        # trunc_normal_(self.class_pos_emb2, std=.02)
        # trunc_normal_(self.class_pos_emb3, std=.02)
        self.mask_token = nn.Parameter(scale * torch.randn(embed_dim))
        # trunc_normal_(self.mask_token, std=.02)
        self.class_pos_emb = nn.Parameter(torch.randn(1, embed_dim // num_heads // 2))
        trunc_normal_(self.class_pos_emb, std=.02)

    # 按照多模态大模型的RoPE写的，而不是3D RoPE
    def rot_pos_emb(self, grid_thw, half_head_dim=None):
        t, w, h = grid_thw[0]

        tpos_ids = torch.arange(t)
        tpos_ids = tpos_ids.view(-1, 1).expand(-1, h * w).flatten()

        hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
        hpos_ids = hpos_ids.reshape(
            h // self.spatial_merge_size,
            self.spatial_merge_size,
            w // self.spatial_merge_size,
            self.spatial_merge_size,
        )
        hpos_ids = hpos_ids.permute(0, 2, 1, 3)
        hpos_ids = hpos_ids.flatten()

        wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
        wpos_ids = wpos_ids.reshape(
            h // self.spatial_merge_size,
            self.spatial_merge_size,
            w // self.spatial_merge_size,
            self.spatial_merge_size,
        )
        wpos_ids = wpos_ids.permute(0, 2, 1, 3)
        wpos_ids = wpos_ids.flatten()

        # import pdb; pdb.set_trace()
        pos_ids = torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1)
        max_grid_size = grid_thw[:,:].max()
        rotary_pos_emb_full = self.vision_rotary_embedding(max_grid_size)
        temporal_pos_emb = rotary_pos_emb_full[tpos_ids][:, half_head_dim*3//4:]
        spatial_pos_emb = rotary_pos_emb_full[pos_ids]
        hpos_emb = spatial_pos_emb[:,0,:half_head_dim*3//8]
        wpos_emb = spatial_pos_emb[:,1,half_head_dim*3//8:half_head_dim*3//4]
        rotary_pos_emb = torch.cat([wpos_emb, hpos_emb, temporal_pos_emb], dim=-1)
        return rotary_pos_emb

    def apply_masks(self, x, masks):
        # x: (b, n, d)
        # mask: (b, n), 1 is keep, 0 is remove
        all_x = []
        for m in masks:
            mask_keep = m.unsqueeze(-1, repeat(1, 1, x.size(-1)))
            all_x += [torch.gather(x, dim=1, index=mask_keep)]
        return torch.cat(all_x, dim=0)

    def forward(self, x: torch.Tensor, twh = None, ids_restore = None):
        """
        Forward pass for the visual transformer.
        
        Args:
            x: Input tensor (image or video)
            twh: Optional tuple of (time, width, height) dimensions
            mask: Optional attention mask
            ids_restore: index to restore the original ordering after masking, shape (B, N_total)
        Returns:
            Output tensor from the model
        """

        # import pdb; pdb.set_trace()
        x = self.proj_embed(x)  # (B, N, D)
        cls_token = x[:, :1, :]

        x = x[:, 1:, :]

        # x.shape
        
        B, N, D = x.shape

        N_total = ids_restore.shape[1]
        N_mask = N_total - N
        # TODO add N_mask
        # import pdb; pdb.set_trace()
        mask_tokens = self.mask_token.expand(B, N_mask, D)
        # 拼接 visible tokens 和 mask tokens
        x_ = torch.cat([x, mask_tokens], dim=1)  
        x = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, D))
        
        # TODO 这里要reshape成 (B, num_frames, ph， pw, D)
        h = int(self.token_per_frame ** 0.5)
        w = int(self.token_per_frame ** 0.5)
        if x.shape[1] // self.token_per_frame == 1:
            x = x.reshape(B, h, w, D)
        else:
            x = x.reshape(B, -1, h, w, D)
        # Determine the spatial dimensions based on inputs
        if len(x.shape) == 4: # Image Input
            twh = (1, x.size(3), x.size(1))
            # tokens_per_frame = x.size(3) // self.patch_size[0] * (x.size(2) // self.patch_size[1])
        else:
            twh = (x.size(1), x.size(3), x.size(2))
        t, _, _ = twh
        # print(t, w, h)
        rotary_pos_emb = self.rot_pos_emb(torch.tensor([twh], device=x.device), self.half_head_dim)
        # print("otary_pos_emb before", rotary_pos_emb.shape)
        x = x.reshape(B, -1, D)
        # 加上cls token
        x = torch.cat([cls_token, x], dim=1)
        rotary_pos_emb = torch.cat(([self.class_pos_emb, rotary_pos_emb]), dim=0)

        # Apply pre-normalization
        x = self.ln_pre(x)

        # print(rotary_pos_emb.shape)
        # print(x.shape)

        # run th rough transformers
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, rotary_pos_emb=rotary_pos_emb)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = x[:,  1:, :]
        mask_pos = (ids_restore >= N)
        x = x[mask_pos].view(B, N_mask, D)

        x = self.predictor_proj(x)
        return x, mask_pos



class VjepaMaskGenerator(object):

    def __init__(
        self,
        crop_size=(224, 224),
        num_frames=16,
        spatial_patch_size=(16, 16),
        temporal_patch_size=2,
        spatial_pred_mask_scale=(0.2, 0.8),
        temporal_pred_mask_scale=(1.0, 1.0),
        aspect_ratio=(0.3, 3.0),
        npred=1,
        max_context_frames_ratio=1.0,
        max_keep=None,
        full_complement=False,
        pred_full_complement=False,
    ):
        super(VjepaMaskGenerator, self).__init__()
        if not isinstance(crop_size, tuple):
            crop_size = (crop_size,) * 2
        if not isinstance(spatial_patch_size, tuple):
            spatial_patch_size = (spatial_patch_size,) * 2
        self.crop_size = crop_size
        self.height, self.width = [crop_size[i] // spatial_patch_size[i] for i in (0, 1)]
        self.duration = num_frames // temporal_patch_size
        self.full_complement = full_complement
        self.pred_full_complement = pred_full_complement

        self.spatial_patch_size = spatial_patch_size
        self.temporal_patch_size = temporal_patch_size

        self.aspect_ratio = aspect_ratio
        self.spatial_pred_mask_scale = spatial_pred_mask_scale
        self.temporal_pred_mask_scale = temporal_pred_mask_scale
        self.npred = npred
        self.max_context_duration = max(
            1, int(self.duration * max_context_frames_ratio)
        )  # maximum number of time-steps (frames) spanned by context mask
        self.max_keep = max_keep  # maximum number of patches to keep in context
        self._itr_counter = Value("i", -1)  # collator is shared across worker processes

    def step(self):
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            v = i.value
        return v

    def _sample_block_size(self, generator, temporal_scale, spatial_scale, aspect_ratio_scale):
        # -- Sample temporal block mask scale
        _rand = torch.rand(1, generator=generator).item()
        min_t, max_t = temporal_scale
        temporal_mask_scale = min_t + _rand * (max_t - min_t)
        t = max(1, int(self.duration * temporal_mask_scale))

        # -- Sample spatial block mask scale
        _rand = torch.rand(1, generator=generator).item()
        min_s, max_s = spatial_scale
        spatial_mask_scale = min_s + _rand * (max_s - min_s)
        spatial_num_keep = int(self.height * self.width * spatial_mask_scale)

        # -- Sample block aspect-ratio
        _rand = torch.rand(1, generator=generator).item()
        min_ar, max_ar = aspect_ratio_scale
        aspect_ratio = min_ar + _rand * (max_ar - min_ar)

        # -- Compute block height and width (given scale and aspect-ratio)
        h = int(round(math.sqrt(spatial_num_keep * aspect_ratio)))
        w = int(round(math.sqrt(spatial_num_keep / aspect_ratio)))
        h = min(h, self.height)
        w = min(w, self.width)

        return (t, h, w)

    def _sample_block_mask(self, b_size):
        t, h, w = b_size
        top = torch.randint(0, self.height - h + 1, (1,))
        left = torch.randint(0, self.width - w + 1, (1,))
        start = torch.randint(0, self.duration - t + 1, (1,))

        mask = torch.ones((self.duration, self.height, self.width), dtype=torch.int32)
        mask[start : start + t, top : top + h, left : left + w] = 0

        # Context mask will only span the first X frames
        # (X=self.max_context_frames)
        if self.max_context_duration < self.duration:
            mask[self.max_context_duration :, :, :] = 0

        # --
        return mask

    def __call__(self, batch_size):
        """
        Create encoder and predictor masks when collating imgs into a batch
        # 1. sample pred block size using seed
        # 2. sample several pred block locations for each image (w/o seed)
        # 3. return pred masks and complement (enc mask)
        """
        seed = self.step()
        g = torch.Generator()
        g.manual_seed(seed)
        p_size = self._sample_block_size(
            generator=g,
            temporal_scale=self.temporal_pred_mask_scale,
            spatial_scale=self.spatial_pred_mask_scale,
            aspect_ratio_scale=self.aspect_ratio,
        )

        collated_masks_pred, collated_masks_enc = [], []
        min_keep_enc = min_keep_pred = self.duration * self.height * self.width
        for _ in range(batch_size):

            empty_context = True
            while empty_context:

                mask_e = torch.ones((self.duration, self.height, self.width), dtype=torch.int32)
                for _ in range(self.npred):
                    mask_e *= self._sample_block_mask(p_size)
                mask_e = mask_e.flatten()

                mask_p = torch.argwhere(mask_e == 0).squeeze()
                mask_e = torch.nonzero(mask_e).squeeze()

                empty_context = len(mask_e) == 0
                if not empty_context:
                    min_keep_pred = min(min_keep_pred, len(mask_p))
                    min_keep_enc = min(min_keep_enc, len(mask_e))
                    collated_masks_pred.append(mask_p)
                    collated_masks_enc.append(mask_e)

        if self.max_keep is not None:
            min_keep_enc = min(min_keep_enc, self.max_keep)

        collated_masks_enc = [cm[:min_keep_enc] for cm in collated_masks_enc]
        collated_masks_pred = [cm[:min_keep_pred] for cm in collated_masks_pred]
        if self.full_complement:  # predictor mask is just complement of encoder mask
            collated_masks_pred = [
                torch.tensor(
                    sorted(list(set(range(int(self.duration * self.height * self.width))) - set(cm.tolist()))),
                    dtype=cm.dtype,
                )
                for cm in collated_masks_enc
            ]
        elif self.pred_full_complement:
            collated_masks_enc = [
                torch.tensor(
                    sorted(list(set(range(int(self.duration * self.height * self.width))) - set(cm.tolist()))),
                    dtype=cm.dtype,
                )
                for cm in collated_masks_pred
            ]

        collated_masks_enc = torch.utils.data.default_collate(collated_masks_enc)
        collated_masks_pred = torch.utils.data.default_collate(collated_masks_pred)

        return collated_masks_enc, collated_masks_pred


# print("hello world")
        

@MODEL_REGISTRY.register()
def PretrainEncoder_small_patch16_224_v0():

    model = PretrainEncoder(
        img_size=224,
        patch_size=14,
        embed_dim=384, 
        depth=12,
        num_heads=6,
        mlp_ratio=4, 
        attn_pool_num_heads=16,
        clip_embed_dim=512,
        num_frames=16,
        num_classes=512,
        use_flash_attn=False,
        use_fused_mlp=True,
        use_fused_rmsnorm=True
    )
    return model


@MODEL_REGISTRY.register()
def PretrainDecoder_small_patch16_224_v0():
    # print("hello world")
    model = PretrainDecoder(
        img_size=224,
        patch_size=14,
        embed_dim=192,
        out_embed_dim=384, 
        depth=12,
        num_heads=3,
        mlp_ratio=4, 
        attn_pool_num_heads=16,
        clip_embed_dim=512,
        num_frames=16,
        num_classes=512,
        use_flash_attn=False,
        use_fused_mlp=True,
        use_fused_rmsnorm=True
    )
    return model

# @MODEL_REGISTRY.register()
# def PretrainDecoder_small_patch16_224_v0():
#     model = PretrainDecoder(
#         img_size=224,
#         patch_size=14,
#         embed_dim=384,
#         out_embed_dim=384, 
#         depth=6,
#         num_heads=6,
#         mlp_ratio=4, 
#         attn_pool_num_heads=16,
#         clip_embed_dim=512,
#         num_frames=16,
#         num_classes=512,
#         use_flash_attn=False,
#         use_fused_mlp=True,
#         use_fused_rmsnorm=True
#     )
#     return model


# model = PretrainEncoder_small_patch16_224_v0()
# input_x = torch.ones((2, 3, 16, 224, 224))
# B, _, T, _, _ = input_x.shape
# device = input_x.device

# k = 4  # 每个样本随机选择 4 个 I-帧

# list_I = torch.empty((B, k), dtype=torch.long, device=device)
# list_P = torch.empty((B, T - k), dtype=torch.long, device=device)

# for b in range(B):
#     i_idx = torch.randperm(T, device=device)[:k]
#     i_idx, _ = torch.sort(i_idx)            # 可选：排序，便于阅读
#     list_I[b] = i_idx

#     mask = torch.ones(T, dtype=torch.bool, device=device)
#     mask[i_idx] = False
#     list_P[b] = torch.arange(T, device=device)[mask]

# output = model(input_x, list_I, list_P)
# print(output)