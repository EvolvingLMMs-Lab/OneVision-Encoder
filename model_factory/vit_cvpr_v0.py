import collections.abc
from collections import OrderedDict
from functools import partial
from itertools import repeat
from typing import Callable, Optional

import numpy as np
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
# UniViT 
# --------------------------------------------------------

class VisualTransformer(nn.Module):
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
            drop_path=0
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
        self.proj = nn.Parameter(scale * torch.randn(embed_dim, num_classes))

        # CLIP-like attention pooling projector
        # TO DO Need To Check
        self.clip_projector = AttentionPoolingBlock(
            dim=embed_dim, 
            num_heads=attn_pool_num_heads, 
            qkv_bias=True, 
            qk_scale=None,
            drop=0., 
            attn_drop=0., 
            drop_path=head_drop_path_rate, 
            norm_layer=partial(nn.LayerNorm, eps=1e-5), 
            out_dim=clip_embed_dim
        )

        # Final classification layers
        # Need To Check
        self.fc_norm = nn.LayerNorm(clip_embed_dim)
        self.fc_dropout = nn.Dropout(p=fc_drop_rate) if fc_drop_rate > 0 else nn.Identity()
        self.head = nn.Linear(clip_embed_dim, num_classes)

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



    def forward(self, x: torch.Tensor, twh = None,):
        """
        Forward pass for the visual transformer.
        
        Args:
            x: Input tensor (image or video)
            twh: Optional tuple of (time, width, height) dimensions
            mask: Optional attention mask
            
        Returns:
            Output tensor from the model
        """
        # Determine the spatial dimensions based on inputs
        if len(x.shape) == 4: # Image Input
            twh = (1, x.size(3) // self.patch_size[0], x.size(2) // self.patch_size[1])
        else:
            twh = (x.size(2), x.size(4) // self.patch_size[0], x.size(3) // self.patch_size[1])
        t, _, _ = twh
        rotary_pos_emb = self.rot_pos_emb(torch.tensor([twh], device=x.device), self.half_head_dim)
        rotary_pos_emb = torch.cat(([self.class_pos_emb, rotary_pos_emb]), dim=0)



        # Patch embedding for images and videos
        if len(x.shape) == 4:      # x (b, c, h, w)
            x = self.conv1(x)
            x = x.flatten(2)
        elif len(x.shape) == 5:    # x (b, c, t, h, w)
            b, c, t, h, w = x.shape
            x = x.reshape(b*t, c, h, w)
            x = self.conv1(x)
            x = x.reshape(b, -1, x.shape[1])


        # 加上cls token
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)


        # Apply pre-normalization
        x = self.ln_pre(x)


        # run through transformers
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, rotary_pos_emb=rotary_pos_emb)
        x = x.permute(1, 0, 2)  # LND -> NLD

        embeddings = x.clone()
        # For images (t=1), just use the class tokne? /// or use attentive probe?
        if t == 1:
            # x = self.ln_post(x[:, 0; :])
            x = self.clip_projector(x)
            x = self.head(self.fc_dropout(x))
        else:
            x = self.clip_projector(x)
            x = self.head(self.fc_dropout(x))
        return {
                "head_embeddings": [x],
                "feature_embeddings": [embeddings],
            }



        
@MODEL_REGISTRY.register()
def UniViT_small_patch16_224_v0():
    model = VisualTransformer(
        img_size=224,
        patch_size=16,
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


        


