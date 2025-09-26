import collections.abc
from collections import OrderedDict
from itertools import repeat
from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint
from torchvision.ops.misc import FrozenBatchNorm2d
from .registry import MODEL_REGISTRY
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import numpy as np
import math
from torch.utils.tensorboard import SummaryWriter
import torch.compiler
from torch import distributed
import os


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)

def init_distributed():
    # 检查是否需要启用分布式
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        # 分布式环境
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        rank = int(os.environ.get("RANK", "0"))
        
        if not distributed.is_initialized():
            distributed.init_process_group(backend="nccl")
            torch.cuda.set_device(local_rank)
            
        return local_rank, world_size, rank
    else:
        # 单机环境 - 不初始化分布式
        print("Running in non-distributed mode")
        local_rank = 0
        world_size = 1
        rank = 0
        
        # 设置设备为0号GPU
        torch.cuda.set_device(local_rank)
        
        return local_rank, world_size, rank

# 在创建模型前调用
# local_rank, world_size, rank = init_distributed()

# @torch.compiler.disable()  # 确保此函数不被编译
# def log_token_cosine_distances(tokens, writer, global_step):
#     """
#     Args:
#         tokens: shape (batch_size, num_tokens, token_dim)
#         writer: TensorBoard SummaryWriter
#         global_step: 当前训练步数（用于TensorBoard记录）
#     """
#     batch_size, num_tokens, token_dim = tokens.shape
    
#     # 1. 归一化（cosine相似度需要单位向量）
#     tokens_norm = F.normalize(tokens, p=2, dim=-1)  # shape: (batch_size, num_tokens, token_dim)
    
#     # 对每个样本单独处理
#     all_distances = []
#     for b in range(batch_size):
#         # 只取一个样本的tokens
#         sample_tokens = tokens_norm[b]  # shape: (num_tokens, token_dim)

#         # # 随机采样200个token
#         # sample_tokens = sample_tokens[torch.randperm(sample_tokens.size(0))[:200]]
        
#         # 计算该样本内所有token的相似度
#         sample_sim = torch.mm(sample_tokens, sample_tokens.T)
#         sample_dist = 1 - sample_sim
        
#         # 排除自身比较
#         sample_dist.fill_diagonal_(float('nan'))
        
#         # 收集有效距离
#         dists = sample_dist.flatten()
#         dists = dists[~torch.isnan(dists)]
#         all_distances.append(dists)

#     # 合并所有样本的距离
#     combined_distances = torch.cat(all_distances)
    
#     # 将距离归一化为0-1
#     min_dist = torch.min(combined_distances)
#     max_dist = torch.max(combined_distances)
#     combined_distances = (combined_distances - min_dist) / (max_dist - min_dist)
    
#     # 落桶到1000个桶中，并进行保存
#     hist = torch.histc(combined_distances.float(), bins=1000, min=0, max=1).detach().cpu().int().numpy()
#     # 6. 记录到TensorBoard
#     # writer.add_histogram(
#     #     "token_cosine_distances", 
#     #     combined_distances.detach().cpu().float().numpy(),  # TensorBoard需要numpy数组
#     #     global_step=global_step
#     # )
#     return hist

@torch.compiler.disable()  # 确保此函数不被编译
def log_token_cosine_distances(tokens, writer, global_step):
    """
    Args:
        tokens: shape (batch_size, num_tokens, token_dim)
        writer: TensorBoard SummaryWriter
        global_step: 当前训练步数（用于TensorBoard记录）
    """
    num_tokens, token_dim = tokens.shape
    tokens = tokens.float()
    
    # 1. 归一化（cosine相似度需要单位向量）
    tokens_norm = F.normalize(tokens, p=2, dim=-1)  # shape: (num_tokens, token_dim)
    
    # 对每个样本单独处理
    all_distances = []

    # 只取一个样本的tokens
    sample_tokens = tokens_norm  # shape: (num_tokens, token_dim)

    # 计算该样本内所有token的相似度
    sample_sim = torch.mm(sample_tokens, sample_tokens.T)
    sample_dist = 1 - sample_sim
    
    # 排除自身比较
    sample_dist.fill_diagonal_(float('nan'))
    
    # 收集有效距离
    dists = sample_dist.flatten()
    dists = dists[~torch.isnan(dists)]
    all_distances.append(dists)

    # 合并所有样本的距离
    combined_distances = torch.cat(all_distances)
    
    # 将距离归一化为0-1
    min_dist = torch.min(combined_distances)
    max_dist = torch.max(combined_distances)
    combined_distances = (combined_distances - min_dist) / (max_dist - min_dist)
    
    # 落桶到1000个桶中，并进行保存
    hist = torch.histc(combined_distances.float(), bins=1000, min=0, max=1).int().numpy()
    # 6. 记录到TensorBoard
    # writer.add_histogram(
    #     "token_cosine_distances", 
    #     combined_distances.detach().cpu().float().numpy(),  # TensorBoard需要numpy数组
    #     global_step=global_step
    # )
    return hist

# @torch.compiler.disable()  # 确保此函数不被编译
# def log_token_l_distances(tokens, writer, global_step):
#     """
#     计算同一图像内token之间的欧式距离并记录到TensorBoard
    
#     Args:
#         tokens: shape (batch_size, num_tokens, token_dim)
#         writer: TensorBoard SummaryWriter
#         global_step: 当前训练步数（用于TensorBoard记录）
#     """
#     batch_size, num_tokens, token_dim = tokens.shape
    
#     # 对每个样本单独处理
#     all_distances = []
#     for b in range(batch_size):
#         # 获取单个样本的tokens
#         sample_tokens = tokens[b]  # shape: (num_tokens, token_dim)

#         # # 随机采样200个token
#         # sample_tokens = sample_tokens[torch.randperm(sample_tokens.size(0))[:200]]
        
#         # 计算该样本内所有token的欧式距离
#         sample_dist = torch.cdist(sample_tokens, sample_tokens, p=2)  # (num_tokens, num_tokens)
        
#         # 排除自身比较
#         sample_dist.fill_diagonal_(float('nan'))
        
#         # 收集有效距离
#         dists = sample_dist.flatten()
#         dists = dists[~torch.isnan(dists)]
#         all_distances.append(dists)

#     # 合并所有样本的距离
#     combined_distances = torch.cat(all_distances)

#     # 将距离归一化为0-1
#     min_dist = torch.min(combined_distances)
#     max_dist = torch.max(combined_distances)
#     combined_distances = (combined_distances - min_dist) / (max_dist - min_dist)
#     # 落桶到1000个桶中，并进行保存
#     hist = torch.histc(combined_distances.float(), bins=1000, min=0, max=1).detach().cpu().int().numpy()

#     # writer.add_histogram("token_euclidean_distances", 
#     #                      combined_distances.detach().cpu().float().numpy(),
#     #                      global_step=global_step)

#     return hist

@torch.compiler.disable()  # 确保此函数不被编译
def log_token_l_distances(tokens, writer, global_step):
    """
    计算同一图像内token之间的欧式距离并记录到TensorBoard
    
    Args:
        tokens: shape (batch_size, num_tokens, token_dim)
        writer: TensorBoard SummaryWriter
        global_step: 当前训练步数（用于TensorBoard记录）
    """
    num_tokens, token_dim = tokens.shape
    
    # 对每个样本单独处理
    all_distances = []
    # 获取单个样本的tokens
    sample_tokens = tokens.float()  # shape: (num_tokens, token_dim)

    # # 随机采样200个token
    # sample_tokens = sample_tokens[torch.randperm(sample_tokens.size(0))[:200]]
    
    # 计算该样本内所有token的欧式距离
    sample_dist = torch.cdist(sample_tokens, sample_tokens, p=2)  # (num_tokens, num_tokens)
    
    # 排除自身比较
    sample_dist.fill_diagonal_(float('nan'))
    
    # 收集有效距离
    dists = sample_dist.flatten()
    dists = dists[~torch.isnan(dists)]
    all_distances.append(dists)

    # 合并所有样本的距离
    combined_distances = torch.cat(all_distances)

    # 将距离归一化为0-1
    min_dist = torch.min(combined_distances)
    max_dist = torch.max(combined_distances)
    combined_distances = (combined_distances - min_dist) / (max_dist - min_dist)
    # 落桶到1000个桶中，并进行保存
    hist = torch.histc(combined_distances.float(), bins=1000, min=0, max=1).int().numpy()

    # writer.add_histogram("token_euclidean_distances", 
    #                      combined_distances.detach().cpu().float().numpy(),
    #                      global_step=global_step)

    return hist

def rotate_half(x):
    """
    旋转输入张量的半个隐藏维度。

    参数:
        x (torch.Tensor): 输入张量。

    返回:
        torch.Tensor: 在最后一个维度上将前后一半交换并取负的张量。

    过程:
    1. 将输入张量的最后一个维度分为两半。
    2. 将后一半取负并与前一半交换位置。
    3. 返回合并后的张量。
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
        is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(attn_mask.size(0), attn_mask.size(1), L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), -1e8)
        else:
            attn_bias += attn_mask

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    # 将attn_weight中nan元素替换为0
    attn_weight = torch.where(torch.isnan(attn_weight), torch.zeros_like(attn_weight), attn_weight)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value



def apply_rotary_pos_emb_vision(tensor: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """
    在视觉领域中对给定张量应用旋转位置嵌入。

    参数:
        tensor (torch.Tensor): 需要应用旋转位置嵌入的输入张量。
        freqs (torch.Tensor): 用于计算余弦和正弦的频率值。

    返回:
        torch.Tensor: 应用了旋转位置嵌入的张量。
    
    过程:
    1. 保存输入张量的原始数据类型。
    2. 将张量转换为浮点类型以进行计算。
    3. 计算频率的余弦和正弦值。
    4. 调整余弦和正弦张量的形状以匹配输入张量。
    5. 通过组合余弦和正弦变换的结果，计算输出张量。
    6. 将输出张量转换回原始数据类型。
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


class VisionRotaryEmbedding(nn.Module):
    """
    VisionRotaryEmbedding 类用于计算视觉模型中的旋转嵌入。

    这种嵌入方法通过结合序列位置和频率信息，为视觉模型提供了位置编码的能力。
    它通过计算给定维度的倒数频率来实现位置编码，并在前向传播时根据输入序列长度生成频率矩阵。

    属性:
    inv_freq (torch.Tensor): 计算得到的倒数频率向量，大小为 (dim/2,)。

    方法:
    __init__(dim: int, theta: float = 10000.0):
        初始化 VisionRotaryEmbedding 类，计算倒数频率。

    forward(seqlen: int) -> torch.Tensor:
        根据输入的序列长度返回频率矩阵。
    """
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        """
        初始化 VisionRotaryEmbedding 类。

        参数:
        dim (int): 要嵌入的维度大小。
        theta (float): 控制频率范围的参数，默认为 10000.0。
        """
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        """
        根据给定的序列长度计算旋转嵌入。

        参数:
        seqlen (int): 输入序列的长度。

        返回:
        torch.Tensor: 生成的频率矩阵，大小为 (seqlen, dim/2)。
        """
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs


# if "in_proj_weight" in k:
#     convert[k.replace("in_proj_weight", "qkv.weight")] = v
# elif "in_proj_bias" in k:
#     convert[k.replace("in_proj_bias", "qkv.bias")] = v
# elif "out_proj.weight" in k:
#     convert[k.replace("out_proj.weight", "proj.weight")] = v
# elif "out_proj.bias" in k:
#     convert[k.replace("out_proj.bias", "proj.bias")] = v

class VisionSdpaAttention(nn.Module):
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

        # Compute Q, K, V matrices
        # Shape: [seq_length, batch_size, dim * 3]
        qkv = self.in_proj(hidden_states)
        # [seq_length, batch_size, 3, num_heads, head_dim]
        qkv = qkv.view(seq_length, batch_size, 3, self.num_heads, -1)
        # [3, batch_size, seq_length, num_heads, head_dim]
        qkv = qkv.permute(2, 1, 0, 3, 4)
        # Each of shape: [batch_size, seq_length, num_heads, head_dim]
        q, k, v = qkv.unbind(0)

        q = apply_rotary_pos_emb_vision(q, rotary_pos_emb)
        k = apply_rotary_pos_emb_vision(k, rotary_pos_emb)
        attention_mask = None
        q = q.permute(0, 2, 1, 3).contiguous()
        k = k.permute(0, 2, 1, 3).contiguous()
        v = v.permute(0, 2, 1, 3).contiguous()
        # q (batch_size, num_heads, seq_length, head_dim)
        # k (batch_size, num_heads, seq_length, head_dim)
        # v (batch_size, num_heads, seq_length, head_dim)

        attn_output = F.scaled_dot_product_attention(q, k, v, attention_mask, dropout_p=0.0)
        attn_output = attn_output.permute(2, 0, 1, 3).contiguous()  # [seq_length, batch_size, num_heads, head_dim]
        attn_output = attn_output.view(seq_length, batch_size, -1)  # [seq_length, batch_size, embedding_dim]
        attn_output = self.out_proj(attn_output)
        return attn_output


class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.to(orig_type)


class QuickGELU(nn.Module):
    # NOTE This is slower than nn.GELU or nn.SiLU and uses more GPU memory
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
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

    def attention(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, rotary_pos_emb: Optional[torch.Tensor] = None):
        if self.attn_type == 'vision':
            assert rotary_pos_emb is not None
            return self.attn(x, rotary_pos_emb=rotary_pos_emb)
        else:
            return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]


    def forward(self,
                x: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None,
                rotary_pos_emb: Optional[torch.Tensor] = None):

        if rotary_pos_emb is not None:
            x = x + self.drop_path(self.ln_attn(self.attention(self.ln_1(x), attn_mask=attn_mask, rotary_pos_emb=rotary_pos_emb)))
            x = x + self.drop_path(self.mlp(self.ln_2(x)))
        else:
            x = x + self.drop_path(self.ln_attn(self.attention(self.ln_1(x), attn_mask=attn_mask)))
            x = x + self.drop_path(self.mlp(self.ln_2(x)))
        return x


class Transformer(nn.Module):
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
                drop_path=drop_path)
            for _ in range(layers)
        ])

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, rotary_pos_emb: Optional[torch.Tensor] = None):
        if self.attn_type == 'vision':
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

_hist_cosine = []
_hist_l = []
_full_clas = {}
_clas_cos = {}
_clas_l = {}

def is_feature_in_list(feature, feature_list, threshold=0.99):
    """检查特征向量是否在列表中(基于余弦相似度)"""
    if len(feature_list) == 0:
        return False
    
    # 将特征向量标准化
    feature_norm = F.normalize(feature.unsqueeze(0), p=2, dim=1)
    
    # 将列表中的特征向量堆叠并标准化
    feature_list_tensor = torch.stack([f for f in feature_list])
    list_norm = F.normalize(feature_list_tensor, p=2, dim=1)
    
    # 计算余弦相似度并检查是否有超过阈值的
    similarities = torch.mm(feature_norm, list_norm.T)
    return torch.any(similarities > threshold).item()

class VisualTransformer(nn.Module):
    def __init__(
            self,
            patch_size: int,
            width: int,
            layers: int,
            heads: int,
            mlp_ratio: float,
            output_dim: int,
            act_layer: Callable = nn.GELU,
            drop_path=0,
            save_feat=False,
            is_mlcd=False,
    ):
        super().__init__()
        self.spatial_merge_size = 1

        self.patch_size = to_2tuple(patch_size)
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        self.writer = SummaryWriter(log_dir="runs/cosine_distances")
        self.writer_l = SummaryWriter(log_dir="runs/l_distances")

        scale = width ** -0.5
        self.width = width
        self.class_embedding = nn.Parameter(scale * torch.randn(width))

        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads, mlp_ratio, act_layer=act_layer, attn_type='vision', drop_path=drop_path)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

        self.vision_rotary_embedding = VisionRotaryEmbedding(width // heads // 2)
        self.class_pos_emb = nn.Parameter(torch.randn(1, width // heads // 2))
        self.heads = heads
        self.steps = 1

        if distributed.is_initialized():
            self.rank = distributed.get_rank()
        else:
            self.rank = 0  # 单机模式下默认为0
        self.step_save = 20
        self.save_feat = save_feat
        self.is_mlcd = is_mlcd



    def rot_pos_emb(self, grid_thw):
        pos_ids = []
        for t, h, w in grid_thw:
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
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.vision_rotary_embedding(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb


    def lock(self, unlocked_groups=0, freeze_bn_stats=False):
        assert unlocked_groups == 0, 'partial locking not currently supported for this model'
        for param in self.parameters():
            param.requires_grad = False
    
    def region_attn(self, embeddings, rotary_pos_emb, boxes_attn):
        seq_length = embeddings.shape[0]
        batch_size = embeddings.shape[1]

        # [B, D] 取每个样本的 class token 
        q_clas = embeddings.permute(1, 0, 2)[:,0,:]
        # q_clas = self.clas_in_proj(q_clas)
        # [B, 1, H, Dh]
        q_clas = q_clas.view(batch_size, 1, self.heads, -1)
        # [B, H, 1, L-1] 除了class token剩下的都要
        q_attn_mask = boxes_attn[:,:,0,1:].unsqueeze(2)

        # V = [B, Q, H, Dh] K是v的复制
        v = embeddings.permute(1, 0, 2).view(batch_size, seq_length, self.heads, -1)
        k = v.clone()

        k = apply_rotary_pos_emb_vision(k, rotary_pos_emb)[:, 1:,:,:]
        v = apply_rotary_pos_emb_vision(v, rotary_pos_emb)[:, 1:,:,:]

        q = q_clas.permute(0, 2, 1, 3).contiguous() # [batch_size, num_heads, seq_length, head_dim]
        k = k.permute(0, 2, 1, 3).contiguous()
        v = v.permute(0, 2, 1, 3).contiguous()


        attn_output = scaled_dot_product_attention(q, k, v, q_attn_mask, dropout_p=0.0)
        attn_output = attn_output.permute(2, 0, 1, 3).contiguous()  # [seq_length, batch_size, num_heads, head_dim]
        attn_output = attn_output.view(1, batch_size, -1)
        attn_output = attn_output.permute(1, 0, 2)
        attn_output = attn_output[:,0,:]
        # attn_output = self.attn_out_proj(attn_output)
        return attn_output
    
    def region(self, embeddings, rotary_pos_emb, boxes_attn):
        bs = embeddings.size(0)
        pred_box_num = boxes_attn.size(1)
        x = embeddings.unsqueeze(1).repeat(1, pred_box_num, 1, 1)
        x = x.view(bs * pred_box_num, -1, self.width)
        boxes_attn = boxes_attn.view(bs * pred_box_num, 1, embeddings.shape[1])
        x = x.permute(1, 0, 2).float()
        boxes_attn = boxes_attn.unsqueeze(1).repeat(1, self.heads, 1, 1)
        
        attn_output = self.region_attn(x, rotary_pos_emb, boxes_attn)
        obj_clas = attn_output @ self.proj

        return obj_clas

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=False):
        self.transformer.grad_checkpointing = enable

    def forward(self, x: torch.Tensor, twh = None, boxes_attn = None, list_label=None):
        global _full_clas
        if list_label:
            box_clas = list_label[0][:,:,0]
        if twh is None:
            twh = (1, x.size(3) // self.patch_size[0], x.size(2) // self.patch_size[1])
        # twh = (1, 24, 24)
        rotary_pos_emb = self.rot_pos_emb(torch.tensor([twh], device=x.device))
        rotary_pos_emb = torch.cat([self.class_pos_emb, rotary_pos_emb], dim=0)

        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        # x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, rotary_pos_emb=rotary_pos_emb)
        x = x.permute(1, 0, 2)  # LND -> NLD
        embeddings = x.clone()
        # 当前进程为0时，保存hist
        if self.save_feat and self.rank == 0:
            global _hist_cosine  # 声明使用全局变量
            global _hist_l  # 声明使用全局变量
            # if embeddings.shape[1] > 1 and not torch.isnan(embeddings).any():
            #     _hist_cosine.append(log_token_cosine_distances(embeddings[:,1:,:].clone(), self.writer, global_step=self.steps))
            #     _hist_l.append(log_token_l_distances(embeddings[:,1:,:].clone(), self.writer_l, global_step=self.steps))
            # # 按照steps保存hist，并且每10000步保存一次
            # if self.steps % self.step_save == 0:
            #     np.save(f"/vlm/yinxie/code/img_var/mlcd_s16_variances_cosine_hist_{str(self.steps//self.step_save).zfill(10)}_rank{self.rank}.npy", np.array(_hist_cosine).reshape(self.step_save,-1))
            #     np.save(f"/vlm/yinxie/code/img_var/mlcd_s16_variances_l_hist_{str(self.steps//self.step_save).zfill(10)}_rank{self.rank}.npy", np.array(_hist_l).reshape(self.step_save,-1))
            #     print(f"/vlm/yinxie/code/img_var/mlcd_s16_variances_cosine_hist_{str(self.steps//self.step_save).zfill(10)}_rank{self.rank}.npy saved")
            #     print(f"/vlm/yinxie/code/img_var/mlcd_s16_variances_l_hist_{str(self.steps//self.step_save).zfill(10)}_rank{self.rank}.npy saved")
            #     _hist_cosine = []
            #     _hist_l = []
            if self.steps % self.step_save == 0:
                # 按照类别计算类内token的距离，并保存hist，清空字典
                for clas_index in _full_clas.keys():
                    if _full_clas[clas_index].shape[0] < 2:
                        print(f"clas_index: {clas_index}, _full_clas[clas_index].shape: {_full_clas[clas_index].shape}")
                        continue
                    # 计算余弦相似度
                    cur_hist_cos = log_token_cosine_distances(_full_clas[clas_index], self.writer, global_step=self.steps)
                    if self.is_mlcd:
                        model_name = 'mlcd'
                    else:
                        model_name = 'rice'
                    # 如果文件夹 /vlm/yinxie/code/clas_var/{model_name}/class_{clas_index} 不存在，则创建文件夹
                    if not os.path.exists(f'/vlm/yinxie/code/clas_var/{model_name}/class_{clas_index}'):
                        os.makedirs(f'/vlm/yinxie/code/clas_var/{model_name}/class_{clas_index}')
                    
                    np.save(f"/vlm/yinxie/code/clas_var/{model_name}/class_{clas_index}/mlcd_cosine_{str(self.steps//self.step_save).zfill(10)}_clas{clas_index}.npy", np.array(cur_hist_cos))
                    # print(f"/vlm/yinxie/code/clas_var/{model_name}/class_{clas_index}/mlcd_cosine_{str(self.steps//self.step_save).zfill(10)}_clas{clas_index}.npy saved")
                    # 计算欧式距离
                    # cur_his_l = log_token_l_distances(_full_clas[clas_index], self.writer_l, global_step=self.steps)
                    # np.save(f"/vlm/yinxie/code/clas_var/{model_name}/mlcd_l_{str(self.steps//self.step_save).zfill(10)}_clas{clas_index}.npy", np.array(cur_his_l))
                    # print(f"/vlm/yinxie/code/clas_var/{model_name}/mlcd_l_{str(self.steps//self.step_save).zfill(10)}_clas{clas_index}.npy saved")
                _full_clas = {}

        self.steps += 1

        x = self.ln_post(x[:, 0, :])
        embeddings = self.ln_post(embeddings)

        if self.proj is not None:
            x = x @ self.proj
        
        

        # 这个non_all_ones是什么意思？

        
        if boxes_attn is not None:                                                             # 若提供了框注意力掩码（说明要做区域级特征）
            # get region data indexes
            non_all_ones = []                                                                  # 用来收集需要做 region 的样本下标
            for i in range(boxes_attn.size(0)):                                                # 遍历 batch 维度（假设 batch = B）
                non_all_ones.append(i)                                                         # 这里直接把所有样本都加入（无过滤）
                # if not torch.all(boxes_attn[i].float() == 1.0):
                #     non_all_ones.append(i)
            non_all_ones = torch.tensor(non_all_ones, device=boxes_attn.device)                # 列表→Tensor，并放到相同设备上

            box_embeddings = embeddings[non_all_ones]                                          # 从所有 tokens 的输出中挑出这些样本 
            real_boxes_attn = boxes_attn[non_all_ones]                                         # 形状：embeddings=[B, L, D] → box_embeddings=[M, L, D]（M=len(non_all_ones)）
            
            obj_clas = self.region(box_embeddings, rotary_pos_emb, real_boxes_attn)            # 进入 region 分支（class作query、只对掩码区域tokens做注意力）
                                                                                               # 返回形状：obj_clas=[M*K, output_dim]（每个框一个向量）

            if list_label:                                                                     # 若传入了标签（用于日志统计）
                box_num = box_clas.shape[1]                                                    # 每图框数 K
                box_clas = box_clas.reshape(-1)                                                # 展平成一维，与 obj_clas 的 [M*K] 对齐
                if self.rank == 0:                                                             # rank 0 记录
                    for clas_index, clas_emb in enumerate(obj_clas.clone()):                   # obj是clas_index 和 clas_emb
                        cur_clas = box_clas[clas_index].cpu().int().item()                     # 当前框的类别
                        # 在一个滑动窗口内，若该特征与前面的重复（实现依赖 is_feature_in_list），就跳过
                        if clas_index != 0 and is_feature_in_list(clas_emb, obj_clas[max(0,clas_index-box_num):clas_index-1]):  
                            continue
                        # 把该框特征按类别聚合到全局字典 _full_clas（用于后续定期画直方图）
                        if cur_clas in _full_clas:
                            _full_clas[cur_clas] = torch.cat([_full_clas[cur_clas], clas_emb.unsqueeze(0).cpu()], dim=0)
                        else:
                            _full_clas[cur_clas] = clas_emb.unsqueeze(0).cpu()

            # Find indices where boxes_attn is not all ones
            # 根据non_all_ones的起始和结束位置拼接张量
            start_idx = 0 if non_all_ones[0] == 0 else non_all_ones[0]                         # 要替换/插入的起始索引（batch维上的下标）
            end_idx = x.size(1) if non_all_ones[-1] == x.size(1) - 1 else non_all_ones[-1] + 1 # 终止索引
            
            parts = []                                                    
            if start_idx > 0:                              
                parts.append(x[:start_idx])                                                     # 保留起始前的一段：x[0:start_idx]，形状大致[*, output_dim]
            parts.append(obj_clas)                                                              # 中间放入 obj_clas，形状是 [M*K, output_dim]
            if end_idx < x.size(1):
                parts.append(x[end_idx:])                                                       
            if not self.is_mlcd:
                x = torch.cat(parts, dim=0)

        return x


@MODEL_REGISTRY.register()
class RoPE2d_ViT_B_16_1024_region(VisualTransformer):
    """ViT_B_16_1024
    创建一个带二维旋转位置编码的ViT-B模型。
    使用16x16的patch，12层，12个头，输出维度1024。
    返回:
    VisualTransformer: 配置的ViT模型实例。
    """
    def __init__(self):
        super().__init__(
            patch_size=16, width=768, layers=12,
            heads=12, mlp_ratio=4, output_dim=1024)


@MODEL_REGISTRY.register()
class RoPE2d_ViT_B_14_1024_region(VisualTransformer):
    def __init__(self):
        super().__init__(
            patch_size=14, width=768, layers=12,
            heads=12, mlp_ratio=4, output_dim=1024)


# def ViT_S_14(input_resolution=224, embedding_size=512):
#     vision_transformer = VisualTransformer(
#         image_size=input_resolution, patch_size=14, width=384,
#         layers=12, heads=6, mlp_ratio=4, output_dim=embedding_size)
#     return vision_transformer

@MODEL_REGISTRY.register()
class RoPE2d_ViT_S_14_1024_region(VisualTransformer):
    def __init__(self):
        super().__init__(
            patch_size=14, width=384, layers=12,
            heads=6, mlp_ratio=4, output_dim=1024)


@MODEL_REGISTRY.register()
class RoPE2d_ViT_S_16_512_region(VisualTransformer):
    def __init__(self):
        super().__init__(
            patch_size=16, width=384, layers=12,
            heads=6, mlp_ratio=4, output_dim=512)

@MODEL_REGISTRY.register()
class RoPE2d_ViT_B_16_512_region(VisualTransformer):
    def __init__(self):
        super().__init__(
            patch_size=16, width=768, layers=12,
            heads=12, mlp_ratio=4, output_dim=1024)

@MODEL_REGISTRY.register()
class RoPE2d_ViT_S_16_512_region_save_feat_mlcd(VisualTransformer):
    def __init__(self):
        super().__init__(
            patch_size=16, width=384, layers=12,
            heads=6, mlp_ratio=4, output_dim=512, save_feat=True, is_mlcd=True)

@MODEL_REGISTRY.register()
class RoPE2d_ViT_S_16_512_region_save_feat(VisualTransformer):
    def __init__(self):
        super().__init__(
            patch_size=16, width=384, layers=12,
            heads=6, mlp_ratio=4, output_dim=512, save_feat=True)



# @MODEL_REGISTRY.register()
# class RoPE2d_ViT_B_16_1024_dp01(VisualTransformer):
#     """ViT_B_16_1024
#     创建一个带二维旋转位置编码的ViT-B模型。
#     使用16x16的patch，12层，12个头，输出维度1024。
#     返回:
#     VisualTransformer: 配置的ViT模型实例。
#     """
#     def __init__(self):
#         super().__init__(
#             patch_size=16, width=768, layers=12,
#             heads=12, mlp_ratio=4, output_dim=1024, drop_path=0.1)


@MODEL_REGISTRY.register()
class RoPE2d_ViT_g_32_1024_region(VisualTransformer):
    """ViT_B_16_1024
    创建一个带二维旋转位置编码的ViT-B模型。
    使用16x16的patch，12层，12个头，输出维度1024。
    返回:
    VisualTransformer: 配置的ViT模型实例。
    """
    def __init__(self):
        super().__init__(
            patch_size=32, width=1408, layers=40,
            heads=16, mlp_ratio=4, output_dim=1024)


@MODEL_REGISTRY.register()
class RoPE2d_ViT_g_14_1024_region(VisualTransformer):
    def __init__(self):
        super().__init__(
            patch_size=14, width=1408, layers=40,
            heads=16, mlp_ratio=4.363636363636363, output_dim=1024)


@MODEL_REGISTRY.register()
class RoPE2d_ViT_bigG_14_1024_region(VisualTransformer):
    def __init__(self):
        super().__init__(
            patch_size=14, width=1664, layers=48,
            heads=16, mlp_ratio=4.9231, output_dim=1024)


@MODEL_REGISTRY.register()
class RoPE2d_ViT_SO400M_region(VisualTransformer):
    def __init__(self):
        super().__init__(
            patch_size=14, width=1152, layers=27,
            heads=16, mlp_ratio=3.73612, output_dim=1024)


@MODEL_REGISTRY.register()
class RoPE2d_ViT_L_14_1024_region(VisualTransformer):
    def __init__(self):
        super().__init__(
            patch_size=14, width=1024, layers=24,
            heads=16, mlp_ratio=4, output_dim=1024)


@MODEL_REGISTRY.register()
class RoPE2d_ViT_t_2_1024_region(VisualTransformer):
    def __init__(self):
        super().__init__(
            patch_size=2, width=128, layers=12,
            heads=8, mlp_ratio=4, output_dim=1024)
