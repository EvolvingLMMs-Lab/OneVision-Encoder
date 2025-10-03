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
from timm.models.registry import register_model  # 用timm注册器

from torch import nn
from torch.utils.checkpoint import checkpoint
from layers import VisionSdpaAttention, VisionRotaryEmbedding, CrossAttention, AttentiveBlock, ResidualAttentionBlock, Transformer, VideoRotaryEmbeddingSimple
from torch.nn import LayerNorm


class LlavaViTEncoder(nn.Module):
    def __init__(
        self,
        patch_size=16,
        img_size=224,
        hidden_size=384,
        head_dim=64,
        num_hidden_layers=12,
        intermediate_size=1536,      # 直接传中间层维度 (例如 384 * 4)
        num_frames=1,
        act_layer=nn.GELU,
        num_key_value_heads=None,    # 预留（当前未使用）
        use_gradient_checkpointing=False,
    ):
        super().__init__()
        self.tubelet_size = 1
        assert hidden_size % head_dim == 0, "hidden_size must be divisible by head_dim"
        num_attention_heads = hidden_size // head_dim
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads

        self.patch_size = to_2tuple(patch_size)

        self.grid_size = (
            num_frames // self.tubelet_size,
            img_size // patch_size,
            img_size // patch_size,
        )

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )

        scale = hidden_size ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(hidden_size))
        self.ln_pre = nn.LayerNorm(hidden_size)

        self.transformer = Transformer(
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            act_layer=act_layer,
            gradient_checkpointing=use_gradient_checkpointing,
        )

        self.spatial_merge_size = 1
        self.half_head_dim = head_dim // 2
        self.video_rope = VideoRotaryEmbeddingSimple(head_dim)


    def mask_mae_style(self, batch_size, t_frames, patches_per_frame, mask_ratio, device):
        """
        MAE-style masking.

        Args:
            batch_size: int
            t_frames:   number of frames (T)
            patches_per_frame: number of spatial patches per frame
            device: torch.device

        Returns:
            visible_indices: (batch_size, n_visible) sorted indices kept per sample
            visible_mask:    (batch_size, total_patches) bool, True=visible
            ids_restore:     (batch_size, total_patches) original index -> position in [visible, masked]
        """
        total_patches = t_frames * patches_per_frame

        i_region = torch.arange(0, patches_per_frame, device=device)          # I-frame (frame 0) patches (always kept)
        p_region_indices = torch.arange(patches_per_frame, total_patches, device=device)  # P-region candidates
        p_region_count = p_region_indices.numel()

        p_keep_count = int(round((1 - mask_ratio) * p_region_count))
        p_keep_count = max(0, min(p_keep_count, p_region_count))

        if p_keep_count > 0:
            rand_scores = torch.rand(batch_size, p_region_count, device=device)
            topk_idx = torch.topk(rand_scores, k=p_keep_count, dim=1, largest=True, sorted=False).indices  # (batch_size, p_keep_count)
            p_kept = p_region_indices[topk_idx]  # (batch_size, p_keep_count)
            visible_indices = torch.cat(
                [i_region.unsqueeze(0).expand(batch_size, -1), p_kept],
                dim=1
            )
        else:
            visible_indices = i_region.unsqueeze(0).expand(batch_size, -1)

        visible_indices = torch.sort(visible_indices, dim=1).values            # (batch_size, n_visible)
        n_visible = visible_indices.shape[1]

        visible_mask = torch.zeros(batch_size, total_patches, dtype=torch.bool, device=device)
        visible_mask.scatter_(1, visible_indices, True)

        vis_int = visible_mask.long()
        mask_int = 1 - vis_int
        vis_rank = torch.cumsum(vis_int, dim=1) - 1
        mask_rank = torch.cumsum(mask_int, dim=1) - 1
        n_visible_col = vis_int.sum(dim=1, keepdim=True)
        ids_restore = torch.where(visible_mask, vis_rank, n_visible_col + mask_rank)

        return visible_indices, visible_mask, ids_restore

    def apply_mask_to_rotary_pos_emb(self, rotary_pos_emb, masks):
        return rotary_pos_emb[masks[0].bool()]


    def forward(self, x: torch.Tensor, mask_ratio=0.5):
        """
        Args:
            x: (batch, channels, height, width) or (batch, channels, t_frames, height, width)

        Returns:
            dict with:
            visible_embeddings: (batch, n_visible, hidden_size)
            mask: (batch, total_patches) float32 (1 = visible, 0 = masked)
            ids_restore: (batch, total_patches)
            visible_indices: (batch, n_visible)
            num_visible: int
            full_sequence_length: int (total_patches)
            patch_grid: (t_frames, h_patches, w_patches)
        """
        if x.dim() == 4:
            x = x.unsqueeze(2)  # (b, c, 1, h, w)

        batch, channels, t_frames, height, width = x.shape
        patch_h, patch_w = self.patch_size

        h_patches = height // patch_h
        w_patches = width // patch_w
        patches_per_frame = h_patches * w_patches
        total_patches = t_frames * patches_per_frame
        device = x.device

        # Patchify: (batch, t_frames, hidden, h_patches, w_patches) -> (batch, total_patches, hidden)
        x_2d = x.permute(0, 2, 1, 3, 4).reshape(batch * t_frames, channels, height, width)
        feats = self.conv1(x_2d)  # (batch*t_frames, hidden, h_patches, w_patches)
        feats = feats.reshape(batch, t_frames, self.hidden_size, h_patches, w_patches).permute(0, 1, 3, 4, 2)
        tokens = feats.reshape(batch, total_patches, self.hidden_size)

        # MAE-style masking
        visible_indices, visible_mask_bool, ids_restore = self.mask_mae_style(
            batch_size=batch,
            t_frames=t_frames,
            patches_per_frame=patches_per_frame,
            mask_ratio=mask_ratio,
            device=device
        )  # (batch, n_visible), (batch, total_patches), (batch, total_patches)
        n_visible = visible_indices.shape[1]

        # Gather visible tokens
        gather_index = visible_indices.unsqueeze(-1).expand(-1, -1, self.hidden_size)  # (batch, n_visible, hidden)
        visible_tokens = torch.gather(tokens, 1, gather_index)  # (batch, n_visible, hidden)

        # Build full RoPE and gather per-sample
        freqs_full = self.video_rope(
            t=t_frames // self.tubelet_size,
            h=h_patches,
            w=w_patches,
            device=device,
            dtype=tokens.dtype
        )  # (total_patches, head_dim//2)

        freqs_visible = freqs_full[visible_indices]  # (batch, n_visible, head_dim//2)

        # LayerNorm & shape -> (n_visible, batch, hidden)
        x_in = self.ln_pre(visible_tokens).permute(1, 0, 2)

        # Transformer with batched RoPE (batch, n_visible, d/2)
        out = self.transformer(x_in, rotary_pos_emb=freqs_visible)  # (n_visible, batch, hidden)
        out = out.permute(1, 0, 2)  # (batch, n_visible, hidden)

        return {
            "visible_embeddings": out,
            "mask": visible_mask_bool.float(),
            "ids_restore": ids_restore,
            "visible_indices": visible_indices,
            "num_visible": n_visible,
            "full_sequence_length": total_patches,
            "patch_grid": (t_frames, h_patches, w_patches),
        }


class LlavaViTDecoder(nn.Module):
    """
    Feature-level MAE-style Decoder（与 LlavaViTEncoder 风格统一）:
      - 输入:
          visible_embeddings : (B, N_vis, encoder_hidden_size)  (来自 Encoder 的可见 token 表示)
          ids_restore        : (B, L_full)  (MAE 风格索引, 原序 -> 在 [visible, masked] 拼接序列中的位置)
          mask               : (B, L_full)  (1=visible, 0=masked)
          patch_grid         : (T, h_patches, w_patches)
      - 过程:
          1) 将 encoder 可见特征投影到解码 hidden_size (若维度不同)
          2) 追加可学习 mask token (数量 = 被遮挡 token 数)
          3) 按 ids_restore 还原到原始顺序 (B, L_full, hidden_size)
          4) 生成全长 3D RoPE (共享一份) 并送入 Transformer
          5) 输出:
              decoded_full   : (B, L_full, out_feature_dim)
              decoded_visible: (B, N_vis, out_feature_dim)
              decoded_masked : (B, N_mask, out_feature_dim)
              mask, ids_restore 原样返回
      - 不做像素重建，只输出特征，供外部特征/教师网络监督。
    """
    def __init__(
        self,
        hidden_size=384,              # 解码器内部维度
        encoder_hidden_size=384,      # Encoder 输出维度（若不同将线性投影）
        head_dim=48,
        num_hidden_layers=8,
        intermediate_size=1536,
        act_layer=nn.GELU,
        num_key_value_heads=None,     # 预留；目前同 encoder 仅用全头注意力
        feature_proj_dim=None,        # 最终输出特征维度 (None 则与 hidden_size 一致)
        use_gradient_checkpointing=False,
    ):
        super().__init__()
        assert hidden_size % head_dim == 0, "hidden_size must be divisible by head_dim"
        num_attention_heads = hidden_size // head_dim
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads  # 预留未用

        # 约束：VideoRotaryEmbeddingSimple 要求 (head_dim//2) % 3 == 0
        assert (head_dim // 2) % 3 == 0, "head_dim//2 must be divisible by 3 for 3D RoPE equal split"

        self.hidden_size = hidden_size
        self.encoder_hidden_size = encoder_hidden_size
        self.head_dim = head_dim
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # 输入特征投影
        if encoder_hidden_size != hidden_size:
            self.proj_in = nn.Linear(encoder_hidden_size, hidden_size, bias=True)
        else:
            self.proj_in = nn.Identity()

        # 可学习 mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        trunc_normal_(self.mask_token, std=0.02)

        # RoPE 与 Transformer
        self.video_rope = VideoRotaryEmbeddingSimple(head_dim)
        self.ln_in = nn.LayerNorm(hidden_size)

        self.transformer = Transformer(
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            act_layer=act_layer,
            gradient_checkpointing=use_gradient_checkpointing,
        )

        # 最终输出特征投影（可对齐 teacher 维度）
        if feature_proj_dim is not None and feature_proj_dim != hidden_size:
            self.feature_head = nn.Linear(hidden_size, feature_proj_dim, bias=True)
            self.out_feature_dim = feature_proj_dim
        else:
            self.feature_head = nn.Identity()
            self.out_feature_dim = hidden_size

    def forward(
        self,
        visible_embeddings: torch.Tensor,  # (B, N_vis, encoder_hidden_size)
        ids_restore: torch.Tensor,         # (B, L_full)
        mask: torch.Tensor,                # (B, L_full) 1=visible 0=masked (float 或 bool)
        patch_grid,                        # (T, h_patches, w_patches)
    ):
        """
        Returns:
            {
                "decoded_full":   (B, L_full, D_out),
                "decoded_visible":(B, N_vis, D_out),
                "decoded_masked": (B, N_mask, D_out),
                "mask":           (B, L_full) (float, 与输入一致类型),
                "ids_restore":    (B, L_full)
            }
        """
        if mask.dtype != torch.bool:
            mask_bool = mask.bool()
        else:
            mask_bool = mask

        B, N_vis, _ = visible_embeddings.shape
        L_full = ids_restore.shape[1]
        T, h_patches, w_patches = patch_grid
        assert L_full == T * h_patches * w_patches, "ids_restore length mismatch patch grid"

        # 投影到 decoder hidden
        vis_dec = self.proj_in(visible_embeddings)  # (B,N_vis,H)

        # 准备 mask tokens
        N_mask = L_full - N_vis
        mask_tokens = self.mask_token.expand(B, N_mask, self.hidden_size)  # (B,N_mask,H)

        # 拼接 [visible, mask] 再根据 ids_restore 复原原序
        x_cat = torch.cat([vis_dec, mask_tokens], dim=1)  # (B,L_full,H)
        gather_index = ids_restore.unsqueeze(-1).expand(-1, -1, self.hidden_size)
        x_full = torch.gather(x_cat, 1, gather_index)     # (B,L_full,H)

        # 构建完整 RoPE (shared)
        freqs_full = self.video_rope(
            t=T,
            h=h_patches,
            w=w_patches,
            device=x_full.device,
            dtype=x_full.dtype
        )  # (L_full, head_dim//2)

        # Transformer 输入 (L,B,C)
        x_in = self.ln_in(x_full).permute(1, 0, 2)
        x_out = self.transformer(x_in, rotary_pos_emb=freqs_full)  # (L_full,B,H)
        x_out = x_out.permute(1, 0, 2)  # (B,L_full,H)

        # 输出投影
        x_out = self.feature_head(x_out)  # (B,L_full,D_out)

        # 拆分可见 / 被遮挡
        decoded_visible = x_out[mask_bool].view(B, N_vis, -1)
        decoded_masked = x_out[~mask_bool].view(B, N_mask, -1)

        return {
            "decoded_full": x_out,
            "decoded_visible": decoded_visible,
            "decoded_masked": decoded_masked,
            "mask": mask.float() if mask.dtype != torch.float32 else mask,
            "ids_restore": ids_restore,
        }


@register_model
def pretrain_encoder_small_patch16_224_v10_03(pretrained: bool = False, **kwargs):
    model = LlavaViTEncoder(
        patch_size=16,
        img_size=224,
        hidden_size=576,
        head_dim=96,
        num_hidden_layers=12,
        intermediate_size=1536,
        num_frames=16,
        act_layer=nn.GELU,
        use_gradient_checkpointing=False,
        **kwargs
    )
    if pretrained:
        pass
    return model


@register_model
def pretrain_decoder_small_patch16_224_v10_03(pretrained: bool = False, **kwargs):
    model = LlavaViTDecoder(
        hidden_size=576,             # decoder hidden
        encoder_hidden_size=576,     # must match encoder hidden_size
        head_dim=96,
        num_hidden_layers=2,
        intermediate_size=1536,      # 384 * 4
        feature_proj_dim=384,        # final feature dimension
        act_layer=nn.GELU,
        use_gradient_checkpointing=False,
        **kwargs
    )
    if pretrained:
        pass
    return model


# ---------------- Main test: encoder + decoder ----------------
if __name__ == "__main__":
    torch.manual_seed(42)

    batch = 2
    channels = 3
    t_frames = 8
    img_size = 224
    mask_ratio = 0.5

    # 构造随机视频
    video = torch.randn(batch, channels, t_frames, img_size, img_size)

    # 初始化模型
    encoder = pretrain_encoder_small_patch16_224_v10_03()
    decoder = pretrain_decoder_small_patch16_224_v10_03()

    # 编码
    with torch.no_grad():
        enc_out = encoder(video, mask_ratio=mask_ratio)

    visible_embeddings = enc_out["visible_embeddings"]     # (B, N_vis, 576)
    mask = enc_out["mask"]                                 # (B, L_full)
    ids_restore = enc_out["ids_restore"]                   # (B, L_full)
    patch_grid = enc_out["patch_grid"]                     # (T, Hp, Wp)
    n_visible = enc_out["num_visible"]
    L_full = enc_out["full_sequence_length"]

    print("=== Encoder Info ===")
    print("video:", video.shape)
    print("patch_grid:", patch_grid)
    print("full_seq_len:", L_full)
    print("visible_embeddings:", visible_embeddings.shape)
    print("mask ratio actual:", (mask.sum() / mask.numel()).item())
    print("ids_restore:", ids_restore.shape)

    # 解码
    with torch.no_grad():
        dec_out = decoder(
            visible_embeddings=visible_embeddings,
            ids_restore=ids_restore,
            mask=mask,
            patch_grid=patch_grid
        )

    decoded_full = dec_out["decoded_full"]
    decoded_visible = dec_out["decoded_visible"]
    decoded_masked = dec_out["decoded_masked"]

    print("\n=== Decoder Info ===")
    print("decoded_full:", decoded_full.shape)         # (B, L_full, D_out)
    print("decoded_visible:", decoded_visible.shape)   # (B, N_vis, D_out)
    print("decoded_masked:", decoded_masked.shape)     # (B, N_mask, D_out)
    print("N_mask:", L_full - n_visible)

    # 简单一致性检查（可见数 + 遮挡数 = 全长）
    assert decoded_visible.size(1) + decoded_masked.size(1) == L_full, "visible+masked != full length"
    print("\nChecks passed.")
