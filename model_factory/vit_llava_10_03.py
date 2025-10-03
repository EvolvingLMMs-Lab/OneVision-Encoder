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
        patch_size=14,
        img_size=224,
        hidden_size=192,
        out_hidden_size=384,
        head_dim=64,                 # 192 // 3
        num_hidden_layers=12,
        intermediate_size=768,       # 192 * 4
        num_frames=16,
        tubelet_size=1,
        act_layer=nn.GELU,
        predictor_embed_dim=384,
        use_gradient_checkpointing=False,
        **kwargs
    )
    if pretrained:
        pass
    return model


# ================= main test =================
if __name__ == "__main__":
    torch.manual_seed(42)

    # config
    batch = 4
    channels = 3
    t_frames = 8
    img_size = 224
    mask_ratio = 0.5
    hidden = 576

    # build random video
    video = torch.randn(batch, channels, t_frames, img_size, img_size)

    # model
    model = pretrain_encoder_small_patch16_224_v10_03()

    with torch.no_grad():
        out = model(video)

    visible_embeddings = out["visible_embeddings"]
    mask = out["mask"]
    ids_restore = out["ids_restore"]
    visible_indices = out["visible_indices"]
    n_visible = out["num_visible"]
    full_seq_len = out["full_sequence_length"]
    patch_grid = out["patch_grid"]

    print("=== Video Encoder Test ===")
    print(f"video shape: {video.shape}")
    print(f"patch grid (t, hp, wp): {patch_grid}")
    print(f"full seq len: {full_seq_len}")
    print(f"visible count: {n_visible}")
    print(f"visible ratio (overall): {(mask.sum() / mask.numel()).item():.4f}")
    print(f"visible_embeddings shape: {visible_embeddings.shape}")
    print(f"mask shape: {mask.shape}")
    print(f"ids_restore shape: {ids_restore.shape}")
    print(f"visible_indices shape: {visible_indices.shape}")
    print("first sample visible_indices (first 30):", visible_indices[0][:30].tolist())

    # verify ids_restore round-trip
    # construct [visible, mask_tokens] then gather back
    dummy_mask_tokens = torch.zeros(full_seq_len - n_visible, hidden)
    cat0 = torch.cat([visible_embeddings[0], dummy_mask_tokens], dim=0)  # (full_seq_len, hidden)
    restored0 = cat0[ids_restore[0]]
    recon_visible = restored0[visible_indices[0]]
    max_diff = (recon_visible - visible_embeddings[0]).abs().max().item()
    print(f"ids_restore round-trip max diff: {max_diff:.6f}")

    # single frame test
    single_img = torch.randn(batch, channels, img_size, img_size)
    with torch.no_grad():
        out_single = model(single_img)
    print("\n=== Single Frame Test ===")
    print("visible_embeddings shape:", out_single["visible_embeddings"].shape)
    print("mask sum:", out_single["mask"].sum().item(), "/", out_single["mask"].numel())
