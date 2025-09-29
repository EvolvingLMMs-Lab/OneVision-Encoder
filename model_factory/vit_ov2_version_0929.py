import math
from typing import Callable, Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# =========================================================
# 基础旋转函数
# =========================================================
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb_vision(tensor: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """
    tensor: (B, L, H, Hd)
    freqs:  (L, Hd/2)  —— “角度”矩阵 (位置 x 频率)
    最终把 Hd 划分为两半做旋转 (经典 RoPE)
    """
    orig_dtype = tensor.dtype
    B, L, H, Hd = tensor.shape
    assert freqs.shape[0] == L, f"freq len {freqs.shape[0]} != seq len {L}"
    assert freqs.shape[1] * 2 == Hd, f"freq dim {freqs.shape[1]} *2 != head_dim {Hd}"
    angles = freqs  # (L, Hd/2)
    cos = angles.cos()  # (L, Hd/2)
    sin = angles.sin()
    # 扩展到 (B,L,H,Hd/2)
    cos = cos[None, :, None, :].expand(B, L, H, -1)
    sin = sin[None, :, None, :].expand(B, L, H, -1)
    x1, x2 = tensor[..., :Hd//2], tensor[..., Hd//2:]
    # 标准公式： [x1*cos - x2*sin , x2*cos + x1*sin]
    rotated_first = x1 * cos - x2 * sin
    rotated_second = x2 * cos + x1 * sin
    out = torch.cat([rotated_first, rotated_second], dim=-1)
    return out.to(orig_dtype)

# =========================================================
# 2D 旧版（仍被 Decoder 使用），保留以兼容
# =========================================================
class VisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        # 返回“角度”矩阵 (seqlen, dim/2)
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs

# =========================================================
# 3D RoPE: 生成 (T*H*W, Hd/2)
# =========================================================
class RotaryEmbedding3D(nn.Module):
    def __init__(self, head_dim: int, theta: float = 10000.0):
        super().__init__()
        if head_dim % 6 != 0:
            raise ValueError(f"head_dim={head_dim} 需满足 head_dim % 6 == 0, 方便 3D RoPE 均分。")
        half = head_dim // 2
        self.part = half // 3            # 每个轴应贡献的频率数量 (= head_dim / 6)
        self.theta = theta
        self.head_dim = head_dim

        # 修复：生成 self.part 个频率（而不是 self.part/2）
        inv_freq = 1.0 / (theta ** (torch.arange(0, self.part, dtype=torch.float) / self.part))
        self.register_buffer("inv_freq_t", inv_freq, persistent=False)
        self.register_buffer("inv_freq_h", inv_freq.clone(), persistent=False)
        self.register_buffer("inv_freq_w", inv_freq.clone(), persistent=False)

    def _axis_angles(self, length: int, inv_freq: torch.Tensor) -> torch.Tensor:
        # (length, self.part)
        pos = torch.arange(length, device=inv_freq.device, dtype=inv_freq.dtype)
        return torch.outer(pos, inv_freq)

    def forward(self, T: int, H: int, W: int) -> torch.Tensor:
        t_ang = self._axis_angles(T, self.inv_freq_t)  # (T, part)
        h_ang = self._axis_angles(H, self.inv_freq_h)  # (H, part)
        w_ang = self._axis_angles(W, self.inv_freq_w)  # (W, part)

        t_full = t_ang[:, None, None, :].expand(T, H, W, -1)
        h_full = h_ang[None, :, None, :].expand(T, H, W, -1)
        w_full = w_ang[None, None, :, :].expand(T, H, W, -1)

        full = torch.cat([t_full, h_full, w_full], dim=-1)  # (T,H,W, 3*part)
        full = full.reshape(T * H * W, -1)                  # (seq, 3*part)

        expected = self.head_dim // 2
        assert full.shape[1] == expected, \
            f"角度矩阵第二维应为 head_dim/2 = {expected}, 实际 {full.shape[1]}"
        return full

# =========================================================
# 注意力 + Block + Transformer
# =========================================================
class VisionSdpaAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 16) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.in_proj = nn.Linear(dim, dim * 3, bias=True)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, hidden_states: torch.Tensor, rotary_pos_emb: torch.Tensor) -> torch.Tensor:
        """
        hidden_states: (L, B, D)
        rotary_pos_emb: (L, head_dim/2) —— “角度”矩阵
        """
        L, B, D = hidden_states.shape
        H = self.num_heads
        assert D % H == 0
        Hd = D // H

        qkv = self.in_proj(hidden_states)  # (L,B,3D)
        qkv = qkv.view(L, B, 3, H, Hd).permute(2, 1, 0, 3, 4)  # (3,B,L,H,Hd)
        q, k, v = qkv.unbind(0)  # (B,L,H,Hd)

        q = apply_rotary_pos_emb_vision(q, rotary_pos_emb)  # (B,L,H,Hd)
        k = apply_rotary_pos_emb_vision(k, rotary_pos_emb)

        q = q.permute(0, 2, 1, 3)  # (B,H,L,Hd)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attn = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        attn = attn.permute(2, 0, 1, 3).reshape(L, B, D)
        return self.out_proj(attn)

class ResidualAttentionBlock(nn.Module):
    def __init__(self,
                 d_model: int,
                 n_head: int,
                 mlp_ratio: float = 4.0,
                 act_layer: Callable = nn.GELU,
                 scale_attn: bool = False,
                 scale_fc: bool = False,
                 drop_path: float = 0.0):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = VisionSdpaAttention(d_model, n_head)
        self.ln_attn = nn.LayerNorm(d_model) if scale_attn else nn.Identity()
        self.ln_2 = nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.LayerNorm(hidden) if scale_fc else nn.Identity(),
            act_layer(),
            nn.Linear(hidden, d_model)
        )
        self.drop_path = nn.Identity() if drop_path <= 0 else DropPath(drop_path)

    def forward(self, x: torch.Tensor, rotary_pos_emb: torch.Tensor):
        h = self.attn(self.ln_1(x), rotary_pos_emb=rotary_pos_emb)
        x = x + self.drop_path(self.ln_attn(h))
        x = x + self.drop_path(self.mlp(self.ln_2(x)))
        return x

class Transformer(nn.Module):
    def __init__(self,
                 width: int,
                 layers: int,
                 heads: int,
                 mlp_ratio: float = 4.0,
                 act_layer: Callable = nn.GELU,
                 drop_path: float = 0.0,
                 use_checkpoint: bool = False):
        super().__init__()
        self.grad_checkpointing = use_checkpoint
        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock(width, heads, mlp_ratio, act_layer, drop_path=drop_path)
            for _ in range(layers)
        ])

    def forward(self, x: torch.Tensor, rotary_pos_emb: torch.Tensor):
        # x: (L,B,D), rotary_pos_emb: (L, Hd/2)
        for blk in self.resblocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(lambda inp: blk(inp, rotary_pos_emb=rotary_pos_emb), x)
            else:
                x = blk(x, rotary_pos_emb=rotary_pos_emb)
        return x

# =========================================================
# Encoder (I 帧全保留 + P 帧按比例保留 + 3D RoPE)
# =========================================================
class PretrainEncoder(nn.Module):
    def __init__(self,
                 in_chans: int = 3,
                 patch_size: int = 16,
                 img_size: int = 224,
                 embed_dim: int = 768,
                 num_heads: int = 12,
                 depth: int = 12,
                 mlp_ratio: float = 4.0,
                 act_layer: Callable = nn.GELU,
                 num_frames: int = 8,
                 tubelet_size: int = 1,
                 drop_path: float = 0.0,
                 keep_ratio_p: float = 0.5,
                 use_checkpoint: bool = False):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = (patch_size, patch_size)
        self.num_heads = num_heads
        self.num_frames = num_frames
        self.keep_ratio_p = keep_ratio_p

        self.conv1 = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        scale = embed_dim ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(embed_dim))
        # CLS 所用的角度（这里直接零向量，更稳）
        self.cls_angle = nn.Parameter(torch.zeros(1, embed_dim // num_heads // 2))

        self.ln_pre = nn.LayerNorm(embed_dim)
        self.transformer = Transformer(embed_dim, depth, num_heads, mlp_ratio,
                                       act_layer=act_layer, drop_path=drop_path,
                                       use_checkpoint=use_checkpoint)

        self.head_dim = embed_dim // num_heads
        self.rope3d = RotaryEmbedding3D(self.head_dim)

    def _embed_patches(self, x: torch.Tensor):
        if x.dim() == 4:
            B, C, H, W = x.shape
            T = 1
            x = self.conv1(x)  # (B, D, H_p, W_p)
            B, D, H_p, W_p = x.shape
            x = x.view(B, D, H_p * W_p).transpose(1, 2)  # (B, N, D)
            return x, T, H_p, W_p
        else:
            B, C, T, H, W = x.shape
            x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
            x = self.conv1(x)
            _, D, H_p, W_p = x.shape
            x = x.reshape(B, T, D, H_p, W_p).permute(0, 1, 3, 4, 2).reshape(B, T * H_p * W_p, D)
            return x, T, H_p, W_p

    def _build_keep_mask(self, B, T, H_p, W_p, list_I, list_P, device):
        patches_per_frame = H_p * W_p
        N = T * patches_per_frame
        keep_mask = torch.zeros(B, N, dtype=torch.bool, device=device)
        frame_is_I = torch.zeros(B, T, dtype=torch.bool, device=device)

        # 默认第0帧为 I
        if list_I is None or list_P is None:
            list_I = torch.zeros(B, 1, dtype=torch.long, device=device)
            if T > 1:
                list_P = torch.arange(1, T, device=device).unsqueeze(0).repeat(B, 1)
            else:
                list_P = torch.zeros(B, 0, dtype=torch.long, device=device)
        else:
            if not torch.is_tensor(list_I):
                list_I = torch.as_tensor(list_I, dtype=torch.long, device=device)
            if not torch.is_tensor(list_P):
                list_P = torch.as_tensor(list_P, dtype=torch.long, device=device)

        for b in range(B):
            i_frames = list_I[b][list_I[b] >= 0]
            frame_is_I[b, i_frames] = True
            for f in i_frames.tolist():
                s, e = f * patches_per_frame, (f + 1) * patches_per_frame
                keep_mask[b, s:e] = True
            p_frames = list_P[b][list_P[b] >= 0]
            for f in p_frames.tolist():
                s, e = f * patches_per_frame, (f + 1) * patches_per_frame
                total = patches_per_frame
                keep_num = max(1, int(total * self.keep_ratio_p))
                rand_idx = torch.rand(total, device=device).topk(keep_num).indices
                fm = torch.zeros(total, dtype=torch.bool, device=device)
                fm[rand_idx] = True
                keep_mask[b, s:e] = fm

        return keep_mask, frame_is_I  # (B,N), (B,T)

    def _apply_keep(self, x, keep_mask):
        B, N, D = x.shape
        counts = keep_mask.sum(1)
        if not torch.all(counts == counts[0]):
            raise ValueError("各样本保留数量不一致。")
        kept = []
        ids_restore = []
        dropped = []
        for b in range(B):
            k_idx = torch.nonzero(keep_mask[b], as_tuple=False).flatten()
            d_idx = torch.nonzero(~keep_mask[b], as_tuple=False).flatten()
            kept.append(x[b, k_idx])
            ids_restore.append(torch.cat([k_idx, d_idx], 0))
            dropped.append(d_idx)
        kept = torch.stack(kept, 0)               # (B, N_kept, D)
        ids_restore = torch.stack(ids_restore, 0) # (B, N)
        dropped = torch.stack(dropped, 0)         # (B, N_drop)
        return kept, ids_restore, dropped

    def forward(self, x: torch.Tensor,
                list_I: Optional[torch.Tensor] = None,
                list_P: Optional[torch.Tensor] = None):
        """
        x: (B,C,T,H,W) 或 (B,C,H,W)
        返回:
            {
              cls, kept_tokens, i_tokens, p_tokens,
              keep_mask, is_I_token, ids_restore, drop_idx, shape_meta
            }
        """
        device = x.device
        x, T, H_p, W_p = self._embed_patches(x)  # (B,N,D)
        B, N_total, D = x.shape
        if T == 1:
            H_p = int(math.sqrt(N_total))
            W_p = H_p

        keep_mask, frame_is_I = self._build_keep_mask(B, T, H_p, W_p, list_I, list_P, device)
        patches_per_frame = H_p * W_p
        frame_ids = torch.arange(N_total, device=device).view(1, -1).repeat(B, 1) // patches_per_frame
        is_I_token = frame_is_I.gather(1, frame_ids)

        kept_tokens, ids_restore, drop_idx = self._apply_keep(x, keep_mask)

        # RoPE: 全量 (T,H,W)
        full_angles = self.rope3d(T, H_p, W_p)  # (N_total, Hd/2)
        kept_indices = ids_restore[:, :kept_tokens.shape[1]]
        # 假设每个样本 kept 顺序一致
        kept_angles = full_angles[kept_indices[0]]  # (N_kept, Hd/2)
        cls_angle = torch.zeros(1, full_angles.shape[1], device=device) + self.cls_angle  # (1, Hd/2)
        angles_seq = torch.cat([cls_angle, kept_angles], 0)  # (1+N_kept, Hd/2)

        cls_tok = self.class_embedding[None, None, :].expand(B, 1, D)
        x_seq = torch.cat([cls_tok, kept_tokens], 1)  # (B, 1+N_kept, D)
        x_seq = self.ln_pre(x_seq)
        x_seq = x_seq.permute(1, 0, 2)  # (L,B,D)
        x_seq = self.transformer(x_seq, rotary_pos_emb=angles_seq)
        x_seq = x_seq.permute(1, 0, 2)

        cls_out = x_seq[:, :1]
        patch_out = x_seq[:, 1:]  # (B,N_kept,D)

        kept_is_I = is_I_token.gather(1, kept_indices)
        i_tokens, p_tokens = [], []
        for b in range(B):
            i_tokens.append(patch_out[b][kept_is_I[b]])
            p_tokens.append(patch_out[b][~kept_is_I[b]])
        i_tokens = torch.stack(i_tokens, 0)
        p_tokens = torch.stack(p_tokens, 0)

        return {
            "cls": cls_out,
            "kept_tokens": patch_out,
            "i_tokens": i_tokens,
            "p_tokens": p_tokens,
            "keep_mask": keep_mask,
            "is_I_token": is_I_token,
            "ids_restore": ids_restore,
            "drop_idx": drop_idx,
            "shape_meta": (T, H_p, W_p)
        }

# =========================================================
# Decoder（重建 masked tokens 特征；简单版：直接预测所有 token）
# =========================================================
class PretrainDecoder(nn.Module):
    def __init__(self,
                 encoder_dim: int,
                 decoder_dim: int,
                 depth: int,
                 num_heads: int,
                 mlp_ratio: float = 4.0,
                 act_layer: Callable = nn.GELU,
                 drop_path: float = 0.0,
                 out_dim: Optional[int] = None):
        super().__init__()
        self.proj_in = nn.Linear(encoder_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        self.transformer = Transformer(decoder_dim, depth, num_heads, mlp_ratio,
                                       act_layer=act_layer, drop_path=drop_path)
        self.norm = nn.LayerNorm(decoder_dim)
        self.out_proj = nn.Linear(decoder_dim, out_dim) if out_dim and out_dim != decoder_dim else nn.Identity()

        self.head_dim = decoder_dim // num_heads
        self.rope3d = RotaryEmbedding3D(self.head_dim)
        self.cls_angle = nn.Parameter(torch.zeros(1, self.head_dim // 2))
        self.cls_embed = nn.Parameter(torch.zeros(decoder_dim))

    def forward(self,
                enc_dict: Dict,
                target_shape: Tuple[int, int, int]):
        """
        enc_dict: encoder 输出
        target_shape: (T, H_p, W_p)
        返回: (B, N_total, out_dim)
        """
        kept_tokens = enc_dict["kept_tokens"]  # (B,N_kept,E)
        ids_restore = enc_dict["ids_restore"]  # (B,N_total)
        B, N_kept, E = kept_tokens.shape
        N_total = ids_restore.shape[1]

        x = self.proj_in(kept_tokens)          # (B,N_kept,Dd)
        mask_tokens = self.mask_token.expand(B, N_total - N_kept, -1)
        x_ = torch.cat([x, mask_tokens], 1)    # (B,N_total,Dd)
        x_full = x_.gather(1, ids_restore.unsqueeze(-1).expand(-1, -1, x_.size(-1)))  # (B,N_total,Dd)

        T, H_p, W_p = target_shape
        angles_full = self.rope3d(T, H_p, W_p)             # (N_total, Hd/2)
        cls_angle = torch.zeros(1, angles_full.shape[1], device=x.device) + self.cls_angle
        angles_seq = torch.cat([cls_angle, angles_full], 0)  # (1+N_total, Hd/2)

        cls_tok = self.cls_embed[None, None, :].expand(B, 1, -1)
        x_seq = torch.cat([cls_tok, x_full], 1)  # (B,1+N_total,Dd)
        x_seq = x_seq.permute(1, 0, 2)           # (L,B,Dd)
        x_seq = self.transformer(x_seq, rotary_pos_emb=angles_seq)
        x_seq = x_seq.permute(1, 0, 2)
        x_seq = self.norm(x_seq)
        return self.out_proj(x_seq[:, 1:, :])    # (B,N_total,out_dim)

# =========================================================
# 简单整体封装 + L2
# =========================================================
class VideoMaskedAutoEncoder(nn.Module):
    def __init__(self,
                 in_chans=3,
                 img_size=224,
                 patch_size=16,
                 num_frames=8,
                 encoder_dim=768,
                 encoder_depth=12,
                 encoder_heads=12,
                 decoder_dim=512,
                 decoder_depth=4,
                 decoder_heads=8,
                 keep_ratio_p=0.5,
                 target_dim=512):
        super().__init__()
        self.encoder = PretrainEncoder(
            in_chans=in_chans,
            patch_size=patch_size,
            img_size=img_size,
            embed_dim=encoder_dim,
            num_heads=encoder_heads,
            depth=encoder_depth,
            num_frames=num_frames,
            keep_ratio_p=keep_ratio_p
        )
        self.decoder = PretrainDecoder(
            encoder_dim=encoder_dim,
            decoder_dim=decoder_dim,
            depth=decoder_depth,
            num_heads=decoder_heads,
            out_dim=target_dim
        )
        self.target_dim = target_dim
        self.num_frames = num_frames
        self.img_size = img_size
        self.patch_size = patch_size

    def forward(self, video: torch.Tensor, teacher_feats: Optional[torch.Tensor] = None):
        """
        video: (B,C,T,H,W)
        teacher_feats: (B, T, H_p, W_p, target_dim) 或 (B, N, target_dim)
        """
        enc_out = self.encoder(video)
        T, H_p, W_p = enc_out["shape_meta"]
        pred = self.decoder(enc_out, (T, H_p, W_p))  # (B,N_total,target_dim)
        if teacher_feats is not None:
            if teacher_feats.dim() == 5:
                teacher = teacher_feats.reshape(teacher_feats.size(0), -1, teacher_feats.size(-1))
            else:
                teacher = teacher_feats
            # 只计算 masked 的 L2
            keep_mask = enc_out["keep_mask"]  # (B,N_total)
            mask = ~keep_mask
            if mask.sum() == 0:
                loss = F.mse_loss(pred, teacher)
            else:
                loss = F.mse_loss(pred[mask], teacher[mask])
            return {"loss": loss, "pred": pred, "enc": enc_out}
        return {"pred": pred, "enc": enc_out}

# =========================================================
# __main__ 测试
# =========================================================
if __name__ == "__main__":
    torch.manual_seed(42)
    B, C, T, H, W = 2, 3, 8, 224, 224
    patch = 14
    video = torch.randn(B, C, T, H, W)

    model = VideoMaskedAutoEncoder(
        in_chans=C,
        img_size=H,
        patch_size=patch,
        num_frames=T,
        encoder_dim=576,
        encoder_depth=4,
        encoder_heads=6,
        decoder_dim=576,
        decoder_depth=2,
        decoder_heads=6,
        keep_ratio_p=0.5,
        target_dim=256
    )

    # 构造 teacher 特征 (与 patch 网格对应)
    H_p = H // patch
    W_p = W // patch
    teacher_feats = torch.randn(B, T, H_p, W_p, 256)

    out = model(video, teacher_feats)
    print("Loss:", out["loss"].item())
    print("Pred shape:", out["pred"].shape)          # (B, T*H_p*W_p, target_dim)
    print("Kept tokens:", out["enc"]["kept_tokens"].shape)
    print("I tokens:", out["enc"]["i_tokens"].shape, "P tokens:", out["enc"]["p_tokens"].shape)
    print("Keep ratio real:", out["enc"]["keep_mask"].float().mean().item())