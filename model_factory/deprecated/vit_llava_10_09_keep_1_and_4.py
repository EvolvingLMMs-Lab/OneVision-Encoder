import torch
from model_factory.layers import TransformerCausal, VideoRotaryEmbeddingSplit466
from timm.models.layers import to_2tuple, trunc_normal_
from timm.models.registry import register_model
from torch import nn
from typing import Optional, Any, Dict

__all__ = [
    "pretrain_encoder_small_patch16_224_v10_09_keep_1_and_4",
    "pretrain_encoder_base_patch16_224_v10_09_keep_1_and_4",
    "pretrain_decoder_small_patch16_224_v10_09_keep_1_and_4",
    "mlcd_decoder_small_patch16_224_v10_09_keep_1_and_4",
    "mlcd_decoder_base_patch16_224_v10_09_keep_1_and_4",
]

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
        use_causal_temporal=True,   # 新增开关：是否启用时间因果
        norm_cls=nn.RMSNorm,
    ):
        super().__init__()

        assert hidden_size % head_dim == 0
        num_attention_heads = hidden_size // head_dim
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads

        self.use_causal_temporal = use_causal_temporal
        self.attn_dropout = attn_dropout
        self.patch_size = to_2tuple(patch_size)

        self.conv1 = nn.Conv2d(
            3, hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False
        )
        scale = hidden_size ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(hidden_size))
        self.ln_pre = norm_cls(hidden_size)

        # 使用带 causal mask 的 Transformer
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

        self.half_head_dim = head_dim // 2
        self.video_rope = VideoRotaryEmbeddingSplit466(head_dim)

    def mask_mae_style(self, batch_size, t_frames, patches_per_frame, mask_ratio, device):
        total_patches = t_frames * patches_per_frame

        # ===== 新增固定 8 帧特殊逻辑（最小侵入修改） =====
        if t_frames == 8:
            # 始终保留：帧0 与 帧4
            i_region = torch.arange(0, patches_per_frame, device=device)  # frame 0
            frame4_base = 4 * patches_per_frame
            frame4_region = torch.arange(frame4_base, frame4_base + patches_per_frame, device=device)

            always_keep_indices = torch.cat([i_region, frame4_region], dim=0)  # (2 * patches_per_frame,)

            # 其余帧(1,2,3,5,6,7)的所有 patch 组成候选集合
            # 原始 p_region_indices = patches_per_frame..total_patches-1 包含帧1-7
            p_region_indices_full = torch.arange(patches_per_frame, total_patches, device=device)
            # 去掉帧4
            mask_frames_candidate = p_region_indices_full[
                ~((p_region_indices_full >= frame4_base) &
                  (p_region_indices_full < frame4_base + patches_per_frame))
            ]
            candidate_count = mask_frames_candidate.numel()

            mask_ratio = float(mask_ratio)
            mask_ratio = max(0.0, min(1.0, mask_ratio))
            p_keep_count = int(round((1 - mask_ratio) * candidate_count))
            p_keep_count = max(0, min(p_keep_count, candidate_count))

            if p_keep_count > 0:
                rand_scores = torch.rand(batch_size, candidate_count, device=device)
                topk_idx = torch.topk(rand_scores, k=p_keep_count, dim=1, largest=True, sorted=False).indices
                kept_dynamic = mask_frames_candidate[topk_idx]   # (B, p_keep_count)
                always_keep_expanded = always_keep_indices.unsqueeze(0).expand(batch_size, -1)
                visible_indices = torch.cat([always_keep_expanded, kept_dynamic], dim=1)
            else:
                visible_indices = always_keep_indices.unsqueeze(0).expand(batch_size, -1)

            # 排序 + 构造 mask / ids_restore 与原逻辑一致
            visible_indices = torch.sort(visible_indices, dim=1).values
            visible_mask = torch.zeros(batch_size, total_patches, dtype=torch.bool, device=device)
            visible_mask.scatter_(1, visible_indices, True)

            vis_int = visible_mask.long()
            mask_int = 1 - vis_int
            vis_rank = torch.cumsum(vis_int, dim=1) - 1
            mask_rank = torch.cumsum(mask_int, dim=1) - 1
            n_visible_col = vis_int.sum(dim=1, keepdim=True)
            ids_restore = torch.where(visible_mask, vis_rank, n_visible_col + mask_rank)
            return visible_indices, visible_mask, ids_restore
        # ===== 固定 8 帧逻辑结束，其他情况走原始实现 =====

        i_region = torch.arange(0, patches_per_frame, device=device)
        p_region_indices = torch.arange(patches_per_frame, total_patches, device=device)
        p_region_count = p_region_indices.numel()

        p_keep_count = int(round((1 - mask_ratio) * p_region_count))
        p_keep_count = max(0, min(p_keep_count, p_region_count))

        if p_keep_count > 0:
            rand_scores = torch.rand(batch_size, p_region_count, device=device)
            topk_idx = torch.topk(rand_scores, k=p_keep_count, dim=1, largest=True, sorted=False).indices
            p_kept = p_region_indices[topk_idx]
            visible_indices = torch.cat([i_region.unsqueeze(0).expand(batch_size, -1), p_kept], dim=1)
        else:
            visible_indices = i_region.unsqueeze(0).expand(batch_size, -1)

        visible_indices = torch.sort(visible_indices, dim=1).values
        visible_mask = torch.zeros(batch_size, total_patches, dtype=torch.bool, device=device)
        visible_mask.scatter_(1, visible_indices, True)

        vis_int = visible_mask.long()
        mask_int = 1 - vis_int
        vis_rank = torch.cumsum(vis_int, dim=1) - 1
        mask_rank = torch.cumsum(mask_int, dim=1) - 1
        n_visible_col = vis_int.sum(dim=1, keepdim=True)
        ids_restore = torch.where(visible_mask, vis_rank, n_visible_col + mask_rank)
        return visible_indices, visible_mask, ids_restore

    def _build_causal_temporal_mask(self, visible_indices, patches_per_frame):
        B, N = visible_indices.shape
        frame_ids = visible_indices // patches_per_frame  # (B,N)
        # frame_ids[:,None,:] -> (B,1,N); frame_ids[:,:,None] -> (B,N,1)
        # 我们需要 mask[i,j] = True 当 frame_j > frame_i
        future = frame_ids.unsqueeze(1) < frame_ids.unsqueeze(2)  # (B,N,N) 这里 frame_i < frame_j
        # 我们想要 mask[i,j] = True 当 frame_j > frame_i ⇒ frame_ids[:, :, None] < frame_ids[:, None, :]
        # 注意上面 future 的定义是 frame_i < frame_j => 等价 frame_j > frame_i
        attention_mask = future  # True=禁止
        return attention_mask

    def forward(self, x: torch.Tensor, mask_ratio=0.5):
        if x.dim() == 4:
            x = x.unsqueeze(2)

        batch, channels, t_frames, height, width = x.shape
        patch_h, patch_w = self.patch_size
        assert height % patch_h == 0 and width % patch_w == 0
        h_patches = height // patch_h
        w_patches = width // patch_w
        patches_per_frame = h_patches * w_patches
        total_patches = t_frames * patches_per_frame
        device = x.device

        # patchify
        x_2d = x.permute(0,2,1,3,4).reshape(batch * t_frames, channels, height, width)
        feats = self.conv1(x_2d)
        feats = feats.reshape(batch, t_frames, self.hidden_size, h_patches, w_patches).permute(0,1,3,4,2)
        tokens = feats.reshape(batch, total_patches, self.hidden_size)

        # masking
        if t_frames == 1:
            # 全部可见，不进行随机遮挡
            visible_indices = torch.arange(total_patches, device=device).unsqueeze(0).expand(batch, -1)  # (B, L)
            visible_mask_bool = torch.ones(batch, total_patches, dtype=torch.bool, device=device)        # (B, L)
            ids_restore = torch.arange(total_patches, device=device).unsqueeze(0).expand(batch, -1)      # (B, L)
        else:
            # masking
            visible_indices, visible_mask_bool, ids_restore = self.mask_mae_style(
                batch_size=batch,
                t_frames=t_frames,
                patches_per_frame=patches_per_frame,
                mask_ratio=mask_ratio,
                device=device
            )
        n_visible = visible_indices.shape[1]

        gather_index = visible_indices.unsqueeze(-1).expand(-1, -1, self.hidden_size)
        visible_tokens = torch.gather(tokens, 1, gather_index)

        # RoPE (full -> select)
        freqs_full = self.video_rope(
            t=t_frames,
            h=h_patches,
            w=w_patches,
            device=device,
            dtype=tokens.dtype
        )
        freqs_visible = freqs_full[visible_indices]  # (B,N_vis,D/2)

        # 构造时间单向 mask (可选)
        if self.use_causal_temporal and t_frames > 1:
            attention_mask = self._build_causal_temporal_mask(visible_indices, patches_per_frame)  # (B,N,N)
        else:
            attention_mask = None

        # LN + 维度调整
        x_in = self.ln_pre(visible_tokens).permute(1, 0, 2)  # (N,B,C)

        # Transformer
        out = self.transformer(
            x_in,
            rotary_pos_emb=freqs_visible,        # (B,N,D/2)
            attention_mask=attention_mask        # (B,N,N) or None
        )
        out = out.permute(1, 0, 2)  # (B,N,C)

        return {
            "visible_embeddings": out,
            "mask": visible_mask_bool.float(),
            "ids_restore": ids_restore,
            "visible_indices": visible_indices,
            "num_visible": n_visible,
            "full_sequence_length": total_patches,
            "patch_grid": (t_frames, h_patches, w_patches),
            "attention_mask_used": attention_mask is not None,
        }


class LlavaViTDecoder(nn.Module):
    """
    Feature-level MAE-style Decoder（支持时间因果，与 Encoder 风格对齐）:

    输入:
        visible_embeddings : (B, N_vis, encoder_hidden_size)
        ids_restore        : (B, L_full)
        mask               : (B, L_full) 1=visible 0=masked
        patch_grid         : (T, h_patches, w_patches)

    流程:
        1. visible_embeddings -> 投影到 hidden_size (若与 encoder_hidden_size 不同)
        2. 添加可学习 mask_token (数量 = N_mask)
        3. 用 ids_restore 还原完整序列顺序 (B, L_full, hidden_size)
        4. 构造全序列 3D RoPE
        5. 构造时间因果 attention_mask (帧级别: 同一帧双向；禁止看到未来帧)
        6. TransformerCausal 前向
        7. 输出全量 / 可见 / 被遮挡 token 特征（不做像素重建）

    可选:
        use_causal_temporal = False 时取消时间因果（全局双向）
        feature_proj_dim 将最终输出特征进一步投影到指定维度。
    """
    def __init__(
        self,
        hidden_size=384,
        encoder_hidden_size=384,
        head_dim=48,
        num_hidden_layers=8,
        intermediate_size=1536,
        act_layer=nn.GELU,
        num_key_value_heads=None,
        feature_proj_dim=None,
        use_gradient_checkpointing=False,
        attn_dropout=0.0,
        use_causal_temporal=True,   # 新增：与 encoder 一致的时间单向注意力开关
        norm_cls=nn.RMSNorm,
    ):
        super().__init__()
        assert hidden_size % head_dim == 0, "hidden_size must be divisible by head_dim"
        num_attention_heads = hidden_size // head_dim
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.hidden_size = hidden_size
        self.encoder_hidden_size = encoder_hidden_size
        self.head_dim = head_dim
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_causal_temporal = use_causal_temporal
        self.attn_dropout = attn_dropout

        # 投影到 decoder hidden
        if encoder_hidden_size != hidden_size:
            self.proj_in = nn.Linear(encoder_hidden_size, hidden_size)
        else:
            self.proj_in = nn.Identity()

        # 学习的 mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        trunc_normal_(self.mask_token, std=0.02)

        # RoPE
        self.video_rope = VideoRotaryEmbeddingSplit466(head_dim)

        # LayerNorm + Transformer (使用支持 attention_mask 的 Causal 版本)
        self.ln_in = norm_cls(hidden_size)
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

        # 输出特征投影（比如对齐 teacher 维）
        if feature_proj_dim is not None and feature_proj_dim != hidden_size:
            self.feature_head = nn.Linear(hidden_size, feature_proj_dim)
            self.out_feature_dim = feature_proj_dim
        else:
            self.feature_head = nn.Identity()
            self.out_feature_dim = hidden_size

    @staticmethod
    def _build_causal_temporal_mask_full(batch_size, total_patches, patches_per_frame, device):
        """
        为完整序列 (含可见+mask token) 构造帧级别因果 mask。
        同一帧内允许双向；未来帧被遮挡。
        返回: (B, L, L) bool, True=禁止注意
        """
        frame_ids = torch.arange(total_patches, device=device) // patches_per_frame  # (L,)
        # frame_i < frame_j => j 是未来帧 => 屏蔽 (query i 禁止看 key j)
        causal = frame_ids.unsqueeze(0) < frame_ids.unsqueeze(1)  # (L,L) True 说明列是未来帧
        causal = causal.unsqueeze(0).expand(batch_size, -1, -1).clone()  # (B,L,L)
        return causal  # True = disallowed

    def forward(
        self,
        visible_embeddings: torch.Tensor,  # (B,N_vis,C_enc)
        ids_restore: torch.Tensor,         # (B,L_full)
        mask: torch.Tensor,                # (B,L_full) 1=visible 0=masked
        patch_grid,                        # (T, h_patches, w_patches)
    ):
        if mask.dtype != torch.bool:
            mask_bool = mask.bool()
        else:
            mask_bool = mask

        B, N_vis, _ = visible_embeddings.shape
        L_full = ids_restore.shape[1]
        T, h_patches, w_patches = patch_grid
        patches_per_frame = h_patches * w_patches
        assert L_full == T * patches_per_frame, "ids_restore length mismatch grid size"

        # 1. 投影 visible
        vis_dec = self.proj_in(visible_embeddings)  # (B,N_vis,H)

        # 2. mask tokens
        N_mask = L_full - N_vis
        mask_tokens = self.mask_token.expand(B, N_mask, self.hidden_size)  # (B,N_mask,H)

        # 3. 还原完整序列
        x_cat = torch.cat([vis_dec, mask_tokens], dim=1)  # (B,L_full,H)
        gather_index = ids_restore.unsqueeze(-1).expand(-1, -1, self.hidden_size)
        x_full = torch.gather(x_cat, 1, gather_index)  # (B,L_full,H)

        # 4. RoPE
        freqs_full = self.video_rope(
            t=T,
            h=h_patches,
            w=w_patches,
            device=x_full.device,
            dtype=x_full.dtype
        )  # (L_full, head_dim//2)

        # 5. 因果 mask
        if self.use_causal_temporal and T > 1:
            attention_mask = self._build_causal_temporal_mask_full(
                batch_size=B,
                total_patches=L_full,
                patches_per_frame=patches_per_frame,
                device=x_full.device
            )  # (B,L_full,L_full)
        else:
            attention_mask = None

        # 6. Transformer
        x_in = self.ln_in(x_full).permute(1, 0, 2)  # (L,B,H)
        x_out = self.transformer(
            x_in,
            rotary_pos_emb=freqs_full,       # (L,D/2)
            attention_mask=attention_mask    # (B,L,L) or None
        )
        x_out = x_out.permute(1, 0, 2)  # (B,L,H)

        # 7. 输出特征
        x_out = self.feature_head(x_out)  # (B,L,D_out)

        decoded_visible = x_out[mask_bool].view(B, N_vis, -1)
        decoded_masked = x_out[~mask_bool].view(B, N_mask, -1)

        return {
            "decoded_full": x_out,
            "decoded_visible": decoded_visible,
            "decoded_masked": decoded_masked,
            "mask": mask.float() if mask.dtype != torch.float32 else mask,
            "ids_restore": ids_restore,
            "attention_mask_used": attention_mask is not None,
        }


class MLCDViTDecoder(nn.Module):
    """ 简化版 Decoder，仅支持全量输入（不区分可见/遮挡），用于快速验证 ViT Encoder 特征质量。
    """
    def __init__(
        self,
        hidden_size=384,
        encoder_hidden_size=384,
        head_dim=48,
        num_hidden_layers=8,
        intermediate_size=1536,
        act_layer=nn.GELU,
        num_key_value_heads=None,
        feature_proj_dim=None,
        use_gradient_checkpointing=False,
        attn_dropout=0.0,
        use_causal_temporal=True,
        norm_cls=nn.RMSNorm,
    ):
        super().__init__()
        assert hidden_size % head_dim == 0
        if num_key_value_heads is None:
            num_key_value_heads = hidden_size // head_dim
        self.hidden_size = hidden_size
        self.encoder_hidden_size = encoder_hidden_size
        self.head_dim = head_dim
        self.num_attention_heads = hidden_size // head_dim
        self.num_key_value_heads = num_key_value_heads
        self.use_causal_temporal = use_causal_temporal

        self.proj_in = nn.Linear(encoder_hidden_size, hidden_size) if encoder_hidden_size != hidden_size else nn.Identity()
        self.video_rope = VideoRotaryEmbeddingSplit466(head_dim)
        self.ln_in = norm_cls(hidden_size)
        self.transformer = TransformerCausal(
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=intermediate_size,
            act_layer=act_layer,
            gradient_checkpointing=use_gradient_checkpointing,
            attn_dropout=attn_dropout,
            norm_cls=norm_cls,
        )
        if feature_proj_dim is not None and feature_proj_dim != hidden_size:
            self.feature_head = nn.Linear(hidden_size, feature_proj_dim)
            self.out_feature_dim = feature_proj_dim
        else:
            self.feature_head = nn.Identity()
            self.out_feature_dim = hidden_size

    def forward(
        self,
        full_embeddings: torch.Tensor,   # (B, L_full, encoder_hidden_size) 已经是完整序列
    ):
        batch_size, seq_len, hidden_dim = full_embeddings.shape
        x_full = self.proj_in(full_embeddings)  # (B,L,H)
        h_patches = w_patches = int(seq_len ** 0.5)

        freqs_full = self.video_rope(
            t=1, h=h_patches, w=w_patches,
            device=x_full.device,
            dtype=x_full.dtype
        )

        x_in = self.ln_in(x_full).permute(1, 0, 2)          # (L,B,H)
        x_out = self.transformer(
            x_in,
            rotary_pos_emb=freqs_full,
            attention_mask=None
        ).permute(1, 0, 2)                                   # (B,L,H)
        x_out = self.feature_head(x_out)

        return {
            "decoded_full": x_out
        }


@register_model
def pretrain_encoder_small_patch16_224_v10_09_keep_1_and_4(pretrained: bool = False, ckpt_path=None,**kwargs):
    """
    ViT Encoder for Video MAE-style pretraining."""
    model = LlavaViTEncoder(
        patch_size=16,
        hidden_size=384,
        head_dim=64,
        num_hidden_layers=12,
        intermediate_size=1536,
        act_layer=nn.GELU,
        use_gradient_checkpointing=False,
        norm_cls=nn.RMSNorm,
    )
    if pretrained:
        assert ckpt_path is not None, "ckpt_path must be provided for pretrained model"
        state_dict = torch.load(ckpt_path, map_location='cpu')
        # replace _orig_mod. in keys
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=True)
    return model

@register_model
def pretrain_encoder_base_patch16_224_v10_09_keep_1_and_4(pretrained: bool = False, ckpt_path=None,**kwargs):
    """
    ViT Encoder for Video MAE-style pretraining."""
    model = LlavaViTEncoder(
        patch_size=16,
        hidden_size=768,
        head_dim=64,
        num_hidden_layers=12,
        intermediate_size=3072,
        act_layer=nn.GELU,
        use_gradient_checkpointing=False,
        norm_cls=nn.RMSNorm,
    )
    if pretrained:
        assert ckpt_path is not None, "ckpt_path must be provided for pretrained model"
        state_dict = torch.load(ckpt_path, map_location='cpu')
        # replace _orig_mod. in keys
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=True)
    return model


@register_model
def pretrain_decoder_small_patch16_224_v10_09_keep_1_and_4(pretrained: bool = False, **kwargs):
    model = LlavaViTDecoder(
        hidden_size=384,             # decoder hidden
        encoder_hidden_size=384,     # must match encoder hidden_size
        head_dim=64,
        num_hidden_layers=3,
        intermediate_size=1536,      # 384 * 4
        feature_proj_dim=384,        # final feature dimension
        act_layer=nn.GELU,
        use_gradient_checkpointing=False,
        norm_cls=nn.RMSNorm,
    )
    if pretrained:
        pass
    return model


@register_model
def mlcd_decoder_small_patch16_224_v10_09_keep_1_and_4(pretrained: bool = False, **kwargs):
    """MLCD Decoder
    """
    model = MLCDViTDecoder(
        hidden_size=384,             # decoder hidden
        encoder_hidden_size=384,     # must match encoder hidden_size
        head_dim=64,
        num_hidden_layers=4,
        intermediate_size=1536,      # 384 * 4
        feature_proj_dim=384,        # final feature dimension
        act_layer=nn.GELU,
        use_gradient_checkpointing=False,
    )
    if pretrained:
        pass
    return model


@register_model
def mlcd_decoder_base_patch16_224_v10_09_keep_1_and_4(pretrained: bool = False, **kwargs):
    """MLCD Decoder
    """
    model = MLCDViTDecoder(
        hidden_size=768,             # decoder hidden
        encoder_hidden_size=768,     # must match encoder hidden_size
        head_dim=64,
        num_hidden_layers=4,
        intermediate_size=3072,      # 384 * 4
        feature_proj_dim=768,        # final feature dimension
        act_layer=nn.GELU,
        use_gradient_checkpointing=False,
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

    encoder = pretrain_encoder_small_patch16_224_v10_09_keep_1_and_4()
    decoder = pretrain_decoder_small_patch16_224_v10_09_keep_1_and_4()

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
