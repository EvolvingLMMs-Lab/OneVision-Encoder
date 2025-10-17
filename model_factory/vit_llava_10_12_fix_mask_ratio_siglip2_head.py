import torch
from model_factory.layers import TransformerCausal, VideoRotaryEmbeddingSplit466, Siglip2MultiheadAttentionPoolingHead
from timm.models.layers import to_2tuple, trunc_normal_
from timm.models.registry import register_model
from torch import nn
from typing import Optional, Any, Dict

__all__ = [
    "pretrain_encoder_small_patch16_224_v10_12_rms_unmask",
    "pretrain_decoder_small_patch16_224_v10_08_rms"
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
        mask_ratio=0.5,             # 遮挡比例
        use_head=False,       # 是否使用 Siglip2MultiheadAttentionPoolingHead
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
        self.mask_ratio = float(mask_ratio)
        self.ln_post = norm_cls(hidden_size)
        self.use_head = use_head
        if use_head:
            self.head = Siglip2MultiheadAttentionPoolingHead(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                intermediate_size=intermediate_size
            )


    def mask_mae_style(self, batch_size, t_frames, patches_per_frame, mask_ratio, device):
        total_patches = t_frames * patches_per_frame
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
        future = frame_ids.unsqueeze(2) < frame_ids.unsqueeze(1)  # (B,N,N) 这里 frame_i < frame_j
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
        if t_frames == 1 or self.mask_ratio == 0.0:
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

        if self.use_head:
            head_output = self.head(self.ln_post(out))  # (B, hidden_size)

        return {
            "visible_embeddings": out,
            "head_output": head_output if self.use_head else None,
            "mask": visible_mask_bool.float(),
            "ids_restore": ids_restore,
            "visible_indices": visible_indices,
            "num_visible": n_visible,
            "full_sequence_length": total_patches,
            "patch_grid": (t_frames, h_patches, w_patches),
            "attention_mask_used": attention_mask is not None,
        }

    # def load_state_dict(self, state_dict, strict: bool = True):
    #     # 不在原地改用户传入的对象，避免副作用；也可用 deepcopy，如果你后面会就地操作 tensor。
    #     sd = dict(state_dict)


    #     # 2) 如果当前模型没有 head，或显式不使用 head，则移除相关权重
    #     no_head = (not hasattr(self, 'head')) or (getattr(self, 'head', None) is None)
    #     use_head = getattr(self, 'use_head', True)
    #     if no_head or (use_head is False):
    #         drop_keys = [k for k in list(sd.keys()) if k.startswith('head.')]
    #         for k in drop_keys:
    #             sd.pop(k)

    #     # 3) 交给父类去正常加载
    #     return super().load_state_dict(sd, strict=strict)


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

        self.ln_post = norm_cls(hidden_size)

    @staticmethod
    def _build_causal_temporal_mask_full(batch_size, total_patches, patches_per_frame, device):
        """
        为完整序列 (含可见+mask token) 构造帧级别因果 mask。
        同一帧内允许双向；未来帧被遮挡。
        返回: (B, L, L) bool, True=禁止注意
        """
        frame_ids = torch.arange(total_patches, device=device) // patches_per_frame  # (L,)
        # frame_i < frame_j => j 是未来帧 => 屏蔽 (query i 禁止看 key j)
        causal = frame_ids.unsqueeze(1) < frame_ids.unsqueeze(0)  # (L,L) True 说明列是未来帧
        causal = causal.unsqueeze(0).expand(batch_size, -1, -1).clone()  # (B,L,L)
        return causal  # True = disallowed

    def forward(
        self,
        visible_embeddings: torch.Tensor,  # (B,N_vis,C_enc)
        ids_restore: torch.Tensor,         # (B,L_full)
        mask: torch.Tensor,                # (B,L_full) 1=visible 0=masked
        patch_grid,                        # (T, h_patches, w_patches)
    ):
        # Check if all tokens are visible (unmask case)
        is_unmask = mask.all().item() if torch.is_tensor(mask) else False
        
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

        if is_unmask:
            # 全部可见，无需 mask tokens 和重排序
            assert N_vis == L_full, "Unmask case requires N_vis == L_full"
            x_full = vis_dec  # (B,L_full,H)
        else:
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


        x_in = x_full.permute(1, 0, 2)  # (L,B,H)
        x_out = self.transformer(
            x_in,
            rotary_pos_emb=freqs_full,       # (L,D/2)
            attention_mask=attention_mask    # (B,L,L) or None
        )
        x_out = x_out.permute(1, 0, 2)  # (B,L,H)

        # x_out = self.ln_post(x_out)

        # 处理返回值
        if is_unmask:
            # 全部可见的情况下
            return {
                "decoded_full": x_out,
                "decoded_visible": x_out,  # 全部为可见
                "decoded_masked": torch.zeros(B, 0, x_out.size(-1), device=x_out.device),  # 空张量
                "mask": mask.float() if mask.dtype != torch.float32 else mask,
                "ids_restore": ids_restore,
                "attention_mask_used": attention_mask is not None,
            }
        else:
            # 正常情况，有可见和遮罩的 tokens
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


@register_model
def pretrain_encoder_small_patch16_224_v10_12_rms_unmask(pretrained: bool = False, ckpt_path=None,**kwargs):
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
        mask_ratio=0.0,  # 不遮挡任何 patch
        use_head=False,
        use_causal_temporal=False
    )
    return model


@register_model
def pretrain_encoder_small_patch16_224_v10_12_rms_unmask_with_head(pretrained: bool = False, ckpt_path=None,**kwargs):
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
        mask_ratio=0.0,  # 不遮挡任何 patch
        use_head=True,
        use_causal_temporal=False
    )
    return model

@register_model
def pretrain_encoder_small_patch16_224_v10_12_rms_unmask_with_head_causal(pretrained: bool = False, ckpt_path=None,**kwargs):
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
        mask_ratio=0.0,  # 不遮挡任何 patch
        use_head=True,
        use_causal_temporal=True
    )
    return model

@register_model
def pretrain_encoder_base_patch16_224_v10_12_rms_unmask_with_head(pretrained: bool = False, ckpt_path=None,**kwargs):
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
        mask_ratio=0.0,  # 不遮挡任何 patch
        use_head=True,
        use_causal_temporal=False
    )
    return model


@register_model
def pretrain_encoder_base_patch16_224_v10_12_rms_unmask_with_head_causal(pretrained: bool = False, ckpt_path=None,**kwargs):
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
        mask_ratio=0.0,  # 不遮挡任何 patch
        use_head=True,
        use_causal_temporal=True
    )
    return model
