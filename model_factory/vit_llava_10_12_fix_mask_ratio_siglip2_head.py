import torch
from model_factory.layers import TransformerCausal, VideoRotaryEmbeddingSplit466, Siglip2MultiheadAttentionPoolingHead
from timm.models.layers import to_2tuple, trunc_normal_
from timm.models.registry import register_model
from torch import nn
from typing import Optional, Any, Dict


__all__ = [
    "pretrain_encoder_small_patch16_224_v10_12_rms_unmask",
    "pretrain_encoder_small_patch16_224_v10_12_rms_unmask_with_head",
    "pretrain_encoder_small_patch16_224_v10_12_rms_mask05_head",
    "pretrain_encoder_small_patch16_224_v10_12_rms_mask08_head",
    "pretrain_encoder_small_patch16_224_v10_12_rms_unmask_with_head_causal",
    "pretrain_encoder_base_patch16_224_v10_12_rms_unmask_with_head",
    "pretrain_encoder_base_patch16_224_v10_12_rms_unmask_with_head_causal",
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
def pretrain_encoder_small_patch16_224_v10_12_rms_mask05_head(pretrained: bool = False, ckpt_path=None,**kwargs):
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
        mask_ratio=0.5,  # 不遮挡任何 patch
        use_head=True,
        use_causal_temporal=False
    )
    return model


@register_model
def pretrain_encoder_small_patch16_224_v10_12_rms_mask08_head(pretrained: bool = False, ckpt_path=None,**kwargs):
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
        mask_ratio=0.8,  # 不遮挡任何 patch
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
