import torch
from model_factory.layers import TransformerCausal, VideoRotaryEmbeddingSplit466, Siglip2MultiheadAttentionPoolingHead
from timm.models.layers import to_2tuple, trunc_normal_
from timm.models.registry import register_model
from torch.nn import functional as F
from torch import nn
from typing import Optional, Any, Dict


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
        use_causal_temporal=False,   # 新增开关：是否启用时间因果
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
    
    def mask_from_given_matrix(self, given_mask: torch.Tensor, t_frames: int, patches_per_frame: int):
        """
        根据外部给定的mask矩阵生成MAE风格的索引和重排信息。
        - given_mask: (batch_size, total_patches)，1表示被mask掉，0表示可见
        - t_frames: 总帧数
        - patches_per_frame: 每帧patch数
        返回:
            visible_indices: (B, N_vis)
            visible_mask_bool: (B, total_patches)
            ids_restore: (B, total_patches)
        """
        device = given_mask.device
        batch_size, total_patches = given_mask.shape
        assert total_patches == t_frames * patches_per_frame, \
            f"given_mask shape mismatch: expected {t_frames * patches_per_frame}, got {total_patches}"

        # 可见部分 mask
        visible_mask = (given_mask == 0)
        visible_mask_bool = visible_mask.bool()

        # 计算 visible indices
        visible_indices = visible_mask.nonzero(as_tuple=False)
        # visible_indices: (N_total_visible, 2) → [batch_idx, patch_idx]
        # 需要按 batch 聚合
        visible_indices = [
            visible_indices[visible_indices[:, 0] == b, 1]
            for b in range(batch_size)
        ]
        # pad 成相同长度 (因为每个样本的可见数量可能不同)
        max_len = max(v.shape[0] for v in visible_indices)
        visible_indices = torch.stack([
            torch.cat([v, v.new_full((max_len - v.shape[0],), v[-1] if v.numel() > 0 else 0)])
            if v.numel() > 0 else torch.zeros(max_len, dtype=torch.long, device=device)
            for v in visible_indices
        ], dim=0)

        # 保证排序一致
        visible_indices = torch.sort(visible_indices, dim=1).values

        # vis_int / mask_int
        vis_int = visible_mask.long()
        mask_int = 1 - vis_int
        vis_rank = torch.cumsum(vis_int, dim=1) - 1
        mask_rank = torch.cumsum(mask_int, dim=1) - 1
        n_visible_col = vis_int.sum(dim=1, keepdim=True)
        ids_restore = torch.where(visible_mask, vis_rank, n_visible_col + mask_rank)

        return visible_indices, visible_mask_bool, ids_restore

    def downsample_mask(self, given_mask: torch.Tensor, patch_size: int = 16):
        """
        将 (B, 1, T, H, W) 的mask下采样为 patch 级别mask。
        假设每个patch内数值相同，因此直接下采样即可。
        
        返回:
            patch_mask: (B, T * H//p * W//p)
            t_frames: T
            patches_per_frame: H//p * W//p
        """
        B, _, T, H, W = given_mask.shape
        p = patch_size
        assert H % p == 0 and W % p == 0, "H, W 必须能整除 patch_size"

        # reshape成 (B*T, 1, H, W)
        mask_2d = given_mask.view(B * T, 1, H, W)
        # 使用最近邻下采样（因为每个patch内值相同）
        mask_down = F.interpolate(mask_2d, scale_factor=1/p, mode='nearest')
        # reshape回 (B, T, H//p, W//p)
        mask_down = mask_down.view(B, T, H//p, W//p)

        patches_per_frame = (H // p) * (W // p)
        patch_mask = mask_down.view(B, T * patches_per_frame)

        return patch_mask, T, patches_per_frame

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

    def forward(self, x: torch.Tensor, ip_mask = None, mask_ratio=0.5):
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

        # import pdb; pdb.set_trace()
        # masking
        if t_frames == 1 or self.mask_ratio == 0.0:
            # 全部可见，不进行随机遮挡
            visible_indices = torch.arange(total_patches, device=device).unsqueeze(0).expand(batch, -1)  # (B, L)
            visible_mask_bool = torch.ones(batch, total_patches, dtype=torch.bool, device=device)        # (B, L)
            ids_restore = torch.arange(total_patches, device=device).unsqueeze(0).expand(batch, -1)      # (B, L)
        elif ip_mask is not None:
            # print("ip_mask", ip_mask.shape)
            ip_mask, t_frames, patches_per_frame = self.downsample_mask(ip_mask, patch_size=16)
            # print("ip_mask after prepare", ip_mask.shape)
            visible_indices, visible_mask_bool, ids_restore = self.mask_from_given_matrix(
                given_mask=ip_mask,
                t_frames=t_frames,
                patches_per_frame=patches_per_frame,
            )
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
        # out = self.ln_post(out)

        if self.use_head:
            head_output = self.head(out)  # (B, hidden_size)


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

    def load_state_dict(self, state_dict, strict: bool = True):
        # 不在原地改用户传入的对象，避免副作用；也可用 deepcopy，如果你后面会就地操作 tensor。
        sd = dict(state_dict)


        # 2) 如果当前模型没有 head，或显式不使用 head，则移除相关权重
        no_head = (not hasattr(self, 'head')) or (getattr(self, 'head', None) is None)
        use_head = getattr(self, 'use_head', True)
        if no_head or (use_head is False):
            drop_keys = [k for k in list(sd.keys()) if k.startswith('head.')]
            for k in drop_keys:
                sd.pop(k)

        # 3) 交给父类去正常加载
        return super().load_state_dict(sd, strict=strict)


@register_model
def pretrain_encoder_small_patch16_224_v10_12_rms_unmask_ip(pretrained: bool = False, ckpt_path=None,**kwargs):
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
        use_head=False
    )
    return model


@register_model
def pretrain_encoder_small_patch16_224_v10_12_rms_mask05_head_ip(pretrained: bool = False, ckpt_path=None,**kwargs):
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
        use_head=True
    )
    return model


@register_model
def pretrain_encoder_small_patch16_224_v10_12_rms_unmask_with_head_ip(pretrained: bool = False, ckpt_path=None,**kwargs):
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
        use_head=True
    )
    return model

