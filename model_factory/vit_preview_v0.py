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

    def mask_by_residual_topk(self, res: torch.Tensor, k_keep: int):
        """
        基于残差 res 的 Top-K 掩码。
        选择 |res| 在 patch 内求和后的得分最高的 K 个 patch 作为可见，其余为 mask。

        Args:
            res:  (B, 1, T, H, W)  —— I 帧建议事先置 0，这样自然会优先选到 P 帧。
            k_keep: int            —— 每个样本保留的可见块数量（Top-K 超参）

        Returns:
            visible_indices: LongTensor (B, K)   —— 选中的线性 patch 索引（按升序）
            visible_mask:    BoolTensor (B, L)   —— L = T * (H/Ph) * (W/Pw)
            ids_restore:     LongTensor (B, L)   —— MAE 风格的还原下标
        """
        assert res.dim() == 5 and res.size(1) == 1, "res 需为 (B,1,T,H,W)"
        B, _, T, H, W = res.shape
        ph, pw = self.patch_size
        assert H % ph == 0 and W % pw == 0, "H/W 必须能被 patch 大小整除"

        hb, wb = H // ph, W // pw        # 每帧的 patch 网格
        L = T * hb * wb                  # 总 patch 数

        # K 边界
        K = int(max(0, min(k_keep, L)))

        # 计算每个 patch 的残差得分（|.| 在 patch 内求和） -> (B, T, hb, wb)
        # 参考：res_c = res[:hb*ph, :wb*pw].reshape(hb, ph, wb, pw); s = |res_c|.sum(axis=(1,3))
        res_abs = res.abs().squeeze(1)                                 # (B,T,H,W)
        scores = res_abs.reshape(B, T, hb, ph, wb, pw).sum(dim=(3, 5)) # (B,T,hb,wb)
        scores = scores.reshape(B, L)                                  # (B, L)

        # 选 Top-K（按 batch 独立进行）
        if K > 0:
            topk_idx = torch.topk(scores, k=K, dim=1, largest=True, sorted=False).indices  # (B, K)
            visible_indices = torch.sort(topk_idx, dim=1).values                           # (B, K) 升序，便于后续索引
        else:
            visible_indices = torch.empty(B, 0, dtype=torch.long, device=res.device)

        # 构造可见 mask
        visible_mask = torch.zeros(B, L, dtype=torch.bool, device=res.device)
        if K > 0:
            visible_mask.scatter_(1, visible_indices, True)

        # MAE 风格 ids_restore：把可见块排在前面、遮挡块排在后面
        vis_int   = visible_mask.long()
        mask_int  = 1 - vis_int
        vis_rank  = torch.cumsum(vis_int, dim=1) - 1
        mask_rank = torch.cumsum(mask_int, dim=1) - 1
        n_visible_col = vis_int.sum(dim=1, keepdim=True)
        ids_restore = torch.where(visible_mask, vis_rank, n_visible_col + mask_rank).long()

        return visible_indices, visible_mask, ids_restore

    def _compute_rope_from_positions(self, patch_positions, visible_indices, device, dtype):
        """
        Compute RoPE frequencies from explicit patch positions.
        
        Args:
            patch_positions: Tensor of shape (num_patches, 3) containing [t, h, w] positions
                for each patch in the sequence. This allows computing RoPE for patches
                from different spatial-temporal locations.
            visible_indices: Tensor of shape (B, N) containing the indices of visible patches.
            device: Target device.
            dtype: Target dtype.
        
        Returns:
            Tensor of shape (B, N_vis, D/2) containing the RoPE frequencies for visible patches.
        """
        t_pos = patch_positions[:, 0].float()
        h_pos = patch_positions[:, 1].float()
        w_pos = patch_positions[:, 2].float()
        
        inv_t = self.video_rope.inv_freq_t.to(device=device, dtype=dtype)
        inv_h = self.video_rope.inv_freq_h.to(device=device, dtype=dtype)
        inv_w = self.video_rope.inv_freq_w.to(device=device, dtype=dtype)
        
        ft = torch.outer(t_pos.to(device), inv_t)
        fh = torch.outer(h_pos.to(device), inv_h)
        fw = torch.outer(w_pos.to(device), inv_w)
        
        freqs_full = torch.cat([ft, fh, fw], dim=-1)
        freqs_visible = freqs_full[visible_indices]
        
        return freqs_visible

    def forward(self, x: torch.Tensor, visible_indices = None, mask_ratio=0.5, patch_positions=None):
        """
        Forward pass for LlavaViTEncoder.
        
        Args:
            x: Input tensor of shape (B, C, H, W) for images or (B, C, T, H, W) for videos.
            visible_indices: Optional indices for visible patches. Can be:
                - None: All patches are visible (no masking)
                - 2D tensor (B, N): Specific visible patch indices
                - 5D tensor (B, 1, T, H, W): Residual tensor for top-k masking
            mask_ratio: Ratio of patches to mask (used for MAE-style training).
            patch_positions: Optional tensor of shape (num_patches, 3) containing [t, h, w] 
                positions for each patch, used for RoPE calculation when patches come from
                different images/videos with varying spatial-temporal positions.
                If provided, this overrides the default position calculation based on grid.
        
        Returns:
            Dictionary with 'visible_embeddings' and optionally 'head_output'.
        """
        if x.dim() == 4:
            x = x.unsqueeze(2)

        # print("x.shape", x.shape)
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
        if t_frames == 1 or visible_indices is None:
            # Use all patches as visible for single frame or when no masking is specified
            visible_indices = torch.arange(total_patches, device=device).unsqueeze(0).expand(batch, -1)  # (B, L)
            visible_mask_bool = torch.ones(batch, total_patches, dtype=torch.bool, device=device)        # (B, L)
            ids_restore = torch.arange(total_patches, device=device).unsqueeze(0).expand(batch, -1)      # (B, L)
        elif visible_indices.ndim == 5:
            # 来自 residual top-k 的掩码
            visible_indices, visible_mask_bool, ids_restore = self.mask_by_residual_topk(
                res=visible_indices,
                k_keep=1536
            )
        elif visible_indices.ndim == 2:
            visible_indices = visible_indices.long()
        else:
            raise ValueError("Invalid visible_indices shape")

        n_visible = visible_indices.shape[1]

        gather_index = visible_indices.unsqueeze(-1).expand(-1, -1, self.hidden_size)
        visible_tokens = torch.gather(tokens, 1, gather_index)

        # RoPE (full -> select)
        if patch_positions is not None:
            # Validate patch_positions shape
            if patch_positions.ndim != 2 or patch_positions.shape[1] != 3:
                raise ValueError(
                    f"patch_positions must have shape (num_patches, 3), got {patch_positions.shape}"
                )
            # Use provided patch positions for RoPE calculation
            freqs_visible = self._compute_rope_from_positions(
                patch_positions, visible_indices, device, tokens.dtype
            )
        else:
            # Default: compute RoPE from grid
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


        x_in = self.ln_pre(visible_tokens).permute(1, 0, 2)  # (N,B,C)

        # Transformer
        out = self.transformer(
            x_in,
            rotary_pos_emb=freqs_visible,        # (B,N,D/2)
            attention_mask=attention_mask        # (B,N,N) or None
        )
        out = out.permute(1, 0, 2)  # (B,N,C)
        out = self.ln_post(out)

        if self.use_head:
            head_output = self.head(out)  # (B, hidden_size)
        else:
            head_output = None

        return {
            "visible_embeddings": out,
            "head_output": head_output if self.use_head else None,
        }


@register_model
def llava_vit_base_rms(pretrained: bool = False, ckpt_path=None,**kwargs):
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
        use_head=True
    )
    return model


@register_model
def llava_vit_small_ln(pretrained: bool = False, ckpt_path=None,**kwargs):
    """
    ViT Encoder for Video MAE-style pretraining."""
    model = LlavaViTEncoder(
        patch_size=16,
        hidden_size=384,
        head_dim=64,
        num_hidden_layers=6,
        intermediate_size=1536,
        act_layer=nn.GELU,
        use_gradient_checkpointing=False,
        norm_cls=nn.LayerNorm,
        use_head=True
    )
    return model


@register_model
def llava_vit_base_ln(pretrained: bool = False, ckpt_path=None,**kwargs):
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
        norm_cls=nn.LayerNorm,
        use_head=True
    )
    return model


@register_model
def llava_vit_large_ln(pretrained: bool = False, ckpt_path=None,**kwargs):
    """
    ViT Encoder for Video MAE-style pretraining."""
    model = LlavaViTEncoder(
        patch_size=14,
        hidden_size=1024,
        head_dim=64,
        num_hidden_layers=24,
        intermediate_size=4096,
        act_layer=nn.GELU,
        use_gradient_checkpointing=False,
        norm_cls=nn.LayerNorm,
        use_head=True
    )
    return model


@register_model
def llava_vit_huge_ln(pretrained: bool = False, ckpt_path=None, **kwargs):
    """
    ViT Encoder for Video MAE-style pretraining (huge size).
    """
    model = LlavaViTEncoder(
        patch_size=14,
        hidden_size=1280,          # 比 large 的 1024 更宽
        head_dim=64,               # 与 hidden_size 对应放大
        num_hidden_layers=32,      # 比 large 的 24 更深
        intermediate_size=5120,    # 比 large 的 4096 更大
        act_layer=nn.GELU,
        use_gradient_checkpointing=False,
        norm_cls=nn.LayerNorm,
        use_head=True
    )
    return model


@register_model
def llava_vit_giant_ln(pretrained: bool = False, ckpt_path=None, **kwargs):
    """
    ViT Encoder for Video MAE-style pretraining (giant size).
    """
    model = LlavaViTEncoder(
        patch_size=14,
        hidden_size=1536,          # 再加一档宽度
        head_dim=96,               # 对应地放大 head 维度
        num_hidden_layers=40,      # 深度进一步增加
        intermediate_size=6144,    # MLP hidden 更大
        act_layer=nn.GELU,
        use_gradient_checkpointing=False,
        norm_cls=nn.LayerNorm,
        use_head=True
    )
    return model
