import torch
from model_factory.layers import TransformerCausal, VisionRotaryEmbedding, Siglip2MultiheadAttentionPoolingHead
from timm.models.layers import to_2tuple, trunc_normal_
from timm.models.registry import register_model
from torch import nn
from typing import Optional, Any, Dict


class VisualTransformer(nn.Module):
    def __init__(
    self,
    patch_size=16,
    hidden_size=384,
    num_head = 6,
    head_dim=64,
    num_hidden_layers=12,
    intermediate_size=1536,
    act_layer=nn.GELU,
    num_key_value_heads=None,
    use_gradient_checkpointing=False,
    attn_dropout=0.0,
    use_causal_temporal=True,   # 新增开关：是否启用时间因果
    norm_cls=nn.RMSNorm,
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
        self.vision_rotary_embedding = VisionRotaryEmbedding(head_dim)
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
        """
        visible_indices: (B, N_vis) 已排序
        返回 attention_mask: (B, N_vis, N_vis)  True=不允许(attend)，False=允许
        规则：
          - 同一帧内全可见（双向），即不屏蔽
          - 不能看未来帧 ⇒ 对于 query i, 若 key j 属于未来帧 (frame_j > frame_i) 则屏蔽
        """
        B, N = visible_indices.shape
        frame_ids = visible_indices // patches_per_frame  # (B,N)
        # frame_ids[:,None,:] -> (B,1,N); frame_ids[:,:,None] -> (B,N,1)
        # 我们需要 mask[i,j] = True 当 frame_j > frame_i
        future = frame_ids.unsqueeze(1) < frame_ids.unsqueeze(2)  # (B,N,N) 这里 frame_i < frame_j
        # 我们想要 mask[i,j] = True 当 frame_j > frame_i ⇒ frame_ids[:, :, None] < frame_ids[:, None, :]
        # 注意上面 future 的定义是 frame_i < frame_j => 等价 frame_j > frame_i
        attention_mask = future  # True=禁止
        attention_mask = torch.where(attention_mask, float('-inf'), 0.0)
        return attention_mask

    def tem_rot_pos_emb(self, grid_thw, head_dim=None):
        t, h, w = grid_thw

        tpos_ids = torch.arange(t)
        tpos_ids = tpos_ids.view(-1, 1).expand(-1, h * w).flatten()

        hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
        hpos_ids = hpos_ids.reshape(
            h,
            1,
            w,
            1,
        )
        hpos_ids = hpos_ids.permute(0, 2, 1, 3)
        hpos_ids = hpos_ids.flatten()

        wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
        wpos_ids = wpos_ids.reshape(
            h,
            1,
            w,
            1,
        )
        wpos_ids = wpos_ids.permute(0, 2, 1, 3)
        wpos_ids = wpos_ids.flatten()

        pos_h_ids = hpos_ids.repeat(t, 1)
        pos_w_ids = wpos_ids.repeat(t, 1)

        max_grid_size = grid_thw.max()

        rotary_pos_emb_full = self.vision_rotary_embedding(max_grid_size)
        temperal_pos_emb = rotary_pos_emb_full[tpos_ids][:, :head_dim//4]
        spatial_h_pos_emb = rotary_pos_emb_full[pos_h_ids][:, :, head_dim//4:head_dim//8*5].flatten(0, 1)
        spatial_w_pos_emb = rotary_pos_emb_full[pos_w_ids][:, :, head_dim//8*5:].flatten(0, 1)
        rotary_pos_emb = torch.cat([temperal_pos_emb, spatial_h_pos_emb, spatial_w_pos_emb], dim=-1)
            
        return rotary_pos_emb
    
    def forward(self, x: torch.Tensor, mask=0.5):
  
        batch, channels, t_frames, height, width = x.shape
        patch_h, patch_w = self.patch_size
        assert height % patch_h == 0 and width % patch_w == 0
        h_patches = height // patch_h
        w_patches = width // patch_w
        patches_per_frame = h_patches * w_patches
        total_patches = t_frames * patches_per_frame
        patch_grid = (t_frames, h_patches, w_patches)
        device = x.device

        x = x.reshape(batch * t_frames, channels, height, width)
        x = self.conv1(x)
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(batch, total_patches, self.hidden_size)

        visible_indices, visible_mask_bool, ids_restore = self.mask_mae_style(
            batch_size=batch,
            t_frames=t_frames,
            patches_per_frame=patches_per_frame,
            mask_ratio=mask,
            device=device
        )
        n_visible = visible_indices.shape[1]

        gather_index = visible_indices.unsqueeze(-1).expand(-1, -1, self.hidden_size)
        x = torch.gather(x, 1, gather_index)

        rotary_pos_emb = self.tem_rot_pos_emb(torch.tensor(patch_grid, device=x.device),  self.head_dim)
        rotary_pos_emb = rotary_pos_emb.repeat(batch, 1)
        rotary_pos_emb = rotary_pos_emb[visible_indices]

        attention_mask = self._build_causal_temporal_mask(visible_indices, patches_per_frame)  # (B,N,N)
        
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)
        
        x = self.transformer(
            x,
            rotary_pos_emb=rotary_pos_emb,        # (B,N,D/2)
            attention_mask=attention_mask        # (B,N,N) or None
        )
        x = x.permute(1, 0, 2)

        if self.use_head:
            x = self.head(x)  # (B, hidden_size)

        return {
            "visible_embeddings": x,
            "mask": visible_mask_bool.float(),
            "ids_restore": ids_restore,
            "visible_indices": visible_indices,
            "num_visible": n_visible,
            "full_sequence_length": total_patches,
            "patch_grid": patch_grid,
            "attention_mask_used": attention_mask is not None,
        }

@register_model
def LLAVA_ViT_S_16_512_casual(pretrained: bool = False, **kwargs):
    model = VisualTransformer(
        patch_size=16,
        hidden_size=384,             # decoder hidden
        head_dim=64,
        num_hidden_layers=12,
        intermediate_size=1536,      # 384 * 4
        act_layer=nn.GELU,
        use_gradient_checkpointing=False,
        norm_cls=nn.RMSNorm,
        use_head=True
    )
    return model
