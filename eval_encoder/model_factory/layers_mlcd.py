import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import LayerNorm
from torch.utils.checkpoint import checkpoint
from typing import Callable, Optional


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_vision(tensor, freqs):
    orig_dtype = tensor.dtype
    tensor = tensor.float()
    if freqs.dim() == 2:
        cos = freqs.cos().unsqueeze(1).repeat(1, 1, 2).unsqueeze(0).float()
        sin = freqs.sin().unsqueeze(1).repeat(1, 1, 2).unsqueeze(0).float()
    else:
        cos = freqs.cos().repeat(1, 1, 2).unsqueeze(2).float()
        sin = freqs.sin().repeat(1, 1, 2).unsqueeze(2).float()
    output = (tensor * cos) + (rotate_half(tensor) * sin)
    return output.to(orig_dtype)


def apply_rotary_pos_emb_video_batched(q: torch.Tensor,
                                       k: torch.Tensor,
                                       freqs: torch.Tensor):
    """
    q,k: (B, L, H, D)
    freqs:
        - (B, L, D/2) per-sample
        - or (L, D/2) shared
    returns rotated q,k
    """
    if freqs.dim() == 2:          # shared
        freqs = freqs.unsqueeze(0)  # (1,L,D/2)
    B, L, H, D = q.shape
    assert freqs.shape[1] == L and freqs.shape[2] * 2 == D, "freqs shape mismatch"
    cos = freqs.cos()
    sin = freqs.sin()
    cos = torch.cat([cos, cos], dim=-1).unsqueeze(2)  # (B,L,1,D)
    sin = torch.cat([sin, sin], dim=-1).unsqueeze(2)
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k


class VisionRotaryEmbedding(nn.Module):
    def __init__(self, dim, theta=10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen):
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        return torch.outer(seq, self.inv_freq)


class VideoRotaryEmbeddingSplit466(nn.Module):
    """
    3D (T,H,W) Rotary 频率构造，按 4:6:6 切分 head_dim//2。
    t_size : 4 * base
    h_size : 6 * base
    w_size : 6 * base
    base = (head_dim//2) // 16
    """
    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even for rotary."
        assert head_dim % 16 == 0, "head_dim must be divisible by 16 (requested)."
        half = head_dim // 2
        assert half % 16 == 0, "head_dim//2 must also be divisible by 16 to split into 4:6:6."

        self.head_dim = head_dim
        self.half = half
        self.base = base

        unit = half // 16
        self.t_size = 4 * unit
        self.h_size = 6 * unit
        self.w_size = 6 * unit

        # 为每个轴单独构建 inv_freq（各自归一化到自身尺度）
        self.register_buffer(
            "inv_freq_t",
            1.0 / (base ** (torch.arange(self.t_size, dtype=torch.float32) / self.t_size)),
            persistent=False
        )
        self.register_buffer(
            "inv_freq_h",
            1.0 / (base ** (torch.arange(self.h_size, dtype=torch.float32) / self.h_size)),
            persistent=False
        )
        self.register_buffer(
            "inv_freq_w",
            1.0 / (base ** (torch.arange(self.w_size, dtype=torch.float32) / self.w_size)),
            persistent=False
        )

    @torch.no_grad()
    def forward(self, t: int, h: int, w: int, device=None, dtype=torch.float32):
        """
        返回:
            freqs: (L, half) 其中 L = t*h*w
        """
        if device is None:
            device = self.inv_freq_t.device

        inv_t = self.inv_freq_t.to(device=device, dtype=dtype)
        inv_h = self.inv_freq_h.to(device=device, dtype=dtype)
        inv_w = self.inv_freq_w.to(device=device, dtype=dtype)

        # 各轴外积
        ft = torch.outer(torch.arange(t, device=device, dtype=dtype), inv_t)  # (t, t_size)
        fh = torch.outer(torch.arange(h, device=device, dtype=dtype), inv_h)  # (h, h_size)
        fw = torch.outer(torch.arange(w, device=device, dtype=dtype), inv_w)  # (w, w_size)

        # 展平序列索引 (与之前风格一致)
        L = t * h * w
        t_ids = torch.arange(t, device=device).repeat_interleave(h * w)     # (L,)
        h_base = torch.arange(h, device=device).repeat_interleave(w)        # (h*w,)
        h_ids = h_base.repeat(t)                                            # (L,)
        w_base = torch.arange(w, device=device).repeat(h)                   # (h*w,)
        w_ids = w_base.repeat(t)                                            # (L,)

        freqs = torch.cat([ft[t_ids], fh[h_ids], fw[w_ids]], dim=-1)        # (L, half)
        assert freqs.shape == (L, self.half)
        return freqs


class VisionSdpaAttentionCausal(nn.Module):
    """
    扩展版本：支持 attention_mask (B,L,L) bool, True=不允许注意。
    用法与之前基本一致，只是多一个参数。
    """
    def __init__(self, hidden_size, num_attention_heads, attn_dropout=0.0):
        super().__init__()
        assert hidden_size % num_attention_heads == 0
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.in_proj = nn.Linear(hidden_size, hidden_size * 3)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.attn_dropout = attn_dropout

    def forward(self, hidden_states, rotary_pos_emb=None, attention_mask=None):
        # hidden_states: (L,B,C)
        L, B, C = hidden_states.shape
        qkv = self.in_proj(hidden_states).view(L, B, 3, self.num_attention_heads, self.head_dim).permute(2,1,0,3,4)
        q, k, v = qkv.unbind(0)  # (B,L,H,D)

        q = apply_rotary_pos_emb_vision(q, rotary_pos_emb)
        k = apply_rotary_pos_emb_vision(k, rotary_pos_emb)

        # (B,H,L,D)
        q = q.permute(0,2,1,3)
        k = k.permute(0,2,1,3)
        v = v.permute(0,2,1,3)

        if attention_mask is not None:
            # attention_mask: (B,L,L) True=禁止
            # PyTorch 允许 (B,1,L,L) 或 (B,L,L)
            attn_mask = attention_mask.unsqueeze(1)  # (B,1,L,L)
        else:
            attn_mask = None

        attn_out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout if self.training else 0.0
        )  # (B,H,L,D)
        attn_out = attn_out.permute(2,0,1,3).contiguous().view(L,B,C)
        return self.out_proj(attn_out)


class ResidualAttentionBlockCausal(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, intermediate_size,
                 act_layer=nn.GELU, attn_dropout=0.0, norm_cls: Callable=nn.LayerNorm):
        super().__init__()
        self.ln_1 = norm_cls(hidden_size)
        self.attn = VisionSdpaAttentionCausal(hidden_size, num_attention_heads, attn_dropout=attn_dropout)
        self.ln_2 = norm_cls(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            act_layer(),
            nn.Linear(intermediate_size, hidden_size),
        )

    def forward(self, x, rotary_pos_emb=None, attention_mask=None):
        x = x + self.attn(self.ln_1(x), rotary_pos_emb=rotary_pos_emb, attention_mask=attention_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class CrossAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v = nn.Linear(hidden_size, hidden_size, bias=False)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x_q, x_k=None, x_v=None):
        # x_*: (B, N, C)
        if x_k is None: x_k = x_q
        if x_v is None: x_v = x_k
        B, Nq, C = x_q.shape
        Nk = x_k.shape[1]

        q = self.q(x_q).view(B, Nq, self.num_heads, self.head_dim).transpose(1, 2)  # (B,H,Nq,D)
        k = self.k(x_k).view(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)  # (B,H,Nk,D)
        v = self.v(x_v).view(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)  # (B,H,Nk,D)

        attn = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)               # (B,H,Nq,D)
        attn = attn.transpose(1, 2).reshape(B, Nq, C)
        return self.proj(attn)


class AttentiveBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, norm_cls=nn.LayerNorm):
        super().__init__()
        self.norm_q = norm_cls(hidden_size)
        self.norm_k = norm_cls(hidden_size)
        self.norm_v = norm_cls(hidden_size)
        self.attn = CrossAttention(hidden_size, num_heads)

    def forward(self, x_q, x_kv, pos_q=None, pos_k=None):
        if pos_q is not None: x_q = x_q + pos_q
        if pos_k is not None: x_kv = x_kv + pos_k
        x_q = self.norm_q(x_q)
        x_k = self.norm_k(x_kv)
        x_v = self.norm_v(x_kv)
        return self.attn(x_q, x_k, x_v)


class AttentionPoolingBlock(AttentiveBlock):
    def forward(self, x):
        # x: (B,N,C) -> pooled (B,C)
        q = x.mean(dim=1, keepdim=True)
        out = super().forward(q, x)
        return out.squeeze(1)


class TransformerCausal(nn.Module):
    def __init__(self,
                 hidden_size,
                 num_hidden_layers,
                 num_attention_heads,
                 intermediate_size,
                 act_layer=nn.GELU,
                 gradient_checkpointing=False,
                 attn_dropout=0.0,
                 norm_cls: Callable=nn.LayerNorm):
        super().__init__()
        self.layers = nn.ModuleList([
            ResidualAttentionBlockCausal(
                hidden_size,
                num_attention_heads,
                intermediate_size,
                act_layer=act_layer,
                attn_dropout=attn_dropout,
                norm_cls=norm_cls
            ) for _ in range(num_hidden_layers)
        ])
        self.grad_checkpointing = gradient_checkpointing

    def forward(self, x, rotary_pos_emb=None, attention_mask=None):
        for blk in self.layers:
            blk: ResidualAttentionBlockCausal
            if self.grad_checkpointing and not torch.jit.is_scripting():
                def cf(t):
                    return blk(t, rotary_pos_emb=rotary_pos_emb, attention_mask=attention_mask)
                x = torch.utils.checkpoint.checkpoint(cf, x)
            else:
                x = blk(x, rotary_pos_emb=rotary_pos_emb, attention_mask=attention_mask)
        return x

class Siglip2MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.activation_fn = F.gelu
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states

class Siglip2MultiheadAttentionPoolingHead(nn.Module):
    """Multihead Attention Pooling."""

    def __init__(self, hidden_size, num_attention_heads, intermediate_size):
        super().__init__()

        self.probe = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.attention = torch.nn.MultiheadAttention(hidden_size, num_attention_heads, batch_first=True)
        self.norm = nn.RMSNorm(hidden_size, )
        self.mlp = Siglip2MLP(hidden_size, intermediate_size)
        self.num_heads = num_attention_heads

    def forward(self, hidden_state: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = hidden_state.shape[0]
        probe = self.probe.repeat(batch_size, 1, 1)

        hidden_state = self.attention(probe, hidden_state, hidden_state)[0]

        residual = hidden_state
        hidden_state = self.norm(hidden_state)
        hidden_state = residual + self.mlp(hidden_state)

        return hidden_state[:, 0]