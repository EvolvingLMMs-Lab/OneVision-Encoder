from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import LayerNorm
from torch.utils.checkpoint import checkpoint


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_vision(tensor, freqs):
    orig_dtype = tensor.dtype
    tensor = tensor.float()
    cos = freqs.cos().unsqueeze(1).repeat(1, 1, 2).unsqueeze(0).float()
    sin = freqs.sin().unsqueeze(1).repeat(1, 1, 2).unsqueeze(0).float()
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


class VideoRotaryEmbeddingSimple(nn.Module):
    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even"
        half = head_dim // 2
        assert half % 3 == 0, "head_dim//2 must be divisible by 3 (t/h/w equal split)"
        self.axis_size = half // 3
        self.head_dim = head_dim
        self.register_buffer(
            "inv_freq",
            1.0 / (base ** (torch.arange(0, self.axis_size, dtype=torch.float32) / self.axis_size)),
            persistent=False
        )

    def forward(self, t: int, h: int, w: int, device=None, dtype=torch.float32):
        if device is None:
            device = self.inv_freq.device
        inv = self.inv_freq.to(device=device, dtype=dtype)
        ft = torch.outer(torch.arange(t, device=device, dtype=dtype), inv)  # (t,a)
        fh = torch.outer(torch.arange(h, device=device, dtype=dtype), inv)  # (h,a)
        fw = torch.outer(torch.arange(w, device=device, dtype=dtype), inv)  # (w,a)

        t_ids = torch.arange(t, device=device).repeat_interleave(h * w)
        h_base = torch.arange(h, device=device).repeat_interleave(w)
        h_ids = h_base.repeat(t)
        w_base = torch.arange(w, device=device).repeat(h)
        w_ids = w_base.repeat(t)

        freqs = torch.cat([ft[t_ids], fh[h_ids], fw[w_ids]], dim=-1)  # (L, head_dim//2)
        return freqs


class VisionSdpaAttention(nn.Module):
    """
    Accepts batched rotary (B,L,D/2) for per-sample variable visible tokens (same L across batch here).
    """
    def __init__(self, hidden_size, num_attention_heads=16, attn_dropout=0.0):
        super().__init__()
        assert hidden_size % num_attention_heads == 0
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.in_proj = nn.Linear(hidden_size, hidden_size * 3, bias=True)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.attn_dropout = attn_dropout

    def forward(self, hidden_states, rotary_pos_emb=None):
        # hidden_states: (L, B, C)
        L, B, C = hidden_states.shape
        qkv = self.in_proj(hidden_states)  # (L,B,3C)
        qkv = qkv.view(L, B, 3, self.num_attention_heads, self.head_dim).permute(2, 1, 0, 3, 4)
        q, k, v = qkv.unbind(0)  # (B,L,H,D)

        if rotary_pos_emb is not None:
            q, k = apply_rotary_pos_emb_video_batched(q, k, rotary_pos_emb)  # (B,L,H,D)

        q = q.permute(0, 2, 1, 3).contiguous()  # (B,H,L,D)
        k = k.permute(0, 2, 1, 3).contiguous()
        v = v.permute(0, 2, 1, 3).contiguous()

        attn = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.attn_dropout if self.training else 0.0
        )  # (B,H,L,D)

        attn = attn.permute(2, 0, 1, 3).contiguous().view(L, B, C)
        return self.out_proj(attn)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, intermediate_size, act_layer=nn.GELU):
        super().__init__()
        self.ln_1 = LayerNorm(hidden_size)
        self.attn = VisionSdpaAttention(hidden_size, num_attention_heads)
        self.ln_2 = LayerNorm(hidden_size)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(hidden_size, intermediate_size)),
            ("act", act_layer()),
            ("c_proj", nn.Linear(intermediate_size, hidden_size)),
        ]))

    def forward(self, x, rotary_pos_emb=None):
        x = x + self.attn(self.ln_1(x), rotary_pos_emb=rotary_pos_emb)
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
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.norm_q = nn.LayerNorm(hidden_size)
        self.norm_k = nn.LayerNorm(hidden_size)
        self.norm_v = nn.LayerNorm(hidden_size)
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


class Transformer(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_hidden_layers,
        num_attention_heads,
        intermediate_size,
        act_layer=nn.GELU,
        gradient_checkpointing=False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.grad_checkpointing = gradient_checkpointing

        self.layers = nn.ModuleList([
            ResidualAttentionBlock(
                hidden_size,
                num_attention_heads,
                intermediate_size,
                act_layer=act_layer
            )
            for _ in range(num_hidden_layers)
        ])

    def enable_gradient_checkpointing(self, enabled=True):
        self.grad_checkpointing = enabled

    def forward(self, x, rotary_pos_emb=None):
        for blk in self.layers:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                def custom_forward(t):
                    return blk(t, rotary_pos_emb=rotary_pos_emb)
                x = checkpoint(custom_forward, x)
            else:
                x = blk(x, rotary_pos_emb=rotary_pos_emb)
        return x
