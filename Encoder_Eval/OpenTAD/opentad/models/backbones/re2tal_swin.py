import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.cuda.amp import custom_fwd, custom_bwd
from einops import rearrange

from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks import DropPath
from mmaction.registry import MODELS
from mmaction.models.backbones.swin import (
    PatchEmbed3D,
    PatchMerging,
    WindowAttention3D,
    Mlp,
    get_window_size,
    compute_mask,
    window_partition,
    window_reverse,
)


def seed_cuda():
    # randomize seeds, use cuda generator if available
    if hasattr(torch.cuda, "default_generators") and len(torch.cuda.default_generators) > 0:
        # GPU
        device_idx = torch.cuda.current_device()
        seed = torch.cuda.default_generators[device_idx].seed()
    else:
        # CPU
        seed = int(torch.seed() % sys.maxsize)
    return seed


def create_coupling(Fm, Gm=None, coupling="additive", implementation_fwd=-1, implementation_bwd=-1, split_dim=1):
    if coupling == "additive":
        fn = AdditiveCoupling(
            Fm,
            Gm,
            implementation_fwd=implementation_fwd,
            implementation_bwd=implementation_bwd,
            split_dim=split_dim,
        )
    else:
        raise NotImplementedError("Unknown coupling method: %s" % coupling)
    return fn


class AdditiveCoupling(nn.Module):
    def __init__(self, Fm, Gm=None, implementation_fwd=-1, implementation_bwd=-1, split_dim=1):
        """
        This computes the output :math:`y` on forward given input :math:`x` and arbitrary modules :math:`Fm` and :math:`Gm` according to:

        :math:`(x1, x2) = x`
        :math:`y1 = x1 + Fm(x2)`
        :math:`y2 = x2 + Gm(y1)`
        :math:`y = (y1, y2)`

        """
        super(AdditiveCoupling, self).__init__()

        self.Gm = Gm
        self.Fm = Fm
        self.split_dim = split_dim

    def forward(self, x, mask_matrix):
        x1, x2 = torch.chunk(x, 2, dim=self.split_dim)
        fmd = self.Fm(x2, mask_matrix)
        y1 = x1 + fmd
        del x1

        gmd = self.Gm(y1)
        y2 = x2 + gmd
        del x2

        out = torch.cat([y1, y2], dim=self.split_dim)
        return out

    def inverse(self, y, mask_matrix):
        y1, y2 = torch.chunk(y, 2, dim=self.split_dim)
        y1, y2 = y1.contiguous(), y2.contiguous()
        gmd = self.Gm.forward(y1)
        x2 = y2 - gmd
        fmd = self.Fm.forward(x2, mask_matrix)
        x1 = y1 - fmd
        x = torch.cat([x1, x2], dim=self.split_dim)

        return x

    def backward_pass_inverse(self, Y_1, Y_2, mask_matrix):
        """
        equations:
        Y_1 = X_1 + F(X_2), F = Attention
        Y_2 = X_2 + G(Y_1), G = MLP

        equations for recomputing the activations:
        X_2 = Y_2 - G(Y_1), G = MLP
        X_1 = Y_1 - F(X_2), F = Attention
        """
        with torch.enable_grad():
            Y_1.requires_grad = True
            g_Y_1 = self.Gm(Y_1)

        with torch.no_grad():
            X_2 = Y_2 - g_Y_1

        with torch.enable_grad():
            X_2.requires_grad = True
            f_X_2 = self.Fm(X_2, mask_matrix)

        with torch.no_grad():
            X_1 = Y_1 - f_X_2

        del Y_2

        # Return tensors to do back propagation on the graph.
        return X_1, X_2, Y_1, g_Y_1, f_X_2

    def backward_pass_grads(self, X_2, Y_1, g_Y_1, f_X_2, dY_1, dY_2):
        with torch.enable_grad():
            g_Y_1.backward(dY_2)
            del g_Y_1

        with torch.no_grad():
            dY_1 = dY_1 + Y_1.grad
            Y_1.grad = None
            del Y_1

        with torch.enable_grad():
            f_X_2.backward(dY_1)
            del f_X_2

        with torch.no_grad():
            dY_2 = dY_2 + X_2.grad
            X_2.grad = None

        return dY_1, dY_2

    def backward_pass(self, Y_1, Y_2, dY_1, dY_2, mask_matrix):
        """
        equations:
        Y_1 = X_1 + F(X_2), F = Attention
        Y_2 = X_2 + G(Y_1), G = MLP

        equations for recomputing of activations:
        X_2 = Y_2 - G(Y_1), G = MLP
        X_1 = Y_1 - F(X_2), F = Attention
        """
        with torch.enable_grad():
            Y_1.requires_grad = True
            g_Y_1 = self.Gm(Y_1)
            g_Y_1.backward(dY_2, retain_graph=True)

        with torch.no_grad():
            X_2 = Y_2 - g_Y_1
            del g_Y_1
            dY_1 = dY_1 + Y_1.grad
            Y_1.grad = None

        with torch.enable_grad():
            X_2.requires_grad = True
            f_X_2 = self.Fm(X_2, mask_matrix)
            f_X_2.backward(dY_1, retain_graph=True)

        with torch.no_grad():
            X_1 = Y_1 - f_X_2
            del f_X_2, Y_1
            dY_2 = dY_2 + X_2.grad
            X_2.grad = None
            X_2 = X_2.detach()

        return X_1, X_2, dY_1, dY_2


class RevBackProp(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, x, attn_mask, blocks):
        for blk in blocks:
            x = blk(x, attn_mask)

        all_tensors = [x.detach(), attn_mask.detach()]
        ctx.save_for_backward(*all_tensors)
        ctx.blocks = blocks
        return x

    @staticmethod
    @custom_bwd
    def backward(ctx, dy):
        # retrieve params from ctx for backward
        Y, mask_matrix = ctx.saved_tensors
        blocks = ctx.blocks

        dY_1, dY_2 = torch.chunk(dy, 2, dim=-1)
        Y_1, Y_2 = torch.chunk(Y, 2, dim=-1)

        for _, blk in enumerate(blocks[::-1]):
            Y_1, Y_2, dY_1, dY_2 = blk.backward_pass(Y_1=Y_1, Y_2=Y_2, dY_1=dY_1, dY_2=dY_2, mask_matrix=mask_matrix)

        dx = torch.cat([dY_1, dY_2], dim=-1)
        del Y_1, Y_2, dY_1, dY_2
        return dx, None, None


class RevBackPropStream(RevBackProp):
    @staticmethod
    def backward(ctx, dy):
        Y, mask_matrix = ctx.saved_tensors
        blocks = ctx.blocks

        dY_1, dY_2 = torch.chunk(dy, 2, dim=-1)
        Y_1, Y_2 = torch.chunk(Y, 2, dim=-1)

        # Run backward staggered on two streams
        s1 = torch.cuda.Stream(device=Y_1.device)
        s2 = torch.cuda.Stream(device=Y_1.device)
        torch.cuda.synchronize()

        # Initial pass
        with torch.cuda.stream(s1):
            blk = blocks[-1]
            X_1, X_2, Y_1, g_Y_1, f_X_2 = blk.backward_pass_inverse(Y_1=Y_1, Y_2=Y_2, mask_matrix=mask_matrix)
            s2.wait_stream(s1)

        # Stagger streams based on iteration
        for i, (this_blk, next_blk) in enumerate(zip(blocks[1:][::-1], blocks[:-1][::-1])):
            if i % 2 == 0:
                stream1 = s1
                stream2 = s2
            else:
                stream1 = s2
                stream2 = s1

            with torch.cuda.stream(stream1):
                dY_1, dY_2 = this_blk.backward_pass_grads(
                    X_2=X_2, Y_1=Y_1, g_Y_1=g_Y_1, f_X_2=f_X_2, dY_1=dY_1, dY_2=dY_2
                )
                stream2.wait_stream(stream1)
            del Y_1, g_Y_1, f_X_2
            with torch.cuda.stream(stream2):
                X_1, X_2, Y_1, g_Y_1, f_X_2 = next_blk.backward_pass_inverse(Y_1=X_1, Y_2=X_2, mask_matrix=mask_matrix)

                stream1.wait_stream(stream2)

        # Last iteration
        with torch.cuda.stream(stream2):
            dY_1, dY_2 = blocks[0].backward_pass_grads(X_2=X_2, Y_1=Y_1, g_Y_1=g_Y_1, f_X_2=f_X_2, dY_1=dY_1, dY_2=dY_2)

        torch.cuda.current_stream().wait_stream(s1)
        torch.cuda.current_stream().wait_stream(s2)

        dx = torch.cat([dY_1, dY_2], dim=-1)

        del X_1, X_2, Y_1, Y_2, dY_1, dY_2, g_Y_1, f_X_2
        return dx, None, None


class SwinTransformerBlock3D_F(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        window_size=(2, 7, 7),
        shift_size=(0, 0, 0),
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=None,
        norm_cfg=dict(type="LN"),
        with_cp=False,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.with_cp = with_cp
        self.drop_path = drop_path
        self.seeds = {}
        self.seeds["droppath"] = seed_cuda()

        assert 0 <= self.shift_size[0] < self.window_size[0], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[1] < self.window_size[1], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[2] < self.window_size[2], "shift_size must in 0-window_size"

        self.norm1 = build_norm_layer(norm_cfg, dim)[1]
        self.attn = WindowAttention3D(
            dim,
            window_size=self.window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

    def forward_part1(self, x, mask_matrix):
        B, D, H, W, C = x.shape
        window_size, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)

        x = self.norm1(x)
        # pad feature maps to multiples of window size
        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (window_size[0] - D % window_size[0]) % window_size[0]
        pad_b = (window_size[1] - H % window_size[1]) % window_size[1]
        pad_r = (window_size[2] - W % window_size[2]) % window_size[2]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
        _, Dp, Hp, Wp, _ = x.shape
        # cyclic shift
        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None
        # partition windows
        x_windows = window_partition(shifted_x, window_size)  # B*nW, Wd*Wh*Ww, C
        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # B*nW, Wd*Wh*Ww, C
        # merge windows
        attn_windows = attn_windows.view(-1, *(window_size + (C,)))
        shifted_x = window_reverse(attn_windows, window_size, B, Dp, Hp, Wp)  # B D' H' W' C
        # reverse cyclic shift
        if any(i > 0 for i in shift_size):
            x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
        else:
            x = shifted_x

        if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
            x = x[:, :D, :H, :W, :].contiguous()
        return x

    def forward(self, x, mask_matrix):
        if self.with_cp:
            x = checkpoint.checkpoint(self.forward_part1, x, mask_matrix)
        else:
            x = self.forward_part1(x, mask_matrix)
        torch.manual_seed(self.seeds["droppath"])
        x = self.drop_path(x)
        return x


class SwinTransformerBlock3D_G(nn.Module):
    def __init__(
        self,
        dim,
        mlp_ratio=4.0,
        drop=0.0,
        drop_path=None,
        act_cfg=dict(type="GELU"),
        norm_cfg=dict(type="LN"),
        with_cp=False,
    ):
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.with_cp = with_cp
        self.drop_path = drop_path
        self.seeds = {}
        self.seeds["droppath"] = seed_cuda()

        self.norm2 = build_norm_layer(norm_cfg, dim)[1]
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_cfg=act_cfg,
            drop=drop,
        )

    def forward_part2(self, x):
        torch.manual_seed(self.seeds["droppath"])
        return self.drop_path(self.mlp(self.norm2(x)))

    def forward(self, x):
        if self.with_cp:
            x = checkpoint.checkpoint(self.forward_part2, x)
        else:
            x = self.forward_part2(x)

        return x


class SwinTransformerBlock3D(nn.Module):
    """Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): Window size.
        shift_size (tuple[int]): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(
        self,
        dim,
        num_heads,
        window_size=(2, 7, 7),
        shift_size=(0, 0, 0),
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_cfg=dict(type="GELU"),
        norm_cfg=dict(type="LN"),
        with_cp=False,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.with_cp = with_cp

        drop_path_in = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        fm = SwinTransformerBlock3D_F(
            dim=dim,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=shift_size,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path_in,
            norm_cfg=norm_cfg,
            with_cp=with_cp,
        )
        gm = SwinTransformerBlock3D_G(
            dim=dim,
            mlp_ratio=mlp_ratio,
            drop=drop,
            drop_path=drop_path_in,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            with_cp=with_cp,
        )

        self.inv_block = create_coupling(Fm=fm, Gm=gm, coupling="additive", split_dim=4)

    def forward(self, x, mask_matrix):
        """Forward function.

        Args:
            x: Input feature, tensor size (B, D, H, W, C).
            mask_matrix: Attention mask for cyclic shift.
        """
        x = self.inv_block(x, mask_matrix)
        return x

    def backward_pass(self, Y_1, Y_2, dY_1, dY_2, mask_matrix):
        return self.inv_block.backward_pass(Y_1=Y_1, Y_2=Y_2, dY_1=dY_1, dY_2=dY_2, mask_matrix=mask_matrix)

    def backward_pass_inverse(self, Y_1, Y_2, mask_matrix):
        return self.inv_block.backward_pass_inverse(Y_1=Y_1, Y_2=Y_2, mask_matrix=mask_matrix)

    def backward_pass_grads(self, X_2, Y_1, g_Y_1, f_X_2, dY_1, dY_2):
        return self.inv_block.backward_pass_grads(X_2=X_2, Y_1=Y_1, g_Y_1=g_Y_1, f_X_2=f_X_2, dY_1=dY_1, dY_2=dY_2)


class BasicLayer(nn.Module):
    """A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (tuple[int]): Local window size. Default: (1,7,7).
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
    """

    def __init__(
        self,
        dim,
        depth,
        num_heads,
        window_size=(1, 7, 7),
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_cfg=dict(type="GELU"),
        norm_cfg=dict(type="LN"),
        downsample=None,
        with_cp=False,
        inv_mode="custom_backprop",
    ):
        super().__init__()
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.depth = depth
        self.with_cp = with_cp

        # build blocks
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock3D(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=(0, 0, 0) if (i % 2 == 0) else self.shift_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    with_cp=with_cp,
                )
                for i in range(depth)
            ]
        )
        self.inv_mode = inv_mode
        self.downsample = downsample
        if self.downsample is not None:
            self.downsample = downsample(embed_dims=dim, norm_cfg=norm_cfg)

    def forward(self, x):
        """Forward function.

        Args:
            x: Input feature, tensor size (B, C, D, H, W).
        """
        x = torch.cat((x, x), dim=1)

        # calculate attention mask for SW-MSA
        B, C, D, H, W = x.shape
        window_size, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)
        x = rearrange(x, "b c d h w -> b d h w c")
        Dp = int(np.ceil(D / window_size[0])) * window_size[0]
        Hp = int(np.ceil(H / window_size[1])) * window_size[1]
        Wp = int(np.ceil(W / window_size[2])) * window_size[2]
        attn_mask = compute_mask(Dp, Hp, Wp, window_size, shift_size, x.device)

        if self.inv_mode == "vanilla":
            x = self.compute_for_inv_blocks(x, attn_mask)
        elif self.inv_mode == "custom_backprop":
            x = RevBackProp.apply(x, attn_mask, self.blocks)
        elif self.inv_mode == "stream_backprop":
            x = RevBackPropStream.apply(x, attn_mask, self.blocks)

        x = x.view(B, D, H, W, -1)

        C = int(C / 2)
        x = (x[:, :, :, :, :C] + x[:, :, :, :, C:]) / 2

        if self.downsample is not None:
            x = self.downsample(x)
        x = rearrange(x, "b d h w c -> b c d h w")
        return x

    def compute_for_inv_blocks(self, x, attn_mask):
        for blk in self.blocks:
            x = blk(x, attn_mask)
        return x


@MODELS.register_module()
class SwinTransformer3D_inv(nn.Module):
    """Swin Transformer backbone.
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        patch_size (int | tuple(int)): Patch size. Default: (4,4,4).
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        depths (tuple[int]): Depths of each Swin Transformer stage.
        num_heads (tuple[int]): Number of attention head of each stage.
        window_size (int): Window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: Truee
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer: Normalization layer. Default: nn.LayerNorm.
        patch_norm (bool): If True, add normalization after patch embedding. Default: False.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
    """

    def __init__(
        self,
        patch_size=(4, 4, 4),
        in_chans=3,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=(2, 7, 7),
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        act_cfg=dict(type="GELU"),
        norm_cfg=dict(type="LN"),
        patch_norm=False,
        frozen_stages=-1,
        with_cp=False,
        inv_mode=["custom_backprop", "custom_backprop", "custom_backprop", "custom_backprop"],
        norm_eval=True,
    ):
        super().__init__()

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval
        self.window_size = window_size
        self.patch_size = patch_size

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed3D(
            patch_size=patch_size,
            in_channels=in_chans,
            embed_dims=embed_dim,
            norm_cfg=norm_cfg if self.patch_norm else None,
            conv_cfg=dict(type="Conv3d"),
        )

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2**i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                act_cfg=act_cfg,
                norm_cfg=norm_cfg,
                downsample=PatchMerging if i_layer < self.num_layers - 1 else None,
                with_cp=with_cp,
                inv_mode=inv_mode[i_layer],
            )
            self.layers.append(layer)

        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))

        # add a norm layer for each output
        self.norm3 = build_norm_layer(norm_cfg, self.num_features)[1]

    def forward(self, x):
        """Forward function."""
        x = self.patch_embed(x)

        x = self.pos_drop(x)
        for layer in self.layers:
            x = layer(x.contiguous())

        x = rearrange(x, "n c d h w -> n d h w c")
        x = self.norm3(x)
        x = rearrange(x, "n d h w c -> n c d h w")
        return x  # [b,c,t,h,w]

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(SwinTransformer3D_inv, self).train(mode)
        self._freeze_stages()

        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, nn.LayerNorm):
                    m.eval()
