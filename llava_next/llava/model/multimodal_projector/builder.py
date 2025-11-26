import torch
import torch.nn as nn
import re

from .pooler_projector import PoolerProjector


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": "identity"}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(nn.Linear(channels, channels), nn.GELU(), nn.Linear(channels, channels))

    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)

class PatchMerger(nn.Module):
    def __init__(self, llm_dim, vit_dim, spatial_merge_size=2):
        super().__init__()
        self.ln_q = torch.nn.LayerNorm(vit_dim, eps=1e-6)
        self.hidden_size = vit_dim * (spatial_merge_size**2)
        self.llm_dim = llm_dim

        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, llm_dim),
        )

    def forward(self, x):
        B, N, C = x.size()
        x = self.mlp(self.ln_q(x).view(B, -1, self.hidden_size))
        return x


class SpatialMergeProjector(nn.Module):
    """
    2x2 Spatial Merge Projector similar to Qwen2VL's merger layer.
    Merges 4 adjacent spatial tokens (2x2 grid) into 1 token.
    """
    def __init__(self, llm_dim, vit_dim, spatial_merge_size=2):
        super().__init__()
        self.spatial_merge_size = spatial_merge_size
        self.ln_q = nn.LayerNorm(vit_dim, eps=1e-6)
        self.hidden_size = vit_dim * (spatial_merge_size ** 2)
        self.llm_dim = llm_dim

        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, llm_dim),
        )

    def forward(self, x, *args, **kwargs):
        """
        Args:
            x: (B, N, C) where N = H * W (number of patches)
        Returns:
            (B, N // (spatial_merge_size^2), llm_dim)
        """
        B, N, C = x.size()
        # Assume square image: H = W = sqrt(N)
        H = W = int(round(N ** 0.5))
        assert H * W == N, f"Expected square grid, got N={N}"

        # Validate divisibility by merge_size
        merge_size = self.spatial_merge_size
        assert H % merge_size == 0 and W % merge_size == 0, \
            f"Grid size ({H}, {W}) not divisible by merge_size {merge_size}"

        # Apply LayerNorm
        x = self.ln_q(x)

        # Reshape to (B, H, W, C)
        x = x.view(B, H, W, C)

        # Merge 2x2 spatial patches
        # (B, H, W, C) -> (B, H//2, 2, W//2, 2, C) -> (B, H//2, W//2, 2, 2, C)
        new_H = H // merge_size
        new_W = W // merge_size
        x = x.view(B, new_H, merge_size, new_W, merge_size, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()  # (B, new_H, new_W, merge_size, merge_size, C)
        x = x.view(B, new_H * new_W, merge_size * merge_size * C)  # (B, N', hidden_size)

        # Project to LLM dimension
        x = self.mlp(x)
        return x

    @property
    def config(self):
        return {"mm_projector_type": "spatial_merge"}


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, "mm_projector_type", "linear")

    if projector_type == "linear":
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    if projector_type == "pooler":
        return PoolerProjector(config, kwargs["vision_cfg"])

    if projector_type == "patch_merger":
        return PatchMerger(
            llm_dim=config.hidden_size,
            vit_dim=config.mm_hidden_size,)

    if projector_type == "spatial_merge":
        return SpatialMergeProjector(
            llm_dim=config.hidden_size,
            vit_dim=config.mm_hidden_size,
            spatial_merge_size=2,
        )


    mlp_gelu_match = re.match(r"^mlp(\d+)x_gelu$", projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    mlp_gelu_resnet_match = re.match(r"^mlp(\d+)x_res(\d+)x_gelu$", projector_type)
    if mlp_gelu_resnet_match:
        mlp_depth = int(mlp_gelu_resnet_match.group(1))
        res_depth = int(mlp_gelu_resnet_match.group(2))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        for _ in range(res_depth):
            modules.append(SimpleResBlock(config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == "identity":
        return IdentityMap()

    raise ValueError(f"Unknown projector type: {projector_type}")
