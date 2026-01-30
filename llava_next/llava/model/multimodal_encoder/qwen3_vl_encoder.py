"""
Qwen3-VL Vision Encoder for LLaVA Framework
Supports specifying output layer for Qwen3-VL-4B ViT encoder

Reference: siglip2_naflex.py
"""

from typing import Optional, Tuple, Union, Dict, List
from dataclasses import dataclass
from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import os
import json

from transformers import PreTrainedModel, PretrainedConfig, AutoConfig, AutoImageProcessor
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from transformers.image_processing_utils import BatchFeature, BaseImageProcessor
from transformers.image_transforms import (
    convert_to_rgb,
    normalize,
    rescale,
    resize,
    to_channel_dimension_format,
)
from transformers.image_utils import (
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    infer_channel_dimension_format,
    is_scaled_image,
    make_flat_list_of_images,
    to_numpy_array,
    valid_images,
    validate_preprocess_arguments,
)
from transformers.utils import TensorType, logging

from llava.utils import rank0_print
from transformers import AutoImageProcessor

logger = logging.get_logger(__name__)

# ==================== Flash Attention Support ====================

try:
    from flash_attn import flash_attn_varlen_func
    from flash_attn.bert_padding import pad_input, unpad_input

    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    flash_attn_varlen_func = None


def is_flash_attn_available():
    """Check if Flash Attention is available"""
    return FLASH_ATTN_AVAILABLE


# ==================== Configuration ====================


@dataclass
class Qwen3VLVisionConfig(PretrainedConfig):
    """Qwen3-VL Vision Encoder configuration class"""

    model_type = "qwen3_vl_vision"

    def __init__(
        self,
        hidden_size: int = 1024,
        intermediate_size: int = 4096,
        num_heads: int = 16,
        depth: int = 24,
        patch_size: int = 16,
        temporal_patch_size: int = 2,
        spatial_merge_size: int = 2,
        in_channels: int = 3,
        num_position_embeddings: int = 2304,
        hidden_act: str = "gelu_pytorch_tanh",
        out_hidden_size: int = 2560,
        deepstack_visual_indexes: List[int] = None,
        layer_norm_eps: float = 1e-6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.depth = depth
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.spatial_merge_size = spatial_merge_size
        self.in_channels = in_channels
        self.num_position_embeddings = num_position_embeddings
        self.hidden_act = hidden_act
        self.out_hidden_size = out_hidden_size
        self.deepstack_visual_indexes = deepstack_visual_indexes or [5, 11, 17]
        self.layer_norm_eps = layer_norm_eps

        # Calculate head_dim
        self.head_dim = hidden_size // num_heads

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs):
        config_path = os.path.join(pretrained_model_name_or_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config_dict = json.load(f)
            return cls(**config_dict)
        else:
            raise FileNotFoundError(f"Config file not found at {config_path}")


# ==================== Image Processor ====================


class Qwen3VLImageProcessor(BaseImageProcessor):
    """Qwen3-VL Image Processor"""

    def __init__(
        self,
        do_resize: bool = True,
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        do_rescale: bool = True,
        rescale_factor: float = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[List[float]] = None,
        image_std: Optional[List[float]] = None,
        do_convert_rgb: bool = True,
        patch_size: int = 16,
        temporal_patch_size: int = 2,
        min_pixels: int = 256 * 28 * 28,  # minimum pixel count
        max_pixels: int = 1280 * 28 * 28,  # maximum pixel count
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.do_resize = do_resize
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else [0.5, 0.5, 0.5]
        self.image_std = image_std if image_std is not None else [0.5, 0.5, 0.5]
        self.do_convert_rgb = do_convert_rgb
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.merge_size = 2

    def smart_resize(
        self,
        height: int,
        width: int,
        factor: int = 28,
        min_pixels: int = 56 * 56,
        max_pixels: int = 14 * 14 * 4 * 1280,
    ) -> Tuple[int, int]:
        """Smart resize image dimensions to be divisible by factor and within pixel range"""
        if height < factor or width < factor:
            raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")

        h_bar = round(height / factor) * factor
        w_bar = round(width / factor) * factor

        if h_bar * w_bar > max_pixels:
            beta = math.sqrt((height * width) / max_pixels)
            h_bar = math.floor(height / beta / factor) * factor
            w_bar = math.floor(width / beta / factor) * factor
        elif h_bar * w_bar < min_pixels:
            beta = math.sqrt(min_pixels / (height * width))
            h_bar = math.ceil(height * beta / factor) * factor
            w_bar = math.ceil(width * beta / factor) * factor

        return h_bar, w_bar

    def preprocess(
        self,
        images: ImageInput,
        do_resize: Optional[bool] = None,
        resample: Optional[PILImageResampling] = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[List[float]] = None,
        image_std: Optional[List[float]] = None,
        do_convert_rgb: Optional[bool] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ) -> BatchFeature:
        do_resize = do_resize if do_resize is not None else self.do_resize
        resample = resample if resample is not None else self.resample
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        do_convert_rgb = do_convert_rgb if do_convert_rgb is not None else self.do_convert_rgb

        images = make_flat_list_of_images(images)

        if not valid_images(images):
            raise ValueError("Invalid image type.")

        if do_convert_rgb:
            images = [convert_to_rgb(image) for image in images]

        # Convert to numpy
        images = [to_numpy_array(image) for image in images]

        # Collect processed data
        all_pixel_values = []
        all_grid_thw = []

        for image in images:
            # Ensure channels last format
            if image.shape[0] == 3:  # channels first
                image = np.transpose(image, (1, 2, 0))

            height, width = image.shape[:2]

            # factor must be a multiple of patch_size * merge_size
            factor = self.patch_size * self.merge_size

            # Smart resize - even if do_resize=False, ensure size compatibility
            # Calculate target size (must be a multiple of factor)
            if do_resize:
                new_height, new_width = self.smart_resize(
                    height,
                    width,
                    factor=factor,
                    min_pixels=self.min_pixels,
                    max_pixels=self.max_pixels,
                )
            else:
                # Even without resize, ensure dimensions are multiples of factor
                new_height = (height // factor) * factor
                new_width = (width // factor) * factor
                # Ensure at least one patch
                new_height = max(new_height, factor)
                new_width = max(new_width, factor)

            # Only resize if dimensions change
            if new_height != height or new_width != width:
                from PIL import Image as PILImage

                pil_image = PILImage.fromarray(
                    (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
                )
                pil_image = pil_image.resize((new_width, new_height), PILImage.BICUBIC)
                image = np.array(pil_image).astype(np.float32)
                if image.max() > 1.0:
                    image = image / 255.0

            height, width = new_height, new_width

            # Rescale
            if do_rescale and image.max() > 1.0:
                image = image * rescale_factor

            # Normalize
            if do_normalize:
                image = (image - np.array(image_mean)) / np.array(image_std)

            # Convert to patches
            image = np.transpose(image, (2, 0, 1))  # (H, W, C) -> (C, H, W)

            # Calculate grid (must be divisible by merge_size)
            grid_t = 1  # static image
            grid_h = height // self.patch_size  # should be divisible by merge_size
            grid_w = width // self.patch_size  # should be divisible by merge_size

            # Handle temporal_patch_size
            if self.temporal_patch_size > 1:
                # Duplicate frames to satisfy temporal_patch_size requirement
                image = np.stack([image] * self.temporal_patch_size, axis=0)  # (T, C, H, W)
            else:
                # temporal_patch_size=1, just add time dimension
                image = image[np.newaxis, ...]  # (1, C, H, W)

            # Reshape image to patches
            # (T, C, H, W) -> (num_patches, C * temporal * patch_h * patch_w)
            patches = self._image_to_patches(image, grid_t, grid_h, grid_w)

            all_pixel_values.append(patches)
            all_grid_thw.append([grid_t, grid_h, grid_w])

        # Stack (for single image)
        pixel_values = np.concatenate(all_pixel_values, axis=0)
        grid_thw = np.array(all_grid_thw)

        return BatchFeature(
            data={
                "pixel_values": pixel_values,
                "grid_thw": grid_thw,
            },
            tensor_type=return_tensors,
        )

    def _image_to_patches(self, image: np.ndarray, grid_t: int, grid_h: int, grid_w: int) -> np.ndarray:
        """Reshape image to patches

        Args:
            image: (T, C, H, W) format image
            grid_t: Temporal grid count
            grid_h: Height grid count
            grid_w: Width grid count

        Returns:
            patches: (num_patches, patch_pixels) format
        """
        temporal_patch_size = self.temporal_patch_size
        patch_size = self.patch_size
        merge_size = self.merge_size

        T, C, H, W = image.shape

        # Reshape to patches
        # (T, C, H, W) -> (T/t, t, C, H/p, p, W/p, p)
        image = image.reshape(
            grid_t,
            temporal_patch_size,
            C,
            grid_h // merge_size,
            merge_size,
            patch_size,
            grid_w // merge_size,
            merge_size,
            patch_size,
        )
        # Permute: (grid_t, grid_h//m, grid_w//m, m, m, t, p, p, C)
        image = image.transpose(0, 3, 6, 4, 7, 1, 5, 8, 2)
        # Flatten
        image = image.reshape(-1, C * temporal_patch_size * patch_size * patch_size)

        return image


# ==================== Model Components ====================


class Qwen3VLVisionMLP(nn.Module):
    """MLP module for Vision Transformer"""

    def __init__(self, config: Qwen3VLVisionConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.linear_fc1 = nn.Linear(self.hidden_size, self.intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(self.intermediate_size, self.hidden_size, bias=True)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return self.linear_fc2(self.act_fn(self.linear_fc1(hidden_state)))


class Qwen3VLVisionPatchEmbed(nn.Module):
    """3D Patch Embedding module"""

    def __init__(self, config: Qwen3VLVisionConfig) -> None:
        super().__init__()
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.in_channels
        self.embed_dim = config.hidden_size

        kernel_size = [self.temporal_patch_size, self.patch_size, self.patch_size]
        self.proj = nn.Conv3d(self.in_channels, self.embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)
        return hidden_states


class Qwen3VLVisionRotaryEmbedding(nn.Module):
    """Rotary Position Embedding for Vision Transformer"""

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs


class Qwen3VLVisionPatchMerger(nn.Module):
    """Patch Merger module"""

    def __init__(self, config: Qwen3VLVisionConfig, use_postshuffle_norm: bool = False) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size * (config.spatial_merge_size**2)
        self.use_postshuffle_norm = use_postshuffle_norm
        self.norm = nn.LayerNorm(
            self.hidden_size if use_postshuffle_norm else config.hidden_size, eps=config.layer_norm_eps
        )
        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(self.hidden_size, config.out_hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x.view(-1, self.hidden_size) if self.use_postshuffle_norm else x).view(-1, self.hidden_size)
        x = self.linear_fc2(self.act_fn(self.linear_fc1(x)))
        return x


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_vision(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    q_embed = q_embed.to(orig_q_dtype)
    k_embed = k_embed.to(orig_k_dtype)
    return q_embed, k_embed


class Qwen3VLVisionAttention(nn.Module):
    """Multi-head Attention module for Vision Transformer - supports Flash Attention 2"""

    def __init__(self, config: Qwen3VLVisionConfig, use_flash_attn: bool = True) -> None:
        super().__init__()
        self.dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.dim // self.num_heads
        self.num_key_value_groups = 1
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True)
        self.proj = nn.Linear(self.dim, self.dim)
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = 0.0

        # Flash Attention config
        self.use_flash_attn = use_flash_attn and FLASH_ATTN_AVAILABLE
        if use_flash_attn and not FLASH_ATTN_AVAILABLE:
            logger.warning_once(
                "Flash Attention 2 is not available. Falling back to eager attention. "
                "Please install flash-attn: pip install flash-attn --no-build-isolation"
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]

        query_states, key_states, value_states = (
            self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        )

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

        if self.use_flash_attn:
            # Flash Attention 2 path
            attn_output = self._flash_attention_forward(query_states, key_states, value_states, cu_seqlens)
        else:
            # Eager Attention path
            attn_output = self._eager_attention_forward(query_states, key_states, value_states, cu_seqlens)

        attn_output = self.proj(attn_output)
        return attn_output

    def _flash_attention_forward(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        """Flash Attention 2 forward pass

        Uses flash_attn_varlen_func to handle variable length sequences
        """
        seq_length = query_states.shape[0]

        # query_states: (seq_len, num_heads, head_dim)
        # Need to convert to (total_tokens, num_heads, head_dim) format
        q = query_states.contiguous()
        k = key_states.contiguous()
        v = value_states.contiguous()

        # Calculate max sequence length
        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()

        # Call flash_attn_varlen_func
        # Input: (total_tokens, num_heads, head_dim)
        # Output: (total_tokens, num_heads, head_dim)
        attn_output = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            dropout_p=self.attention_dropout if self.training else 0.0,
            softmax_scale=self.scaling,
            causal=False,  # Vision does not need causal mask
        )

        # Output shape: (seq_len, num_heads, head_dim) -> (seq_len, hidden_dim)
        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        return attn_output

    def _eager_attention_forward(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        """Eager Attention forward pass (original implementation)"""
        seq_length = query_states.shape[0]

        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        splits = [torch.split(tensor, lengths.tolist(), dim=2) for tensor in (query_states, key_states, value_states)]

        attn_outputs = []
        for q, k, v in zip(*splits):
            attn_weights = torch.matmul(q, k.transpose(-1, -2)) * self.scaling
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
            attn_output = torch.matmul(attn_weights, v)
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_outputs.append(attn_output)

        attn_output = torch.cat(attn_outputs, dim=1)
        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        return attn_output


class Qwen3VLVisionBlock(nn.Module):
    """Transformer Block for Vision Transformer"""

    def __init__(self, config: Qwen3VLVisionConfig, use_flash_attn: bool = True) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attn = Qwen3VLVisionAttention(config, use_flash_attn=use_flash_attn)
        self.mlp = Qwen3VLVisionMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


# ==================== Main Vision Encoder ====================


class Qwen3VLVisionEncoder(PreTrainedModel):
    """Qwen3-VL Vision Encoder main model - supports Flash Attention 2"""

    config_class = Qwen3VLVisionConfig
    base_model_prefix = "visual"
    _no_split_modules = ["Qwen3VLVisionBlock"]
    supports_gradient_checkpointing = True

    def __init__(self, config: Qwen3VLVisionConfig, use_flash_attn: bool = True) -> None:
        super().__init__(config)
        self.config = config
        self.use_flash_attn = use_flash_attn and FLASH_ATTN_AVAILABLE

        if use_flash_attn and FLASH_ATTN_AVAILABLE:
            logger.info("Using Flash Attention 2 for vision encoder")
        elif use_flash_attn and not FLASH_ATTN_AVAILABLE:
            logger.warning("Flash Attention 2 requested but not available, falling back to eager attention")

        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size

        self.patch_embed = Qwen3VLVisionPatchEmbed(config)
        self.pos_embed = nn.Embedding(config.num_position_embeddings, config.hidden_size)
        self.num_grid_per_side = int(config.num_position_embeddings**0.5)

        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = Qwen3VLVisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.ModuleList(
            [Qwen3VLVisionBlock(config, use_flash_attn=self.use_flash_attn) for _ in range(config.depth)]
        )

        self.merger = Qwen3VLVisionPatchMerger(config, use_postshuffle_norm=False)

        self.deepstack_visual_indexes = config.deepstack_visual_indexes
        self.deepstack_merger_list = nn.ModuleList(
            [
                Qwen3VLVisionPatchMerger(config, use_postshuffle_norm=True)
                for _ in range(len(config.deepstack_visual_indexes))
            ]
        )

        self.gradient_checkpointing = False

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        merge_size = self.spatial_merge_size
        max_hw = int(grid_thw[:, 1:].max().item())
        freq_table = self.rotary_pos_emb(max_hw)
        device = freq_table.device

        total_tokens = int(torch.prod(grid_thw, dim=1).sum().item())
        pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)

        offset = 0
        for num_frames, height, width in grid_thw:
            merged_h, merged_w = height // merge_size, width // merge_size
            block_rows = torch.arange(merged_h, device=device)
            block_cols = torch.arange(merged_w, device=device)
            intra_row = torch.arange(merge_size, device=device)
            intra_col = torch.arange(merge_size, device=device)

            row_idx = block_rows[:, None, None, None] * merge_size + intra_row[None, None, :, None]
            col_idx = block_cols[None, :, None, None] * merge_size + intra_col[None, None, None, :]

            row_idx = row_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
            col_idx = col_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)

            coords = torch.stack((row_idx, col_idx), dim=-1)
            if num_frames > 1:
                coords = coords.repeat(num_frames, 1)

            num_tokens = coords.shape[0]
            pos_ids[offset : offset + num_tokens] = coords
            offset += num_tokens

        embeddings = freq_table[pos_ids]
        embeddings = embeddings.flatten(1)
        return embeddings

    def fast_pos_embed_interpolate(self, grid_thw: torch.Tensor) -> torch.Tensor:
        grid_ts, grid_hs, grid_ws = grid_thw[:, 0], grid_thw[:, 1], grid_thw[:, 2]
        device = grid_thw.device

        idx_list = [[] for _ in range(4)]
        weight_list = [[] for _ in range(4)]

        for t, h, w in zip(grid_ts, grid_hs, grid_ws):
            h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h)
            w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w)

            h_idxs_floor = h_idxs.int()
            w_idxs_floor = w_idxs.int()
            h_idxs_ceil = (h_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
            w_idxs_ceil = (w_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)

            dh = h_idxs - h_idxs_floor
            dw = w_idxs - w_idxs_floor

            base_h = h_idxs_floor * self.num_grid_per_side
            base_h_ceil = h_idxs_ceil * self.num_grid_per_side

            indices = [
                (base_h[None].T + w_idxs_floor[None]).flatten(),
                (base_h[None].T + w_idxs_ceil[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_floor[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),
            ]

            weights = [
                ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
                ((1 - dh)[None].T * dw[None]).flatten(),
                (dh[None].T * (1 - dw)[None]).flatten(),
                (dh[None].T * dw[None]).flatten(),
            ]

            for i in range(4):
                idx_list[i].extend(indices[i].tolist())
                weight_list[i].extend(weights[i].tolist())

        idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=device)
        weight_tensor = torch.tensor(weight_list, dtype=self.pos_embed.weight.dtype, device=device)
        pos_embeds = self.pos_embed(idx_tensor).to(device) * weight_tensor[:, :, None]
        patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]

        patch_pos_embeds = patch_pos_embeds.split([h * w for h, w in zip(grid_hs, grid_ws)])

        patch_pos_embeds_permute = []
        merge_size = self.config.spatial_merge_size
        for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws):
            pos_embed = pos_embed.repeat(t, 1)
            pos_embed = (
                pos_embed.view(t, h // merge_size, merge_size, w // merge_size, merge_size, -1)
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            patch_pos_embeds_permute.append(pos_embed)
        patch_pos_embeds = torch.cat(patch_pos_embeds_permute)
        return patch_pos_embeds

    def forward(
        self,
        hidden_states: torch.Tensor,
        grid_thw: torch.Tensor,
        output_hidden_states: bool = True,
        output_layer_idx: Optional[int] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], Optional[Tuple[torch.Tensor, ...]]]:
        """
        Forward pass

        Args:
            hidden_states: Input pixel values
            grid_thw: (num_images, 3) - for each image (temporal, height, width)
            output_hidden_states: Whether to output all hidden states
            output_layer_idx: Specify output layer index (e.g., -1 for last layer, -2 for second to last)

        Returns:
            last_hidden_state: Final output (after merger)
            deepstack_features: Deep Stack feature list
            hidden_states: Hidden states from all layers
        """
        hidden_states = self.patch_embed(hidden_states)
        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
        hidden_states = hidden_states + pos_embeds

        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0, dtype=torch.int32
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        all_hidden_states = () if output_hidden_states else None
        deepstack_feature_lists = []

        for layer_num, blk in enumerate(self.blocks):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
                **kwargs,
            )

            if layer_num in self.deepstack_visual_indexes:
                deepstack_feature = self.deepstack_merger_list[self.deepstack_visual_indexes.index(layer_num)](
                    hidden_states
                )
                deepstack_feature_lists.append(deepstack_feature)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # Select output based on output_layer_idx
        if output_layer_idx is not None and all_hidden_states is not None:
            selected_hidden_state = all_hidden_states[output_layer_idx]
        else:
            selected_hidden_state = hidden_states

        last_hidden_state = self.merger(selected_hidden_state)

        return last_hidden_state, deepstack_feature_lists, all_hidden_states


# ==================== LLaVA Vision Tower ====================


class Qwen3VLVisionTower(nn.Module):
    """
    Qwen3-VL Vision Tower for LLaVA Framework

    Supports:
    - Specify output layer (output_layer_idx)
    - Optional merger usage
    - Deep Stack feature output
    - Flash Attention 2
    """

    def __init__(
        self,
        vision_tower: str,
        vision_tower_cfg=None,
        delay_load: bool = False,
        output_layer_idx: int = -1,  # Use last layer by default
        use_merger: bool = False,  # Optional merger usage for projection
        use_flash_attn: bool = True,  # Whether to use Flash Attention 2
        **kwargs,
    ):
        super().__init__()

        self.is_loaded = False
        self.vision_tower_name = vision_tower
        self.select_layer = vision_tower_cfg.mm_vision_select_layer
        self.output_layer_idx = self.select_layer
        self.use_merger = use_merger
        self.use_flash_attn = use_flash_attn

        # Load config
        self.config = self._load_config(vision_tower)

        # Create image processor - use AutoImageProcessor to load only image processor, no tokenizer needed
        self.image_processor = AutoImageProcessor.from_pretrained(vision_tower)
        # self.image_processor = Qwen3VLImageProcessor(
        #     patch_size=self.config.patch_size,
        #     temporal_patch_size=self.config.temporal_patch_size,
        # )

        if not delay_load:
            rank0_print(f"Loading Qwen3-VL vision tower: {vision_tower}")
            self.load_model()
        elif getattr(vision_tower_cfg, "unfreeze_mm_vision_tower", False):
            rank0_print(f"The checkpoint contains `vision_tower` weights: `unfreeze_mm_vision_tower`: True.")
            self.load_model()
        elif hasattr(vision_tower_cfg, "mm_tunable_parts") and "mm_vision_tower" in vision_tower_cfg.mm_tunable_parts:
            rank0_print(
                f"The checkpoint contains `vision_tower` weights: `mm_tunable_parts` contains `mm_vision_tower`."
            )
            self.load_model()
        else:
            self.cfg_only = self.config

    def _load_config(self, vision_tower_path: str) -> Qwen3VLVisionConfig:
        """Load config file"""
        config_path = os.path.join(vision_tower_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config_dict = json.load(f)
            return Qwen3VLVisionConfig(**config_dict)
        else:
            # Use default config
            rank0_print(f"Config not found at {config_path}, using default Qwen3-VL-4B config")
            return Qwen3VLVisionConfig()

    def load_model(self, device_map=None):
        if self.is_loaded:
            rank0_print(f"{self.vision_tower_name} is already loaded, skipping.")
            return

        # Create model - supports Flash Attention 2
        self.vision_tower = Qwen3VLVisionEncoder(self.config, use_flash_attn=self.use_flash_attn)

        # Load weights
        weights_path = os.path.join(self.vision_tower_name, "model.safetensors")
        if os.path.exists(weights_path):
            from safetensors.torch import load_file

            state_dict = load_file(weights_path)
            self.vision_tower.load_state_dict(state_dict)
            rank0_print(f"Loaded weights from {weights_path}")
        else:
            rank0_print(f"Warning: Weights not found at {weights_path}")

        self.is_loaded = True

    def forward(
        self,
        images: Union[torch.Tensor, Dict],
        grid_thw: Optional[torch.Tensor] = None,
        output_layer_idx: Optional[int] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            images: Input images, can be:
                - dict: {'pixel_values': tensor, 'grid_thw': tensor}
                - tensor: Preprocessed patches, supports the following formats:
                    - (total_patches, patch_dim): All patches concatenated
                    - (batch, num_patches, patch_dim): batch format
            grid_thw: Grid info (batch, 3) or (num_images, 3)
            output_layer_idx: Override default output layer index

        Returns:
            image_features: Image features
                - If input is (batch, num_patches, dim)，output is (batch, num_tokens, hidden_dim)
                - If input is (total_patches, dim)，output is (total_tokens, hidden_dim)
        """
        layer_idx = output_layer_idx if output_layer_idx is not None else self.output_layer_idx
        # assert 3==4, f'images.shape={images.shape},grid_thw={grid_thw}'
        # Record whether this is batch format input
        is_batch_format = False
        batch_size = None
        num_patches_per_image = None

        if isinstance(images, dict):
            # Dict input
            pixel_values = images["pixel_values"].to(device=self.device, dtype=self.dtype)
            grid_thw = images["grid_thw"].to(device=self.device)
        elif isinstance(images, torch.Tensor):
            pixel_values = images.to(device=self.device, dtype=self.dtype)

            # Check input dimensions
            if pixel_values.dim() == 3:
                # (batch, num_patches, patch_dim) format
                is_batch_format = True
                batch_size = pixel_values.shape[0]
                num_patches_per_image = pixel_values.shape[1]

                # Infer grid dimensions
                # num_patches = grid_t * grid_h * grid_w
                # Assume grid_t = 1, grid_h = grid_w (square)
                # num_patches = grid_h * grid_w
                # So grid_h = grid_w = sqrt(num_patches)
                side = int(math.sqrt(num_patches_per_image))

                if grid_thw is None:
                    # Create grid_thw - use patch count directly
                    grid_thw = torch.tensor([[1, side, side]] * batch_size, device=self.device, dtype=torch.long)
                else:
                    grid_thw = grid_thw.to(device=self.device)

                # Reshape (batch, num_patches, dim) to (total_patches, dim)
                pixel_values = pixel_values.reshape(-1, pixel_values.shape[-1])

            elif pixel_values.dim() == 2:
                # (total_patches, patch_dim) format - original format
                if grid_thw is None:
                    # Try to infer grid_thw (assume single image)
                    num_patches = pixel_values.shape[0]
                    side = int(math.sqrt(num_patches))
                    grid_thw = torch.tensor([[1, side, side]], device=self.device)
                else:
                    grid_thw = grid_thw.to(device=self.device)
            else:
                raise ValueError(f"Unsupported tensor dimension: {pixel_values.dim()}, expected 2 or 3")
        else:
            raise ValueError(f"Unsupported input type: {type(images)}")

        # Forward pass
        last_hidden_state, deepstack_features, all_hidden_states = self.vision_tower(
            pixel_values,
            grid_thw,
            output_hidden_states=True,
            output_layer_idx=layer_idx if not self.use_merger else None,
        )

        if self.use_merger:
            # Use merger output
            image_features = last_hidden_state
        else:
            # Use specified layer raw output (without merger)
            if all_hidden_states is not None:
                image_features = all_hidden_states[layer_idx]
            else:
                image_features = last_hidden_state

        # If input is batch format, also convert output back to batch format
        if is_batch_format and batch_size is not None:
            # Calculate token count for each image
            # Merger will merge merge_size x merge_size patches into 1 token
            merge_size = self.config.spatial_merge_size
            tokens_per_image = num_patches_per_image  # token count unchanged after merger (already merged patches)

            total_tokens = image_features.shape[0]
            tokens_per_image = total_tokens // batch_size

            image_features = image_features.reshape(batch_size, tokens_per_image, -1)

        # Inverse transform: restore 2x2 merge reordered features to original patch order
        # Qwen2-VL preprocessing order is: (batch, grid_t, grid_h//m, grid_w//m, m, m, ...)
        # i.e., arranged by 2x2 blocks, need to restore to row-by-row scanning order
        image_features = self._reverse_merge_reorder(image_features, grid_thw)

        return image_features

    def _reverse_merge_reorder(self, features: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """
        Inverse transform: restore 2x2 merge reordered features to original patch order

        Qwen2-VL preprocessing reorder logic:
            patches.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
            i.e.: (batch, grid_t, grid_h//m, grid_w//m, m, m, channel, temporal, patch_h, patch_w)

        This means patches are arranged in the following order:
            - First iterate grid_t
            - Then iterate grid_h // merge_size (block rows)
            - Then iterate grid_w // merge_size (block cols)
            - Then iterate merge_size x merge_size within each block

        We need to restore to standard row-by-row scanning order:
            - First iterate grid_t
            - Then iterate grid_h (row by row)
            - Then iterate grid_w (column by column)

        Args:
            features: (batch, num_tokens, hidden_dim) or (total_tokens, hidden_dim)
            grid_thw: (num_images, 3) - for each image (temporal, height, width)

        Returns:
            reordered_features: restored to original order features
        """
        merge_size = self.config.spatial_merge_size

        # Check input dimensions
        if features.dim() == 2:
            # (total_tokens, hidden_dim) format - need to split by grid_thw
            return self._reverse_merge_reorder_flat(features, grid_thw, merge_size)
        elif features.dim() == 3:
            # (batch, num_tokens, hidden_dim) format
            return self._reverse_merge_reorder_batch(features, grid_thw, merge_size)
        else:
            raise ValueError(f"Unsupported features dimension: {features.dim()}")

    def _reverse_merge_reorder_batch(
        self, features: torch.Tensor, grid_thw: torch.Tensor, merge_size: int
    ) -> torch.Tensor:
        """Process inverse transform for batch format"""
        batch_size, num_tokens, hidden_dim = features.shape

        reordered_list = []
        for i in range(batch_size):
            grid_t, grid_h, grid_w = grid_thw[i].tolist()
            feat = features[i]  # (num_tokens, hidden_dim)

            # Calculate block count
            block_h = grid_h // merge_size
            block_w = grid_w // merge_size

            # Reshape to block structure: (grid_t, block_h, block_w, merge_size, merge_size, hidden_dim)
            feat = feat.view(grid_t, block_h, block_w, merge_size, merge_size, hidden_dim)

            # Reverse transform: from (t, bh, bw, mh, mw, d) to (t, bh, mh, bw, mw, d)
            # Then reshape to (t, grid_h, grid_w, d)
            feat = feat.permute(0, 1, 3, 2, 4, 5)  # (t, bh, mh, bw, mw, d)
            feat = feat.reshape(grid_t, grid_h, grid_w, hidden_dim)

            # Flatten back to (num_tokens, hidden_dim)
            feat = feat.reshape(-1, hidden_dim)
            reordered_list.append(feat)

        # Stack back to batch format
        return torch.stack(reordered_list, dim=0)

    def _reverse_merge_reorder_flat(
        self, features: torch.Tensor, grid_thw: torch.Tensor, merge_size: int
    ) -> torch.Tensor:
        """Process inverse transform for flat format (total_tokens, hidden_dim)"""
        hidden_dim = features.shape[-1]

        # Calculate token count for each image
        tokens_per_image = (grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]).tolist()

        # Split by image
        feat_list = torch.split(features, tokens_per_image, dim=0)

        reordered_list = []
        for feat, (grid_t, grid_h, grid_w) in zip(feat_list, grid_thw.tolist()):
            grid_t, grid_h, grid_w = int(grid_t), int(grid_h), int(grid_w)

            # Calculate block count
            block_h = grid_h // merge_size
            block_w = grid_w // merge_size

            # Reshape to block structure
            feat = feat.view(grid_t, block_h, block_w, merge_size, merge_size, hidden_dim)

            # Inverse transform
            feat = feat.permute(0, 1, 3, 2, 4, 5)  # (t, bh, mh, bw, mw, d)
            feat = feat.reshape(grid_t, grid_h, grid_w, hidden_dim)

            # Flatten
            feat = feat.reshape(-1, hidden_dim)
            reordered_list.append(feat)

        return torch.cat(reordered_list, dim=0)

    def get_deepstack_features(
        self,
        images: Union[torch.Tensor, Dict],
        grid_thw: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """Get Deep Stack features"""
        if isinstance(images, dict):
            pixel_values = images["pixel_values"].to(device=self.device, dtype=self.dtype)
            grid_thw = images["grid_thw"].to(device=self.device)
        else:
            pixel_values = images.to(device=self.device, dtype=self.dtype)
            grid_thw = grid_thw.to(device=self.device)

        _, deepstack_features, _ = self.vision_tower(
            pixel_values,
            grid_thw,
            output_hidden_states=False,
        )
        return deepstack_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        for p in self.vision_tower.parameters():
            return p.dtype
        return torch.float32

    @property
    def device(self):
        for p in self.vision_tower.parameters():
            return p.device
        return torch.device("cpu")

    @property
    def hidden_size(self):
        """Return output dimension"""
        if self.use_merger:
            return self.config.out_hidden_size
        else:
            return self.config.hidden_size

    @property
    def num_patches(self):
        """Default patch count"""
        return (448 // self.config.patch_size) ** 2

    @property
    def num_patches_per_side(self):
        return 448 // self.config.patch_size

    @property
    def image_size(self):
        return 448
