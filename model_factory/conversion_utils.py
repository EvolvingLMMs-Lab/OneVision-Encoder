#!/usr/bin/env python3
# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Common utilities for model conversion scripts.

This module contains shared functionality used across multiple conversion scripts:
- convert_llava_vit_to_hf.py
- convert_llava_vit_packing_to_hf.py
- convert_vit_preview_v0_hf_to_packing.py
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import requests
from io import BytesIO
from torchvision import transforms
from transformers import CLIPImageProcessor


# CLIP Specific Constants
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


def get_real_coco_image(size=448):
    """
    Download a real COCO image and preprocess to Tensor (using CLIP mean/std, Float32)
    
    Args:
        size: Target image size (default: 448)
    
    Returns:
        Tensor of shape [1, 3, size, size]
    """
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"  # COCO cat image
    print(f"--> Downloading real image from {url} (Target Size: {size})...")
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert("RGB")
    except Exception as e:
        print(f"[Error] Failed to download image: {e}. Generating random noise as fallback.")
        img = Image.fromarray(np.random.randint(0, 255, (size, size, 3), dtype=np.uint8))

    transform = transforms.Compose([
        transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])

    return transform(img).unsqueeze(0)  # [1, 3, size, size]


def interpolate_frame_indices(frame_indices: torch.Tensor, total_frames: torch.Tensor, target_frames: int = 64) -> torch.Tensor:
    """
    Interpolate frame indices from original video frame count to target frame count.
    
    Args:
        frame_indices: [B, seq_len] Original frame indices
        total_frames: [B] Total frames for each video
        target_frames: Target frame count (default 64)
    
    Returns:
        interpolated_indices: [B, seq_len] Interpolated frame indices in range [0, target_frames-1]
    """
    bs, seq_len = frame_indices.shape
    device = frame_indices.device

    total_frames_float = total_frames.float().view(bs, 1)
    frame_indices_float = frame_indices.float()

    # Interpolation formula: new_idx = (old_idx / (total_frames - 1)) * (target_frames - 1)
    # Handle total_frames = 1 case
    total_frames_safe = torch.clamp(total_frames_float - 1, min=1.0)
    interpolated_indices = (frame_indices_float / total_frames_safe) * (target_frames - 1)

    # Round and convert to integer
    interpolated_indices = torch.round(interpolated_indices).long()

    # Ensure indices are in valid range
    interpolated_indices = torch.clamp(interpolated_indices, 0, target_frames - 1)

    return interpolated_indices


def get_synthesized_video(real_image_tensor, num_frames=8, size=224):
    """
    Create a synthesized video by stacking the real image multiple times.
    
    Args:
        real_image_tensor: Real image tensor of shape (1, C, H, W)
        num_frames: Number of frames to create
        size: Target size for each frame
    
    Returns:
        video_tensor: Synthesized video tensor of shape (1, C, T, H, W)
    """
    # Resize the image to target size if needed
    if real_image_tensor.shape[-1] != size or real_image_tensor.shape[-2] != size:
        real_image_tensor = F.interpolate(
            real_image_tensor.float(),
            size=(size, size),
            mode='bicubic',
            align_corners=False
        )

    # Stack the image to create video frames: (1, C, H, W) -> (1, C, T, H, W)
    video_tensor = real_image_tensor.unsqueeze(2).repeat(1, 1, num_frames, 1, 1)

    return video_tensor


def compute_patch_positions_with_interpolated_temporal(
    interpolated_indices: torch.Tensor,
    h_patches: int,
    w_patches: int,
    device: torch.device
) -> torch.Tensor:
    """
    Compute patch positions with interpolated temporal positions for RoPE.
    
    This function computes patch positions where the temporal positions are
    based on the interpolated frame indices, matching the src_model's RoPE
    positions when using visible_index.
    
    Args:
        interpolated_indices: [B, num_frames] Interpolated frame indices in 64-frame context
        h_patches: Number of patches in height dimension
        w_patches: Number of patches in width dimension
        device: Target device
    
    Returns:
        patch_positions: Tensor of shape (total_patches, 3) with [t, h, w] positions
    """
    bs, num_frames = interpolated_indices.shape
    total_patches = bs * num_frames * h_patches * w_patches
    
    # Pre-allocate tensor for better performance
    positions = torch.zeros((total_patches, 3), dtype=torch.long, device=device)
    
    idx = 0
    for b in range(bs):
        for frame_idx in range(num_frames):
            # Get the interpolated temporal position (in 64-frame context)
            t_pos = interpolated_indices[b, frame_idx].item()
            
            # Generate spatial positions for this frame
            for h in range(h_patches):
                for w in range(w_patches):
                    positions[idx, 0] = t_pos
                    positions[idx, 1] = h
                    positions[idx, 2] = w
                    idx += 1
    
    return positions


def compute_cosine_similarity(feat1, feat2, name="Feature"):
    """
    Compute cosine similarity between two feature tensors and report statistics.
    
    Args:
        feat1: First feature tensor
        feat2: Second feature tensor
        name: Name of the feature for reporting
    
    Returns:
        dict with 'min_cos', 'mean_cos', 'max_diff', 'pass' (bool)
    """
    # Convert to float for better precision
    feat1_flat = feat1.flatten(0, -2).float() if feat1.dim() > 2 else feat1.float()
    feat2_flat = feat2.flatten(0, -2).float() if feat2.dim() > 2 else feat2.float()
    
    # Handle shape mismatch
    if feat1_flat.shape[0] != feat2_flat.shape[0]:
        print(f"    [Warning] {name} shape mismatch: {feat1_flat.shape} vs {feat2_flat.shape}")
        min_len = min(feat1_flat.shape[0], feat2_flat.shape[0])
        feat1_flat = feat1_flat[:min_len]
        feat2_flat = feat2_flat[:min_len]
    
    # Compute differences
    diff = (feat1_flat - feat2_flat).abs().max().item()
    cos_sim = F.cosine_similarity(feat1_flat, feat2_flat, dim=-1)
    
    min_cos = cos_sim.min().item()
    mean_cos = cos_sim.mean().item()
    
    # Report results
    print(f"    [{name}] Max Diff:       {diff:.6f}")
    print(f"    [{name}] Min Cosine Sim: {min_cos:.8f} (Mean: {mean_cos:.8f})")
    
    passed = min_cos > 0.99
    print(f"    {'✅' if passed else '❌'} {name}: {'PASS' if passed else 'FAIL'}")
    
    return {
        'min_cos': min_cos,
        'mean_cos': mean_cos,
        'max_diff': diff,
        'pass': passed
    }


def save_model_with_processor(model, output_dir, image_size=448) -> bool:
    """
    Save a model along with its CLIPImageProcessor configuration.
    
    Args:
        model: Model to save (must have save_pretrained method)
        output_dir: Output directory path
        image_size: Target image size for the processor
    
    Returns:
        bool: True if save was successful, False otherwise
    
    Note:
        This function does not catch exceptions from save operations.
        Callers should handle filesystem errors (permissions, disk space, etc.)
    """
    if not hasattr(model, "save_pretrained"):
        print("❌ Error: Model does not have save_pretrained method.")
        return False
    
    print(f"\n--> Saving Model to {output_dir}...")
    model.save_pretrained(output_dir)
    
    print(f"    Saving CLIPImageProcessor config (CLIP Defaults + {image_size})...")
    processor = CLIPImageProcessor(
        size=image_size,
        crop_size=image_size,
        image_mean=CLIP_MEAN,
        image_std=CLIP_STD,
        resample=3,
        do_center_crop=True,
        do_normalize=True,
        do_resize=True,
        feature_extractor_type="CLIPFeatureExtractor"
    )
    processor.save_pretrained(output_dir)
    
    print("✅ Model and CLIP Processor saved.")
    return True


def move_model_to_device(model, dtype=torch.bfloat16) -> torch.device:
    """
    Move model to CUDA if available, otherwise CPU, and cast to specified dtype.
    
    Args:
        model: Model to move (modified in-place)
        dtype: Target dtype (default: torch.bfloat16)
    
    Returns:
        torch.device: The device the model was moved to (for informational purposes)
    
    Note:
        - Model is modified in-place
        - Callers should handle CUDA OOM errors or dtype conversion issues
        - Return value can be used for moving other tensors to the same device
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"--> [CUDA DETECTED] Moving model to {device} and casting to {dtype}...")
        model.to(device, dtype=dtype)
    else:
        device = torch.device("cpu")
        print(f"--> [WARNING] CUDA not available. Using CPU with {dtype} (may be slow).")
        model.to(device, dtype=dtype)
    
    return device
