#!/usr/bin/env python3
# coding=utf-8
"""
Common utilities for vision model packing alignment scripts.

This module contains shared functions used across multiple alignment scripts
(align_aim_v2_packing.py, align_dinov3_packing.py, align_siglip2_packing.py).
"""

import math
import os
import torch
import torch.nn.functional as F
import numpy as np

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


def convert_to_patches(pixel_values, patch_size):
    """
    Convert image tensor to patches for packing format.
    
    Args:
        pixel_values (torch.Tensor): Input tensor of shape [bs, channels, height, width]
        patch_size (int): Size of each patch
    
    Returns:
        torch.Tensor: Patches of shape [total_num_patches, channels * patch_size * patch_size]
        torch.Tensor: grid_thw of shape [bs, 3] containing [t, h, w] for each image
    """
    batch_size, channels, height, width = pixel_values.shape
    num_patches_height = height // patch_size
    num_patches_width = width // patch_size
    
    # Reshape to patches: [bs, channels, num_patches_h, patch_size, num_patches_w, patch_size]
    patches = pixel_values.reshape(
        batch_size, channels,
        num_patches_height, patch_size,
        num_patches_width, patch_size
    )
    
    # Rearrange to: [bs, num_patches_h, num_patches_w, patch_size, patch_size, channels]
    patches = patches.permute(0, 2, 4, 3, 5, 1)
    
    # Flatten patches: [bs, num_patches_h * num_patches_w, patch_size * patch_size * channels]
    patches = patches.reshape(
        batch_size,
        num_patches_height * num_patches_width,
        patch_size * patch_size * channels
    )
    
    # Concatenate all batches: [total_num_patches, patch_dim]
    packed_patches = patches.reshape(-1, patch_size * patch_size * channels)
    
    # Create grid_thw: [bs, 3] where each row is [t=1, h=num_patches_height, w=num_patches_width]
    grid_thw = torch.tensor(
        [[1, num_patches_height, num_patches_width]] * batch_size,
        dtype=torch.long,
        device=pixel_values.device
    )
    
    return packed_patches, grid_thw


def round_up_to_multiple(value, multiple):
    """
    Round up a value to the nearest multiple.
    
    Args:
        value (int): Value to round up
        multiple (int): Multiple to round to
    
    Returns:
        int: Rounded value
    """
    return int(math.ceil(value / multiple) * multiple)


def generate_test_image(path, width, height):
    """
    Generate a test image with random colors.
    
    Args:
        path (str): Path to save the image
        width (int): Image width
        height (int): Image height
    
    Returns:
        str: Path to the generated image
    """
    if not PIL_AVAILABLE:
        raise ImportError("PIL is required for image generation")
    
    # Create random image
    img_array = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    
    # Save image
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)
    
    return path


def load_image_as_tensor(image_path, device):
    """
    Load an image and convert it to a tensor.
    
    Args:
        image_path (str): Path to the image
        device (str): Device to load tensor on
    
    Returns:
        torch.Tensor: Image tensor of shape [1, 3, H, W] normalized to [0, 1]
    """
    if not PIL_AVAILABLE:
        raise ImportError("PIL is required for image loading")
    
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img)
    
    # Convert to tensor and normalize
    img_tensor = torch.from_numpy(img_array).float() / 255.0
    
    # Rearrange to [C, H, W]
    img_tensor = img_tensor.permute(2, 0, 1)
    
    # Add batch dimension
    img_tensor = img_tensor.unsqueeze(0)
    
    return img_tensor.to(device)


def compute_similarity_metrics(feat1: torch.Tensor, feat2: torch.Tensor):
    """
    Compute similarity metrics between two feature tensors.
    
    Args:
        feat1: First feature tensor
        feat2: Second feature tensor
    
    Returns:
        dict: Dictionary containing similarity metrics
    """
    # Ensure same shape
    assert feat1.shape == feat2.shape, f"Shape mismatch: {feat1.shape} vs {feat2.shape}"
    
    # Compute metrics
    max_diff = (feat1 - feat2).abs().max().item()
    mean_diff = (feat1 - feat2).abs().mean().item()
    
    # Flatten for cosine similarity computation
    feat1_flat = feat1.reshape(-1, feat1.shape[-1])
    feat2_flat = feat2.reshape(-1, feat2.shape[-1])
    
    cos_sim = F.cosine_similarity(feat1_flat, feat2_flat, dim=-1)
    min_cos = cos_sim.min().item()
    mean_cos = cos_sim.mean().item()
    max_cos = cos_sim.max().item()
    
    return {
        'max_diff': max_diff,
        'mean_diff': mean_diff,
        'min_cosine': min_cos,
        'mean_cosine': mean_cos,
        'max_cosine': max_cos
    }


def print_metrics(metrics, prefix="", indent=0):
    """
    Print similarity metrics in a formatted way.
    
    Args:
        metrics (dict): Dictionary containing metrics from compute_similarity_metrics
        prefix (str): Prefix to add to the output (e.g., resolution info)
        indent (int): Number of spaces to indent the output
    """
    indent_str = " " * indent
    if prefix:
        print(f"{indent_str}Results for {prefix}:")
        print(f"{indent_str}{'-' * 80}")
    print(f"{indent_str}Max Diff:        {metrics['max_diff']:.6f}")
    print(f"{indent_str}Mean Diff:       {metrics['mean_diff']:.6f}")
    print(f"{indent_str}Min Cosine Sim:  {metrics['min_cosine']:.8f}")
    print(f"{indent_str}Mean Cosine Sim: {metrics['mean_cosine']:.8f}")
    print(f"{indent_str}Max Cosine Sim:  {metrics['max_cosine']:.8f}")
