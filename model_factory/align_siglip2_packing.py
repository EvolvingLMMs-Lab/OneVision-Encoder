#!/usr/bin/env python3
# coding=utf-8
"""
Siglip2 Naflex Packing Alignment Script

This script verifies consistency between:
- vit_siglip2.py (Siglip2Naflex) - standard format that accepts [B, C, H, W] images
- vit_siglip2_packing_hf.py (Siglip2NaflexPacking) - packing format that accepts pre-patchified input

The script performs the following:
1. Loads both standard and packing models with the same checkpoint
2. Creates random test images in standard format [B, C, H, W]
3. Processes through standard model directly
4. Converts images to packing format (pre-patchified patches + grid_thw)
5. Processes through packing model
6. Compares outputs using cosine similarity and absolute difference metrics
7. Reports pass/fail based on similarity threshold

Expected Result:
Both models should produce identical (or near-identical) outputs since they share
the same weights and architecture, just with different I/O formats.

Usage:
    python align_siglip2_packing.py --ckpt <model_checkpoint> [--device cuda]
    
Example:
    python align_siglip2_packing.py \
        --ckpt google/siglip2-so400m-patch16-naflex \
        --device cuda \
        --batch_size 2 \
        --image_size 224 \
        --threshold 0.99
"""

import argparse
import torch
import torch.nn.functional as F
import numpy as np
from vit_siglip2 import Siglip2Naflex
from vit_siglip2_packing_hf import Siglip2NaflexPacking


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
    
    # Create grid_thw: [bs, 3] where each row is [t, h, w]
    # For single images, t=1
    grid_thw = torch.tensor(
        [[1, num_patches_height, num_patches_width]] * batch_size,
        dtype=torch.long,
        device=pixel_values.device
    )
    
    return packed_patches, grid_thw


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
        'max_cosine': max_cos,
    }


def test_alignment(standard_model, packing_model, test_input, patch_size, device):
    """
    Test alignment between standard and packing models.
    
    Args:
        standard_model: Siglip2Naflex model
        packing_model: Siglip2NaflexPacking model
        test_input: Input tensor of shape [bs, channels, height, width]
        patch_size: Patch size
        device: Device to run on
    
    Returns:
        dict: Dictionary containing test results
    """
    print(f"\nTesting with input shape: {test_input.shape}")
    
    # Move input to device
    test_input = test_input.to(device)
    
    # Get output from standard model
    print("Running standard model (Siglip2Naflex)...")
    with torch.no_grad():
        standard_output = standard_model(test_input)
    
    print(f"Standard model output shape: {standard_output.shape}")
    
    # Convert input to packing format
    print("Converting input to packing format...")
    packed_input, grid_thw = convert_to_patches(test_input, patch_size)
    print(f"Packed input shape: {packed_input.shape}")
    print(f"grid_thw shape: {grid_thw.shape}")
    print(f"grid_thw values:\n{grid_thw}")
    
    # Get output from packing model
    print("Running packing model (Siglip2NaflexPacking)...")
    with torch.no_grad():
        packing_output = packing_model(packed_input, grid_thw)
    
    print(f"Packing model output shape: {packing_output.shape}")
    
    # Reshape packing output to match standard output format
    batch_size = test_input.shape[0]
    num_patches_per_image = packing_output.shape[0] // batch_size
    packing_output_reshaped = packing_output.reshape(batch_size, num_patches_per_image, -1)
    print(f"Packing model output reshaped: {packing_output_reshaped.shape}")
    
    # Compute similarity metrics
    print("\nComputing similarity metrics...")
    metrics = compute_similarity_metrics(standard_output, packing_output_reshaped)
    
    return metrics, standard_output, packing_output_reshaped


def main():
    parser = argparse.ArgumentParser(
        description="Verify alignment between Siglip2Naflex and Siglip2NaflexPacking"
    )
    parser.add_argument("--ckpt", type=str, default="google/siglip2-so400m-patch16-naflex",
                       help="Model checkpoint path")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to run on (default: cuda if available)")
    parser.add_argument("--batch_size", type=int, default=2,
                       help="Batch size for testing (default: 2)")
    parser.add_argument("--image_size", type=int, default=224,
                       help="Image size for testing (default: 224)")
    parser.add_argument("--threshold", type=float, default=0.99,
                       help="Cosine similarity threshold for pass/fail (default: 0.99)")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Siglip2 Naflex Packing Alignment Script")
    print("=" * 80)
    print(f"Model checkpoint: {args.ckpt}")
    print(f"Device: {args.device}")
    print(f"Similarity threshold: {args.threshold}")
    
    # Initialize models
    print("\nInitializing models...")
    print("Loading standard model (Siglip2Naflex)...")
    standard_model = Siglip2Naflex(ckpt=args.ckpt, device=args.device)
    
    print("Loading packing model (Siglip2NaflexPacking)...")
    packing_model = Siglip2NaflexPacking(ckpt=args.ckpt, device=args.device)
    
    patch_size = packing_model.patch_size
    print(f"Patch size: {patch_size}")
    
    # Validate image size is divisible by patch size
    if args.image_size % patch_size != 0:
        raise ValueError(
            f"Image size ({args.image_size}) must be divisible by patch size ({patch_size})"
        )
    
    # Create test input
    print(f"\nCreating test input: [{args.batch_size}, 3, {args.image_size}, {args.image_size}]")
    test_input = torch.randn(args.batch_size, 3, args.image_size, args.image_size)
    
    # Run alignment test
    print("\n" + "=" * 80)
    print("Running Alignment Test")
    print("=" * 80)
    
    metrics, standard_output, packing_output = test_alignment(
        standard_model, packing_model, test_input, patch_size, args.device
    )
    
    # Display results
    print("\n" + "=" * 80)
    print("Results")
    print("=" * 80)
    print(f"Max Diff:        {metrics['max_diff']:.6f}")
    print(f"Mean Diff:       {metrics['mean_diff']:.6f}")
    print(f"Min Cosine Sim:  {metrics['min_cosine']:.8f}")
    print(f"Mean Cosine Sim: {metrics['mean_cosine']:.8f}")
    print(f"Max Cosine Sim:  {metrics['max_cosine']:.8f}")
    
    # Pass/Fail
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    
    if metrics['min_cosine'] > args.threshold:
        print(f"✅ PASS: Models are aligned (min cosine similarity {metrics['min_cosine']:.8f} > {args.threshold})")
        return 0
    else:
        print(f"❌ FAIL: Models are NOT aligned (min cosine similarity {metrics['min_cosine']:.8f} <= {args.threshold})")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
