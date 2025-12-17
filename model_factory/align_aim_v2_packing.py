#!/usr/bin/env python3
# coding=utf-8
"""
AIMv2 Packing Alignment Script

This script verifies consistency between:
- vit_aim_v2.py (AIMv2) - standard format that accepts [B, C, H, W] images
- vit_aim_v2_packing_hf.py (AIMv2Packing) - packing format that accepts pre-patchified input

The script performs the following:
1. Loads both standard and packing models with the same checkpoint
2. Creates random test images in standard format [B, C, H, W] OR loads real images
3. Processes through standard model directly
4. Converts images to packing format (pre-patchified patches + grid_thw)
5. Processes through packing model
6. Compares outputs using cosine similarity and absolute difference metrics
7. Reports pass/fail based on similarity threshold

Expected Result:
Both models should produce identical (or near-identical) outputs since they share
the same weights and architecture, just with different I/O formats.

Note: AIMv2 includes CLS token, so the output will exclude this prefix token when
comparing patch representations.

Usage:
    # Test with random tensors
    python align_aim_v2_packing.py --ckpt <model_checkpoint> [--device cuda]

    # Test with real images from model_factory/images/
    python align_aim_v2_packing.py --ckpt <model_checkpoint> --use_real_images

Example:
    python align_aim_v2_packing.py \
        --ckpt apple/aimv2-large-patch14-224 \
        --device cuda \
        --batch_size 2 \
        --image_size 224 \
        --threshold 0.99

    python align_aim_v2_packing.py \
        --ckpt apple/aimv2-large-patch14-224 \
        --device cuda \
        --use_real_images \
        --image_dir model_factory/images
"""

import argparse
import math
import os
import traceback
import torch
import torch.nn.functional as F
import numpy as np
from vit_aim_v2 import AIMv2
from vit_aim_v2_packing_hf import AIMv2Packing
from alignment_utils import (
    convert_to_patches,
    round_up_to_multiple,
    generate_test_image,
    load_image_as_tensor,
    compute_similarity_metrics,
    print_metrics,
    PIL_AVAILABLE
)

# Note: PIL availability is checked in alignment_utils


def test_alignment(standard_model, packing_model, test_input, patch_size, device):
    """
    Test alignment between standard and packing models.

    Args:
        standard_model: AIMv2 model
        packing_model: AIMv2Packing model
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
    print("Running standard model (AIMv2)...")
    with torch.no_grad():
        standard_output = standard_model(test_input)

    print(f"Standard model output shape: {standard_output.shape}")

    # Aimv2VisionModel already excludes CLS token from last_hidden_state
    # So we can use the output directly without removing any prefix tokens
    standard_patch_tokens = standard_output
    print(f"Standard model patch tokens shape: {standard_patch_tokens.shape}")

    # Convert input to packing format
    print("Converting input to packing format...")
    packed_input, grid_thw = convert_to_patches(test_input, patch_size)
    print(f"Packed input shape: {packed_input.shape}")
    print(f"grid_thw shape: {grid_thw.shape}")
    print(f"grid_thw values:\n{grid_thw}")

    # Get output from packing model
    print("Running packing model (AIMv2Packing)...")
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            packing_output = packing_model(packed_input, grid_thw)

    print(f"Packing model output shape: {packing_output.shape}")

    # Reshape packing output to match standard output format
    batch_size = test_input.shape[0]
    num_patches_per_image = packing_output.shape[0] // batch_size
    packing_output_reshaped = packing_output.reshape(batch_size, num_patches_per_image, -1)
    print(f"Packing model output reshaped: {packing_output_reshaped.shape}")

    # Compute similarity metrics
    print("\nComputing similarity metrics...")
    metrics = compute_similarity_metrics(standard_patch_tokens, packing_output_reshaped)

    return metrics, standard_patch_tokens, packing_output_reshaped


def main():
    parser = argparse.ArgumentParser(
        description="Verify alignment between AIMv2 and AIMv2Packing"
    )
    parser.add_argument("--ckpt", type=str, default="apple/aimv2-large-patch14-224",
                       help="Model checkpoint path")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to run on (default: cuda if available)")
    parser.add_argument("--batch_size", type=int, default=2,
                       help="Batch size for testing (default: 2)")
    parser.add_argument("--image_size", type=int, default=224,
                       help="Image size for testing (default: 224)")
    parser.add_argument("--threshold", type=float, default=0.99,
                       help="Cosine similarity threshold for pass/fail (default: 0.99)")
    parser.add_argument("--use_real_images", action="store_true",
                       help="Use real images from model_factory/images/ directory")
    parser.add_argument("--image_dir", type=str, default="model_factory/images",
                       help="Directory containing test images (default: model_factory/images)")

    args = parser.parse_args()

    print("=" * 80)
    print("AIMv2 Packing Alignment Script")
    print("=" * 80)
    print(f"Model checkpoint: {args.ckpt}")
    print(f"Device: {args.device}")
    print(f"Similarity threshold: {args.threshold}")
    print(f"Use real images: {args.use_real_images}")

    # Initialize models
    print("\nInitializing models...")
    print("Loading standard model (AIMv2)...")
    standard_model = AIMv2(ckpt=args.ckpt, device=args.device)

    print("Loading packing model (AIMv2Packing)...")
    packing_model = AIMv2Packing.from_pretrained(args.ckpt, trust_remote_code=True)
    packing_model = packing_model.to(args.device).eval()

    patch_size = packing_model.config.patch_size
    print(f"Patch size: {patch_size}")

    all_tests_passed = True
    test_results = []

    if args.use_real_images:
        if not PIL_AVAILABLE:
            print("\n❌ ERROR: PIL is not available. Cannot load real images.")
            print("Please install Pillow: pip install Pillow")
            return 1

        # Get the script's directory to construct absolute paths
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Handle image directory path
        if os.path.isabs(args.image_dir):
            image_dir = args.image_dir
        elif args.image_dir == "model_factory/images":
            # Default case: use images directory relative to script
            image_dir = os.path.join(script_dir, "images")
        else:
            # User provided a relative path
            image_dir = os.path.abspath(args.image_dir)

        # Define test images
        test_images = [
            ("1.jpg", 384, 384),  # Default size if needs to be generated
            ("2.jpg", 512, 512),  # Default size if needs to be generated
        ]

        print("\n" + "=" * 80)
        print("Testing with Real Images")
        print("=" * 80)

        # Check and generate images if needed
        for img_name, width, height in test_images:
            img_path = os.path.join(image_dir, img_name)
            if not os.path.exists(img_path):
                print(f"\nImage not found: {img_path}")
                print(f"Generating test image...")
                generate_test_image(img_path, width, height)

        # Load and test each image individually
        for img_name, _, _ in test_images:
            img_path = os.path.join(image_dir, img_name)
            print(f"\n{'=' * 80}")
            print(f"Testing image: {img_name}")
            print(f"{'=' * 80}")

            try:
                # Load image
                test_input = load_image_as_tensor(img_path, args.device)

                # Get actual image dimensions
                _, _, height, width = test_input.shape
                print(f"Image dimensions: {width}x{height}")

                # Check if dimensions are divisible by patch size
                if height % patch_size != 0 or width % patch_size != 0:
                    print(f"⚠️  Warning: Image dimensions ({width}x{height}) are not divisible by patch size ({patch_size})")
                    print(f"Resizing to nearest multiple of patch size...")

                    # Round up to nearest multiple of patch size
                    new_height = round_up_to_multiple(height, patch_size)
                    new_width = round_up_to_multiple(width, patch_size)

                    test_input = F.interpolate(
                        test_input,
                        size=(new_height, new_width),
                        mode='bilinear',
                        align_corners=False
                    )
                    print(f"Resized to: {new_width}x{new_height}")

                # Run alignment test
                metrics, standard_output, packing_output = test_alignment(
                    standard_model, packing_model, test_input, patch_size, args.device
                )

                # Display results
                print(f"\nResults for {img_name}:")
                print("-" * 80)
                print(f"Max Diff:        {metrics['max_diff']:.6f}")
                print(f"Mean Diff:       {metrics['mean_diff']:.6f}")
                print(f"Min Cosine Sim:  {metrics['min_cosine']:.8f}")
                print(f"Mean Cosine Sim: {metrics['mean_cosine']:.8f}")
                print(f"Max Cosine Sim:  {metrics['max_cosine']:.8f}")

                # Check if test passed
                test_passed = metrics['min_cosine'] > args.threshold
                test_results.append((img_name, test_passed, metrics))

                if test_passed:
                    print(f"✅ PASS: {img_name} (min cosine similarity {metrics['min_cosine']:.8f} > {args.threshold})")
                else:
                    print(f"❌ FAIL: {img_name} (min cosine similarity {metrics['min_cosine']:.8f} <= {args.threshold})")
                    all_tests_passed = False

            except Exception as e:
                print(f"❌ ERROR processing {img_name}: {e}")
                all_tests_passed = False
                traceback.print_exc()

        # Test with both images together in a batch
        print(f"\n{'=' * 80}")
        print(f"Testing with batched real images")
        print(f"{'=' * 80}")

        try:
            # Load both images
            img1_path = os.path.join(image_dir, test_images[0][0])
            img2_path = os.path.join(image_dir, test_images[1][0])

            img1 = load_image_as_tensor(img1_path, args.device)
            img2 = load_image_as_tensor(img2_path, args.device)

            # Get dimensions
            _, _, h1, w1 = img1.shape
            _, _, h2, w2 = img2.shape

            # Resize both to a common size (use max dimensions, rounded up to patch size)
            target_h = round_up_to_multiple(max(h1, h2), patch_size)
            target_w = round_up_to_multiple(max(w1, w2), patch_size)

            print(f"Resizing images to common size: {target_w}x{target_h}")

            img1_resized = F.interpolate(img1, size=(target_h, target_w), mode='bilinear', align_corners=False)
            img2_resized = F.interpolate(img2, size=(target_h, target_w), mode='bilinear', align_corners=False)

            # Batch images together
            batch_input = torch.cat([img1_resized, img2_resized], dim=0)

            # Run alignment test
            metrics, standard_output, packing_output = test_alignment(
                standard_model, packing_model, batch_input, patch_size, args.device
            )

            # Display results
            print(f"\nResults for batched images:")
            print("-" * 80)
            print(f"Max Diff:        {metrics['max_diff']:.6f}")
            print(f"Mean Diff:       {metrics['mean_diff']:.6f}")
            print(f"Min Cosine Sim:  {metrics['min_cosine']:.8f}")
            print(f"Mean Cosine Sim: {metrics['mean_cosine']:.8f}")
            print(f"Max Cosine Sim:  {metrics['max_cosine']:.8f}")

            # Check if test passed
            test_passed = metrics['min_cosine'] > args.threshold
            test_results.append(("batch", test_passed, metrics))

            if test_passed:
                print(f"✅ PASS: Batched images (min cosine similarity {metrics['min_cosine']:.8f} > {args.threshold})")
            else:
                print(f"❌ FAIL: Batched images (min cosine similarity {metrics['min_cosine']:.8f} <= {args.threshold})")
                all_tests_passed = False

        except Exception as e:
            print(f"❌ ERROR processing batched images: {e}")
            all_tests_passed = False
            traceback.print_exc()

        # Test with 10 random-sized images in a batch (packing format advantage)
        print(f"\n{'=' * 80}")
        print(f"Testing with 10 random-sized images (packing format)")
        print(f"{'=' * 80}")

        try:
            # Generate 10 random images with different sizes (up to 1000x1000)
            np.random.seed(42)  # For reproducibility
            batch_images = []
            image_sizes = []
            
            for i in range(10):
                # Random size up to 1000x1000, rounded to nearest multiple of patch_size
                h = np.random.randint(224, 1001)
                w = np.random.randint(224, 1001)
                
                # Round to multiple of patch_size
                h = round_up_to_multiple(h, patch_size)
                w = round_up_to_multiple(w, patch_size)
                
                image_sizes.append((h, w))
                
                # Generate random image
                img = torch.randn(1, 3, h, w, device=args.device)
                batch_images.append(img)
            
            print(f"Generated 10 images with sizes: {image_sizes}")
            
            # Process each image individually through standard model
            print("Processing images through standard model...")
            standard_outputs = []
            for i, img in enumerate(batch_images):
                with torch.no_grad():
                    output = standard_model(img)
                standard_outputs.append(output.squeeze(0))  # Remove batch dim
            
            # Stack all outputs
            standard_output_list = standard_outputs
            
            # Convert all images to packing format
            print("Converting images to packing format...")
            all_patches = []
            grid_thw_list = []
            
            for img in batch_images:
                patches, grid = convert_to_patches(img, patch_size)
                all_patches.append(patches)
                grid_thw_list.append(grid)
            
            # Concatenate into single packed format
            packed_input = torch.cat(all_patches, dim=0)
            grid_thw = torch.cat(grid_thw_list, dim=0)
            
            print(f"Packed input shape: {packed_input.shape}")
            print(f"grid_thw shape: {grid_thw.shape}")
            
            # Process through packing model
            print("Processing through packing model...")
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    packing_output = packing_model(packed_input, grid_thw)
            
            print(f"Packing model output shape: {packing_output.shape}")
            
            # Split packing output back to individual images
            packing_outputs = []
            start_idx = 0
            for i in range(10):
                t, h, w = grid_thw[i]
                num_patches = int(t * h * w)
                packing_outputs.append(packing_output[start_idx:start_idx + num_patches])
                start_idx += num_patches
            
            # Compute similarity metrics for each image
            print("\nComputing similarity metrics for each image...")
            all_min_cosine = []
            all_mean_cosine = []
            all_max_diff = []
            all_mean_diff = []
            
            for i in range(10):
                std_out = standard_output_list[i].unsqueeze(0)
                pack_out = packing_outputs[i].unsqueeze(0)
                
                # Compute metrics
                diff = (std_out - pack_out).abs()
                max_diff = diff.max().item()
                mean_diff = diff.mean().item()
                
                # Cosine similarity per patch
                std_norm = std_out / (std_out.norm(dim=-1, keepdim=True) + 1e-8)
                pack_norm = pack_out / (pack_out.norm(dim=-1, keepdim=True) + 1e-8)
                cosine_sim = (std_norm * pack_norm).sum(dim=-1)
                
                min_cosine = cosine_sim.min().item()
                mean_cosine = cosine_sim.mean().item()
                
                all_min_cosine.append(min_cosine)
                all_mean_cosine.append(mean_cosine)
                all_max_diff.append(max_diff)
                all_mean_diff.append(mean_diff)
                
                print(f"  Image {i+1} ({image_sizes[i][1]}x{image_sizes[i][0]}): "
                      f"min_cos={min_cosine:.4f}, mean_cos={mean_cosine:.4f}, "
                      f"max_diff={max_diff:.4f}")
            
            # Aggregate metrics
            overall_min_cosine = min(all_min_cosine)
            overall_mean_cosine = np.mean(all_mean_cosine)
            overall_max_diff = max(all_max_diff)
            overall_mean_diff = np.mean(all_mean_diff)
            
            print(f"\nOverall results for 10 random-sized images:")
            print("-" * 80)
            print(f"Max Diff:        {overall_max_diff:.6f}")
            print(f"Mean Diff:       {overall_mean_diff:.6f}")
            print(f"Min Cosine Sim:  {overall_min_cosine:.8f}")
            print(f"Mean Cosine Sim: {overall_mean_cosine:.8f}")
            
            # Check if test passed
            test_passed = overall_min_cosine > args.threshold
            metrics = {
                'max_diff': overall_max_diff,
                'mean_diff': overall_mean_diff,
                'min_cosine': overall_min_cosine,
                'mean_cosine': overall_mean_cosine,
                'max_cosine': max(all_min_cosine)  # Using min_cosine per image as representative
            }
            test_results.append(("10_random_sizes", test_passed, metrics))
            
            if test_passed:
                print(f"✅ PASS: 10 random-sized images (min cosine similarity {overall_min_cosine:.8f} > {args.threshold})")
            else:
                print(f"❌ FAIL: 10 random-sized images (min cosine similarity {overall_min_cosine:.8f} <= {args.threshold})")
                all_tests_passed = False
                
        except Exception as e:
            print(f"❌ ERROR processing random-sized batch: {e}")
            all_tests_passed = False
            traceback.print_exc()

    else:
        # Multi-resolution random tensor tests
        # Test resolutions: 224, 336, 448 (multiples of 14 for patch14 models)
        test_resolutions = [224, 336, 448]

        # Validate all resolutions are divisible by patch size
        for res in test_resolutions:
            if res % patch_size != 0:
                raise ValueError(
                    f"Resolution {res} must be divisible by patch size ({patch_size})"
                )

        print("\n" + "=" * 80)
        print("Multi-Resolution Alignment Tests")
        print("=" * 80)
        print(f"Testing resolutions: {test_resolutions}")

        # Test 1: Individual resolution tests
        print("\n" + "=" * 80)
        print("Test 1: Individual Resolution Tests")
        print("=" * 80)

        for resolution in test_resolutions:
            print(f"\n{'=' * 80}")
            print(f"Testing resolution: {resolution}x{resolution}")
            print(f"{'=' * 80}")

            # Create test input
            test_input = torch.randn(args.batch_size, 3, resolution, resolution)
            print(f"Input shape: {test_input.shape}")

            try:
                # Run alignment test
                metrics, standard_output, packing_output = test_alignment(
                    standard_model, packing_model, test_input, patch_size, args.device
                )

                # Display results
                print(f"\nResults for {resolution}x{resolution}:")
                print("-" * 80)
                print(f"Max Diff:        {metrics['max_diff']:.6f}")
                print(f"Mean Diff:       {metrics['mean_diff']:.6f}")
                print(f"Min Cosine Sim:  {metrics['min_cosine']:.8f}")
                print(f"Mean Cosine Sim: {metrics['mean_cosine']:.8f}")
                print(f"Max Cosine Sim:  {metrics['max_cosine']:.8f}")

                # Check if test passed
                test_passed = metrics['min_cosine'] > args.threshold
                test_results.append((f"{resolution}x{resolution}", test_passed, metrics))

                if test_passed:
                    print(f"✅ PASS: {resolution}x{resolution} (min cosine similarity {metrics['min_cosine']:.8f} > {args.threshold})")
                else:
                    print(f"❌ FAIL: {resolution}x{resolution} (min cosine similarity {metrics['min_cosine']:.8f} <= {args.threshold})")
                    all_tests_passed = False

            except Exception as e:
                print(f"❌ ERROR testing {resolution}x{resolution}: {e}")
                all_tests_passed = False
                traceback.print_exc()

        # Test 2: Mixed resolution batch test (all three resolutions together)
        print(f"\n{'=' * 80}")
        print(f"Test 2: Mixed Resolution Batch Test")
        print(f"{'=' * 80}")
        print(f"Testing all resolutions together in packing format: {test_resolutions}")

        try:
            # Create images with different resolutions
            images = []
            for resolution in test_resolutions:
                img = torch.randn(1, 3, resolution, resolution, device=args.device)
                images.append(img)

            print(f"Created {len(images)} images with resolutions: {test_resolutions}")

            # Process each through standard model separately
            print("\nRunning standard model on each image separately...")
            standard_outputs = []
            for i, (img, resolution) in enumerate(zip(images, test_resolutions)):
                with torch.no_grad():
                    output = standard_model(img)
                # Aimv2VisionModel already excludes CLS token from last_hidden_state
                # So we can use the output directly
                standard_outputs.append(output.squeeze(0))  # Remove batch dimension
                print(f"  Image {i+1} ({resolution}x{resolution}): output shape {output.shape}")

            # Convert all images to packing format
            print("\nConverting to packing format...")
            all_patches = []
            grid_thw_list = []

            for i, (img, resolution) in enumerate(zip(images, test_resolutions)):
                packed_patches, grid_thw_single = convert_to_patches(img, patch_size)
                all_patches.append(packed_patches)
                grid_thw_list.append(grid_thw_single)
                num_patches = packed_patches.shape[0]
                print(f"  Image {i+1} ({resolution}x{resolution}): {num_patches} patches")

            # Concatenate all patches
            packed_input = torch.cat(all_patches, dim=0)
            grid_thw = torch.cat(grid_thw_list, dim=0)

            print(f"Packed input shape: {packed_input.shape}")
            print(f"grid_thw shape: {grid_thw.shape}")
            print(f"grid_thw values:\n{grid_thw}")

            # Process through packing model
            print("\nRunning packing model...")
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    packing_output = packing_model(packed_input, grid_thw)

            print(f"Packing model output shape: {packing_output.shape}")

            # Split packing output back into individual images
            packing_outputs = []
            start_idx = 0
            for i, resolution in enumerate(test_resolutions):
                num_patches = (resolution // patch_size) ** 2
                packing_outputs.append(packing_output[start_idx:start_idx + num_patches])
                start_idx += num_patches
                print(f"  Image {i+1} output shape: {packing_outputs[-1].shape}")

            # Compare each image's output
            print("\nComparing outputs for each resolution...")
            all_mixed_passed = True
            for i, resolution in enumerate(test_resolutions):
                standard_out = standard_outputs[i]
                packing_out = packing_outputs[i]

                # Reshape to add batch dimension for metric computation
                standard_out = standard_out.unsqueeze(0)
                packing_out = packing_out.unsqueeze(0)

                metrics = compute_similarity_metrics(standard_out, packing_out)

                print(f"\n  Results for {resolution}x{resolution} in mixed batch:")
                print(f"    Max Diff:        {metrics['max_diff']:.6f}")
                print(f"    Mean Diff:       {metrics['mean_diff']:.6f}")
                print(f"    Min Cosine Sim:  {metrics['min_cosine']:.8f}")
                print(f"    Mean Cosine Sim: {metrics['mean_cosine']:.8f}")
                print(f"    Max Cosine Sim:  {metrics['max_cosine']:.8f}")

                test_passed = metrics['min_cosine'] > args.threshold
                if not test_passed:
                    all_mixed_passed = False
                    all_tests_passed = False

            if all_mixed_passed:
                print(f"\n✅ PASS: Mixed resolution batch test (all resolutions aligned)")
                # Compute minimum and mean cosine similarity across all resolutions
                min_cosines = []
                mean_cosines = []
                for i in range(len(test_resolutions)):
                    metrics = compute_similarity_metrics(
                        standard_outputs[i].unsqueeze(0),
                        packing_outputs[i].unsqueeze(0)
                    )
                    min_cosines.append(metrics['min_cosine'])
                    mean_cosines.append(metrics['mean_cosine'])
                test_results.append(("mixed_batch", True, {
                    "min_cosine": min(min_cosines),
                    "mean_cosine": sum(mean_cosines) / len(mean_cosines)
                }))
            else:
                print(f"\n❌ FAIL: Mixed resolution batch test (some resolutions misaligned)")
                # Still compute metrics even for failed tests
                min_cosines = []
                mean_cosines = []
                for i in range(len(test_resolutions)):
                    try:
                        metrics = compute_similarity_metrics(
                            standard_outputs[i].unsqueeze(0),
                            packing_outputs[i].unsqueeze(0)
                        )
                        min_cosines.append(metrics['min_cosine'])
                        mean_cosines.append(metrics['mean_cosine'])
                    except:
                        min_cosines.append(0.0)
                        mean_cosines.append(0.0)
                test_results.append(("mixed_batch", False, {
                    "min_cosine": min(min_cosines) if min_cosines else 0.0,
                    "mean_cosine": sum(mean_cosines) / len(mean_cosines) if mean_cosines else 0.0
                }))

        except Exception as e:
            print(f"❌ ERROR in mixed resolution batch test: {e}")
            all_tests_passed = False
            traceback.print_exc()

    # Final summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)

    if len(test_results) > 1:
        print(f"\nTest Results:")
        for name, passed, metrics in test_results:
            # Check pass/fail for both min and mean thresholds
            min_pass = metrics['min_cosine'] > args.threshold
            mean_pass = metrics['mean_cosine'] > args.threshold

            min_status = "✅" if min_pass else "❌"
            mean_status = "✅" if mean_pass else "❌"

            print(f"  {name}:")
            print(f"    {min_status} Min cosine:  {metrics['min_cosine']:.8f} ({'PASS' if min_pass else 'FAIL'})")
            print(f"    {mean_status} Mean cosine: {metrics['mean_cosine']:.8f} ({'PASS' if mean_pass else 'FAIL'})")

    if all_tests_passed:
        print(f"\n✅ ALL TESTS PASSED: Models are aligned")
        return 0
    else:
        print(f"\n❌ SOME TESTS FAILED: Models may not be aligned")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
