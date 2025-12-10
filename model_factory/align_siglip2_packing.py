#!/usr/bin/env python3
# coding=utf-8
"""
Siglip2 Naflex Packing Alignment Script

This script verifies consistency between:
- vit_siglip2.py (Siglip2Naflex) - standard format that accepts [B, C, H, W] images
- vit_siglip2_packing_hf.py (Siglip2NaflexPacking) - packing format that accepts pre-patchified input

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

Usage:
    # Test with random tensors
    python align_siglip2_packing.py --ckpt <model_checkpoint> [--device cuda]
    
    # Test with real images from model_factory/images/
    python align_siglip2_packing.py --ckpt <model_checkpoint> --use_real_images
    
Example:
    python align_siglip2_packing.py \
        --ckpt google/siglip2-so400m-patch16-naflex \
        --device cuda \
        --batch_size 2 \
        --image_size 224 \
        --threshold 0.99
    
    python align_siglip2_packing.py \
        --ckpt google/siglip2-so400m-patch16-naflex \
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
from vit_siglip2 import Siglip2Naflex
from vit_siglip2_packing_hf import Siglip2NaflexPacking

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Warning: PIL not available. Real image tests will be disabled.")


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


def round_up_to_multiple(value, multiple):
    """
    Round up a value to the nearest multiple.
    
    Args:
        value (int): Value to round up
        multiple (int): Multiple to round up to
    
    Returns:
        int: Rounded up value (at least `multiple`)
    """
    return max(multiple, ((value + multiple - 1) // multiple) * multiple)


def generate_test_image(path, width, height):
    """
    Generate a test image with random colors if it doesn't exist.
    
    Args:
        path: Path to save the image
        width: Width of the image
        height: Height of the image
    """
    if not PIL_AVAILABLE:
        raise ImportError("PIL is required to generate images. Please install Pillow.")
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Create a random colorful image
    img_array = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    img = Image.fromarray(img_array, 'RGB')
    img.save(path)
    print(f"Generated test image: {path} ({width}x{height})")


def load_image_as_tensor(image_path, device):
    """
    Load an image from disk and convert it to a tensor.
    
    Args:
        image_path: Path to the image file
        device: Device to load the tensor to
    
    Returns:
        torch.Tensor: Image tensor of shape [1, 3, H, W] with values in [0, 1]
    """
    if not PIL_AVAILABLE:
        raise ImportError("PIL is required to load images. Please install Pillow.")
    
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img, dtype=np.float32) / 255.0  # Normalize to [0, 1]
    
    # Convert to tensor: [H, W, C] -> [C, H, W] -> [1, C, H, W]
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
    
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
    parser.add_argument("--use_real_images", action="store_true",
                       help="Use real images from model_factory/images/ directory")
    parser.add_argument("--image_dir", type=str, default="model_factory/images",
                       help="Directory containing test images (default: model_factory/images)")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Siglip2 Naflex Packing Alignment Script")
    print("=" * 80)
    print(f"Model checkpoint: {args.ckpt}")
    print(f"Device: {args.device}")
    print(f"Similarity threshold: {args.threshold}")
    print(f"Use real images: {args.use_real_images}")
    
    # Initialize models
    print("\nInitializing models...")
    print("Loading standard model (Siglip2Naflex)...")
    standard_model = Siglip2Naflex(ckpt=args.ckpt, device=args.device)
    
    print("Loading packing model (Siglip2NaflexPacking)...")
    packing_model = Siglip2NaflexPacking(ckpt=args.ckpt, device=args.device)
    
    patch_size = packing_model.patch_size
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
                import traceback
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
            import traceback
            traceback.print_exc()
        
    else:
        # Multi-resolution random tensor tests
        # Test resolutions: 224, 384, 512
        test_resolutions = [224, 384, 512]
        
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
                
                test_passed = metrics['min_cosine'] > args.threshold
                if not test_passed:
                    all_mixed_passed = False
                    all_tests_passed = False
            
            if all_mixed_passed:
                print(f"\n✅ PASS: Mixed resolution batch test (all resolutions aligned)")
                test_results.append(("mixed_batch", True, {"min_cosine": min([compute_similarity_metrics(standard_outputs[i].unsqueeze(0), packing_outputs[i].unsqueeze(0))['min_cosine'] for i in range(len(test_resolutions))])}))
            else:
                print(f"\n❌ FAIL: Mixed resolution batch test (some resolutions misaligned)")
                test_results.append(("mixed_batch", False, {"min_cosine": 0.0}))
                
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
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"  {status}: {name} (min cosine: {metrics['min_cosine']:.8f})")
    
    if all_tests_passed:
        print(f"\n✅ ALL TESTS PASSED: Models are aligned")
        return 0
    else:
        print(f"\n❌ SOME TESTS FAILED: Models may not be aligned")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
