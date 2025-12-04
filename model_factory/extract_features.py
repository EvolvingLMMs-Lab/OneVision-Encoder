#!/usr/bin/env python3
# coding=utf-8
"""
Feature Extraction and Verification Tool

This tool extracts second-to-last layer features from both packing and non-packing 
HuggingFace ViT models and immediately verifies their consistency.

Usage:
    python extract_features.py \\
        --hf_model_path /path/to/hf_model \\
        --packing_model_path /path/to/packing_model \\
        --image1 1.jpg \\
        --image2 2.jpg \\
        --video1 1.mp4 \\
        --video2 2.mp4 \\
        --output_dir ./features \\
        --threshold 0.99

Input Requirements:
    - Images: Native resolution (no resizing)
    - Videos: 8 frames at native resolution
    
Output:
    - features_hf.npz: Features from non-packing HF model
    - features_packing.npz: Features from packing model
    - metadata.json: Metadata about inputs and models
    - Console output: Consistency verification results
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import cv2

try:
    from model_factory.vit_preview_v0_hf import LlavaViTModel
    from model_factory.vit_preview_v0_packing_hf import (
        LlavaViTPackingModel,
        compute_patch_positions_from_grid_thw,
    )
    from model_factory.conversion_utils import (
        interpolate_frame_indices,
        compute_patch_positions_with_interpolated_temporal,
        CLIP_MEAN,
        CLIP_STD,
    )
except ImportError as e:
    print(f"[Error] Failed to import required modules: {e}")
    print("Please ensure model_factory is in your PYTHONPATH")
    import sys
    sys.exit(1)


def load_image(image_path: str, device: torch.device) -> Tuple[torch.Tensor, Dict]:
    """
    Load image at native resolution and preprocess with CLIP normalization.
    
    Args:
        image_path: Path to image file
        device: Target device
    
    Returns:
        Tuple of (image_tensor, metadata)
        - image_tensor: [1, 3, H, W]
        - metadata: Dict with resolution info
    """
    print(f"Loading image: {image_path}")
    img = Image.open(image_path).convert("RGB")
    width, height = img.size
    
    # Convert to tensor and normalize with CLIP stats
    img_tensor = torch.from_numpy(np.array(img)).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1)  # [H, W, C] -> [C, H, W]
    
    # Apply CLIP normalization
    mean = torch.tensor(CLIP_MEAN).view(3, 1, 1)
    std = torch.tensor(CLIP_STD).view(3, 1, 1)
    img_tensor = (img_tensor - mean) / std
    
    img_tensor = img_tensor.unsqueeze(0).to(device)  # [1, C, H, W]
    
    metadata = {
        "path": image_path,
        "width": width,
        "height": height,
        "shape": list(img_tensor.shape),
    }
    
    print(f"  Resolution: {width}x{height}")
    return img_tensor, metadata


def load_video(video_path: str, num_frames: int, device: torch.device) -> Tuple[torch.Tensor, Dict]:
    """
    Load video at native resolution with uniform frame sampling.
    
    Args:
        video_path: Path to video file
        num_frames: Number of frames to extract (default: 8)
        device: Target device
    
    Returns:
        Tuple of (video_tensor, metadata)
        - video_tensor: [1, 3, T, H, W]
        - metadata: Dict with video info
    """
    print(f"Loading video: {video_path}")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"  Total frames: {total_frames}, FPS: {fps}, Resolution: {width}x{height}")
    
    # Uniformly sample frames (handle single-frame case)
    if total_frames == 1:
        frame_indices = np.array([0] * num_frames, dtype=int)
    else:
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    print(f"  Sampling frames: {frame_indices.tolist()}")
    
    frames = []
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            raise ValueError(f"Failed to read frame {frame_idx} from {video_path}")
        
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    
    cap.release()
    
    # Convert to tensor [T, H, W, C] -> [C, T, H, W]
    video_array = np.stack(frames, axis=0)  # [T, H, W, C]
    video_tensor = torch.from_numpy(video_array).float() / 255.0
    video_tensor = video_tensor.permute(3, 0, 1, 2)  # [C, T, H, W]
    
    # Apply CLIP normalization
    mean = torch.tensor(CLIP_MEAN).view(3, 1, 1, 1)
    std = torch.tensor(CLIP_STD).view(3, 1, 1, 1)
    video_tensor = (video_tensor - mean) / std
    
    video_tensor = video_tensor.unsqueeze(0).to(device)  # [1, C, T, H, W]
    
    metadata = {
        "path": video_path,
        "width": width,
        "height": height,
        "total_frames": total_frames,
        "sampled_frames": num_frames,
        "frame_indices": frame_indices.tolist(),
        "fps": fps,
        "shape": list(video_tensor.shape),
    }
    
    return video_tensor, metadata


def extract_hf_features(
    model: LlavaViTModel,
    images: List[torch.Tensor],
    videos: List[torch.Tensor],
    device: torch.device
) -> Dict[str, np.ndarray]:
    """
    Extract second-to-last layer features from non-packing HF model.
    
    Args:
        model: LlavaViTModel instance
        images: List of image tensors [1, 3, H, W]
        videos: List of video tensors [1, 3, T, H, W]
        device: Device to run on
    
    Returns:
        Dict mapping input names to feature arrays
    """
    print("\n=== Extracting features from HF model ===")
    model.eval()
    features = {}
    
    with torch.no_grad():
        # Process images
        for i, img in enumerate(images, 1):
            print(f"Processing image {i}...")
            img = img.to(device, dtype=torch.bfloat16)
            output = model(img, output_hidden_states=True)
            # Get second-to-last layer: hidden_states[-2]
            feat = output.hidden_states[-2].cpu().float().numpy()
            features[f"image{i}"] = feat
            print(f"  Feature shape: {feat.shape}")
        
        # Process videos
        for i, video in enumerate(videos, 1):
            print(f"Processing video {i}...")
            video = video.to(device, dtype=torch.bfloat16)
            
            # Videos need special handling with visible_indices
            bs, C, T, H, W = video.shape
            patch_size = model.config.patch_size
            h_patches = H // patch_size
            w_patches = W // patch_size
            frame_tokens = h_patches * w_patches
            target_frames = 64
            
            # Compute interpolated frame indices
            frame_indices = torch.arange(T).unsqueeze(0).to(device)
            total_frames_tensor = torch.tensor([T]).to(device)
            interpolated_indices = interpolate_frame_indices(
                frame_indices, total_frames_tensor, target_frames
            )
            
            # Create padded video
            padded_video = torch.zeros(bs, C, target_frames, H, W, device=device, dtype=video.dtype)
            frame_idx_expanded = interpolated_indices.view(bs, 1, T, 1, 1).expand(bs, C, T, H, W)
            padded_video.scatter_(dim=2, index=frame_idx_expanded, src=video)
            
            # Compute visible_index
            per = torch.arange(frame_tokens, device=device)
            visible_index = (interpolated_indices.unsqueeze(-1) * frame_tokens + per).reshape(bs, -1)
            visible_index = visible_index.clamp_max(target_frames * frame_tokens - 1)
            
            output = model(padded_video, visible_indices=visible_index, output_hidden_states=True)
            # Get second-to-last layer
            feat = output.hidden_states[-2].cpu().float().numpy()
            features[f"video{i}"] = feat
            print(f"  Feature shape: {feat.shape}")
    
    return features


def extract_packing_features(
    model: LlavaViTPackingModel,
    images: List[torch.Tensor],
    videos: List[torch.Tensor],
    device: torch.device
) -> Dict[str, np.ndarray]:
    """
    Extract second-to-last layer features from packing model.
    
    Args:
        model: LlavaViTPackingModel instance
        images: List of image tensors [1, 3, H, W]
        videos: List of video tensors [1, 3, T, H, W]
        device: Device to run on
    
    Returns:
        Dict mapping input names to feature arrays
    """
    print("\n=== Extracting features from Packing model ===")
    model.eval()
    features = {}
    
    # Get model config
    patch_size = model.config.patch_size
    channels = 3
    target_frames = 64
    
    with torch.no_grad():
        # Process images
        for i, img in enumerate(images, 1):
            print(f"Processing image {i}...")
            img = img.to(device, dtype=torch.bfloat16)
            
            bs, C, H, W = img.shape
            h_patches = H // patch_size
            w_patches = W // patch_size
            
            # Reshape to patches
            patches = img.view(bs, C, h_patches, patch_size, w_patches, patch_size)
            patches = patches.permute(0, 2, 4, 1, 3, 5).contiguous()
            seq_len = bs * h_patches * w_patches
            patch_dim = patch_size * patch_size * C
            hidden_states = patches.view(seq_len, patch_dim)
            
            # Create grid_thw and patch_positions
            grid_thw = torch.tensor([[1, h_patches, w_patches]], dtype=torch.long, device=device)
            patch_positions = compute_patch_positions_from_grid_thw(grid_thw)
            
            output = model(
                hidden_states=hidden_states,
                grid_thw=grid_thw,
                patch_positions=patch_positions,
                output_hidden_states=True
            )
            # Get second-to-last layer
            feat = output.hidden_states[-2].cpu().float().numpy()
            features[f"image{i}"] = feat
            print(f"  Feature shape: {feat.shape}")
        
        # Process videos
        for i, video in enumerate(videos, 1):
            print(f"Processing video {i}...")
            video = video.to(device, dtype=torch.bfloat16)
            
            bs, C, T, H, W = video.shape
            h_patches = H // patch_size
            w_patches = W // patch_size
            
            # Compute interpolated indices for RoPE
            frame_indices = torch.arange(T).unsqueeze(0).to(device)
            total_frames_tensor = torch.tensor([T]).to(device)
            interpolated_indices = interpolate_frame_indices(
                frame_indices, total_frames_tensor, target_frames
            )
            
            # Reshape to patches
            patches = video.view(bs, C, T, h_patches, patch_size, w_patches, patch_size)
            patches = patches.permute(0, 2, 3, 5, 1, 4, 6).contiguous()
            seq_len = bs * T * h_patches * w_patches
            patch_dim = patch_size * patch_size * C
            hidden_states = patches.view(seq_len, patch_dim)
            
            # Create grid_thw and patch_positions
            grid_thw = torch.tensor([[T, h_patches, w_patches]], dtype=torch.long, device=device)
            patch_positions = compute_patch_positions_with_interpolated_temporal(
                interpolated_indices, h_patches, w_patches, device
            )
            
            output = model(
                hidden_states=hidden_states,
                grid_thw=grid_thw,
                patch_positions=patch_positions,
                output_hidden_states=True
            )
            # Get second-to-last layer
            feat = output.hidden_states[-2].cpu().float().numpy()
            features[f"video{i}"] = feat
            print(f"  Feature shape: {feat.shape}")
    
    return features


def verify_feature_consistency(hf_features: Dict[str, np.ndarray], 
                               packing_features: Dict[str, np.ndarray],
                               threshold: float = 0.99) -> bool:
    """
    Verify consistency between HF and packing model features.
    
    Args:
        hf_features: Features from HF model
        packing_features: Features from packing model
        threshold: Cosine similarity threshold for pass/fail
    
    Returns:
        True if all features pass, False otherwise
    """
    print("\n" + "=" * 80)
    print("Feature Consistency Verification")
    print("=" * 80)
    print(f"Similarity threshold: {threshold}")
    
    results = {}
    
    for key in hf_features.keys():
        if key not in packing_features:
            print(f"\n⚠️  Warning: {key} not found in packing features")
            continue
        
        feat1 = hf_features[key]
        feat2 = packing_features[key]
        
        print(f"\n--- {key} ---")
        print(f"HF shape:      {feat1.shape}")
        print(f"Packing shape: {feat2.shape}")
        
        # Convert to torch tensors
        feat1_t = torch.from_numpy(feat1).float()
        feat2_t = torch.from_numpy(feat2).float()
        
        # Flatten for comparison (preserve last dimension which is feature dimension)
        if feat1_t.dim() == 2:
            feat1_flat = feat1_t
        else:
            feat1_flat = feat1_t.reshape(-1, feat1_t.shape[-1])
        
        if feat2_t.dim() == 2:
            feat2_flat = feat2_t
        else:
            feat2_flat = feat2_t.reshape(-1, feat2_t.shape[-1])
        
        # Handle shape mismatch
        if feat1_flat.shape[0] != feat2_flat.shape[0]:
            print(f"⚠️  Shape mismatch: {feat1_flat.shape} vs {feat2_flat.shape}")
            min_len = min(feat1_flat.shape[0], feat2_flat.shape[0])
            feat1_flat = feat1_flat[:min_len]
            feat2_flat = feat2_flat[:min_len]
            print(f"    Comparing first {min_len} tokens")
        
        # Compute metrics
        max_diff = (feat1_flat - feat2_flat).abs().max().item()
        mean_diff = (feat1_flat - feat2_flat).abs().mean().item()
        
        cos_sim = F.cosine_similarity(feat1_flat, feat2_flat, dim=-1)
        min_cos = cos_sim.min().item()
        mean_cos = cos_sim.mean().item()
        max_cos = cos_sim.max().item()
        
        print(f"Max Diff:        {max_diff:.6f}")
        print(f"Mean Diff:       {mean_diff:.6f}")
        print(f"Min Cosine Sim:  {min_cos:.8f}")
        print(f"Mean Cosine Sim: {mean_cos:.8f}")
        print(f"Max Cosine Sim:  {max_cos:.8f}")
        
        # Pass/Fail
        if min_cos > threshold:
            print(f"✅ {key}: PASS (min cosine > {threshold})")
            results[key] = True
        else:
            print(f"❌ {key}: FAIL (min cosine <= {threshold})")
            results[key] = False
    
    # Summary
    print("\n" + "=" * 80)
    print("Verification Summary")
    print("=" * 80)
    
    total = len(results)
    passed = sum(results.values())
    failed = total - passed
    
    print(f"Total comparisons: {total}")
    print(f"Passed:            {passed}")
    print(f"Failed:            {failed}")
    
    if failed == 0:
        print("\n✅ All features match! Models are consistent.")
        return True
    else:
        print(f"\n❌ {failed} feature(s) do not match. Please investigate.")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Extract second-to-last layer features from ViT models and verify consistency"
    )
    parser.add_argument("--hf_model_path", type=str, required=True,
                       help="Path to non-packing HF model")
    parser.add_argument("--packing_model_path", type=str, required=True,
                       help="Path to packing model")
    parser.add_argument("--image1", type=str, required=True,
                       help="Path to first image (e.g., 1.jpg)")
    parser.add_argument("--image2", type=str, required=True,
                       help="Path to second image (e.g., 2.jpg)")
    parser.add_argument("--video1", type=str, required=True,
                       help="Path to first video (e.g., 1.mp4)")
    parser.add_argument("--video2", type=str, required=True,
                       help="Path to second video (e.g., 2.mp4)")
    parser.add_argument("--num_frames", type=int, default=8,
                       help="Number of frames to sample from videos (default: 8)")
    parser.add_argument("--output_dir", type=str, default="./features",
                       help="Output directory for features (default: ./features)")
    parser.add_argument("--threshold", type=float, default=0.99,
                       help="Cosine similarity threshold for verification (default: 0.99)")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load models
    print("\n=== Loading Models ===")
    print(f"Loading HF model from: {args.hf_model_path}")
    hf_model = LlavaViTModel.from_pretrained(args.hf_model_path, torch_dtype=torch.bfloat16)
    hf_model.to(device)
    hf_model.eval()
    print("✅ HF model loaded")
    
    print(f"Loading Packing model from: {args.packing_model_path}")
    packing_model = LlavaViTPackingModel.from_pretrained(args.packing_model_path, torch_dtype=torch.bfloat16)
    packing_model.to(device)
    packing_model.eval()
    print("✅ Packing model loaded")
    
    # Load inputs
    print("\n=== Loading Inputs ===")
    image1, img1_meta = load_image(args.image1, device)
    image2, img2_meta = load_image(args.image2, device)
    video1, vid1_meta = load_video(args.video1, args.num_frames, device)
    video2, vid2_meta = load_video(args.video2, args.num_frames, device)
    
    images = [image1, image2]
    videos = [video1, video2]
    
    # Extract features
    hf_features = extract_hf_features(hf_model, images, videos, device)
    packing_features = extract_packing_features(packing_model, images, videos, device)
    
    # Save features
    print("\n=== Saving Features ===")
    hf_output = os.path.join(args.output_dir, "features_hf.npz")
    np.savez(hf_output, **hf_features)
    print(f"✅ HF features saved to: {hf_output}")
    
    packing_output = os.path.join(args.output_dir, "features_packing.npz")
    np.savez(packing_output, **packing_features)
    print(f"✅ Packing features saved to: {packing_output}")
    
    # Save metadata
    metadata = {
        "hf_model_path": args.hf_model_path,
        "packing_model_path": args.packing_model_path,
        "num_frames": args.num_frames,
        "device": str(device),
        "inputs": {
            "image1": img1_meta,
            "image2": img2_meta,
            "video1": vid1_meta,
            "video2": vid2_meta,
        },
        "features": {
            "hf": {k: list(v.shape) for k, v in hf_features.items()},
            "packing": {k: list(v.shape) for k, v in packing_features.items()},
        }
    }
    
    metadata_output = os.path.join(args.output_dir, "metadata.json")
    with open(metadata_output, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✅ Metadata saved to: {metadata_output}")
    
    # Verify consistency
    all_passed = verify_feature_consistency(hf_features, packing_features, args.threshold)
    
    print("\n=== Feature Extraction and Verification Complete ===")
    print(f"All outputs saved to: {args.output_dir}")
    
    # Return exit code based on verification result
    return 0 if all_passed else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
