#!/usr/bin/env python3
# coding=utf-8
"""
Feature Verification Tool

This tool loads and compares features extracted by extract_features.py
to verify consistency between packing and non-packing models.

Usage:
    python verify_features.py --features_dir ./features
"""

import argparse
import json
import os
import numpy as np
import torch
import torch.nn.functional as F


def load_features(features_dir: str):
    """Load features and metadata from directory."""
    hf_path = os.path.join(features_dir, "features_hf.npz")
    packing_path = os.path.join(features_dir, "features_packing.npz")
    metadata_path = os.path.join(features_dir, "metadata.json")
    
    if not os.path.exists(hf_path):
        raise FileNotFoundError(f"HF features not found: {hf_path}")
    if not os.path.exists(packing_path):
        raise FileNotFoundError(f"Packing features not found: {packing_path}")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")
    
    hf_features = np.load(hf_path)
    packing_features = np.load(packing_path)
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    return hf_features, packing_features, metadata


def compute_similarity(feat1: np.ndarray, feat2: np.ndarray, name: str, threshold: float = 0.99):
    """Compute and display similarity metrics between two features."""
    print(f"\n--- {name} ---")
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
        print(f"✅ {name}: PASS (min cosine > {threshold})")
        return True
    else:
        print(f"❌ {name}: FAIL (min cosine <= {threshold})")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Verify and compare extracted features"
    )
    parser.add_argument("--features_dir", type=str, default="./features",
                       help="Directory containing extracted features (default: ./features)")
    parser.add_argument("--threshold", type=float, default=0.99,
                       help="Cosine similarity threshold for pass/fail (default: 0.99)")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Feature Verification Tool")
    print("=" * 80)
    print(f"Similarity threshold: {args.threshold}")
    
    # Load features
    print(f"\nLoading features from: {args.features_dir}")
    hf_features, packing_features, metadata = load_features(args.features_dir)
    
    # Display metadata
    print("\n=== Metadata ===")
    print(f"HF Model:      {metadata['hf_model_path']}")
    print(f"Packing Model: {metadata['packing_model_path']}")
    print(f"Device:        {metadata['device']}")
    print(f"Num Frames:    {metadata['num_frames']}")
    
    print("\n=== Inputs ===")
    for input_name, input_meta in metadata['inputs'].items():
        print(f"{input_name}: {input_meta['path']}")
        if 'width' in input_meta:
            print(f"  Resolution: {input_meta['width']}x{input_meta['height']}")
        if 'sampled_frames' in input_meta:
            print(f"  Frames: {input_meta['sampled_frames']}/{input_meta['total_frames']}")
    
    # Compare features
    print("\n" + "=" * 80)
    print("Feature Comparison")
    print("=" * 80)
    
    results = {}
    for key in hf_features.keys():
        if key in packing_features:
            passed = compute_similarity(hf_features[key], packing_features[key], key, args.threshold)
            results[key] = passed
        else:
            print(f"\n⚠️  Warning: {key} not found in packing features")
    
    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    
    total = len(results)
    passed = sum(results.values())
    failed = total - passed
    
    print(f"Total comparisons: {total}")
    print(f"Passed:            {passed}")
    print(f"Failed:            {failed}")
    
    if failed == 0:
        print("\n✅ All features match! Models are consistent.")
        return 0
    else:
        print(f"\n❌ {failed} feature(s) do not match. Please investigate.")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
