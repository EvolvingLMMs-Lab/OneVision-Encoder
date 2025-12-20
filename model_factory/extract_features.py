#!/usr/bin/env python3
# coding=utf-8
"""
Feature Extraction and Verification Tool

This tool extracts second-to-last layer features from both packing and non-packing
HuggingFace ViT models and immediately verifies their consistency.

Usage (simple, with root only):
    python extract_features.py \\
        --hf_model_path /path/to/hf_model \\
        --packing_model_path /path/to/packing_model \\
        --root ./images \\
        --output_dir ./features \\
        --threshold 0.99

It will automatically use:
    <root>/1.jpg
    <root>/2.jpg
    <root>/1.mp4
    <root>/2.mp4

You can still override any of them with explicit paths:
    --image1 some/other/1.jpg  etc.
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


PATCH_DIVISOR = 14  # H, W 都要能被 14 整除


def _ceil_to_multiple(x: int, k: int) -> int:
    """返回 >= x 的最小的 k 的倍数。"""
    return ((x + k - 1) // k) * k


def load_image(image_path: str, device: torch.device) -> Tuple[torch.Tensor, Dict]:
    """
    加载图片，并把分辨率 resize 到可以被 14 整除，然后做 CLIP 归一化。

    返回:
        image_tensor: [1, 3, H, W] (H, W 已经是 14 的倍数)
        metadata: 包含原始/resize 后分辨率
    """
    print(f"Loading image: {image_path}")
    img = Image.open(image_path).convert("RGB")
    orig_width, orig_height = img.size

    # 调整到 14 的倍数
    new_width = _ceil_to_multiple(orig_width, PATCH_DIVISOR)
    new_height = _ceil_to_multiple(orig_height, PATCH_DIVISOR)

    if (new_width, new_height) != (orig_width, orig_height):
        print(f"  Resize from {orig_width}x{orig_height} to {new_width}x{new_height} (multiple of {PATCH_DIVISOR})")
        img = img.resize((new_width, new_height), Image.BICUBIC)
    else:
        print(f"  Resolution already multiple of {PATCH_DIVISOR}: {orig_width}x{orig_height}")

    width, height = img.size

    img_tensor = torch.from_numpy(np.array(img)).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1)  # [H, W, C] -> [C, H, W]

    mean = torch.tensor(CLIP_MEAN).view(3, 1, 1)
    std = torch.tensor(CLIP_STD).view(3, 1, 1)
    img_tensor = (img_tensor - mean) / std

    img_tensor = img_tensor.unsqueeze(0).to(device)

    metadata = {
        "path": image_path,
        "orig_width": orig_width,
        "orig_height": orig_height,
        "width": width,
        "height": height,
        "shape": list(img_tensor.shape),
    }

    print(f"  Final resolution: {width}x{height}")
    return img_tensor, metadata


def load_video(video_path: str, num_frames: int, device: torch.device) -> Tuple[torch.Tensor, Dict]:
    """
    加载视频，抽帧，然后把每帧 resize 到高宽都能被 14 整除，再做 CLIP 归一化。

    返回:
        video_tensor: [1, 3, T, H, W] (H, W 为 14 的倍数)
        metadata: 包含原始/resize 后分辨率
    """
    print(f"Loading video: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"  Total frames: {total_frames}, FPS: {fps}, Resolution: {orig_width}x{orig_height}")

    if total_frames == 1:
        frame_indices = np.array([0] * num_frames, dtype=int)
    else:
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    print(f"  Sampling frames: {frame_indices.tolist()}")

    # 计算 resize 后分辨率（对所有帧统一）
    new_width = _ceil_to_multiple(orig_width, PATCH_DIVISOR)
    new_height = _ceil_to_multiple(orig_height, PATCH_DIVISOR)
    if (new_width, new_height) != (orig_width, orig_height):
        print(f"  Resize frames from {orig_width}x{orig_height} to {new_width}x{new_height} (multiple of {PATCH_DIVISOR})")
    else:
        print(f"  Frame resolution already multiple of {PATCH_DIVISOR}: {orig_width}x{orig_height}")

    frames = []
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            raise ValueError(f"Failed to read frame {frame_idx} from {video_path}")

        # BGR -> RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # resize 到 14 的倍数
        if (new_width, new_height) != (orig_width, orig_height):
            frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

        frames.append(frame)

    cap.release()

    video_array = np.stack(frames, axis=0)  # [T, H, W, C]
    video_tensor = torch.from_numpy(video_array).float() / 255.0
    video_tensor = video_tensor.permute(3, 0, 1, 2)  # [C, T, H, W]

    mean = torch.tensor(CLIP_MEAN).view(3, 1, 1, 1)
    std = torch.tensor(CLIP_STD).view(3, 1, 1, 1)
    video_tensor = (video_tensor - mean) / std

    video_tensor = video_tensor.unsqueeze(0).to(device)

    _, _, T, H, W = video_tensor.shape

    metadata = {
        "path": video_path,
        "orig_width": orig_width,
        "orig_height": orig_height,
        "width": W,
        "height": H,
        "total_frames": total_frames,
        "sampled_frames": num_frames,
        "frame_indices": frame_indices.tolist(),
        "fps": fps,
        "shape": list(video_tensor.shape),
    }

    print(f"  Final frame resolution: {W}x{H}")
    return video_tensor, metadata


def extract_hf_features(
    model: LlavaViTModel,
    images: List[torch.Tensor],
    videos: List[torch.Tensor],
    device: torch.device
) -> Dict[str, np.ndarray]:
    print("\n=== Extracting features from HF model ===")
    model.eval()
    features = {}

    with torch.no_grad():
        for i, img in enumerate(images, 1):
            print(f"Processing image {i}...")
            img = img.to(device, dtype=torch.bfloat16)
            output = model(img, output_hidden_states=True)
            feat = output.hidden_states[-2].cpu().float().numpy()
            features[f"image{i}"] = feat
            print(f"  Feature shape: {feat.shape}")

        for i, video in enumerate(videos, 1):
            print(f"Processing video {i}...")
            video = video.to(device, dtype=torch.bfloat16)

            bs, C, T, H, W = video.shape
            patch_size = model.config.patch_size
            h_patches = H // patch_size
            w_patches = W // patch_size
            frame_tokens = h_patches * w_patches
            target_frames = 64

            frame_indices = torch.arange(T).unsqueeze(0).to(device)
            total_frames_tensor = torch.tensor([T]).to(device)
            interpolated_indices = interpolate_frame_indices(
                frame_indices, total_frames_tensor, target_frames
            )

            padded_video = torch.zeros(bs, C, target_frames, H, W, device=device, dtype=video.dtype)
            frame_idx_expanded = interpolated_indices.view(bs, 1, T, 1, 1).expand(bs, C, T, H, W)
            padded_video.scatter_(dim=2, index=frame_idx_expanded, src=video)

            per = torch.arange(frame_tokens, device=device)
            visible_index = (interpolated_indices.unsqueeze(-1) * frame_tokens + per).reshape(bs, -1)
            visible_index = visible_index.clamp_max(target_frames * frame_tokens - 1)

            output = model(padded_video, visible_indices=visible_index, output_hidden_states=True)
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
    print("\n=== Extracting features from Packing model ===")
    model.eval()
    features = {}

    patch_size = model.config.patch_size
    channels = 3
    target_frames = 64

    with torch.no_grad():
        for i, img in enumerate(images, 1):
            print(f"Processing image {i}...")
            img = img.to(device, dtype=torch.bfloat16)

            bs, C, H, W = img.shape
            h_patches = H // patch_size
            w_patches = W // patch_size

            patches = img.view(bs, C, h_patches, patch_size, w_patches, patch_size)
            patches = patches.permute(0, 2, 4, 1, 3, 5).contiguous()
            seq_len = bs * h_patches * w_patches
            patch_dim = patch_size * patch_size * C
            hidden_states = patches.view(seq_len, patch_dim)

            grid_thw = torch.tensor([[1, h_patches, w_patches]], dtype=torch.long, device=device)
            patch_positions = compute_patch_positions_from_grid_thw(grid_thw)

            output = model(
                hidden_states=hidden_states,
                grid_thw=grid_thw,
                patch_positions=patch_positions,
                output_hidden_states=True
            )
            feat = output.hidden_states[-2].cpu().float().numpy()
            features[f"image{i}"] = feat
            print(f"  Feature shape: {feat.shape}")

        for i, video in enumerate(videos, 1):
            print(f"Processing video {i}...")
            video = video.to(device, dtype=torch.bfloat16)

            bs, C, T, H, W = video.shape
            h_patches = H // patch_size
            w_patches = W // patch_size

            frame_indices = torch.arange(T).unsqueeze(0).to(device)
            total_frames_tensor = torch.tensor([T]).to(device)
            interpolated_indices = interpolate_frame_indices(
                frame_indices, total_frames_tensor, target_frames
            )

            patches = video.view(bs, C, T, h_patches, patch_size, w_patches, patch_size)
            patches = patches.permute(0, 2, 3, 5, 1, 4, 6).contiguous()
            seq_len = bs * T * h_patches * w_patches
            patch_dim = patch_size * patch_size * C
            hidden_states = patches.view(seq_len, patch_dim)

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
            feat = output.hidden_states[-2].cpu().float().numpy()
            features[f"video{i}"] = feat
            print(f"  Feature shape: {feat.shape}")

    return features


def verify_feature_consistency(hf_features: Dict[str, np.ndarray],
                               packing_features: Dict[str, np.ndarray],
                               threshold: float = 0.99) -> bool:
    print("\n" + "=" * 80)
    print("Feature Consistency Verification")
    print("=" * 80)
    print(f"Similarity threshold: {threshold}")

    results = {}

    for key in hf_features.keys():
        if key not in packing_features:
            print(f"\n[Warning] {key} not found in packing features")
            continue

        feat1 = hf_features[key]
        feat2 = packing_features[key]

        print(f"\n--- {key} ---")
        print(f"HF shape:      {feat1.shape}")
        print(f"Packing shape: {feat2.shape}")

        feat1_t = torch.from_numpy(feat1).float()
        feat2_t = torch.from_numpy(feat2).float()

        if feat1_t.dim() == 2:
            feat1_flat = feat1_t
        else:
            feat1_flat = feat1_t.reshape(-1, feat1_t.shape[-1])

        if feat2_t.dim() == 2:
            feat2_flat = feat2_t
        else:
            feat2_flat = feat2_t.reshape(-1, feat2_t.shape[-1])

        if feat1_flat.shape[0] != feat2_flat.shape[0]:
            print(f"[Warning] Shape mismatch: {feat1_flat.shape} vs {feat2_flat.shape}")
            min_len = min(feat1_flat.shape[0], feat2_flat.shape[0])
            feat1_flat = feat1_flat[:min_len]
            feat2_flat = feat2_flat[:min_len]
            print(f"  Comparing first {min_len} tokens")

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

        if min_cos > threshold:
            print(f"[OK] {key}: PASS (min cosine > {threshold})")
            results[key] = True
        else:
            print(f"[FAIL] {key}: FAIL (min cosine <= {threshold})")
            results[key] = False

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
        print("\nAll features match! Models are consistent.")
        return True
    else:
        print(f"\n{failed} feature(s) do not match. Please investigate.")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Extract second-to-last layer features from ViT models and verify consistency"
    )
    parser.add_argument("--hf_model_path", type=str, required=True,
                        help="Path to non-packing HF model")
    parser.add_argument("--packing_model_path", type=str, required=True,
                        help="Path to packing model")

    parser.add_argument("--root", type=str, default="./images",
                        help="Root directory for default inputs (default: ./images)")

    parser.add_argument("--image1", type=str, default=None,
                        help="Path to first image (default: <root>/1.jpg)")
    parser.add_argument("--image2", type=str, default=None,
                        help="Path to second image (default: <root>/2.jpg)")
    parser.add_argument("--video1", type=str, default=None,
                        help="Path to first video (default: <root>/1.mp4)")
    parser.add_argument("--video2", type=str, default=None,
                        help="Path to second video (default: <root>/2.mp4)")

    parser.add_argument("--num_frames", type=int, default=8,
                        help="Number of frames to sample from videos (default: 8)")
    parser.add_argument("--output_dir", type=str, default="./features",
                        help="Output directory for features (default: ./features)")
    parser.add_argument("--threshold", type=float, default=0.99,
                        help="Cosine similarity threshold for verification (default: 0.99)")

    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()

    image1_path = Path(args.image1) if args.image1 is not None else root / "1.jpg"
    image2_path = Path(args.image2) if args.image2 is not None else root / "2.jpg"
    video1_path = Path(args.video1) if args.video1 is not None else root / "1.mp4"
    video2_path = Path(args.video2) if args.video2 is not None else root / "2.mp4"

    print(f"Root directory: {root}")
    print(f"Using files:")
    print(f"  image1: {image1_path}")
    print(f"  image2: {image2_path}")
    print(f"  video1: {video1_path}")
    print(f"  video2: {video2_path}")

    for p in [image1_path, image2_path, video1_path, video2_path]:
        if not p.is_file():
            raise FileNotFoundError(f"Input file not found: {p}")

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("\n=== Loading Models ===")
    print(f"Loading HF model from: {args.hf_model_path}")
    hf_model = LlavaViTModel.from_pretrained(args.hf_model_path, torch_dtype=torch.bfloat16)
    hf_model.to(device)
    hf_model.eval()
    print("HF model loaded")

    print(f"Loading Packing model from: {args.packing_model_path}")
    packing_model = LlavaViTPackingModel.from_pretrained(args.packing_model_path, torch_dtype=torch.bfloat16)
    packing_model.to(device)
    packing_model.eval()
    print("Packing model loaded")

    print("\n=== Loading Inputs ===")
    image1, img1_meta = load_image(str(image1_path), device)
    image2, img2_meta = load_image(str(image2_path), device)
    video1, vid1_meta = load_video(str(video1_path), args.num_frames, device)
    video2, vid2_meta = load_video(str(video2_path), args.num_frames, device)

    images = [image1, image2]
    videos = [video1, video2]

    hf_features = extract_hf_features(hf_model, images, videos, device)
    packing_features = extract_packing_features(packing_model, images, videos, device)

    print("\n=== Saving Features ===")
    hf_output = os.path.join(args.output_dir, "features_hf.npz")
    np.savez(hf_output, **hf_features)
    print(f"HF features saved to: {hf_output}")

    packing_output = os.path.join(args.output_dir, "features_packing.npz")
    np.savez(packing_output, **packing_features)
    print(f"Packing features saved to: {packing_output}")

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
    print(f"Metadata saved to: {metadata_output}")

    all_passed = verify_feature_consistency(hf_features, packing_features, args.threshold)

    print("\n=== Feature Extraction and Verification Complete ===")
    print(f"All outputs saved to: {args.output_dir}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
