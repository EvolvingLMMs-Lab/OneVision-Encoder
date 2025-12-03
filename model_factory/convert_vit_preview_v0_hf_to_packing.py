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
Weight Conversion Tool: vit_preview_v0_hf.py → vit_preview_v0_packing_hf.py

This script converts weights from the HuggingFace ViT model format (vit_preview_v0_hf.py)
to the packing model format (vit_preview_v0_packing_hf.py) which uses grid_thw input
similar to Qwen2VL.

The conversion includes comprehensive tests to verify that the packing model produces
consistent outputs with the original HF model.

Reference: convert_llava_vit_packing_to_hf.py
"""

import argparse
import os
from pathlib import Path
import timm
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
import requests
from io import BytesIO
from torchvision import transforms
from transformers import CLIPImageProcessor

# Import model definitions
try:
    from model_factory import vit_preview_v0_hf
    from model_factory.vit_preview_v0_hf import LlavaViTModel
    from model_factory import vit_preview_v0_packing_hf
    from model_factory.vit_preview_v0_packing_hf import (
        LlavaViTPackingModel,
        compute_patch_positions_from_grid_thw,
    )
except ImportError:
    print("[Warning] Could not import model definitions directly. Ensure they are in PYTHONPATH.")

# CLIP Specific Constants
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


def get_real_coco_image(size=448):
    """
    Download a real COCO image and preprocess to Tensor (using CLIP mean/std, Float32)
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


def remap_state_dict_hf_to_packing(hf_state_dict):
    """
    Remap state dict from HF model format to packing model format.
    
    The main differences:
    1. HF model uses separate q_proj, k_proj, v_proj
       Packing model uses combined qkv projection
    2. HF model has embeddings.patch_embedding
       Packing model has patch_embed.proj
    3. Attention projection names differ slightly
    
    Args:
        hf_state_dict: State dict from vit_preview_v0_hf model
        
    Returns:
        new_dict: Remapped state dict for vit_preview_v0_packing_hf model
    """
    print("[Remap] Starting state dict remapping from HF to Packing format...")
    new_dict = {}
    qkv_cache = {}

    for k, v in hf_state_dict.items():
        new_k = k
        
        # Embeddings remapping
        if k.startswith("embeddings.patch_embedding."):
            new_k = k.replace("embeddings.patch_embedding.", "patch_embed.proj.")
        
        # LayerNorm remapping - names stay the same
        # (no changes needed for layernorm_pre.* and layernorm_post.*)
        
        # Encoder layers remapping
        # Most encoder layer names stay the same, except attention projections
        elif k.startswith("encoder.layers."):
            # Attention remapping - need to combine Q, K, V into QKV
            if ".self_attn.q_proj." in new_k or ".self_attn.k_proj." in new_k or ".self_attn.v_proj." in new_k:
                # Extract layer number and parameter type
                layer_match = new_k.split("encoder.layers.")[1].split(".")[0]
                param_type = new_k.split(".")[-1]  # weight or bias
                prefix = f"encoder.layers.{layer_match}.self_attn"
                
                if prefix not in qkv_cache:
                    qkv_cache[prefix] = {}
                
                if ".q_proj." in new_k:
                    proj_type = "q"
                elif ".k_proj." in new_k:
                    proj_type = "k"
                elif ".v_proj." in new_k:
                    proj_type = "v"
                
                if param_type not in qkv_cache[prefix]:
                    qkv_cache[prefix][param_type] = {}
                qkv_cache[prefix][param_type][proj_type] = v
                continue  # Don't add to new_dict yet
                
            elif ".self_attn.out_proj." in new_k:
                new_k = new_k.replace(".self_attn.out_proj.", ".self_attn.proj.")
        
        # Video RoPE remapping
        elif k.startswith("video_rope."):
            new_k = k.replace("video_rope.", "rotary_emb.")
        
        # Head and other layers keep their names (no elif needed as new_k already = k)
        
        new_dict[new_k] = v

    # Combine Q, K, V projections into QKV
    print(f"[Remap] Combining Q/K/V into QKV for {len(qkv_cache)} layers...")
    for prefix, params_dict in qkv_cache.items():
        for param_type, proj_dict in params_dict.items():
            if 'q' in proj_dict and 'k' in proj_dict and 'v' in proj_dict:
                # Concatenate Q, K, V along dimension 0
                qkv_combined = torch.cat([
                    proj_dict['q'],
                    proj_dict['k'],
                    proj_dict['v']
                ], dim=0)
                new_dict[f"{prefix}.qkv.{param_type}"] = qkv_combined
            else:
                print(f"[Warning] Incomplete QKV for {prefix}, param_type={param_type}")

    return new_dict


def verify_consistency_packing(hf_model, packing_model, real_image_tensor):
    """
    Verify consistency between the HF model and the packing model with grid_thw input.
    
    This function tests that the packing model (which uses grid_thw input like Qwen2VL)
    produces consistent outputs with the original HF model.
    """
    print("\n=== Verifying Consistency with Packing Model (grid_thw input - bfloat16) ===")

    hf_model.eval()
    packing_model.eval()

    device = next(hf_model.parameters()).device
    print(f"    Running on Device: {device}")

    dtype = next(hf_model.parameters()).dtype
    print(f"    Model Dtype: {dtype}")

    # Prepare input tensor
    input_tensor = real_image_tensor.to(device, dtype=torch.bfloat16)
    print(f"    Input Shape: {input_tensor.shape} | Dtype: {input_tensor.dtype}")

    # Get patch size from HF model
    patch_size = hf_model.config.patch_size
    print(f"    Patch Size: {patch_size}")

    # Calculate grid dimensions
    bs, channels, height, width = input_tensor.shape
    h_patches = height // patch_size
    w_patches = width // patch_size
    t_frames = 1

    # Create grid_thw tensor
    grid_thw = torch.tensor([[t_frames, h_patches, w_patches]], dtype=torch.long, device=device)
    print(f"    grid_thw: {grid_thw}")

    with torch.no_grad():
        # HF model forward
        try:
            hf_out = hf_model(input_tensor)
            hf_feat = hf_out.last_hidden_state
            hf_head = hf_out.pooler_output
        except Exception as e:
            print(f"    [Error] HF model forward failed: {e}")
            import traceback
            traceback.print_exc()
            return

        # Packing model forward - expects input in [seq_len, patch_dim] format
        try:
            # Reshape image to patches: (B, C, H, W) -> (seq_len, patch_dim)
            patches = input_tensor.view(
                bs, channels, h_patches, patch_size, w_patches, patch_size
            )
            patches = patches.permute(0, 2, 4, 1, 3, 5).contiguous()
            seq_len = bs * t_frames * h_patches * w_patches
            patch_dim = patch_size * patch_size * channels
            hidden_states = patches.view(seq_len, patch_dim)

            # Compute patch_positions from grid_thw for RoPE calculation
            patch_positions = compute_patch_positions_from_grid_thw(grid_thw)

            print(f"    Packing input shape: {hidden_states.shape} (seq_len={seq_len}, patch_dim={patch_dim})")
            print(f"    patch_positions shape: {patch_positions.shape}")

            packing_out = packing_model(
                hidden_states=hidden_states,
                grid_thw=grid_thw,
                patch_positions=patch_positions,
            )
            packing_feat = packing_out.last_hidden_state
            packing_head = packing_out.pooler_output
        except Exception as e:
            print(f"    [Error] Packing model forward failed: {e}")
            import traceback
            traceback.print_exc()
            return

    # Compare outputs
    if hf_feat is not None and packing_feat is not None:
        # Reshape for comparison (hf_feat: (B, N, C), packing_feat: (total_N, C))
        hf_feat_flat = hf_feat.flatten(0, -2).float()  # (B*N, C)
        packing_feat_flat = packing_feat.float()  # (total_N, C)

        # Check if shapes match
        if hf_feat_flat.shape[0] != packing_feat_flat.shape[0]:
            print(f"    [Warning] Shape mismatch: hf {hf_feat_flat.shape} vs packing {packing_feat_flat.shape}")
            min_len = min(hf_feat_flat.shape[0], packing_feat_flat.shape[0])
            hf_feat_flat = hf_feat_flat[:min_len]
            packing_feat_flat = packing_feat_flat[:min_len]

        diff_feat = (hf_feat_flat - packing_feat_flat).abs().max().item()
        cos_sim = F.cosine_similarity(hf_feat_flat, packing_feat_flat, dim=-1)

        min_cos = cos_sim.min().item()
        mean_cos = cos_sim.mean().item()

        print(f"    [Packing Feature] Max Diff:       {diff_feat:.6f}")
        print(f"    [Packing Feature] Min Cosine Sim: {min_cos:.8f} (Mean: {mean_cos:.8f})")

        if min_cos > 0.99:
            print("    ✅ Packing Feature: PASS")
        else:
            print("    ❌ Packing Feature: FAIL")

    if hf_head is not None and packing_head is not None:
        diff_head = (hf_head - packing_head).abs().max().item()
        hf_head_f = hf_head.float()
        packing_head_f = packing_head.float()
        cos_sim_head = F.cosine_similarity(hf_head_f, packing_head_f, dim=-1)

        min_cos_head = cos_sim_head.min().item()
        mean_cos_head = cos_sim_head.mean().item()

        print(f"    [Packing Head]    Max Diff:       {diff_head:.6f}")
        print(f"    [Packing Head]    Min Cosine Sim: {min_cos_head:.8f} (Mean: {mean_cos_head:.8f})")

        if min_cos_head > 0.99:
            print("    ✅ Packing Head:    PASS")
        else:
            print("    ❌ Packing Head:    FAIL")


def interpolate_frame_indices(frame_indices: torch.Tensor, total_frames: torch.Tensor, target_frames: int = 64) -> torch.Tensor:
    """
    Interpolate frame indices from original video frame count to target frame count.
    """
    bs, seq_len = frame_indices.shape
    device = frame_indices.device

    total_frames_float = total_frames.float().view(bs, 1)
    frame_indices_float = frame_indices.float()

    total_frames_safe = torch.clamp(total_frames_float - 1, min=1.0)
    interpolated_indices = (frame_indices_float / total_frames_safe) * (target_frames - 1)
    interpolated_indices = torch.round(interpolated_indices).long()
    interpolated_indices = torch.clamp(interpolated_indices, 0, target_frames - 1)

    return interpolated_indices


def get_synthesized_video(real_image_tensor, num_frames=8, size=224):
    """
    Create a synthesized video by stacking the real image multiple times.
    """
    if real_image_tensor.shape[-1] != size or real_image_tensor.shape[-2] != size:
        real_image_tensor = F.interpolate(
            real_image_tensor.float(),
            size=(size, size),
            mode='bicubic',
            align_corners=False
        )

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
    """
    bs, num_frames = interpolated_indices.shape
    patches_per_frame = h_patches * w_patches

    positions = []
    for b in range(bs):
        for frame_idx in range(num_frames):
            t_pos = interpolated_indices[b, frame_idx].item()
            for h in range(h_patches):
                for w in range(w_patches):
                    positions.append([t_pos, h, w])

    return torch.tensor(positions, dtype=torch.long, device=device)


def verify_video_consistency_packing(hf_model, packing_model, real_image_tensor, num_frames=8, image_size=224):
    """
    Verify consistency between the HF model and the packing model with video input.
    """
    print(f"\n=== Verifying Video Consistency with Packing Model ({num_frames} frames - bfloat16) ===")

    hf_model.eval()
    packing_model.eval()

    device = next(hf_model.parameters()).device
    print(f"    Running on Device: {device}")

    dtype = next(hf_model.parameters()).dtype
    print(f"    Model Dtype: {dtype}")

    # Get patch size from HF model
    patch_size = hf_model.config.patch_size

    # Calculate grid dimensions
    channels = 3
    h_patches = image_size // patch_size
    w_patches = image_size // patch_size
    frame_tokens = h_patches * w_patches
    target_frames = 64  # HF model expects 64-frame context

    bs = 1

    # Create synthesized video
    video_tensor = get_synthesized_video(real_image_tensor, num_frames=num_frames, size=image_size)
    video_tensor = video_tensor.to(device, dtype=torch.bfloat16)
    print(f"    Video Input Shape: {video_tensor.shape} (B, C, T, H, W)")

    # Compute interpolated frame indices for 64-frame context
    frame_indices = torch.arange(num_frames).unsqueeze(0).to(device)
    total_frames_tensor = torch.tensor([num_frames]).to(device)
    interpolated_indices = interpolate_frame_indices(
        frame_indices, total_frames_tensor, target_frames
    )

    print(f"    Original frame indices: {frame_indices[0].tolist()}")
    print(f"    Interpolated indices (in 64-frame context): {interpolated_indices[0].tolist()}")

    with torch.no_grad():
        # HF model forward with visible_indices
        try:
            # Create 64-frame padded video
            padded_videos = torch.zeros(bs, channels, target_frames, image_size, image_size,
                                        device=device, dtype=video_tensor.dtype)

            # Scatter original frames into interpolated positions
            seq_len = frame_indices.shape[1]
            frame_idx_expanded = interpolated_indices.view(bs, 1, seq_len, 1, 1).expand(
                bs, channels, seq_len, image_size, image_size
            )
            padded_videos.scatter_(dim=2, index=frame_idx_expanded, src=video_tensor)

            # Compute visible_index
            per = torch.arange(frame_tokens, device=device)
            visible_index = (interpolated_indices.unsqueeze(-1) * frame_tokens + per).reshape(bs, -1)
            visible_index = visible_index.clamp_max(target_frames * frame_tokens - 1)

            print(f"    Padded video shape: {padded_videos.shape}")
            print(f"    visible_index shape: {visible_index.shape}")

            hf_out = hf_model(padded_videos, visible_indices=visible_index)
            hf_feat = hf_out.last_hidden_state
            hf_head = hf_out.pooler_output
        except Exception as e:
            print(f"    [Error] HF model forward failed: {e}")
            import traceback
            traceback.print_exc()
            return

        # Packing model forward
        try:
            # Reshape video to patches
            patches = video_tensor.view(
                bs, channels, num_frames, h_patches, patch_size, w_patches, patch_size
            )
            patches = patches.permute(0, 2, 3, 5, 1, 4, 6).contiguous()
            total_seq_len = bs * num_frames * h_patches * w_patches
            patch_dim = patch_size * patch_size * channels
            hidden_states = patches.view(total_seq_len, patch_dim)

            # Compute patch_positions with interpolated temporal positions
            patch_positions = compute_patch_positions_with_interpolated_temporal(
                interpolated_indices, h_patches, w_patches, device
            )

            grid_thw = torch.tensor([[num_frames, h_patches, w_patches]], dtype=torch.long, device=device)

            print(f"    Packing input shape: {hidden_states.shape}")
            print(f"    patch_positions shape: {patch_positions.shape}")
            print(f"    grid_thw: {grid_thw.tolist()}")

            packing_out = packing_model(
                hidden_states=hidden_states,
                grid_thw=grid_thw,
                patch_positions=patch_positions,
            )
            packing_feat = packing_out.last_hidden_state
            packing_head = packing_out.pooler_output
        except Exception as e:
            print(f"    [Error] Packing model forward failed: {e}")
            import traceback
            traceback.print_exc()
            return

    # Compare outputs
    if hf_feat is not None and packing_feat is not None:
        hf_feat_flat = hf_feat.flatten(0, -2).float()
        packing_feat_flat = packing_feat.float()

        if hf_feat_flat.shape[0] != packing_feat_flat.shape[0]:
            print(f"    [Warning] Shape mismatch: hf {hf_feat_flat.shape} vs packing {packing_feat_flat.shape}")
            min_len = min(hf_feat_flat.shape[0], packing_feat_flat.shape[0])
            hf_feat_flat = hf_feat_flat[:min_len]
            packing_feat_flat = packing_feat_flat[:min_len]

        diff_feat = (hf_feat_flat - packing_feat_flat).abs().max().item()
        cos_sim = F.cosine_similarity(hf_feat_flat, packing_feat_flat, dim=-1)

        min_cos = cos_sim.min().item()
        mean_cos = cos_sim.mean().item()

        print(f"    [Video Packing Feature] Max Diff:       {diff_feat:.6f}")
        print(f"    [Video Packing Feature] Min Cosine Sim: {min_cos:.8f} (Mean: {mean_cos:.8f})")

        if min_cos > 0.99:
            print("    ✅ Video Packing Feature: PASS")
        else:
            print("    ❌ Video Packing Feature: FAIL")

    if hf_head is not None and packing_head is not None:
        diff_head = (hf_head - packing_head).abs().max().item()
        hf_head_f = hf_head.float()
        packing_head_f = packing_head.float()
        cos_sim_head = F.cosine_similarity(hf_head_f, packing_head_f, dim=-1)

        min_cos_head = cos_sim_head.min().item()
        mean_cos_head = cos_sim_head.mean().item()

        print(f"    [Video Packing Head]    Max Diff:       {diff_head:.6f}")
        print(f"    [Video Packing Head]    Min Cosine Sim: {min_cos_head:.8f} (Mean: {mean_cos_head:.8f})")

        if min_cos_head > 0.99:
            print("    ✅ Video Packing Head:    PASS")
        else:
            print("    ❌ Video Packing Head:    FAIL")


def verify_mixed_video_image_consistency_packing(hf_model, packing_model, real_image_tensor, num_frames=8, video_size=224, image_size=448):
    """
    Verify consistency with mixed video+image input.
    """
    print(f"\n=== Verifying Mixed Video+Image Consistency ({num_frames} frames + image - bfloat16) ===")

    hf_model.eval()
    packing_model.eval()

    device = next(hf_model.parameters()).device
    print(f"    Running on Device: {device}")

    patch_size = hf_model.config.patch_size
    channels = 3
    target_frames = 64

    # Prepare video
    video_h_patches = video_size // patch_size
    video_w_patches = video_size // patch_size
    video_frame_tokens = video_h_patches * video_w_patches

    video_tensor = get_synthesized_video(real_image_tensor, num_frames=num_frames, size=video_size)
    video_tensor = video_tensor.to(device, dtype=torch.bfloat16)
    print(f"    Video Input Shape: {video_tensor.shape}")

    frame_indices = torch.arange(num_frames).unsqueeze(0).to(device)
    total_frames_tensor = torch.tensor([num_frames]).to(device)
    interpolated_indices = interpolate_frame_indices(frame_indices, total_frames_tensor, target_frames)

    # Prepare image
    image_h_patches = image_size // patch_size
    image_w_patches = image_size // patch_size

    image_tensor = real_image_tensor.to(device, dtype=torch.bfloat16)
    if image_tensor.shape[-1] != image_size or image_tensor.shape[-2] != image_size:
        image_tensor = F.interpolate(
            image_tensor.float(),
            size=(image_size, image_size),
            mode='bicubic',
            align_corners=False
        ).to(dtype=torch.bfloat16)
    print(f"    Image Input Shape: {image_tensor.shape}")

    bs = 1

    with torch.no_grad():
        # Process video with HF model
        try:
            padded_videos = torch.zeros(bs, channels, target_frames, video_size, video_size,
                                        device=device, dtype=video_tensor.dtype)
            seq_len = frame_indices.shape[1]
            frame_idx_expanded = interpolated_indices.view(bs, 1, seq_len, 1, 1).expand(
                bs, channels, seq_len, video_size, video_size
            )
            padded_videos.scatter_(dim=2, index=frame_idx_expanded, src=video_tensor)

            per = torch.arange(video_frame_tokens, device=device)
            visible_index = (interpolated_indices.unsqueeze(-1) * video_frame_tokens + per).reshape(bs, -1)
            visible_index = visible_index.clamp_max(target_frames * video_frame_tokens - 1)

            hf_video_out = hf_model(padded_videos, visible_indices=visible_index)
            hf_video_feat = hf_video_out.last_hidden_state
        except Exception as e:
            print(f"    [Error] HF video forward failed: {e}")
            return

        # Process image with HF model
        try:
            hf_image_out = hf_model(image_tensor)
            hf_image_feat = hf_image_out.last_hidden_state
        except Exception as e:
            print(f"    [Error] HF image forward failed: {e}")
            return

        # Process combined input with packing model
        try:
            # Prepare video patches
            video_patches = video_tensor.view(
                bs, channels, num_frames, video_h_patches, patch_size, video_w_patches, patch_size
            )
            video_patches = video_patches.permute(0, 2, 3, 5, 1, 4, 6).contiguous()
            video_seq_len = bs * num_frames * video_h_patches * video_w_patches
            patch_dim = patch_size * patch_size * channels
            video_hidden_states = video_patches.view(video_seq_len, patch_dim)

            video_patch_positions = compute_patch_positions_with_interpolated_temporal(
                interpolated_indices, video_h_patches, video_w_patches, device
            )

            # Prepare image patches
            image_patches = image_tensor.view(
                bs, channels, image_h_patches, patch_size, image_w_patches, patch_size
            )
            image_patches = image_patches.permute(0, 2, 4, 1, 3, 5).contiguous()
            image_seq_len = bs * 1 * image_h_patches * image_w_patches
            image_hidden_states = image_patches.view(image_seq_len, patch_dim)

            image_patch_positions = []
            for h in range(image_h_patches):
                for w in range(image_w_patches):
                    image_patch_positions.append([0, h, w])
            image_patch_positions = torch.tensor(image_patch_positions, dtype=torch.long, device=device)

            # Combine
            combined_hidden_states = torch.cat([video_hidden_states, image_hidden_states], dim=0)
            combined_patch_positions = torch.cat([video_patch_positions, image_patch_positions], dim=0)

            combined_grid_thw = torch.tensor([
                [num_frames, video_h_patches, video_w_patches],
                [1, image_h_patches, image_w_patches]
            ], dtype=torch.long, device=device)

            print(f"    Combined input shape: {combined_hidden_states.shape}")

            packing_out = packing_model(
                hidden_states=combined_hidden_states,
                grid_thw=combined_grid_thw,
                patch_positions=combined_patch_positions,
            )
            packing_feat = packing_out.last_hidden_state

            packing_video_feat = packing_feat[:video_seq_len]
            packing_image_feat = packing_feat[video_seq_len:]
        except Exception as e:
            print(f"    [Error] Packing model forward failed: {e}")
            import traceback
            traceback.print_exc()
            return

    # Compare video
    print("\n    --- Video Comparison ---")
    if hf_video_feat is not None and packing_video_feat is not None:
        hf_video_flat = hf_video_feat.flatten(0, -2).float()
        packing_video_flat = packing_video_feat.float()

        if hf_video_flat.shape[0] != packing_video_flat.shape[0]:
            min_len = min(hf_video_flat.shape[0], packing_video_flat.shape[0])
            hf_video_flat = hf_video_flat[:min_len]
            packing_video_flat = packing_video_flat[:min_len]

        cos_sim = F.cosine_similarity(hf_video_flat, packing_video_flat, dim=-1)
        min_cos = cos_sim.min().item()
        mean_cos = cos_sim.mean().item()

        print(f"    [Mixed Video] Min Cosine Sim: {min_cos:.8f} (Mean: {mean_cos:.8f})")
        video_pass = min_cos > 0.99
        print(f"    {'✅' if video_pass else '❌'} Mixed Video Feature: {'PASS' if video_pass else 'FAIL'}")
    else:
        video_pass = False

    # Compare image
    print("\n    --- Image Comparison ---")
    if hf_image_feat is not None and packing_image_feat is not None:
        hf_image_flat = hf_image_feat.flatten(0, -2).float()
        packing_image_flat = packing_image_feat.float()

        if hf_image_flat.shape[0] != packing_image_flat.shape[0]:
            min_len = min(hf_image_flat.shape[0], packing_image_flat.shape[0])
            hf_image_flat = hf_image_flat[:min_len]
            packing_image_flat = packing_image_flat[:min_len]

        cos_sim = F.cosine_similarity(hf_image_flat, packing_image_flat, dim=-1)
        min_cos = cos_sim.min().item()
        mean_cos = cos_sim.mean().item()

        print(f"    [Mixed Image] Min Cosine Sim: {min_cos:.8f} (Mean: {mean_cos:.8f})")
        image_pass = min_cos > 0.99
        print(f"    {'✅' if image_pass else '❌'} Mixed Image Feature: {'PASS' if image_pass else 'FAIL'}")
    else:
        image_pass = False

    print("\n    --- Overall Summary ---")
    if video_pass and image_pass:
        print("    ✅ Mixed Video+Image Consistency: ALL PASS")
    else:
        print("    ❌ Mixed Video+Image Consistency: SOME FAILED")


def verify_multi_sample_consistency_packing(hf_model, packing_model, real_image_tensor):
    """
    Verify consistency with multiple samples (3 images + 2 videos).
    """
    print("\n=== Verifying Multi-Sample Consistency (3 images + 2 videos - bfloat16) ===")

    hf_model.eval()
    packing_model.eval()

    device = next(hf_model.parameters()).device
    patch_size = hf_model.config.patch_size
    channels = 3
    target_frames = 64
    num_frames = 8

    # Define image and video sizes
    image_sizes = [224, 336, 1008]
    video_sizes = [378, 518]

    print(f"    Image resolutions: {image_sizes}")
    print(f"    Video resolutions: {video_sizes}")

    # Prepare images
    image_tensors = []
    image_h_patches_list = []
    image_w_patches_list = []
    for img_size in image_sizes:
        h_patches = img_size // patch_size
        w_patches = img_size // patch_size
        image_h_patches_list.append(h_patches)
        image_w_patches_list.append(w_patches)

        img_tensor = real_image_tensor.to(device, dtype=torch.bfloat16)
        if img_tensor.shape[-1] != img_size or img_tensor.shape[-2] != img_size:
            img_tensor = F.interpolate(
                img_tensor.float(),
                size=(img_size, img_size),
                mode='bicubic',
                align_corners=False
            ).to(dtype=torch.bfloat16)
        image_tensors.append(img_tensor)

    # Prepare videos
    video_tensors = []
    video_h_patches_list = []
    video_w_patches_list = []
    interpolated_indices_list = []
    for vid_size in video_sizes:
        h_patches = vid_size // patch_size
        w_patches = vid_size // patch_size
        video_h_patches_list.append(h_patches)
        video_w_patches_list.append(w_patches)

        vid_tensor = get_synthesized_video(real_image_tensor, num_frames=num_frames, size=vid_size)
        vid_tensor = vid_tensor.to(device, dtype=torch.bfloat16)
        video_tensors.append(vid_tensor)

        frame_indices = torch.arange(num_frames).unsqueeze(0).to(device)
        total_frames_tensor = torch.tensor([num_frames]).to(device)
        interpolated_indices = interpolate_frame_indices(frame_indices, total_frames_tensor, target_frames)
        interpolated_indices_list.append(interpolated_indices)

    bs = 1

    with torch.no_grad():
        # Process each sample with HF model
        hf_image_feats = []
        hf_video_feats = []

        for img_tensor in image_tensors:
            hf_out = hf_model(img_tensor)
            hf_image_feats.append(hf_out.last_hidden_state)

        for vid_tensor, interpolated_indices, h_patches, w_patches in zip(
            video_tensors, interpolated_indices_list, video_h_patches_list, video_w_patches_list
        ):
            vid_size = vid_tensor.shape[-1]
            frame_tokens = h_patches * w_patches

            padded_videos = torch.zeros(bs, channels, target_frames, vid_size, vid_size,
                                        device=device, dtype=vid_tensor.dtype)
            seq_len = num_frames
            frame_idx_expanded = interpolated_indices.view(bs, 1, seq_len, 1, 1).expand(
                bs, channels, seq_len, vid_size, vid_size
            )
            padded_videos.scatter_(dim=2, index=frame_idx_expanded, src=vid_tensor)

            per = torch.arange(frame_tokens, device=device)
            visible_index = (interpolated_indices.unsqueeze(-1) * frame_tokens + per).reshape(bs, -1)
            visible_index = visible_index.clamp_max(target_frames * frame_tokens - 1)

            hf_out = hf_model(padded_videos, visible_indices=visible_index)
            hf_video_feats.append(hf_out.last_hidden_state)

        # Process all samples together with packing model
        patch_dim = patch_size * patch_size * channels
        all_hidden_states = []
        all_patch_positions = []
        grid_thw_list = []
        seq_lengths = []

        # Prepare image patches
        for img_tensor, h_patches, w_patches in zip(image_tensors, image_h_patches_list, image_w_patches_list):
            img_patches = img_tensor.view(bs, channels, h_patches, patch_size, w_patches, patch_size)
            img_patches = img_patches.permute(0, 2, 4, 1, 3, 5).contiguous()
            img_seq_len = bs * 1 * h_patches * w_patches
            img_hidden_states = img_patches.view(img_seq_len, patch_dim)
            all_hidden_states.append(img_hidden_states)
            seq_lengths.append(img_seq_len)

            img_patch_positions = []
            for h in range(h_patches):
                for w in range(w_patches):
                    img_patch_positions.append([0, h, w])
            img_patch_positions = torch.tensor(img_patch_positions, dtype=torch.long, device=device)
            all_patch_positions.append(img_patch_positions)

            grid_thw_list.append([1, h_patches, w_patches])

        # Prepare video patches
        for vid_tensor, interpolated_indices, h_patches, w_patches in zip(
            video_tensors, interpolated_indices_list, video_h_patches_list, video_w_patches_list
        ):
            vid_patches = vid_tensor.view(bs, channels, num_frames, h_patches, patch_size, w_patches, patch_size)
            vid_patches = vid_patches.permute(0, 2, 3, 5, 1, 4, 6).contiguous()
            vid_seq_len = bs * num_frames * h_patches * w_patches
            vid_hidden_states = vid_patches.view(vid_seq_len, patch_dim)
            all_hidden_states.append(vid_hidden_states)
            seq_lengths.append(vid_seq_len)

            vid_patch_positions = compute_patch_positions_with_interpolated_temporal(
                interpolated_indices, h_patches, w_patches, device
            )
            all_patch_positions.append(vid_patch_positions)

            grid_thw_list.append([num_frames, h_patches, w_patches])

        # Combine all samples
        combined_hidden_states = torch.cat(all_hidden_states, dim=0)
        combined_patch_positions = torch.cat(all_patch_positions, dim=0)
        combined_grid_thw = torch.tensor(grid_thw_list, dtype=torch.long, device=device)

        print(f"    Combined input shape: {combined_hidden_states.shape}")

        packing_out = packing_model(
            hidden_states=combined_hidden_states,
            grid_thw=combined_grid_thw,
            patch_positions=combined_patch_positions,
        )
        packing_feat = packing_out.last_hidden_state

        # Split outputs
        packing_image_feats = []
        packing_video_feats = []
        current_idx = 0
        for i in range(len(image_tensors)):
            sample_len = seq_lengths[i]
            packing_image_feats.append(packing_feat[current_idx:current_idx + sample_len])
            current_idx += sample_len
        for i in range(len(video_tensors)):
            sample_len = seq_lengths[len(image_tensors) + i]
            packing_video_feats.append(packing_feat[current_idx:current_idx + sample_len])
            current_idx += sample_len

    # Compare all samples
    all_pass = True

    for i in range(len(image_tensors)):
        hf_feat = hf_image_feats[i].flatten(0, -2).float()
        packing_feat = packing_image_feats[i].float()

        if hf_feat.shape[0] != packing_feat.shape[0]:
            min_len = min(hf_feat.shape[0], packing_feat.shape[0])
            hf_feat = hf_feat[:min_len]
            packing_feat = packing_feat[:min_len]

        cos_sim = F.cosine_similarity(hf_feat, packing_feat, dim=-1)
        min_cos = cos_sim.min().item()

        print(f"    [Image {i+1} (res={image_sizes[i]})] Min Cos: {min_cos:.8f}")
        if min_cos <= 0.99:
            all_pass = False

    for i in range(len(video_tensors)):
        hf_feat = hf_video_feats[i].flatten(0, -2).float()
        packing_feat = packing_video_feats[i].float()

        if hf_feat.shape[0] != packing_feat.shape[0]:
            min_len = min(hf_feat.shape[0], packing_feat.shape[0])
            hf_feat = hf_feat[:min_len]
            packing_feat = packing_feat[:min_len]

        cos_sim = F.cosine_similarity(hf_feat, packing_feat, dim=-1)
        min_cos = cos_sim.min().item()

        print(f"    [Video {i+1} (res={video_sizes[i]})] Min Cos: {min_cos:.8f}")
        if min_cos <= 0.99:
            all_pass = False

    if all_pass:
        print("    ✅ Multi-Sample Consistency: ALL PASS")
    else:
        print("    ❌ Multi-Sample Consistency: SOME FAILED")


def verify_saved_model_loading_packing(hf_model, output_dir, real_image_tensor):
    """
    Verify that the saved packing model can be loaded and produces consistent results.
    """
    print("\n=== Verifying Loaded Saved Packing Model ===")
    print(f"--> Loading from: {output_dir}")

    device = next(hf_model.parameters()).device

    try:
        print("    Loading Image Processor...")
        image_processor = CLIPImageProcessor.from_pretrained(output_dir)

        print("    Loading Vision Tower (LlavaViTPackingModel) with torch_dtype=bfloat16...")
        vision_tower = LlavaViTPackingModel.from_pretrained(output_dir, torch_dtype=torch.bfloat16)
        vision_tower.to(device)
        vision_tower.eval()

        print("    ✅ Successfully loaded model")
    except Exception as e:
        print(f"    ❌ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return

    # Test with image
    patch_size = hf_model.config.patch_size
    input_tensor = real_image_tensor.to(device, dtype=torch.bfloat16)

    bs, channels, height, width = input_tensor.shape
    h_patches = height // patch_size
    w_patches = width // patch_size
    t_frames = 1

    grid_thw = torch.tensor([[t_frames, h_patches, w_patches]], dtype=torch.long, device=device)

    with torch.no_grad():
        hf_out = hf_model(input_tensor).last_hidden_state

        patches = input_tensor.view(bs, channels, h_patches, patch_size, w_patches, patch_size)
        patches = patches.permute(0, 2, 4, 1, 3, 5).contiguous()
        seq_len = bs * t_frames * h_patches * w_patches
        patch_dim = patch_size * patch_size * channels
        hidden_states = patches.view(seq_len, patch_dim)

        patch_positions = compute_patch_positions_from_grid_thw(grid_thw)

        tgt_out = vision_tower(
            hidden_states=hidden_states,
            grid_thw=grid_thw,
            patch_positions=patch_positions,
        ).last_hidden_state

    hf_feat_flat = hf_out.flatten(0, -2).float()
    tgt_feat_flat = tgt_out.float()

    if hf_feat_flat.shape[0] != tgt_feat_flat.shape[0]:
        min_len = min(hf_feat_flat.shape[0], tgt_feat_flat.shape[0])
        hf_feat_flat = hf_feat_flat[:min_len]
        tgt_feat_flat = tgt_feat_flat[:min_len]

    cos_sim = F.cosine_similarity(hf_feat_flat, tgt_feat_flat, dim=-1)
    min_cos = cos_sim.min().item()

    print(f"    [Reloaded Model] Min Cosine Sim: {min_cos:.8f}")
    if min_cos > 0.99:
        print("    ✅ Reloaded Model Verification: PASS")
    else:
        print("    ❌ Reloaded Model Verification: FAIL")


def convert_and_save_packing(hf_model_name, packing_model_name, weight_path, output_dir):
    """
    Main conversion function.
    """
    print(f"=== HF to Packing Model Conversion ===")
    print(f"Source:  {hf_model_name}")
    print(f"Target:  {packing_model_name}")
    print(f"Weights: {weight_path}")
    print(f"Output:  {output_dir}")

    # Create HF model
    print("\n--> Creating HF Model...")
    hf_model = timm.create_model(hf_model_name, pretrained=False)

    print("--> Loading weights into HF model...")
    checkpoint = torch.load(weight_path, map_location='cpu')
    state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    hf_model.load_state_dict(state_dict, strict=False)

    # Create Packing model
    print("\n--> Creating Packing Model...")
    packing_model = timm.create_model(packing_model_name, pretrained=False)

    print("--> Remapping State Dict...")
    packing_state_dict = remap_state_dict_hf_to_packing(hf_model.state_dict())

    print("--> Loading weights into Packing model...")
    missing, unexpected = packing_model.load_state_dict(packing_state_dict, strict=False)

    # Filter out non-critical missing keys
    # Note: 'attn.bias' keys are internal PyTorch artifacts for attention masks
    # and don't represent actual learnable parameters
    real_missing = [k for k in missing if "attn.bias" not in k]
    if len(real_missing) > 0:
        print(f"    [Warning] Missing keys ({len(real_missing)}):")
        for k in real_missing[:5]:
            print(f"      {k}")
    else:
        print("    Load OK (No critical missing keys).")

    # Move to device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"\n--> Moving models to {device} and casting to bfloat16...")
        hf_model.to(device, dtype=torch.bfloat16)
        packing_model.to(device, dtype=torch.bfloat16)
    else:
        device = torch.device("cpu")
        print(f"\n--> [WARNING] CUDA not available. Tests may be slow.")
        hf_model.to(device, dtype=torch.bfloat16)
        packing_model.to(device, dtype=torch.bfloat16)

    print("\n--> Fetching real image for verification (448x448)...")
    real_img = get_real_coco_image(size=448)

    # Run all verification tests
    verify_consistency_packing(hf_model, packing_model, real_img)
    verify_video_consistency_packing(hf_model, packing_model, real_img, num_frames=8, image_size=224)
    verify_mixed_video_image_consistency_packing(hf_model, packing_model, real_img, num_frames=8, video_size=224, image_size=448)
    verify_multi_sample_consistency_packing(hf_model, packing_model, real_img)

    if output_dir:
        print(f"\n--> Saving Packing Model to {output_dir}...")
        if hasattr(packing_model, "save_pretrained"):
            packing_model.save_pretrained(output_dir)

            print("    Saving CLIPImageProcessor config...")
            processor = CLIPImageProcessor(
                size=448,
                crop_size=448,
                image_mean=CLIP_MEAN,
                image_std=CLIP_STD,
                resample=3,
                do_center_crop=True,
                do_normalize=True,
                do_resize=True,
                feature_extractor_type="CLIPFeatureExtractor"
            )
            processor.save_pretrained(output_dir)

            print("✅ Packing Model (bf16) and CLIP Processor saved.")

            verify_saved_model_loading_packing(hf_model, output_dir, real_img)
        else:
            print("❌ Error: Target model does not have save_pretrained method.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert vit_preview_v0_hf to vit_preview_v0_packing_hf format")
    parser.add_argument("model_name", type=str, help="Source HF model name")
    parser.add_argument("weight_path", type=str, help="Path to .pth checkpoint")
    parser.add_argument("--target_model_name", type=str, default=None, help="Target packing model name")
    parser.add_argument("--output_dir", type=str, default=None)

    args = parser.parse_args()

    tgt_name = args.target_model_name
    if tgt_name is None:
        # Convert name like hf_llava_vit_huge_ln to hf_llava_vit_packing_huge_ln
        # This handles the standard naming convention for LLaVA ViT models
        if "llava_vit_" in args.model_name:
            # Find first occurrence and insert "packing_" after it
            parts = args.model_name.split("llava_vit_", 1)
            tgt_name = parts[0] + "llava_vit_packing_" + parts[1]
        else:
            # Fallback: just append "_packing" to the model name
            tgt_name = args.model_name + "_packing"

    out_dir = args.output_dir
    if out_dir is None:
        p = Path(args.weight_path)
        out_dir = os.path.join(p.parent, f"{p.stem}_packing")

    convert_and_save_packing(args.model_name, tgt_name, args.weight_path, out_dir)
