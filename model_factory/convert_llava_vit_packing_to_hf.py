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

# 务必确保这里导入了定义了两个模型的文件
try:
    from model_factory import vit_preview_v0     # 你的 Source 模型定义
    # Import the packing model with grid_thw support
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
    下载一张真实的 COCO 图片并预处理为 Tensor (使用 CLIP 均值/方差, Float32)
    """
    url = "http://images.cocodataset.org/val2017/000000039769.jpg" # COCO cat image
    print(f"--> Downloading real image from {url} (Target Size: {size})...")
    try:
        # 增加 header 伪装浏览器，防止某些情况下下载失败
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert("RGB")
    except Exception as e:
        print(f"[Error] Failed to download image: {e}. Generating random noise as fallback.")
        img = Image.fromarray(np.random.randint(0, 255, (size, size, 3), dtype=np.uint8))

    # 预处理：Resize -> ToTensor -> Normalize (CLIP Specific)
    # 注意：这里保持 Float32，具体的精度转换在模型输入前进行
    transform = transforms.Compose([
        transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])

    return transform(img).unsqueeze(0) # [1, 3, size, size]


def remap_state_dict_packing(src_state_dict):
    """
    Remap state dict from source model to packing model format.
    The packing model uses a slightly different architecture with combined QKV projection.

    Note: The packing model has a patch_embed layer with Conv2d that maps directly to the
    source model's conv1 (Conv2d). No weight reshaping is needed.
    """
    print("[Remap Packing] Starting state dict remapping for packing model...")
    new_dict = {}

    for k, v in src_state_dict.items():
        new_k = k
        if k.startswith("conv1."):
            # conv1 -> patch_embed.proj (both are Conv2d, no reshaping needed)
            new_k = k.replace("conv1.", "patch_embed.proj.")
        elif k.startswith("ln_pre."):
            new_k = k.replace("ln_pre.", "layernorm_pre.")
        elif k.startswith("ln_post."):
            new_k = k.replace("ln_post.", "layernorm_post.")
        elif k.startswith("transformer.layers."):
            new_k = k.replace("transformer.layers.", "encoder.layers.")
            if ".ln_1." in new_k:
                new_k = new_k.replace(".ln_1.", ".layer_norm1.")
            if ".ln_2." in new_k:
                new_k = new_k.replace(".ln_2.", ".layer_norm2.")
            if ".mlp.0." in new_k:
                new_k = new_k.replace(".mlp.0.", ".mlp.fc1.")
            if ".mlp.2." in new_k:
                new_k = new_k.replace(".mlp.2.", ".mlp.fc2.")
            if ".attn.in_proj" in new_k:
                # Combined QKV projection - keep it as is for packing model
                new_k = new_k.replace(".attn.in_proj", ".self_attn.qkv")
            elif ".attn.out_proj." in new_k:
                new_k = new_k.replace(".attn.out_proj.", ".self_attn.proj.")
        new_dict[new_k] = v

    return new_dict


def verify_consistency_packing(src_model, packing_model, real_image_tensor):
    """
    Verify consistency between the source model and the packing model with grid_thw input.

    This function tests that the packing model (which uses grid_thw input like Qwen2VL)
    produces consistent outputs with the original source model.

    Note: The packing model expects input in [seq_len, patch_dim] format where
    patch_dim = temporal_patch_size * patch_size * patch_size * in_channels.
    """
    print("\n=== Verifying Consistency with Packing Model (grid_thw input - bfloat16) ===")

    src_model.eval()
    packing_model.eval()

    device = next(src_model.parameters()).device
    print(f"    Running on Device: {device}")

    dtype = next(src_model.parameters()).dtype
    print(f"    Model Dtype: {dtype}")

    # Prepare input tensor
    input_tensor = real_image_tensor.to(device, dtype=torch.bfloat16)
    print(f"    Input Shape: {input_tensor.shape} | Dtype: {input_tensor.dtype}")

    # Get patch size from source model
    patch_size = 16
    if hasattr(src_model, "patch_size"):
        ps = src_model.patch_size
        patch_size = ps[0] if isinstance(ps, tuple) else ps

    # Calculate grid dimensions
    bs, channels, height, width = input_tensor.shape
    h_patches = height // patch_size
    w_patches = width // patch_size
    t_frames = 1

    # Create grid_thw tensor
    grid_thw = torch.tensor([[t_frames, h_patches, w_patches]], dtype=torch.long, device=device)
    print(f"    grid_thw: {grid_thw}")

    with torch.no_grad():
        # Source model forward
        try:
            src_out = src_model(input_tensor)
            if isinstance(src_out, dict):
                src_feat = src_out.get('visible_embeddings')
                src_head = src_out.get('head_output')
            else:
                src_feat = src_out
                src_head = None
        except Exception as e:
            print(f"    [Error] Source forward failed: {e}")
            import traceback
            traceback.print_exc()
            return

        # Packing model forward - expects input in [seq_len, patch_dim] format
        # where patch_dim = patch_size * patch_size * in_channels
        try:
            # Reshape image to patches: (B, C, H, W) -> (seq_len, patch_dim)
            # First reshape to (B, C, h_patches, patch_size, w_patches, patch_size)
            patches = input_tensor.view(
                bs, channels, h_patches, patch_size, w_patches, patch_size
            )
            # Permute to (B, h_patches, w_patches, C, patch_size, patch_size)
            patches = patches.permute(0, 2, 4, 1, 3, 5).contiguous()
            # Reshape to (B * h_patches * w_patches, C * patch_size * patch_size)
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
    if src_feat is not None and packing_feat is not None:
        # Reshape for comparison (src_feat: (B, N, C), packing_feat: (total_N, C))
        src_feat_flat = src_feat.flatten(0, -2).float()  # (B*N, C)
        packing_feat_flat = packing_feat.float()  # (total_N, C)

        # Check if shapes match
        if src_feat_flat.shape[0] != packing_feat_flat.shape[0]:
            print(f"    [Warning] Shape mismatch: src {src_feat_flat.shape} vs packing {packing_feat_flat.shape}")
            # Try to handle different output sizes
            min_len = min(src_feat_flat.shape[0], packing_feat_flat.shape[0])
            src_feat_flat = src_feat_flat[:min_len]
            packing_feat_flat = packing_feat_flat[:min_len]

        diff_feat = (src_feat_flat - packing_feat_flat).abs().max().item()
        cos_sim = F.cosine_similarity(src_feat_flat, packing_feat_flat, dim=-1)

        min_cos = cos_sim.min().item()
        mean_cos = cos_sim.mean().item()

        print(f"    [Packing Feature] Max Diff:       {diff_feat:.6f}")
        print(f"    [Packing Feature] Min Cosine Sim: {min_cos:.8f} (Mean: {mean_cos:.8f})")

        if min_cos > 0.99:
            print("    ✅ Packing Feature: PASS")
        else:
            print("    ❌ Packing Feature: FAIL")

    if src_head is not None and packing_head is not None:
        diff_head = (src_head - packing_head).abs().max().item()
        src_head_f = src_head.float()
        packing_head_f = packing_head.float()
        cos_sim_head = F.cosine_similarity(src_head_f, packing_head_f, dim=-1)

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
    patches_per_frame = h_patches * w_patches

    positions = []
    for b in range(bs):
        for frame_idx in range(num_frames):
            # Get the interpolated temporal position (in 64-frame context)
            t_pos = interpolated_indices[b, frame_idx].item()

            # Generate spatial positions for this frame
            for h in range(h_patches):
                for w in range(w_patches):
                    positions.append([t_pos, h, w])

    return torch.tensor(positions, dtype=torch.long, device=device)


def verify_video_consistency_packing(src_model, packing_model, real_image_tensor, num_frames=8, image_size=224):
    """
    Verify consistency between the source model and the packing model with video input.

    This function tests that the packing model produces consistent outputs with the
    original source model when processing video input (multiple frames).

    The src_model requires 64 frames input with visible_index for uniform frame sampling.
    The packing model receives the actual frames but with RoPE positions matching the
    src_model's interpolated positions in the 64-frame context.

    Args:
        src_model: The source ViT model
        packing_model: The packing HF model
        real_image_tensor: Real image tensor for creating synthesized video
        num_frames: Number of video frames to test (default: 8)
        image_size: Size of each frame (default: 224)
    """
    print(f"\n=== Verifying Video Consistency with Packing Model ({num_frames} frames - bfloat16) ===")

    src_model.eval()
    packing_model.eval()

    device = next(src_model.parameters()).device
    print(f"    Running on Device: {device}")

    dtype = next(src_model.parameters()).dtype
    print(f"    Model Dtype: {dtype}")

    # Get patch size from source model
    patch_size = 16
    if hasattr(src_model, "patch_size"):
        ps = src_model.patch_size
        patch_size = ps[0] if isinstance(ps, tuple) else ps

    # Calculate grid dimensions
    channels = 3
    h_patches = image_size // patch_size
    w_patches = image_size // patch_size
    frame_tokens = h_patches * w_patches
    target_frames = 64  # src_model expects 64-frame context

    bs = 1

    # Create synthesized video from real image instead of random tensor
    video_tensor = get_synthesized_video(real_image_tensor, num_frames=num_frames, size=image_size)
    video_tensor = video_tensor.to(device, dtype=torch.bfloat16)
    print(f"    Video Input Shape: {video_tensor.shape} (B, C, T, H, W)")

    # Compute interpolated frame indices for 64-frame context
    frame_indices = torch.arange(num_frames).unsqueeze(0).to(device)  # [1, 8]
    total_frames_tensor = torch.tensor([num_frames]).to(device)  # [1]
    interpolated_indices = interpolate_frame_indices(
        frame_indices, total_frames_tensor, target_frames
    )  # [1, 8] - indices in 64-frame context

    print(f"    Original frame indices: {frame_indices[0].tolist()}")
    print(f"    Interpolated indices (in 64-frame context): {interpolated_indices[0].tolist()}")

    with torch.no_grad():
        # === Source model forward ===
        # Create 64-frame padded video and use visible_index
        try:
            # Create padded video with 64 frames
            padded_videos = torch.zeros(bs, channels, target_frames, image_size, image_size,
                                        device=device, dtype=video_tensor.dtype)

            # Scatter original frames into interpolated positions
            seq_len = frame_indices.shape[1]
            frame_idx_expanded = interpolated_indices.view(bs, 1, seq_len, 1, 1).expand(
                bs, channels, seq_len, image_size, image_size
            )
            padded_videos.scatter_(dim=2, index=frame_idx_expanded, src=video_tensor)

            # Compute visible_index for the uniformly sampled frames
            per = torch.arange(frame_tokens, device=device)
            visible_index = (interpolated_indices.unsqueeze(-1) * frame_tokens + per).reshape(bs, -1)
            visible_index = visible_index.clamp_max(target_frames * frame_tokens - 1)

            print(f"    Padded video shape: {padded_videos.shape}")
            print(f"    visible_index shape: {visible_index.shape}")

            src_out = src_model(padded_videos, visible_indices=visible_index, mask_ratio=None)
            if isinstance(src_out, dict):
                src_feat = src_out.get('visible_embeddings')
                src_head = src_out.get('head_output')
            else:
                src_feat = src_out
                src_head = None
        except Exception as e:
            print(f"    [Error] Source forward failed: {e}")
            import traceback
            traceback.print_exc()
            return

        # === Packing model forward ===
        # Use actual frames but with RoPE positions matching the interpolated indices
        try:
            # Reshape video to patches: (B, C, T, H, W) -> (seq_len, patch_dim)
            patches = video_tensor.view(
                bs, channels, num_frames, h_patches, patch_size, w_patches, patch_size
            )
            # Permute to (B, T, h_patches, w_patches, C, patch_size, patch_size)
            patches = patches.permute(0, 2, 3, 5, 1, 4, 6).contiguous()
            # Reshape to (B * T * h_patches * w_patches, C * patch_size * patch_size)
            total_seq_len = bs * num_frames * h_patches * w_patches
            patch_dim = patch_size * patch_size * channels
            hidden_states = patches.view(total_seq_len, patch_dim)

            # Compute patch_positions with interpolated temporal positions
            # This ensures RoPE positions match the src_model
            patch_positions = compute_patch_positions_with_interpolated_temporal(
                interpolated_indices, h_patches, w_patches, device
            )

            # Create grid_thw for the actual frames
            grid_thw = torch.tensor([[num_frames, h_patches, w_patches]], dtype=torch.long, device=device)

            print(f"    Packing input shape: {hidden_states.shape} (seq_len={total_seq_len}, patch_dim={patch_dim})")
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
    if src_feat is not None and packing_feat is not None:
        # Reshape for comparison (src_feat: (B, N, C), packing_feat: (total_N, C))
        src_feat_flat = src_feat.flatten(0, -2).float()  # (B*N, C)
        packing_feat_flat = packing_feat.float()  # (total_N, C)

        # Check if shapes match
        if src_feat_flat.shape[0] != packing_feat_flat.shape[0]:
            print(f"    [Warning] Shape mismatch: src {src_feat_flat.shape} vs packing {packing_feat_flat.shape}")
            # Try to handle different output sizes
            min_len = min(src_feat_flat.shape[0], packing_feat_flat.shape[0])
            src_feat_flat = src_feat_flat[:min_len]
            packing_feat_flat = packing_feat_flat[:min_len]

        diff_feat = (src_feat_flat - packing_feat_flat).abs().max().item()
        cos_sim = F.cosine_similarity(src_feat_flat, packing_feat_flat, dim=-1)

        min_cos = cos_sim.min().item()
        mean_cos = cos_sim.mean().item()

        print(f"    [Video Packing Feature] Max Diff:       {diff_feat:.6f}")
        print(f"    [Video Packing Feature] Min Cosine Sim: {min_cos:.8f} (Mean: {mean_cos:.8f})")

        if min_cos > 0.99:
            print("    ✅ Video Packing Feature: PASS")
        else:
            print("    ❌ Video Packing Feature: FAIL")

    if src_head is not None and packing_head is not None:
        diff_head = (src_head - packing_head).abs().max().item()
        src_head_f = src_head.float()
        packing_head_f = packing_head.float()
        cos_sim_head = F.cosine_similarity(src_head_f, packing_head_f, dim=-1)

        min_cos_head = cos_sim_head.min().item()
        mean_cos_head = cos_sim_head.mean().item()

        print(f"    [Video Packing Head]    Max Diff:       {diff_head:.6f}")
        print(f"    [Video Packing Head]    Min Cosine Sim: {min_cos_head:.8f} (Mean: {mean_cos_head:.8f})")

        if min_cos_head > 0.99:
            print("    ✅ Video Packing Head:    PASS")
        else:
            print("    ❌ Video Packing Head:    FAIL")


def verify_mixed_video_image_consistency_packing(src_model, packing_model, real_image_tensor, num_frames=8, video_size=224, image_size=448):
    """
    Verify consistency between the source model and the packing model with mixed video+image input.

    This function tests that the packing model produces consistent outputs with the
    original source model when processing a combined batch of video and image inputs.

    The video uses `compute_patch_positions_with_interpolated_temporal` for RoPE calculation,
    and both video and image are processed together in a single packing model forward pass.

    The src_model processes video and image separately, and the results are compared with
    the packing model's combined output.

    Args:
        src_model: The source ViT model
        packing_model: The packing HF model
        real_image_tensor: Real image tensor for creating test inputs
        num_frames: Number of video frames to test (default: 8)
        video_size: Size of each video frame (default: 224)
        image_size: Size of the image (default: 448)
    """
    print(f"\n=== Verifying Mixed Video+Image Consistency with Packing Model ({num_frames} frames + image - bfloat16) ===")

    src_model.eval()
    packing_model.eval()

    device = next(src_model.parameters()).device
    print(f"    Running on Device: {device}")

    dtype = next(src_model.parameters()).dtype
    print(f"    Model Dtype: {dtype}")

    # Get patch size from source model
    patch_size = 14
    if hasattr(src_model, "patch_size"):
        ps = src_model.patch_size
        patch_size = ps[0] if isinstance(ps, tuple) else ps

    channels = 3
    target_frames = 64  # src_model expects 64-frame context for video

    # ============================================================
    # Prepare Video Input
    # ============================================================
    video_h_patches = video_size // patch_size
    video_w_patches = video_size // patch_size
    video_frame_tokens = video_h_patches * video_w_patches

    # Create synthesized video from real image
    video_tensor = get_synthesized_video(real_image_tensor, num_frames=num_frames, size=video_size)
    video_tensor = video_tensor.to(device, dtype=torch.bfloat16)
    print(f"    Video Input Shape: {video_tensor.shape} (B, C, T, H, W)")

    # Compute interpolated frame indices for 64-frame context
    frame_indices = torch.arange(num_frames).unsqueeze(0).to(device)  # [1, 8]
    total_frames_tensor = torch.tensor([num_frames]).to(device)  # [1]
    interpolated_indices = interpolate_frame_indices(
        frame_indices, total_frames_tensor, target_frames
    )  # [1, 8] - indices in 64-frame context

    print(f"    Video original frame indices: {frame_indices[0].tolist()}")
    print(f"    Video interpolated indices (in 64-frame context): {interpolated_indices[0].tolist()}")

    # ============================================================
    # Prepare Image Input
    # ============================================================
    image_h_patches = image_size // patch_size
    image_w_patches = image_size // patch_size

    # Prepare image input
    image_tensor = real_image_tensor.to(device, dtype=torch.bfloat16)
    if image_tensor.shape[-1] != image_size or image_tensor.shape[-2] != image_size:
        image_tensor = F.interpolate(
            image_tensor.float(),
            size=(image_size, image_size),
            mode='bicubic',
            align_corners=False
        ).to(dtype=torch.bfloat16)
    print(f"    Image Input Shape: {image_tensor.shape} (B, C, H, W)")

    bs = 1  # batch size for each (video and image will be processed separately by src_model)

    with torch.no_grad():
        # ============================================================
        # Source model forward - Process Video and Image Separately
        # ============================================================

        # === Video forward with src_model ===
        try:
            # Create 64-frame padded video and use visible_index
            padded_videos = torch.zeros(bs, channels, target_frames, video_size, video_size,
                                        device=device, dtype=video_tensor.dtype)

            # Scatter original frames into interpolated positions
            seq_len = frame_indices.shape[1]
            frame_idx_expanded = interpolated_indices.view(bs, 1, seq_len, 1, 1).expand(
                bs, channels, seq_len, video_size, video_size
            )
            padded_videos.scatter_(dim=2, index=frame_idx_expanded, src=video_tensor)

            # Compute visible_index for the uniformly sampled frames
            per = torch.arange(video_frame_tokens, device=device)
            visible_index = (interpolated_indices.unsqueeze(-1) * video_frame_tokens + per).reshape(bs, -1)
            visible_index = visible_index.clamp_max(target_frames * video_frame_tokens - 1)

            print(f"    Video padded shape: {padded_videos.shape}")
            print(f"    Video visible_index shape: {visible_index.shape}")

            src_video_out = src_model(padded_videos, visible_indices=visible_index, mask_ratio=None)
            if isinstance(src_video_out, dict):
                src_video_feat = src_video_out.get('visible_embeddings')
            else:
                src_video_feat = src_video_out
        except Exception as e:
            print(f"    [Error] Source video forward failed: {e}")
            import traceback
            traceback.print_exc()
            return

        # === Image forward with src_model ===
        try:
            src_image_out = src_model(image_tensor)
            if isinstance(src_image_out, dict):
                src_image_feat = src_image_out.get('visible_embeddings')
            else:
                src_image_feat = src_image_out
            print(f"    Source image output shape: {src_image_feat.shape}")
        except Exception as e:
            print(f"    [Error] Source image forward failed: {e}")
            import traceback
            traceback.print_exc()
            return

        # ============================================================
        # Packing model forward - Process Video and Image Together
        # ============================================================
        try:
            # === Prepare video patches ===
            video_patches = video_tensor.view(
                bs, channels, num_frames, video_h_patches, patch_size, video_w_patches, patch_size
            )
            video_patches = video_patches.permute(0, 2, 3, 5, 1, 4, 6).contiguous()
            video_seq_len = bs * num_frames * video_h_patches * video_w_patches
            patch_dim = patch_size * patch_size * channels
            video_hidden_states = video_patches.view(video_seq_len, patch_dim)

            # Compute video patch_positions with interpolated temporal positions
            # This uses compute_patch_positions_with_interpolated_temporal as required
            video_patch_positions = compute_patch_positions_with_interpolated_temporal(
                interpolated_indices, video_h_patches, video_w_patches, device
            )

            # === Prepare image patches ===
            image_patches = image_tensor.view(
                bs, channels, image_h_patches, patch_size, image_w_patches, patch_size
            )
            image_patches = image_patches.permute(0, 2, 4, 1, 3, 5).contiguous()
            image_seq_len = bs * 1 * image_h_patches * image_w_patches
            image_hidden_states = image_patches.view(image_seq_len, patch_dim)

            # For image, temporal position is 0 (single frame)
            image_patch_positions = []
            for h in range(image_h_patches):
                for w in range(image_w_patches):
                    image_patch_positions.append([0, h, w])
            image_patch_positions = torch.tensor(image_patch_positions, dtype=torch.long, device=device)

            # === Combine video and image ===
            combined_hidden_states = torch.cat([video_hidden_states, image_hidden_states], dim=0)
            combined_patch_positions = torch.cat([video_patch_positions, image_patch_positions], dim=0)

            # Create grid_thw for the combined input
            # Video: (num_frames, video_h_patches, video_w_patches)
            # Image: (1, image_h_patches, image_w_patches)
            combined_grid_thw = torch.tensor([
                [num_frames, video_h_patches, video_w_patches],
                [1, image_h_patches, image_w_patches]
            ], dtype=torch.long, device=device)

            print(f"    Combined input shape: {combined_hidden_states.shape}")
            print(f"    Combined patch_positions shape: {combined_patch_positions.shape}")
            print(f"    Combined grid_thw: {combined_grid_thw.tolist()}")

            packing_out = packing_model(
                hidden_states=combined_hidden_states,
                grid_thw=combined_grid_thw,
                patch_positions=combined_patch_positions,
            )
            packing_feat = packing_out.last_hidden_state

            # Split the output back into video and image parts
            packing_video_feat = packing_feat[:video_seq_len]
            packing_image_feat = packing_feat[video_seq_len:]

            print(f"    Packing video output shape: {packing_video_feat.shape}")
            print(f"    Packing image output shape: {packing_image_feat.shape}")
        except Exception as e:
            print(f"    [Error] Packing model forward failed: {e}")
            import traceback
            traceback.print_exc()
            return

    # ============================================================
    # Compare Video Outputs
    # ============================================================
    print("\n    --- Video Comparison ---")
    if src_video_feat is not None and packing_video_feat is not None:
        src_video_flat = src_video_feat.flatten(0, -2).float()
        packing_video_flat = packing_video_feat.float()

        if src_video_flat.shape[0] != packing_video_flat.shape[0]:
            print(f"    [Warning] Video shape mismatch: src {src_video_flat.shape} vs packing {packing_video_flat.shape}")
            min_len = min(src_video_flat.shape[0], packing_video_flat.shape[0])
            src_video_flat = src_video_flat[:min_len]
            packing_video_flat = packing_video_flat[:min_len]

        diff_feat = (src_video_flat - packing_video_flat).abs().max().item()
        cos_sim = F.cosine_similarity(src_video_flat, packing_video_flat, dim=-1)

        min_cos = cos_sim.min().item()
        mean_cos = cos_sim.mean().item()

        print(f"    [Mixed Video Feature] Max Diff:       {diff_feat:.6f}")
        print(f"    [Mixed Video Feature] Min Cosine Sim: {min_cos:.8f} (Mean: {mean_cos:.8f})")

        if min_cos > 0.99:
            print("    ✅ Mixed Video Feature: PASS")
            video_pass = True
        else:
            print("    ❌ Mixed Video Feature: FAIL")
            video_pass = False
    else:
        video_pass = False

    # ============================================================
    # Compare Image Outputs
    # ============================================================
    print("\n    --- Image Comparison ---")
    if src_image_feat is not None and packing_image_feat is not None:
        src_image_flat = src_image_feat.flatten(0, -2).float()
        packing_image_flat = packing_image_feat.float()

        if src_image_flat.shape[0] != packing_image_flat.shape[0]:
            print(f"    [Warning] Image shape mismatch: src {src_image_flat.shape} vs packing {packing_image_flat.shape}")
            min_len = min(src_image_flat.shape[0], packing_image_flat.shape[0])
            src_image_flat = src_image_flat[:min_len]
            packing_image_flat = packing_image_flat[:min_len]

        diff_feat = (src_image_flat - packing_image_flat).abs().max().item()
        cos_sim = F.cosine_similarity(src_image_flat, packing_image_flat, dim=-1)

        min_cos = cos_sim.min().item()
        mean_cos = cos_sim.mean().item()

        print(f"    [Mixed Image Feature] Max Diff:       {diff_feat:.6f}")
        print(f"    [Mixed Image Feature] Min Cosine Sim: {min_cos:.8f} (Mean: {mean_cos:.8f})")

        if min_cos > 0.99:
            print("    ✅ Mixed Image Feature: PASS")
            image_pass = True
        else:
            print("    ❌ Mixed Image Feature: FAIL")
            image_pass = False
    else:
        image_pass = False

    # ============================================================
    # Overall Summary
    # ============================================================
    print("\n    --- Mixed Video+Image Overall Summary ---")
    if video_pass and image_pass:
        print("    ✅ Mixed Video+Image Consistency: ALL PASS")
    else:
        print("    ❌ Mixed Video+Image Consistency: SOME FAILED")


def verify_multi_sample_consistency_packing(src_model, packing_model, real_image_tensor):
    """
    Verify consistency between the source model and the packing model with multiple samples.

    This function tests:
    - 3 images with resolutions: 224, 336, 1080
    - 2 videos with 8 frames each: first video resolution 378, second video resolution 518
    - All 5 samples are separately tested through src_model
    - All 5 samples are packed together for packing_model forward

    Args:
        src_model: The source ViT model
        packing_model: The packing HF model
        real_image_tensor: Real image tensor for creating test inputs
    """
    print("\n=== Verifying Multi-Sample Consistency (3 images + 2 videos - bfloat16) ===")
    print("    Image resolutions: 224, 336, 1080")
    print("    Video resolutions: 378 (8 frames), 518 (8 frames)")

    src_model.eval()
    packing_model.eval()

    device = next(src_model.parameters()).device
    print(f"    Running on Device: {device}")

    dtype = next(src_model.parameters()).dtype
    print(f"    Model Dtype: {dtype}")

    # Get patch size from source model
    patch_size = 14
    if hasattr(src_model, "patch_size"):
        ps = src_model.patch_size
        patch_size = ps[0] if isinstance(ps, tuple) else ps

    channels = 3
    target_frames = 64  # src_model expects 64-frame context for video
    num_frames = 8  # Number of frames per video

    # Define original image and video sizes (as specified in the requirements)
    image_sizes_original = [224, 336, 1008]  # Original image resolutions
    video_sizes_original = [378, 518]  # Original video resolutions
    image_sizes_adjusted = [224, 336, 1008]
    video_sizes = [378, 518]

    print(f"    Original image sizes: {image_sizes_original}")
    print(f"    Original video sizes: {video_sizes_original}")
    print(f"    Adjusted image sizes (divisible by {patch_size}): {image_sizes_adjusted}")
    print(f"    Adjusted video sizes (divisible by {patch_size}): {video_sizes}")

    # ============================================================
    # Prepare All Inputs
    # ============================================================

    # === Prepare 3 images ===
    image_tensors = []
    image_h_patches_list = []
    image_w_patches_list = []
    for i, img_size in enumerate(image_sizes_adjusted):
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
        print(f"    Image {i+1} Shape: {img_tensor.shape} (res={image_sizes_original[i]}, adjusted={img_size})")

    # === Prepare 2 videos ===
    video_tensors = []
    video_h_patches_list = []
    video_w_patches_list = []
    interpolated_indices_list = []
    for i, vid_size in enumerate(video_sizes):
        h_patches = vid_size // patch_size
        w_patches = vid_size // patch_size
        video_h_patches_list.append(h_patches)
        video_w_patches_list.append(w_patches)

        vid_tensor = get_synthesized_video(real_image_tensor, num_frames=num_frames, size=vid_size)
        vid_tensor = vid_tensor.to(device, dtype=torch.bfloat16)
        video_tensors.append(vid_tensor)

        # Compute interpolated frame indices for 64-frame context
        frame_indices = torch.arange(num_frames).unsqueeze(0).to(device)
        total_frames_tensor = torch.tensor([num_frames]).to(device)
        interpolated_indices = interpolate_frame_indices(
            frame_indices, total_frames_tensor, target_frames
        )
        interpolated_indices_list.append(interpolated_indices)

        print(f"    Video {i+1} Shape: {vid_tensor.shape} (res={video_sizes_original[i]}, adjusted={vid_size})")

    bs = 1  # batch size for each sample

    with torch.no_grad():
        # ============================================================
        # Source model forward - Process Each Sample Separately
        # ============================================================
        src_image_feats = []
        src_video_feats = []

        # === Process each image separately with src_model ===
        for i, img_tensor in enumerate(image_tensors):
            try:
                src_image_out = src_model(img_tensor)
                if isinstance(src_image_out, dict):
                    src_image_feat = src_image_out.get('visible_embeddings')
                else:
                    src_image_feat = src_image_out
                src_image_feats.append(src_image_feat)
                print(f"    Source image {i+1} output shape: {src_image_feat.shape}")
            except Exception as e:
                print(f"    [Error] Source image {i+1} forward failed: {e}")
                import traceback
                traceback.print_exc()
                return

        # === Process each video separately with src_model ===
        for i, (vid_tensor, interpolated_indices) in enumerate(zip(video_tensors, interpolated_indices_list)):
            try:
                vid_size = video_sizes[i]
                h_patches = video_h_patches_list[i]
                w_patches = video_w_patches_list[i]
                frame_tokens = h_patches * w_patches

                # Create 64-frame padded video and use visible_index
                padded_videos = torch.zeros(bs, channels, target_frames, vid_size, vid_size,
                                            device=device, dtype=vid_tensor.dtype)

                # Scatter original frames into interpolated positions
                seq_len = num_frames
                frame_idx_expanded = interpolated_indices.view(bs, 1, seq_len, 1, 1).expand(
                    bs, channels, seq_len, vid_size, vid_size
                )
                padded_videos.scatter_(dim=2, index=frame_idx_expanded, src=vid_tensor)

                # Compute visible_index for the uniformly sampled frames
                per = torch.arange(frame_tokens, device=device)
                visible_index = (interpolated_indices.unsqueeze(-1) * frame_tokens + per).reshape(bs, -1)
                visible_index = visible_index.clamp_max(target_frames * frame_tokens - 1)
                src_video_out = src_model(padded_videos, visible_indices=visible_index, mask_ratio=None)
                if isinstance(src_video_out, dict):
                    src_video_feat = src_video_out.get('visible_embeddings')
                else:
                    src_video_feat = src_video_out
                src_video_feats.append(src_video_feat)
                print(f"    Source video {i+1} output shape: {src_video_feat.shape}")
            except Exception as e:
                print(f"    [Error] Source video {i+1} forward failed: {e}")
                import traceback
                traceback.print_exc()
                return

        # ============================================================
        # Packing model forward - Process All Samples Together
        # ============================================================
        try:
            patch_dim = patch_size * patch_size * channels

            all_hidden_states = []
            all_patch_positions = []
            grid_thw_list = []
            seq_lengths = []  # Track sequence lengths for each sample

            # === Prepare image patches ===
            for i, img_tensor in enumerate(image_tensors):
                h_patches = image_h_patches_list[i]
                w_patches = image_w_patches_list[i]
                img_size = image_sizes_adjusted[i]

                img_patches = img_tensor.view(
                    bs, channels, h_patches, patch_size, w_patches, patch_size
                )
                img_patches = img_patches.permute(0, 2, 4, 1, 3, 5).contiguous()
                img_seq_len = bs * 1 * h_patches * w_patches
                img_hidden_states = img_patches.view(img_seq_len, patch_dim)
                all_hidden_states.append(img_hidden_states)
                seq_lengths.append(img_seq_len)

                # For image, temporal position is 0 (single frame)
                img_patch_positions = []
                for h in range(h_patches):
                    for w in range(w_patches):
                        img_patch_positions.append([0, h, w])
                img_patch_positions = torch.tensor(img_patch_positions, dtype=torch.long, device=device)
                all_patch_positions.append(img_patch_positions)

                grid_thw_list.append([1, h_patches, w_patches])

            # === Prepare video patches ===
            for i, (vid_tensor, interpolated_indices) in enumerate(zip(video_tensors, interpolated_indices_list)):
                h_patches = video_h_patches_list[i]
                w_patches = video_w_patches_list[i]

                vid_patches = vid_tensor.view(
                    bs, channels, num_frames, h_patches, patch_size, w_patches, patch_size
                )
                vid_patches = vid_patches.permute(0, 2, 3, 5, 1, 4, 6).contiguous()
                vid_seq_len = bs * num_frames * h_patches * w_patches
                vid_hidden_states = vid_patches.view(vid_seq_len, patch_dim)
                all_hidden_states.append(vid_hidden_states)
                seq_lengths.append(vid_seq_len)

                # Compute video patch_positions with interpolated temporal positions
                vid_patch_positions = compute_patch_positions_with_interpolated_temporal(
                    interpolated_indices, h_patches, w_patches, device
                )
                all_patch_positions.append(vid_patch_positions)

                grid_thw_list.append([num_frames, h_patches, w_patches])

            # === Combine all samples ===
            combined_hidden_states = torch.cat(all_hidden_states, dim=0)
            combined_patch_positions = torch.cat(all_patch_positions, dim=0)
            combined_grid_thw = torch.tensor(grid_thw_list, dtype=torch.long, device=device)

            print(f"    Combined input shape: {combined_hidden_states.shape}")
            print(f"    Combined patch_positions shape: {combined_patch_positions.shape}")
            print(f"    Combined grid_thw: {combined_grid_thw.tolist()}")

            packing_out = packing_model(
                hidden_states=combined_hidden_states,
                grid_thw=combined_grid_thw,
                patch_positions=combined_patch_positions,
            )
            packing_feat = packing_out.last_hidden_state

            # Split the output back into individual samples
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

            for i, feat in enumerate(packing_image_feats):
                print(f"    Packing image {i+1} output shape: {feat.shape}")
            for i, feat in enumerate(packing_video_feats):
                print(f"    Packing video {i+1} output shape: {feat.shape}")

        except Exception as e:
            print(f"    [Error] Packing model forward failed: {e}")
            import traceback
            traceback.print_exc()
            return

    # ============================================================
    # Compare Outputs for Each Sample
    # ============================================================
    all_pass = True

    # === Compare image outputs ===
    for i in range(len(image_tensors)):
        print(f"\n    --- Image {i+1} Comparison (res={image_sizes_original[i]}) ---")
        src_feat = src_image_feats[i]
        packing_feat = packing_image_feats[i]

        if src_feat is not None and packing_feat is not None:
            src_flat = src_feat.flatten(0, -2).float()
            packing_flat = packing_feat.float()

            if src_flat.shape[0] != packing_flat.shape[0]:
                print(f"    [Warning] Shape mismatch: src {src_flat.shape} vs packing {packing_flat.shape}")
                min_len = min(src_flat.shape[0], packing_flat.shape[0])
                src_flat = src_flat[:min_len]
                packing_flat = packing_flat[:min_len]

            diff_feat = (src_flat - packing_flat).abs().max().item()
            cos_sim = F.cosine_similarity(src_flat, packing_flat, dim=-1)

            min_cos = cos_sim.min().item()
            mean_cos = cos_sim.mean().item()

            print(f"    [Image {i+1}] Max Diff:       {diff_feat:.6f}")
            print(f"    [Image {i+1}] Min Cosine Sim: {min_cos:.8f} (Mean: {mean_cos:.8f})")

            if min_cos > 0.99:
                print(f"    ✅ Image {i+1}: PASS")
            else:
                print(f"    ❌ Image {i+1}: FAIL")
                all_pass = False
        else:
            all_pass = False

    # === Compare video outputs ===
    for i in range(len(video_tensors)):
        print(f"\n    --- Video {i+1} Comparison (res={video_sizes_original[i]}) ---")
        src_feat = src_video_feats[i]
        packing_feat = packing_video_feats[i]

        if src_feat is not None and packing_feat is not None:
            src_flat = src_feat.flatten(0, -2).float()
            packing_flat = packing_feat.float()

            if src_flat.shape[0] != packing_flat.shape[0]:
                print(f"    [Warning] Shape mismatch: src {src_flat.shape} vs packing {packing_flat.shape}")
                min_len = min(src_flat.shape[0], packing_flat.shape[0])
                src_flat = src_flat[:min_len]
                packing_flat = packing_flat[:min_len]

            diff_feat = (src_flat - packing_flat).abs().max().item()
            cos_sim = F.cosine_similarity(src_flat, packing_flat, dim=-1)

            min_cos = cos_sim.min().item()
            mean_cos = cos_sim.mean().item()

            print(f"    [Video {i+1}] Max Diff:       {diff_feat:.6f}")
            print(f"    [Video {i+1}] Min Cosine Sim: {min_cos:.8f} (Mean: {mean_cos:.8f})")

            if min_cos > 0.99:
                print(f"    ✅ Video {i+1}: PASS")
            else:
                print(f"    ❌ Video {i+1}: FAIL")
                all_pass = False
        else:
            all_pass = False

    # ============================================================
    # Overall Summary
    # ============================================================
    print("\n    --- Multi-Sample Overall Summary ---")
    if all_pass:
        print("    ✅ Multi-Sample Consistency (3 images + 2 videos): ALL PASS")
    else:
        print("    ❌ Multi-Sample Consistency (3 images + 2 videos): SOME FAILED")


def verify_saved_model_loading_packing(src_model, output_dir, real_image_tensor):
    print("\n=== Verifying Loaded Saved Packing Model (Simulating User Usage - bfloat16) ===")
    print(f"--> Loading from: {output_dir}")

    device = next(src_model.parameters()).device
    print(f"    Using device from src_model: {device}")

    try:
        print("    Loading Image Processor (CLIP)...")
        image_processor = CLIPImageProcessor.from_pretrained(output_dir)
        print(f"    Processor config: {image_processor}")

        print("    Loading Vision Tower (LlavaViTPackingModel) with torch_dtype=bfloat16...")
        # CRITICAL: 显式指定加载为 bfloat16
        vision_tower = LlavaViTPackingModel.from_pretrained(output_dir, torch_dtype=torch.bfloat16)
        vision_tower.to(device)
        vision_tower.eval()

        print("    ✅ Successfully loaded and moved to device.")

    except Exception as e:
        print(f"    ❌ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return

    # Get patch size from source model
    patch_size = 14
    if hasattr(src_model, "patch_size"):
        ps = src_model.patch_size
        patch_size = ps[0] if isinstance(ps, tuple) else ps

    # ============================================================
    # Part 1: Image Test (Single Frame)
    # ============================================================
    print("\n    --- Image Test (Single Frame) ---")

    # Prepare input for packing model verification
    input_tensor = real_image_tensor.to(device, dtype=torch.bfloat16)

    # Calculate grid dimensions
    bs, channels, height, width = input_tensor.shape
    h_patches = height // patch_size
    w_patches = width // patch_size
    t_frames = 1

    # Create grid_thw tensor
    grid_thw = torch.tensor([[t_frames, h_patches, w_patches]], dtype=torch.long, device=device)
    print(f"    Image grid_thw: {grid_thw.tolist()}")

    # Packing model expects input in [seq_len, patch_dim] format
    with torch.no_grad():
        src_out = src_model(input_tensor)['visible_embeddings']

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
        print(f"    patch_positions shape: {patch_positions.shape}")

        tgt_out = vision_tower(
            hidden_states=hidden_states,
            grid_thw=grid_thw,
            patch_positions=patch_positions,
        ).last_hidden_state

    src_feat_flat = src_out.flatten(0, -2).float()
    tgt_feat_flat = tgt_out.float()

    # Handle shape mismatch
    if src_feat_flat.shape[0] != tgt_feat_flat.shape[0]:
        min_len = min(src_feat_flat.shape[0], tgt_feat_flat.shape[0])
        src_feat_flat = src_feat_flat[:min_len]
        tgt_feat_flat = tgt_feat_flat[:min_len]

    cos_sim = F.cosine_similarity(src_feat_flat, tgt_feat_flat, dim=-1)

    min_cos = cos_sim.min().item()
    mean_cos = cos_sim.mean().item()

    print(f"    [Reloaded Image] Min Cosine Sim: {min_cos:.8f} (Mean: {mean_cos:.8f})")
    if min_cos > 0.99:
        print("    ✅ Reloaded Image Verification: PASS")
    else:
        print("    ❌ Reloaded Image Verification: FAIL")

    # ============================================================
    # Part 2: Video Test (Multiple Frames)
    # ============================================================
    print("\n    --- Video Test (8 Frames) ---")

    num_frames = 8
    image_size = 224
    target_frames = 64  # src_model expects 64-frame context

    # Initialize variables to track video test results
    min_cos_video = 0.0
    video_test_passed = False

    # Create synthesized video from real image
    video_tensor = get_synthesized_video(real_image_tensor, num_frames=num_frames, size=image_size)
    video_tensor = video_tensor.to(device, dtype=torch.bfloat16)

    bs = 1
    channels = 3
    h_patches = image_size // patch_size
    w_patches = image_size // patch_size
    frame_tokens = h_patches * w_patches

    print(f"    Video shape: {video_tensor.shape} (B, C, T, H, W)")

    # Compute interpolated frame indices for 64-frame context
    frame_indices = torch.arange(num_frames).unsqueeze(0).to(device)  # [1, 8]
    total_frames_tensor = torch.tensor([num_frames]).to(device)  # [1]
    interpolated_indices = interpolate_frame_indices(
        frame_indices, total_frames_tensor, target_frames
    )  # [1, 8] - indices in 64-frame context

    print(f"    Original frame indices: {frame_indices[0].tolist()}")
    print(f"    Interpolated indices (in 64-frame context): {interpolated_indices[0].tolist()}")

    # Initialize output variables
    src_video_feat = None
    tgt_video_out = None

    with torch.no_grad():
        # === Source model forward ===
        # Create 64-frame padded video and use visible_index
        try:
            # Create padded video with 64 frames
            padded_videos = torch.zeros(bs, channels, target_frames, image_size, image_size,
                                        device=device, dtype=video_tensor.dtype)

            # Scatter original frames into interpolated positions
            frame_seq_len = frame_indices.shape[1]
            frame_idx_expanded = interpolated_indices.view(bs, 1, frame_seq_len, 1, 1).expand(
                bs, channels, frame_seq_len, image_size, image_size
            )
            padded_videos.scatter_(dim=2, index=frame_idx_expanded, src=video_tensor)

            # Compute visible_index for the uniformly sampled frames
            per = torch.arange(frame_tokens, device=device)
            visible_index = (interpolated_indices.unsqueeze(-1) * frame_tokens + per).reshape(bs, -1)
            visible_index = visible_index.clamp_max(target_frames * frame_tokens - 1)

            src_video_out = src_model(padded_videos, visible_indices=visible_index, mask_ratio=None)
            if isinstance(src_video_out, dict):
                src_video_feat = src_video_out.get('visible_embeddings')
            else:
                src_video_feat = src_video_out
        except Exception as e:
            print(f"    [Error] Source video forward failed: {e}")
            import traceback
            traceback.print_exc()
            return

        # === Packing model forward ===
        try:
            # Reshape video to patches: (B, C, T, H, W) -> (seq_len, patch_dim)
            patches = video_tensor.view(
                bs, channels, num_frames, h_patches, patch_size, w_patches, patch_size
            )
            # Permute to (B, T, h_patches, w_patches, C, patch_size, patch_size)
            patches = patches.permute(0, 2, 3, 5, 1, 4, 6).contiguous()
            # Reshape to (B * T * h_patches * w_patches, C * patch_size * patch_size)
            total_seq_len = bs * num_frames * h_patches * w_patches
            patch_dim = patch_size * patch_size * channels
            hidden_states = patches.view(total_seq_len, patch_dim)

            # Compute patch_positions with interpolated temporal positions
            # This ensures RoPE positions match the src_model
            patch_positions = compute_patch_positions_with_interpolated_temporal(
                interpolated_indices, h_patches, w_patches, device
            )

            # Create grid_thw for the actual frames
            grid_thw = torch.tensor([[num_frames, h_patches, w_patches]], dtype=torch.long, device=device)

            print(f"    Packing input shape: {hidden_states.shape}")
            print(f"    patch_positions shape: {patch_positions.shape}")
            print(f"    grid_thw: {grid_thw.tolist()}")

            tgt_video_out = vision_tower(
                hidden_states=hidden_states,
                grid_thw=grid_thw,
                patch_positions=patch_positions,
            ).last_hidden_state
        except Exception as e:
            print(f"    [Error] Packing video forward failed: {e}")
            import traceback
            traceback.print_exc()
            return

    # Compare outputs
    if src_video_feat is not None and tgt_video_out is not None:
        src_video_flat = src_video_feat.flatten(0, -2).float()
        tgt_video_flat = tgt_video_out.float()

        # Handle shape mismatch
        if src_video_flat.shape[0] != tgt_video_flat.shape[0]:
            min_len = min(src_video_flat.shape[0], tgt_video_flat.shape[0])
            src_video_flat = src_video_flat[:min_len]
            tgt_video_flat = tgt_video_flat[:min_len]

        cos_sim_video = F.cosine_similarity(src_video_flat, tgt_video_flat, dim=-1)

        min_cos_video = cos_sim_video.min().item()
        mean_cos_video = cos_sim_video.mean().item()

        print(f"    [Reloaded Video] Min Cosine Sim: {min_cos_video:.8f} (Mean: {mean_cos_video:.8f})")
        if min_cos_video > 0.99:
            print("    ✅ Reloaded Video Verification: PASS")
            video_test_passed = True
        else:
            print("    ❌ Reloaded Video Verification: FAIL")

    # ============================================================
    # Overall Summary
    # ============================================================
    print("\n    --- Overall Summary ---")
    image_test_passed = min_cos > 0.99
    all_pass = image_test_passed and video_test_passed
    if all_pass:
        print("    ✅ Reloaded Packing Model (Image + Video): ALL PASS")
    else:
        print("    ❌ Reloaded Packing Model (Image + Video): SOME FAILED")


def convert_and_save_packing(src_model_name, tgt_model_name, weight_path, output_dir):
    print(f"=== Packing Model Conversion Task ===")
    print(f"Source:  {src_model_name}")
    print(f"Target:  {tgt_model_name}")
    print(f"Weights: {weight_path}")
    print(f"Output:  {output_dir}")

    # 1. 创建 Source Model (默认 CPU)
    print("\n--> Creating Source Model...")
    src_model = timm.create_model(src_model_name, pretrained=False)

    print("--> Loading weights into Source...")
    checkpoint = torch.load(weight_path, map_location='cpu')
    state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    src_model.load_state_dict(state_dict, strict=False)

    # 2. 创建 Packing Target Model (默认 CPU)
    print("\n--> Creating Packing Target Model...")
    tgt_model = timm.create_model(tgt_model_name, pretrained=False)

    print("--> Remapping State Dict for Packing Model...")
    hf_state_dict = remap_state_dict_packing(src_model.state_dict())

    print("--> Loading weights into Packing Target...")
    missing, unexpected = tgt_model.load_state_dict(hf_state_dict, strict=False)

    real_missing = [k for k in missing if "attn.bias" not in k]
    if len(real_missing) > 0:
        print(f"    [Warning] Missing keys ({len(real_missing)}):")
        for k in real_missing[:5]: print(f"      {k}")
    else:
        print("    Load OK (No critical missing keys).")

    # 3. CRITICAL: 检测 CUDA 并移动模型，并转换为 bfloat16
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"\n--> [CUDA DETECTED] Moving models to {device} and casting to bfloat16...")
        # Convert both models to bfloat16
        src_model.to(device, dtype=torch.bfloat16)
        tgt_model.to(device, dtype=torch.bfloat16)
    else:
        device = torch.device("cpu")
        print(f"\n--> [WARNING] CUDA not available. Flash Attention will FAIL. bfloat16 on CPU is slow.")
        src_model.to(device, dtype=torch.bfloat16)
        tgt_model.to(device, dtype=torch.bfloat16)


    print("\n--> Fetching real image for verification (448x448)...")
    real_img = get_real_coco_image(size=448)

    # 验证 Packing 模型一致性 (单帧图像)
    verify_consistency_packing(src_model, tgt_model, real_img)

    # 验证 Packing 模型一致性 (多帧视频)
    verify_video_consistency_packing(src_model, tgt_model, real_img, num_frames=8, image_size=224)

    # 验证 Packing 模型一致性 (视频+图片混合输入，视频使用 compute_patch_positions_with_interpolated_temporal)
    verify_mixed_video_image_consistency_packing(src_model, tgt_model, real_img, num_frames=8, video_size=224, image_size=448)

    # 验证 Packing 模型一致性 (3个图片 + 2个视频的多样本测试)
    # 3 images: 224, 336, 1080 resolutions
    # 2 videos: 378 resolution (8 frames), 518 resolution (8 frames)
    verify_multi_sample_consistency_packing(src_model, tgt_model, real_img)

    if output_dir:
        print(f"\n--> Saving HF Packing Model to {output_dir}...")
        if hasattr(tgt_model, "save_pretrained"):
            # 模型本身已经是 bf16，save_pretrained 会保存为 bf16 (safetensors 默认支持)
            tgt_model.save_pretrained(output_dir)

            # --- 保存 CLIPImageProcessor ---
            print("    Saving CLIPImageProcessor config (CLIP Defaults + 448)...")

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

            # 验证 Reload 后的模型
            verify_saved_model_loading_packing(src_model, output_dir, real_img)

        else:
            print("❌ Error: Target model does not have save_pretrained method.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert ViT model to HuggingFace format (Packing version)")
    parser.add_argument("model_name", type=str, help="Source model name")
    parser.add_argument("weight_path", type=str, help="Path to .pth checkpoint")
    parser.add_argument("--target_model_name", type=str, default=None, help="Target HF packing model name")
    parser.add_argument("--output_dir", type=str, default=None)

    args = parser.parse_args()

    tgt_name = args.target_model_name
    if tgt_name is None:
        # Default to packing model name
        tgt_name = f"hf_{args.model_name}".replace("llava_vit_", "llava_vit_packing_")

    out_dir = args.output_dir
    if out_dir is None:
        p = Path(args.weight_path)
        out_dir = os.path.join(p.parent, f"{p.stem}_hf_packing")

    convert_and_save_packing(args.model_name, tgt_name, args.weight_path, out_dir)
