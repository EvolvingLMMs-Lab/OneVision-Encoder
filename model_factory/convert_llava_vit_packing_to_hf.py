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
    patch_size = 16
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
        else:
            print("    ❌ Reloaded Video Verification: FAIL")
    
    # ============================================================
    # Overall Summary
    # ============================================================
    print("\n    --- Overall Summary ---")
    all_pass = min_cos > 0.99 and (min_cos_video if 'min_cos_video' in dir() else 0) > 0.99
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
