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
    from model_factory.vit_preview_v0_packing_hf import LlavaViTPackingModel
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
    """
    print("[Remap Packing] Starting state dict remapping for packing model...")
    new_dict = {}

    for k, v in src_state_dict.items():
        new_k = k
        if k.startswith("conv1."):
            # conv1 -> embeddings.proj (2D conv -> 3D conv)
            new_k = k.replace("conv1.", "embeddings.proj.")
            if k == "conv1.weight":
                # Source 2D conv: (out_channels, in_channels, H, W)
                # Target 3D conv: (out_channels, in_channels, T, H, W) where T=1
                # Use unsqueeze(2) to add temporal dimension at position 2
                v = v.unsqueeze(2)
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
    _, _, height, width = input_tensor.shape
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

        # Packing model forward - need to prepare pixel values in the right format
        try:
            # The packing model expects pixel values in (N, C, T_patch, H_patch, W_patch) format
            # where N is the total number of patches
            bs = input_tensor.shape[0]
            total_patches = t_frames * h_patches * w_patches * bs

            # Reshape input to patches for the packing model
            # input_tensor: (B, C, H, W) -> (B, C, 1, H, W) -> patches
            pixel_values_5d = input_tensor.unsqueeze(2)  # (B, C, 1, H, W)

            # Reshape to (B, C, T, H_patches, patch_size, W_patches, patch_size)
            pixel_values_patches = pixel_values_5d.view(
                bs, 3, t_frames, h_patches, patch_size, w_patches, patch_size
            )
            # Permute to (B, T, H_patches, W_patches, C, T_patch, H_patch_size, W_patch_size)
            pixel_values_patches = pixel_values_patches.permute(0, 2, 3, 5, 1, 2, 4, 6).contiguous()
            # Reshape to (B * T * H * W, C, 1, patch_size, patch_size)
            pixel_values_packed = pixel_values_patches.view(
                total_patches, 3, 1, patch_size, patch_size
            )

            packing_out = packing_model(pixel_values=pixel_values_packed, grid_thw=grid_thw)
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

    # Prepare input for packing model verification
    input_tensor = real_image_tensor.to(device, dtype=torch.bfloat16)

    # Get patch size from source model
    patch_size = 16
    if hasattr(src_model, "patch_size"):
        ps = src_model.patch_size
        patch_size = ps[0] if isinstance(ps, tuple) else ps

    # Calculate grid dimensions
    _, _, height, width = input_tensor.shape
    h_patches = height // patch_size
    w_patches = width // patch_size
    t_frames = 1
    bs = input_tensor.shape[0]
    total_patches = t_frames * h_patches * w_patches * bs

    # Create grid_thw tensor
    grid_thw = torch.tensor([[t_frames, h_patches, w_patches]], dtype=torch.long, device=device)

    # Prepare pixel values for packing model
    pixel_values_5d = input_tensor.unsqueeze(2)  # (B, C, 1, H, W)
    pixel_values_patches = pixel_values_5d.view(
        bs, 3, t_frames, h_patches, patch_size, w_patches, patch_size
    )
    pixel_values_patches = pixel_values_patches.permute(0, 2, 3, 5, 1, 2, 4, 6).contiguous()
    pixel_values_packed = pixel_values_patches.view(
        total_patches, 3, 1, patch_size, patch_size
    )

    with torch.no_grad():
        src_out = src_model(input_tensor)['visible_embeddings']
        tgt_out = vision_tower(pixel_values=pixel_values_packed, grid_thw=grid_thw).last_hidden_state

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

    print(f"    [Reloaded Packing Model] Min Cosine Sim: {min_cos:.8f} (Mean: {mean_cos:.8f})")
    if min_cos > 0.99:
        print("    ✅ Reloaded Packing Model Verification: PASS")
    else:
        print("    ❌ Reloaded Packing Model Verification: FAIL")


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

    # 验证 Packing 模型一致性
    verify_consistency_packing(src_model, tgt_model, real_img)

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
