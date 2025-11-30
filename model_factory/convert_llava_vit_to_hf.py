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
    from model_factory import vit_preview_v0_hf  # 你的 HF 模型定义
    # 为了模拟你的 MLCDVisionModel，我们将刚刚定义的 HF 类赋值给它
    from model_factory.vit_preview_v0_hf import LlavaViTModel as MLCDVisionModel
    from model_factory.vit_preview_v0_hf import (
        LlavaViTAttention,
        LlavaViTFlashAttention2,
        LLAVA_VIT_ATTENTION_CLASSES,
    )
    from model_factory import vit_preview_v0     # 你的 Source 模型定义
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


def interpolate_frame_indices(frame_indices: torch.Tensor, total_frames: torch.Tensor, target_frames: int = 64) -> torch.Tensor:
    bs, seq_len = frame_indices.shape
    total_frames_float = total_frames.float().view(bs, 1)
    frame_indices_float = frame_indices.float()
    total_frames_safe = torch.clamp(total_frames_float - 1, min=1.0)
    interpolated_indices = (frame_indices_float / total_frames_safe) * (target_frames - 1)
    interpolated_indices = torch.round(interpolated_indices).long()
    interpolated_indices = torch.clamp(interpolated_indices, 0, target_frames - 1)
    return interpolated_indices


def remap_state_dict(src_state_dict):
    print("[Remap] Starting state dict remapping...")
    new_dict = {}
    qkv_cache = {}

    for k, v in src_state_dict.items():
        new_k = k
        if k.startswith("conv1."): new_k = k.replace("conv1.", "embeddings.patch_embedding.")
        elif k.startswith("ln_pre."): new_k = k.replace("ln_pre.", "layernorm_pre.")
        elif k.startswith("ln_post."): new_k = k.replace("ln_post.", "layernorm_post.")
        elif k.startswith("transformer.layers."):
            new_k = k.replace("transformer.layers.", "encoder.layers.")
            if ".ln_1." in new_k: new_k = new_k.replace(".ln_1.", ".layer_norm1.")
            if ".ln_2." in new_k: new_k = new_k.replace(".ln_2.", ".layer_norm2.")
            if ".mlp.0." in new_k: new_k = new_k.replace(".mlp.0.", ".mlp.fc1.")
            if ".mlp.2." in new_k: new_k = new_k.replace(".mlp.2.", ".mlp.fc2.")
            if ".attn." in new_k: new_k = new_k.replace(".attn.", ".self_attn.")
            if ".proj." in new_k and "in_proj" not in new_k: new_k = new_k.replace(".proj.", ".out_proj.")
            if "in_proj" in new_k:
                prefix = new_k.rsplit(".in_proj", 1)[0]
                param_type = new_k.split(".")[-1]
                if prefix not in qkv_cache: qkv_cache[prefix] = {}
                qkv_cache[prefix][param_type] = v
                continue
        new_dict[new_k] = v

    print(f"[Remap] Splitting QKV for {len(qkv_cache)} layers...")
    for prefix, params in qkv_cache.items():
        for p_type, tensor in params.items():
            dim = tensor.shape[0] // 3
            q, k, v_part = torch.split(tensor, dim, dim=0)
            new_dict[f"{prefix}.q_proj.{p_type}"] = q
            new_dict[f"{prefix}.k_proj.{p_type}"] = k
            new_dict[f"{prefix}.v_proj.{p_type}"] = v_part

    return new_dict


def verify_consistency(src_model, tgt_model, real_image_tensor):
    print(f"\n=== Verifying Consistency (Real Image Input - bfloat16) ===")

    # 确保模型处于 eval 模式
    src_model.eval()
    tgt_model.eval()

    # 获取当前设备 (src_model 应该已经被移动到了 CUDA)
    device = next(src_model.parameters()).device
    print(f"    Running on Device: {device}")

    # 检查模型精度
    dtype = next(src_model.parameters()).dtype
    print(f"    Model Dtype: {dtype}")

    # 将输入移动到 GPU 并转换为 bfloat16
    input_tensor = real_image_tensor.to(device, dtype=torch.bfloat16)
    print(f"    Input Shape: {input_tensor.shape} | Dtype: {input_tensor.dtype}")

    with torch.no_grad():
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

        try:
            tgt_out = tgt_model(pixel_values=input_tensor)
            tgt_feat = tgt_out.last_hidden_state
            tgt_head = tgt_out.pooler_output
        except Exception as e:
            print(f"    [Error] Target forward failed: {e}")
            import traceback
            traceback.print_exc()
            return

    # 注意：比较时建议转回 float32，避免 bf16 精度问题导致数值差异看起来过大
    if src_feat is not None and tgt_feat is not None:
        diff_feat = (src_feat - tgt_feat).abs().max().item()
        src_feat_flat = src_feat.flatten(0, -2).float()
        tgt_feat_flat = tgt_feat.flatten(0, -2).float()
        cos_sim = F.cosine_similarity(src_feat_flat, tgt_feat_flat, dim=-1)

        min_cos = cos_sim.min().item()
        mean_cos = cos_sim.mean().item()  # Added Mean

        print(f"    [Image Feature] Max Diff:       {diff_feat:.6f}")
        print(f"    [Image Feature] Min Cosine Sim: {min_cos:.8f} (Mean: {mean_cos:.8f})")

        if min_cos > 0.99:
            print("    ✅ Image Feature: PASS")
        else:
            print("    ❌ Image Feature: FAIL")

    if src_head is not None and tgt_head is not None:
        diff_head = (src_head - tgt_head).abs().max().item()
        src_head_f = src_head.float()
        tgt_head_f = tgt_head.float()
        cos_sim_head = F.cosine_similarity(src_head_f, tgt_head_f, dim=-1)

        min_cos_head = cos_sim_head.min().item()
        mean_cos_head = cos_sim_head.mean().item() # Added Mean

        print(f"    [Image Head]    Max Diff:       {diff_head:.6f}")
        print(f"    [Image Head]    Min Cosine Sim: {min_cos_head:.8f} (Mean: {mean_cos_head:.8f})")

        if min_cos_head > 0.99:
            print("    ✅ Image Head:    PASS")
        else:
            print("    ❌ Image Head:    FAIL")


def verify_consistency_video(src_model, tgt_model, real_image_tensor_high_res):
    print("\n=== Verifying Consistency (Real Video Sampling Input - bfloat16) ===")
    src_model.eval()
    tgt_model.eval()

    device = next(src_model.parameters()).device
    print(f"    Running on Device: {device}")

    # Downsample image to 224 for video verification (Operation on Float32 usually safer for interpolation)
    print("    [Video Mode] Resizing input from 448 to 224 for video consistency check...")

    # Keep on Float32 for interpolation to avoid artifacts, then cast
    real_image_tensor_high_res = real_image_tensor_high_res.to(device, dtype=torch.float32)
    real_image_tensor = F.interpolate(real_image_tensor_high_res, size=(224, 224), mode='bicubic', align_corners=False)

    bs = 1
    original_frames = 8
    C, H, W = 3, 224, 224

    # CRITICAL: Convert video input to bfloat16
    videos = real_image_tensor.unsqueeze(2).repeat(bs, 1, original_frames, 1, 1).to(device, dtype=torch.bfloat16)

    frame_indices = torch.arange(original_frames).unsqueeze(0).repeat(bs, 1).to(device)
    total_frames_tensor = torch.tensor([original_frames]*bs).to(device)

    patch_size = 16
    if hasattr(src_model, "patch_size"):
        ps = src_model.patch_size
        patch_size = ps[0] if isinstance(ps, tuple) else ps

    grid_h, grid_w = H // patch_size, W // patch_size
    frame_tokens = grid_h * grid_w
    target_frames = 64

    print(f"    Video Shape: {videos.shape}, Dtype: {videos.dtype}")

    with torch.no_grad():
        interpolated_indices = interpolate_frame_indices(frame_indices, total_frames_tensor.view(-1), target_frames)
        # padded_videos inherits dtype from videos (bf16)
        padded_videos = torch.zeros(bs, C, target_frames, H, W, device=device, dtype=videos.dtype)

        seq_len = frame_indices.shape[1]
        frame_idx_expanded = interpolated_indices.view(bs, 1, seq_len, 1, 1).expand(bs, C, seq_len, H, W)
        padded_videos.scatter_(dim=2, index=frame_idx_expanded, src=videos)
        per = torch.arange(frame_tokens, device=device)
        visible_index = (interpolated_indices.unsqueeze(-1) * frame_tokens + per).reshape(bs, -1)
        visible_index = visible_index.clamp_max(target_frames * frame_tokens - 1)

        try:
            src_out_dict = src_model(padded_videos, visible_indices=visible_index, mask_ratio=None)
            src_feat = src_out_dict["visible_embeddings"]
            src_head = src_out_dict.get("head_output", None)
        except Exception as e:
            print(f"    [Error] Source forward failed: {e}")
            return

        try:
            tgt_out = tgt_model(pixel_values=padded_videos, visible_indices=visible_index)
            tgt_feat = tgt_out.last_hidden_state
            tgt_head = tgt_out.pooler_output
        except Exception as e:
            print(f"    [Error] Target forward failed: {e}")
            return

    if src_feat is not None and tgt_feat is not None:
        diff_feat = (src_feat - tgt_feat).abs().max().item()
        src_feat_flat = src_feat.flatten(0, -2).float()
        tgt_feat_flat = tgt_feat.flatten(0, -2).float()
        cos_sim = F.cosine_similarity(src_feat_flat, tgt_feat_flat, dim=-1)

        min_cos = cos_sim.min().item()
        mean_cos = cos_sim.mean().item()

        print(f"    [Video Feature] Max Diff:       {diff_feat:.6f}")
        print(f"    [Video Feature] Min Cosine Sim: {min_cos:.8f} (Mean: {mean_cos:.8f})")
        if min_cos > 0.99: print("    ✅ Video Feature: PASS")
        else: print("    ❌ Video Feature: FAIL")

    if src_head is not None and tgt_head is not None:
        diff_head = (src_head - tgt_head).abs().max().item()
        src_head_f = src_head.float()
        tgt_head_f = tgt_head.float()
        cos_sim_head = F.cosine_similarity(src_head_f, tgt_head_f, dim=-1)

        min_cos_head = cos_sim_head.min().item()
        mean_cos_head = cos_sim_head.mean().item() # Already present here, just making sure

        print(f"    [Video Head]    Max Diff:       {diff_head:.6f}")
        print(f"    [Video Head]    Min Cosine Sim: {min_cos_head:.8f} (Mean: {mean_cos_head:.8f})")
        if min_cos_head > 0.99: print("    ✅ Video Head:    PASS")
        else: print("    ❌ Video Head:    FAIL")


def verify_saved_model_loading(src_model, output_dir, real_image_tensor):
    print("\n=== Verifying Loaded Saved Model (Simulating User Usage - bfloat16) ===")
    print(f"--> Loading from: {output_dir}")

    device = next(src_model.parameters()).device
    print(f"    Using device from src_model: {device}")

    try:
        print("    Loading Image Processor (CLIP)...")
        image_processor = CLIPImageProcessor.from_pretrained(output_dir)
        print(f"    Processor config: {image_processor}")

        print("    Loading Vision Tower (MLCDVisionModel) with torch_dtype=bfloat16...")
        # CRITICAL: 显式指定加载为 bfloat16
        vision_tower = MLCDVisionModel.from_pretrained(output_dir, torch_dtype=torch.bfloat16)
        vision_tower.to(device)
        vision_tower.eval()

        print("    ✅ Successfully loaded and moved to device.")

    except Exception as e:
        print(f"    ❌ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return

    # Cosine Sim Check
    # CRITICAL: Input must be bfloat16
    input_tensor = real_image_tensor.to(device, dtype=torch.bfloat16)

    with torch.no_grad():
        src_out = src_model(input_tensor)['visible_embeddings']
        tgt_out = vision_tower(pixel_values=input_tensor).last_hidden_state

    src_feat_flat = src_out.flatten(0, -2).float()
    tgt_feat_flat = tgt_out.flatten(0, -2).float()
    cos_sim = F.cosine_similarity(src_feat_flat, tgt_feat_flat, dim=-1)

    min_cos = cos_sim.min().item()
    mean_cos = cos_sim.mean().item() # Added Mean

    print(f"    [Reloaded Model] Min Cosine Sim: {min_cos:.8f} (Mean: {mean_cos:.8f})")
    if min_cos > 0.99:
        print("    ✅ Reloaded Model Verification: PASS")
    else:
        print("    ❌ Reloaded Model Verification: FAIL")


def convert_and_save(src_model_name, tgt_model_name, weight_path, output_dir):
    print(f"=== Conversion Task ===")
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

    # 2. 创建 Target Model (默认 CPU)
    print("\n--> Creating Target Model...")
    tgt_model = timm.create_model(tgt_model_name, pretrained=False)

    print("--> Remapping State Dict...")
    hf_state_dict = remap_state_dict(src_model.state_dict())

    print("--> Loading weights into Target...")
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

    # 验证内存中的模型
    verify_consistency(src_model, tgt_model, real_img)

    # 验证视频 (会自动 resize 到 224)
    verify_consistency_video(src_model, tgt_model, real_img)

    if output_dir:
        print(f"\n--> Saving HF Model to {output_dir}...")
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

            print("✅ Model (bf16) and CLIP Processor saved.")

            # 验证 Reload 后的模型
            verify_saved_model_loading(src_model, output_dir, real_img)

        else:
            print("❌ Error: Target model does not have save_pretrained method.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", type=str, help="Source model name")
    parser.add_argument("weight_path", type=str, help="Path to .pth checkpoint")
    parser.add_argument("--target_model_name", type=str, default=None, help="Target HF model name")
    parser.add_argument("--output_dir", type=str, default=None)

    args = parser.parse_args()

    tgt_name = args.target_model_name
    if tgt_name is None:
        tgt_name = f"hf_{args.model_name}"

    out_dir = args.output_dir
    if out_dir is None:
        p = Path(args.weight_path)
        out_dir = os.path.join(p.parent, f"{p.stem}_hf")

    convert_and_save(args.model_name, tgt_name, args.weight_path, out_dir)
