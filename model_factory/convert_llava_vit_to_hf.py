import argparse
import os
from pathlib import Path

import numpy as np
import timm
import torch
from model_factory import vit_preview_v0_hf, vit_preview_v0

# =============================================================================
# 映射逻辑 (Source -> Target)
# =============================================================================
def remap_state_dict(src_state_dict):
    """
    将 Source Model (LlavaViTEncoder) 的权重 Key 映射到 Target Model (HFLlavaViTModel)
    """
    new_dict = {}
    for k, v in src_state_dict.items():
        new_k = k

        # --- 1. Patch Embeddings ---
        # Source: conv1.weight -> Target: embeddings.patch_embedding.weight
        if k.startswith("conv1."):
            new_k = k.replace("conv1.", "embeddings.patch_embedding.")

        # --- 2. Pre/Post Norm ---
        # Source: ln_pre -> Target: layernorm_pre
        elif k.startswith("ln_pre."):
            new_k = k.replace("ln_pre.", "layernorm_pre.")
        # Source: ln_post -> Target: layernorm_post
        elif k.startswith("ln_post."):
            new_k = k.replace("ln_post.", "layernorm_post.")

        # --- 3. Transformer Layers ---
        # Source: transformer.layers.0. -> Target: encoder.layers.0.
        elif k.startswith("transformer.layers."):
            new_k = k.replace("transformer.layers.", "encoder.layers.")

            # Layer Norms
            # Source: norm1 -> Target: layer_norm1
            new_k = new_k.replace(".norm1.", ".layer_norm1.")
            new_k = new_k.replace(".norm2.", ".layer_norm2.")

            # Attention
            # Source: attn -> Target: self_attn
            new_k = new_k.replace(".attn.", ".self_attn.")

            # Projections
            # Source 代码若用 TransformerCausal，通常内部有 proj/q_proj 等
            # 你的 HF 代码用的是: q_proj, k_proj, v_proj, out_proj

            # 映射 output projection
            if ".proj." in new_k:
                new_k = new_k.replace(".proj.", ".out_proj.")

        # --- 4. Head (Siglip2MultiheadAttentionPoolingHead) ---
        # Source: head -> Target: head (名字通常一致)

        # --- 5. RoPE ---
        # Source: video_rope -> Target: video_rope (名字通常一致)

        new_dict[new_k] = v

    return new_dict

# =============================================================================
# 验证逻辑
# =============================================================================
def verify_consistency(src_model, tgt_model):
    print("\n=== Verifying Consistency ===")
    src_model.eval()
    tgt_model.eval()

    # 1. 固定随机种子
    torch.manual_seed(42)
    np.random.seed(42)

    # 2. 构造输入 (假设是标准的 Image/Video Input)
    # timm 里的 patch_embed 通常存了 img_size, 如果没有就默认 224
    img_size = 224
    if hasattr(src_model, 'patch_size'):
        # 尝试获取 patch_size 属性
        pass

    # 构造输入: [Batch, Channel, Frames, Height, Width]
    # 如果是纯图片模型，Source 可能只接受 [B, C, H, W]
    # 这里根据 Source 模型的 conv1 维度来判断
    is_video = False
    if hasattr(src_model, 'conv1'):
        # 如果 conv1 是 Conv3d 则肯定是视频，如果是 Conv2d 可能是图片或 (B*T) 图片
        # 你的代码里 conv1 是 Conv2d，但在 forward 里把 T 拆出来了
        is_video = True

    B, C, T, H, W = 1, 3, 8, img_size, img_size

    # 构造输入 Tensor
    input_tensor = torch.randn(B, C, T, H, W)

    print(f"    Input Shape: {input_tensor.shape}")

    with torch.no_grad():
        # --- Source Forward ---
        # 你的 Source forward 签名: forward(x, visible_indices=None, ...)
        # 它返回一个 dict: {'visible_embeddings': ..., 'head_output': ...}
        try:
            src_out_dict = src_model(input_tensor)
            if isinstance(src_out_dict, dict):
                src_feat = src_out_dict.get('visible_embeddings')
                src_head = src_out_dict.get('head_output')
            else:
                src_feat = src_out_dict # 假设直接返回 feature
                src_head = None
        except Exception as e:
            print(f"    [Error] Source model forward failed: {e}")
            return

        # --- Target Forward (HF) ---
        # HF forward 通常接受 pixel_values
        # 你的 HF forward 签名: forward(pixel_values, ...)
        try:
            tgt_out = tgt_model(pixel_values=input_tensor)
            # HF 通常返回 BaseModelOutputWithPooling
            # last_hidden_state, pooler_output
            tgt_feat = tgt_out.last_hidden_state
            tgt_head = tgt_out.pooler_output
        except Exception as e:
            print(f"    [Error] Target model forward failed: {e}")
            return

    # 3. 比较特征
    # 需要确保两个 tensor 维度对齐
    # Source: (B, N, C)
    # Target: (B, N, C)

    if src_feat is not None and tgt_feat is not None:
        diff_feat = (src_feat - tgt_feat).abs().max().item()
        print(f"    Feature Max Diff: {diff_feat:.6f}")
        if diff_feat < 1e-4:
            print("    ✅ Feature: PASS")
        else:
            print("    ❌ Feature: FAIL (Difference too large)")

    # 4. 比较 Head Output (如果用了 Head)
    if src_head is not None and tgt_head is not None:
        diff_head = (src_head - tgt_head).abs().max().item()
        print(f"    Head Max Diff:    {diff_head:.6f}")
        if diff_head < 1e-4:
            print("    ✅ Head:    PASS")
        else:
            print("    ❌ Head:    FAIL")
    elif src_head is None and tgt_head is None:
        print("    [Info] No head output to compare (both None).")
    else:
        print("    [Warning] Head output mismatch (one is None, one is Tensor).")


# =============================================================================
# 主转换函数
# =============================================================================
def convert_and_save(src_model_name, tgt_model_name, weight_path, output_dir):
    print(f"=== Conversion Task ===")
    print(f"Source:  {src_model_name}")
    print(f"Target:  {tgt_model_name}")
    print(f"Weights: {weight_path}")
    print(f"Output:  {output_dir}")

    # 1. 创建 Source Model
    print(f"\n--> Creating Source Model (timm)...")
    try:
        # num_classes=0 避免加载分类头
        src_model = timm.create_model(src_model_name, pretrained=False)
    except Exception as e:
        print(f"Error creating source model '{src_model_name}': {e}")
        return

    # 2. 加载权重到 Source
    print(f"--> Loading weights from {weight_path}...")
    checkpoint = torch.load(weight_path, map_location='cpu')
    state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))

    missing, unexpected = src_model.load_state_dict(state_dict, strict=False)
    print(f"    Source Load: Missing={len(missing)}, Unexpected={len(unexpected)}")

    # 3. 创建 Target Model
    print(f"\n--> Creating Target Model (HF style via timm)...")
    try:
        tgt_model = timm.create_model(tgt_model_name, pretrained=False)
    except Exception as e:
        print(f"Error creating target model '{tgt_model_name}': {e}")
        return

    # 4. 映射并加载到 Target
    print(f"--> Mapping and Loading State Dict...")
    hf_state_dict = remap_state_dict(src_model.state_dict())

    missing, unexpected = tgt_model.load_state_dict(hf_state_dict, strict=False)
    print(f"    Target Load: Missing={len(missing)}, Unexpected={len(unexpected)}")

    if len(missing) > 0:
        for k in missing:
            print(f"    [Missing] {k}")
    if len(unexpected) > 0:
        for k in unexpected:
            print(f"    [Unexpected] {k}")

    # 5. 验证一致性
    verify_consistency(src_model, tgt_model)

    # 6. 保存
    print(f"\n--> Saving HF pretrained to {output_dir}")
    if hasattr(tgt_model, "save_pretrained"):
        tgt_model.save_pretrained(output_dir)
        print("✅ Done.")
    else:
        print("❌ Error: Target model does not support 'save_pretrained'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", type=str, help="Source model name (timm registry)")
    parser.add_argument("weight_path", type=str, help="Path to .pth checkpoint")
    parser.add_argument("--target_model_name", type=str, default=None, help="Target model name (if different)")
    parser.add_argument("--output_dir", type=str, default=None)

    args = parser.parse_args()

    # 1. 推断 Target Name (如果未提供)
    # 假设你的 HF 模型注册名为 "hf_" + source_name (例如 llava_vit_base -> hf_llava_vit_base)
    # 或者是你在 timm 里注册的另一个名字
    tgt_name = args.target_model_name
    if tgt_name is None:
        # 这里需要你自己在 timm 注册时保证命名规则，或者手动传入
        # 假设我们简单地在前面加 hf_ 前缀来尝试寻找目标模型
        # 或者你注册 Target 模型时用的就是比如 'hf_llava_vit_base'
        tgt_name = f"hf_{args.model_name}"
        print(f"[Info] Guessing target model name: {tgt_name}")

    # 2. 推断 Output Dir
    out_dir = args.output_dir
    if out_dir is None:
        p = Path(args.weight_path)
        out_dir = os.path.join(p.parent, f"{p.stem}_hf")

    # 重要：确保你的模型代码在这里被 import，否则 timm 找不到模型
    # import my_model_definitions

    convert_and_save(args.model_name, tgt_name, args.weight_path, out_dir)
