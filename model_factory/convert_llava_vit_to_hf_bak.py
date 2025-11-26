import argparse
import torch
import timm
import numpy as np
# 确保导入你的模型定义
from model_factory import vit_preview_v0_hf, vit_preview_v0

def remap_state_dict(src_state_dict):
    new_dict = {}
    qkv_cache = {}
    for k, v in src_state_dict.items():
        new_k = k
        # 1. 基础替换
        if k.startswith("conv1."): new_k = k.replace("conv1.", "embeddings.patch_embedding.")
        elif k.startswith("ln_pre."): new_k = k.replace("ln_pre.", "layernorm_pre.")
        elif k.startswith("ln_post."): new_k = k.replace("ln_post.", "layernorm_post.")
        elif k.startswith("transformer.layers."):
            new_k = k.replace("transformer.layers.", "encoder.layers.")
            # Layer 内部
            if ".ln_1." in new_k: new_k = new_k.replace(".ln_1.", ".layer_norm1.")
            if ".ln_2." in new_k: new_k = new_k.replace(".ln_2.", ".layer_norm2.")
            if ".mlp.0." in new_k: new_k = new_k.replace(".mlp.0.", ".mlp.fc1.")
            if ".mlp.2." in new_k: new_k = new_k.replace(".mlp.2.", ".mlp.fc2.")
            if ".attn." in new_k: new_k = new_k.replace(".attn.", ".self_attn.")
            # Proj (小心不覆盖 in_proj)
            if ".proj." in new_k and "in_proj" not in new_k: new_k = new_k.replace(".proj.", ".out_proj.")

            # QKV Split 逻辑
            if "in_proj" in new_k:
                prefix = new_k.rsplit(".in_proj", 1)[0]
                param_type = new_k.split(".")[-1]
                if prefix not in qkv_cache: qkv_cache[prefix] = {}
                qkv_cache[prefix][param_type] = v
                continue
        new_dict[new_k] = v

    # 处理 QKV
    for prefix, params in qkv_cache.items():
        for p_type, tensor in params.items():
            dim = tensor.shape[0] // 3
            q, k, v_part = torch.split(tensor, dim, dim=0)
            new_dict[f"{prefix}.q_proj.{p_type}"] = q
            new_dict[f"{prefix}.k_proj.{p_type}"] = k
            new_dict[f"{prefix}.v_proj.{p_type}"] = v_part
    return new_dict

def debug_layers(src_model, tgt_model):
    print("\n=== LAYER-WISE DEBUG ===")
    src_model.eval()
    tgt_model.eval()
    torch.manual_seed(42)

    # Input
    img_size = 224
    B, C, H, W = 1, 3, img_size, img_size
    x = torch.randn(B, C, H, W)
    print(f"Input: {x.shape}")

    # 获取 Patch Size 以计算 Grid
    # src_model.patch_size 可能是 (14, 14) 或 14
    if hasattr(src_model, "patch_size"):
        ps = src_model.patch_size
        if isinstance(ps, tuple): ps = ps[0]
    else:
        ps = 16 # fallback

    grid_h = H // ps
    grid_w = W // ps
    print(f"Patch Size: {ps}, Grid: {grid_h}x{grid_w}, Tokens: {grid_h*grid_w}")

    # --- 1. Patch Embedding ---
    src_x_2d = x.unsqueeze(2).permute(0,2,1,3,4).reshape(B, C, H, W)
    src_emb = src_model.conv1(src_x_2d)
    src_emb = src_emb.flatten(2).transpose(1, 2)

    tgt_emb = tgt_model.embeddings(x)

    diff_emb = (src_emb - tgt_emb).abs().max().item()
    print(f"[1] Patch Embedding Diff: {diff_emb:.6f} " + ("✅" if diff_emb < 1e-4 else "❌"))

    # --- 2. Pre-Norm ---
    src_pre = src_model.ln_pre(src_emb)
    tgt_pre = tgt_model.layernorm_pre(tgt_emb)

    diff_pre = (src_pre - tgt_pre).abs().max().item()
    print(f"[2] Pre-Norm Diff:        {diff_pre:.6f} " + ("✅" if diff_pre < 1e-4 else "❌"))

    # --- 3. Layer 0 Components ---
    src_l0_in = src_pre.permute(1, 0, 2) # (L, B, C)
    tgt_l0_in = tgt_pre # (B, L, C)

    src_layer0 = src_model.transformer.layers[0]
    tgt_layer0 = tgt_model.encoder.layers[0]

    # 构造 RoPE (使用动态计算的 grid)
    # Source
    src_rope = src_model.video_rope(t=1, h=grid_h, w=grid_w, device=x.device)
    src_rope = src_rope.unsqueeze(0)

    # Target
    tgt_rope = tgt_model.video_rope(t=1, h=grid_h, w=grid_w, device=x.device)
    tgt_rope = torch.cat([tgt_rope, tgt_rope], dim=-1).unsqueeze(0)

    # Norm 1
    src_ln1 = src_layer0.ln_1(src_l0_in)
    tgt_ln1 = tgt_layer0.layer_norm1(tgt_l0_in)

    diff_ln1 = (src_ln1.permute(1,0,2) - tgt_ln1).abs().max().item()
    print(f"[3] Layer0 LN1 Diff:      {diff_ln1:.6f} " + ("✅" if diff_ln1 < 1e-4 else "❌"))


    # RoPE Freq Check
    src_rope_raw = src_model.video_rope(t=1, h=grid_h, w=grid_w, device=x.device)
    tgt_rope_raw = tgt_model.video_rope(t=1, h=grid_h, w=grid_w, device=x.device)
    print(f"RoPE Raw Diff: {(src_rope_raw - tgt_rope_raw).abs().max().item():.6f}")

    # 把这一段改成这样，暂时禁用 RoPE
    with torch.no_grad():
        src_attn_out = src_layer0.attn(src_ln1, rotary_pos_emb=None)
        tgt_attn_out, _ = tgt_layer0.self_attn(tgt_ln1, rotary_pos_emb=None)

    diff_attn = (src_attn_out.permute(1,0,2) - tgt_attn_out).abs().max().item()
    print(f"[4] Layer0 Attn Diff (No RoPE): {diff_attn:.6f} " + ("✅" if diff_attn < 1e-4 else "❌"))

    # Attention
    # Source
    with torch.no_grad():
        src_attn_out = src_layer0.attn(src_ln1, rotary_pos_emb=src_rope)
        tgt_attn_out, _ = tgt_layer0.self_attn(tgt_ln1, rotary_pos_emb=tgt_rope)

    diff_attn = (src_attn_out.permute(1,0,2) - tgt_attn_out).abs().max().item()
    print(f"[4] Layer0 Attn Diff:     {diff_attn:.6f} " + ("✅" if diff_attn < 1e-4 else "❌"))

    if diff_attn > 1e-4:
        print("    !!! Attention Mismatch Found !!! Checking details...")
        # Check Q Projection
        src_qkv = src_layer0.attn.in_proj(src_ln1) # (L, B, 3C)
        tgt_q = tgt_layer0.self_attn.q_proj(tgt_ln1) # (B, L, C)

        C = src_qkv.shape[2] // 3
        src_q = src_qkv[..., :C] # (L, B, C)

        diff_q = (src_q.permute(1,0,2) - tgt_q).abs().max().item()
        print(f"    Q Proj Diff:          {diff_q:.6f}")

        if diff_q < 1e-4:
            print("    (Q Proj matches, so Linear weights are correct. Problem is likely in RoPE or Attention logic)")
        else:
            print("    (Q Proj mismatch, implies QKV Split or Linear weight loading is wrong)")

def convert_and_debug(src_name, tgt_name, weight_path):
    print(f"Loading Source: {src_name}")
    src_model = timm.create_model(src_name, pretrained=False)
    src_model.load_state_dict(torch.load(weight_path, map_location='cpu'), strict=False)

    print(f"Loading Target: {tgt_name}")
    tgt_model = timm.create_model(tgt_name, pretrained=False)
    tgt_model.load_state_dict(remap_state_dict(src_model.state_dict()), strict=False)

    debug_layers(src_model, tgt_model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", type=str)
    parser.add_argument("weight_path", type=str)
    parser.add_argument("--target_model_name", type=str, default=None)
    args = parser.parse_args()

    tgt = args.target_model_name if args.target_model_name else f"hf_{args.model_name}"
    convert_and_debug(args.model_name, tgt, args.weight_path)
