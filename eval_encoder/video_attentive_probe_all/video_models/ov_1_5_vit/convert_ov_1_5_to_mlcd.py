from transformers import Qwen2Tokenizer, AutoProcessor, AutoConfig
from transformers import MLCDVisionModel
from transformers import CLIPImageProcessor
from transformers import Qwen2VLImageProcessor, AutoProcessor
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import torch
import numpy as np
from transformers import Qwen2Tokenizer, logging
from safetensors.torch import load_file
from PIL import Image, ImageDraw
from huggingface_hub import hf_hub_download, snapshot_download
import requests
from io import BytesIO
import sys

import argparse

def load_ori_vit(model_name):
    model = MLCDVisionModel.from_pretrained(model_name)
    preprocess = AutoProcessor.from_pretrained(model_name)
    return model, preprocess

REVERSE_VIT_KEYS_TO_MODIFY_MAPPING = {
    "visual.": "vision_model.",
    # "embeddings.": "embeddings.",
    # "visual.patch_embed.proj.": "model.visual.patch_embedding.",
    "blocks.": "encoder.layers.",
    "class_embedding": "embeddings.class_embedding",
    "patch_embed.proj.": "embeddings.patch_embedding.",
    "pre_layernorm": "pre_layrnorm",
    ".norm": ".layer_norm",
    ".attn.proj.": ".self_attn.out_proj.",
}

def split_qkv_weights(state_dict, block_prefix):
    """
    Args:
        state_dict: 包含合并后的 qkv 权重和偏置的 dict
        block_prefix: block 前缀
        qkv_dim: 每个 q/k/v 的输出 dim（假设 q/k/v 的输出 dim 一样）
    Returns:
        dict: 包含拆分后的 q_proj, k_proj, v_proj 权重和偏置
    """
    # 取出合并后的权重和偏置
    qkv_weight = state_dict[f"{block_prefix}.attn.qkv.weight"]
    qkv_bias = state_dict[f"{block_prefix}.attn.qkv.bias"]

    qkv_dim = qkv_weight.shape[0] // 3  # 每个 q/k/v 的输出维度

    # 拆分权重
    q_w = qkv_weight[:qkv_dim]
    k_w = qkv_weight[qkv_dim:2*qkv_dim]
    v_w = qkv_weight[2*qkv_dim:3*qkv_dim]

    # 拆分偏置
    q_b = qkv_bias[:qkv_dim]
    k_b = qkv_bias[qkv_dim:2*qkv_dim]
    v_b = qkv_bias[2*qkv_dim:3*qkv_dim]

    return {
        f"{block_prefix}.self_attn.q_proj.weight": q_w,
        f"{block_prefix}.self_attn.k_proj.weight": k_w,
        f"{block_prefix}.self_attn.v_proj.weight": v_w,
        f"{block_prefix}.self_attn.q_proj.bias": q_b,
        f"{block_prefix}.self_attn.k_proj.bias": k_b,
        f"{block_prefix}.self_attn.v_proj.bias": v_b,
    }

def convert_state_dict(state_dict):
        new_state_dict = {}
        for key, value in state_dict.items():
            if 'visual' not in key or 'merge' in key:
                continue
            # if key.endswith(".inv_freq"):
            #     continue
            for key_to_modify, new_key in REVERSE_VIT_KEYS_TO_MODIFY_MAPPING.items():
                if key_to_modify in key:
                    key = key.replace(key_to_modify, new_key)

            new_state_dict[key] = value

        new_state_dict2 = {}
        for key, value in new_state_dict.items():
            if key.startswith("vision_model.encoder.layers.") and "attn." in key and "qkv." in key:
                block_index = key.split('.')[3]
                block_prefix = f"vision_model.encoder.layers.{block_index}"
                split_res = split_qkv_weights(new_state_dict, block_prefix)
                new_state_dict2.update(split_res)
            else:
                new_state_dict2[key] = value
        return new_state_dict2

def convert_vlm2rice(vlm_model_name, rice_model_name, save_path):
    # Load VLM model
    vlm_weights = {}
    for weight_path in os.listdir(vlm_model_name):
        if weight_path.endswith('.safetensors'):
            weight_file = os.path.join(vlm_model_name, weight_path)
            vlm_weights.update(load_file(weight_file))
    
    convert_vlm_weights = convert_state_dict(vlm_weights)

    # Load RICE model
    rice_model, preprocess = load_ori_vit(rice_model_name)

    rice_model.load_state_dict(convert_vlm_weights,strict=False)

    print('Loaded VLM and RICE models.')


    # Save converted model
    os.makedirs(save_path, exist_ok=True)
    rice_model.save_pretrained(save_path)
    preprocess.save_pretrained(save_path)
    # rice_tokenizer.save_pretrained(save_path)
    print(f"Converted model saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert VLM model to RICE format")
    parser.add_argument("--vlm_model_name", type=str, default='/vlm/xiangan/datasets/date_2025_08_10_pretrain_rice_vl_3b_1x_epoch_53m_pretrain_small_res_32768_mbs_1_gbs_896_59743_steps_huggingface', help="Path or name of the VLM model")
    parser.add_argument("--rice_model_name", type=str, default='DeepGlint-AI/rice-vit-large-patch14-560', help="Path or name of the RICE model")
    parser.add_argument("--save_path", type=str, default='/vlm/yinxie/code/checkpoints/rice-vit-large-patch14-560-w3b-af53mid', help="Path to save the converted model")
    args = parser.parse_args()

    convert_vlm2rice(args.vlm_model_name, args.rice_model_name, args.save_path)