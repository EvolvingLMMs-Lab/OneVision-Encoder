#!/bin/bash
# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

# 模型配置
MODEL_FAMILY="llava_vit_sampling"
MODEL_NAME="llava_vit_base_ln"
MODEL_WEIGHT="/video_vit/xiangan/checkpoint_llava_vit/2025_11_19_new_b16_continue_80gpus_how_to_100m_continue/00040000/backbone.pt"
MODEL_WEIGHT="/video_vit/xiangan/checkpoint_llava_vit/2025_11_23_new_b16_continue_80gpus_how_to_100m_num_frames_16/00064000/backbone.pt"
MODEL_WEIGHT="/video_vit/xiangan/checkpoint_llava_vit/2025_11_23_new_b16_continue_80gpus_how_to_100m_num_frames_16/00076000/backbone.pt"
FRAMES_TOKEN_NUM=196
EMBEDDING_SIZE=768
REPORT_DIR_SUFFIX="_8frames"

# 运行评估
run_attentive_probe
