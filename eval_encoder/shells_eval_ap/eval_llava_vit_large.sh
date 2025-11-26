#!/bin/bash
# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 模型配置
MODEL_FAMILY="llava_vit_sampling"
MODEL_NAME="llava_vit_large_ln"
MODEL_WEIGHT=$1
FRAMES_TOKEN_NUM=256
EMBEDDING_SIZE=1024

# 运行评估
run_attentive_probe
