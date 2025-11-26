#!/bin/bash
# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

# 模型配置
MODEL_FAMILY="llava_vit_sampling"
MODEL_NAME="llava_vit_base_ln"
MODEL_WEIGHT=$1
FRAMES_TOKEN_NUM=196
EMBEDDING_SIZE=768
NUM_FRAMES=16
REPORT_DIR_SUFFIX="_16frames"

# 运行评估
run_attentive_probe
