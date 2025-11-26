#!/bin/bash
# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

# 模型配置
MODEL_FAMILY="dinov3"
MODEL_NAME="dinov3_base"
FRAMES_TOKEN_NUM=196
EMBEDDING_SIZE=768
INPUT_SIZE=224

# 运行评估
run_attentive_probe
