#!/bin/bash
# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

# 模型配置
MODEL_FAMILY="siglip2"
MODEL_NAME="siglip2_large_patch16_256"
FRAMES_TOKEN_NUM=256
EMBEDDING_SIZE=1024
INPUT_SIZE=256

# 运行评估
run_attentive_probe
