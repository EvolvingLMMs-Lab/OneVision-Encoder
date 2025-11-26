#!/bin/bash
# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

# 模型配置
MODEL_FAMILY="metaclip"
MODEL_NAME="metaclip_large14_fullcc"
FRAMES_TOKEN_NUM=256
EMBEDDING_SIZE=1024

# 自定义数据集列表
DATASETS=(
    # "ssv2"
    # "diving48"
    # "perception_test"
    # "epic_verb"
    # "epic_noun"
    "hmdb51"
    # "k400"
    "charadesego"
)

# 运行评估
run_attentive_probe
