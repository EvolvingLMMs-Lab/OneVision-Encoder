#!/bin/bash
# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

# 模型配置
MODEL_FAMILY="metaclip"
MODEL_NAME="metaclip_base16_fullcc"
FRAMES_TOKEN_NUM=196
EMBEDDING_SIZE=768

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
