#!/bin/bash
# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_codec.sh"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Model configuration
MODEL_FAMILY="ov_encoder_codec"
MODEL_NAME="ov_encoder_large"
MODEL_WEIGHT=$1
FRAMES_TOKEN_NUM=256
EMBEDDING_SIZE=1024
NUM_FRAMES=64
INPUT_SIZE=224
REPORT_DIR_SUFFIX="_2kpatches_codec"

# Codec-specific parameters
K_KEEP=2048
PATCH_SIZE=14
MV_COMPENSATE="similarity"
CENTER_PRIOR=0.0
STATIC_FALLBACK=1

DEFAULT_DATASETS=(
    "ssv2"
    "diving48"
    "perception_test"
    "epic_verb"
    "epic_noun"
    "hmdb51"
    "k400"
    "charadesego"
)

# Run evaluation
run_attentive_probe_codec
