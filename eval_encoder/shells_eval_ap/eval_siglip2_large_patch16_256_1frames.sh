#!/bin/bash
# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

# Model configuration
MODEL_FAMILY="siglip2"
MODEL_NAME="siglip2_large_patch16_256"
FRAMES_TOKEN_NUM=256
EMBEDDING_SIZE=1024
INPUT_SIZE=256
NUM_FRAMES=1
REPORT_DIR_SUFFIX="_1frames"

DATASETS=(
    # "ssv2"
    # "diving48"
    "perception_test"
    # "epic_verb"
    # "epic_noun"
    # "hmdb51"
    # "k400"
    # "charadesego"
)

# Run evaluation
run_attentive_probe
