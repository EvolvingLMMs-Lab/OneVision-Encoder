#!/bin/bash
# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

# Model configuration
MODEL_FAMILY="siglip"
MODEL_NAME="siglip_base"
FRAMES_TOKEN_NUM=196
EMBEDDING_SIZE=768
NUM_FRAMES=8
REPORT_DIR_SUFFIX="_8frames"

# Custom dataset list
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

# Run evaluation
run_attentive_probe
