#!/bin/bash
# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

# Model configuration
MODEL_FAMILY="metaclip"
MODEL_NAME="metaclip_large14_fullcc"
FRAMES_TOKEN_NUM=256
EMBEDDING_SIZE=1024

# Custom dataset list
DATASETS=(
    "ssv2"
    "diving48"
    "perception_test"
    "epic_verb"
    "epic_noun"
    "hmdb51"
    "k400"
    "charadesego"
)

NUM_FRAMES=16
REPORT_DIR_SUFFIX="_16frames"

# Run evaluation
run_attentive_probe
