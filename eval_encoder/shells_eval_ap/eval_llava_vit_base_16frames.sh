#!/bin/bash
# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

# Model configuration
MODEL_FAMILY="llava_vit_sampling"
MODEL_NAME="llava_vit_base_ln"
MODEL_WEIGHT=$1
FRAMES_TOKEN_NUM=196
EMBEDDING_SIZE=768
NUM_FRAMES=16
REPORT_DIR_SUFFIX="_16frames"

# Run evaluation
run_attentive_probe
