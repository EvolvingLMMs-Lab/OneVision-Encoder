#!/bin/bash
# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

# Model configuration
MODEL_FAMILY="dinov3"
MODEL_NAME="dinov3_base"
FRAMES_TOKEN_NUM=196
EMBEDDING_SIZE=768
INPUT_SIZE=224
NUM_FRAMES=8
REPORT_DIR_SUFFIX="_8frames"

# Run evaluation
run_attentive_probe
