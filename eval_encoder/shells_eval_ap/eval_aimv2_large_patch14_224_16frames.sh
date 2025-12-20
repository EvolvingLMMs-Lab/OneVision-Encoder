#!/bin/bash
# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

# Model configuration
MODEL_FAMILY="aimv2"
MODEL_NAME="aimv2_large_patch14_native_ap"
FRAMES_TOKEN_NUM=256
EMBEDDING_SIZE=1024

NUM_FRAMES=16
REPORT_DIR_SUFFIX="_16frames"

# Run evaluation
run_attentive_probe
