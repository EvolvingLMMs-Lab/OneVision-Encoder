#!/bin/bash
# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

# Model configuration
MODEL_FAMILY="siglip2"
MODEL_NAME="siglip2_so400m_patch16_naflex"
FRAMES_TOKEN_NUM=256
EMBEDDING_SIZE=1024
INPUT_SIZE=256
NUM_FRAMES=16
REPORT_DIR_SUFFIX="_16frames"

# Run evaluation
run_attentive_probe
