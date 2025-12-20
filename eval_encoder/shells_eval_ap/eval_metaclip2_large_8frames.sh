#!/bin/bash
# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

# Model configuration
MODEL_FAMILY="metaclip"
MODEL_NAME="metaclip2_large14"
FRAMES_TOKEN_NUM=256
EMBEDDING_SIZE=1024

NUM_FRAMES=8
REPORT_DIR_SUFFIX="_8frames"

# Run evaluation
run_attentive_probe
