#!/bin/bash
# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Model configuration
MODEL_FAMILY="llava_vit_sampling"
MODEL_NAME="llava_vit_large_ln"
MODEL_WEIGHT=$1
FRAMES_TOKEN_NUM=256
EMBEDDING_SIZE=1024

# Run evaluation
run_attentive_probe
