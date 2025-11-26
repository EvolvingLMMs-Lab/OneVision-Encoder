#!/bin/bash
# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

# Model configuration
MODEL_FAMILY="llava_vit_sampling"
MODEL_NAME="llava_vit_base_ln"
# Note: Multiple MODEL_WEIGHT assignments track checkpoint evolution; only the last one is used
MODEL_WEIGHT="/video_vit/xiangan/checkpoint_llava_vit/2025_11_19_new_b16_continue_80gpus_how_to_100m_continue/00040000/backbone.pt"
MODEL_WEIGHT="/video_vit/xiangan/checkpoint_llava_vit/2025_11_23_new_b16_continue_80gpus_how_to_100m_num_frames_16/00064000/backbone.pt"
MODEL_WEIGHT="/video_vit/xiangan/checkpoint_llava_vit/2025_11_23_new_b16_continue_80gpus_how_to_100m_num_frames_16/00076000/backbone.pt"
FRAMES_TOKEN_NUM=196
EMBEDDING_SIZE=768
REPORT_DIR_SUFFIX="_8frames"

# Run evaluation
run_attentive_probe
