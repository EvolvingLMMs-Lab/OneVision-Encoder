#!/bin/bash
# ============================================================================
# OneVision Encoder Evaluation Script
# ============================================================================
# This script evaluates OneVision Encoder models on various benchmarks.
#
# IMPORTANT: Video benchmarks (videomme, perceptiontest, mvbench, etc.) should
#            be tested ONE AT A TIME due to:
#            1. Large memory consumption for video processing
#            2. Different codec offline assets for each benchmark
#            3. Easier debugging and result tracking
#
# Usage:
#   # For image benchmarks (can run multiple together):
#   TASKS="ai2d,chartqa,docvqa_val" bash scripts/eval/eval_ov_encoder.sh
#
#   # For video benchmarks (run ONE at a time):
#   TASKS="videomme" bash scripts/eval/eval_ov_encoder.sh
#   TASKS="perceptiontest_val_mc" bash scripts/eval/eval_ov_encoder.sh
#   TASKS="mvbench" bash scripts/eval/eval_ov_encoder.sh
# ============================================================================

# Get the absolute path of the repo directory based on script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

export PYTHONPATH=${REPO_DIR}:${REPO_DIR}/lmms-eval
export HF_HOME=${REPO_DIR}/.huggingface_cache
export HF_ENDPOINT=https://hf-mirror.com

echo "[DEBUG] REPO_DIR=${REPO_DIR}"
echo "[DEBUG] HF_HOME=${HF_HOME}"

# ============ Offline Codec Mode Configuration ============
# Codec mode is ONLY used for VIDEO tasks that require frame extraction.
# Image tasks (chartqa, docvqa, etc.) do NOT use codec at all.
#
# When USE_OFFLINE_CODEC=1 (codec mode):
#   - For VIDEO tasks: use precomputed mosaic images + position indices
#   - Requires offline assets (precomputed by preprocess_video_benchmark.sh)
#
# When USE_OFFLINE_CODEC=0 (frame extraction mode):
#   - For VIDEO tasks: extract 8 frames uniformly from each video at runtime
#   - No preprocessing required, but slower
#
# Auto-detect based on MODEL_PATH:
#   - Models with "qwen3vl-vit" or "siglip2" in name -> USE_OFFLINE_CODEC=0 (frame extraction)
#   - Other models (onevision-encoder) -> USE_OFFLINE_CODEC=1 (codec)
# ===========================================================

# ============ Configuration (MODIFY THESE) ============
# Model path - change this to your model checkpoint
MODEL_PATH="${MODEL_PATH:-trained_model/must_contain_llava_in_name}"

# Tasks to evaluate - can be overridden via environment variable
# NOTE: For video benchmarks, test ONE at a time!
TASKS="${TASKS:-ai2d,chartqa,docvqa_val}"

# Model configuration
RUN_PORT="${RUN_PORT:-12456}"
MODEL_NAME="${MODEL_NAME:-onevision_encoder}"
CONV_TEMPLATE="${CONV_TEMPLATE:-qwen_1_5}"
NUM_GPUS="${NUM_GPUS:-8}"
# ======================================================

# Auto-detect USE_OFFLINE_CODEC based on MODEL_PATH
if [[ "$MODEL_PATH" == *"qwen3vl-vit"* ]] || [[ "$MODEL_PATH" == *"siglip2"* ]]; then
    USE_OFFLINE_CODEC=0
    echo "[AUTO] Detected qwen3vl-vit/siglip2 model -> USE_OFFLINE_CODEC=0 (8 frames per video)"
else
    USE_OFFLINE_CODEC=1
    echo "[AUTO] Detected codec model (onevision-encoder) -> USE_OFFLINE_CODEC=1 (mosaic + positions)"
fi

# Base directory for offline assets - MUST match HF_HOME to ensure video paths match
OFFLINE_CACHE_DIR="${HF_HOME}"

# Video task patterns that use codec
VIDEO_TASK_PATTERNS="videomme|perceptiontest|mvbench|nextqa|temporalbench|video_mmmu|tomato|longvideobench|egoschema|mlvu|mmvu"

# Task to offline directory mapping
declare -A TASK_OFFLINE_MAP
TASK_OFFLINE_MAP["videomme"]="videomme_offline"
TASK_OFFLINE_MAP["perceptiontest"]="perceptiontest_offline"
TASK_OFFLINE_MAP["perceptiontest_val_mc"]="perceptiontest_val_mc_offline"
TASK_OFFLINE_MAP["mvbench"]="mvbench_offline"
TASK_OFFLINE_MAP["nextqa_mc_test"]="nextqa_mc_test_offline"
TASK_OFFLINE_MAP["temporalbench"]="temporalbench_offline"
TASK_OFFLINE_MAP["temporalbench_long_qa"]="temporalbench_long_qa_offline"
TASK_OFFLINE_MAP["video_mmmu"]="video_mmmu_offline"
TASK_OFFLINE_MAP["tomato"]="tomato_offline"
TASK_OFFLINE_MAP["longvideobench"]="longvideobench_offline"
TASK_OFFLINE_MAP["longvideobench_val_v"]="longvideobench_val_v_offline"
TASK_OFFLINE_MAP["egoschema"]="egoschema_offline"
TASK_OFFLINE_MAP["mlvu"]="mlvu_offline"
TASK_OFFLINE_MAP["mlvu_dev"]="mlvu_dev_offline"
TASK_OFFLINE_MAP["mmvu"]="mmvu_offline"
TASK_OFFLINE_MAP["mmvu_val"]="mmvu_val_offline"

# Function to check if tasks contain any video tasks
is_video_task() {
    local tasks="$1"
    if [[ "$tasks" =~ $VIDEO_TASK_PATTERNS ]]; then
        return 0
    else
        return 1
    fi
}

# Function to setup offline codec for video tasks
setup_offline_codec() {
    local tasks="$1"
    local offline_roots=""
    for pattern in "${!TASK_OFFLINE_MAP[@]}"; do
        if [[ "$tasks" == *"$pattern"* ]]; then
            local offline_dir="${OFFLINE_CACHE_DIR}/${TASK_OFFLINE_MAP[$pattern]}/assets"
            if [ -d "$offline_dir" ]; then
                if [ -z "$offline_roots" ]; then
                    offline_roots="$offline_dir"
                else
                    offline_roots="$offline_roots:$offline_dir"
                fi
                echo "  [OK] Found: $pattern -> $offline_dir"
            else
                echo "  [WARNING] Not found: $pattern -> $offline_dir"
            fi
        fi
    done
    echo "$offline_roots"
}

# Codec parameters (only used when USE_OFFLINE_CODEC=1)
export LLAVA_CODEC_VISIDX_MODE=pack_topk
export LLAVA_CODEC_SEQ_LEN_FRAMES=64
export LLAVA_CODEC_NUM_IMAGES=8
export LLAVA_CODEC_SQUARE_SIZE=576
export LLAVA_CODEC_PATCH_SIZE=16

# ============ Auto-detect Codec Mode ============
echo "========================================"
echo "Task: $TASKS"
echo "Model: $MODEL_PATH"
echo "========================================"

# Check if multiple video tasks are specified
VIDEO_COUNT=0
for pattern in videomme perceptiontest mvbench nextqa temporalbench video_mmmu tomato longvideobench egoschema mlvu mmvu; do
    if [[ "$TASKS" == *"$pattern"* ]]; then
        VIDEO_COUNT=$((VIDEO_COUNT + 1))
    fi
done

if [ "$VIDEO_COUNT" -gt 1 ]; then
    echo ""
    echo "========================================"
    echo "[WARNING] Multiple video benchmarks detected!"
    echo "It is recommended to run video benchmarks ONE AT A TIME."
    echo "Example:"
    echo "  TASKS=\"videomme\" bash $0"
    echo "  TASKS=\"perceptiontest_val_mc\" bash $0"
    echo "========================================"
    echo ""
fi

# Step 1: Check if any task is video task
if is_video_task "$TASKS"; then
    echo "[INFO] Detected VIDEO task(s)"
    
    # Step 2: If codec model, search for offline assets
    if [ "$USE_OFFLINE_CODEC" = "1" ]; then
        echo "[INFO] Codec model detected, searching for precomputed assets..."
        OFFLINE_ROOTS=$(setup_offline_codec "$TASKS")
        
        if [ -n "$OFFLINE_ROOTS" ]; then
            echo "----------------------------------------"
            echo "Mode: CODEC (precomputed mosaic + positions)"
            echo "LLAVA_CODEC_OFFLINE_ROOT=$OFFLINE_ROOTS"
            echo "----------------------------------------"
            export LLAVA_CODEC_USE_OFFLINE=1
            export LLAVA_CODEC_OFFLINE_ROOT="$OFFLINE_ROOTS"
        else
            echo "----------------------------------------"
            echo "[ERROR] Codec model but no offline assets found!"
            echo "Please run: bash scripts/precompute_codec_patch/preprocess_video_benchmark.sh <task>"
            echo "----------------------------------------"
            exit 1
        fi
    else
        echo "----------------------------------------"
        echo "Mode: FRAME EXTRACTION (8 frames per video)"
        echo "----------------------------------------"
        unset LLAVA_CODEC_USE_OFFLINE
        unset LLAVA_CODEC_OFFLINE_ROOT
    fi
else
    echo "[INFO] Detected IMAGE task(s) only"
    echo "----------------------------------------"
    echo "Mode: Standard image evaluation"
    echo "----------------------------------------"
    unset LLAVA_CODEC_USE_OFFLINE
    unset LLAVA_CODEC_OFFLINE_ROOT
fi
echo "========================================"
# ================================================

# Run the evaluation script with the specified parameters
python -m accelerate.commands.launch \
    --main_process_port=$RUN_PORT \
    --num_processes=$NUM_GPUS \
    -m lmms_eval \
    --model llava_ov_encoder \
    --model_args pretrained=$MODEL_PATH,conv_template=$CONV_TEMPLATE \
    --tasks $TASKS \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix ${MODEL_NAME}_$(date +%Y%m%d) \
    --output_path ./eval_log/${MODEL_NAME}/
