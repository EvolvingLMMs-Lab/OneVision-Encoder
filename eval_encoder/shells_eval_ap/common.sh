#!/bin/bash
# ============================================================================
# Common Configuration Script - for attentive_probe evaluation
# Usage: Source this file in model scripts, then call run_attentive_probe
# ============================================================================

# Environment setup
export PYTHONPATH=../

# Default dataset list (can be overridden in calling script)
DEFAULT_DATASETS=(
    "ssv2"
    "diving48"
    "perception_test"
    "epic_verb"
    "epic_noun"
    "hmdb51"
    "k400"
    "charadesego"
)

# ============================================================================
# Get batch size based on dataset name
# Args: $1 - dataset name
# ============================================================================
get_batch_size() {
    local dataset="$1"
    if [[ "$dataset" == "ssv2" || "$dataset" == "diving48" || "$dataset" == "perception_test" ]]; then
        echo 4
    elif [[ "$dataset" == "hmdb51" ]]; then
        echo 2
    else
        echo 16
    fi
}

# ============================================================================
# Get epochs based on dataset name
# Args: $1 - dataset name
# ============================================================================
get_epochs() {
    local dataset="$1"
    if [[ "$dataset" == "hmdb51" ]]; then
        echo 30
    elif [[ "$dataset" == "diving48" ]]; then
        echo 30
    else
        echo 10
    fi
}

# ============================================================================
# Run attentive_probe evaluation
# Required variables to set before calling:
#   - MODEL_FAMILY: model family (required)
#   - MODEL_NAME: model name (required)
#   - MODEL_WEIGHT: model weight path (optional, default "NULL")
#   - FRAMES_TOKEN_NUM: token count (optional, default 196)
#   - EMBEDDING_SIZE: embedding dimension (optional, default 768)
#   - INPUT_SIZE: input size (optional, not passed if unset)
#   - NUM_FRAMES: number of frames (optional, not passed if unset)
#   - DATASETS: dataset array (optional, uses DEFAULT_DATASETS if unset/empty)
#   - REPORT_DIR_SUFFIX: report directory suffix (optional, e.g. "_16frames")
# ============================================================================
run_attentive_probe() {
    # Set default values
    MODEL_WEIGHT="${MODEL_WEIGHT:-NULL}"
    FRAMES_TOKEN_NUM="${FRAMES_TOKEN_NUM:-196}"
    EMBEDDING_SIZE="${EMBEDDING_SIZE:-768}"
    REPORT_DIR_SUFFIX="${REPORT_DIR_SUFFIX:-}"

    # Use custom datasets or default datasets
    if [[ -z "${DATASETS+x}" ]] || [[ ${#DATASETS[@]} -eq 0 ]]; then
        DATASETS=("${DEFAULT_DATASETS[@]}")
    fi

    # Build report directory
    BASE_REPORT_DIR="result_attentive_probe/${MODEL_FAMILY}/${MODEL_NAME}${REPORT_DIR_SUFFIX}"

    # Loop through each dataset for testing
    for DATASET in "${DATASETS[@]}"; do
        BATCH_SIZE=$(get_batch_size "$DATASET")
        EPOCHS=$(get_epochs "$DATASET")

        echo "DATASET=$DATASET, BATCH_SIZE=$BATCH_SIZE"

        echo "========================================================"
        echo "Start testing dataset: ${DATASET}"
        echo "Model: ${MODEL_NAME}"
        echo "Batch Size: ${BATCH_SIZE}"
        echo "Report Dir: ${BASE_REPORT_DIR}/${DATASET}"
        echo "========================================================"

        # Build output directory
        SAVE_DIR="${BASE_REPORT_DIR}/${DATASET}"
        mkdir -p "$SAVE_DIR"

        # Build extra arguments
        EXTRA_ARGS=""
        if [[ -n "${INPUT_SIZE}" ]]; then
            EXTRA_ARGS="${EXTRA_ARGS} --input_size ${INPUT_SIZE}"
        fi
        if [[ -n "${NUM_FRAMES}" ]]; then
            EXTRA_ARGS="${EXTRA_ARGS} --num_frames ${NUM_FRAMES}"
        fi

        torchrun --nproc_per_node 8 --master_port 15555 \
            attentive_probe.py \
            --eval_freq 1 \
            --default_lr_list 0.0001 \
            --default_epoch "${EPOCHS}" \
            --batch_size ${BATCH_SIZE} \
            --default_weight_decay 0 \
            --dali_py_num_workers 8 \
            --model_family "${MODEL_FAMILY}" \
            --model_name "${MODEL_NAME}" \
            --model_weight "${MODEL_WEIGHT}" \
            --dataset "${DATASET}" \
            --save_report "${SAVE_DIR}" \
            --frames_token_num ${FRAMES_TOKEN_NUM} \
            --embedding_size ${EMBEDDING_SIZE} \
            ${EXTRA_ARGS}

        echo "Finished testing ${DATASET}"
        echo ""
    done
}
