#!/bin/bash
# ============================================================================
# Common Configuration Script - for attentive_probe_codec evaluation
# Usage: Source this file in model scripts, then call run_attentive_probe_codec
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
    elif [[ "$dataset" == "perception_test" ]]; then
        echo 30
    else
        echo 10
    fi
}

# ============================================================================
# Get codec parameters based on dataset name
# Args: $1 - dataset name
# Returns: Sets CODEC_MV_COMPENSATE, CODEC_STATIC_ABS_THRESH, CODEC_STATIC_REL_THRESH
# ============================================================================
get_codec_params() {
    local dataset="$1"
    if [[ "$dataset" == "diving48" || "$dataset" == "perception_test" ]]; then
        # Parameters for diving48 and perception_test
        CODEC_MV_COMPENSATE="similarity"
        CODEC_STATIC_ABS_THRESH="126"
        CODEC_STATIC_REL_THRESH="0.38"
    else
        # Parameters for other datasets
        CODEC_MV_COMPENSATE="similarity"
        CODEC_STATIC_ABS_THRESH="116"
        CODEC_STATIC_REL_THRESH="0.55"
    fi
}

# ============================================================================
# Run attentive_probe_codec evaluation
# Required variables to set before calling:
#   - MODEL_FAMILY: model family (required)
#   - MODEL_NAME: model name (required)
#   - MODEL_WEIGHT: model weight path (optional, default "NULL")
#   - FRAMES_TOKEN_NUM: token count (optional, default 196)
#   - EMBEDDING_SIZE: embedding dimension (optional, default 768)
#   - INPUT_SIZE: input size (optional, not passed if unset)
#   - NUM_FRAMES: number of frames (optional, not passed if unset)
#   - K_keep: number of top-K patches to keep (optional, default 2048)
#   - DATASETS: dataset array (optional, uses DEFAULT_DATASETS if unset/empty)
#   - REPORT_DIR_SUFFIX: report directory suffix (optional, e.g. "_64frames_codec")
# ============================================================================
run_attentive_probe_codec() {
    # Set default values
    MODEL_WEIGHT="${MODEL_WEIGHT:-NULL}"
    FRAMES_TOKEN_NUM="${FRAMES_TOKEN_NUM:-196}"
    EMBEDDING_SIZE="${EMBEDDING_SIZE:-768}"
    K_keep="${K_keep:-2048}"
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
        
        # Get codec-specific parameters for this dataset
        get_codec_params "$DATASET"

        echo "DATASET=$DATASET, BATCH_SIZE=$BATCH_SIZE"
        echo "Codec params: mv_compensate=${CODEC_MV_COMPENSATE}, static_abs_thresh=${CODEC_STATIC_ABS_THRESH}, static_rel_thresh=${CODEC_STATIC_REL_THRESH}"

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
            attentive_probe_codec.py \
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
            --K_keep ${K_keep} \
            --mv_compensate ${CODEC_MV_COMPENSATE} \
            --static_abs_thresh ${CODEC_STATIC_ABS_THRESH} \
            --static_rel_thresh ${CODEC_STATIC_REL_THRESH} \
            --static_fallback 1 \
            ${EXTRA_ARGS}

        echo "Finished testing ${DATASET}"
        echo ""
    done
}
