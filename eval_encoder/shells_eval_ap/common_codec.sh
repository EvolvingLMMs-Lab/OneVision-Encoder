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
# Run attentive_probe_codec evaluation
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
#   - K_KEEP: number of top-K patches to keep as visible (optional, default 2048)
#   - PATCH_SIZE: patch size for residual patching (optional, default 14)
#   - CACHE_DIR: directory to store/load visible_indices cache (optional)
#   - MV_COMPENSATE: MV global compensation (optional, default "similarity")
#   - MV_USE_INCONSISTENCY: use MV local variance flag (optional)
#   - MV_INCON_KSIZE: neighborhood size for MV inconsistency (optional, default 3)
#   - RES_USE_GRAD: use gradient-based residual energy flag (optional)
#   - CENTER_PRIOR: center Gaussian prior strength (optional, default 0.0)
#   - CENTER_SIGMA: center Gaussian sigma (optional, default 0.35)
#   - STATIC_FALLBACK: enable static-video hybrid fallback (optional, default 1)
#   - STATIC_ABS_THRESH: absolute low-energy threshold (optional, default 2.0)
#   - STATIC_REL_THRESH: relative contrast threshold (optional, default 0.15)
#   - STATIC_UNIFORM_FRAMES: uniformly-picked frames in hybrid fallback (optional, default 4)
# ============================================================================
run_attentive_probe_codec() {
    # Set default values
    MODEL_WEIGHT="${MODEL_WEIGHT:-NULL}"
    FRAMES_TOKEN_NUM="${FRAMES_TOKEN_NUM:-196}"
    EMBEDDING_SIZE="${EMBEDDING_SIZE:-768}"
    REPORT_DIR_SUFFIX="${REPORT_DIR_SUFFIX:-}"
    K_KEEP="${K_KEEP:-2048}"
    PATCH_SIZE="${PATCH_SIZE:-14}"
    MV_COMPENSATE="${MV_COMPENSATE:-similarity}"
    MV_INCON_KSIZE="${MV_INCON_KSIZE:-3}"
    CENTER_PRIOR="${CENTER_PRIOR:-0.0}"
    CENTER_SIGMA="${CENTER_SIGMA:-0.35}"
    STATIC_FALLBACK="${STATIC_FALLBACK:-1}"
    STATIC_ABS_THRESH="${STATIC_ABS_THRESH:-2.0}"
    STATIC_REL_THRESH="${STATIC_REL_THRESH:-0.15}"
    STATIC_UNIFORM_FRAMES="${STATIC_UNIFORM_FRAMES:-4}"

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
        echo "K_keep: ${K_KEEP}"
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
        if [[ -n "${CACHE_DIR}" ]]; then
            EXTRA_ARGS="${EXTRA_ARGS} --cache_dir ${CACHE_DIR}"
        fi
        if [[ -n "${MV_USE_INCONSISTENCY}" ]] && [[ "${MV_USE_INCONSISTENCY}" == "true" ]]; then
            EXTRA_ARGS="${EXTRA_ARGS} --mv_use_inconsistency"
        fi
        if [[ -n "${RES_USE_GRAD}" ]] && [[ "${RES_USE_GRAD}" == "true" ]]; then
            EXTRA_ARGS="${EXTRA_ARGS} --res_use_grad"
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
            --K_keep ${K_KEEP} \
            --patch_size ${PATCH_SIZE} \
            --mv_compensate ${MV_COMPENSATE} \
            --mv_incon_ksize ${MV_INCON_KSIZE} \
            --center_prior ${CENTER_PRIOR} \
            --center_sigma ${CENTER_SIGMA} \
            --static_fallback ${STATIC_FALLBACK} \
            --static_abs_thresh ${STATIC_ABS_THRESH} \
            --static_rel_thresh ${STATIC_REL_THRESH} \
            --static_uniform_frames ${STATIC_UNIFORM_FRAMES} \
            ${EXTRA_ARGS}

        echo "Finished testing ${DATASET}"
        echo ""
    done
}
