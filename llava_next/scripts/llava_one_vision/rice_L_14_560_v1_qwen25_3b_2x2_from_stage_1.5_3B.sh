#!/bin/bash

# ==============================================================================
#                            Configuration
# ==============================================================================

# --- Environment Setup ---
# Set the Python path to the current directory
export PYTHONPATH=$(pwd)

# (Optional) Multi-node/GPU environment variables - uncomment to use
# export OMP_NUM_THREADS=8
# export NCCL_IB_DISABLE=0
# export NCCL_IB_GID_INDEX=3
# export NCCL_SOCKET_IFNAME=eth0
# export NCCL_DEBUG=INFO
# export NUM_GPUS=8
# export NNODES=1
# export RANK=0
# export ADDR="localhost"
# export PORT="29500"

# --- Core Model & Data Configuration ---
# Paths to the pre-trained models and initial checkpoint
LLM_VERSION="/vlm/pretrain_models/Qwen/Qwen2.5-3B-Instruct"
VISION_MODEL_VERSION="/vlm/xiangan/pretrain_models/deepglint/rice-vit-large-patch14-560-v1"
MODEL_CHECKPOINT="checkpoints/emova-_vlm_xiangan_pretrain_models_deepglint_rice-vit-large-patch14-560-v1-_vlm_pretrain_models_Qwen_Qwen2.5-3B-Instruct-mlp2x_gelu-pretrain_7M_vqa_stage_1_5/"

# Specific dataset for this training run
TRAIN_DATA_PATH="/vlm/data/train_images/lmms-lab/LLaVA-OneVision-Data/llava_format.json"
TRAIN_IMAGE_FOLDER="/" # Root directory for images, as paths in JSON are absolute

# --- Run & Naming Configuration ---
# Clean up model version strings for use in names (replaces '/' with '_')
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

# Define the type of projector and the prompt version
mm_projector_type="patch_merger"
PROMPT_VERSION="qwen_1_5"

# Construct a descriptive name for the training run and define the output directory
BASE_RUN_NAME="LLaVA-OV-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-${mm_projector_type}-Align-EMOVA3.5M-Finetune-OV3.5M-14nodes-4x4"
OUTPUT_DIR="./checkpoints/${BASE_RUN_NAME}"

# --- Training Hyperparameters ---
EPOCHS=1
PER_DEVICE_BATCH_SIZE=2
PER_DEVICE_EVAL_BATCH_SIZE=4
GRAD_ACCUM_STEPS=1
LEARNING_RATE=1e-5
WARMUP_RATIO=0.03
SAVE_STEPS=500
MAX_SEQ_LEN=12000
DATALOADER_WORKERS=2

# ==============================================================================
#                         Pre-flight Check
# ==============================================================================
# Print key configuration parameters for debugging before starting the run
echo "================ Training Configuration ================"
echo "RUN NAME:        ${BASE_RUN_NAME}"
echo "MODEL CHECKPOINT:${MODEL_CHECKPOINT}"
echo "DATA PATH:       ${TRAIN_DATA_PATH}"
echo "IMAGE FOLDER:    ${TRAIN_IMAGE_FOLDER}"
echo "BATCH SIZE:      ${PER_DEVICE_BATCH_SIZE} (per device)"
echo "LEARNING RATE:   ${LEARNING_RATE}"
echo "MAX SEQ LENGTH:  ${MAX_SEQ_LEN}"
echo "OUTPUT DIR:      ${OUTPUT_DIR}"
echo "========================================================"


# ==============================================================================
#                         Training Execution
# ==============================================================================
# Launch the multi-node training script using DeepSpeed
deepspeed --hostfile hostfile_12nodes.txt \
    --master_port 65534 \
    llava/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path "${MODEL_CHECKPOINT}" \
    --version "${PROMPT_VERSION}" \
    --data_path "${TRAIN_DATA_PATH}" \
    --image_folder "${TRAIN_IMAGE_FOLDER}" \
    --vision_tower "${VISION_MODEL_VERSION}" \
    --mm_projector_type "${mm_projector_type}" \
    --mm_tunable_parts "mm_vision_tower,mm_mlp_adapter,mm_language_model" \
    --mm_vision_tower_lr 2e-6 \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio "anyres" \
    --image_grid_pinpoints "[(560, 560), (1120, 560), (1680, 560), (2240, 560), (560, 1120), (1120, 1120), (1680, 1120), (2240, 1120), (560, 1680), (1120, 1680), (1680, 1680), (2240, 1680), (560, 2240), (1120, 2240), (1680, 2240), (2240, 2240)]" \
    --mm_patch_merge_type "spatial_unpad" \
    --group_by_modality_length True \
    --bf16 True \
    --run_name "${BASE_RUN_NAME}" \
    --output_dir "${OUTPUT_DIR}" \
    --num_train_epochs "${EPOCHS}" \
    --per_device_train_batch_size "${PER_DEVICE_BATCH_SIZE}" \
    --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRAD_ACCUM_STEPS}" \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps "${SAVE_STEPS}" \
    --save_total_limit 1 \
    --learning_rate "${LEARNING_RATE}" \
    --weight_decay 0. \
    --warmup_ratio "${WARMUP_RATIO}" \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length "${MAX_SEQ_LEN}" \
    --dataloader_num_workers "${DATALOADER_WORKERS}" \
    --lazy_preprocess True \
    --dataloader_drop_last True \
    --report_to "wandb" \
    --torch_compile True \
    --torch_compile_backend "inductor"
