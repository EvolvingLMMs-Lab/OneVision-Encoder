#!/bin/bash

# =================================================================
#            Multimodal Model Pre-training Script
# =================================================================
#
# This script handles the setup and execution of a DeepSpeed-based
# pre-training job for a large multimodal model.
#

# --- 1. Environment & Node Configuration ---
# Sets up the environment for distributed training.
export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO
export NUM_GPUS=8
export NNODES=1
export RANK=0
export ADDR="localhost"
export PORT="29500"
export PYTHONPATH=$(pwd)


# --- 2. Model & Data Paths ---
# Define the locations for models and datasets.
LLM_VERSION="/vlm/pretrain_models/Qwen/Qwen2.5-7B-Instruct"
VISION_MODEL_VERSION="/vlm/xiangan/pretrain_models/deepglint/rice-vit-bigG-patch14-560-v2"
DATA_ROOT="/vlm/data/pretrain_data"


# --- 3. Run Configuration ---
# Create a unique run name based on model versions for easy tracking.
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"
PROMPT_VERSION=plain
BASE_RUN_NAME="${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}_patch_merger_558k_stage_1"

echo "================================================="
echo "Starting Pre-training Run: ${BASE_RUN_NAME}"
echo "================================================="


# --- 4. DeepSpeed Training Launch ---
# Execute the training script with specified hyperparameters.
deepspeed --hostfile hostfile_14nodes.txt llava/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    \
    --model_name_or_path ${LLM_VERSION} \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type patch_merger \
    --mm_tunable_parts mm_mlp_adapter \
    --mm_vision_select_layer -2 \
    --mm_vision_select_feature patch \
    \
    --data_path ${DATA_ROOT}/blip_laion_cc_sbu_558k.json \
    --image_folder /vlm/data/train_images/LLaVA-Pretrain \
    --version ${PROMPT_VERSION} \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    \
    --bf16 True \
    --output_dir /vlm/xiangan/checkpoints_rice_vl/projectors/${BASE_RUN_NAME} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --save_steps 50000 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 2 \
    --lazy_preprocess True

echo "================================================="
echo "Run ${BASE_RUN_NAME} Finished."
echo "================================================="
