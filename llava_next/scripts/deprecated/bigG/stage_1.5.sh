#!/bin/bash

# =========================================================================
# 1. 环境与路径配置 (Environment & Path Configuration)
# =========================================================================
export PYTHONPATH=$(pwd)

# --- 模型与数据路径定义 ---
LLM_VERSION="/vlm/pretrain_models/Qwen/Qwen2.5-7B-Instruct"
VISION_MODEL_VERSION="/vlm/xiangan/pretrain_models/deepglint/rice-vit-bigG-patch14-560-v2"
PRETRAINED_PROJECTOR="/vlm/xiangan/checkpoints_rice_vl/projectors/_vlm_xiangan_pretrain_models_deepglint_rice-vit-bigG-patch14-560-v2-_vlm_pretrain_models_Qwen_Qwen2.5-7B-Instruct_patch_merger_558k_stage_1/mm_projector.bin"

# --- 实验命名 ---
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"
BASE_RUN_NAME="emova-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-patch_merger-pretrain_7M_vqa_stage_1_5"


# =========================================================================
# 2. 启动训练 (Launch Training)
# =========================================================================
echo "Starting Training Run: ${BASE_RUN_NAME}"

deepspeed --hostfile hostfile_14nodes.txt \
    llava/train/train_mem.py \
    --model_name_or_path ${LLM_VERSION} \
    --vision_tower ${VISION_MODEL_VERSION} \
    --pretrain_mm_mlp_adapter "${PRETRAINED_PROJECTOR}" \
    --mm_tunable_parts mm_vision_tower,mm_mlp_adapter,mm_language_model \
    --version qwen_1_5 \
    --mm_vision_select_layer -2 \
    --mm_projector_type patch_merger \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --mm_vision_select_feature patch \
    --data_path /vlm/data/train_images/Emova-ollm/emova-alignment-7m/llava_format_vqa.json \
    --image_folder / \
    --lazy_preprocess True \
    --image_aspect_ratio anyres \
    --image_grid_pinpoints "[(560, 560), (1120, 560), (560, 1120), (1120, 1120), (1680, 560), (560, 1680)]" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --output_dir /vlm/xiangan/checkpoints_rice_vl/${BASE_RUN_NAME} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --mm_vision_tower_lr 2e-6 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4500 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --deepspeed scripts/zero3.json
