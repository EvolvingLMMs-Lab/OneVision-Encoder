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
export PYTHONPATH=$(pwd)

LLM_VERSION="/vlm/pretrain_models/Qwen/Qwen2.5-3B-Instruct"
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="/vlm/xiangan/pretrain_models/deepglint/rice-vit-large-patch14-560-v1"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"
DATA_ROOT="/vlm/data/pretrain_data"

############### Pretrain ################

PROMPT_VERSION=qwen_1_5

BASE_RUN_NAME="emova-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-mlp2x_gelu-pretrain_7M_vqa_stage_1_5"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

deepspeed --hostfile list_host_v12 \
    llava/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path ${LLM_VERSION} \
    --version ${PROMPT_VERSION} \
    --data_path /vlm/data/train_images/Emova-ollm/emova-alignment-7m/llava_format_vqa.json \
    --image_folder / \
    --vision_tower ${VISION_MODEL_VERSION} \
    --pretrain_mm_mlp_adapter "/vlm/xiangan/unicom_unit/checkpoints/projectors/emova-_vlm_xiangan_pretrain_models_deepglint_rice-vit-large-patch14-560-v1-_vlm_pretrain_models_Qwen_Qwen2.5-3B-Instruct-mlp2x_gelu-pretrain_blip558k_plain_stage_1/mm_projector.bin" \
    --mm_tunable_parts mm_vision_tower,mm_mlp_adapter,mm_language_model \
    --mm_vision_select_layer -2 \
    --mm_projector_type patch_merger \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio anyres \
    --image_grid_pinpoints "[(560, 560), (1120, 560), (560, 1120), (1120, 1120), (1680, 560), (560, 1680)]" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --output_dir ./checkpoints/${BASE_RUN_NAME} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --mm_vision_tower_lr 1e-5 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 5000 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --mm_vision_select_feature patch