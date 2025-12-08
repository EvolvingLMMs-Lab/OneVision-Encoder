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

LLM_VERSION="/vlm/pretrain_models/Qwen/Qwen2.5-7B-Instruct"
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="/video_vit/pretrain_models/deepglint/hevc/hevc_vit_ocr_packing_12_06_00068000_l14_flash_attn"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"
DATA_ROOT="/vlm/data/train_images/LLaVA-NeXT-Data"


PROJECTOR_NAME="hevc_vit_ocr_packing_flashattn_qwen25_select_layer_m2"
PROMPT_VERSION="qwen_1_5"


BASE_RUN_NAME="llavanext-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-mlp2x_gelu-pretrain_blip558k-finetune_llavanext780k-select_layer_m2"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

deepspeed --hostfile host_80 \
    --master_port 65534 \
    llava/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path ${LLM_VERSION} \
    --version ${PROMPT_VERSION} \
    --data_path ${DATA_ROOT}/llava_next_raw_format/llava_next_raw_format_processed.json \
    --image_folder ${DATA_ROOT}/llava_next_raw_format \
    --pretrain_mm_mlp_adapter /video_vit/xiangan/checkpoint_llava_next/projectors/${PROJECTOR_NAME}/mm_projector.bin \
    --mm_tunable_parts mm_vision_tower,mm_mlp_adapter,mm_language_model \
    --mm_vision_tower_lr 2e-6 \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio anyres \
    --image_grid_pinpoints "[(448, 896), (896, 448), (896, 896), (1344, 448), (448, 1344)]" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name $BASE_RUN_NAME \
    --output_dir "/video_vit/xiangan/checkpoint_llava_next/${BASE_RUN_NAME}" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 32768 \
    --dataloader_num_workers 2 \
    --lazy_preprocess True \
    --report_to wandb \
    --dataloader_drop_last True
