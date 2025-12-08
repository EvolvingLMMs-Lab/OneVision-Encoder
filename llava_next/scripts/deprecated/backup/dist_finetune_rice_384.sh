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
VISION_MODEL_VERSION="/vlm/xiangan/pretrain_models/deepglint/RICE/rice-vit-large-patch14-378"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"
DATA_ROOT="/vlm/data/train_images"
PROJECTOR_NAME="llavanext-_vlm_xiangan_pretrain_models_deepglint_RICE_rice-vit-large-patch14-378-_vlm_pretrain_models_Qwen_Qwen2.5-7B-Instruct-mlp2x_gelu-pretrain_blip558k_plain"

PROMPT_VERSION="qwen_1_5"


BASE_RUN_NAME="llavanext-RICE-vit-l-14-378px-qwen2.5-mlp2x_gelu-pretrain_blip558k-finetune_llavanext780k-8nodes"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

deepspeed --hostfile hostfile.txt \
    --master_port 65534 \
    llava/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path ${LLM_VERSION} \
    --version ${PROMPT_VERSION} \
    --data_path ${DATA_ROOT}/llava_next_raw_format/llava_next_raw_format_processed.json \
    --image_folder ${DATA_ROOT}/llava_next_raw_format \
    --pretrain_mm_mlp_adapter ./checkpoints/projectors/${PROJECTOR_NAME}/mm_projector.bin \
    --mm_tunable_parts mm_vision_tower,mm_mlp_adapter,mm_language_model \
    --mm_vision_tower_lr 2e-6 \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio anyres \
    --image_grid_pinpoints "[(378, 756), (756, 378), (756, 756), (1134, 378), (378, 1134)]" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name $BASE_RUN_NAME \
    --output_dir "./checkpoints/${BASE_RUN_NAME}" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --report_to wandb \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True 

