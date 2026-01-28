export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0
export PYTHONPATH=$(pwd)


export RANK=0
export NUM_GPUS=8
export NNODES=1
export ADDR="localhost"
PROMPT_VERSION="qwen_1_5"

LLM_VERSION="Qwen/Qwen3-4B-Instruct-2507"
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="lmms-lab/onevision-encoder-large"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

PROMPT_VERSION=plain

BASE_RUN_NAME="date$(date +%m%d)_llavanext-onevision-encoder-large_-2hid-qwen3-4b-instruct-pretrain_blip558k_plain"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

deepspeed --master_port 65535 \
    llava/train/train_mem.py \
    --deepspeed scripts/zero2_noopt.json \
    --model_name_or_path ${LLM_VERSION} \
    --version ${PROMPT_VERSION} \
    --data_path blip_laion_cc_sbu_558k.json \
    --image_folder image_root \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_tunable_parts="mm_mlp_adapter" \
    --mm_vision_select_layer -2 \
    --mm_projector_type mlp2x_gelu \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir checkpoints/projectors/${BASE_RUN_NAME} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --mm_patch_merge_type flat \
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
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --run_name $BASE_RUN_NAME