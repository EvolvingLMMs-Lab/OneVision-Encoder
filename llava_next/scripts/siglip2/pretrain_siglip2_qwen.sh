export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO
export PYTHONPATH=$(pwd)
export http_proxy=http://172.16.5.77:8889
export https_proxy=http://172.16.5.77:8889
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# export CUDA_VISIBLE_DEVICES=4,5,6,7


export RANK=0
export NUM_GPUS=8  # 每个节点使用的 GPU 数量
export NNODES=1    # 节点总数
export ADDR="localhost"  # 主节点地址
export PORT=29502  # 主节点端口
PROMPT_VERSION="qwen_1_5"

LLM_VERSION="/vlm/pretrain_models/Qwen/Qwen2.5-1.5B-Instruct"
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="/vlm/pretrain_models/SigLIP2/siglip2-so400m-patch16-naflex"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

############### Pretrain ################

PROMPT_VERSION=plain

BASE_RUN_NAME="llavanext-siglip2packing_-2hidnorm-qwen2.5-1.5b-instruct-pretrain_blip558k_plain-1220"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node="${NUM_GPUS}" --nnodes="${NNODES}" --node_rank="${RANK}" --master_addr="${ADDR}" --master_port="${PORT}" \
    llava/train/train_mem.py \
    --deepspeed scripts/zero2.json \
    --model_name_or_path ${LLM_VERSION} \
    --version ${PROMPT_VERSION} \
    --data_path /mnt/vlmdata/data/pretrain_data/blip_laion_cc_sbu_558k.json \
    --image_folder /mnt/vlmdata/data/train_images/LLaVA-Pretrain/images \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_tunable_parts="mm_mlp_adapter" \
    --mm_vision_select_layer -2 \
    --mm_projector_type mlp2x_gelu \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir /vlm/yinxie/code/checkpoints/projectors/${BASE_RUN_NAME} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
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

# You can delete the sdpa attn_implementation if you want to use flash attn