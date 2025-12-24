export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0
export PYTHONPATH=$(pwd)
export CUDA_VISIBLE_DEVICES=6,7

LLM_VERSION="/vlm/pretrain_models/Qwen/Qwen2.5-1.5B-Instruct"
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="/video_vit/pretrain_models/deepglint/onevision-encoder-large"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"
export WANDB_MODE=disabled


export PORT=29502
PROMPT_VERSION="qwen_1_5"

BASE_RUN_NAME="./checkpoints/date1220_llavanext-llavavit_-2hid-qwen2.5-1.5b-sigvid-8nodes"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

mkdir -p $BASE_RUN_NAME
cp $0 $BASE_RUN_NAME/$(basename $0)

deepspeed --master_port 65535 \
    llava/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path ${LLM_VERSION} \
    --version ${PROMPT_VERSION} \
    --data_path /rice_vl/llava_video_8f_imgs_1027/video_800k_llavanextsig_740k_shuffled.jsonl \
    --image_folder /rice_vl/llava_video_8f_imgs_1027 \
    --pretrain_mm_mlp_adapter="/vlm/yinxie/code/checkpoints/projectors/llavanext-llavavit_-2hid-qwen2.5-1.5b-instruct-pretrain_blip558k_plain-1220-dist/mm_projector.bin" \
    --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model" \
    --mm_vision_tower_lr=2e-6 \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio anyres \
    --image_grid_pinpoints "[(574, 1120), (1120, 574), (1120, 1120), (1694, 574), (574, 1694)]" \
    --mm_patch_merge_type flat \
    --bf16 True \
    --run_name $BASE_RUN_NAME \
    --output_dir $BASE_RUN_NAME \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 20 \
    --learning_rate 1e-5 \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 321120 \
    --gradient_checkpointing True \
    --dataloader_num_workers 1 \
    --lazy_preprocess True \
    --dataloader_drop_last True \
    --attn_implementation flash_attention_2 | tee $BASE_RUN_NAME/train.log

# You can delete the sdpa attn_implementation if you want to use flash attn
