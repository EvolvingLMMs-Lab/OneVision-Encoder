export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0
export PYTHONPATH=$(pwd)

# Monitor if a specific PID exists, if so wait for 1 minute
PID_TO_MONITOR=493372

if [ -z "$PID_TO_MONITOR" ]; then
    echo "Usage: $0 <pid_to_monitor>"
    exit 1
fi

# Monitor the given PID, sleep if it exists
while kill -0 "$PID_TO_MONITOR" 2>/dev/null; do
    echo "PID $PID_TO_MONITOR is still running. Waiting 2 minutes..."
    sleep 120
done

LLM_VERSION="Qwen/Qwen3-4B-Instruct-2507"
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="lmms-lab/onevision-encoder-large"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"
export WANDB_MODE=disabled


export PORT=29502  
PROMPT_VERSION="qwen_1_5"

BASE_RUN_NAME="./checkpoints/date$(date +%m%d)_llavanext-onevision-encoder-large_-2hid-qwen3-4b-sigvid"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

mkdir -p $BASE_RUN_NAME
cp $0 $BASE_RUN_NAME/$(basename $0)

deepspeed --master_port 65535 \
    llava/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path ${LLM_VERSION} \
    --version ${PROMPT_VERSION} \
    --data_path video_800k_mixed_codec8f_or_8fkeep.jsonl \
    --image_folder root \
    --pretrain_mm_mlp_adapter="checkpoints/projectors/date$(date +%m%d)_llavanext-onevision-encoder-large_-2hid-qwen3-4b-instruct-pretrain_blip558k_plain/mm_projector.bin" \
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
    --gradient_accumulation_steps 16 \
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
