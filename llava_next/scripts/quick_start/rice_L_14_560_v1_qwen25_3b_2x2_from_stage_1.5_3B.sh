export PYTHONPATH=$(pwd)

# 基本配置参数 - 更清晰地组织
LLM_VERSION="/vlm/pretrain_models/Qwen/Qwen2.5-3B-Instruct"
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="/vlm/xiangan/pretrain_models/deepglint/rice-vit-large-patch14-560-v1"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"
DATA_ROOT="/vlm/data/train_images/LLaVA-NeXT-Data"
PROMPT_VERSION="qwen_1_5"

# 模型配置
MODEL_PATH="/vlm/xiangan/unicom_unit/checkpoints/emova-_vlm_xiangan_pretrain_models_deepglint_rice-vit-large-patch14-560-v1-_vlm_pretrain_models_Qwen_Qwen2.5-3B-Instruct-mlp2x_gelu-pretrain_7M_vqa_stage_1_5/"
mm_projector_type="patch_merger"
BASE_RUN_NAME="llava_next-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-${mm_projector_type}-pretrain_blip558k_emova_alignment_7m_finetune_llavanext7805_4x4"
OUTPUT_DIR="./checkpoints/${BASE_RUN_NAME}"

# 训练参数 - 便于调整
EPOCHS=1
BATCH_SIZE=4
EVAL_BATCH_SIZE=4
GRAD_ACCUM=4
LR=1e-5
WARMUP_RATIO=0.03
SAVE_STEPS=500
MAX_SEQ_LEN=8192
NUM_WORKERS=4  # 增加了数据加载器的线程数

# 打印关键配置参数用于调试
echo "================ 训练配置 ================"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"
echo "MODEL_PATH: ${MODEL_PATH}"
echo "DATA_PATH: ${DATA_ROOT}/llava_next_raw_format/llava_next_raw_format_processed.json"
echo "BATCH_SIZE: ${BATCH_SIZE} (x 节点数 x GPU数/节点)"
echo "LEARNING_RATE: ${LR}"
echo "MAX_SEQ_LEN: ${MAX_SEQ_LEN}"
echo "========================================="

# 启动训练 - 使用localhost替代主机文件
deepspeed --include localhost \
    --master_port 65534 \
    llava/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path ${MODEL_PATH} \
    --version ${PROMPT_VERSION} \
    --data_path ${DATA_ROOT}/llava_next_raw_format/llava_next_raw_format_processed.json \
    --image_folder ${DATA_ROOT}/llava_next_raw_format \
    --mm_tunable_parts mm_vision_tower,mm_mlp_adapter,mm_language_model \
    --mm_vision_tower_lr 2e-6 \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type ${mm_projector_type} \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio anyres \
    --image_grid_pinpoints "[(560, 560), (1120, 560), (1680, 560), (560, 1120), (1120, 1120), (560, 1680)]" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name ${BASE_RUN_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs ${EPOCHS} \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --per_device_eval_batch_size ${EVAL_BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps ${SAVE_STEPS} \
    --save_total_limit 1 \
    --learning_rate ${LR} \
    --weight_decay 0. \
    --warmup_ratio ${WARMUP_RATIO} \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length ${MAX_SEQ_LEN} \
    --dataloader_num_workers ${NUM_WORKERS} \
    --lazy_preprocess True \
    --report_to wandb \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True