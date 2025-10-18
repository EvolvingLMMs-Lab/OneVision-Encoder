TRAIN_DATA_ROOT_PATH="${TRAIN_DATA_ROOT_PATH:-/video_vit/fewshot_video/ActionRecognition}"
TRAIN_DATA_CSV_PATH="${TRAIN_DATA_CSV_PATH:-/video_vit/fewshot_video/ActionRecognition}"
VAL_DATA_ROOT_PATH="${VAL_DATA_ROOT_PATH:-/video_vit/eval_data/val/}"
VAL_DATA_CSV_PATH="${VAL_DATA_CSV_PATH:-/video_vit/eval_data/annotation/}"

# ---------------- Distributed defaults -----------------
NUM_GPUS="${NUM_GPUS:-8}"        # 每节点 GPU 数
NNODES="${NNODES:-1}"            # 总节点数
RANK="${RANK:-0}"                # 当前节点 rank
ADDR="${ADDR:-127.0.0.1}"        # 主节点地址 (MASTER_ADDR)
PORT="${PORT:-32509}"            # 主节点端口 (MASTER_PORT)



# ============== Output & Model Defaults ==============
OUTPUT="${OUTPUT:-output}"              # 训练/评估产物输出目录
MODEL_FAMILY="${MODEL_FAMILY:-NULL}"           # 逻辑模型名称（自定义 tag）
MODEL_NAME="${MODEL_NAME:-NULL}"
CKPT_PATH="${CKPT_PATH:-model.pt}"            # 微调初始权重 / 预训练 ckpt 路径

EMBEDDING_SIZE="${EMBEDDING_SIZE:-1024}"
NUM_FRAMES="${NUM_FRAMES:-8}"
NUM_EPOCH="${NUM_EPOCH:-40}"
INPUT_SIZE="${INPUT_SIZE:-224}"
TUBELET_SIZE="${TUBELET_SIZE:-1}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-0.0001}"
EVAL_FREQ="${EVAL_FREQ:-10}"

# 如果外部没传，则给默认
DATASETS="${DATASETS:-ssv2}"
# 去掉所有空格（防止有人写成 "k400, ssv2,k600"）
DATASETS="${DATASETS// /}"
# 拆成数组
IFS=',' read -r -a DATASET_ARRAY <<< "$DATASETS"

for SEED in 1
do
    # for DATASET in ssv2 k400 k600 k700 hmdb51 ucf101 epic_verb epic_noun perception_test diving48 CharadesEgo  CharadesEgo_v1_only1st CharadesEgo_v1_only3rd
    for DATASET in "${DATASET_ARRAY[@]}";
    do
        echo "当前 SEED=$SEED, DATASET=$DATASET"
        for NUM_SHOTS in 50
        do
            echo "SEED: $SEED"
            echo "DATASET: $DATASET"
            echo "NUM_SHOTS: $NUM_SHOTS"

            FLASH=1 torchrun --nproc_per_node="${NUM_GPUS}" --nnodes="${NNODES}" \
                --node_rank="${RANK}" --master_addr="${ADDR}" --master_port="${PORT}" \
                video_attentive_probe_all/ac_export_feature_and_attentive_probe_latest.py \
                --embedding_size ${EMBEDDING_SIZE} \
                --dataset ${DATASET} \
                --default_epoch ${NUM_EPOCH} \
                --seed ${SEED} \
                --num_shots ${NUM_SHOTS} \
                --num_step 8 \
                --train_data_root_path ${TRAIN_DATA_ROOT_PATH} \
                --train_data_csv_path ${TRAIN_DATA_CSV_PATH} \
                --val_data_root_path ${VAL_DATA_ROOT_PATH} \
                --val_data_csv_path ${VAL_DATA_CSV_PATH} \
                --save_report ${OUTPUT} \
                --batch_size ${BATCH_SIZE} \
                --model_family ${MODEL_FAMILY} \
                --model_name ${MODEL_NAME} \
                --ckpt_path ${CKPT_PATH} \
                --num_frames ${NUM_FRAMES} \
                --input_size ${INPUT_SIZE} \
                --tubelet_size ${TUBELET_SIZE} \
                --default_lr_list ${LR} \
                --eval_freq ${EVAL_FREQ}
        done
    done
done

# Due to the small dataset size, the following dataset raises errors when using 8 GPUs with a large batch size. 
# export NUM_GPUS=1
# export NNODES=1
# export RANK=0
# export ADDR="127.0.0.1"
# export PORT="32509"


# TRAIN_DATA_ROOT_PATH=/path/to/train/video
# TRAIN_DATA_CSV_PATH=/path/to/train/video
# VAL_DATA_ROOT_PATH=/path/to/val/video
# VAL_DATA_CSV_PATH=/path/to/val/csv
# OUTPUT=/path/to/output
# MODEL_NAME='umt'

# FINETUNE=/path/to/ckpt
# model='vit_large_patch16_224'
# EMBEDDING_SIZE=1024
# PATCH_SIZE=16
# NUM_FRAMES=8
# INPUT_SIZE=224
# TUBELET_SIZE=1
# BATCH_SIZE=32

# for SEED in 1
# do
#     for DATASET in RareAct Drone_Action
#     do
#         for NUM_SHOTS in 50
#         do
#             echo "SEED: $SEED"
#             echo "DATASET: $DATASET"
#             echo "NUM_SHOTS: $NUM_SHOTS"

#             FLASH=1 torchrun --nproc_per_node="${NUM_GPUS}" --nnodes="${NNODES}" \
#                 --node_rank="${RANK}" --master_addr="${ADDR}" --master_port="${PORT}" \
#                 ac_export_feature_and_attentive_probe.py \
#                 --embedding_size ${EMBEDDING_SIZE} \
#                 --data_set ${DATASET} \
#                 --seed ${SEED} \
#                 --num_shots ${NUM_SHOTS} \
#                 --num_step 8 \
#                 --train_data_root_path ${TRAIN_DATA_ROOT_PATH} \
#                 --train_data_csv_path ${TRAIN_DATA_CSV_PATH} \
#                 --val_data_root_path ${VAL_DATA_ROOT_PATH} \
#                 --val_data_csv_path ${VAL_DATA_CSV_PATH} \
#                 --save_report ${OUTPUT} \
#                 --batch_size ${BATCH_SIZE} \
#                 --model_name ${MODEL_NAME} \
#                 --model ${model} \
#                 --finetune ${FINETUNE} \
#                 --num_frames ${NUM_FRAMES} \
#                 --input_size ${INPUT_SIZE} \
#                 --tubelet_size ${TUBELET_SIZE} \
#                 --patch_size ${PATCH_SIZE}
#         done
#     done
# done
