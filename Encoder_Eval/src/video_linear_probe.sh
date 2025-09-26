export NUM_GPUS=8
export NNODES=1
export RANK=0
export ADDR="127.0.0.1"
export PORT="32509"
#pt=pretrain ppt=post-pretrain ft=finetune

TRAIN_DATA_ROOT_PATH=/path/to/train/video
TRAIN_DATA_CSV_PATH=/path/to/train/video
VAL_DATA_ROOT_PATH=/path/to/val/video
VAL_DATA_CSV_PATH=/path/to/val/csv
OUTPUT=/path/to/output
MODEL_NAME='umt'


FINETUNE=/path/to/ckpt
model='vit_large_patch16_224'
EMBEDDING_SIZE=768
PATCH_SIZE=16
NUM_FRAMES=8
INPUT_SIZE=224
TUBELET_SIZE=1
BATCH_SIZE=32

for SEED in 1
do
    for DATASET in ssv2 k400 k600 k700 hmdb51 ucf101 epic_verb epic_noun perception_test diving48 CharadesEgo  CharadesEgo_v1_only1st CharadesEgo_v1_only3rd
    do
        for NUM_SHOTS in 50
        do
            echo "SEED: $SEED"
            echo "DATASET: $DATASET"
            echo "NUM_SHOTS: $NUM_SHOTS"

            FLASH=1 torchrun --nproc_per_node="${NUM_GPUS}" --nnodes="${NNODES}" \
                --node_rank="${RANK}" --master_addr="${ADDR}" --master_port="${PORT}" \
                ac_export_feature_and_linear_probe.py \
                --embedding_size ${EMBEDDING_SIZE} \
                --data_set ${DATASET} \
                --seed ${SEED} \
                --num_shots ${NUM_SHOTS} \
                --num_step 8 \
                --train_data_root_path ${TRAIN_DATA_ROOT_PATH} \
                --train_data_csv_path ${TRAIN_DATA_CSV_PATH} \
                --val_data_root_path ${VAL_DATA_ROOT_PATH} \
                --val_data_csv_path ${VAL_DATA_CSV_PATH} \
                --save_report ${OUTPUT} \
                --batch_size ${BATCH_SIZE} \
                --model_name ${MODEL_NAME} \
                --model ${model} \
                --finetune ${FINETUNE} \
                --num_frames ${NUM_FRAMES} \
                --input_size ${INPUT_SIZE} \
                --tubelet_size ${TUBELET_SIZE} \
                --patch_size ${PATCH_SIZE}
        done
    done
done


# Due to the small dataset size, the following dataset raises errors when using 8 GPUs with a large batch size. 
FINETUNE=/path/to/ckpt
model='vit_large_patch16_224'
EMBEDDING_SIZE=768
PATCH_SIZE=16
NUM_FRAMES=8
INPUT_SIZE=224
TUBELET_SIZE=1
BATCH_SIZE=32

for SEED in 1
do
    for DATASET in RareAct Drone_Action
    do
        for NUM_SHOTS in 10
        do
            echo "SEED: $SEED"
            echo "DATASET: $DATASET"
            echo "NUM_SHOTS: $NUM_SHOTS"

            FLASH=1 torchrun --nproc_per_node="${NUM_GPUS}" --nnodes="${NNODES}" \
                --node_rank="${RANK}" --master_addr="${ADDR}" --master_port="${PORT}" \
                ac_export_feature_and_linear_probe.py \
                --embedding_size ${EMBEDDING_SIZE} \
                --data_set ${DATASET} \
                --seed ${SEED} \
                --num_shots ${NUM_SHOTS} \
                --num_step 8 \
                --train_data_root_path ${TRAIN_DATA_ROOT_PATH} \
                --train_data_csv_path ${TRAIN_DATA_CSV_PATH} \
                --val_data_root_path ${VAL_DATA_ROOT_PATH} \
                --val_data_csv_path ${VAL_DATA_CSV_PATH} \
                --save_report ${OUTPUT} \
                --batch_size ${BATCH_SIZE} \
                --model_name ${MODEL_NAME} \
                --model ${model} \
                --finetune ${FINETUNE} \
                --num_frames ${NUM_FRAMES} \
                --input_size ${INPUT_SIZE} \
                --tubelet_size ${TUBELET_SIZE} \
                --patch_size ${PATCH_SIZE}
        done
    done
done
