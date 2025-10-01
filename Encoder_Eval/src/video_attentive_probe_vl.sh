export DALI_DEBUG=1
export NUM_GPUS=8
export NNODES=1
export RANK=0
export ADDR="127.0.0.1"
export PORT="32590"
#pt=pretrain ppt=post-pretrain ft=finetune
# export CUDA_VISIBLE_DEVICES=4,5,6,7
# TRAIN_DATA_ROOT_PATH=/video_vit/eval_data/train
# TRAIN_DATA_CSV_PATH=/video_vit/eval_data/annotation
TRAIN_DATA_ROOT_PATH=/video_vit/video_encoder_eval/video_linear_probe/fewshot_video/ActionRecognition
TRAIN_DATA_CSV_PATH=/video_vit/video_encoder_eval/video_linear_probe/fewshot_video/ActionRecognition
VAL_DATA_ROOT_PATH=/video_vit/eval_data/val/
VAL_DATA_CSV_PATH=/video_vit/eval_data/annotation/
OUTPUT=/video_vit/feilong/LLaVA-ViT/Encoder_Eval/output/ov2_ssv2
mkdir -p "$OUTPUT"

MODEL_NAME="ricevl"

FINETUNE=/vlm/yinxie/code/checkpoints/rice-vit-large-patch14-560-w8b
# FINETUNE= /video_vit/feilong/LLaVA-ViT/checkpoints/llava_vit.py/00012501/backbone.pt
model="ricevl"

EMBEDDING_SIZE=1024
PATCH_SIZE=16
NUM_FRAMES=16
INPUT_SIZE=560
TUBELET_SIZE=1
BATCH_SIZE=32

for SEED in 1
do
    for DATASET in ssv2 #k400 k600 k700 epic_verb epic_noun perception_test diving48 CharadesEgo  CharadesEgo_v1_only1st CharadesEgo_v1_only3rd
    do
        for NUM_SHOTS in 50
        do
            echo "SEED: $SEED"
            echo "DATASET: $DATASET"
            echo "NUM_SHOTS: $NUM_SHOTS"

            FLASH=1 torchrun --nproc_per_node="${NUM_GPUS}" --nnodes="${NNODES}" \
                --node_rank="${RANK}" --master_addr="${ADDR}" --master_port="${PORT}" \
                video_attentive_probe_all/ac_export_feature_and_attentive_probe_latest.py \
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