MODEL_NAME=pretrain_encoder_base_patch16_224_v11_09_ln_head_ip
CKPT_PATH=/video_vit/xiangan/checkpoint_llava_vit/continue_with_mlcd_1536_tokens_b16_mix_three_input_residual_mv_new_b16/00030000/backbone.pt

mkdir -p output/three_input_b16_mv_sampling
mkdir -p output/three_input_b16_mv_residual
mkdir -p output/three_input_b16_mv_residual_mv
mkdir -p output/three_input_b16_mv_residual_mv_new

# 定义数据集数组
# DATASETS=(ssv2 ucf101 perception_test hmdb51)
DATASETS=(ssv2 perception_test hmdb51 ucf101)

GPU_ID=0
BASE_PORT=12344

# 第一个 for 循环：sampling 配置
# OUTPUT_DIR=three_input_b16_mv_sampling
# MODEL_FAMILY=llava_vit_sampling
# SCRIPT=video_attentive_probe.sh

# for dataset in "${DATASETS[@]}"; do
#     PORT=$((BASE_PORT + GPU_ID))
    
#     PORT=$PORT \
#     OUTPUT=output/$OUTPUT_DIR \
#     CUDA_VISIBLE_DEVICES=$GPU_ID \
#     MODEL_FAMILY=$MODEL_FAMILY \
#     EMBEDDING_SIZE=768 \
#     NUM_FRAMES=64 \
#     PYTHONPATH=../ \
#     MODEL_NAME=$MODEL_NAME \
#     DATASETS=$dataset \
#     CKPT_PATH=$CKPT_PATH \
#     bash $SCRIPT > output/$OUTPUT_DIR/${dataset}.log &
    
#     GPU_ID=$((GPU_ID + 1))
# done

# 第二个 for 循环：residual 配置
OUTPUT_DIR=three_input_b16_mv_residual_mv_new
MODEL_FAMILY=llava_vit_mv
SCRIPT=video_attentive_probe_ip.sh

for dataset in "${DATASETS[@]}"; do
    PORT=$((BASE_PORT + GPU_ID))
    
    PORT=$PORT \
    OUTPUT=output/$OUTPUT_DIR \
    CUDA_VISIBLE_DEVICES=$GPU_ID \
    MODEL_FAMILY=$MODEL_FAMILY \
    EMBEDDING_SIZE=768 \
    NUM_FRAMES=64 \
    USING_MV=1 \
    PYTHONPATH=../ \
    MODEL_NAME=$MODEL_NAME \
    DATASETS=$dataset \
    CKPT_PATH=$CKPT_PATH \
    bash $SCRIPT > output/$OUTPUT_DIR/${dataset}.log &
    
    GPU_ID=$((GPU_ID + 1))
done