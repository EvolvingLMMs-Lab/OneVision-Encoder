NUM_GPUS=8
NNODES=1
ADDR=localhost:1

#feature extraction
WORLD_SIZE=24 #多

DATA_PATH=/video_vit/TAD-Dataset
CONFIG_PATH=configs/actionformer
CHECKPOINT_PATH=exps
EMBEDDING_SIZE=1024

#dinov3
MODEL_NAME=dinov3  #如果是huggingface文件就是这个样子的
CKPT=/video_vit/video_encoder_eval/video_linear_probe/checkpoint/facebook/dinov3-vitl16-pretrain-lvd1689m #.pth 文件或者huggingface的名字
MODEL_TYPES=dinov3

#mlcd
# MODEL_NAME=mlcd 
# CKPT=/video_vit/video_encoder_eval/video_linear_probe/checkpoint/deepglint/mlcd-vit-bigG-patch14-224  #.pth 文件或者huggingface的名字
# MODEL_TYPES=mlcd

# PE-core
# MODEL_NAME=PE-Core-L14-336    #如果是huggingface文件就是这个样子的
# CKPT=PE-Core-L14-336  #.pth 文件或者huggingface的名字
# MODEL_TYPES=pe

#internvideo
# CKPT=/video_vit/video_encoder_eval/video_linear_probe/checkpoint/internvideo_v1/internvideov1-videomae-L-16____UnlabeledHybrid_1M_and_ppt_K700_65W.pth
# MODEL_NAME=vit_large_patch16_224
# MODEL_TYPES=internvideo_v1

#univit
# MODEL_NAME=internvideo2_tem_dense_urope_tube_small_patch16_224_fc_512_v1 #如果是pth的话这个必须要是模型名
# CKPT=pretrained/backbone_tube248_dense_moreepoch.pt
# MODEL_NAME=vit_large_patch16_224
# MODEL_TYPES=univit   # 用空格分隔，别写逗号

END_NAME=dinov3_test #文件后缀名
DATASETS=(charades fineaction)


for DATASET in "${DATASETS[@]}"; do
    echo "==> [${DATASET}] feature extraction with ${MODEL_NAME}"
    python data/test_featrue_extraction.py \
        --data_set "${DATASET}" \
        --data_path "${DATA_PATH}/${DATASET}/raw_data/video" \
        --save_path "${DATA_PATH}/${DATASET}/features/${MODEL_NAME}${END_NAME}" \
        --model_name "${MODEL_NAME}" \
        --model_type "${MODEL_TYPES}" \
        --ckpt_path "${CKPT}" \
        --world_size "${WORLD_SIZE}" \
        --anno_file "${DATA_PATH}/${DATASET}/annotations/${DATASET}.json" \

    echo "==> generate data and model config file"
    python tools/generate_config.py \
        --dataset "${DATASET}" \
        --data_path "${DATA_PATH}" \
        --feature_path "${DATA_PATH}/${DATASET}/features/${MODEL_NAME}${END_NAME}" \
        --embedding_size "${EMBEDDING_SIZE}" \
        --exp_save_path "${CHECKPOINT_PATH}/${DATASET}/actionformer_${MODEL_NAME}${END_NAME}" \
        --model_config_path "configs/actionformer/${DATASET}_${MODEL_NAME}${END_NAME}.py" \
        --data_config_path "../_base_/datasets/${DATASET}/feature_${MODEL_NAME}${END_NAME}.py" \

    echo "==> [${DATASET}] training (${MODEL_NAME})"
    torchrun \
        --nnodes="${NNODES}" \
        --nproc_per_node="${NUM_GPUS}" \
        --rdzv_backend=c10d \
        --rdzv_endpoint="${ADDR}" \
        tools/train.py \
        configs/actionformer/${DATASET}_${MODEL_NAME}${END_NAME}.py

    echo "==> [${DATASET}] testing (${MODEL_NAME})"
    torchrun \
        --nnodes="${NNODES}" \
        --nproc_per_node="${NUM_GPUS}" \
        --rdzv_backend=c10d \
        --rdzv_endpoint="${ADDR}" \
        tools/test.py \
        configs/actionformer/${DATASET}_${MODEL_NAME}${END_NAME}.py \
        --checkpoint "${CHECKPOINT_PATH}/${DATASET}/actionformer_${MODEL_NAME}${END_NAME}/gpu${NUM_GPUS}_id0/checkpoint/best.pth"
done
