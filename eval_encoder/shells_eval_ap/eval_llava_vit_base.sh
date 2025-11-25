#!/bin/bash

# 环境变量设置
export PYTHONPATH=../

# 模型配置
MODEL_FAMILY="llava_vit_sampling"
MODEL_NAME="llava_vit_base_ln"
MODEL_WEIGHT="/video_vit/xiangan/checkpoint_llava_vit/2025_11_19_new_b16_continue_80gpus_how_to_100m_continue/00040000/backbone.pt"

# 修改点：使用变量拼接路径
BASE_REPORT_DIR="result_attentive_probe/${MODEL_FAMILY}/${MODEL_NAME}"

# 要测试的数据集列表
DATASETS=(
    "ssv2"
    "diving48"
    "perception_test"
    "epic_verb"
    "epic_noun"
    "hmdb51"
    "k400"
    "charadesego"
)

# 循环遍历每个数据集进行测试
for DATASET in "${DATASETS[@]}"; do
    # ==========================================
    # 根据数据集名称动态设置 Batch Size
    # ==========================================
    # 设置 BATCH_SIZE：hmdb51 -> 2，ssv2/diving48/perception_test -> 4，其它 -> 16
    if [[ "$DATASET" == "ssv2" || "$DATASET" == "diving48" || "$DATASET" == "perception_test" ]]; then
        BATCH_SIZE=4
    elif [[ "$DATASET" == "hmdb51" ]]; then
        BATCH_SIZE=2
    else
        BATCH_SIZE=16
    fi

    if [[ "$DATASET" == "hmdb51" ]]; then
        EPOCHS=30
    else
        EPOCHS=10
    fi

    echo "DATASET=$DATASET, BATCH_SIZE=$BATCH_SIZE"

    echo "========================================================"
    echo "Start testing dataset: ${DATASET}"
    echo "Model: ${MODEL_NAME}"
    echo "Batch Size: ${BATCH_SIZE}"
    echo "Report Dir: ${BASE_REPORT_DIR}/${DATASET}"
    echo "========================================================"

    # 构建输出目录
    SAVE_DIR="${BASE_REPORT_DIR}/${DATASET}"
    mkdir -p "$SAVE_DIR"

    torchrun --nproc_per_node 8 --master_port 15555 \
        attentive_probe.py \
        --eval_freq 1 \
        --default_lr_list 0.0001 \
        --default_epoch "${EPOCHS}" \
        --batch_size ${BATCH_SIZE} \
        --default_weight_decay 0 \
        --dali_py_num_workers 8 \
        --model_family "${MODEL_FAMILY}" \
        --model_name "${MODEL_NAME}" \
        --model_weight "${MODEL_WEIGHT}" \
        --dataset "${DATASET}" \
        --save_report "${SAVE_DIR}" \
        --frames_token_num 196 \
        --embedding_size 768

    echo "Finished testing ${DATASET}"
    echo ""
done
