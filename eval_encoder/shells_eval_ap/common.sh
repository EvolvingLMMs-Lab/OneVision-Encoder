#!/bin/bash
# ============================================================================
# 通用配置脚本 - 用于 attentive_probe 评估
# 使用方法: 在各个模型脚本中 source 此文件后调用 run_attentive_probe
# ============================================================================

# 环境变量设置
export PYTHONPATH=../

# 默认数据集列表 (可在调用脚本中覆盖)
DEFAULT_DATASETS=(
    "ssv2"
    "diving48"
    "perception_test"
    "epic_verb"
    "epic_noun"
    "hmdb51"
    "k400"
    "charadesego"
)

# ============================================================================
# 根据数据集名称获取 batch size
# 参数: $1 - 数据集名称
# ============================================================================
get_batch_size() {
    local dataset="$1"
    if [[ "$dataset" == "ssv2" || "$dataset" == "diving48" || "$dataset" == "perception_test" ]]; then
        echo 4
    elif [[ "$dataset" == "hmdb51" ]]; then
        echo 2
    else
        echo 16
    fi
}

# ============================================================================
# 根据数据集名称获取 epochs
# 参数: $1 - 数据集名称
# ============================================================================
get_epochs() {
    local dataset="$1"
    if [[ "$dataset" == "hmdb51" ]]; then
        echo 30
    else
        echo 10
    fi
}

# ============================================================================
# 运行 attentive_probe 评估
# 需要预先设置的变量:
#   - MODEL_FAMILY: 模型系列 (必需)
#   - MODEL_NAME: 模型名称 (必需)
#   - MODEL_WEIGHT: 模型权重路径 (可选, 默认 "NULL")
#   - FRAMES_TOKEN_NUM: token 数量 (可选, 默认 196)
#   - EMBEDDING_SIZE: embedding 维度 (可选, 默认 768)
#   - INPUT_SIZE: 输入尺寸 (可选, 不设置则不传递此参数)
#   - NUM_FRAMES: 帧数 (可选, 不设置则不传递此参数)
#   - DATASETS: 数据集数组 (可选, 默认使用 DEFAULT_DATASETS)
#   - REPORT_DIR_SUFFIX: 报告目录后缀 (可选, 如 "_16frames")
# ============================================================================
run_attentive_probe() {
    # 设置默认值
    MODEL_WEIGHT="${MODEL_WEIGHT:-NULL}"
    FRAMES_TOKEN_NUM="${FRAMES_TOKEN_NUM:-196}"
    EMBEDDING_SIZE="${EMBEDDING_SIZE:-768}"
    REPORT_DIR_SUFFIX="${REPORT_DIR_SUFFIX:-}"

    # 使用自定义数据集或默认数据集
    if [[ ${#DATASETS[@]} -eq 0 ]]; then
        DATASETS=("${DEFAULT_DATASETS[@]}")
    fi

    # 构建报告目录
    BASE_REPORT_DIR="result_attentive_probe/${MODEL_FAMILY}/${MODEL_NAME}${REPORT_DIR_SUFFIX}"

    # 循环遍历每个数据集进行测试
    for DATASET in "${DATASETS[@]}"; do
        BATCH_SIZE=$(get_batch_size "$DATASET")
        EPOCHS=$(get_epochs "$DATASET")

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

        # 构建额外参数
        EXTRA_ARGS=""
        if [[ -n "${INPUT_SIZE}" ]]; then
            EXTRA_ARGS="${EXTRA_ARGS} --input_size ${INPUT_SIZE}"
        fi
        if [[ -n "${NUM_FRAMES}" ]]; then
            EXTRA_ARGS="${EXTRA_ARGS} --num_frames ${NUM_FRAMES}"
        fi

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
            --frames_token_num ${FRAMES_TOKEN_NUM} \
            --embedding_size ${EMBEDDING_SIZE} \
            ${EXTRA_ARGS}

        echo "Finished testing ${DATASET}"
        echo ""
    done
}
