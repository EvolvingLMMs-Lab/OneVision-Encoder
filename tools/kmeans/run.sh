#!/bin/bash

# ================= 配置区域 =================
PYTHON_SCRIPT="step1_extract_video_features.py"
BASE_INPUT="/video_vit/dataset/clips_square_aug_panda70M_meta_llava_vit_256/list_all_valid_pandas_70M"
BASE_OUTPUT="/video_vit/dataset/clips_square_aug_panda70M_meta_llava_vit_256/output_list_all_valid_pandas_70M"

HOSTS=(
    "172.16.5.19"
    "172.16.5.27"
    "172.16.5.81"
    "172.16.5.82"
    "172.16.5.85"
    "172.16.5.86"
    "172.16.5.87"
    "172.16.5.88"
    "172.16.5.89"
    "172.16.5.90"
    "172.16.5.91"
    "172.16.5.92"
    "172.16.5.93"
    "172.16.5.94"
    "172.16.5.95"
    "172.16.5.96"
)

# ================= 自动计算逻辑 =================
LOCAL_IP=$(hostname -I | grep -o "172.16.5.[0-9]*" | head -n 1)
FILE_INDEX=-1
for i in "${!HOSTS[@]}"; do
    if [[ "${HOSTS[$i]}" == "${LOCAL_IP}" ]]; then
        FILE_INDEX=$i
        break
    fi
done

if [ $FILE_INDEX -eq -1 ]; then
    echo "Error: Current IP ($LOCAL_IP) not found in HOSTS list."
    exit 1
fi

SUFFIX=$(printf "%03d" $FILE_INDEX)
FULL_INPUT_PATH="${BASE_INPUT}_part_${SUFFIX}"
FULL_OUTPUT_PATH="${BASE_OUTPUT}_part_${SUFFIX}"

echo "------------------------------------------------"
echo "Node IP       : $LOCAL_IP"
echo "Task Index    : $FILE_INDEX"
echo "Input File    : $FULL_INPUT_PATH"
echo "Output File   : $FULL_OUTPUT_PATH"
echo "------------------------------------------------"

# ================= 构造命令 =================

# 必须分开写，避免 argparse 报错
CMD_ARGS=(
    "torchrun"
    "--nnodes=1"
    "--nproc_per_node=8"
    "--node_rank=0"
    "--master_addr=127.0.0.1"
    "--master_port=29507"
    "$PYTHON_SCRIPT"
    "--input"       "$FULL_INPUT_PATH"
    "--output"      "$FULL_OUTPUT_PATH"
    "--batch_size"  "32"
    "--num_frames"  "8"
)

echo -e "\n[INFO] 即将执行以下命令:\n"

# 获取数组长度
len=${#CMD_ARGS[@]}

# 循环打印，智能处理换行符
for (( i=0; i<len; i++ )); do
    current="${CMD_ARGS[$i]}"

    # 下一个索引和值
    next_idx=$((i + 1))
    next_val="${CMD_ARGS[$next_idx]}"

    # 判断是否为 "参数名 参数值" 的组合 (以-开头，且下一个不以-开头)
    if [[ "$current" == -* ]] && [[ $next_idx -lt $len ]] && [[ "$next_val" != -* ]]; then
        # 是组合，打印在一行
        echo -n "    $current $next_val"

        # 检查这组参数是否是整个命令的结尾
        if [[ $((i + 2)) -eq $len ]]; then
            echo ""  # 最后一行，不加斜杠
        else
            echo " \\" # 不是最后，加斜杠
        fi

        # 跳过下一个，因为已经打印了
        ((i++))
    else
        # 是单个参数（如 torchrun 或 脚本名）
        echo -n "    $current"

        if [[ $next_idx -eq $len ]]; then
            echo ""
        else
            echo " \\"
        fi
    fi
done

echo -e "\n[INFO] 3秒后开始执行...\n"
sleep 3

# ================= 执行 =================
"${CMD_ARGS[@]}"