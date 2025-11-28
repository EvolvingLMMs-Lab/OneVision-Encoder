#!/bin/bash

# ================= 配置区域 =================
PYTHON_SCRIPT="step3_kmeans.py"

# 基础路径前缀 (注意：后缀部分会在逻辑中拼接)
BASE_INPUT_DIR="/video_vit/dataset/clips_square_aug_panda70M_meta_llava_vit_256"
BASE_OUTPUT_DIR="/video_vit/dataset/clips_square_aug_panda70M_meta_llava_vit_256"

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
# 获取本机 IP
LOCAL_IP=$(hostname -I | grep -o "172.16.5.[0-9]*" | head -n 1)
FILE_INDEX=-1

# 匹配 IP 获取索引
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

# 格式化索引 (例如 0 -> 000)
SUFFIX=$(printf "%03d" $FILE_INDEX)

# 构造具体的文件路径
# Input: list_all_valid_pandas_70M_part_000_feat_processed_dim_512_frames_8
FULL_INPUT_PATH="${BASE_INPUT_DIR}/list_all_valid_pandas_70M_part_${SUFFIX}_feat_processed_dim_512_frames_8"

# Output: centers_part_800000_part_000
FULL_OUTPUT_PATH="${BASE_OUTPUT_DIR}/centers_800000_part_${SUFFIX}"

echo "------------------------------------------------"
echo "Node IP       : $LOCAL_IP"
echo "Task Index    : $FILE_INDEX"
echo "Input File    : $FULL_INPUT_PATH"
echo "Output File   : $FULL_OUTPUT_PATH"
echo "------------------------------------------------"

# ================= 构造命令 =================

# 使用数组构造命令，方便处理空格和特殊字符
CMD_ARGS=(
    "python"
    "$PYTHON_SCRIPT"
    "--input"       "$FULL_INPUT_PATH"
    "--num_classes" "800000"
    "--output"      "$FULL_OUTPUT_PATH"
)

echo -e "\n[INFO] 即将执行以下命令:\n"

# 获取数组长度
len=${#CMD_ARGS[@]}

# 循环打印，美化显示
for (( i=0; i<len; i++ )); do
    current="${CMD_ARGS[$i]}"

    # 下一个索引和值
    next_idx=$((i + 1))
    next_val="${CMD_ARGS[$next_idx]}"

    # 判断是否为 "参数名 参数值" 的组合 (以-开头，且下一个不以-开头)
    if [[ "$current" == -* ]] && [[ $next_idx -lt $len ]] && [[ "$next_val" != -* ]]; then
        echo -n "    $current $next_val"

        if [[ $((i + 2)) -eq $len ]]; then
            echo ""
        else
            echo " \\"
        fi
        ((i++))
    else
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