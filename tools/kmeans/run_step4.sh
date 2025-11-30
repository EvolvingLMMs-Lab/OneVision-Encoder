#!/bin/bash

# ===================== 配置区域 =====================

# 1. 机器 IP 列表
HOSTS='
172.16.5.19
172.16.5.27
172.16.5.81
172.16.5.82
172.16.5.85
172.16.5.86
172.16.5.87
172.16.5.88
172.16.5.89
172.16.5.90
172.16.5.91
172.16.5.92
172.16.5.93
172.16.5.94
172.16.5.95
172.16.5.96
'

# 2. 脚本和参数
PY_SCRIPT=step4_collision.py
SCRIPT_ARGS='--input /video_vit/dataset/configs_for_llava_vit_versions_0_0_1_add_pandas70M/list_all_npy --class_center /video_vit/dataset/configs_for_llava_vit_versions_0_0_1_add_pandas70M/centers_howto100m_k710_panda70m_800000_8frames_4096.npy'

# 3. 环境配置
GPUS_PER_NODE=8
MASTER_PORT=29504
TORCHRUN_CMD=/root/miniconda3/envs/dino/bin/torchrun

# ===================================================

# 过滤空行并转为数组
HOST_LIST=($(echo "$HOSTS" | grep -v '^\s*$'))
MASTER_ADDR=${HOST_LIST[0]}
NUM_NODES=${#HOST_LIST[@]}
# 获取当前工作目录
CURRENT_DIR=$(pwd)

echo -----------------------------------------------------------
echo 🚀 Launching torchrun on $NUM_NODES nodes...
echo Master: $MASTER_ADDR
echo WorkDir: $CURRENT_DIR
echo -----------------------------------------------------------

for (( i=0; i<${NUM_NODES}; i++ )); do
    HOST=${HOST_LIST[$i]}
    NODE_RANK=$i

    echo Processing Node $NODE_RANK: $HOST

    # 关键修改：
    # 1. cd $CURRENT_DIR: 切换到和当前一样的目录，确保相对路径脚本能找到
    # 2. export PATH=\"$PATH\": 把当前的 PATH 变量带过去，确保能找到 torchrun 和 python

    CMD="cd $CURRENT_DIR; export PATH=\"$PATH\"; nohup $TORCHRUN_CMD \
      --nproc_per_node=$GPUS_PER_NODE \
      --nnodes=$NUM_NODES \
      --node_rank=$NODE_RANK \
      --master_addr=$MASTER_ADDR \
      --master_port=$MASTER_PORT \
      $PY_SCRIPT $SCRIPT_ARGS"

    # 注意：上面我把日志重定向加上了 > /dev/null 2>&1，防止 nohup 在当前目录生成大量 nohup.out 文件
    # 如果你需要日志，把 > /dev/null 改成 > log_node_${NODE_RANK}.txt

    ssh -n $HOST "$CMD" &

    echo "  -> Done"
    sleep 0.1
done

echo -----------------------------------------------------------
echo ✅ All jobs started.