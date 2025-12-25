#!/bin/bash

# ===================== Configuration =====================

# 1. Cluster host list - configure your hosts here
HOSTS='
# node-01
# node-02
# Add your hosts...
'

# 2. Script and arguments
PY_SCRIPT=step4_collision.py
INPUT_PATH="${DATA_DIR:-/path/to/data}/list_all_npy"
CENTER_PATH="${DATA_DIR:-/path/to/data}/centers.npy"
SCRIPT_ARGS="--input $INPUT_PATH --class_center $CENTER_PATH"

# 3. Environment config
GPUS_PER_NODE=8
MASTER_PORT=${MASTER_PORT:-29504}
TORCHRUN_CMD=${TORCHRUN_CMD:-torchrun}

# ===================================================

# Filter empty lines and convert to array
HOST_LIST=($(echo "$HOSTS" | grep -v '^\s*$' | grep -v '^#'))
MASTER_ADDR=${HOST_LIST[0]}
NUM_NODES=${#HOST_LIST[@]}
CURRENT_DIR=$(pwd)

echo -----------------------------------------------------------
echo "🚀 Launching torchrun on $NUM_NODES nodes..."
echo "Master: $MASTER_ADDR"
echo "WorkDir: $CURRENT_DIR"
echo -----------------------------------------------------------

for (( i=0; i<${NUM_NODES}; i++ )); do
    HOST=${HOST_LIST[$i]}
    NODE_RANK=$i

    echo "Processing Node $NODE_RANK: $HOST"

    CMD="cd $CURRENT_DIR; export PATH=\"$PATH\"; nohup $TORCHRUN_CMD \
      --nproc_per_node=$GPUS_PER_NODE \
      --nnodes=$NUM_NODES \
      --node_rank=$NODE_RANK \
      --master_addr=$MASTER_ADDR \
      --master_port=$MASTER_PORT \
      $PY_SCRIPT $SCRIPT_ARGS"

    ssh -n $HOST "$CMD" &

    echo "  -> Done"
    sleep 0.1
done

echo -----------------------------------------------------------
echo "✅ All jobs started."
