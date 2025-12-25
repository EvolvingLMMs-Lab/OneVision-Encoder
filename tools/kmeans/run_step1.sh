#!/bin/bash

# ================= Configuration =================
PYTHON_SCRIPT="step1_extract_video_features.py"
BASE_INPUT="${DATA_DIR:-/path/to/data}/list_all_valid"
BASE_OUTPUT="${DATA_DIR:-/path/to/data}/output_list_all_valid"

# Configure your cluster hosts here
HOSTS=(
    # "node-01"
    # "node-02"
    # Add your hosts...
)

# ================= Auto-compute logic =================
LOCAL_IP=$(hostname -I | head -n 1 | awk '{print $1}')
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

# ================= Build command =================
CMD_ARGS=(
    "torchrun"
    "--nnodes=1"
    "--nproc_per_node=8"
    "--node_rank=0"
    "--master_addr=127.0.0.1"
    "--master_port=${MASTER_PORT:-29507}"
    "$PYTHON_SCRIPT"
    "--input"       "$FULL_INPUT_PATH"
    "--output"      "$FULL_OUTPUT_PATH"
    "--batch_size"  "32"
    "--num_frames"  "8"
)

echo -e "\n[INFO] Executing command:\n"
printf "    %s \\\n" "${CMD_ARGS[@]}"

echo -e "\n[INFO] Starting in 3 seconds...\n"
sleep 3

# ================= Execute =================
"${CMD_ARGS[@]}"
