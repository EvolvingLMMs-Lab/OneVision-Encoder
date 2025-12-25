#!/bin/bash

# ================= Configuration =================
PYTHON_SCRIPT="step3_kmeans.py"

BASE_INPUT_DIR="${DATA_DIR:-/path/to/data}"
BASE_OUTPUT_DIR="${DATA_DIR:-/path/to/data}"

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
FULL_INPUT_PATH="${BASE_INPUT_DIR}/features_part_${SUFFIX}"
FULL_OUTPUT_PATH="${BASE_OUTPUT_DIR}/centers_800000_part_${SUFFIX}"

echo "------------------------------------------------"
echo "Node IP       : $LOCAL_IP"
echo "Task Index    : $FILE_INDEX"
echo "Input File    : $FULL_INPUT_PATH"
echo "Output File   : $FULL_OUTPUT_PATH"
echo "------------------------------------------------"

# ================= Build command =================
CMD_ARGS=(
    "python"
    "$PYTHON_SCRIPT"
    "--input"       "$FULL_INPUT_PATH"
    "--num_classes" "800000"
    "--output"      "$FULL_OUTPUT_PATH"
)

echo -e "\n[INFO] Executing command:\n"
printf "    %s \\\n" "${CMD_ARGS[@]}"

echo -e "\n[INFO] Starting in 3 seconds...\n"
sleep 3

# ================= Execute =================
"${CMD_ARGS[@]}"
