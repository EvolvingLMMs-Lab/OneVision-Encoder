#!/bin/bash
set -e  # Exit immediately on error

export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 1. Define Node IP List (Host List)
list_hostname=(
)

# 2. Get local IP address for eth0 using 'ifconfig'
# We filter for 'inet ' to get IPv4, print the second column, and remove 'addr:' if present (for compatibility)
current_ip=$(ifconfig eth0 | grep 'inet ' | awk '{print $2}' | sed 's/addr://')

if [ -z "$current_ip" ]; then
    echo "Error: Could not detect IP address for interface eth0 using ifconfig."
    echo "Please ensure 'net-tools' is installed and eth0 is active."
    exit 1
fi

echo "Detected Local IP (eth0): $current_ip"

# 3. Set Master Address
# Logic: Default to the FIRST IP in the list (172.16.5.19)
# This ensures all nodes connect to the same master.
master_addr="${MASTER_ADDR:-${list_hostname[0]}}"
master_port="${MASTER_PORT:-29500}"

# Calculate total number of nodes
nnode=${#list_hostname[@]}

# 4. Determine node_rank based on IP
node_rank=-1
for idx in "${!list_hostname[@]}"; do
  if [[ "${list_hostname[$idx]}" == "$current_ip" ]]; then
    node_rank=$idx
    break
  fi
done

# If current IP is not in the list, exit with error
if [[ $node_rank -eq -1 ]]; then
    echo "Error: Current IP ($current_ip) not found in list_hostname!"
    exit 1
fi

echo "=========================="
echo "Distributed Training Configuration"
echo "=========================="
echo "master_addr=$master_addr"
echo "master_port=$master_port"
echo "nnode=$nnode"
echo "node_rank=$node_rank"
echo "HOSTNAME=$HOSTNAME"
echo "GPU devices:  0,1,2,3,4,5,6,7"
echo "Output directory: ckpts/$(basename "$0" .sh)"
echo "=========================="

# Create output directory
mkdir -p "ckpts/$(basename "$0" .sh)"

init_backbone=NULL
path_pfc_si=NULL
path_pfc_ocr=NULL
path_pfc_codec=NULL

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --master_addr $master_addr --master_port $master_port \
  --nnode $nnode --node_rank $node_rank --nproc_per_node 8 \
  -m \
  training.train \
  --model_name ov_encoder_large \
  --image_size 448 \
  --num_frames 8 \
  --image_size_video 224 \
  --embedding_size 1024 \
  --list_batch_sizes 16 8 16 \
  --lr 0.00005 \
  --warmup_ratio 0.0001 \
  --list_datasets onevision_encoder_si_ssd onevision_encoder_ocr_ssd onevision_encoder_video_codec  \
  --output "ckpts/$(basename "$0" .sh)" \
  --init_backbone $init_backbone \
  --list_init_partial_fc_paths $path_pfc_si $path_pfc_ocr $path_pfc_codec \
  --list_sample_rates 0.1 0.1 0.1 \
  --list_lr_pfc_weights 1 1 1 \
  --list_loss_weights 1 1 1 \
  --num_sampled_data 320000000 \
  --finetune_backbone 1 \
  --backward_passes_per_step 4 \
  --num_tokens_per_frame 256 \
  --residual_ratio 0.5 \
  --frame_sampling_ratio 0.4