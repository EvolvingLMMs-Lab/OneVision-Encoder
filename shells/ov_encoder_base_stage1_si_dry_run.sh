#!/bin/bash
set -e  # Exit immediately on error

export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Hostname list - Configure according to actual environment
list_hostname=(
  localhost
)

# Master node address and port
master_addr="${MASTER_ADDR:-127.0.0.1}"
master_port="${MASTER_PORT:-29500}"

# Calculate total number of nodes
nnode=${#list_hostname[@]}

# Build hostname to noderank mapping
declare -A hostname2noderank
for idx in "${!list_hostname[@]}"; do
  hostname2noderank["${list_hostname[$idx]}"]=$idx
done

# Current node's rank
node_rank=${hostname2noderank[$HOSTNAME]}

# Special handling: for single-machine training or localhost scenario
if [[ $nnode -eq 0 ]] || [[ "${list_hostname[0]}" == "localhost" ]] || [[ -z "$node_rank" ]]; then
  echo "Single-machine training detected, automatically setting node_rank=0"
  node_rank=0
  nnode=1
  # If host list is empty or only has localhost, ensure local address is used
  if [[ $nnode -eq 0 ]] || [[ "${list_hostname[0]}" == "localhost" ]]; then
    master_addr="127.0.0.1"
  fi
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

torchrun \
  --master_addr "$master_addr" \
  --master_port "$master_port" \
  --nnode "$nnode" \
  --node_rank "$node_rank" \
  --nproc_per_node 8 \
  -m training.train \
  --debug 1 \
  --model_name ov_encoder_base \
  --image_size 224 \
  --embedding_size 768 \
  --list_batch_sizes 128 \
  --lr 0.001 \
  --warmup_ratio 0.1 \
  --list_datasets onevision_encoder_si_dry_run \
  --output "ckpts/$(basename "$0" .sh)" \
  --list_sample_rates 0.1 \
  --list_lr_pfc_weights 1 \
  --num_sampled_data 13000000000 \
  --finetune_backbone 1 \
  --backward_passes_per_step 1

echo "=========================="
echo "Training completed!"
echo "=========================="