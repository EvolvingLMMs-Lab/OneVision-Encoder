import os
import random
import sys
import time

finetune_backbone = 1

master_addr = "localhost"
master_port = 18889 + 222
nnode = 1
node_rank = 0

backward_passes_per_step = 1
ckpt_interval = 1000
debug = 0


list_dataset = [
    # "LAION225_COYO400_IN14_HOI31",
    "ssv2_v0",
]
list_dataset = " ".join(list_dataset)


list_batch_size = [32]
list_batch_size = [str(x) for x in list_batch_size]
list_batch_size = " ".join(list_batch_size)


list_margin = "0.3"
list_sample_rate = "0.1"
list_filter = "0.75"

list_lr_pfc_weight = ["1"]
list_lr_pfc_weight = " ".join(list_lr_pfc_weight)


init_backbone = "NULL"
init_decoder_backbone = "NULL"
list_init_partial_fc = [
    "NULL",
    "NULL"
]
list_init_partial_fc = " ".join(list_init_partial_fc)

model_name = "PretrainEncoder_small_patch16_224_v0"
model_decoder_name = "PretrainDecoder_small_patch16_224_v0"
embedding_size = 512
lr = 0.001
num_sampled_data =  6_500_000
num_frames = 16

image_size = 224
opt = "adamw"
output = f"./checkpoints/{os.path.basename(__file__)}"
output_decoder = f"./checkpoints_decoder/{os.path.basename(__file__)}"

random_diff = 10
repeat_pfc = 3

save_pfc = 1
frequent = 20
warmup_ratio = 0.01
weight_decay = 0.05
weight_decay_pfc = 0.05
workers = 32


# 获取当前脚本所在目录的绝对路径
current_directory = os.path.dirname(os.path.abspath(__file__))

# 获取上一级目录
parent_directory = os.path.dirname(current_directory)

cmd = ""
cmd += f" PYTHONPATH={parent_directory}/"
cmd += " CUDA_DEVICE_MAX_CONNECTIONS=1"

# cmd += " CUDA_LAUNCH_BLOCKING=1"
cmd += " NCCL_SOCKET_IFNAME=eth0"
cmd += " NCCL_SOCKET_NTHREADS=32"
cmd += " NCCL_NSOCKS_PERTHREAD=4"
cmd += " NCCL_ALGO=Ring"

cmd += " NCCL_IB_GID_INDEX=3"
cmd += " NCCL_DEBUG=INFO"
cmd += " NCCL_IB_DISABLE=0"
cmd += " NCCL_IB_HCA=mlx5_bond_0,mlx5_bond_1,mlx5_bond_2,mlx5_bond_3,mlx5_bond_4,mlx5_bond_5,mlx5_bond_6,mlx5_bond_7"
cmd += " NCCL_NET_GDR_LEVEL=2"
cmd += " NCCL_IB_QPS_PER_CONNECTION=4"
cmd += " NCCL_IB_TC=160"
cmd += " NCCL_IB_TIMEOUT=22"
cmd += " GLOO_SOCKET_IFNAME=eth0"
cmd += " USE_CHECKPOINT=0"
# cmd += " CUDA_VISIBLE_DEVICES=4,5,6,7"
cmd += " CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7"

cmd += " torchrun --nproc_per_node 8"
cmd += f" --nnodes {nnode}"
cmd += f" --node_rank {node_rank}"
cmd += f" --master_addr {master_addr}"
cmd += f" --master_port {master_port}"
cmd += " training/train_all_llava.py"
cmd += f" --backward_passes_per_step {backward_passes_per_step}"
cmd += f" --ckpt_interval {ckpt_interval}"
cmd += f" --debug {debug}"
cmd += f" --list_batch_size {list_batch_size}"
cmd += f" --list_dataset {list_dataset}"
cmd += f" --list_filter {list_filter}"
cmd += f" --list_margin {list_margin}"
cmd += f" --list_sample_rate {list_sample_rate}"
cmd += f" --list_lr_pfc_weight {list_lr_pfc_weight}"
cmd += f" --list_init_partial_fc {list_init_partial_fc}"
cmd += f" --image_size {image_size}"
cmd += f" --embedding_size {embedding_size}"
cmd += f" --lr {lr}"
cmd += f" --num_sampled_data {num_sampled_data}"
cmd += f" --num_frames {num_frames}"
cmd += f" --model_name {model_name}"
cmd += f" --model_decoder_name {model_decoder_name}"
cmd += f" --opt {opt}"
cmd += f" --output {output}"
cmd += f" --output_decoder {output_decoder}"
cmd += f" --random_diff {random_diff}"
cmd += f" --repeat_pfc {repeat_pfc}"

# cmd += f" --init_mode {init_mode}"
cmd += f" --init_backbone {init_backbone}"
cmd += f" --init_decoder_backbone {init_decoder_backbone}"
# cmd += f" --init_partial_fc {init_partial_fc}"

cmd += f" --save_pfc {save_pfc}"
cmd += f" --frequent {frequent}"
cmd += f" --warmup_ratio {warmup_ratio}"
cmd += f" --weight_decay {weight_decay}"
cmd += f" --weight_decay_pfc {weight_decay_pfc}"
cmd += f" --workers {workers}"

print(cmd)
os.system(cmd)
