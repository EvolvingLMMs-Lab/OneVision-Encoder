import os
import sys

script = sys.argv[1]

cmd = ""
cmd += " CUDA_DEVICE_MAX_CONNECTIONS=1"
cmd += " NCCL_SOCKET_IFNAME=eth0"
cmd += " NCCL_SOCKET_NTHREADS=32"
cmd += " NCCL_NSOCKS_PERTHREAD=4"
cmd += " NCCL_ALGO=Ring"

cmd += " NCCL_IB_GID_INDEX=3"
cmd += " NCCL_IB_DISABLE=0"
cmd += " NCCL_IB_HCA=mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_1"
cmd += " NCCL_NET_GDR_LEVEL=2"
cmd += " NCCL_IB_QPS_PER_CONNECTION=4"
cmd += " NCCL_IB_TC=160"
cmd += " NCCL_IB_TIMEOUT=22"
cmd += " GLOO_SOCKET_IFNAME=eth0"
cmd += " http_proxy=http://172.16.5.79:18000"
cmd += " https_proxy=http://172.16.5.79:18000"
cmd += " HF_ENDPOINT=https://hf-mirror.com"
cmd += " TOKENIZERS_PARALLELISM=false"

# export http_proxy=http://172.16.5.79:18000
# export https_proxy=http://172.16.5.79:18000
# cmd += " USE_CHECKPOINT=1"

cmd += " CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "
cmd += f" bash {script}"
print(cmd)
os.system(cmd)
