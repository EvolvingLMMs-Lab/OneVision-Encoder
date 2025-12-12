
# 例如设置最小丢弃 15%，第一帧不参与比例计算且不丢弃
# export RES_MIN_DROP_RATIO=0.50

# # 保存整段序列可视化
# export HEVC_SHARDS=1
# export VIZ_MASK=1
# export VIZ_MASK_FRAMES=all
# export VIZ_MASK_INTERVAL=1
# export VIZ_MASK_SAMPLES=1
# export UMT_HEVC_Y_ONLY=1


# export CUDA_DEVICE_MAX_CONNECTIONS=1
# export NCCL_SOCKET_IFNAME=eth0
# export NCCL_SOCKET_NTHREADS=8
# export NCCL_NSOCKS_PERTHREAD=1
# export NCCL_IB_GID_INDEX=3
# export NCCL_DEBUG=INFO
# export NCCL_IB_DISABLE=0
# export NCCL_IB_HCA=mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_1
# export NCCL_NET_GDR_LEVEL=2
# export NCCL_IB_QPS_PER_CONNECTION=8
# export NCCL_IB_TC=160
# export NCCL_IB_TIMEOUT=22
# export USE_CHECKPOINT=0
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 主机名列表
list_hostname=(
  instance-5-37
)

# 主节点地址和端口
master_addr="172.16.5.37"
master_port=$((18889 + 305))

# 计算节点总数
nnode=${#list_hostname[@]}

# 构建主机名到noderank的映射
declare -A hostname2noderank
for idx in "${!list_hostname[@]}"; do
  hostname2noderank["${list_hostname[$idx]}"]=$idx
done

# 当前节点的 rank
node_rank=${hostname2noderank[$HOSTNAME]}

echo "master_addr=$master_addr"
echo "master_port=$master_port"
echo "nnode=$nnode"
echo "node_rank=$node_rank"


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --master_addr $master_addr --master_port $master_port \
  --nnode $nnode --node_rank $node_rank --nproc_per_node 8 \
  -m \
  training.train_univit_12_02_l14_448_sampling_using_huggingface_model \
  --model_name hf_llava_vit_huge_ln \
  --image_size 224 \
  --embedding_size 1024 \
  --list_batch_sizes 16 \
  --debug 1 \
  --lr 0.0008 \
  --warmup_ratio 0.1 \
  --list_datasets llava_vit_si_2025_12_12 \
  --output /video_vit/xiangan/checkpoint_llava_vit/`basename $0 .sh` \
  --list_sample_rates 0.1 \
  --num_sampled_data 20_000_000_000 \
  --finetune_backbone 1 \
  --backward_passes_per_step 1
