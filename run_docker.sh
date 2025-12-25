# Docker run script for OneVision Encoder training
# Configure data mount paths according to your environment

docker run \
    -it \
    --gpus all \
    --ipc host \
    --net host \
    --privileged \
    --cap-add IPC_LOCK \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --rm \
    -v "$(pwd)":/workspace/OneVision-Encoder \
    -v /data_3:/data_3 \
    -w /workspace/OneVision-Encoder \
    -e NCCL_TIMEOUT=1800 \
    -e CUDA_DEVICE_MAX_CONNECTIONS=1 \
    -e NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth0} \
    -e NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-3} \
    -e NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0} \
    -e NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5_0}" \
    -e NCCL_NET_GDR_LEVEL=${NCCL_NET_GDR_LEVEL:-2} \
    -e NCCL_IB_QPS_PER_CONNECTION=${NCCL_IB_QPS_PER_CONNECTION:-4} \
    -e NCCL_IB_TC=${NCCL_IB_TC:-160} \
    -e NCCL_IB_TIMEOUT=${NCCL_IB_TIMEOUT:-22} \
    -e NCCL_CROSS_NIC=${NCCL_CROSS_NIC:-1} \
    -e NCCL_MIN_NCHANNELS=${NCCL_MIN_NCHANNELS:-8} \
    -e NCCL_MAX_NCHANNELS=${NCCL_MAX_NCHANNELS:-16} \
    llava_vit:25.11.22 \
    bash -c "service ssh restart; bash"
