# -e http_proxy=http://172.16.5.77:8889 \
# -e https_proxy=http://172.16.5.77:8889 \
# -e http_proxy=http://172.16.5.79:18000 \
# -e https_proxy=http://172.16.5.79:18000 \

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
    -v "$(pwd)":/workspace/LLaVA-ViT \
    -v /train_tmp:/train_tmp \
    -v /vlm:/vlm \
    -v /video_vit:/video_vit \
    -v /rice_ocr:/rice_ocr \
    -v /data_0:/data_0 \
    -v /data_1:/data_1 \
    -v /data_2:/data_2 \
    -v /data_3:/data_3 \
    -w /workspace/LLaVA-ViT/ \
    -e NCCL_TIMEOUT=1800 \
    -e CUDA_DEVICE_MAX_CONNECTIONS=1 \
    -e NCCL_SOCKET_IFNAME=eth0 \
    -e NCCL_IB_GID_INDEX=3 \
    -e NCCL_IB_DISABLE=0 \
    -e NCCL_IB_HCA="mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_1" \
    -e NCCL_NET_GDR_LEVEL=2 \
    -e NCCL_IB_QPS_PER_CONNECTION=4 \
    -e NCCL_IB_TC=160 \
    -e NCCL_IB_TIMEOUT=22 \
    -e NCCL_CROSS_NIC=1 \
    -e NCCL_MIN_NCHANNELS=8 \
    -e NCCL_MAX_NCHANNELS=16 \
    -e http_proxy=http://172.16.5.77:8889 \
    -e https_proxy=http://172.16.5.77:8889 \
    llava_vit:25.11.22 \
    bash -c "service ssh restart; bash; "