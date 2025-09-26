import os

import numpy as np

from dataset.registry import DATASET_REGISTRY

from .properties import Property

# 获取分布式训练环境变量 (Get distributed training environment variables)
rank = int(os.getenv("RANK", "0"))          # 全局进程排名 (Global process rank)
local_rank = int(os.getenv("LOCAL_RANK", "0"))  # 本地进程排名 (Local process rank)
world_size = int(os.getenv("WORLD_SIZE", "1"))  # 总进程数 (Total number of processes)


list_prefix = []


if os.path.exists("/data_2/list_videos_frames64_kinetics_ssv2"):
    assert os.path.exists("/data_2/labels_kinetics_200000.npy")

    with open("/data_2/list_videos_frames64_kinetics_ssv2", "r") as f:
        lines = f.readlines()
        lines = [x.strip() for x in lines]
    label = np.load("/data_2/labels_kinetics_200000.npy")

    list_prefix = []
    for i in range(len(label)):
        list_prefix.append(
            (lines[i], label[i])
        )

if os.path.exists("/data_2/list_videos_frames64_ssv2"):
    assert os.path.exists("/data_2/labels_ssv2_200000.npy")

    with open("/data_2/list_videos_frames64_ssv2", "r") as f:
        lines = f.readlines()
        lines = [x.strip() for x in lines]
    label = np.load("/data_2/labels_ssv2_200000.npy")

    for i in range(len(label)):
        list_prefix.append(
            (lines[i], label[i])
        )


_mlcd_video_v4_6gpus_decord = Property(
    num_class=200_000,
    num_example=0,
    name="mlcd_video_v4_6gpus_decord",
    prefix=list_prefix,
    num_shards=8,
    shard_id=local_rank,
    label_select=0,
    label_start=0,
    dali_type="decord",
    random_diff=10,
    pfc_type="partial_fc",
)

@DATASET_REGISTRY.register()
def mlcd_video_v4_6gpus_decord():
    return _mlcd_video_v4_6gpus_decord
