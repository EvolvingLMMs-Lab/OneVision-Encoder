import os
import random

import numpy as np

from dataset.registry import DATASET_REGISTRY

from .properties import Property

# 获取分布式训练环境变量 (Get distributed training environment variables)
rank = int(os.getenv("RANK", "0"))          # 全局进程排名 (Global process rank)
local_rank = int(os.getenv("LOCAL_RANK", "0"))  # 本地进程排名 (Local process rank)
world_size = int(os.getenv("WORLD_SIZE", "1"))  # 总进程数 (Total number of processes)


mlcd_video_v2_6gpus = None


list_prefix = [
    "/data_3/dataset_mlcd_video_v2_6gpus/k710_ssv2_20k_50k_100k_200k_500k_data_2",
    "/data_3/dataset_mlcd_video_v2_6gpus/k710_ssv2_20k_50k_100k_200k_500k_data_3",
    "/data_3/dataset_mlcd_video_v2_6gpus/k710_ssv2_20k_50k_100k_200k_500k_data_6",
    "/data_3/dataset_mlcd_video_v2_6gpus/k710_ssv2_20k_50k_100k_200k_500k_data_7",
    "/data_6/dataset_mlcd_video_v2_6gpus/intern_vid_train_VM-2-58-tencentos_20k_50k_100k_200k_500k",
    "/data_7/dataset_mlcd_video_v2_6gpus/intern_vid_train_VM-2-85-tencentos_20k_50k_100k_200k_500k",
] * 1000

node_rank = rank // 8
# Create a random number generator with fixed seed
rng = np.random.RandomState(node_rank)
indices = rng.permutation(len(list_prefix))
list_prefix = [list_prefix[i] for i in indices]


_mlcd_video_v2_6gpus_20k = Property(
    num_class=20_000,
    num_example=0,
    name="mlcd_video_v2_6gpus_20k",
    prefix=list_prefix,
    num_shards=8,
    shard_id=local_rank,
    label_select=0,
    label_start=0,
    dali_type="video",
)
_mlcd_video_v2_6gpus_50k = Property(
    num_class=50_000,
    num_example=0,
    name="mlcd_video_v2_6gpus_50k",
    prefix=list_prefix,
    num_shards=8,
    shard_id=local_rank,
    label_select=10,
    label_start=0,
    dali_type="video",
)
_mlcd_video_v2_6gpus_100k = Property(
    num_class=100_000,
    num_example=0,
    name="mlcd_video_v2_6gpus_100k",
    prefix=list_prefix,
    num_shards=8,
    shard_id=local_rank,
    label_select=20,
    label_start=0,
    dali_type="video",
)
_mlcd_video_v2_6gpus_200k = Property(
    num_class=200_000,
    num_example=0,
    name="mlcd_video_v2_6gpus_200k",
    prefix=list_prefix,
    num_shards=8,
    shard_id=local_rank,
    label_select=30,
    label_start=0,
    dali_type="video",
)
_mlcd_video_v2_6gpus_500k = Property(
    num_class=500_000,
    num_example=0,
    name="mlcd_video_v2_6gpus_500k",
    prefix=list_prefix,
    num_shards=8,
    shard_id=local_rank,
    label_select=40,
    label_start=0,
    dali_type="video",
)



@DATASET_REGISTRY.register()
def mlcd_video_v2_6gpus_20k():
    return _mlcd_video_v2_6gpus_20k


@DATASET_REGISTRY.register()
def mlcd_video_v2_6gpus_50k():
    return _mlcd_video_v2_6gpus_50k


@DATASET_REGISTRY.register()
def mlcd_video_v2_6gpus_100k():
    return _mlcd_video_v2_6gpus_100k


@DATASET_REGISTRY.register()
def mlcd_video_v2_6gpus_200k():
    return _mlcd_video_v2_6gpus_200k


@DATASET_REGISTRY.register()
def mlcd_video_v2_6gpus_500k():
    return _mlcd_video_v2_6gpus_500k