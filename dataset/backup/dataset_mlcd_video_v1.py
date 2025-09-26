import os

from .properties import Property
from dataset.registry import DATASET_REGISTRY

# 获取分布式训练环境变量 (Get distributed training environment variables)
rank = int(os.getenv("RANK", "0"))          # 全局进程排名 (Global process rank)
local_rank = int(os.getenv("LOCAL_RANK", "0"))  # 本地进程排名 (Local process rank)
world_size = int(os.getenv("WORLD_SIZE", "1"))  # 总进程数 (Total number of processes)

# 视频数据集路径列表 (Video dataset path list)
hostname = os.environ["HOSTNAME"]  # 获取主机名 (Get hostname)

mlcd_video_v1 = None
list_prefix = None


if hostname == "VM-2-85-tencentos":

    if local_rank < 4:
        list_prefix_select = [
            "/data_2/dataset_mlcd_video_v1/k710_ssv2_20k_50k_100k_200k_500k",
            "/data_3/dataset_mlcd_video_v1/k710_ssv2_20k_50k_100k_200k_500k",
            "/data_6/dataset_mlcd_video_v1/k710_ssv2_20k_50k_100k_200k_500k",
            "/data_7/dataset_mlcd_video_v1/k710_ssv2_20k_50k_100k_200k_500k",
        ]
        list_prefix = list_prefix_select[local_rank:local_rank+1]
        num_shards = 1
        shard_id = 0
    else:

        list_prefix = [
            "/data_7/dataset_mlcd_video_v1/intern_vid_train_VM-2-85-tencentos_20k_50k_100k_200k_500k"
        ]
        num_shards = 4
        shard_id = local_rank - 4

elif hostname == "VM-2-58-tencentos":
    list_prefix = [
        "/data_7/dataset_mlcd_video_v1/intern_vid_train_VM-2-58-tencentos_20k_50k_100k_200k_500k"
    ]
    num_shards = 8
    shard_id = local_rank
else:
    raise


if os.environ["HOSTNAME"] in \
    [
        "VM-2-58-tencentos",
        "VM-2-85-tencentos",
    ]:

    _mlcd_video_v1_20k = Property(
        num_class=20_000,
        num_example=0,
        name="mlcd_video_v1_20k",
        prefix=list_prefix,
        num_shards=num_shards,
        shard_id=shard_id,
        label_select=0,
        label_start=0,
        dali_type="video",
    )
    _mlcd_video_v1_50k = Property(
        num_class=50_000,
        num_example=0,
        name="mlcd_video_v1_50k",
        prefix=list_prefix,
        num_shards=num_shards,
        shard_id=shard_id,
        label_select=10,
        label_start=0,
        dali_type="video",
    )
    _mlcd_video_v1_100k = Property(
        num_class=100_000,
        num_example=0,
        name="mlcd_video_v1_100k",
        prefix=list_prefix,
        num_shards=num_shards,
        shard_id=shard_id,
        label_select=20,
        label_start=0,
        dali_type="video",
    )
    _mlcd_video_v1_200k = Property(
        num_class=200_000,
        num_example=0,
        name="mlcd_video_v1_200k",
        prefix=list_prefix,
        num_shards=num_shards,
        shard_id=shard_id,
        label_select=30,
        label_start=0,
        dali_type="video",
    )
    _mlcd_video_v1_500k = Property(
        num_class=500_000,
        num_example=0,
        name="mlcd_video_v1_500k",
        prefix=list_prefix,
        num_shards=num_shards,
        shard_id=shard_id,
        label_select=40,
        label_start=0,
        dali_type="video",
    )
else:
    _mlcd_video_v1_20k = None
    _mlcd_video_v1_50k = None
    _mlcd_video_v1_100k = None
    _mlcd_video_v1_200k = None
    _mlcd_video_v1_500k = None


@DATASET_REGISTRY.register()
def mlcd_video_v1_20k():
    return _mlcd_video_v1_20k


@DATASET_REGISTRY.register()
def mlcd_video_v1_50k():
    return _mlcd_video_v1_50k


@DATASET_REGISTRY.register()
def mlcd_video_v1_100k():
    return _mlcd_video_v1_100k


@DATASET_REGISTRY.register()
def mlcd_video_v1_200k():
    return _mlcd_video_v1_200k


@DATASET_REGISTRY.register()
def mlcd_video_v1_500k():
    return _mlcd_video_v1_500k