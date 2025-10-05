import os

from dataset.registry import DATASET_REGISTRY

from .properties import Property

rank = int(os.getenv("RANK", "0"))          # 全局进程排名 (Global process rank)
local_rank = int(os.getenv("LOCAL_RANK", "0"))  # 本地进程排名 (Local process rank)
world_size = int(os.getenv("WORLD_SIZE", "1"))  # 总进程数 (Total number of processes)


@DATASET_REGISTRY.register()
def k710_ssv2_diving48_ssd():
    """ 临时使用 tmpfs 进行训练"""
    with open("/data_1/video_vit/list_mp4", "r", encoding="utf-8") as f:
        lines = f.readlines()
    lines = [x.strip().split(",")[0] for x in lines]
    return Property(
        prefix=lines,
        name="k710_ssv2_diving48_ssd",
        num_example=0,
        num_shards=world_size,
        shard_id=rank,
        dali_type="decord",
    )
