import os

import numpy as np

from dataset.registry import DATASET_REGISTRY

from .properties import Property

rank = int(os.getenv("RANK", "0"))          # 全局进程排名 (Global process rank)
local_rank = int(os.getenv("LOCAL_RANK", "0"))  # 本地进程排名 (Local process rank)
world_size = int(os.getenv("WORLD_SIZE", "1"))  # 总进程数 (Total number of processes)


list_prefix = []


def init_prefix():
    list_prefix = []

    path = "/video_vit/train_UniViT/videos_frames64_kinetics_ssv2/list_videos_frames64_kinetics_ssv2_new"
    if os.path.exists(path):
        with open(path, "r") as f:
            lines = f.readlines()
            lines = [x.strip() for x in lines]
    
    for i in range(len(lines)):
        list_prefix.append(lines[i])
    
    return list_prefix



@DATASET_REGISTRY.register()
def ssv2_v0():
    list_prefix = init_prefix()
    _ssv2_v0 = Property(
        prefix=list_prefix,
        name="ssv2_v0",
        num_example=0,
        num_shards=8,
        shard_id=local_rank,
        dali_type="decord",
        pfc_type=["unmask", "mask"],
    )
    return _ssv2_v0