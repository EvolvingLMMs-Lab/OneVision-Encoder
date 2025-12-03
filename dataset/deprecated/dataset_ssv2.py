import os

import numpy as np

from dataset.registry import DATASET_REGISTRY
from .properties import Property

rank = int(os.getenv("RANK", "0"))          # 全局进程排名 (Global process rank)
local_rank = int(os.getenv("LOCAL_RANK", "0"))  # 本地进程排名 (Local process rank)
world_size = int(os.getenv("WORLD_SIZE", "1"))  # 总进程数 (Total number of processes)


@DATASET_REGISTRY.register()
def ssv2_tmpfs():
    """ 临时使用 tmpfs 进行训练"""
    with open("/train_tmp/ssv2_train_new.csv", "r", encoding="utf-8") as f:
        lines = f.readlines()
    lines = [x.strip().split(",")[0] for x in lines]
    lines = [f"/train_tmp/{x}" for x in lines]
    return Property(
        prefix=lines,
        name="ssv2_tmpfs",
        num_example=0,
        num_shards=world_size,
        shard_id=rank,
        dali_type="decord",
    )
