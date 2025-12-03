import os

import numpy as np

from dataset.registry import DATASET_REGISTRY

from .properties import Property

# 获取分布式训练环境变量 (Get distributed training environment variables)
rank = int(os.getenv("RANK", "0"))          # 全局进程排名 (Global process rank)
local_rank = int(os.getenv("LOCAL_RANK", "0"))  # 本地进程排名 (Local process rank)
world_size = int(os.getenv("WORLD_SIZE", "1"))  # 总进程数 (Total number of processes)


list_prefix = "NULL"


if os.path.exists("/data_7/list_mp4_preprocessed_PE_Video"):
    assert os.path.exists("/data_7/label_preprocessed_PE_Video.npy")

    with open("/data_7/list_mp4_preprocessed_PE_Video", "r") as f:
        lines = f.readlines()
        lines = [x.strip() for x in lines]
    label = np.load("/data_7/label_preprocessed_PE_Video.npy")

    list_prefix = []
    for i in range(len(label)):
        list_prefix.append(
            (lines[i], label[i])
        )


# list_prefix = list_prefix[:1000]

_mlcd_peVideo_decord_torch_100k = Property(
    num_class=100_000,
    num_example=0,
    name="mlcd_peVideo_decord_torch_100k",
    prefix=list_prefix,
    num_shards=world_size,
    shard_id=rank,
    label_select=0,
    label_start=0,
    dali_type="decord_torch",
    random_diff=10,
    pfc_type="partial_fc",
)

@DATASET_REGISTRY.register()
def mlcd_peVideo_decord_torch_100k():
    return _mlcd_peVideo_decord_torch_100k
