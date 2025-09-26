import os

from dataset.registry import DATASET_REGISTRY

from .properties import Property

# 获取分布式训练环境变量 (Get distributed training environment variables)
rank = int(os.getenv("RANK", "0"))          # 全局进程排名 (Global process rank)
local_rank = int(os.getenv("LOCAL_RANK", "0"))  # 本地进程排名 (Local process rank)
world_size = int(os.getenv("WORLD_SIZE", "1"))  # 总进程数 (Total number of processes)


list_prefix = [
    "/data_7/dataset_k710/train_k710"
]


_mlcd_k710 = Property(
    num_class=710,
    num_example=0,
    name="mlcd_k710",
    prefix=list_prefix,
    num_shards=world_size,
    shard_id=rank,
    label_select=0,
    label_start=0,
    dali_type="video",
    random_diff=1,
    pfc_type="parallel_softmax",
)

@DATASET_REGISTRY.register()
def mlcd_k710():
    return _mlcd_k710
