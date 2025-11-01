import os
import numpy as np

from dataset.registry import DATASET_REGISTRY

from .properties import Property

rank = int(os.getenv("RANK", "0"))              # 全局进程排名 (Global process rank)
local_rank = int(os.getenv("LOCAL_RANK", "0"))  # 本地进程排名 (Local process rank)
world_size = int(os.getenv("WORLD_SIZE", "1"))  # 总进程数 (Total number of processes)


# name: str,                      # Dataset name / 数据集名称
# prefixes: List[str],            # Data path prefixes / 数据路径前缀（多个）
# num_classes: int,               # Total number of classes / 类别总数
# num_examples: int,              # Total number of examples / 样本总数
# label_start: int = 0,           # Label starting offset / 标签起始偏移
# label_select: int = 0,          # Label selection index / 标签选择索引
# num_shards: int = world_size,   # Number of data shards / 数据分片数量
# shard_id: int = rank,           # Current process shard ID / 当前进程的分片ID
# dali_type: str = "origin",      # Data loader type / 数据加载器类型
# random_diff: int = 10,          # Random difference parameter / 随机差异参数
# pfc_types: tuple = ("partial_fc",),  # PFC variants / PFC 类型（可多选）
# mp4_list_path: Optional[str] = None,    # Path to mp4 list file / mp4列表文件路径
# label_list_path: Optional[str] = None,  # Path to label list file / 标签列表文件路径


@DATASET_REGISTRY.register()
def k710_ssv2_univit_pfs_ip_path():
    """
    """
    with open("/video_vit/train_UniViT/mp4_list.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()
    lines = [x.strip().split(",")[0] for x in lines]
    return Property(
        name="k710_ssv2_diving48_pfs",
        prefixes=lines,
        num_classes=200000,
        num_examples=6089886,
        num_shards=world_size,
        shard_id=rank,
        dali_type="decord",
        label=["/video_vit/train_UniViT/list_merged.npy", "/video_vit/train_UniViT/merged_visible_indices_uint16.npy"]
    )
