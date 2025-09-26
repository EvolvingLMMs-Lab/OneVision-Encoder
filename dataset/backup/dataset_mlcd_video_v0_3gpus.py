import os

from .properties import Property
from dataset.registry import DATASET_REGISTRY

# 获取分布式训练环境变量 (Get distributed training environment variables)
rank = int(os.getenv("RANK", "0"))          # 全局进程排名 (Global process rank)
local_rank = int(os.getenv("LOCAL_RANK", "0"))  # 本地进程排名 (Local process rank)
world_size = int(os.getenv("WORLD_SIZE", "1"))  # 总进程数 (Total number of processes)

# 视频数据集路径列表 (Video dataset path list)
mlcd_video_n1_v0_lp = [
    "/data_2/datasets_video_mlcd/pre_shuffled_train_00_10k_20k_50k_100k_200k",
    "/data_3/datasets_video_mlcd/pre_shuffled_train_01_10k_20k_50k_100k_200k",
    "/data_6/datasets_video_mlcd/pre_shuffled_train_02_10k_20k_50k_100k_200k",
    "/data_7/datasets_video_mlcd/pre_shuffled_train_03_10k_20k_50k_100k_200k",
]

# n1: 单节点配置 (single node configuration)
# lp: list_prefix 路径前缀列表
# 数据分配逻辑 (Data distribution logic):
# 四个SSD分给八张卡 (Four SSDs distributed to eight GPUs)
# 第一个SSD给 local_rank 0, 1
# 第二个SSD给 local_rank 2, 3
# 第三个SSD给 local_rank 4, 5
# 第四个SSD给 local_rank 6, 7
mlcd_video_n1_v0_lp = mlcd_video_n1_v0_lp[local_rank // 2: local_rank // 2 + 1]

# 数据分片逻辑 (Data sharding logic):
# 每个rec文件被分成2片 (Each rec file is divided into 2 shards)
# local_rank 0 使用 shard_id 0
# local_rank 1 使用 shard_id 1
# local_rank 2 使用 shard_id 0
# local_rank 3 使用 shard_id 1
# 以此类推 (and so on)

num_nodes = world_size // 8
node_rank = rank // 8
num_shards = 2 * num_nodes
shards_id = local_rank % 2 + node_rank * 2

if os.environ["HOSTNAME"] in \
    [
        "VM-2-58-tencentos",
        "VM-2-85-tencentos",
        "VM-2-39-tencentos",
    ]:
    # 创建不同规模的数据集配置 (Create dataset configurations of different scales)
    _mlcd_video_n1_v0_20k = Property(
        num_class=20_000,        # 类别总数 (Total number of classes)
        num_example=0,           # 样本数量（0表示自动计算）(Number of examples, 0 means auto-calculation)
        name="mlcd_video_n1_v0_20k",
        prefix=mlcd_video_n1_v0_lp,  # 数据路径 (Data path)
        num_shards=num_shards,   # 数据分片数 (Number of data shards)
        shard_id=shards_id,      # 当前进程使用的分片ID (Shard ID used by current process)
        label_select=10,         # 标签选择参数 (Label selection parameter)
        label_start=0,           # 标签起始位置 (Label start position)
        dali_type="video",       # 数据类型为视频 (Data type is video)
    )
    # 50k类别配置 (50k classes configuration) 
    _mlcd_video_n1_v0_50k = Property(
        num_class=50_000,
        num_example=0,
        name="mlcd_video_n1_v0_50k",
        prefix=mlcd_video_n1_v0_lp,
        num_shards=num_shards,   # 数据分片数 (Number of data shards)
        shard_id=shards_id,      # 当前进程使用的分片ID (Shard ID used by current process)
        label_select=20,
        label_start=0,
        dali_type="video",
    )
    # 100k类别配置 (100k classes configuration)
    _mlcd_video_n1_v0_100k = Property(
        num_class=100_000,
        num_example=0,
        name="mlcd_video_n1_v0_100k",
        prefix=mlcd_video_n1_v0_lp,
        num_shards=num_shards,   # 数据分片数 (Number of data shards)
        shard_id=shards_id,      # 当前进程使用的分片ID (Shard ID used by current process)
        label_select=30,
        label_start=0,
        dali_type="video",
    )
    # 200k类别配置 (200k classes configuration)
    _mlcd_video_n1_v0_200k = Property(
        num_class=200_000,
        num_example=0,
        name="mlcd_video_n1_v0_200k",
        prefix=mlcd_video_n1_v0_lp,
        num_shards=num_shards,   # 数据分片数 (Number of data shards)
        shard_id=shards_id,      # 当前进程使用的分片ID (Shard ID used by current process)
        label_select=40,
        label_start=0,
        dali_type="video",
    )
else:
    # 非指定主机设置为None (Set to None for non-specified hosts)
    _mlcd_video_n1_v0_20k = None
    _mlcd_video_n1_v0_50k = None
    _mlcd_video_n1_v0_100k = None
    _mlcd_video_n1_v0_200k = None


# 注册各个数据集到注册表 (Register datasets to the registry)
@DATASET_REGISTRY.register()
def mlcd_video_n1_v0_3gpus_20k():
    """返回20k类别的视频数据集 (Return video dataset with 20k classes)"""
    return _mlcd_video_n1_v0_20k


@DATASET_REGISTRY.register()
def mlcd_video_n1_v0_3gpus_50k():
    """返回50k类别的视频数据集 (Return video dataset with 50k classes)"""
    return _mlcd_video_n1_v0_50k

@DATASET_REGISTRY.register()
def mlcd_video_n1_v0_3gpus_100k():
    """返回100k类别的视频数据集 (Return video dataset with 100k classes)"""
    return _mlcd_video_n1_v0_100k

@DATASET_REGISTRY.register()
def mlcd_video_n1_v0_3gpus_200k():
    """返回200k类别的视频数据集 (Return video dataset with 200k classes)"""
    return _mlcd_video_n1_v0_200k
