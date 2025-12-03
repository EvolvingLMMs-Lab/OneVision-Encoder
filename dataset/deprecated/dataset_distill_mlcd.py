import os
import glob

from dataset.registry import DATASET_REGISTRY

from .properties import Property


@DATASET_REGISTRY.register()
def RICE_in_pfs():
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))

    patterns = [
        "/vlm/data/coyo400m_resized448/*.rec",
        "/vlm/data/LAION224M_HOI31M_IN13M_labeled_2024_03_05/*.rec",
    ]
    all_files = [f for pattern in patterns for f in glob.glob(pattern)]
    all_files = [x.replace(".rec", "") for x in all_files]
    all_files = sorted(all_files)

    # 添加文件检查
    if not all_files:
        raise FileNotFoundError("未找到数据集文件")

    # 计算节点信息
    gpus_per_node = 8
    node_rank = rank // gpus_per_node
    num_nodes = world_size // gpus_per_node
    list_prefix = all_files[node_rank::num_nodes]

    return Property(
        num_classes=2000000,
        num_examples=0,
        prefixes=list_prefix,
        name="MLCD_in_pfs",
        label_select=0,
        label_start=0,
        num_shards=8,
        shard_id=local_rank,
        dali_type="origin",
        random_diff=10,
    )


@DATASET_REGISTRY.register()
def distill_mlcd_coyo_laion_shuffled():
    """ 蒸馏MLCD"""
    rank = int(os.getenv("RANK", "0"))          # 全局进程排名 (Global process rank)
    world_size = int(os.getenv("WORLD_SIZE", "1"))  # 总进程数 (Total number of processes)

    with open("/video_vit/pretrain_video_datas/mlcd/list_coyo", "r", encoding="utf-8") as f:
        list_coyo = f.readlines()
    list_coyo = [x.strip() for x in list_coyo]

    with open("/video_vit/pretrain_video_datas/mlcd/list_laion", "r", encoding="utf-8") as f:
        list_laion = f.readlines()
    list_laion = [x.strip() for x in list_laion]
    list_prefix = list_laion + list_coyo
    list_prefix = list_prefix * 10

    import random
    seed = 42
    random.shuffle(list_prefix, random.seed(seed))
    list_prefix = list_prefix * 10
    list_prefix = list_prefix[rank::world_size]

    list_prefix = [x.replace(".rec", "") for x in list_prefix]
    list_prefix = [x for x in list_prefix if os.path.exists(f"{x}.rec") and os.path.exists(f"{x}.idx")]

    return Property(
        prefixes=list_prefix,
        name="coyo_laion",
        num_classes=2_000_000,
        num_examples=0,
        num_shards=1,
        shard_id=0,
        dali_type="origin",
    )



@DATASET_REGISTRY.register()
def mlcd_coyo_laion():
    """ mlcd_coyo_laion"""
    rank = int(os.getenv("RANK", "0"))          # 全局进程排名 (Global process rank)
    world_size = int(os.getenv("WORLD_SIZE", "1"))  # 总进程数 (Total number of processes)

    with open("/video_vit/pretrain_video_datas/mlcd/list_coyo", "r", encoding="utf-8") as f:
        list_coyo = f.readlines()
    list_coyo = [x.strip() for x in list_coyo]

    with open("/video_vit/pretrain_video_datas/mlcd/list_laion", "r", encoding="utf-8") as f:
        list_laion = f.readlines()
    list_laion = [x.strip() for x in list_laion]
    list_prefix = list_laion + list_coyo

    import random
    seed = 42
    random.shuffle(list_prefix, random.seed(seed))
    list_prefix = list_prefix * 10
    list_prefix = list_prefix[rank::world_size]

    list_prefix = [x.replace(".rec", "") for x in list_prefix]
    list_prefix = [x for x in list_prefix if os.path.exists(f"{x}.rec") and os.path.exists(f"{x}.idx")]

    return Property(
        prefixes=list_prefix,
        name="coyo_laion",
        num_classes=2000000,
        num_examples=0,
        num_shards=1,
        shard_id=0,
        dali_type="origin",
    )
