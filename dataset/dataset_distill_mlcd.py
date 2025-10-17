import os

from dataset.registry import DATASET_REGISTRY

from .properties import Property


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
        num_classes=2_000_000,
        num_examples=0,
        num_shards=1,
        shard_id=0,
        dali_type="origin",
    )
