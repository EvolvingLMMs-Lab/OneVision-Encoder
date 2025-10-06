import os

from dataset.registry import DATASET_REGISTRY

from .properties import Property


@DATASET_REGISTRY.register()
def distill_mlcd_coyo_laion():
    """ 蒸馏MLCD"""
    rank = int(os.getenv("RANK", "0"))          # 全局进程排名 (Global process rank)
    world_size = int(os.getenv("WORLD_SIZE", "1"))  # 总进程数 (Total number of processes)

    if rank % 2 == 0:
        with open("/video_vit/pretrain_video_datas/mlcd/list_coyo", "r", encoding="utf-8") as f:
            list_coyo = f.readlines()
        print(list_coyo)
        list_coyo = [x.strip() for x in list_coyo]
        list_prefix = list_coyo
    else:
        with open("/video_vit/pretrain_video_datas/mlcd/list_laion", "r", encoding="utf-8") as f:
            list_laion = f.readlines()
        list_laion = [x.strip() for x in list_laion]
        list_prefix = list_laion

    list_prefix = [x.replace(".rec", "") for x in list_prefix]
    print(list_prefix)
    list_prefix = [x for x in list_prefix if os.path.exists(f"{x}.rec") and os.path.exists(f"{x}.idx")]
    return Property(
        prefix=list_prefix,
        name="coyo_laion",
        num_example=0,
        num_shards=world_size // 2,
        shard_id=rank // 2,
        dali_type="origin",
    )
