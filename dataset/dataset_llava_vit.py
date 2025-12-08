import glob
import logging
import math
import os

from dataset.registry import DATASET_REGISTRY

from .properties import Property

logger = logging.getLogger(__file__)

list_coyo = [
    "/vlm/data/coyo400m_part1/coyo700m_00",
    "/vlm/data/coyo400m_part1/coyo700m_01",
    "/vlm/data/coyo400m_part1/coyo700m_02",
    "/vlm/data/coyo400m_part1/coyo700m_03",
    "/vlm/data/coyo400m_part1/coyo700m_04",
    "/vlm/data/coyo400m_part1/coyo700m_05",
    "/vlm/data/coyo400m_part1/coyo700m_06",
    "/vlm/data/coyo400m_part1/coyo700m_07",
    "/vlm/data/coyo400m_part1/coyo700m_08",
    "/vlm/data/coyo400m_part1/coyo700m_09",
    "/vlm/data/coyo400m_part2/coyo700m_10",
    "/vlm/data/coyo400m_part2/coyo700m_11",
    "/vlm/data/coyo400m_part2/coyo700m_12",
    "/vlm/data/coyo400m_part2/coyo700m_13",
    "/vlm/data/coyo400m_part2/coyo700m_14",
    "/vlm/data/coyo400m_part2/coyo700m_15",
    "/vlm/data/coyo400m_part2/coyo700m_16",
    "/vlm/data/coyo400m_part2/coyo700m_17",
    "/vlm/data/coyo400m_part2/coyo700m_18",
    "/vlm/data/coyo400m_part2/coyo700m_19",
    "/vlm/data/coyo400m_part3/coyo700m_20",
    "/vlm/data/coyo400m_part3/coyo700m_21",
    "/vlm/data/coyo400m_part3/coyo700m_22",
    "/vlm/data/coyo400m_part3/coyo700m_24",
    "/vlm/data/coyo400m_part3/coyo700m_25",
    "/vlm/data/coyo400m_part3/coyo700m_26",
    "/vlm/data/coyo400m_part3/coyo700m_27",
    "/vlm/data/coyo400m_part3/coyo700m_28",
    "/vlm/data/coyo400m_part3/coyo700m_29",
    "/vlm/data/coyo400m_part4/coyo700m_30",
    "/vlm/data/coyo400m_part4/coyo700m_31",
]
list_laion = [
    "/vlm/data/LAION224M_HOI31M_IN13M_labeled_2024_03_05/LAION224M_HOI31M_IN13M_labeled_2024_03_05_VM-2-20-tencentos",
    "/vlm/data/LAION224M_HOI31M_IN13M_labeled_2024_03_05/LAION224M_HOI31M_IN13M_labeled_2024_03_05_VM-2-21-tencentos",
    "/vlm/data/LAION224M_HOI31M_IN13M_labeled_2024_03_05/LAION224M_HOI31M_IN13M_labeled_2024_03_05_VM-2-23-tencentos",
    "/vlm/data/LAION224M_HOI31M_IN13M_labeled_2024_03_05/LAION224M_HOI31M_IN13M_labeled_2024_03_05_VM-2-28-tencentos",
    "/vlm/data/LAION224M_HOI31M_IN13M_labeled_2024_03_05/LAION224M_HOI31M_IN13M_labeled_2024_03_05_VM-2-34-tencentos",
    "/vlm/data/LAION224M_HOI31M_IN13M_labeled_2024_03_05/LAION224M_HOI31M_IN13M_labeled_2024_03_05_VM-2-58-tencentos",
    "/vlm/data/LAION224M_HOI31M_IN13M_labeled_2024_03_05/LAION224M_HOI31M_IN13M_labeled_2024_03_05_VM-2-62-tencentos",
    "/vlm/data/LAION224M_HOI31M_IN13M_labeled_2024_03_05/LAION224M_HOI31M_IN13M_labeled_2024_03_05_VM-2-85-tencentos",
]

@DATASET_REGISTRY.register()
def llava_vit_si():
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))

    all_files = list_coyo + list_laion

    # 添加文件检查
    if not all_files:
        raise FileNotFoundError("未找到数据集文件")

    # ========== 分组策略 ==========
    group_size = 4  # 每组 4 张卡（一台机器）
    group_id = rank // group_size
    group_count = max(1, math.ceil(world_size / group_size))

    # 组内 shard 配置
    group_start = group_id * group_size
    group_end = min(world_size, group_start + group_size)
    num_shards = group_end - group_start  # 当前组的实际卡数
    shard_id = rank - group_start         # 当前 rank 在组内的索引

    # ========== 文件分配：按组分配，组内共享 ==========
    # 每个组获取自己的文件子集
    list_prefix = all_files[group_id::group_count]

    # 打印分片信息，方便调试
    print(f"[Rank {rank}] Group {group_id}/{group_count}, "
          f"Shard {shard_id}/{num_shards}, "
          f"分配到 {len(list_prefix)} 个文件")
    if len(list_prefix) > 0:
        print(f"[Rank {rank}] 文件列表: {list_prefix}")

    return Property(
        num_classes=2000000,
        num_examples=0,
        prefixes=list_prefix,
        name="MLCD_in_pfs",
        label_select=0,
        label_start=0,
        num_shards=num_shards,  # 组内卡数
        shard_id=shard_id,      # 组内索引
        dali_type="origin",
        random_diff=10,
    )


@DATASET_REGISTRY.register()
def llava_vit_si_ssd():
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))

    patterns = [
        "/data_*/mlcd/coyo700m*/*.rec",
        "/data_*/mlcd/LAION224M_HOI31M_IN13M_labeled*/*.rec",
    ]

    all_files = [f for pattern in patterns for f in glob.glob(pattern)]
    all_files = [x.replace(".rec", "") for x in all_files]
    all_files = sorted(all_files)

    list_prefix = all_files
    if len(list_prefix) == 0:
        raise RuntimeError(f"No rec prefixes found for patterns: {patterns}")

    # 固定按 8 张卡来划分（目标卡数）
    target_cards = 8

    # 计算每个 prefix 应该被多少张卡读取（尽量均匀分配）
    num_prefix = len(list_prefix)
    base = target_cards // num_prefix
    rem = target_cards % num_prefix
    # 前 rem 个 prefix 多分配 1 张卡
    group_sizes = [base + (1 if i < rem else 0) for i in range(num_prefix)]

    # 计算每个 prefix 在 card 空间中的起始 card index（[0..target_cards-1]）
    start_indices = []
    acc = 0
    for s in group_sizes:
        start_indices.append(acc)
        acc += s

    # 将当前进程的 local_rank 映射到 0..target_cards-1 的 card_index
    # 这样即便 local_rank 超过 8，也可以循环映射到 8 个逻辑卡上
    card_index = local_rank % target_cards

    # 找到 card_index 对应属于哪个 prefix（保证 group_sizes 非零时能找到）
    prefix_idx = None
    for i, start in enumerate(start_indices):
        if group_sizes[i] == 0:
            continue
        if start <= card_index < start + group_sizes[i]:
            prefix_idx = i
            break

    # 如果没有找到（可能的情况：num_prefix > target_cards 且当前 card_index 落在没有分配的 prefix 之后）
    # 在这种情况下，我们将把 card 分配给最后一个有分配的 prefix（安全回退）
    if prefix_idx is None:
        # 找最近一个 group_sizes > 0 的 prefix（应该存在，因为 target_cards > 0）
        for i in range(num_prefix - 1, -1, -1):
            if group_sizes[i] > 0:
                prefix_idx = i
                break

    assigned_prefix = [list_prefix[prefix_idx]]
    shard_id = card_index - start_indices[prefix_idx]
    num_shards = group_sizes[prefix_idx]

    # 打印调试信息，便于确认分配
    logger.info(f"[llava_vit_si_ssd] all_prefixes={list_prefix}")
    logger.info(f"[llava_vit_si_ssd] group_sizes={group_sizes}")
    logger.info(f"[llava_vit_si_ssd] start_indices={start_indices}")
    logger.info(f"[llava_vit_si_ssd] local_rank={local_rank} -> card_index={card_index}")
    logger.info(f"[llava_vit_si_ssd] assigned_prefix={assigned_prefix}, shard_id={shard_id}, num_shards={num_shards}")

    return Property(
        num_classes=2000000,
        num_examples=0,
        prefixes=assigned_prefix,
        name="llava_vit_si_ssd",
        label_select=0,
        label_start=0,
        num_shards=num_shards,
        shard_id=shard_id,
        dali_type="origin",
        random_diff=10,
    )

@DATASET_REGISTRY.register()
def llava_vit_ocr_ssd():
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))

    patterns = [
        "/data_*/llava_vit_ocr_obelics/*ocr_labeled/*.rec",
        "/data_*/llava_vit_ocr_zero250m/*ocr_labeled/*.rec",
    ]

    all_files = [f for pattern in patterns for f in glob.glob(pattern)]
    all_files = [x.replace(".rec", "") for x in all_files]
    all_files = sorted(all_files)

    list_prefix = all_files
    if len(list_prefix) == 0:
        raise RuntimeError(f"No rec prefixes found for patterns: {patterns}")

    # 固定按 8 张卡来划分（目标卡数）
    target_cards = 8

    # 计算每个 prefix 应该被多少张卡读取（尽量均匀分配）
    num_prefix = len(list_prefix)
    base = target_cards // num_prefix
    rem = target_cards % num_prefix
    # 前 rem 个 prefix 多分配 1 张卡
    group_sizes = [base + (1 if i < rem else 0) for i in range(num_prefix)]

    # 计算每个 prefix 在 card 空间中的起始 card index（[0..target_cards-1]）
    start_indices = []
    acc = 0
    for s in group_sizes:
        start_indices.append(acc)
        acc += s

    # 将当前进程的 local_rank 映射到 0..target_cards-1 的 card_index
    # 这样即便 local_rank 超过 8，也可以循环映射到 8 个逻辑卡上
    card_index = local_rank % target_cards

    # 找到 card_index 对应属于哪个 prefix（保证 group_sizes 非零时能找到）
    prefix_idx = None
    for i, start in enumerate(start_indices):
        if group_sizes[i] == 0:
            continue
        if start <= card_index < start + group_sizes[i]:
            prefix_idx = i
            break

    # 如果没有找到（可能的情况：num_prefix > target_cards 且当前 card_index 落在没有分配的 prefix 之后）
    # 在这种情况下，我们将把 card 分配给最后一个有分配的 prefix（安全回退）
    if prefix_idx is None:
        # 找最近一个 group_sizes > 0 的 prefix（应该存在，因为 target_cards > 0）
        for i in range(num_prefix - 1, -1, -1):
            if group_sizes[i] > 0:
                prefix_idx = i
                break

    assigned_prefix = [list_prefix[prefix_idx]]
    shard_id = card_index - start_indices[prefix_idx]
    num_shards = group_sizes[prefix_idx]

    # 打印调试信息，便于确认分配
    logger.info(f"[llava_vit_ocr_ssd] all_prefixes={list_prefix}")
    logger.info(f"[llava_vit_ocr_ssd] group_sizes={group_sizes}")
    logger.info(f"[llava_vit_ocr_ssd] start_indices={start_indices}")
    logger.info(f"[llava_vit_ocr_ssd] local_rank={local_rank} -> card_index={card_index}")
    logger.info(f"[llava_vit_ocr_ssd] assigned_prefix={assigned_prefix}, shard_id={shard_id}, num_shards={num_shards}")

    return Property(
        num_classes=365187,
        num_examples=0,
        prefixes=assigned_prefix,
        name="llava_vit_ocr_ssd",
        label_select=0,
        label_start=0,
        num_shards=num_shards,
        shard_id=shard_id,
        dali_type="origin",
        random_diff=100,
    )


@DATASET_REGISTRY.register()
def howto100m_kinetics_104948429_400000_split_80():
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))

    assert world_size <= 80

    list_mp4_label_path = f"/video_vit/dataset/configs_for_llava_vit_versions_0_0_0/preshuffled_trainset_split_80/preshuffled_trainset_part.lst_{rank:03d}"
    # with open(list_mp4_label, "r", encoding="utf-8") as f:
        # lines = f.readlines()

    # lines = [x.strip().split(",")[0] for x in lines]
    return Property(
        name="howto100m_kinetics_104948429_400000_split_80",
        prefixes=[list_mp4_label_path],
        num_classes=400000,
        num_examples=104948429 // world_size,
        num_shards=1,
        shard_id=0,
        dali_type="decord",
    )


@DATASET_REGISTRY.register()
def howto100m_kinetics_104948429_400000_split_128():
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))

    assert world_size <= 128

    list_mp4_label_path = f"/video_vit/dataset/configs_for_llava_vit_versions_0_0_0/preshuffled_trainset_split_128/preshuffled_trainset_part.lst_{rank:03d}"
    # with open(list_mp4_label, "r", encoding="utf-8") as f:
        # lines = f.readlines()

    # lines = [x.strip().split(",")[0] for x in lines]
    return Property(
        name="howto100m_kinetics_104948429_400000_split_128",
        prefixes=[list_mp4_label_path],
        num_classes=400000,
        num_examples=104948429 // world_size,
        num_shards=1,
        shard_id=0,
        dali_type="decord",
    )

@DATASET_REGISTRY.register()
def howto100m_panda70m_kinetics_126409811_400000_split_128():
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))

    assert world_size <= 128

    list_mp4_label_path = f"/video_vit/dataset/configs_for_llava_vit_versions_0_0_1_add_pandas70M/train_how_to_100m_panda70m_k710_split_128/part_{rank:03d}"
    # with open(list_mp4_label, "r", encoding="utf-8") as f:
        # lines = f.readlines()

    # lines = [x.strip().split(",")[0] for x in lines]
    return Property(
        name="howto100m_panda70m_kinetics_126409811_400000_split_128",
        prefixes=[list_mp4_label_path],
        num_classes=400000,
        num_examples=104948429 // world_size,
        num_shards=1,
        shard_id=0,
        dali_type="decord",
    )


@DATASET_REGISTRY.register()
def configs_for_llava_vit_versions_0_0_2_add_pandas70M():
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))

    assert world_size <= 128

    list_mp4_label_path = f"/video_vit/dataset/configs_for_llava_vit_versions_0_0_2_add_pandas70M/train_how_to_100m_panda70m_k710_split_128/part_{rank:03d}"
    # with open(list_mp4_label, "r", encoding="utf-8") as f:
        # lines = f.readlines()

    # lines = [x.strip().split(",")[0] for x in lines]
    return Property(
        name="configs_for_llava_vit_versions_0_0_2_add_pandas70M",
        prefixes=[list_mp4_label_path],
        num_classes=400000,
        num_examples=116693632 // world_size,
        num_shards=1,
        shard_id=0,
        dali_type="decord",
    )


@DATASET_REGISTRY.register()
def configs_for_llava_vit_versions_0_0_2_add_pandas70M_filtered():
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))

    assert world_size <= 128

    list_mp4_label_path = f"/video_vit/dataset/configs_for_llava_vit_versions_0_0_2_add_pandas70M/train_how_to_100m_panda70m_k710_filtered_split_128/part_{rank:03d}"
    # with open(list_mp4_label, "r", encoding="utf-8") as f:
        # lines = f.readlines()

    # lines = [x.strip().split(",")[0] for x in lines]
    return Property(
        name="configs_for_llava_vit_versions_0_0_2_add_pandas70M_filtered",
        prefixes=[list_mp4_label_path],
        num_classes=400000,
        num_examples=1269187 ** 128,
        num_shards=1,
        shard_id=0,
        dali_type="decord",
    )


@DATASET_REGISTRY.register()
def configs_for_llava_vit_versions_0_0_2_add_pandas70M_split_80():
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))

    assert world_size <= 80

    list_mp4_label_path = f"/video_vit/dataset/configs_for_llava_vit_versions_0_0_2_add_pandas70M/train_how_to_100m_panda70m_k710_split_80/part_{rank:03d}"
    # with open(list_mp4_label, "r", encoding="utf-8") as f:
        # lines = f.readlines()

    # lines = [x.strip().split(",")[0] for x in lines]
    return Property(
        name="configs_for_llava_vit_versions_0_0_2_add_pandas70M",
        prefixes=[list_mp4_label_path],
        num_classes=400000,
        num_examples=116693632 // world_size,
        num_shards=1,
        shard_id=0,
        dali_type="decord",
    )

@DATASET_REGISTRY.register()
def fake_data():
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    return Property(
        name="fake_data",
        prefixes=["/video_vit/xiangan/LLaVA-ViT/fake_data/fake_data"],
        num_classes=10000,
        num_examples=0,
        num_shards=world_size,
        shard_id=rank,
        dali_type="origin",
    )
