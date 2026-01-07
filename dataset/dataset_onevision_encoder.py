import glob
import logging
import os

from dataset.registry import DATASET_REGISTRY

from .properties import Property


logger = logging.getLogger(__file__)

rank = int(os.getenv("RANK", "0"))
local_rank = int(os.getenv("LOCAL_RANK", "0"))
world_size = int(os.getenv("WORLD_SIZE", "1"))


@DATASET_REGISTRY.register()
def onevision_encoder_si_dry_run():
    """Dry run configuration for onevision encoder SI dataset.

    This function creates a minimal configuration with no actual data prefixes,
    useful for testing and debugging without loading real datasets.

    Returns:
        Property: Dataset property configuration with empty prefixes and basic settings.
    """
    return Property(
        num_classes=20000,
        num_examples=0,
        prefixes=[],
        name="onevision_encoder_si_dry_run",
        label_select=0,
        label_start=0,
        num_shards=world_size,
        shard_id=local_rank,
        dali_type="origin",
        random_diff=10,
    )


@DATASET_REGISTRY.register()
def onevision_encoder_si_ssd():
    """Configure onevision encoder SI dataset from SSD storage.

    This function loads COYO-700M and LAION datasets from SSD storage and distributes
    the data prefixes across 8 target cards.  Each card is assigned specific prefixes
    based on the local rank, ensuring even distribution of data shards.

    The function handles cases where the number of prefixes doesn't divide evenly
    by the target card count, allocating extra cards to the first few prefixes.

    Returns:
        Property: Dataset property configuration with assigned prefixes and shard information.

    Raises:
        RuntimeError: If no .rec files are found matching the specified patterns.
    """
    patterns = [
        "/data_*/coyo400m/*.rec",
        "/data_*/laion260m/*.rec",
    ]

    all_files = [f for pattern in patterns for f in glob.glob(pattern)]
    all_files = [x.replace(".rec", "") for x in all_files]
    all_files = sorted(all_files)

    list_prefix = all_files
    if len(list_prefix) == 0:
        raise RuntimeError(f"No rec prefixes found for patterns: {patterns}")

    # Fixed division by 8 cards (target card count)
    target_cards = 8

    # Calculate how many cards should read each prefix (distribute evenly)
    num_prefix = len(list_prefix)
    base = target_cards // num_prefix
    rem = target_cards % num_prefix
    # Allocate 1 extra card to the first rem prefixes
    group_sizes = [base + (1 if i < rem else 0) for i in range(num_prefix)]

    # Calculate the starting card index for each prefix in card space ([0..target_cards-1])
    start_indices = []
    acc = 0
    for s in group_sizes:
        start_indices.append(acc)
        acc += s

    # Map current process's local_rank to card_index in range [0..target_cards-1]
    # This way, even if local_rank exceeds 8, it can cyclically map to 8 logical cards
    card_index = local_rank % target_cards

    # Find which prefix the card_index belongs to (ensure it can be found when group_sizes is non-zero)
    prefix_idx = None
    for i, start in enumerate(start_indices):
        if group_sizes[i] == 0:
            continue
        if start <= card_index < start + group_sizes[i]:
            prefix_idx = i
            break

    # If not found (possible case: num_prefix > target_cards and current card_index falls after unassigned prefixes)
    # In this case, we assign the card to the last prefix that has allocation (safe fallback)
    if prefix_idx is None:
        # Find the nearest prefix with group_sizes > 0 (should exist since target_cards > 0)
        for i in range(num_prefix - 1, -1, -1):
            if group_sizes[i] > 0:
                prefix_idx = i
                break

    assigned_prefix = [list_prefix[prefix_idx]]
    shard_id = card_index - start_indices[prefix_idx]
    num_shards = group_sizes[prefix_idx]

    # Print debug information for verification
    logger.info(f"[onevision_encoder_si_ssd] all_prefixes={list_prefix}")
    logger.info(f"[onevision_encoder_si_ssd] group_sizes={group_sizes}")
    logger.info(f"[onevision_encoder_si_ssd] start_indices={start_indices}")
    logger.info(f"[onevision_encoder_si_ssd] local_rank={local_rank} -> card_index={card_index}")
    logger.info(f"[onevision_encoder_si_ssd] assigned_prefix={assigned_prefix}, shard_id={shard_id}, num_shards={num_shards}")

    return Property(
        num_classes=2000000,
        num_examples=0,
        prefixes=assigned_prefix,
        name="onevision_encoder_si_ssd",
        label_select=0,
        label_start=0,
        num_shards=num_shards,
        shard_id=shard_id,
        dali_type="origin",
        random_diff=10,
    )


@DATASET_REGISTRY.register()
def onevision_encoder_si_cfs_single_node():
    """Configure onevision encoder SI dataset from CFS storage.

    This function loads COYO-400M and LAION-260M datasets from CFS storage and assigns
    all data prefixes to workers based on global rank distribution.

    WARNING: This is NOT a recommended approach as it can cause severe data imbalance.
    """
    patterns = [
        "/datasets_ov_encoder/coyo400m/*.rec",
        "/datasets_ov_encoder/laion260m/*.rec",
    ]

    all_files = [f for pattern in patterns for f in glob.glob(pattern)]
    all_files = [x.replace(".rec", "") for x in all_files]
    all_files = sorted(all_files)

    return Property(
        num_classes=2000000,
        num_examples=0,
        prefixes=all_files,
        name="onevision_encoder_si_ssd",
        label_select=0,
        label_start=0,
        num_shards=world_size,
        shard_id=rank,
        dali_type="origin",
        random_diff=10,
    )


@DATASET_REGISTRY.register()
def onevision_encoder_ocr_ssd():
    """Configure onevision encoder OCR dataset from SSD storage.

    This function loads OCR-labeled datasets (Obelics and Zero250M) from SSD storage
    and distributes the data prefixes across 8 target cards. Each card is assigned
    specific prefixes based on the local rank, ensuring even distribution of data shards.

    Similar to onevision_encoder_si_ssd, this handles uneven prefix distribution
    and supports cyclical mapping for cases with more than 8 GPUs.

    Returns:
        Property: Dataset property configuration with assigned prefixes, OCR data type,
                 and shard information.

    Raises:
        RuntimeError: If no .rec files are found matching the specified patterns.
    """
    patterns = [
        "/data_*/onevision_encoder_ocr_obelics/*ocr_labeled/*.rec",
        "/data_*/onevision_encoder_ocr_zero250m/*ocr_labeled/*. rec",
    ]

    all_files = [f for pattern in patterns for f in glob.glob(pattern)]
    all_files = [x.replace(".rec", "") for x in all_files]
    all_files = sorted(all_files)

    list_prefix = all_files
    if len(list_prefix) == 0:
        raise RuntimeError(f"No rec prefixes found for patterns: {patterns}")

    # Fixed division by 8 cards (target card count)
    target_cards = 8

    # Calculate how many cards should read each prefix (distribute evenly)
    num_prefix = len(list_prefix)
    base = target_cards // num_prefix
    rem = target_cards % num_prefix
    # Allocate 1 extra card to the first rem prefixes
    group_sizes = [base + (1 if i < rem else 0) for i in range(num_prefix)]

    # Calculate the starting card index for each prefix in card space ([0..target_cards-1])
    start_indices = []
    acc = 0
    for s in group_sizes:
        start_indices.append(acc)
        acc += s

    # Map current process's local_rank to card_index in range [0..target_cards-1]
    # This way, even if local_rank exceeds 8, it can cyclically map to 8 logical cards
    card_index = local_rank % target_cards

    # Find which prefix the card_index belongs to (ensure it can be found when group_sizes is non-zero)
    prefix_idx = None
    for i, start in enumerate(start_indices):
        if group_sizes[i] == 0:
            continue
        if start <= card_index < start + group_sizes[i]:
            prefix_idx = i
            break

    # If not found (possible case: num_prefix > target_cards and current card_index falls after unassigned prefixes)
    # In this case, we assign the card to the last prefix that has allocation (safe fallback)
    if prefix_idx is None:
        # Find the nearest prefix with group_sizes > 0 (should exist since target_cards > 0)
        for i in range(num_prefix - 1, -1, -1):
            if group_sizes[i] > 0:
                prefix_idx = i
                break

    assigned_prefix = [list_prefix[prefix_idx]]
    shard_id = card_index - start_indices[prefix_idx]
    num_shards = group_sizes[prefix_idx]

    # Print debug information for verification
    logger.info(f"[onevision_encoder_ocr_ssd] all_prefixes={list_prefix}")
    logger.info(f"[onevision_encoder_ocr_ssd] group_sizes={group_sizes}")
    logger.info(f"[onevision_encoder_ocr_ssd] start_indices={start_indices}")
    logger.info(f"[onevision_encoder_ocr_ssd] local_rank={local_rank} -> card_index={card_index}")
    logger.info(f"[onevision_encoder_ocr_ssd] assigned_prefix={assigned_prefix}, shard_id={shard_id}, num_shards={num_shards}")

    return Property(
        num_classes=365187,
        num_examples=0,
        prefixes=assigned_prefix,
        name="onevision_encoder_ocr_ssd",
        label_select=0,
        label_start=0,
        num_shards=num_shards,
        shard_id=shard_id,
        dali_type="ocr",
        random_diff=100,
    )


@DATASET_REGISTRY.register()
def onevision_encoder_video_codec():
    """Configure onevision encoder with square indexed filtered dataset (version 0.0.2).

    This function sets up a video dataset configuration using the Decord residual data loader.
    The dataset uses square aspect ratio frames with indexing for efficient access.
    Designed for distributed training with up to 128 workers.
    Currently configured with placeholder path that should be updated with actual data location.

    Returns:
        Property: Dataset property configuration for square indexed video data
                 with Decord residual data type.

    Raises:
        AssertionError: If world_size exceeds 128.
    """
    assert world_size <= 128
    list_mp4_label_path = f"train_how_to_100m_panda70m_k710_square_with_index_filtered_split_128/part_{rank:03d}"

    return Property(
        name="onevision_encoder_video_codec",
        prefixes=[list_mp4_label_path],
        num_classes=400000,
        num_examples=1629324 * 128,
        num_shards=1,
        shard_id=0,
        dali_type="decord_residual",
    )
