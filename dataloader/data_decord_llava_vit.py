import logging
import os
from typing import Any, Dict, List

import decord
import numpy as np
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.pipeline import pipeline_def
from nvidia.dali.plugin.pytorch import DALIGenericIterator

logger = logging.getLogger(__file__)

# ----------------------------------------------------------------------------
# 1. DALI Iterator Wrapper (已修改 - 返回 indices, total_frames 和 video_visible_indices)
# ----------------------------------------------------------------------------
class DALIWarper:
    def __init__(self, dali_iter: DALIGenericIterator, steps_per_epoch: int):
        self.iter = dali_iter
        self.steps_per_epoch = steps_per_epoch

    def __next__(self) -> Dict[str, object]:
        data_dict = self.iter.__next__()[0]
        return {
            "videos": data_dict["videos"],
            "labels": data_dict["labels"],
            "indices": data_dict["indices"],
            "total_frames": data_dict["total_frames"],
            "video_visible_indices": data_dict["video_visible_indices"]
        }

    def __iter__(self):
        return self

    def __len__(self) -> int:
        return self.steps_per_epoch

    def reset(self):
        self.iter.reset()

# ----------------------------------------------------------------------------
# 2. DALI External Source for Video Data (已修改 - 返回 indices, total_frames 和 video_visible_indices)
# ----------------------------------------------------------------------------
class VideoExternalSource:
    def __init__(self, mode: str, source_params: dict):
        self.mode = mode
        self.file_list: list = source_params["file_list"]
        self.num_shards: int = source_params["num_shards"]
        self.shard_id: int = source_params["shard_id"]
        self.batch_size: int = source_params["batch_size"]
        self.sequence_length: int = source_params["sequence_length"]
        self.use_rgb: bool = source_params["use_rgb"]
        self.seed: int = source_params["seed"]
        self.decord_num_threads: int = source_params["decord_num_threads"]

        self.shard_size = len(self.file_list) // self.num_shards
        self.shard_offset = self.shard_size * self.shard_id
        # 【删除】 self.full_iterations 不再需要用于抛出异常
        # self.full_iterations = self.shard_size // self.batch_size

        self.perm = None
        self.last_seen_epoch = -1

        # 只用标准logger，不加file handler
        self.logger = logging.getLogger(__file__)

    def _get_frame_indices(self, num_frames: int) -> list:
        if num_frames < self.sequence_length:
            indices = list(range(num_frames))
            indices += [num_frames - 1] * (self.sequence_length - num_frames)
            return indices
        seg_size = float(num_frames - 1) / self.sequence_length
        if self.mode == "train":
            indices = [int(seg_size * i + np.random.uniform(0, seg_size)) for i in range(self.sequence_length)]
        else:
            indices = [int(seg_size * i + seg_size / 2) for i in range(self.sequence_length)]
        return indices

    def _load_video_data(self, video_path: str):
        vr = decord.VideoReader(video_path, num_threads=self.decord_num_threads, ctx=decord.cpu(0))
        num_frames = len(vr)
        frame_indices = self._get_frame_indices(num_frames)
        video_data = vr.get_batch(frame_indices).asnumpy()
        if self.use_rgb:
            video_data = video_data[:, :, :, ::-1]
        return video_data, np.array(frame_indices, dtype=np.int64), num_frames

    def _get_valid_sample(self, sample_idx: int, depth=0) -> tuple:
        if depth > 3:  # 防止极端递归
            self.logger.info("Too many attempts, fallback to first sample.")
            sample_line = self.file_list[0]
        else:
            sample_line = self.file_list[sample_idx]
        parts = sample_line.strip().split()
        if len(parts) < 12:
            self.logger.info(f"Invalid line format (not enough columns): {sample_line}")
            new_idx = np.random.randint(0, len(self.file_list))
            return self._get_valid_sample(new_idx, depth + 1)
        video_path = parts[0]
        video_label = [int(x) for x in parts[1:11]]
        video_visible_indices_path = parts[11]
        try:
            video_data, frame_indices, total_frames = self._load_video_data(video_path)
            video_visible_indices = np.load(video_visible_indices_path, mmap_mode="r")
            if isinstance(video_visible_indices, np.ndarray):
                video_visible_indices = video_visible_indices.astype(np.int16)
            return video_data, np.array(video_label, dtype=np.int64), frame_indices, np.int64([total_frames]), video_visible_indices
        except Exception as e:
            self.logger.info(f"Failed to load video: {video_path}, error: {e}")
            new_idx = np.random.randint(0, len(self.file_list))
            return self._get_valid_sample(new_idx, depth + 1)

    def __call__(self, sample_info):
        if self.last_seen_epoch != sample_info.epoch_idx:
            self.last_seen_epoch = sample_info.epoch_idx
            rng = np.random.default_rng(seed=self.seed + sample_info.epoch_idx)
            self.perm = rng.permutation(len(self.file_list))
        idx_in_shard = sample_info.idx_in_epoch % self.shard_size
        sample_idx = self.perm[idx_in_shard + self.shard_offset]
        return self._get_valid_sample(sample_idx, depth=0)

# ----------------------------------------------------------------------------
# 3. DALI Pipeline Definition (已修改 - 处理 indices, total_frames 和 video_visible_indices)
# ----------------------------------------------------------------------------
@pipeline_def(enable_conditionals=True)
def dali_video_pipeline(mode: str, source_params: Dict[str, Any]):
    input_size = source_params["input_size"]
    mean = source_params["mean"]
    std = source_params["std"]

    # ===> 现在返回 5 个输出: videos, labels, indices, total_frames, video_visible_indices <===
    videos, labels, indices, total_frames, video_visible_indices = fn.external_source(
        source=VideoExternalSource(mode, source_params),
        num_outputs=5,
        batch=False,
        parallel=True,
        dtype=[types.UINT8, types.INT64, types.INT64, types.INT64, types.INT16],
        layout=["FHWC", "C", "C", "C", "C"]
    )

    videos = videos.gpu()
    labels = labels.gpu()
    indices = indices.gpu()
    total_frames = total_frames.gpu()
    video_visible_indices = video_visible_indices.gpu()

    # 直接resize到input_size，因为视频已经是256x256的正方形
    videos = fn.resize(videos, resize_x=input_size, resize_y=input_size, antialias=True, interp_type=types.INTERP_CUBIC)

    if mode == "train":
        # 亮度/对比度
        if fn.random.coin_flip(dtype=types.BOOL, probability=0.3):
            videos = fn.brightness_contrast(
                videos,
                contrast=fn.random.uniform(range=(0.6, 1.4)),
                brightness=fn.random.uniform(range=(-0.125, 0.125)),
                device="gpu",
            )
        # 饱和度
        if fn.random.coin_flip(dtype=types.BOOL, probability=0.3):
            videos = fn.saturation(
                videos,
                saturation=fn.random.uniform(range=[0.6, 1.4]),
                device="gpu",
            )
        # 色相
        if fn.random.coin_flip(dtype=types.BOOL, probability=0.3):
            videos = fn.hue(
                videos,
                hue=fn.random.uniform(range=[-0.2, 0.2]),
                device="gpu",
            )

        # 色彩空间转换
        if fn.random.coin_flip(dtype=types.BOOL, probability=0.1):
            videos = fn.color_space_conversion(
                videos,
                image_type=types.RGB,
                output_type=types.BGR,
                device="gpu",
            )

    videos = fn.crop_mirror_normalize(videos, dtype=types.FLOAT, output_layout="CFHW", mean=[m * 255.0 for m in mean], std=[s * 255.0 for s in std])
    return videos, labels, indices, total_frames, video_visible_indices

# ----------------------------------------------------------------------------
# 4. Main Dataloader Function (已修改 - output_map 增加 indices, total_frames 和 video_visible_indices)
# ----------------------------------------------------------------------------
def get_dali_dataloader(
    data_root_path: str,
    data_csv_path: str,
    mode: str = "val",
    batch_size: int = 32,
    sequence_length: int = 16,
    input_size: int = 224,
    short_side_size: int = 224,
    use_rgb: bool = True,
    mean: List[float] = [0.48145466, 0.4578275, 0.40821073],
    std: List[float] = [0.26862954, 0.26130258, 0.27577711],
    dali_num_threads: int = 4,
    dali_py_num_workers: int = 8,
    decord_num_threads: int = 2,
    seed: int = 0,
    shard_id = None,
    num_shards = None,
) -> DALIWarper:
    print(f"[{mode} loader] Reading for: {data_csv_path}")
    file_list = []
    try:
        with open(data_csv_path, "r") as f:
            for line in f:
                file_list.append(line.rstrip("\n"))  # 每行原始内容，无分割处理
    except FileNotFoundError:
        raise FileNotFoundError(f"Data list file not found at: {data_csv_path}")
    if not file_list:
        raise ValueError(f"File list from {data_csv_path} is empty.")

    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    if num_shards is None or shard_id is None:
        num_shards = world_size
        shard_id = rank

    source_params = {
        "num_shards": num_shards, "shard_id": shard_id, "file_list": file_list,
        "batch_size": batch_size, "sequence_length": sequence_length, "seed": seed + rank,
        "use_rgb": use_rgb, "input_size": input_size, "short_side_size": short_side_size,
        "mean": mean, "std": std,
        "decord_num_threads": decord_num_threads,
    }

    pipe = dali_video_pipeline(
        batch_size=batch_size, num_threads=dali_num_threads, device_id=local_rank,
        seed=seed + rank, py_num_workers=dali_py_num_workers, py_start_method="forkserver",
        prefetch_queue_depth=8, mode=mode, source_params=source_params,
    )
    pipe.build()

    dali_iter = DALIGenericIterator(
        pipelines=[pipe],
        output_map=["videos", "labels", "indices", "total_frames", "video_visible_indices"],
        auto_reset=True
    )
    steps_per_epoch = len(file_list) // batch_size // num_shards
    dataloader = DALIWarper(dali_iter=dali_iter, steps_per_epoch=steps_per_epoch)

    print(f"[{mode} loader] DALI pipeline built. Total samples: {len(file_list)}, Steps per epoch: {steps_per_epoch}.")
    return dataloader
