#
# Created by anxiangsir
# Date: 2025-11-13 12:26:36 (UTC)
#
 
import os
import warnings
from typing import Any, Dict, List, Tuple

import decord
import numpy as np
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.pipeline import pipeline_def
from nvidia.dali.plugin.pytorch import DALIGenericIterator
try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False



# ----------------------------------------------------------------------------
# 1. DALI Iterator Wrapper (已修改 - 返回 indices 和 total_frames)
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
            "total_frames": data_dict["total_frames"]
        }

    def __iter__(self):
        return self

    def __len__(self) -> int:
        return self.steps_per_epoch

    def reset(self):
        self.iter.reset()

# ----------------------------------------------------------------------------
# 2. DALI External Source for Video Data (已修改 - 返回 indices 和 total_frames)
# ----------------------------------------------------------------------------
class VideoExternalSource:
    def __init__(self, mode: str, source_params: Dict[str, Any]):
        self.mode = mode
        self.file_list: List[Tuple[str, int]] = source_params["file_list"]
        self.num_shards: int = source_params["num_shards"]
        self.shard_id: int = source_params["shard_id"]
        self.batch_size: int = source_params["batch_size"]
        self.sequence_length: int = source_params["sequence_length"]
        self.use_rgb: bool = source_params["use_rgb"]
        self.seed: int = source_params["seed"]

        # ===> decord 线程数参数 <===
        self.decord_num_threads: int = source_params["decord_num_threads"]

        self.shard_size = len(self.file_list) // self.num_shards
        self.shard_offset = self.shard_size * self.shard_id
        self.full_iterations = self.shard_size // self.batch_size

        self.perm = None
        self.last_seen_epoch = -1
        self.fallback_example = self.file_list[0] if self.file_list else ("", 0)

    def _get_frame_indices(self, num_frames: int) -> List[int]:
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

    def _load_video_data(self, video_path: str) -> Tuple[np.ndarray, np.ndarray, int]:
        # ===> 在此处使用可配置的线程数，并返回 indices 和 total_frames <===
        vr = decord.VideoReader(video_path, num_threads=self.decord_num_threads, ctx=decord.cpu(0))
        num_frames = len(vr)
        frame_indices = self._get_frame_indices(num_frames)
        video_data = vr.get_batch(frame_indices).asnumpy()
        if self.use_rgb:
            video_data = video_data[:, :, :, ::-1]
        # 返回 video_data, indices, 和 total_frames
        return video_data, np.array(frame_indices, dtype=np.int64), num_frames

    def __call__(self, sample_info) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if sample_info.iteration >= self.full_iterations:
            raise StopIteration
        if self.last_seen_epoch != sample_info.epoch_idx:
            self.last_seen_epoch = sample_info.epoch_idx
            rng = np.random.default_rng(seed=self.seed + sample_info.epoch_idx)
            self.perm = rng.permutation(len(self.file_list))
        sample_idx = self.perm[sample_info.idx_in_epoch + self.shard_offset]
        video_path, video_label = self.file_list[sample_idx]
        try:
            video_data, frame_indices, total_frames = self._load_video_data(video_path)
        except Exception as e:
            warnings.warn(f"Failed to load video: {video_path}, error: {e}. Using fallback.")
            fallback_path, _ = self.fallback_example
            if not fallback_path: raise IOError(f"Fallback video path is empty!")
            video_data, frame_indices, total_frames = self._load_video_data(fallback_path)

        return video_data, np.int64([int(video_label)]), frame_indices, np.int64([total_frames])


def preprocess_videos(videos, mode, input_size, mean, std):
    # 统一 resize + 中心裁剪
    videos = fn.resize(
        videos,
        device="gpu",
        resize_shorter=input_size,
        interp_type=types.INTERP_CUBIC,
    )
    videos = fn.crop_mirror_normalize(
        videos,
        device="gpu",
        crop=[input_size, input_size],
        crop_pos_x=0.5,
        crop_pos_y=0.5,
        dtype=types.UINT8,
        output_layout="FHWC",
    )

    if mode == "train":
        # 亮度/对比度
        if fn.random.coin_flip(dtype=types.BOOL, probability=0.8):
            videos = fn.brightness_contrast(
                videos,
                contrast=fn.random.uniform(range=(0.6, 1.4)),
                brightness=fn.random.uniform(range=(-0.125, 0.125)),
                device="gpu",
            )

        # 饱和度
        if fn.random.coin_flip(dtype=types.BOOL, probability=0.8):
            videos = fn.saturation(
                videos,
                saturation=fn.random.uniform(range=[0.6, 1.4]),
                device="gpu",
            )

        # 色相
        if fn.random.coin_flip(dtype=types.BOOL, probability=0.8):
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

    # 统一归一化到 FLOAT / CFHW
    videos = fn.crop_mirror_normalize(
        videos,
        dtype=types.FLOAT,
        output_layout="CFHW",
        mean=[m * 255.0 for m in mean],
        std=[m * 255.0 for m in std],
        device="gpu",
    )
    return videos


# ----------------------------------------------------------------------------
# 3. DALI Pipeline Definition (已修改 - 处理 indices 和 total_frames)
# ----------------------------------------------------------------------------
@pipeline_def(enable_conditionals=True)
def dali_video_pipeline(mode: str, source_params: Dict[str, Any]):
    short_side_size = source_params["short_side_size"]
    input_size = source_params["input_size"]
    mean = source_params["mean"]
    std = source_params["std"]

    # ===> 现在返回 4 个输出: videos, labels, indices, total_frames <===
    videos, labels, indices, total_frames = fn.external_source(
        source=VideoExternalSource(mode, source_params),
        num_outputs=4,
        batch=False,
        parallel=True,
        dtype=[types.UINT8, types.INT64, types.INT64, types.INT64],
        layout=["FHWC", "C", "C", "C"]
    )

    videos = videos.gpu()
    labels = labels.gpu()
    indices = indices.gpu()
    total_frames = total_frames.gpu()

    videos = preprocess_videos(videos, mode, input_size, mean, std)
    return videos, labels, indices, total_frames


# ----------------------------------------------------------------------------
# 4. Main Dataloader Function (已修改 - output_map 增加 indices 和 total_frames)
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
) -> DALIWarper:
    """
    """
    print(f"[{mode} loader] Reading for: {data_csv_path}")
    file_list = []
    try:
        with open(data_csv_path, "r", encoding="utf-8") as f:
            for line in f:
                offset_path, label = line.strip().split(",")
                full_path = os.path.join(data_root_path, offset_path)
                file_list.append((full_path, int(label)))
    except FileNotFoundError:
        raise FileNotFoundError(f"Data list file not found at: {data_csv_path}")
    if not file_list:
        raise ValueError(f"File list from {data_csv_path} is empty.")

    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))

    source_params = {
        "num_shards": world_size, "shard_id": rank, "file_list": file_list,
        "batch_size": batch_size, "sequence_length": sequence_length, "seed": seed + rank,
        "use_rgb": use_rgb, "input_size": input_size, "short_side_size": short_side_size,
        "mean": mean, "std": std,
        "decord_num_threads": decord_num_threads,
    }

    pipe = dali_video_pipeline(
        batch_size=batch_size, num_threads=dali_num_threads, device_id=local_rank,
        seed=seed + rank, py_num_workers=dali_py_num_workers, py_start_method="forkserver",
        prefetch_queue_depth=2, mode=mode, source_params=source_params,
    )
    pipe.build()

    # ===> output_map 增加 "indices" 和 "total_frames" <===
    dali_iter = DALIGenericIterator(
        pipelines=[pipe],
        output_map=["videos", "labels", "indices", "total_frames"],
        auto_reset=True
    )
    steps_per_epoch = len(file_list) // world_size // batch_size
    dataloader = DALIWarper(dali_iter=dali_iter, steps_per_epoch=steps_per_epoch)

    print(f"[{mode} loader] DALI pipeline built. Total samples: {len(file_list)}, Steps per epoch: {steps_per_epoch}.")
    return dataloader
