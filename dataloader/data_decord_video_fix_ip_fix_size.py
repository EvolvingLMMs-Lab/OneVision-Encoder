import os

import decord
import numpy as np
import nvidia.dali.fn as fn
import nvidia.dali.types as types

from nvidia.dali.auto_aug import rand_augment
from nvidia.dali.pipeline import pipeline_def
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy

rank = int(os.environ.get("RANK", "0"))
local_rank = int(os.environ.get("LOCAL_RANK", "0"))
world_size = int(os.environ.get("WORLD_SIZE", "1"))


class DALIWarper(object):
    def __init__(self, dali_iter, step_data_num, mode="train"):
        self.iter = dali_iter
        self.step_data_num = step_data_num
        assert(mode in ["train", "val", "test"])
        self.mode = mode

    def __next__(self):
        try:
            data_dict = self.iter.__next__()[0]
            videos = data_dict["videos"]
            visible_indices = data_dict["visible_indices"]
            labels = data_dict["labels"]
            return {
                "pixel_values": videos,
                "visible_indices": visible_indices,
                "labels": labels
            }
        except StopIteration:
            self.iter.reset()
            return self.__next__()

    def __iter__(self):
        return self

    def __len__(self):
        return self.step_data_num

    def reset(self):
        self.iter.reset()


class ExternalInputCallable:
    def __init__(
        self, source_params):

        self.file_list = source_params.get("file_list", None)
        self.label_path = source_params.get("label_path", None)
        self.visible_indices_path = source_params.get("visible_indices_path", None)
        
        if self.file_list is None:
            raise ValueError("file_list is None")

        self.num_shards = source_params.get("num_shards", world_size)
        self.shard_id = source_params.get("shard_id", rank)

        self.batch_size = source_params.get("batch_size", 32)
        self.input_size = source_params.get("input_size", 224)

        self.short_side_size = source_params.get("short_side_size", 256)
        self.sequence_length = source_params.get("sequence_length", 16)
        self.stride = source_params.get("stride", 8)
        self.use_sparse_sampling = source_params.get("use_sparse_sampling", False)
        self.use_rgb = source_params.get("use_rgb", True)
        self.use_flip = source_params.get("use_flip", True)
        self.seed = source_params.get("seed", 0)
        self.reprob = source_params.get("reprob", 0.0)

        # self.label = None
        # self.video_visible_indices = None
        # if self.label_path is not None:
        #     self.label = np.load(self.label_path)
        # if self.visible_indices_path is not None:
        #     self.video_visible_indices = np.load(self.visible_indices_path, mmap_mode="r")

        self.perm = None
        self.last_seen_epoch = None
        self.replace_example_info = self.file_list[0]

        self.mode = "train"

        # If the dataset size is not divisible by number of shards, the trailing samples will be omitted.
        self.shard_size = len(self.file_list) // self.num_shards
        self.shard_offset = self.shard_size * self.shard_id
        # drop last batch
        self.full_iterations = self.shard_size // self.batch_size

    # def __getstate__(self):
    #     state = dict(self.__dict__)
    #     # 不要把 mmap arrays pickled 过去
    #     state["label"] = None
    #     state["video_visible_indices"] = None
    #     return state


    # def __setstate__(self, state):
    #     self.__dict__.update(state)
    #     # 在子进程里按路径重新打开 mmap
    #     if self.label is None and self.label_path is not None:
    #         self.label = np.load(self.label_path)
    #     if self.video_visible_indices is None and self.visible_indices_path is not None:
    #         self.video_visible_indices = np.load(self.visible_indices_path, mmap_mode="r")


    def sparse_sampling_get_frameid_data(
            self,
            video_path,
            sequence_length,
            test_info):

        decord_vr = decord.VideoReader(video_path, num_threads=1, ctx=decord.cpu(0))
        duration = len(decord_vr)

        if duration >= sequence_length:
            frame_id_list = list(range(0, min(duration, sequence_length)))
        else:
            frame_id_list = list(range(duration)) + [duration - 1] * (sequence_length - duration)
        
        decord_vr.seek(0)
        video_data = decord_vr.get_batch(frame_id_list).asnumpy()
        return video_data

    def get_label_and_visible_indices(self, video_path):
        # label:    /video_vit/dataset/clips_square_aug_k710_ssv2/6/3/rank_068_sample_0000145663_label.npy
        # video:    /video_vit/dataset/clips_square_aug_k710_ssv2_hevc_v2/6/3/rank_068_sample_0000145663.mp4
        # residual: /video_vit/dataset/clips_square_aug_k710_ssv2_hevc_v2_residual/6/3/rank_068_sample_0000145663.visidx.npy

        label_path = video_path.replace("clips_square_aug_k710_ssv2_hevc_v2", "clips_square_aug_k710_ssv2")
        label_path = label_path.replace(".mp4", "_label.npy")

        video_visible_indices_path = video_path.replace("clips_square_aug_k710_ssv2_hevc_v2", "clips_square_aug_k710_ssv2_hevc_v2_residual")
        video_visible_indices_path = video_visible_indices_path.replace(".mp4", ".visidx.npy")

        video_label = np.load(label_path)
        video_visible_indices = np.load(video_visible_indices_path, mmap_mode="r")

        return video_label, video_visible_indices

    def __call__(self, sample_info):
        # sample_info
        # idx_in_epoch – 0-based index of the sample within epoch
        # idx_in_batch – 0-based index of the sample within batch
        # iteration – number of current batch within epoch
        # epoch_idx – number of current epoch
        if sample_info.iteration >= self.full_iterations:
            # Indicate end of the epoch
            raise StopIteration

        if self.last_seen_epoch != sample_info.epoch_idx:
            self.last_seen_epoch = sample_info.epoch_idx
            cur_seed = self.seed + sample_info.epoch_idx
            self.perm = np.random.default_rng(seed=cur_seed).permutation(len(self.file_list))

        sample_idx = self.perm[sample_info.idx_in_epoch + self.shard_offset]

        video_path = self.file_list[sample_idx]

        # label:    /video_vit/dataset/clips_square_aug_k710_ssv2/6/3/rank_068_sample_0000145663_label.npy
        # video:    /video_vit/dataset/clips_square_aug_k710_ssv2_hevc_v2/6/3/rank_068_sample_0000145663.mp4
        # residual: /video_vit/dataset/clips_square_aug_k710_ssv2_hevc_v2_residual/6/3/rank_068_sample_0000145663.visidx.npy
        test_info = None
        # print(video_path)
        try:
            video_data = self.sparse_sampling_get_frameid_data(video_path, self.sequence_length, test_info)
            video_label, video_visible_indices = self.get_label_and_visible_indices(video_path)
        except Exception as e:
            print("Error loading video:" , video_path)
            print(e)

            video_path = self.replace_example_info
            video_label, video_visible_indices = self.get_label_and_visible_indices(video_path)
            video_data = self.sparse_sampling_get_frameid_data(video_path, self.sequence_length, test_info)

        if self.mode == "test":
            chunk_nb, split_nb, video_idx = test_info
            return (
                    video_data,
                    np.int64([int(video_label)]),
                    np.int64([int(chunk_nb)]),
                    np.int64([int(split_nb)]),
                    np.int64([int(video_idx)])
                    )
        else:
            if isinstance(video_label, int):
                video_label = np.int64(np.array([video_label]))
            elif isinstance(video_label, np.ndarray):
                video_label = video_label.astype(np.int64)
            if isinstance(video_visible_indices, int):
                video_visible_indices = np.int16(np.array([video_visible_indices]))
            elif isinstance(video_visible_indices, np.ndarray):
                video_visible_indices = video_visible_indices.astype(np.int16)
            return video_data, video_visible_indices, video_label


@pipeline_def()
def dali_pipeline(mode, source_params):
    if mode == "train":
        videos, visible_indices, labels = fn.external_source(
            source      = ExternalInputCallable(source_params),
            num_outputs = 3,
            batch       = False,
            parallel    = True,
            dtype       = [types.UINT8, types.INT16, types.INT64],
            layout      = ["FHWC", "C", "C"]
        )

        videos = videos.gpu()
        # videos = fn.resize(
        #     videos,
        #     resize_x=source_params['input_size'],
        #     resize_y=source_params['input_size'],
        #     interp_type         = types.INTERP_LINEAR
        # )

        videos = fn.crop_mirror_normalize(
            videos,
            device        = "gpu",
            dtype         = types.FLOAT,
            output_layout = "CFHW",
            mean          = source_params['mean'],
            std           = source_params['std']
        )
        visible_indices = visible_indices.gpu()
        labels = labels.gpu()
        return videos, visible_indices, labels


def dali_dataloader(
    file_list,
    dali_num_threads,
    dali_py_num_workers,
    batch_size,
    input_size         = 224,
    sequence_length    = 16,
    stride             = 8,
    mode               = "train",
    seed               = 0,
    short_side_size    = 256,
    num_shards         = None,
    shard_id           = None,
):

    if isinstance(input_size, list):
        input_size = input_size[0]

    mean = [x * 255 for x in [0.48145466, 0.4578275, 0.40821073]]
    std = [x * 255 for x in [0.26862954, 0.26130258, 0.27577711]]

    source_params = {
        # Basic parameters
        "batch_size":           batch_size,
        "seed":                 seed + rank,
        "num_shards":           num_shards, 
        "shard_id":             shard_id,
        "file_list":            file_list,

        # Size parameters
        # "input_size":           input_size,
        # "short_side_size":      short_side_size,
        "sequence_length":      sequence_length,
        "stride":               stride,
        
        # Feature flags
        "use_sparse_sampling":  True,
        "use_rgb":              True,
        "use_flip":             True,
        
        # Augmentation parameters
        "mean":                 mean,
        "std":                  std,
        "reprob":               0,
    }

    pipe = dali_pipeline(
        batch_size           = batch_size,
        num_threads          = dali_num_threads,
        device_id            = local_rank,
        seed                 = seed + rank,
        py_num_workers       = dali_py_num_workers,
        py_start_method      = 'spawn',
        prefetch_queue_depth = 2,
        mode                 = mode,
        source_params        = source_params
    )
    pipe.build()

    # Define output mapping based on mode
    output_map = ['videos', 'visible_indices', 'labels']
    if mode == "test":
        output_map.extend(['chunk_nb', 'split_nb', 'sample_idx'])

    dataloader = DALIWarper(
        dali_iter = DALIGenericIterator(
            pipelines           = pipe,
            output_map          = output_map,
            auto_reset          = True,
            size                = -1,
            last_batch_padded   = False,
            last_batch_policy   = LastBatchPolicy.FILL,
            prepare_first_batch = False
        ),
        step_data_num = len(file_list) // world_size // batch_size,
        mode          = mode
    )
    return dataloader
