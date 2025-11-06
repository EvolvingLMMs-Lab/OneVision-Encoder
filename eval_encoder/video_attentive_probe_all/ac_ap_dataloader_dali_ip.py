import os
import logging
import traceback
import numpy as np
import nvidia.dali.fn as fn

import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
import decord
from nvidia.dali.pipeline import pipeline_def
import glob
try:
    from .hevc_feature_decoder import ResPipeReader
except Exception:
    from hevc_feature_decoder import ResPipeReader



class DALIWarper(object):
    def __init__(self, dali_iter, step_data_num):
        self.iter = dali_iter
        self.step_data_num = step_data_num

    def __next__(self):
        data_dict = self.iter.__next__()[0]
        videos = data_dict["videos"]
        labels = data_dict["labels"]
        res_zero_masks = data_dict["res_zero_masks"]
        # print("videos shape:", videos.shape)
        # print("res_zero_masks shape:", res_zero_masks.shape)
        return videos, res_zero_masks, labels


    def __iter__(self):
        return self

    def __len__(self):
        return self.step_data_num

    def reset(self):
        self.iter.reset()


class ExternalInputCallable:
    def __init__(self, mode, source_params):

        self.mode = mode

        self.file_list = source_params['file_list']
        self.num_shards = source_params['num_shards']
        self.shard_id = source_params['shard_id']
        self.batch_size = source_params['batch_size']
        self.sequence_length = source_params['sequence_length']
        self.use_rgb = source_params['use_rgb']
        self.seed = source_params['seed']

        # If the dataset size is not divisible by number of shards, the trailing samples will be omitted.
        self.shard_size = len(self.file_list) // self.num_shards
        self.shard_offset = self.shard_size * self.shard_id
        # drop last batch
        self.full_iterations = self.shard_size // self.batch_size
        # so that we don't have to recompute the `self.perm` for every sample
        self.perm = None
        self.last_seen_epoch = None
        self.replace_example_info = self.file_list[0]
        self.gop_size = 16
        self.enable_res_zero_mask = True
        self.hevc_y_only = True
        self.tokeq_target_frames = 8


    def get_frame_id_list(self, video_path, sequence_length):
        decord_vr = decord.VideoReader(video_path, num_threads=1, ctx=decord.cpu(0))
        duration = len(decord_vr)
        
        if self.mode in ["train", "val"]:
            # 所有帧索引
            all_index = list(range(0, int(duration), 1))
            average_duration = duration // sequence_length

            if average_duration > 0:
                frame_id_list = list(
                    np.multiply(list(range(sequence_length)), average_duration))
            else:
            # 拼接 I 和 P，保证顺序
                if duration >= sequence_length:
                    frame_id_list = list(range(0, min(duration, sequence_length)))
                else:
                    frame_id_list = list(range(duration)) + [duration - 1] * (sequence_length - duration)
            
            try:
                key_idx = None
                if hasattr(decord_vr, "get_key_indices"):
                    key_idx = decord_vr.get_key_indices()
                elif hasattr(decord_vr, "get_keyframes"):
                    key_idx = decord_vr.get_keyframes()
                if key_idx is not None:
                    # key_idx 可能是 NDArray；转成 Python list 的整型帧号集合
                    I_list = np.asarray(key_idx).tolist()
                    I_list = [int(i) for i in I_list if int(i) in frame_id_list]
                    if len(I_list) >= self.tokeq_target_frames:
                        # 如果 I 帧过多，优先保留前面的
                        I_list = I_list[:self.tokeq_target_frames]
                        P_list = []
                    else:
                        P_list = [i for i in range(len(frame_id_list)) if i not in I_list]
            except Exception:
                # 保底处理：忽略异常，后续用默认策略
                print("没有读取成功")
                gop = max(1,int(self.gop_size))
                I_list = [i for i, fid in enumerate(frame_id_list)if(int(fid)% gop)== 0]
                # 其余为 P 帧
                P_list = [i for i in range(len(frame_id_list))if i not in I_list]
                # Map absolute frame id -> position in the sampled sequence
                frame_ids = frame_id_list
                pos_map = {fid: i for i, fid in enumerate(frame_ids)}

            frame_ids = frame_id_list
            pos_map = {fid: i for i, fid in enumerate(frame_ids)}


                        # 读取视频帧
            decord_vr.seek(0)
            video_data = decord_vr.get_batch(frame_id_list).asnumpy()

            # 转成 numpy array
            I_list = np.array(I_list, dtype=np.int64)
            P_list = np.array(P_list, dtype=np.int64)
            I_pos_set = set(I_list.tolist())
            P_pos_set = set(P_list.tolist())

            num_I = int(len(I_list))
            num_P = int(len(P_list))
            if self.enable_res_zero_mask:
                _prev_y_only = os.environ.get("UMT_HEVC_Y_ONLY", None)
                try:
                    if self.hevc_y_only:
                        os.environ["UMT_HEVC_Y_ONLY"] = "1"

                    # --- residual read with ResPipeReader; prefix fast path if frame_ids are 0..F-1 ---
                    F_sel = len(frame_id_list)
                    wanted = set(frame_id_list)
                    idx2pos = {fid: i for i, fid in enumerate(frame_id_list)}
                    I_pos_set = set(int(pos_map.get(fid, -1)) for fid in I_list if int(pos_map.get(fid, -1)) >= 0)

                    is_prefix = all(fid == i for i, fid in enumerate(frame_id_list))
                    prefix_fast = (int(os.environ.get("HEVC_PREFIX_FAST", "1")) != 0)

                    # Classic list-based accumulation (no FAST_POSTPROC)
                    residuals_y = [None] * F_sel
                    H0 = W0 = None
                    dtype0 = None

                    def _ensure_y(arr):
                        nonlocal H0, W0, dtype0
                        # ResPipeReader yields Y or (Y,U,V); we only need Y here
                        y = arr[0] if isinstance(arr, tuple) else arr
                        y = np.asarray(y)
                        if y.ndim == 3:
                            y = np.squeeze(y)
                        if H0 is None:
                            H0, W0 = int(y.shape[0]), int(y.shape[1])
                            dtype0 = y.dtype
                        return y

                    # t_read = _t.time()
                    hevc_threads = getattr(self, 'hevc_n_parallel', 6)


                    # Sparse sampling path (single reader)
                    # tA = _t.time()
                    rdr = ResPipeReader(video_path, nb_frames=None, n_parallel=hevc_threads)
                    # hevc_init_ms += (_t.time() - tA) * 1000.0
                    cur_idx = 0
                    try:
                        for res in rdr.next_residual():
                            if cur_idx in wanted:
                                pos = idx2pos[cur_idx]
                                if pos in I_pos_set:
                                    if H0 is None:
                                        y0 = _ensure_y(res)
                                        H0, W0, dtype0 = y0.shape[0], y0.shape[1], y0.dtype
                                    residuals_y[pos] = np.full((H0, W0), 255, dtype=dtype0 or np.uint8)
                                else:
                                    y = _ensure_y(res)
                                    residuals_y[pos] = y
                                if all(x is not None for x in residuals_y):
                                    break
                            cur_idx += 1
                    finally:
                        try:
                            rdr.close()
                        except Exception:
                            pass


                    # If still missing shapes, fall back to video dims
                    if dtype0 is None:
                        dtype0 = np.uint8
                        H0, W0 = video_data.shape[1], video_data.shape[2]
                    for i in range(F_sel):
                        if residuals_y[i] is None:
                            residuals_y[i] = np.zeros((H0, W0), dtype=dtype0)

                    residuals_y = np.stack(residuals_y, axis=0)[..., np.newaxis]
                    combined_data = np.concatenate([video_data, residuals_y], axis=-1)

                    if H0 != video_data.shape[1] or W0 != video_data.shape[2]:
                        print("[warn] residual尺寸与视频不一致: res=(%d,%d) video=(%d,%d)" % (H0, W0, video_data.shape[1], video_data.shape[2]))
                finally:
                    # 恢复环境变量
                    if _prev_y_only is None:
                        os.environ.pop("UMT_HEVC_Y_ONLY", None)
                    else:
                        os.environ["UMT_HEVC_Y_ONLY"] = _prev_y_only

            return combined_data

    def __call__(self, sample_info):
        #sample_info
        #idx_in_epoch – 0-based index of the sample within epoch
        #idx_in_batch – 0-based index of the sample within batch
        #iteration – number of current batch within epoch
        #epoch_idx – number of current epoch
        if sample_info.iteration >= self.full_iterations:
            # Indicate end of the epoch
            raise StopIteration
        if self.last_seen_epoch != sample_info.epoch_idx:
            self.last_seen_epoch = sample_info.epoch_idx
            cur_seed = self.seed + sample_info.epoch_idx
            self.perm = np.random.default_rng(seed=cur_seed).permutation(len(self.file_list))
            
        sample_idx = self.perm[sample_info.idx_in_epoch + self.shard_offset]

        example_info = self.file_list[sample_idx]
        video_path, video_label = example_info
        # video_path = _maybe_swap_to_hevc(video_path)
        # print(video_path, video_label)
        try:
            combined_data = self.get_frame_id_list(video_path, self.sequence_length)
        except:
            # print("error", video_path)
            # print(video_path)
            video_path, video_label = self.replace_example_info
            # print(video_path, video_label)
            combined_data = self.get_frame_id_list(video_path, self.sequence_length)
        return combined_data, np.int64([int(video_label)])

@pipeline_def(enable_conditionals=True)
def dali_pipeline(mode, source_params):
    
    short_side_size = source_params['short_side_size']
    input_size = source_params['input_size']
    mean = source_params['mean']
    std = source_params['std']
    if not source_params['multi_views']:
        if mode == "train":
            combined_data, labels = fn.external_source(
                source = ExternalInputCallable(mode, source_params),
                num_outputs = 2,
                batch = False,
                parallel = True,
                dtype = [types.UINT8, types.INT64],
                layout = ["FHWC", "C"]
            )
            combined_data = combined_data.gpu()
            combined_data = fn.resize(
                combined_data,
                device="gpu",
                resize_shorter=input_size,
                interp_type=types.INTERP_CUBIC
            )
            combined_data = fn.crop_mirror_normalize(
                combined_data,
                device="gpu",
                crop=[input_size, input_size],
                crop_pos_x=0.5,   # 中心裁剪
                crop_pos_y=0.5,
                dtype=types.UINT8,
                output_layout="FHWC"
            )


            video_channels = source_params.get('video_channels', 3)  # 例如 RGB=3
            videos = fn.slice(
                combined_data,
                start=[0],
                shape=[video_channels],
                axes=[3]   # C 通道在第3维
            )

            res_zero_masks = fn.slice(
                combined_data,
                start=[video_channels],
                shape=[1],
                axes=[3]
            )

            # combined_data = fn.resize(combined_data, size=[input_size, input_size], 
            #                     interp_type=types.INTERP_LINEAR, device="gpu")
            videos = fn.slice(combined_data, axes=[3], start=0, end=3)
            brightness_contrast_probability = fn.random.coin_flip(dtype=types.BOOL, probability=0.8)
            if brightness_contrast_probability:
                videos = fn.brightness_contrast(videos, contrast=fn.random.uniform(range=(0.6, 1.4)),
                                                brightness=fn.random.uniform(range=(-0.125, 0.125)), device="gpu")
            saturation_probability = fn.random.coin_flip(dtype=types.BOOL, probability=0.8)
            if saturation_probability:
                videos = fn.saturation(videos, saturation=fn.random.uniform(range=[0.6, 1.4]), device="gpu")
            hue_probability = fn.random.coin_flip(dtype=types.BOOL, probability=0.8)
            if hue_probability:
                videos = fn.hue(videos, hue=fn.random.uniform(range=[-0.2, 0.2]), device="gpu")
            color_space_probability = fn.random.coin_flip(dtype=types.BOOL, probability=0.1)
            if color_space_probability:
                videos = fn.color_space_conversion(videos, image_type=types.RGB, output_type=types.BGR, device="gpu")
            videos = fn.crop_mirror_normalize(videos, dtype=types.FLOAT, output_layout = "CFHW",
                                            mean=[m*255.0 for m in mean], std=[m*255.0 for m in std], device="gpu")

            # Make mask follow the same spatial transforms as videos (flip + resize) before layout change
            # res_zero_masks = fn.resize(
            #     res_zero_masks,
            #     size=[input_size, input_size],
            #     interp_type=types.INTERP_NN
            # )
            res_zero_masks = fn.transpose(res_zero_masks, perm=[3, 0, 1, 2])  # FHWC -> CFHW
            # res_zero_masks = fn.cast(res_zero_masks, dtype=types.FLOAT) / 255.0

            labels = labels.gpu()
            return videos, res_zero_masks, labels
        else:
            combined_data, labels = fn.external_source(
                source = ExternalInputCallable(mode, source_params),
                num_outputs = 2,
                batch = False,
                parallel = True,
                dtype = [types.UINT8, types.INT64],
                layout = ["FHWC", "C"]
            )
            combined_data = combined_data.gpu()
            combined_data = fn.resize(
                combined_data,
                device="gpu",
                resize_shorter=input_size,
                interp_type=types.INTERP_CUBIC
            )
            combined_data = fn.crop_mirror_normalize(
                combined_data,
                device="gpu",
                crop=[input_size, input_size],
                crop_pos_x=0.5,   # 中心裁剪
                crop_pos_y=0.5,
                dtype=types.UINT8,
                output_layout="FHWC"
            )

            

            video_channels = source_params.get('video_channels', 3)  # 例如 RGB=3
            videos = fn.slice(
                combined_data,
                start=[0],
                shape=[video_channels],
                axes=[3]   # C 通道在第3维
            )

            res_zero_masks = fn.slice(
                combined_data,
                start=[video_channels],
                shape=[1],
                axes=[3]
            )
            
            # Make mask follow the same spatial transforms as videos (flip + resize) before layout change
            # res_zero_masks = fn.resize(
            #     res_zero_masks,
            #     size=[input_size, input_size],
            #     interp_type=types.INTERP_NN
            # )
            videos = fn.crop_mirror_normalize(videos, dtype=types.FLOAT, output_layout = "CFHW",
                                            mean=[m*255.0 for m in mean], std=[m*255.0 for m in std], device="gpu")
            res_zero_masks = fn.transpose(res_zero_masks, perm=[3, 0, 1, 2])  # FHWC -> CFHW
            # res_zero_masks = fn.cast(res_zero_masks, dtype=types.FLOAT) / 255.0
            labels = labels.gpu()
            return videos, res_zero_masks, labels


def dali_dataloader(data_root_path,
                    data_csv_path,
                    data_set,
                    dali_num_threads = 4,
                    dali_py_num_workers = 8,
                    batch_size = 32,
                    input_size = 224,
                    short_side_size = 239,
                    sequence_length = 16,
                    use_rgb = False,
                    multi_views=False,
                    mean = [0.48145466, 0.4578275, 0.40821073],
                    std = [0.26862954, 0.26130258, 0.27577711],
                    mode = "val",
                    num_shots = None,
                    seed = 0):

    if num_shots is not None:
        if mode == "train":
            txt_file_name = "{}_{}_{}_replaced.txt".format(data_set, mode, "fewshot{}".format(num_shots))
        else:
            txt_file_name = "{}_hevc_{}.txt".format(data_set, mode)
            if data_set != "ssv2":
                old_data_set = data_set
                data_set = data_set + "_hevc"
        file_list = []

        with open(os.path.join(data_csv_path, data_set, txt_file_name), 'r') as file:
            reader = file.readlines()
            if mode == "train":
                for line in reader:
                    offset_viedo_path, video_label = line.strip().split(',')
                    if data_set == "ssv2":
                        video_path = os.path.join(data_root_path, "ssv2_hevc", offset_viedo_path)
                    elif data_set == "ucf101":
                        video_path = os.path.join(data_root_path, "ucf101_hevc", offset_viedo_path)
                    elif data_set == "k400":
                        video_path = os.path.join(data_root_path, "k400_hevc", offset_viedo_path)
                    elif data_set == "hmdb51":
                        video_path = os.path.join(data_root_path, "hmdb51_hevc", offset_viedo_path)
                    elif data_set == "perception_test":
                        video_path = os.path.join(data_root_path, "perception_test_hevc", offset_viedo_path)
                    else:
                        video_path = os.path.join(data_root_path, data_set, offset_viedo_path)
                    # print(video_path)
                    file_list.append([video_path, int(video_label)])
            else:

                for line in reader:
                    offset_viedo_path, video_label = line.strip().split(',')
                    if data_set in ["ssv2", "ucf101_hevc", "k400_hevc", "hmdb51_hevc", "perception_test_hevc"]:
                        video_path = offset_viedo_path
                    else:
                        video_path = os.path.join(data_root_path, data_set, offset_viedo_path)
                    # print(video_path)
                    file_list.append([video_path, int(video_label)])
                # data_set = old_data_set 
    else:
        raise NotImplementedError("This function is not implemented yet")
           

    rank = int(os.getenv("RANK", 0))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))

    source_params = {
        "num_shards": world_size,
        "shard_id": rank,
        "file_list": file_list,
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "seed": seed + rank,
        "use_rgb": use_rgb,
        "input_size": input_size,
        "short_side_size": short_side_size,
        "multi_views": multi_views,
        "mean": mean,
        "std": std,
        "enable_res_zero_mask":  True,
        "res_block_size":        16,
        "res_min_drop_ratio":    float(os.environ.get("RES_MIN_DROP_RATIO", "0.0")),
        "hevc_y_only":           True,
    }


    pipe = dali_pipeline(
        batch_size = batch_size,
        num_threads = dali_num_threads,
        device_id = local_rank,
        seed = seed + rank,
        py_num_workers = dali_py_num_workers,
        py_start_method = 'spawn',
        prefetch_queue_depth = 1,
        mode = mode,
        source_params = source_params,
    )
    pipe.build()


    dataloader = DALIWarper(
        dali_iter = DALIGenericIterator(pipelines=pipe,
            output_map=['videos', 'res_zero_masks', 'labels'],
            auto_reset=False,
            size=-1,
            last_batch_padded=False,
            last_batch_policy=LastBatchPolicy.FILL,
            prepare_first_batch=False),
        step_data_num = len(file_list) // world_size // batch_size,
    )
    
    return dataloader