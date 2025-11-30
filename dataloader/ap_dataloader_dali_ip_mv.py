import os
import logging
import traceback
import numpy as np
import nvidia.dali.fn as fn
from typing import List, Tuple, Dict, Any

import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
import decord
from nvidia.dali.pipeline import pipeline_def
import glob
try:
    from .hevc_feature_decoder_mv import HevcFeatureReader
except Exception:
    from hevc_feature_decoder_mv import HevcFeatureReader
try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False

import warnings
warnings.filterwarnings("ignore")

def _fuse_energy(norm_mv: np.ndarray, norm_res: np.ndarray, mode: str = "weighted", w_mv: float = 1.0, w_res: float = 1.0):
    """Fuse two normalized maps into one normalized map in [0,1]."""
    mode = (mode or "weighted").lower()
    if mode == "max":
        fused = np.maximum(norm_mv, norm_res)
    elif mode == "sum":
        fused = np.clip(norm_mv + norm_res, 0.0, 1.0)
    elif mode == "geomean":
        fused = np.sqrt(np.clip(norm_mv, 0.0, 1.0) * np.clip(norm_res, 0.0, 1.0))
    else:  # weighted
        denom = float(w_mv + w_res) if (w_mv + w_res) != 0 else 1.0
        fused = (float(w_mv) * norm_mv + float(w_res) * norm_res) / denom
    return np.clip(fused, 0.0, 1.0).astype(np.float32)

def _residual_energy_norm(res_y: np.ndarray, pct: float = 95.0):
    """Return (norm_HxW_float32_in_[0,1], scale_max_level). No gamma/colormap."""
    x = np.abs(res_y.astype(np.float32) - 128.0)
    a = float(np.percentile(x, pct))
    a = max(a, 1.0)
    norm = np.clip(x / a, 0.0, 1.0)
    return norm.astype(np.float32), a

def _mv_energy_norm(
    mvx: np.ndarray,
    mvy: np.ndarray,
    H: int,
    W: int,
    mv_unit_div: float = 4.0,
    pct: float = 95.0,
):
    """Return (norm_HxW_float32_in_[0,1], scale_max_px). No gamma/colormap."""
    vx = mvx.astype(np.float32) / float(mv_unit_div)
    vy = mvy.astype(np.float32) / float(mv_unit_div)
    mag = np.sqrt(vx * vx + vy * vy)  # pixels
    a = float(np.percentile(mag, pct))
    a = max(a, 1e-6)
    norm = np.clip(mag / a, 0.0, 1.0)
    norm_u = cv2.resize(norm, (W, H), interpolation=cv2.INTER_NEAREST)
    return norm_u.astype(np.float32), a

class DALIWarper(object):
    def __init__(self, dali_iter, step_data_num):
        self.iter = dali_iter
        self.step_data_num = step_data_num

    def __next__(self):
        data_dict = self.iter.__next__()[0]

        return {
            "videos": data_dict["videos"],
            "res_zero_masks": data_dict["res_zero_masks"],
            "labels": data_dict["labels"],
            "indices": data_dict["indices"],
            "total_frames": data_dict["total_frames"]
        }
    
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


    def get_frame_id_list(self, video_path, sequence_length,    
                          mv_unit_div: float = 4.0,   # quarter-pel -> pixel
                          mv_pct: float = 95.0,       # MV 归一化分位数（传给 _mv_energy_norm）
                          res_pct: float = 95.0,      # 残差归一化分位数（传给 _residual_energy_norm）
                          fuse_mode: str = "weighted",
                          w_mv: float = 1.0,
                          w_res: float = 1.0,):
        
        decord_vr = decord.VideoReader(video_path, num_threads=2, ctx=decord.cpu(0))
        duration = len(decord_vr)

        if self.mode in ["train", "val"]:

            # 按照每一个seq进行分group
            # average_duration = duration // sequence_length
            
            # if average_duration > 0:
            #     frame_id_list = list(
            #         np.multiply(list(range(sequence_length)), average_duration))
            # else:
            #     if duration >= sequence_length:
            #         frame_id_list = list(range(0, min(duration, sequence_length)))
            #     else:
            #         frame_id_list = list(range(duration)) + [duration - 1] * (sequence_length - duration)
            
            frame_id_list = np.linspace(0, duration - 1, sequence_length, dtype=int).tolist()
            try:
                key_idx = None
                if hasattr(decord_vr, "get_key_indices"):
                    key_idx = decord_vr.get_key_indices()
                elif hasattr(decord_vr, "get_keyframes"):
                    key_idx = decord_vr.get_keyframes()
                if key_idx is not None:
                    # key_idx 可能是 NDArray；转成 Python list 的整型帧号集合 只保留一帧为I帧
                    I_list = np.asarray(key_idx)
                    I_list = I_list.tolist()[0] if I_list.ndim > 1 else I_list.tolist()
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
                # gop = max(1,int(self.gop_size))
                # I_list = [i for i, fid in enumerate(frame_id_list)if(int(fid)% gop)== 0]
                # 第一帧为I帧
                I_list = [0]
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

            if self.enable_res_zero_mask:
                _prev_y_only = os.environ.get("UMT_HEVC_Y_ONLY", None)
                try:
                    if self.hevc_y_only:
                        os.environ["UMT_HEVC_Y_ONLY"] = "1"

                    # --- residual read with HevcFeatureReader; prefix fast path if frame_ids are 0..F-1 ---
                    F_sel = len(frame_id_list)
                    wanted = set(frame_id_list)
                    idx2pos = {fid: i for i, fid in enumerate(frame_id_list)}
                    I_pos_set = set(int(pos_map.get(fid, -1)) for fid in I_list if int(pos_map.get(fid, -1)) >= 0)

                    # Classic list-based accumulation (no FAST_POSTPROC)
                    residuals_y = [None] * F_sel
                    H0 = W0 = None
                    dtype0 = None

                    hevc_threads = getattr(self, 'hevc_n_parallel', 6)
                    rdr = HevcFeatureReader(video_path, nb_frames=None, n_parallel=hevc_threads)
                    cur_idx = 0
                    try:
                        for frame_tuple, meta in rdr.nextFrameEx():
                            if cur_idx in wanted:
                                pos = idx2pos[cur_idx]
                                (
                                    frame_type,
                                    quadtree_stru,
                                    rgb,
                                    mv_x_L0,
                                    mv_y_L0,
                                    mv_x_L1,
                                    mv_y_L1,
                                    ref_off_L0,
                                    ref_off_L1,
                                    size,
                                    residual,
                                ) = frame_tuple

                                # I 帧：直接置 0（与你 residual 逻辑一致）
                                if pos in I_pos_set:
                                    if H0 is None:
                                        # 用残差Y来确定输出尺寸/类型
                                        y0 = residual if residual.ndim == 2 else cv2.cvtColor(residual, cv2.COLOR_BGR2YUV)[:, :, 0]
                                        y0 = np.asarray(y0)
                                        H0, W0, dtype0 = int(y0.shape[0]), int(y0.shape[1]), y0.dtype
                                    residuals_y[pos] = np.zeros((H0, W0), dtype=dtype0 or np.uint8)

                                else:
                                    # 1) 取 MV (L0) 并上采样到 H×W
                                    mvx_hw = rdr._upsample_mv_to_hw(mv_x_L0.astype(np.float32))
                                    mvy_hw = rdr._upsample_mv_to_hw(mv_y_L0.astype(np.float32))

                                    # 2) 取残差 Y
                                    Y_res = residual if residual.ndim == 2 else cv2.cvtColor(residual, cv2.COLOR_BGR2YUV)[:, :, 0]

                                    # 初始化输出尺寸/类型（只在第一次命中时做）
                                    if H0 is None:
                                        H0, W0, dtype0 = int(Y_res.shape[0]), int(Y_res.shape[1]), Y_res.dtype

                                    # 若当前帧的尺寸与 H0×W0 不一致，做一次 resize 对齐（极少见，兜底）
                                    if (Y_res.shape[0] != H0) or (Y_res.shape[1] != W0):
                                        Y_res = cv2.resize(Y_res, (W0, H0), interpolation=cv2.INTER_AREA)
                                    if (mvx_hw.shape[0] != H0) or (mvx_hw.shape[1] != W0):
                                        mvx_hw = cv2.resize(mvx_hw, (W0, H0), interpolation=cv2.INTER_NEAREST)
                                        mvy_hw = cv2.resize(mvy_hw, (W0, H0), interpolation=cv2.INTER_NEAREST)

                                    # 3) 归一化到 [0,1]
                                    #    下面这些超参请确保在外层有定义；如果没有，你也可以给个默认值：
                                    #    mv_unit_div, mv_pct, res_pct, fuse_mode, w_mv, w_res
                                    mv_norm, _ = _mv_energy_norm(mvx_hw, mvy_hw, H0, W0, mv_unit_div=mv_unit_div, pct=mv_pct)
                                    res_norm, _ = _residual_energy_norm(Y_res, pct=res_pct)

                                    # 4) 融合（weighted/sum/max/geomean 均可，默认 weighted）
                                    fused = _fuse_energy(mv_norm, res_norm, mode=fuse_mode, w_mv=w_mv, w_res=w_res)

                                    # 写回你原来的容器（保持最小改动，用 uint8 存）
                                    residuals_y[pos] = (np.clip(fused, 0.0, 1.0) * 255.0).astype(dtype0 or np.uint8)

                                # 结束条件
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
                
            return combined_data, duration, frame_id_list
    
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
        try:
            combined_data, duration, frame_id_list = self.get_frame_id_list(video_path, self.sequence_length)
        except:
            video_path, video_label = self.replace_example_info
            combined_data, duration, frame_id_list = self.get_frame_id_list(video_path, self.sequence_length)
        
        # print(np.array(frame_id_list, dtype=np.int64),np.int64([duration]))
        return combined_data, np.int64([int(video_label)]), np.array(frame_id_list, dtype=np.int64), np.int64([duration])

@pipeline_def(enable_conditionals=True)
def dali_pipeline(mode, source_params):
    
    short_side_size = source_params['short_side_size']
    input_size = source_params['input_size']
    mean = source_params['mean']
    std = source_params['std']
    
    if not source_params['multi_views']:
        if mode == "train":
            combined_data, labels, indices, total_frames = fn.external_source(
                source = ExternalInputCallable(mode, source_params),
                num_outputs = 4,
                batch = False,
                parallel = True,
                dtype = [types.UINT8, types.INT64],
                layout = ["FHWC", "C"]
            )
            combined_data = combined_data.gpu()
            combined_data = fn.resize(combined_data, device="gpu", resize_shorter=input_size, interp_type=types.INTERP_CUBIC)
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
            videos = fn.slice(combined_data, start=[0], shape=[video_channels], axes=[3])

            res_zero_masks = fn.slice(combined_data, start=[video_channels], shape=[1], axes=[3])

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

            res_zero_masks = fn.transpose(res_zero_masks, perm=[3, 0, 1, 2])  # FHWC -> CFHW
            labels = labels.gpu()
            indices = indices.gpu()
            total_frames = total_frames.gpu()

            return videos, res_zero_masks, labels, indices, total_frames
        else:
            combined_data, labels, indices, total_frames = fn.external_source(
                source = ExternalInputCallable(mode, source_params),
                num_outputs = 4,
                batch = False,
                parallel = True,
                dtype = [types.UINT8, types.INT64],
                layout = ["FHWC", "C"]
            )
            combined_data = combined_data.gpu()
            combined_data = fn.resize(combined_data, device="gpu", resize_shorter=input_size, interp_type=types.INTERP_CUBIC)
            combined_data = fn.crop_mirror_normalize(combined_data, device="gpu", crop=[input_size, input_size], crop_pos_x=0.5, crop_pos_y=0.5, dtype=types.UINT8, output_layout="FHWC")

            video_channels = source_params.get('video_channels', 3)  # 例如 RGB=3
            videos = fn.slice(combined_data, start=[0], shape=[video_channels], axes=[3])

            res_zero_masks = fn.slice(combined_data, start=[video_channels], shape=[1], axes=[3])
            
            videos = fn.crop_mirror_normalize(videos, dtype=types.FLOAT, output_layout = "CFHW", mean=[m*255.0 for m in mean], std=[m*255.0 for m in std], device="gpu")
            res_zero_masks = fn.transpose(res_zero_masks, perm=[3, 0, 1, 2])  # FHWC -> CFHW
            labels = labels.gpu()
            indices = indices.gpu()
            total_frames = total_frames.gpu()

            return videos, res_zero_masks, labels, indices, total_frames


def get_dali_dataloader(data_root_path,
                    data_csv_path,
                    data_set,
                    dali_num_threads = 4,
                    dali_py_num_workers = 8,
                    batch_size = 32,
                    input_size = 224,
                    short_side_size = 239,
                    sequence_length = 16,
                    decord_num_threads: int = 2,
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
                    elif data_set == "diving48":
                        video_path = os.path.join(data_root_path, "diving48_hevc", offset_viedo_path)
                    else:
                        video_path = os.path.join(data_root_path, data_set, offset_viedo_path)
                    file_list.append([video_path, int(video_label)])
            else:
                for line in reader:
                    offset_viedo_path, video_label = line.strip().split(',')
                    if data_set in ["ssv2", "ucf101_hevc", "k400_hevc", "hmdb51_hevc", "perception_test_hevc", "diving48_hevc"]:
                        video_path = offset_viedo_path
                    else:
                        video_path = os.path.join(data_root_path, data_set, offset_viedo_path)
                    file_list.append([video_path, int(video_label)])
    elif num_shots is None:
        print(f"[{mode} loader] Reading for: {data_csv_path}")
        file_list = []
        try:
            with open(data_csv_path, "r") as f:
                for line in f:
                    offset_path, label = line.strip().split(",")
                    full_path = os.path.join(data_root_path, offset_path)
                    file_list.append((full_path, int(label)))
        except FileNotFoundError:
            raise FileNotFoundError(f"Data list file not found at: {data_csv_path}")
        if not file_list:
            raise ValueError(f"File list from {data_csv_path} is empty.")
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
            output_map=['videos', 'res_zero_masks','labels', "indices", "total_frames"],
            auto_reset=False,
            size=-1,
            last_batch_padded=False,
            last_batch_policy=LastBatchPolicy.FILL,
            prepare_first_batch=False),
        step_data_num = len(file_list) // world_size // batch_size,
    )
    return dataloader
