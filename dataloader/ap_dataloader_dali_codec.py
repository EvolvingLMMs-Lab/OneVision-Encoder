import os
import logging
import traceback
import numpy as np
import nvidia.dali.fn as fn
from typing import List, Tuple, Dict, Any
import torch
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
import decord
from nvidia.dali.pipeline import pipeline_def
from nvidia.dali.pipeline import DataNode
import nvidia.dali.math as dali_math
import pickle
import glob
import hashlib
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
import cv2
import numpy as np

def resize_and_center_crop_residuals(residuals_y, input_size):
    """
    residuals_y: numpy 数组，形状 (F, H, W) 或 (F, H, W, 1)
    返回: res_zero_masks，形状 (F, input_size, input_size, 1)，dtype=uint8
    """
    # 如果是 (F, H, W, 1)，先去掉最后一维
    if residuals_y.ndim == 4 and residuals_y.shape[-1] == 1:
        residuals_y = residuals_y[..., 0]  # -> (F, H, W)

    F, H, W = residuals_y.shape

    # 按你 DALI 里的逻辑：resize_shorter=input_size -> 按短边缩放
    scale = input_size / min(H, W)
    new_w = int(round(W * scale))
    new_h = int(round(H * scale))

    # 中心裁剪坐标（所有帧分辨率相同，只算一次）
    x1 = (new_w - input_size) // 2
    y1 = (new_h - input_size) // 2
    x2 = x1 + input_size
    y2 = y1 + input_size

    # 预分配输出: (F, input_size, input_size, 1) 对应 DALI 的 "FHWC"
    res_zero_masks = np.empty((F, input_size, input_size, 1), dtype=np.uint8)

    for i in range(F):
        frame = residuals_y[i]  # (H, W)

        # INTER_CUBIC 对应 DALI 的 INTERP_CUBIC
        resized_long = cv2.resize(
            frame,
            (new_w, new_h),
            interpolation=cv2.INTER_CUBIC
        )  # (new_h, new_w)

        cropped = resized_long[y1:y2, x1:x2]  # (input_size, input_size)

        # DALI 输出是 UINT8，这里也转成 uint8；如果你 residual 本来是 float，可以自定义归一化/阈值
        res_zero_masks[i, :, :, 0] = cropped.astype(np.uint8)

    return res_zero_masks

import numpy as np

def compute_visible_indices_cpu(
    residuals_y: np.ndarray,
    patch_size: int | tuple[int, int],
    K: int,
) -> np.ndarray:
    """
    CPU 版的 Top-K 可见 patch 选择逻辑，对应 PyTorch 版 mask_by_residual_topk 的单样本情形 (B=1)，
    只返回 visible_indices（不返回 mask 和 ids_restore）。

    Args:
        residuals_y: np.ndarray
            形状 (F, H, W, 1) 或 (F, H, W)，表示单个样本的残差。
            注意：建议在调用前就把 residuals_y 处理成“带符号”的残差，
            即与训练时喂给 mask_by_residual_topk 的 res.abs() 语义一致。
            如果 residuals_y 目前是 uint8 (0~255)，你可以在外部先做:
                residuals_y = residuals_y.astype(np.int16) - 128
        patch_size: int 或 (ph, pw)
            patch 的高宽。
        input_size: int
            目标输入尺寸 H=W=input_size。
            如果 residuals_y 的 H,W 已经是 input_size，可以不用它；
            此参数主要是为了兼容原函数签名。
        sequence_length: int
            序列长度 F（帧数），主要用于接口兼容；实际以 residuals_y 的第 0 维为准。
        K: int
            要保留的 Top-K patch 数量（k_keep）。

    Returns:
        visible_indices: np.ndarray, 形状 (K',)，dtype=int32
            选中的 patch 线性索引（升序），K' = clamp(K, 0, L)，L 为总 patch 数。
    """
    # ---------- 1. 统一 residuals_y 形状 ----------
    # 支持 (F, H, W, 1) 或 (F, H, W)
    if residuals_y.ndim == 4 and residuals_y.shape[-1] == 1:
        residuals_y = residuals_y.squeeze(-1)  # (F, H, W)

    if residuals_y.ndim != 3:
        raise ValueError(f"residuals_y 必须是 (F,H,W) 或 (F,H,W,1)，当前形状: {residuals_y.shape}")

    F, H, W = residuals_y.shape  # 实际使用的 F,H,W 以数据为准
    # sequence_length 和 input_size 主要是接口兼容，如果你想强约束可加检查：
    # if F != sequence_length:
    #     raise ValueError(f"sequence_length={sequence_length}, 但 residuals_y.shape[0]={F}")
    # if H != input_size or W != input_size:
    #     raise ValueError(f"input_size={input_size}, 但 residuals_y 形状为 H={H}, W={W}")

    # ---------- 2. patch 网格划分 ----------
    if isinstance(patch_size, int):
        ph = pw = patch_size
    else:
        ph, pw = patch_size

    if H % ph != 0 or W % pw != 0:
        raise ValueError(
            f"H/W 必须能被 patch 大小整除，当前 H={H}, W={W}, ph={ph}, pw={pw}"
        )

    hb, wb = H // ph, W // pw  # 每帧的 patch 网格数
    L = F * hb * wb            # 总 patch 数

    # ---------- 3. K 边界处理（与 PyTorch 版一致） ----------
    K_clamped = int(max(0, min(K, L)))
    if K_clamped == 0:
        return np.empty((0,), dtype=np.int32)

    # ---------- 4. 计算每个 patch 的残差得分 ----------
    # PyTorch 版是：res_abs = res.abs().squeeze(1);
    #               scores = res_abs.reshape(B,T,hb,ph,wb,pw).sum(dim=(3,5))
    # 这里是单样本 (B=1)，且 residuals_y 已经是绝对值或带符号残差：
    res_abs = np.abs(residuals_y)  # (F,H,W)

    # reshape 成 (F, hb, ph, wb, pw)，对 patch 内求和 -> (F, hb, wb)
    res_reshaped = res_abs.reshape(F, hb, ph, wb, pw)
    patch_scores = res_reshaped.sum(axis=(2, 4))      # (F, hb, wb)

    # 展平为一维 (L,) —— 对应 PyTorch 版 scores.reshape(B,L) 中 B=1 的情况
    patch_scores_flat = patch_scores.reshape(L)       # (L,)

    # ---------- 5. 选 Top-K 索引（与 PyTorch 版逻辑对应） ----------
    # PyTorch: topk_idx = torch.topk(scores, k=K, dim=1, largest=True, sorted=False).indices
    #         visible_indices = torch.sort(topk_idx, dim=1).values
    # NumPy 等价写法：
    # - argpartition 到倒数第 K_clamped 个，再取这 K_clamped 个；
    # - 再排序，得到升序索引。
    topk_indices = np.argpartition(patch_scores_flat, -K_clamped)[-K_clamped:]
    visible_indices = np.sort(topk_indices).astype(np.int32)  # (K_clamped,)

    return visible_indices


def _get_cache_path(video_path: str, cache_dir: str) -> str:
    """
    Generate a cache file path for a given video path.
    
    Args:
        video_path: Path to the video file
        cache_dir: Directory to store cache files
        
    Returns:
        Path to the cache pkl file
    """
    if not cache_dir:
        return None
    
    # Create a unique filename based on the video path
    # Use the video path hash to avoid filesystem issues with long paths
    video_hash = hashlib.md5(video_path.encode()).hexdigest()
    cache_filename = f"visible_indices_{video_hash}.pkl"
    cache_path = os.path.join(cache_dir, cache_filename)
    
    return cache_path

def save_cache(visible_indices: np.ndarray, video_path: str, cache_dir: str) -> None:
    """
    Save visible_indices to a pkl file.
    
    Args:
        visible_indices: numpy array of visible indices to save
        video_path: Path to the video file (used to generate cache filename)
        cache_dir: Directory to store cache files
    """
    if not cache_dir:
        return
    
    cache_path = _get_cache_path(video_path, cache_dir)
    if cache_path is None:
        return
    
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    
    # Save to a temporary file first, then rename atomically
    temp_path = cache_path + '.tmp'
    try:
        with open(temp_path, 'wb') as f:
            pickle.dump(visible_indices, f)
        os.replace(temp_path, cache_path)
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        # Don't raise - caching is optional
        print(f"Warning: Failed to save cache for {video_path}: {e}")

def get_cache(video_path: str, cache_dir: str) -> np.ndarray:
    """
    Load visible_indices from a pkl file.
    
    Args:
        video_path: Path to the video file (used to generate cache filename)
        cache_dir: Directory where cache files are stored
        
    Returns:
        numpy array of visible indices, or None if cache doesn't exist
    """
    if not cache_dir:
        return None
    
    cache_path = _get_cache_path(video_path, cache_dir)
    if cache_path is None or not os.path.exists(cache_path):
        return None
    
    try:
        with open(cache_path, 'rb') as f:
            visible_indices = pickle.load(f)
        return visible_indices
    except Exception as e:
        print(f"Warning: Failed to load cache for {video_path}: {e}")
        return None

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
            "visible_indices": data_dict["visible_indices"],
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
        self.input_size = source_params['input_size']
        self.patch_size = source_params['patch_size']
        self.cache_dir = source_params.get('cache_dir', '')
        self.K_keep = source_params.get('K_keep')
        
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
                    # combined_data = np.concatenate([video_data, residuals_y], axis=-1)

                    if H0 != video_data.shape[1] or W0 != video_data.shape[2]:
                        print("[warn] residual尺寸与视频不一致: res=(%d,%d) video=(%d,%d)" % (H0, W0, video_data.shape[1], video_data.shape[2]))
                finally:
                    if _prev_y_only is None:
                        os.environ.pop("UMT_HEVC_Y_ONLY", None)
                    else:
                        os.environ["UMT_HEVC_Y_ONLY"] = _prev_y_only
                
            return video_data, residuals_y, duration, frame_id_list
    
    def __call__(self, sample_info):
        if sample_info.iteration >= self.full_iterations:
            raise StopIteration
        if self.last_seen_epoch != sample_info.epoch_idx:
            self.last_seen_epoch = sample_info.epoch_idx
            cur_seed = self.seed + sample_info.epoch_idx
            self.perm = np.random.default_rng(seed=cur_seed).permutation(len(self.file_list))
        sample_idx = self.perm[sample_info.idx_in_epoch + self.shard_offset]
        example_info = self.file_list[sample_idx]
        video_path, video_label = example_info
        
        # Try to load visible_indices from cache first
        visible_indices = get_cache(video_path, self.cache_dir)
        
        if visible_indices is None:
            try:
                video_data, residuals_y, duration, frame_id_list = self.get_frame_id_list(video_path, self.sequence_length)
            except:
                video_path, video_label = self.replace_example_info
                video_data, residuals_y, duration, frame_id_list = self.get_frame_id_list(video_path, self.sequence_length)
            residuals_y = resize_and_center_crop_residuals(residuals_y, input_size=self.input_size)
            visible_indices = compute_visible_indices_cpu(
                residuals_y=residuals_y,
                patch_size=self.patch_size,
                K=self.K_keep,
            )
            save_cache(visible_indices, video_path, self.cache_dir)
        else:
            try:
                video_data, duration, frame_id_list = self.get_frame_id_list_simple(video_path, self.sequence_length)
            except:
                video_path, video_label = self.replace_example_info
                video_data, duration, frame_id_list = self.get_frame_id_list_simple(video_path, self.sequence_length)
        
        visible_indices = visible_indices.astype(np.int64)
        
        return video_data, visible_indices, np.int64([int(video_label)]), np.array(frame_id_list, dtype=np.int64), np.int64([duration])
    
    def get_frame_id_list_simple(self, video_path, sequence_length):
        """Load video frames without residual computation (for cache hits)"""
        decord_vr = decord.VideoReader(video_path, num_threads=2, ctx=decord.cpu(0))
        duration = len(decord_vr)
        frame_id_list = np.linspace(0, duration - 1, sequence_length, dtype=int).tolist()
        decord_vr.seek(0)
        video_data = decord_vr.get_batch(frame_id_list).asnumpy()
        return video_data, duration, frame_id_list
    

@pipeline_def(enable_conditionals=True)
def dali_pipeline(mode, source_params):
    
    short_side_size = source_params['short_side_size']
    input_size = source_params['input_size']
    mean = source_params['mean']
    std = source_params['std']
    
    if mode == "train":
        video, visible_indices, labels, indices, total_frames = fn.external_source(
            source = ExternalInputCallable(mode, source_params),
            num_outputs = 5,
            batch = False,
            parallel = True,
            dtype = [types.UINT8, types.INT64, types.INT64, types.INT64, types.INT64],
            layout = ["FHWC", "", "", "", ""]
        )
            
        video = video.gpu()
        visible_indices = visible_indices.gpu()
        
        video = fn.resize(video, device="gpu", resize_shorter=input_size, interp_type=types.INTERP_CUBIC)
        videos = fn.crop_mirror_normalize(
            video,
            device="gpu",
            crop=[input_size, input_size],
            crop_pos_x=0.5,   # 中心裁剪
            crop_pos_y=0.5,
            dtype=types.UINT8,
            output_layout="FHWC"
        )

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

        labels = labels.gpu()
        indices = indices.gpu()
        total_frames = total_frames.gpu()
        return videos, visible_indices, labels, indices, total_frames
    else:
        video, visible_indices, labels, indices, total_frames = fn.external_source(
            source = ExternalInputCallable(mode, source_params),
            num_outputs = 5,
            batch = False,
            parallel = True,
            dtype = [types.UINT8, types.INT64, types.INT64, types.INT64, types.INT64],
            layout = ["FHWC", "", "", "", ""]
        )
        video = video.gpu()
        visible_indices = visible_indices.gpu()
        
        video = fn.resize(video, device="gpu", resize_shorter=input_size, interp_type=types.INTERP_CUBIC)
        videos = fn.crop_mirror_normalize(
            video,
            device="gpu",
            crop=[input_size, input_size],
            crop_pos_x=0.5,   # 中心裁剪
            crop_pos_y=0.5,
            dtype=types.UINT8,
            output_layout="FHWC"
        )
        
        videos = fn.crop_mirror_normalize(videos, dtype=types.FLOAT, output_layout = "CFHW",
                                        mean=[m*255.0 for m in mean], std=[m*255.0 for m in std], device="gpu")
        
        labels = labels.gpu()
        indices = indices.gpu()
        total_frames = total_frames.gpu()
        return videos, visible_indices, labels, indices, total_frames


def get_dali_dataloader_codec(
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
    feature_extract: bool = True,
    patch_size: int = 14,
    cache_dir: str = None,
    K_keep: int = 2048,
) -> DALIWarper:
    """
    Create a DALI dataloader for video data with IP-MV (motion vector) features.
    
    Args:
        cache_dir: Directory to store cached residuals. If None, uses data_root_path/cache_residuals.
                   Set to empty string "" to disable caching.
    """
    print(f"[{mode} loader] Reading for: {data_csv_path}")
    
    # Set default cache directory if not specified
    if cache_dir is None:
        cache_dir = os.path.join(data_root_path, "cache_residuals")
    
    # Print cache status
    rank = int(os.getenv("RANK", "0"))
    if cache_dir:
        if rank == 0:
            print(f"[{mode} loader] Cache directory: {cache_dir}")
            if not os.path.exists(cache_dir):
                print(f"[{mode} loader] Creating cache directory (first run will be slower)")
            else:
                print(f"[{mode} loader] Using existing cache (should be faster)")
    else:
        if rank == 0:
            print(f"[{mode} loader] Caching disabled")

    file_list = []
    try:
        with open(data_csv_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) >= 2:
                    offset_path, label = parts[0], parts[1]
                    full_path = os.path.join(data_root_path, offset_path)
                    file_list.append((full_path, int(label)))
    except FileNotFoundError:
        raise FileNotFoundError(f"Data list file not found at: {data_csv_path}")
    
    if not file_list:
        raise ValueError(f"File list from {data_csv_path} is empty.")

    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))

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
        "mean": mean,
        "std": std,
        "enable_res_zero_mask":  True,
        "res_block_size":        16,
        "res_min_drop_ratio":    float(os.environ.get("RES_MIN_DROP_RATIO", "0.0")),
        "hevc_y_only":           True,
        "patch_size": patch_size,
        "cache_dir": cache_dir,
        "K_keep": K_keep,
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
            output_map=['videos', 'visible_indices','labels', "indices", "total_frames"],
            auto_reset=False,
            size=-1,
            last_batch_padded=False,
            last_batch_policy=LastBatchPolicy.FILL,
            prepare_first_batch=False),
        step_data_num = len(file_list) // world_size // batch_size,
    )
    return dataloader
