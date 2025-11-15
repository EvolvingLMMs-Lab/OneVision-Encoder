import os
import logging
import traceback
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
import torch
import decord
import numpy as np
import cv2
import glob
import re
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.auto_aug import rand_augment
from nvidia.dali.pipeline import pipeline_def

try:
    from .hevc_feature_decoder import HevcFeatureReader
except Exception:
    from hevc_feature_decoder import HevcFeatureReader

# ---- DALI v1 file-only logger ----
def _get_dali_v1_logger():
    logger = logging.getLogger("dali_v1_logger")
    if getattr(logger, "_configured", False):
        return logger
    logger.propagate = False
    logger.setLevel(logging.INFO)
    log_dir = os.environ.get("LLAVA_OUTPUT_DIR") or os.environ.get("OUTPUT_DIR") or os.getcwd()
    try:
        os.makedirs(log_dir, exist_ok=True)
    except Exception:
        log_dir = os.getcwd()
    rank = int(os.environ.get("RANK", "0"))
    fh = logging.FileHandler(os.path.join(log_dir, f"dali_v1_rank{rank:02d}.log"))
    fmt = logging.Formatter(f"rank-id:{rank:03d}:%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(fmt)
    fh.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(fh)
    logger._configured = True
    return logger

# ---- file_list expansion helpers ----
_VIDEO_EXTS = {'.mp4', '.mkv', '.avi', '.mov', '.webm', '.m4v', '.hevc', '.h265', '.265'}
_LIST_EXTS  = {'.txt', '.lst', '.list', '.csv'}

def _is_video_path(p: str) -> bool:
    return os.path.splitext(p)[1].lower() in _VIDEO_EXTS

def _is_list_path(p: str) -> bool:
    return os.path.splitext(p)[1].lower() in _LIST_EXTS

def _parse_line_to_example(line: str):
    s = line.strip()
    if not s or s.startswith('#'):
        return None
    # Accept common formats: "path label" | "path,label" | "label path" | "path"
    if ',' in s:
        a, b = s.split(',', 1)
        a, b = a.strip(), b.strip()
        if _is_video_path(a):
            path, lab = a, b
        else:
            path, lab = b, a
        try:
            lab = int(re.findall(r"-?\d+", str(lab))[0])
        except Exception:
            lab = 0
        return (path, lab)
    toks = s.split()
    if len(toks) == 1:
        return (toks[0], 0)
    if len(toks) >= 2:
        if _is_video_path(toks[0]) and not _is_video_path(toks[1]):
            p = toks[0]
            try:
                lab = int(re.findall(r"-?\d+", toks[1])[0])
            except Exception:
                lab = 0
            return (p, lab)
        if _is_video_path(toks[1]) and not _is_video_path(toks[0]):
            p = toks[1]
            try:
                lab = int(re.findall(r"-?\d+", toks[0])[0])
            except Exception:
                lab = 0
            return (p, lab)
        # Fallback: first token is path, second token is label
        p = toks[0]
        try:
            lab = int(re.findall(r"-?\d+", toks[1])[0])
        except Exception:
            lab = 0
        return (p, lab)
    return None

def _read_list_file(path: str):
    logger = _get_dali_v1_logger()
    examples = []
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                ex = _parse_line_to_example(line)
                if ex is not None:
                    examples.append(ex)
    except Exception as e:
        logger.warning(f"[file_list] failed to read list '{path}': {e}")
    return examples

def _expand_file_list(arg: str):
    """Return a list of (video_path, label) tuples from a directory / glob / list file / single video path."""
    logger = _get_dali_v1_logger()
    if not isinstance(arg, str):
        return []
    arg = arg.strip()
    if not arg:
        return []
    # Directory: prefer list files; otherwise scan video files
    if os.path.isdir(arg):
        list_files = []
        for ext in sorted(_LIST_EXTS):
            list_files.extend(sorted(glob.glob(os.path.join(arg, f"*{ext}"))))
        examples = []
        if list_files:
            for lf in list_files:
                examples.extend(_read_list_file(lf))
            if not examples:
                logger.warning(f"[file_list] directory '{arg}' has list files but none yielded examples")
            return examples
        vids = []
        for fname in sorted(os.listdir(arg)):
            p = os.path.join(arg, fname)
            if os.path.isfile(p) and _is_video_path(p):
                vids.append(p)
        if vids:
            return [(p, 0) for p in vids]
        # Try recursive search for videos in subdirectories
        rec_globs = []
        for ext in sorted(_VIDEO_EXTS):
            rec_globs.extend(sorted(glob.glob(os.path.join(arg, '**', f'*{ext}'), recursive=True)))
        if rec_globs:
            return [(p, 0) for p in rec_globs]
        logger.warning(f"[file_list] directory '{arg}' contains no list files or video files (even recursively)")
        return []
    # Glob pattern
    if any(ch in arg for ch in ['*', '?', '[']):
        paths = sorted(glob.glob(arg))
        vids  = [p for p in paths if os.path.isfile(p) and _is_video_path(p)]
        if vids:
            return [(p, 0) for p in vids]
        lists = [p for p in paths if os.path.isfile(p) and _is_list_path(p)]
        examples = []
        for lf in lists:
            examples.extend(_read_list_file(lf))
        return examples
    # Single file
    if os.path.isfile(arg):
        # Known list file by extension
        if _is_list_path(arg):
            exs = _read_list_file(arg)
            if not exs:
                logger.warning(f"[file_list] list '{arg}' produced no examples")
            return exs
        # Known video file by extension
        if _is_video_path(arg):
            return [(arg, 0)]
        # Otherwise, try to treat it as a plain-text list file without extension
        exs = _read_list_file(arg)
        if exs:
            logger.info(f"[file_list] treated '{arg}' as a list file without extension; {len(exs)} examples")
            return exs
        # Fallback: treat as a single path (may be remote or produced later)
        logger.warning(f"[file_list] '{arg}' is a file but neither list nor known video format; treating as a single path with label 0")
        return [(arg, 0)]
    # Fallback: treat as single path (may be remote or non-existing yet)
    if _is_video_path(arg):
        return [(arg, 0)]
    logger.warning(f"[file_list] given as string; failed to read list '{arg}'. Using it as a single video path with label 0.")
    return [(arg, 0)]

def _maybe_swap_to_hevc(p: str) -> str:
    """
    If the path points to the non-HEVC directory:
      .../videos_frames64_kinetics_ssv2/videos_frames64_kinetics_ssv2/
    rewrite it to the HEVC directory:
      .../videos_frames64_kinetics_ssv2/videos_frames64_kinetics_ssv2_hevc/
    Only apply the rewrite if the target file exists. Otherwise, return the original path.
    """
    try:
        if not isinstance(p, str):
            return p
        old_seg = "/videos_frames64_kinetics_ssv2/videos_frames64_kinetics_ssv2/"
        new_seg = "/videos_frames64_kinetics_ssv2/videos_frames64_kinetics_ssv2_hevc/"
        if old_seg in p:
            cand = p.replace(old_seg, new_seg)
            if os.path.exists(cand):
                return cand
    except Exception:
        pass
    return p

def _build_zero_res_block_mask(residual: np.ndarray,
                               block: int = 16,
                               thresh: int = 0,
                               min_drop_ratio: float = 0.0) -> np.ndarray:
    """
    生成像素级 mask (H, W, 1)，值为 255 的像素属于被丢弃的块。
    - residual: 2D (H,W) 或 3D (H,W,C)；若 3D 取第 1 通道（Y）
    - block:    分块大小（像素）
    - thresh:   |residual-128| 的块内绝对值之和 <= thresh 视为“全零块”
    - min_drop_ratio: 最小丢弃比例（0 表示只丢全零块；>0 表示至少丢这么多比例的块）
    """
    if residual.ndim == 3:
        residual = residual[..., 0]
    res = residual.astype(np.int16) - 128
    H, W = res.shape
    hb, wb = H // block, W // block
    if hb == 0 or wb == 0:
        return np.zeros((H, W, 1), dtype=np.uint8)

    # (hb, wb, block, block)
    res_c = res[:hb * block, :wb * block].reshape(hb, block, wb, block)
    # 每个块的 |.| 总和 -> (hb, wb)
    s = np.abs(res_c).sum(axis=(1, 3))

    # 基于“全零块”的初始丢弃集合
    blk_mask = (s <= thresh).astype(np.uint8)  # 1 表示丢弃
    total_blocks = hb * wb
    if min_drop_ratio and min_drop_ratio > 0:
        target_k = int(np.ceil(min_drop_ratio * total_blocks))
        cur_k = int(blk_mask.sum())
        if cur_k < target_k:
            # 需要额外丢弃若干个“残差最小”的块
            need = target_k - cur_k
            # 展平成一维，按 s 从小到大排序
            flat_s = s.reshape(-1)
            order = np.argsort(flat_s)  # 小 -> 大
            flat_mask = blk_mask.reshape(-1).astype(bool)
            # 跳过已经选中的（全零）块
            extras = [idx for idx in order if not flat_mask[idx]]
            if need > 0 and len(extras) > 0:
                pick = extras[:need]
                flat_mask[pick] = True
                blk_mask = flat_mask.reshape(hb, wb).astype(np.uint8)

    # 上采样回像素级
    mask_small = blk_mask
    mask = np.repeat(np.repeat(mask_small, block, axis=0), block, axis=1)
    # 需要的话做 padding
    if mask.shape[0] != H or mask.shape[1] != W:
        pad = np.zeros((H, W), dtype=np.uint8)
        pad[:mask.shape[0], :mask.shape[1]] = mask
        mask = pad
    return (mask[..., None] * 255).astype(np.uint8)

def _build_zero_res_block_mask_topk(residual: np.ndarray,
                                    block: int = 16,
                                    drop_ratio: float = 0.0) -> np.ndarray:
    """
    生成像素级 mask (H, W, 1)，值为 255 的像素属于被丢弃的块。
    直接丢弃残差最小的 top-k 块，而不是只丢全零块。

    参数：
    - residual: 2D (H,W) 或 3D (H,W,C)；若 3D 取第 1 通道（Y）
    - block:    分块大小（像素）
    - drop_ratio: 丢弃比例 (0~1)，表示丢掉残差最小的这部分块
    """
    if residual.ndim == 3:
        residual = residual[..., 0]
    res = residual.astype(np.int16) - 128
    H, W = res.shape
    hb, wb = H // block, W // block
    if hb == 0 or wb == 0:
        return np.zeros((H, W, 1), dtype=np.uint8)

    # (hb, wb, block, block)
    res_c = res[:hb * block, :wb * block].reshape(hb, block, wb, block)
    # 每个块的 |.| 总和 -> (hb, wb)
    s = np.abs(res_c).sum(axis=(1, 3))

    total_blocks = hb * wb
    # print("drop_ratio", drop_ratio)
    k = int(np.ceil(drop_ratio * total_blocks)) if drop_ratio > 0 else 0

    blk_mask = np.zeros((hb, wb), dtype=np.uint8)
    if k > 0:
        # 展平成一维并按残差能量从小到大排序
        flat_s = s.reshape(-1)
        order = np.argsort(flat_s)  # 小 → 大
        pick = order[:k]  # 选前k个最小的块
        flat_mask = np.zeros_like(flat_s, dtype=bool)
        flat_mask[pick] = True
        blk_mask = flat_mask.reshape(hb, wb).astype(np.uint8)

    # 上采样回像素级
    mask = np.repeat(np.repeat(blk_mask, block, axis=0), block, axis=1)
    if mask.shape[0] != H or mask.shape[1] != W:
        pad = np.zeros((H, W), dtype=np.uint8)
        pad[:mask.shape[0], :mask.shape[1]] = mask
        mask = pad
    return (mask[..., None] * 255).astype(np.uint8)


# ---- Visualization helper ----
def _viz_mask_apply_and_save(video_data: np.ndarray,
                             res_zero_masks: np.ndarray,
                             out_dir: str,
                             tag: str,
                             frames: int = 1) -> None:
    """
    Save a few visualization frames where "masked blocks" are painted black and others keep original pixels.
    Args:
      video_data: (F,H,W,3) uint8 RGB from decord
      res_zero_masks: (F,H,W,1) uint8 {0,255}
      out_dir: directory to save images
      tag: filename prefix
      frames: how many frames to dump (starting from 0..)
    """
    try:
        if video_data is None or res_zero_masks is None:
            return
        if not isinstance(video_data, np.ndarray) or not isinstance(res_zero_masks, np.ndarray):
            return
        if video_data.ndim != 4 or video_data.shape[-1] != 3:
            return
        # Normalize mask to (F,H,W,1) uint8
        m = res_zero_masks
        if m.ndim == 3:
            m = m[..., None]
        if m.shape[0] != video_data.shape[0]:
            # Length mismatch; bail out
            return
        F = video_data.shape[0]
        frames = max(1, int(frames))
        idxs = list(range(min(frames, F)))
        os.makedirs(out_dir, exist_ok=True)
        for t in idxs:
            frame = video_data[t]  # (H,W,3) uint8 RGB
            mask  = m[t, ..., 0]   # (H,W) uint8 0/255
            # Apply: black out masked pixels (mask==255)
            vis = frame.copy()
            if mask.dtype != np.uint8:
                mask = mask.astype(np.uint8)
            # boolean mask
            mm = mask > 0
            vis[mm] = 0
            # Save as PNG (BGR for OpenCV)
            bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(out_dir, f"{tag}_f{t:03d}.png"), bgr)
    except Exception:
        # best-effort only; never crash data loader
        pass


rank = int(os.environ.get("RANK", "0"))
local_rank = int(os.environ.get("LOCAL_RANK", "0"))
world_size = int(os.environ.get("WORLD_SIZE", "1"))

# Logger for mask ratio logging (file only, no stdout)
def _get_mask_logger():
    logger = logging.getLogger("mask_ratio_logger")
    if getattr(logger, "_configured", False):
        return logger
    logger.propagate = False  # do not print to stdout
    logger.setLevel(logging.INFO)
    # decide log dir
    log_dir = os.environ.get("LLAVA_OUTPUT_DIR") or os.environ.get("OUTPUT_DIR") or os.path.join(os.getcwd(), "output")
    try:
        os.makedirs(log_dir, exist_ok=True)
    except Exception:
        log_dir = os.getcwd()
    fh = logging.FileHandler(os.path.join(log_dir, f"mask_ratio_rank{rank:02d}.log"))
    fmt = logging.Formatter(f"rank-id:{rank:03d}:%(asctime)s - %(message)s")
    fh.setFormatter(fmt)
    fh.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(fh)
    logger._configured = True
    return logger

class DALIWarper(object):
    def __init__(self, dali_iter, step_data_num, mode="train"):
        self.iter = dali_iter
        self.step_data_num = step_data_num
        assert(mode in ["train", "val", "test"])
        self.mode = mode
        # Mask logging state
        self._step = 0
        self._mask_logger = _get_mask_logger()
        self._mask_log_interval = int(os.environ.get("MASK_LOG_INTERVAL", "50"))
        self._mask_log_samples = int(os.environ.get("MASK_LOG_SAMPLES", "2"))
        # Shape logging/sanity check state
        self._log_shapes_once = False
        self._max_shape_logs = 3

    def __next__(self):
        logger = _get_dali_v1_logger()
        try:
            data = self.iter.__next__()  # fetch one batch from DALIGenericIterator
            self._saw_stop_once = False  # reset flag on success
        except StopIteration:
            if not getattr(self, "_saw_stop_once", False):
                try:
                    logger.warning("[StopIteration] DALI iterator reached end; resetting and retrying once.")
                except Exception:
                    pass
                self._saw_stop_once = True
                try:
                    self.iter.reset()
                    data = self.iter.__next__()
                except StopIteration:
                    try:
                        logger.error("[StopIteration] Iterator stopped again after reset; propagating.")
                    except Exception:
                        pass
                    raise
            else:
                try:
                    logger.error("[StopIteration] Repeated StopIteration; propagating.")
                except Exception:
                    pass
                raise
        except Exception:
            try:
                logger.error("[Exception] DALI iterator raised an exception:\n%s", traceback.format_exc())
            except Exception:
                pass
            raise

        # DALIGenericIterator returns a one-element list of dicts
        data_dict = data[0]
        # Ensure expected key name for training code
        if 'pixel_values' not in data_dict:
            if 'videos' in data_dict:
                data_dict['pixel_values'] = data_dict['videos']

        # ---- Sanity check shapes & clamp index lists to valid [0, F-1] ----
        try:
            pv = data_dict.get('pixel_values', None)
            if isinstance(pv, torch.Tensor) and pv.ndim >= 4:
                # Expected [B, C, F, H, W]
                B = int(pv.shape[0])
                C = int(pv.shape[1]) if pv.ndim >= 2 else -1
                # Heuristic: pick the smallest of spatial dims as H/W, assume the remaining middle is F
                if pv.ndim == 5:
                    Fdim = 2
                elif pv.ndim == 4:
                    # Some pipelines may output [B, F, H, W] after normalization (no channel); fall back
                    Fdim = 1
                else:
                    Fdim = 2
                F = int(pv.shape[Fdim])

                if not self._log_shapes_once and self._max_shape_logs > 0:
                    try:
                        _get_dali_v1_logger().info(
                            f"[batch_shapes] pixel_values={tuple(pv.shape)} I_lists={tuple(data_dict.get('I_lists', torch.tensor([])).shape)} "
                            f"P_lists={tuple(data_dict.get('P_lists', torch.tensor([])).shape)} res_zero_masks={tuple(data_dict.get('res_zero_masks', torch.tensor([])).shape)} labels={tuple(data_dict.get('labels', torch.tensor([])).shape)}"
                        )
                        self._max_shape_logs -= 1
                        if self._max_shape_logs <= 0:
                            self._log_shapes_once = True
                    except Exception:
                        pass

                for key in ('I_lists', 'P_lists'):
                    idx = data_dict.get(key, None)
                    if isinstance(idx, torch.Tensor):
                        # Ensure long dtype and 2D [B, L]
                        if idx.dtype != torch.long:
                            idx = idx.long()
                        if idx.ndim == 1 and B == 1:
                            idx = idx.unsqueeze(0)
                        elif idx.ndim > 2:
                            idx = idx.view(B, -1)
                        # Clamp to valid frame range
                        before_min = int(idx.min().item()) if idx.numel() else 0
                        before_max = int(idx.max().item()) if idx.numel() else -1
                        idx = torch.clamp(idx, 0, max(0, F - 1))
                        after_min = int(idx.min().item()) if idx.numel() else 0
                        after_max = int(idx.max().item()) if idx.numel() else -1
                        if before_min < 0 or before_max >= F:
                            try:
                                _get_dali_v1_logger().warning(
                                    f"[index_clamp] key={key} clamped from min/max=({before_min},{before_max}) to ({after_min},{after_max}) with F={F}"
                                )
                            except Exception:
                                pass
                        data_dict[key] = idx
        except Exception:
            # Never crash data path due to debugging utilities
            pass
        # ---- Mask ratio logging (whole sequence except the first frame) ----
        try:
            m = data_dict.get('res_zero_masks', None)
            if isinstance(m, torch.Tensor) and m.ndim in (4, 5):
                m_t = m.detach()
                if m_t.is_cuda:
                    m_t = m_t.cpu()
                m_f = m_t.float()
                # Expect [B,1,F,H,W] or [B,F,H,W]
                if m_f.ndim == 5:
                    m_f = m_f[:, 0]   # -> [B,F,H,W]
                if m_f.ndim == 4 and m_f.shape[1] >= 1:
                    m_ex_first = m_f[:, 1:]  # exclude the first frame
                    if m_ex_first.numel() > 0:
                        ratios = m_ex_first.mean(dim=(1, 2, 3))  # per-sample ratio
                        K = int(min(self._mask_log_samples, ratios.shape[0]))
                        vals = [f"{float(ratios[i].item()):.3f}" for i in range(K)]
                        self._mask_logger.info(f"[mask] ratios(exclude-first): {', '.join(vals)}")
        except Exception:
            pass

        return data_dict

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

        # Robust file_list expansion (directory / glob / list file / single video)
        # self._file_list_arg = self.file_list
        # entries = _expand_file_list(self._file_list_arg)
        # if not entries:
        #     _get_dali_v1_logger().error(f"[file_list] No examples resolved from '{self._file_list_arg}'.")
        #     raise ValueError(f"No examples resolved from '{self._file_list_arg}'")
        # self.file_list = entries
        # try:
        #     _get_dali_v1_logger().info(f"[file_list] resolved {len(self.file_list)} examples from '{self._file_list_arg}'")
        # except Exception:
        #     pass
        self.num_shards = source_params.get("num_shards", world_size) or world_size
        self.shard_id   = source_params.get("shard_id", rank) or rank
        self.label = source_params.get("label")

        # --- config from source_params ---
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

        self.res_block_size = source_params.get("res_block_size", 16)
        self.res_zero_thresh = source_params.get("res_zero_thresh", 0)
        self.enable_res_zero_mask = source_params.get("enable_res_zero_mask", True)
        self.res_min_drop_ratio = float(source_params.get("res_min_drop_ratio", 0.5))
        self.hevc_y_only = bool(source_params.get("hevc_y_only", True))

        # ---- visualization config (env-driven) ----
        self.viz_mask = int(os.environ.get("VIZ_MASK", "0")) != 0
        self.viz_mask_interval = int(os.environ.get("VIZ_MASK_INTERVAL", "200"))
        self.viz_mask_samples = int(os.environ.get("VIZ_MASK_SAMPLES", "2"))
        _vmf_raw = os.environ.get("VIZ_MASK_FRAMES", "1").strip().lower()
        if _vmf_raw in ("all", "0", "-1"):
            self.viz_mask_frames = -1
        else:
            try:
                self.viz_mask_frames = int(_vmf_raw)
            except Exception:
                self.viz_mask_frames = 1
        base_out = os.environ.get("LLAVA_OUTPUT_DIR") or os.environ.get("OUTPUT_DIR") or os.path.join(os.getcwd(), "output")
        self.viz_mask_dir = os.path.join(base_out, "viz_mask")
        try:
            os.makedirs(self.viz_mask_dir, exist_ok=True)
        except Exception:
            pass

        self.perm = None
        self.last_seen_epoch = None
        self.replace_example_info = self.file_list[0]
        self.replace_example_info = _maybe_swap_to_hevc(self.replace_example_info)

        self.mode = "train"

        import math
        # Shard with ceil so small uneven splits still produce data
        self.shard_size = int(math.ceil(len(self.file_list) / float(self.num_shards)))
        self.shard_offset = self.shard_size * int(self.shard_id)
        # Number of iterations for this shard (at least 1 if there are entries)
        self.full_iterations = max(1 if len(self.file_list) > 0 else 0,
                                   int(math.ceil(self.shard_size / float(self.batch_size))))
    def __getstate__(self):
        # Return only serializable fields; everything stored is JSON/pickle-friendly already.
        return dict(self.__dict__)

    def __setstate__(self, state):
        self.__dict__.update(state)


    def sparse_sampling_get_frameid_data(
            self,
            video_path,
            sequence_length,
            test_info):

        decord_vr = decord.VideoReader(video_path, num_threads=8, ctx=decord.cpu(0))
        duration = len(decord_vr)

        if self.mode in ["train"]:
            # 所有帧索引
            all_index = list(range(0, int(duration), 1))

            # === I/P 帧规则 ===
            i_index = [0]                     # I帧 = 第1帧（索引0）
            p_index = all_index[1:]           # P帧 = 剩下的帧

            # 拼接 I 和 P，保证顺序
            frame_id_list = i_index + p_index

            # 截断或补齐到 sequence_length
            if len(frame_id_list) >= sequence_length:
                frame_id_list = frame_id_list[:sequence_length]
            else:
                frame_id_list = frame_id_list + [frame_id_list[-1]] * (sequence_length - len(frame_id_list))

            # Map absolute frame id -> position in the sampled sequence
            frame_ids = frame_id_list
            pos_map = {fid: i for i, fid in enumerate(frame_ids)}

            # 记录 I_list / P_list 在序列中的位置
            I_list = [0]  # 第一帧一定是 I
            P_list = list(range(1, len(frame_id_list)))  # 剩下的都是 P

            # 读取视频帧
            decord_vr.seek(0)
            video_data = decord_vr.get_batch(frame_id_list).asnumpy()

            # 转成 numpy array
            I_list = np.array(I_list, dtype=np.int64)
            P_list = np.array(P_list, dtype=np.int64)

            res_zero_masks = None
            # import pdb; pdb.set_trace()
            if self.enable_res_zero_mask:
                try:
                    # Optionally force HEVC reader to output Y-only residuals (safer + faster for mask)
                    _prev_y_only = os.environ.get("UMT_HEVC_Y_ONLY", None)
                    try:
                        if self.hevc_y_only:
                            os.environ["UMT_HEVC_Y_ONLY"] = "1"
                        rdr = HevcFeatureReader(video_path, nb_frames=None, n_parallel=6)
                        masks_list = []
                        # import pdb; pdb.set_trace()
                        # Iterate through frames and collect only requested indices
                        wanted = set(frame_id_list)
                        cur_idx = 0
                        for (frame_tuple, meta) in rdr.nextFrameEx():
                            if cur_idx in wanted:
                                residual = frame_tuple[-1]
                                # Ensure Y-only residual for mask logic
                                if isinstance(residual, np.ndarray) and residual.ndim == 3:
                                    residual_y = residual[..., 0]
                                else:
                                    residual_y = residual  # already Y
                                seq_pos = pos_map.get(cur_idx, -1)
                                if seq_pos == 0:
                                    # 第一帧：永不丢弃，也不计入 min_drop_ratio 计算
                                    if isinstance(residual_y, np.ndarray) and residual_y.ndim >= 2:
                                        H, W = residual_y.shape[:2]
                                    else:
                                        H, W = video_data.shape[1], video_data.shape[2]
                                    mask_hw1 = np.zeros((H, W, 1), dtype=np.uint8)
                                else:
                                    mask_hw1 = _build_zero_res_block_mask(
                                        residual_y,
                                        block=self.res_block_size,
                                        thresh=self.res_zero_thresh,
                                        min_drop_ratio=self.res_min_drop_ratio
                                    )
                                    # print("drop_ratio", self.res_min_drop_ratio)
                                    # mask_hw1 = _build_zero_res_block_mask_topk(
                                    #     residual_y,
                                    #     block=self.res_block_size,
                                    #     drop_ratio=self.res_min_drop_ratio
                                    # )
                                masks_list.append((cur_idx, mask_hw1))
                                if len(masks_list) == len(frame_id_list):
                                    break
                            cur_idx += 1
                        rdr.close()
                    finally:
                        # Restore previous env to avoid side-effects
                        if _prev_y_only is None:
                            os.environ.pop("UMT_HEVC_Y_ONLY", None)
                        else:
                            os.environ["UMT_HEVC_Y_ONLY"] = _prev_y_only
                    # Order masks to match frame_id_list
                    masks_dict = {idx: m for (idx, m) in masks_list}
                    masks_ordered = [masks_dict.get(fid, np.zeros_like(masks_list[0][1]) if masks_list else np.zeros((video_data.shape[1], video_data.shape[2], 1), dtype=np.uint8)) for fid in frame_id_list]
                    res_zero_masks = np.stack(masks_ordered, axis=0)  # (F,H,W,1)
                except Exception as _e:
                    # On any failure, fall back to zeros of matching spatial size / sequence length
                    H, W = video_data.shape[1], video_data.shape[2]
                    res_zero_masks = np.zeros((len(frame_id_list), H, W, 1), dtype=np.uint8)
            else:
                H, W = video_data.shape[1], video_data.shape[2]
                res_zero_masks = np.zeros((len(frame_id_list), H, W, 1), dtype=np.uint8)
            # print("video_label", video_label)
            # print("video_label_type", type(video_label))
            return video_data, I_list, P_list, res_zero_masks


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

        total_n = len(self.file_list)
        if total_n == 0:
            raise StopIteration
        # Map to local index within this shard and wrap if needed
        local_idx = (sample_info.idx_in_epoch % max(1, self.shard_size))
        global_idx = self.shard_offset + local_idx
        if global_idx >= total_n:
            global_idx = global_idx % total_n
        sample_idx = self.perm[global_idx]

        example_info = self.file_list[sample_idx]
        video_label = self.label[sample_idx]
        video_path = example_info
        video_path = _maybe_swap_to_hevc(video_path)
        test_info = None
        
        # # Robust handling for tuple or not
        # if isinstance(example_info, (list, tuple)):
        #     video_path = example_info[0]
        #     video_label = example_info[1] if len(example_info) > 1 else 0
        # else:
        #     video_path = str(example_info)
        #     video_label = 0

        try:
            video_data, I_list, P_list, res_zero_masks = self.sparse_sampling_get_frameid_data(video_path, self.sequence_length, test_info)
        except Exception:
            try:
                _get_mask_logger().warning("read_fail: %s", str(video_path))
            except Exception:
                pass
            video_path = self.replace_example_info
            video_data, I_list, P_list, res_zero_masks = self.sparse_sampling_get_frameid_data(video_path, self.sequence_length, test_info)

        # # ---- optional visualization: save masked frames (pre-augmentation) ----
        # try:
        #     if (self.mode == "train" and self.viz_mask and
        #         (sample_info.iteration % max(1, self.viz_mask_interval) == 0) and
        #         (sample_info.idx_in_batch < max(1, self.viz_mask_samples)) and
        #         (res_zero_masks is not None)):
        #         # build a short tag for filenames
        #         _vname = os.path.basename(video_path).replace(os.sep, "_")
        #         _tag = f"rank{rank:02d}_it{sample_info.iteration:06d}_ib{sample_info.idx_in_batch:02d}_{_vname}"
        #         _frames_to_save = (video_data.shape[0] if self.viz_mask_frames <= 0 else self.viz_mask_frames)
        #         _viz_mask_apply_and_save(video_data, res_zero_masks, self.viz_mask_dir, _tag, frames=_frames_to_save)
        #         # also log a line into the mask logger (file-only)
        #         try:
        #             _get_mask_logger().info("[viz] it=%d ib=%d saved=%d dir=%s tag=%s",
        #                                      int(sample_info.iteration),
        #                                      int(sample_info.idx_in_batch),
        #                                      int(min(self.viz_mask_frames, video_data.shape[0] if hasattr(video_data, 'shape') else 1)),
        #                                      self.viz_mask_dir, _tag)
        #         except Exception:
        #             pass
        except Exception:
            pass

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
            return video_data, I_list, P_list, res_zero_masks, video_label


def _count_file_list_entries(file_list):
    if isinstance(file_list, (list, tuple)):
        return len(file_list)
    if isinstance(file_list, str):
        try:
            exs = _expand_file_list(file_list)
            return max(1, len(exs))
        except Exception:
            return 1
    return 1


@pipeline_def()
def dali_pipeline(mode, source_params):

    parallel_flag = (os.environ.get("DALI_PARALLEL_ES", "1") != "0")
    if mode == "train":
        videos, I_lists, P_lists, res_zero_masks, labels = fn.external_source(
            source      = ExternalInputCallable(source_params),
            num_outputs = 5,
            batch       = False,
            parallel    = parallel_flag,
            dtype       = [types.UINT8, types.INT64, types.INT64, types.UINT8, types.INT64],
            layout      = ["FHWC", "C", "C", "FHWC", "C"],
        )
        
        videos = videos.gpu()
        res_zero_masks = res_zero_masks.gpu()
        # Shared flip decision for videos & masks (so they stay aligned)
        mirror = None
        # teacher_videos = fn.identity(videos)
        # videos = rand_augment.rand_augment(videos, n=4, m=7, fill_value=128, monotonic_mag=True)
        videos_student = fn.random_resized_crop(
            videos,
            random_area         = (0.50, 1.0),
            random_aspect_ratio = (0.75, 1.3333),
            size                = [source_params['input_size'], source_params['input_size']],
            num_attempts        = 10,
            antialias           = True,
            interp_type         = types.INTERP_LINEAR
        )
        # teacher_videos = fn.random_resized_crop(
        #     videos,
        #     random_area         = (0.50, 1.0),
        #     random_aspect_ratio = (0.75, 1.3333),
        #     size                = [source_params['input_size'] * 2, source_params['input_size'] * 2],
        #     num_attempts        = 10,
        #     antialias           = True,
        #     interp_type         = types.INTERP_LINEAR
        # )
        if source_params['reprob'] > 0:
            erase_probability = fn.random.coin_flip(dtype=types.BOOL, probability=source_params['reprob'])
            if erase_probability:
                mask = videos * 0
                # anchor=(y0, x0, y1, x1, …);shape=(h0, w0, h1, w1, …)
                mask = fn.erase(
                    mask,
                    device     = "gpu",
                    axis_names = "HW",
                    fill_value = 255,
                    anchor     = fn.random.uniform(range=(0, source_params['input_size']), shape=(2,)),
                    shape      = fn.random.uniform(range=(20, 90), shape=(2,))
                )
                noise = fn.random.normal(videos, device="gpu", dtype=types.INT8)
                videos_student = (videos_student & (255 - mask)) | (noise & mask)

                # teacher_mask = teacher_videos * 0
                # teacher_mask = fn.erase(
                #     teacher_mask,
                #     device     = "gpu",
                #     axis_names = "HW",
                #     fill_value = 255,
                #     anchor     = fn.random.uniform(range=(0, source_params['input_size'] * 2), shape=(2,)),
                #     shape      = fn.random.uniform(range=(20, 90), shape=(2,))
                # )
                # teacher_noise = fn.random.normal(teacher_videos, device="gpu", dtype=types.INT8)
                # teacher_videos = (teacher_videos & (255 - teacher_mask)) | (teacher_noise & teacher_mask)

            else:
                # align dali-types
                mask = videos_student * 0
                videos_student = (videos_student & (255 + mask))
                # teacher_mask = teacher_videos * 0
                # teacher_videos = (teacher_videos & (255 + teacher_mask))
                
        if source_params['use_flip']:
            mirror = fn.random.coin_flip(probability=0.5)
            videos_student = fn.flip(
                videos_student,
                device     = "gpu",
                horizontal = mirror
            )

            # teacher_videos = fn.flip(
            #     teacher_videos,
            #     device     = "gpu",
            #     horizontal = fn.random.coin_flip(probability=0.5)
            # )

        videos_student = fn.crop_mirror_normalize(
            videos_student,
            device        = "gpu",
            dtype         = types.FLOAT,
            output_layout = "CFHW",
            mean          = source_params['mean'],
            std           = source_params['std']
        )
        # teacher_videos = fn.crop_mirror_normalize(
        #     teacher_videos,
        #     device        = "gpu",
        #     dtype         = types.FLOAT,
        #     output_layout = "CFHW",
        #     mean          = source_params['mean'],
        #     std           = source_params['std']
        # )

        # Make mask follow the same spatial transforms as videos (flip + resize) before layout change
        if source_params['use_flip'] and mirror is not None:
            res_zero_masks = fn.flip(res_zero_masks, device="gpu", horizontal=mirror)
        res_zero_masks = fn.resize(
            res_zero_masks,
            size=[source_params['input_size'], source_params['input_size']],
            interp_type=types.INTERP_NN
        )
        res_zero_masks = fn.transpose(res_zero_masks, perm=[3, 0, 1, 2])  # FHWC -> CFHW
        res_zero_masks = fn.cast(res_zero_masks, dtype=types.FLOAT) / 255.0

        labels = labels.gpu()
        return videos_student, I_lists, P_lists, res_zero_masks, labels
    


def dali_dataloader(
    file_list,
    label,
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
        "label":                label,
        
        # Size parameters
        "input_size":           input_size,
        "short_side_size":      short_side_size,
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

        "enable_res_zero_mask":  True,
        "res_block_size":        16,
        "res_min_drop_ratio":    float(os.environ.get("RES_MIN_DROP_RATIO", "0.0")),
        "hevc_y_only":           True,

    }
    # Provide output dir to dataloader logger if available from caller
    if "OUTPUT_DIR" not in os.environ:
        # Best-effort: if caller has a conventional output/ directory, use it
        _default_out = os.path.join(os.getcwd(), "output")
        try:
            os.makedirs(_default_out, exist_ok=True)
        except Exception:
            pass
        os.environ.setdefault("OUTPUT_DIR", _default_out)
        
    pipe = dali_pipeline(
        batch_size           = batch_size,
        num_threads          = dali_num_threads,
        device_id            = local_rank,
        seed                 = seed + rank,
        py_num_workers       = dali_py_num_workers,
        py_start_method      = os.environ.get('DALI_PY_START_METHOD', 'spawn'),
        prefetch_queue_depth = 2,
        mode                 = mode,
        source_params        = source_params
    )
    pipe.build()

    # Define output mapping based on mode
    output_map = ['pixel_values', 'I_lists', 'P_lists', 'res_zero_masks', 'labels']
    if mode == "test":
        output_map.extend(['chunk_nb', 'split_nb', 'sample_idx'])

    dali_iter = DALIGenericIterator(
        pipelines           = pipe,
        output_map          = output_map,
        auto_reset          = True,
        size                = -1,
        last_batch_padded   = False,
        last_batch_policy   = LastBatchPolicy.FILL,
        prepare_first_batch = False,
    )
    _total_entries = _count_file_list_entries(file_list)
    _num_shards_eff = num_shards or world_size
    _steps = max(1, (_total_entries // _num_shards_eff) // batch_size)
    dataloader = DALIWarper(
        dali_iter     = dali_iter,
        step_data_num = _steps,
        mode          = mode
    )
    return dataloader
