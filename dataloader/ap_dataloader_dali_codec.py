import os
import logging
import traceback
import math
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

# ------- MV global motion model cache (per-process) -------
_MV_GLOBAL_DESIGN_CACHE = {}


def _mv_get_design_matrix(Hm: int, Wm: int, mode: str):
    """Return cached (x_flat, y_flat, A_full) for the MV grid.

    Coordinates are normalized to [-1, 1] to improve conditioning.

    similarity params [tx, ty, a, b]:
      dx = tx + a*x - b*y
      dy = ty + b*x + a*y

    affine params [a0, a1, a2, b0, b1, b2]:
      dx = a0 + a1*x + a2*y
      dy = b0 + b1*x + b2*y
    """
    key = (int(Hm), int(Wm), str(mode).lower())
    if key in _MV_GLOBAL_DESIGN_CACHE:
        return _MV_GLOBAL_DESIGN_CACHE[key]

    Hm = int(Hm)
    Wm = int(Wm)
    ys = np.linspace(-1.0, 1.0, Hm, dtype=np.float32)
    xs = np.linspace(-1.0, 1.0, Wm, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    x = xx.reshape(-1).astype(np.float32)
    y = yy.reshape(-1).astype(np.float32)

    m = str(mode).lower()
    if m == "similarity":
        # 2N x 4
        N = x.shape[0]
        A = np.zeros((2 * N, 4), dtype=np.float32)
        # dx rows
        A[0::2, 0] = 1.0
        A[0::2, 2] = x
        A[0::2, 3] = -y
        # dy rows
        A[1::2, 1] = 1.0
        A[1::2, 2] = y
        A[1::2, 3] = x
    elif m == "affine":
        # 2N x 6
        N = x.shape[0]
        A = np.zeros((2 * N, 6), dtype=np.float32)
        # dx rows
        A[0::2, 0] = 1.0
        A[0::2, 1] = x
        A[0::2, 2] = y
        # dy rows
        A[1::2, 3] = 1.0
        A[1::2, 4] = x
        A[1::2, 5] = y
    else:
        raise ValueError(f"Unknown design mode: {mode}")

    _MV_GLOBAL_DESIGN_CACHE[key] = (x, y, A)
    return x, y, A


def _mv_predict_from_params(x_flat: np.ndarray, y_flat: np.ndarray, params: np.ndarray, mode: str):
    """Predict dx, dy for all grid points."""
    m = str(mode).lower()
    if m == "similarity":
        tx, ty, a, b = [float(v) for v in params.reshape(-1)[:4]]
        dx = tx + a * x_flat - b * y_flat
        dy = ty + b * x_flat + a * y_flat
        return dx.astype(np.float32), dy.astype(np.float32)
    elif m == "affine":
        a0, a1, a2, b0, b1, b2 = [float(v) for v in params.reshape(-1)[:6]]
        dx = a0 + a1 * x_flat + a2 * y_flat
        dy = b0 + b1 * x_flat + b2 * y_flat
        return dx.astype(np.float32), dy.astype(np.float32)
    else:
        raise ValueError(f"Unknown predict mode: {mode}")


def _mv_fit_global_model(vx: np.ndarray, vy: np.ndarray, mode: str,
                         max_points: int = 5000,
                         trim_ratio: float = 0.7,
                         iters: int = 2):
    """Robustly fit a global camera-motion model on MV grid and return predicted (vx_cam, vy_cam).

    Uses iterative trimming: fit -> residual -> keep smallest trim_ratio -> refit.
    Falls back to median translation if not enough valid points.
    """
    Hm, Wm = vx.shape
    x_flat, y_flat, A_full = _mv_get_design_matrix(Hm, Wm, mode=mode)

    vxf = vx.reshape(-1).astype(np.float32)
    vyf = vy.reshape(-1).astype(np.float32)

    mag1 = np.abs(vxf) + np.abs(vyf)
    valid = np.isfinite(mag1) & (mag1 > 1e-3)
    idx = np.nonzero(valid)[0]

    if idx.size < 64:
        vx_cam = np.full_like(vx, float(np.median(vx)), dtype=np.float32)
        vy_cam = np.full_like(vy, float(np.median(vy)), dtype=np.float32)
        return vx_cam, vy_cam

    # deterministic subsampling to cap cost
    if idx.size > int(max_points):
        step = int(math.ceil(idx.size / float(max_points)))
        idx = idx[::step]

    def build_system(sel_idx: np.ndarray):
        rows0 = 2 * sel_idx
        rows1 = 2 * sel_idx + 1
        rows = np.concatenate([rows0, rows1], axis=0)
        A = A_full[rows]
        b = np.concatenate([vxf[sel_idx], vyf[sel_idx]], axis=0).astype(np.float32)
        return A, b

    sel = idx
    params = None
    for _ in range(max(1, int(iters))):
        A, b = build_system(sel)
        try:
            params = np.linalg.lstsq(A, b, rcond=None)[0].astype(np.float32)
        except Exception:
            params = None
            break

        dx_all, dy_all = _mv_predict_from_params(x_flat, y_flat, params, mode=mode)
        rx = vxf[sel] - dx_all[sel]
        ry = vyf[sel] - dy_all[sel]
        r = np.sqrt(rx * rx + ry * ry)

        if sel.size < 128:
            break

        keep_n = int(max(64, round(sel.size * float(trim_ratio))))
        if keep_n >= sel.size:
            break
        order = np.argsort(r)
        sel = sel[order[:keep_n]]

    if params is None:
        vx_cam = np.full_like(vx, float(np.median(vx)), dtype=np.float32)
        vy_cam = np.full_like(vy, float(np.median(vy)), dtype=np.float32)
        return vx_cam, vy_cam

    dx_all, dy_all = _mv_predict_from_params(x_flat, y_flat, params, mode=mode)
    vx_cam = dx_all.reshape(Hm, Wm).astype(np.float32)
    vy_cam = dy_all.reshape(Hm, Wm).astype(np.float32)
    return vx_cam, vy_cam

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

# ---- Center prior helper ----
def _apply_center_prior(fused: np.ndarray, center_prior: float = 0.0, center_sigma: float = 0.35) -> np.ndarray:
    """Apply a center-Gaussian prior to fused map. fused is HxW float32 in [0,1]."""
    cp = float(center_prior)
    if cp <= 0.0:
        return fused

    H, W = int(fused.shape[0]), int(fused.shape[1])
    if H <= 1 or W <= 1:
        return fused

    sigma = float(center_sigma) * float(min(H, W))
    sigma = max(sigma, 1.0)

    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    cy = (H - 1) * 0.5
    cx = (W - 1) * 0.5
    d2 = (xx - cx) ** 2 + (yy - cy) ** 2
    w = np.exp(-0.5 * d2 / (sigma * sigma)).astype(np.float32)
    m = float(w.max())
    if m > 0:
        w = w / m

    out = fused.astype(np.float32) * (1.0 + cp * w)
    return np.clip(out, 0.0, 1.0).astype(np.float32)

def _residual_energy_norm(res_y: np.ndarray, pct: float = 95.0, use_grad: bool = False):
    """Return (norm_HxW_float32_in_[0,1], scale_max_level). No gamma/colormap."""
    if not use_grad:
        x = np.abs(res_y.astype(np.float32) - 128.0)
    else:
        # Edge/structure energy: less biased to texture/noise than pure magnitude
        r = res_y.astype(np.float32)
        gx = np.abs(np.roll(r, -1, axis=1) - np.roll(r, 1, axis=1))
        gy = np.abs(np.roll(r, -1, axis=0) - np.roll(r, 1, axis=0))
        x = 0.5 * (gx + gy)

    a = float(np.percentile(x, pct))
    a = max(a, 1.0)
    norm = np.clip(x / a, 0.0, 1.0)
    return norm.astype(np.float32), a
import cv2
import numpy as np

def resize_and_center_crop_residuals(residuals_y, input_size):
    """
    Resize and center crop residuals.
    
    Args:
        residuals_y: numpy array, shape (F, H, W) or (F, H, W, 1)
    
    Returns:
        res_zero_masks: shape (F, input_size, input_size, 1), dtype=uint8
    """
    # If shape is (F, H, W, 1), remove last dimension
    if residuals_y.ndim == 4 and residuals_y.shape[-1] == 1:
        residuals_y = residuals_y[..., 0]  # -> (F, H, W)

    F, H, W = residuals_y.shape

    # Following DALI logic: resize_shorter=input_size -> scale by short side
    scale = input_size / min(H, W)
    new_w = int(round(W * scale))
    new_h = int(round(H * scale))

    # Center crop coordinates (all frames have same resolution, calculate once)
    x1 = (new_w - input_size) // 2
    y1 = (new_h - input_size) // 2
    x2 = x1 + input_size
    y2 = y1 + input_size

    # Pre-allocate output: (F, input_size, input_size, 1) corresponding to DALI's "FHWC"
    res_zero_masks = np.empty((F, input_size, input_size, 1), dtype=np.uint8)

    for i in range(F):
        frame = residuals_y[i]  # (H, W)

        # INTER_CUBIC corresponds to DALI's INTERP_CUBIC
        resized_long = cv2.resize(
            frame,
            (new_w, new_h),
            interpolation=cv2.INTER_CUBIC
        )  # (new_h, new_w)

        cropped = resized_long[y1:y2, x1:x2]  # (input_size, input_size)

        # DALI outputs UINT8, convert to uint8 here; if residual is float, customize normalization/threshold
        res_zero_masks[i, :, :, 0] = cropped.astype(np.uint8)

    return res_zero_masks

def compute_visible_indices_cpu(
    residuals_y: np.ndarray,
    patch_size: int | tuple[int, int],
    K: int,
    static_fallback: bool = True,
    static_abs_thresh: float = 2.0,
    static_rel_thresh: float = 0.15,
    static_uniform_frames: int = 4,
) -> np.ndarray:
    """
    CPU version of Top-K visible patch selection logic, corresponding to PyTorch mask_by_residual_topk single sample case (B=1),
    returns only visible_indices (not mask and ids_restore).

    Args:
        residuals_y: np.ndarray
            Shape (F, H, W, 1) or (F, H, W), representing residuals for a single sample.
            Note: It's recommended to process residuals_y into "signed" residuals before calling,
            i.e., consistent with res.abs() semantics fed to mask_by_residual_topk during training.
            If residuals_y is currently uint8 (0~255), you can do this externally first:
                residuals_y = residuals_y.astype(np.int16) - 128
        patch_size: int or (ph, pw)
            Patch height and width.
        input_size: int
            Target input size H=W=input_size.
            If residuals_y's H,W are already input_size, you may not need it;
            This parameter is mainly for compatibility with original function signature.
        sequence_length: int
            Sequence length F (frame count), mainly for interface compatibility; actually uses residuals_y dimension 0.
        K: int
            Number of Top-K patches to keep (k_keep).

    Returns:
        visible_indices: np.ndarray, shape (K',), dtype=int32
            Selected patch linear indices (ascending), K' = clamp(K, 0, L), L is total patch count.
    """
    # ---------- 1. Unify residuals_y shape ----------
    # Support (F, H, W, 1) or (F, H, W)
    if residuals_y.ndim == 4 and residuals_y.shape[-1] == 1:
        residuals_y = residuals_y.squeeze(-1)  # (F, H, W)

    if residuals_y.ndim != 3:
        raise ValueError(f"residuals_y must be (F,H,W) or (F,H,W,1), current shape: {residuals_y.shape}")

    F, H, W = residuals_y.shape  # Actual F,H,W based on data
    # sequence_length and input_size are mainly for interface compatibility, add checks if strict constraint needed:
    
    # ---------- 2. Patch grid division ----------
    if isinstance(patch_size, int):
        ph = pw = patch_size
    else:
        ph, pw = patch_size

    if H % ph != 0 or W % pw != 0:
        raise ValueError(
            f"H/W must be divisible by patch size, current H={H}, W={W}, ph={ph}, pw={pw}"
        )

    hb, wb = H // ph, W // pw  # Patch grid count per frame
    L = F * hb * wb            # Total patch count

    # ---------- 3. K boundary handling (consistent with PyTorch version) ----------
    K_clamped = int(max(0, min(K, L)))
    if K_clamped == 0:
        return np.empty((0,), dtype=np.int32)

    # ---------- 4. Calculate residual score for each patch ----------
    # PyTorch version: res_abs = res.abs().squeeze(1);
    #               scores = res_abs.reshape(B,T,hb,ph,wb,pw).sum(dim=(3,5))
    # Here is single sample (B=1), and residuals_y is already absolute value or signed residual:
    res_abs = np.abs(residuals_y)  # (F,H,W)

    # Reshape to (F, hb, ph, wb, pw), sum over patch interior -> (F, hb, wb)
    res_reshaped = res_abs.reshape(F, hb, ph, wb, pw)
    patch_scores = res_reshaped.sum(axis=(2, 4))      # (F, hb, wb)

    # Flatten to 1D (L,) - corresponding to PyTorch version scores.reshape(B,L) with B=1
    patch_scores_flat = patch_scores.reshape(L)       # (L,)

    # ---------- 4.5 Static/low-energy video fallback (Hybrid): uniform few frames + remaining Top-K ----------
    # patch_scores_flat is patch interior energy sum (proportional to ph*pw).
    # Use patch interior average intensity for static detection: patch_mean ~ [0,255].
    if bool(static_fallback):
        area = float(ph * pw)
        if area <= 0:
            area = 1.0
        patch_mean = patch_scores_flat.astype(np.float32) / area

        # Adaptive detection: absolute low energy + relative low contrast (flat distribution)
        p95 = float(np.percentile(patch_mean, 95.0))
        p50 = float(np.percentile(patch_mean, 50.0))
        rel_contrast = float((p95 - p50) / max(p95, 1e-6))

        is_static = (p95 < float(static_abs_thresh)) or (
            (p95 < float(static_abs_thresh) * 2.0) and (rel_contrast < float(static_rel_thresh))
        )

        if is_static:
            patches_per_frame = hb * wb

            # 1) Uniformly select a few frames (default 4 frames), deduplicate and fill
            f_u = int(static_uniform_frames) if int(static_uniform_frames) > 0 else 1
            f_u = max(1, min(F, f_u))
            if f_u >= F:
                t_list = list(range(F))
            else:
                t_list = np.linspace(0, F - 1, f_u, dtype=int).tolist()
                t_list = list(dict.fromkeys(t_list))
                if len(t_list) < f_u:
                    need = f_u - len(t_list)
                    for t in range(F):
                        if t not in t_list:
                            t_list.append(t)
                            need -= 1
                            if need <= 0:
                                break

            # 2) First try to cover as many patches as possible on these frames (if budget sufficient take all; otherwise uniformly subsample by budget)
            uniform_idx = []
            max_uniform = len(t_list) * patches_per_frame

            if K_clamped >= max_uniform:
                for t in t_list:
                    base = t * patches_per_frame
                    uniform_idx.extend(range(base, base + patches_per_frame))
            else:
                # Insufficient budget: distribute K evenly to these frames, uniformly sample spatially per frame
                k_base = K_clamped // len(t_list)
                k_rem = K_clamped % len(t_list)
                for j, t in enumerate(t_list):
                    k_t = k_base + (1 if j < k_rem else 0)
                    if k_t <= 0:
                        continue
                    if k_t >= patches_per_frame:
                        pos_list = list(range(patches_per_frame))
                    else:
                        step = float(patches_per_frame) / float(k_t)
                        pos_list = [min(patches_per_frame - 1, int(round(i * step))) for i in range(k_t)]
                        pos_list = list(dict.fromkeys(pos_list))
                        if len(pos_list) < k_t:
                            need = k_t - len(pos_list)
                            for p in range(patches_per_frame):
                                if p not in pos_list:
                                    pos_list.append(p)
                                    need -= 1
                                    if need <= 0:
                                        break
                    base = t * patches_per_frame
                    uniform_idx.extend([base + p for p in pos_list])

            uniform_idx = np.asarray(uniform_idx, dtype=np.int32)
            if uniform_idx.size > K_clamped:
                uniform_idx = uniform_idx[:K_clamped]

            # 3) Remaining tokens continue Top-K (excluding uniform_idx)
            k_remain = int(K_clamped - uniform_idx.size)
            if k_remain <= 0:
                return np.sort(uniform_idx).astype(np.int32)

            scores = patch_scores_flat.astype(np.float32).copy()
            scores[uniform_idx] = -np.inf

            remain_candidates = int(np.isfinite(scores).sum())
            if k_remain >= remain_candidates:
                topk_rem = np.where(np.isfinite(scores))[0].astype(np.int32)
            else:
                topk_rem = np.argpartition(scores, -k_remain)[-k_remain:].astype(np.int32)

            selected = np.unique(np.concatenate([uniform_idx, topk_rem], axis=0))
            if selected.size > K_clamped:
                selected = selected[:K_clamped]
            elif selected.size < K_clamped:
                all_idx = np.arange(L, dtype=np.int32)
                mask = np.ones(L, dtype=bool)
                mask[selected] = False
                extra = all_idx[mask][: (K_clamped - selected.size)]
                selected = np.concatenate([selected, extra], axis=0)

            return np.sort(selected).astype(np.int32)

    # ---------- 5. Select Top-K indices (corresponding to PyTorch version logic) ----------
    topk_indices = np.argpartition(patch_scores_flat, -K_clamped)[-K_clamped:]
    visible_indices = np.sort(topk_indices).astype(np.int32)  # (K_clamped,)

    return visible_indices


# ---- helper: fast box filter / local variance (numpy only) ----

def _box_filter2d(x: np.ndarray, k: int = 3) -> np.ndarray:
    """Fast kxk box filter using integral image (numpy only)."""
    k = int(k)
    if k <= 1:
        return x.astype(np.float32)
    pad = k // 2
    xp = np.pad(x.astype(np.float32), ((pad, pad), (pad, pad)), mode="edge")
    ii = np.cumsum(np.cumsum(xp, axis=0), axis=1)
    s = ii[k:, k:] - ii[:-k, k:] - ii[k:, :-k] + ii[:-k, :-k]
    return s / float(k * k)


def _local_var2d(x: np.ndarray, k: int = 3) -> np.ndarray:
    """Local variance in a kxk window."""
    x = x.astype(np.float32)
    m1 = _box_filter2d(x, k)
    m2 = _box_filter2d(x * x, k)
    v = m2 - m1 * m1
    return np.maximum(v, 0.0).astype(np.float32)

def _get_cache_path(video_path: str, cache_dir: str, cache_key: str = "") -> str:
    """
    Generate a cache file path for a given video path.
    
    Args:
        video_path: Path to the video file
        cache_dir: Directory to store cache files
        cache_key: Additional key for cache uniqueness
    Returns:
        Path to the cache pkl file
    """
    if not cache_dir:
        return None
    
    # Create a unique filename based on the video path and cache_key
    base = video_path if not cache_key else (video_path + "|" + str(cache_key))
    video_hash = hashlib.md5(base.encode()).hexdigest()
    cache_filename = f"visible_indices_{video_hash}.pkl"
    cache_path = os.path.join(cache_dir, cache_filename)
    
    return cache_path

def save_cache(visible_indices: np.ndarray, video_path: str, cache_dir: str, cache_key: str = "") -> None:
    """
    Save visible_indices to a pkl file.
    
    Args:
        visible_indices: numpy array of visible indices to save
        video_path: Path to the video file (used to generate cache filename)
        cache_dir: Directory to store cache files
        cache_key: Additional key for cache uniqueness
    """
    if not cache_dir:
        return
    
    cache_path = _get_cache_path(video_path, cache_dir, cache_key=cache_key)
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

def get_cache(video_path: str, cache_dir: str, cache_key: str = "") -> np.ndarray:
    """
    Load visible_indices from a pkl file.
    
    Args:
        video_path: Path to the video file (used to generate cache filename)
        cache_dir: Directory where cache files are stored
        cache_key: Additional key for cache uniqueness
    Returns:
        numpy array of visible indices, or None if cache doesn't exist
    """
    if not cache_dir:
        return None
    
    cache_path = _get_cache_path(video_path, cache_dir, cache_key=cache_key)
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
    compensate: str = "none",
    use_inconsistency: bool = False,
    incon_ksize: int = 3,
):
    """Return (norm_HxW_float32_in_[0,1], scale_max_px). No gamma/colormap.

    compensate:
      - 'none'       : no compensation
      - 'median'     : subtract global median translation per frame (helps remove camera pan/track motion)
      - 'mean'       : subtract global mean translation per frame
      - 'similarity' : fit a global similarity camera-motion field (translation + rotation + isotropic scale)
      - 'affine'     : fit a global affine camera-motion field (adds anisotropic scale + shear)
    """
    vx = mvx.astype(np.float32) / float(mv_unit_div)
    vy = mvy.astype(np.float32) / float(mv_unit_div)

    c = (compensate or "none").lower()
    if c == "median":
        vx = vx - np.median(vx)
        vy = vy - np.median(vy)
    elif c == "mean":
        vx = vx - float(np.mean(vx))
        vy = vy - float(np.mean(vy))
    elif c == "none":
        pass
    elif c in ("similarity", "affine"):
        # Fit a global camera-motion model on the MV grid (handles zoom/rotation better than pure translation)
        try:
            vx_cam, vy_cam = _mv_fit_global_model(vx, vy, mode=c)
            vx = vx - vx_cam
            vy = vy - vy_cam
        except Exception:
            # fallback to median translation
            vx = vx - np.median(vx)
            vy = vy - np.median(vy)
    else:
        raise ValueError(f"Unknown mv compensate mode: {compensate}")

    mag = np.sqrt(vx * vx + vy * vy)  # pixels

    if use_inconsistency:
        k = int(incon_ksize)
        k = 3 if k < 3 else k
        if (k % 2) == 0:
            k += 1
        score = _local_var2d(mag, k=k)
    else:
        score = mag.astype(np.float32)

    a = float(np.percentile(score, pct))
    a = max(a, 1e-6)
    norm = np.clip(score / a, 0.0, 1.0)
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
        self.mv_compensate = source_params.get('mv_compensate', 'none')
        self.mv_use_inconsistency = bool(source_params.get('mv_use_inconsistency', False))
        self.mv_incon_ksize = int(source_params.get('mv_incon_ksize', 3))
        self.res_use_grad = bool(source_params.get('res_use_grad', False))

        # optional: duplicate masking & center prior (online residual/mv energy)
        self.mask_all_duplicates = bool(source_params.get('mask_all_duplicates', False))
        self.center_prior = float(source_params.get('center_prior', 0.0) or 0.0)
        self.center_sigma = float(source_params.get('center_sigma', 0.35) or 0.35)
        # optional: static-video hybrid fallback (uniform few frames + remaining topk)
        self.static_fallback = bool(source_params.get('static_fallback', True))
        self.static_abs_thresh = float(source_params.get('static_abs_thresh', 2.0) or 2.0)
        self.static_rel_thresh = float(source_params.get('static_rel_thresh', 0.15) or 0.15)
        self.static_uniform_frames = int(source_params.get('static_uniform_frames', 4) or 4)

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
                          mv_pct: float = 95.0,       # MV normalization percentile (passed to _mv_energy_norm)
                          res_pct: float = 95.0,      # Residual normalization percentile (passed to _residual_energy_norm)
                          fuse_mode: str = "weighted",
                          w_mv: float = 1.0,
                          w_res: float = 1.0,
                          mv_compensate: str = "none",
                          mv_use_inconsistency: bool = False,
                          mv_incon_ksize: int = 3,
                          res_use_grad: bool = False):

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
                    # key_idx may be NDArray; convert to Python list of integer frame numbers, keep only one I-frame
                    I_list = np.asarray(key_idx)
                    I_list = I_list.tolist()[0] if I_list.ndim > 1 else I_list.tolist()
                    I_list = [int(i) for i in I_list if int(i) in frame_id_list]
                    if len(I_list) >= self.tokeq_target_frames:
                        # If too many I frames, prioritize keeping the first ones
                        I_list = I_list[:self.tokeq_target_frames]
                        P_list = []
                    else:
                        P_list = [i for i in range(len(frame_id_list)) if i not in I_list]
            except Exception:
                # Fallback handling: ignore exception, use default strategy
                print("Failed to read key indices")
                # First frame is I-frame
                I_list = [0]
                # Rest are P frames
                P_list = [i for i in range(len(frame_id_list)) if i not in I_list]
                # Map absolute frame id -> position in the sampled sequence
                frame_ids = frame_id_list
                pos_map = {fid: i for i, fid in enumerate(frame_ids)}
            
            frame_ids = frame_id_list
            pos_map = {fid: i for i, fid in enumerate(frame_ids)}
            
            # Read video frames
            decord_vr.seek(0)
            video_data = decord_vr.get_batch(frame_id_list).asnumpy()

            # Convert to numpy array
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

                                # I frame: set to 0 directly (consistent with residual logic)
                                if pos in I_pos_set:
                                    if H0 is None:
                                        # Use residual Y to determine output dimensions/type
                                        y0 = residual if residual.ndim == 2 else cv2.cvtColor(residual, cv2.COLOR_BGR2YUV)[:, :, 0]
                                        y0 = np.asarray(y0)
                                        H0, W0, dtype0 = int(y0.shape[0]), int(y0.shape[1]), y0.dtype
                                    residuals_y[pos] = np.zeros((H0, W0), dtype=dtype0 or np.uint8)

                                else:
                                    # 1) Extract MV (L0) and upsample to H×W
                                    mvx_hw = rdr._upsample_mv_to_hw(mv_x_L0.astype(np.float32))
                                    mvy_hw = rdr._upsample_mv_to_hw(mv_y_L0.astype(np.float32))

                                    # 2) Extract residual Y
                                    Y_res = residual if residual.ndim == 2 else cv2.cvtColor(residual, cv2.COLOR_BGR2YUV)[:, :, 0]

                                    # Initialize output dimensions/type (only on first hit)
                                    if H0 is None:
                                        H0, W0, dtype0 = int(Y_res.shape[0]), int(Y_res.shape[1]), Y_res.dtype

                                    # If current frame dimensions don't match H0×W0, resize to align (very rare, fallback)
                                    if (Y_res.shape[0] != H0) or (Y_res.shape[1] != W0):
                                        Y_res = cv2.resize(Y_res, (W0, H0), interpolation=cv2.INTER_AREA)
                                    if (mvx_hw.shape[0] != H0) or (mvx_hw.shape[1] != W0):
                                        mvx_hw = cv2.resize(mvx_hw, (W0, H0), interpolation=cv2.INTER_NEAREST)
                                        mvy_hw = cv2.resize(mvy_hw, (W0, H0), interpolation=cv2.INTER_NEAREST)

                                    # 3) Normalize to [0,1]
                                    mv_norm, _ = _mv_energy_norm(
                                        mvx_hw, mvy_hw, H0, W0,
                                        mv_unit_div=mv_unit_div,
                                        pct=mv_pct,
                                        compensate=mv_compensate,
                                        use_inconsistency=bool(mv_use_inconsistency),
                                        incon_ksize=int(mv_incon_ksize),
                                    )
                                    res_norm, _ = _residual_energy_norm(Y_res, pct=res_pct, use_grad=bool(res_use_grad))

                                    # 4) Fuse + center prior
                                    fused = _fuse_energy(mv_norm, res_norm, mode=fuse_mode, w_mv=w_mv, w_res=w_res)
                                    fused = _apply_center_prior(fused, center_prior=self.center_prior, center_sigma=self.center_sigma)

                                    # Write back to container (keep uint8 storage)
                                    residuals_y[pos] = (np.clip(fused, 0.0, 1.0) * 255.0).astype(dtype0 or np.uint8)

                                # End condition
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
                        print("[warn] Residual dimensions don't match video: res=(%d,%d) video=(%d,%d)" % (H0, W0, video_data.shape[1], video_data.shape[2]))
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
        cache_key = (
            f"K{self.K_keep}_mvcomp{self.mv_compensate}_mvincon{int(self.mv_use_inconsistency)}_ks{int(self.mv_incon_ksize)}_resgrad{int(self.res_use_grad)}"
            f"_dup{int(self.mask_all_duplicates)}_cp{self.center_prior:.4f}_cs{self.center_sigma:.4f}"
            f"_st{int(self.static_fallback)}_sta{self.static_abs_thresh:.3f}_str{self.static_rel_thresh:.3f}_stf{int(self.static_uniform_frames)}"
        )
        visible_indices = get_cache(video_path, self.cache_dir, cache_key=cache_key)

        if visible_indices is None:
            try:
                video_data, residuals_y, duration, frame_id_list = self.get_frame_id_list(
                    video_path,
                    self.sequence_length,
                    mv_compensate=self.mv_compensate,
                    mv_use_inconsistency=self.mv_use_inconsistency,
                    mv_incon_ksize=self.mv_incon_ksize,
                    res_use_grad=self.res_use_grad,
                )
            except:
                video_path, video_label = self.replace_example_info
                video_data, residuals_y, duration, frame_id_list = self.get_frame_id_list(
                    video_path,
                    self.sequence_length,
                    mv_compensate=self.mv_compensate,
                    mv_use_inconsistency=self.mv_use_inconsistency,
                    mv_incon_ksize=self.mv_incon_ksize,
                    res_use_grad=self.res_use_grad,
                )
            residuals_y = resize_and_center_crop_residuals(residuals_y, input_size=self.input_size)
            visible_indices = compute_visible_indices_cpu(
                residuals_y=residuals_y,
                patch_size=self.patch_size,
                K=self.K_keep,
                static_fallback=self.static_fallback,
                static_abs_thresh=self.static_abs_thresh,
                static_rel_thresh=self.static_rel_thresh,
                static_uniform_frames=self.static_uniform_frames,
            )
            save_cache(visible_indices, video_path, self.cache_dir, cache_key=cache_key)
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
            crop_pos_x=0.5,   # Center crop
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
            crop_pos_x=0.5,   # Center crop
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
    mv_compensate: str = "similarity",
    mv_use_inconsistency: bool = True,
    mv_incon_ksize: int = 3,
    res_use_grad: bool = False,
    mask_all_duplicates: bool = False,
    center_prior: float = 0.3,
    center_sigma: float = 0.35,
    static_fallback: bool = True,
    static_abs_thresh: float = 116.0,
    static_rel_thresh: float = 0.55,
    static_uniform_frames: int = 4,
) -> DALIWarper:
    """
    Create a DALI dataloader for video data with motion vector features from codec.
    
    Args:
        cache_dir: Directory to store cached residuals. If None, uses data_root_path/cache_residuals.
                   Set to empty string "" to disable caching.
        mv_compensate: 'none'|'median'|'mean'|'similarity'|'affine' for MV global compensation (camera-motion removal)
        mv_use_inconsistency: Whether to use local variance of MV magnitude (highlights moving subject vs background)
        mv_incon_ksize: Neighborhood size (odd >=3) for MV inconsistency
        res_use_grad: Use gradient-based residual energy (more structural, less texture-driven)
        mask_all_duplicates: If uniform sampling produces duplicate frame ids, fully mask those sampled time steps
        center_prior: Strength of center Gaussian prior applied to fused map (0 disables)
        center_sigma: Gaussian sigma as a fraction of min(H,W)
        static_fallback: Enable hybrid fallback for almost-static videos (uniform few frames + remaining Top-K)
        static_abs_thresh: Absolute low-energy threshold on patch mean intensity (~0..255)
        static_rel_thresh: Relative contrast threshold (0..1), smaller means flatter distribution
        static_uniform_frames: Number of uniformly-picked frames for the hybrid part (e.g., 4)
    """
    print(f"[{mode} loader] Reading from: {data_csv_path}")
    
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
        "mv_compensate": mv_compensate,
        "mv_use_inconsistency": mv_use_inconsistency,
        "mv_incon_ksize": mv_incon_ksize,
        "res_use_grad": res_use_grad,
        "mask_all_duplicates": bool(mask_all_duplicates),
        "center_prior": float(center_prior),
        "center_sigma": float(center_sigma),
        "static_fallback": bool(static_fallback),
        "static_abs_thresh": float(static_abs_thresh),
        "static_rel_thresh": float(static_rel_thresh),
        "static_uniform_frames": int(static_uniform_frames),
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
