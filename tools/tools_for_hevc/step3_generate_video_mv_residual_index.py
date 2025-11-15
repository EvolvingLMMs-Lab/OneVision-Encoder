import os
import sys
import math
import argparse
import traceback
import numpy as np
import decord

try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False
try:
    from PIL import Image
    _HAS_PIL = True
except Exception:
    _HAS_PIL = False

import torch
from numpy.lib.format import open_memmap

import numpy as np
# ---- 你的残差读取器 ----
from hevc_feature_decoder_mv import HevcFeatureReader

# ===== 新增：读取分布式环境变量 =====
RANK = int(os.environ.get("RANK", "0"))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
# =====================================


def _y_from_yuv_bytes(buf: bytes, H: int, W: int, cw: int, ch: int, tight: bool = True) -> np.ndarray:
    if tight:
        Ysz = H * W
        y = np.frombuffer(buf, dtype=np.uint8, count=Ysz)
        return y.reshape(H, W)
    else:
        Ysz = cw * ch
        y = np.frombuffer(buf, dtype=np.uint8, count=Ysz).reshape(ch, cw)
        return y[:H, :W]

def _reshape_mv_from_bytes(buf: bytes, H: int, W: int) -> np.ndarray:
    H4, W4 = (H >> 2), (W >> 2)
    cnt = H4 * W4
    mv = np.frombuffer(buf, dtype=np.int16, count=cnt)
    return mv.reshape(H4, W4)

def _reshape_ref_from_bytes(buf: bytes, H: int, W: int) -> np.ndarray:
    H4, W4 = (H >> 2), (W >> 2)
    cnt = H4 * W4
    ref = np.frombuffer(buf, dtype=np.uint8, count=cnt)
    return ref.reshape(H4, W4)

def _reshape_size_from_bytes(buf: bytes, H: int, W: int) -> np.ndarray:
    H8, W8 = (H >> 3), (W >> 3)
    cnt = H8 * W8
    sz = np.frombuffer(buf, dtype=np.uint8, count=cnt)
    return sz.reshape(H8, W8)

# ---------- 可视化工具（带无 OpenCV 兜底） ----------
def _viz_mv_to_hsv_bgr(mvx: np.ndarray, mvy: np.ndarray, full_hw: tuple = None) -> np.ndarray:
    """
    将 MV (x,y) 可视化为 HSV→BGR 图。若无 OpenCV，则退化为灰度幅值图的三通道堆叠。
    mvx, mvy: (H/4,W/4) 或 (H,W)；若与 full_hw=(H,W) 匹配 1/4 分辨率，将先最近邻上采样。
    return: uint8 BGR (H,W,3)
    """
    assert mvx.shape == mvy.shape
    Hs, Ws = mvx.shape
    if full_hw is not None and (Hs * 4 == full_hw[0] and Ws * 4 == full_hw[1]):
        if _HAS_CV2:
            mvx_u = cv2.resize(mvx, (full_hw[1], full_hw[0]), interpolation=cv2.INTER_NEAREST)
            mvy_u = cv2.resize(mvy, (full_hw[1], full_hw[0]), interpolation=cv2.INTER_NEAREST)
        else:
            ys = (np.linspace(0, Hs-1, full_hw[0])).astype(np.int32)
            xs = (np.linspace(0, Ws-1, full_hw[1])).astype(np.int32)
            mvx_u = mvx[ys[:,None], xs[None,:]]
            mvy_u = mvy[ys[:,None], xs[None,:]]
    else:
        mvx_u, mvy_u = mvx, mvy

    if _HAS_CV2:
        ang = np.arctan2(-mvy_u, mvx_u)  # y-down
        ang = (ang + np.pi) / (2*np.pi)  # 0..1
        mag = np.sqrt(mvx_u.astype(np.float32)**2 + mvy_u.astype(np.float32)**2)
        s = np.percentile(mag, 95) if np.isfinite(mag).all() else 1.0
        s = max(float(s), 1e-6)
        mag = np.clip(mag / s, 0.0, 1.0)
        hsv = np.zeros((mvx_u.shape[0], mvx_u.shape[1], 3), dtype=np.uint8)
        hsv[..., 0] = (ang * 179.0).astype(np.uint8)  # Hue in [0,179]
        hsv[..., 1] = 255
        hsv[..., 2] = (mag * 255.0).astype(np.uint8)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return bgr
    else:
        # 无 OpenCV：用幅值灰度图复制到 3 个通道
        mag = np.sqrt(mvx_u.astype(np.float32)**2 + mvy_u.astype(np.float32)**2)
        s = np.percentile(mag, 95) if np.isfinite(mag).all() else 1.0
        s = max(float(s), 1e-6)
        g = np.clip(mag / s, 0.0, 1.0)
        g = (g * 255.0).astype(np.uint8)
        return np.stack([g, g, g], axis=-1)

def _viz_residual_y(res_y: np.ndarray, signed: bool = True) -> np.ndarray:
    """
    将残差 Y (H,W) 可视化。默认以 128 为中心显示正负偏差；若无 OpenCV 则直接输出灰度或双边归一化后的灰度。
    return: uint8 BGR (H,W,3) 若有 OpenCV，否则 uint8 灰度 (H,W) 或 (H,W,3)
    """
    if res_y.ndim != 2:
        res_y = np.squeeze(res_y)
        if res_y.ndim != 2:
            raise ValueError(f"unexpected residual shape: {res_y.shape}")
    img = res_y.astype(np.float32)
    if signed:
        img = img - 128.0
        a = np.percentile(np.abs(img), 95)
        a = max(float(a), 1.0)
        img = (img + a) / (2*a)  # [-a,a]→[0,1]
    else:
        a = np.percentile(img, 95)
        a = max(float(a), 1.0)
        img = np.clip(img / a, 0.0, 1.0)
    vis_u8 = (img * 255.0).astype(np.uint8)
    if _HAS_CV2:
        return cv2.applyColorMap(vis_u8, cv2.COLORMAP_TURBO)
    else:
        # 无 OpenCV：退化为三通道灰度
        return np.stack([vis_u8, vis_u8, vis_u8], axis=-1)


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

# ---------- 小工具 ----------
def _resize_u8_gray(arr_u8: np.ndarray, size: int = 224) -> np.ndarray:
    if arr_u8.ndim != 2:
        raise ValueError(f"expect HxW, got shape {arr_u8.shape}")
    if _HAS_CV2:
        interp = cv2.INTER_AREA if (arr_u8.shape[0] > size or arr_u8.shape[1] > size) else cv2.INTER_LINEAR
        return cv2.resize(arr_u8, (size, size), interpolation=interp)
    elif _HAS_PIL:
        return np.array(Image.fromarray(arr_u8).resize((size, size), resample=Image.BILINEAR), dtype=np.uint8)
    else:
        H, W = arr_u8.shape
        ys = (np.linspace(0, H-1, size)).astype(np.int32)
        xs = (np.linspace(0, W-1, size)).astype(np.int32)
        return arr_u8[ys[:,None], xs[None,:]]

def _maybe_swap_to_hevc(p: str) -> str:
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


def _load_list_file(list_path: str):
    with open(list_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]
    return lines  # 每行一个视频路径

def mask_by_residual_topk(res_torch: torch.Tensor, k_keep: int, patch_size: int):
    assert res_torch.dim() == 5 and res_torch.size(1) == 1, "res 需为 (B,1,T,H,W)"
    B, _, T, H, W = res_torch.shape
    ph = pw = patch_size
    assert H % ph == 0 and W % pw == 0, "H/W 必须能被 patch 大小整除"
    hb = H // ph
    wb = W // pw
    L = T * hb * wb

    K = int(max(0, min(k_keep, L)))
    res_abs = res_torch.abs().squeeze(1)  # (B,T,H,W)
    scores = res_abs.reshape(B, T, hb, ph, wb, pw).sum(dim=(3,5)).reshape(B, L)

    if K > 0:
        topk_idx = torch.topk(scores, k=K, dim=1, largest=True, sorted=False).indices
        visible_indices = torch.sort(topk_idx, dim=1).values
    else:
        visible_indices = torch.empty(B, 0, dtype=torch.long, device=res_torch.device)
    return visible_indices  # (B,K)

def process_one_video_mv_res(
    video_path: str,
    seq_len: int,
    patch_size: int,
    K: int,
    hevc_n_parallel: int,
    hevc_y_only: bool,          # 仅保留签名一致，这里不直接使用
    *,
    mv_unit_div: float = 4.0,   # quarter-pel -> pixel
    mv_pct: float = 95.0,       # MV 归一化分位数（传给 _mv_energy_norm）
    res_pct: float = 95.0,      # 残差归一化分位数（传给 _residual_energy_norm）
    fuse_mode: str = "weighted",
    w_mv: float = 1.0,
    w_res: float = 1.0,
) -> np.ndarray:
    """
    读取视频前 seq_len 帧的 MV(L0) 与残差(Y)，I 位置置 0；按 _mv_energy_norm/_residual_energy_norm 归一化，
    用 _fuse_energy 融合后在 224×224 上做 patch Top-K，返回 (K,) int32。
    异常时返回全 0（便于断点续跑）。
    """
    try:
        os.environ["UMT_HEVC_Y_ONLY"] = "1" if hevc_y_only else "0"
        prefix_fast = int(os.environ.get("HEVC_PREFIX_FAST", "1")) != 0

        video_path = _maybe_swap_to_hevc(video_path)
        vr = decord.VideoReader(video_path, num_threads=max(4, hevc_n_parallel), ctx=decord.cpu(0))
        duration = len(vr)

        T = seq_len
        if duration >= T:
            frame_id_list = list(range(T))
        else:
            frame_id_list = list(range(duration)) + [duration - 1] * (T - duration)

        # I 帧（相对 T 片段的位置）
        key_idx = None
        if hasattr(vr, "get_key_indices"):
            key_idx = vr.get_key_indices()
        elif hasattr(vr, "get_keyframes"):
            key_idx = vr.get_keyframes()
        if key_idx is not None:
            I_global = set(int(i) for i in np.asarray(key_idx).tolist())
        else:
            I_global = set(int(i) for i in range(0, duration, 16))
        I_pos = set([i for i, fid in enumerate(frame_id_list) if fid in I_global])

        # 读残差（Y 通道），I 位置置 0
        Tsel = T
        # --- 用 HevcFeatureReader 与 C 端严格对齐（读取顺序与字段布局由 C 端决定） ---
        rdr = HevcFeatureReader(video_path, nb_frames=seq_len, n_parallel=hevc_n_parallel)
        H, W = rdr.height, rdr.width

        T = int(seq_len)
        fused_list = [None] * T  # 存每帧融合后的 [0,1] map，shape=(H,W)

        # 提供一个小工具：把残差转成 Y 通道（若是 BGR）
        def _residual_y(residual: np.ndarray) -> np.ndarray:
            if residual.ndim == 2:
                return residual
            if residual.ndim == 3 and residual.shape[2] == 3:
                # BGR -> Y
                return cv2.cvtColor(residual, cv2.COLOR_BGR2YUV)[:, :, 0]
            # 其它形状做兜底（尽量 squeeze 到 H×W）
            r = np.squeeze(residual)
            if r.ndim == 2:
                return r
            raise ValueError(f"Unexpected residual shape: {residual.shape}")

        # 逐帧读取，保持首 T 帧（不足用最后一帧补齐）
        frames_collected = 0
        last_fused = np.zeros((H, W), dtype=np.float32)

        try:
            if all(fid == i for i, fid in enumerate(frame_id_list)) and prefix_fast:
                # 连续帧：直接取前 Tsel 帧
                it = rdr.nextFrameEx()
                for i in range(Tsel):
                    frame_tuple, meta = next(it)
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

                    if i in I_pos:
                        fused_list[i] = np.zeros((H, W), dtype=np.float32)
                        continue

                    mvx_hw = rdr._upsample_mv_to_hw(mv_x_L0.astype(np.float32))
                    mvy_hw = rdr._upsample_mv_to_hw(mv_y_L0.astype(np.float32))
                    mv_norm, _ = _mv_energy_norm(
                        mvx_hw, mvy_hw, H, W,
                        mv_unit_div=mv_unit_div, pct=mv_pct
                    )

                    Y_res = _residual_y(residual)
                    res_norm, _ = _residual_energy_norm(Y_res, pct=res_pct)

                    fused = _fuse_energy(
                        mv_norm, res_norm,
                        mode=fuse_mode, w_mv=w_mv, w_res=w_res
                    )
                    fused_list[i] = fused
                    last_fused = fused
            else:
                # 非连续：顺序扫描并映射
                wanted = set(frame_id_list)
                idx2pos = {fid: i for i, fid in enumerate(frame_id_list)}
                filled = 0
                cur_idx = 0
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

                        if pos in I_pos:
                            fused_list[pos] = np.zeros((H, W), dtype=np.float32)
                        else:
                            mvx_hw = rdr._upsample_mv_to_hw(mv_x_L0.astype(np.float32))
                            mvy_hw = rdr._upsample_mv_to_hw(mv_y_L0.astype(np.float32))
                            mv_norm, _ = _mv_energy_norm(
                                mvx_hw, mvy_hw, H, W,
                                mv_unit_div=mv_unit_div, pct=mv_pct
                            )
                            Y_res = _residual_y(residual)
                            res_norm, _ = _residual_energy_norm(Y_res, pct=res_pct)
                            fused = _fuse_energy(
                                mv_norm, res_norm,
                                mode=fuse_mode, w_mv=w_mv, w_res=w_res
                            )
                            fused_list[pos] = fused
                            last_fused = fused

                        filled += 1
                        if filled >= Tsel:
                            break
                    cur_idx += 1
        finally:
            rdr.close()
        # 若不足 T 帧，则用最后一帧的 fused 重复补齐
        for i in range(frames_collected, T):
            fused_list[i] = last_fused.copy()

        # 兜底（极端异常下）
        for i in range(T):
            if fused_list[i] is None:
                fused_list[i] = np.zeros((H, W), dtype=np.float32)

        fused = np.stack(fused_list, axis=0)  # (T, H, W)

        # resize -> 224×224，并转成"强度"型 uint8（不需要居中）
        fused224_u8 = np.empty((T, 224, 224), dtype=np.uint8)
        for i in range(T):
            fused224_u8[i] = _resize_u8_gray((fused[i] * 255.0).astype(np.uint8), size=224)

        # 用你现有的 Top-K（按 patch 求和）
        score_int16 = fused224_u8.astype(np.int16)  # (T,224,224)
        try:
            # Fast path when PyTorch has NumPy bridge available
            res_torch = torch.from_numpy(score_int16).to(torch.int16).unsqueeze(0).unsqueeze(1)
        except Exception:
            # Fallback: avoid NumPy bridge entirely (PyTorch built without NumPy)
            res_torch = torch.tensor(score_int16.tolist(), dtype=torch.int16).unsqueeze(0).unsqueeze(1)

        with torch.no_grad():
            vis_idx = mask_by_residual_topk(res_torch, K, patch_size)  # (1,K)

        try:
            vis_idx_np = vis_idx.squeeze(0).cpu().numpy().astype(np.int32)
        except Exception:
            # Fallback when PyTorch is built without NumPy bridge
            vis_idx_np = np.asarray(
                vis_idx.squeeze(0).cpu().to(torch.int32).tolist(),
                dtype=np.int32
            )
        return vis_idx_np

    except Exception:
        import traceback, sys
        sys.stderr.write(f"[WARN][MV+RES] failed: {video_path}\n{traceback.format_exc()}\n")
        return np.zeros((K,), dtype=np.int32)


# ---------- 单视频可视化调试分支 ----------
def _debug_dump_video(video_path: str, out_dir: str, frames: int, hevc_n_parallel: int = 1):
    os.makedirs(out_dir, exist_ok=True)
    rdr = HevcFeatureReader(video_path, nb_frames=frames, n_parallel=hevc_n_parallel)
    H, W = rdr.height, rdr.width
    cnt = 0
    try:
        for (frame_tuple, meta) in rdr.nextFrameEx():
            if cnt >= frames:
                break
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
            raw_mode = False
            if isinstance(meta, dict) and meta.get("raw_mode", False):
                raw_mode = True
                Hc = int(meta.get("coded_height", H))
                Wc = int(meta.get("coded_width", W))
                tight = bool(meta.get("tight_planes", True))
            # 只做 L0；I 帧也可视化（通常为 0）
            if raw_mode:
                mvx = rdr._upsample_mv_to_hw(_reshape_mv_from_bytes(mv_x_L0, H, W).astype(np.float32))
                mvy = rdr._upsample_mv_to_hw(_reshape_mv_from_bytes(mv_y_L0, H, W).astype(np.float32))
            else:
                mvx = rdr._upsample_mv_to_hw(mv_x_L0.astype(np.float32))
                mvy = rdr._upsample_mv_to_hw(mv_y_L0.astype(np.float32))
            mv_bgr = _viz_mv_to_hsv_bgr(mvx, mvy, full_hw=(H, W))

            # 残差取 Y
            if raw_mode:
                y_res = _y_from_yuv_bytes(residual, H, W, Wc, Hc, tight=tight)
            else:
                if residual.ndim == 3 and residual.shape[2] == 3:
                    if _HAS_CV2:
                        y_res = cv2.cvtColor(residual, cv2.COLOR_BGR2YUV)[:, :, 0]
                    else:
                        # 无 OpenCV：简单取第一通道兜底
                        y_res = residual[:, :, 0]
                else:
                    y_res = np.squeeze(residual)
            res_bgr = _viz_residual_y(y_res, signed=True)

            # 写文件
            out_prefix = os.path.join(out_dir, f"{cnt:05d}")
            if _HAS_CV2:
                cv2.imwrite(out_prefix + "_mv_hsv.png", mv_bgr)
                cv2.imwrite(out_prefix + "_residual_viz.png", res_bgr)
                # RGB/Y 也顺手导出
                if raw_mode:
                    y_vis = _y_from_yuv_bytes(rgb, H, W, Wc, Hc, tight=tight)
                    cv2.imwrite(out_prefix + "_rgb.png", cv2.cvtColor(y_vis, cv2.COLOR_GRAY2BGR))
                elif isinstance(rgb, np.ndarray) and rgb.ndim == 2:
                    cv2.imwrite(out_prefix + "_rgb.png", cv2.cvtColor(rgb, cv2.COLOR_GRAY2BGR))
                elif isinstance(rgb, np.ndarray) and rgb.ndim == 3:
                    cv2.imwrite(out_prefix + "_rgb.png", rgb)
            else:
                if _HAS_PIL:
                    Image.fromarray(mv_bgr).save(out_prefix + "_mv_hsv.png")
                    Image.fromarray(res_bgr).save(out_prefix + "_residual_viz.png")
                    if raw_mode:
                        y_vis = _y_from_yuv_bytes(rgb, H, W, Wc, Hc, tight=tight)
                        Image.fromarray(np.stack([y_vis]*3, axis=-1)).save(out_prefix + "_rgb.png")
                    elif isinstance(rgb, np.ndarray):
                        rgb_img = rgb if (rgb.ndim == 3 and rgb.shape[2] == 3) else np.stack([rgb]*3, axis=-1)
                        Image.fromarray(rgb_img.astype(np.uint8)).save(out_prefix + "_rgb.png")
                else:
                    # 最简兜底：只保存 .npy
                    pass
            # 方便深入排查，保存原始数组
            np.save(out_prefix + "_mvx_L0.npy", mvx.astype(np.float32))
            np.save(out_prefix + "_mvy_L0.npy", mvy.astype(np.float32))
            np.save(out_prefix + "_residual_y.npy", y_res.astype(np.uint8))
            cnt += 1
    finally:
        rdr.close()
    print(f"[debug] dumped {cnt} frames to {out_dir}")


# ===== 新增：按样本生成输出路径的工具函数 =====
def _make_out_path(video_path: str, src: str, dst: str, suffix: str = ".visidx.npy") -> str:
    """基于原始视频路径做字符串 replace，保持层级不变，只改前缀，并将扩展改为 .visidx.npy"""
    if src not in video_path:
        raise ValueError(f"--out_replace_src '{src}' 不在视频路径中：{video_path}")
    replaced = video_path.replace(src, dst)
    stem, _ = os.path.splitext(replaced)
    out_path = stem + suffix
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    return out_path
# ==============================================


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", type=str, default="/video_vit/dataset/clips_square_aug_k710_ssv2_hevc_v2/shuf_merged_list", help="视频列表文件（每行一个路径）")
    # ===== 新增：分文件输出相关参数 =====
    ap.add_argument("--out_replace_src", default="clips_square_aug_k710_ssv2_hevc_v2", help="输出路径替换：原始路径中的子串（如原根目录）")
    ap.add_argument("--out_replace_dst", default="clips_square_aug_k710_ssv2_hevc_v2_residual_mv", help="输出路径替换：替换成的子串（如新根目录）")
    ap.add_argument("--out_suffix", default=".visidx.npy", help="每个视频结果文件后缀，默认 .visidx.npy")
    ap.add_argument("--overwrite",  type=int, default=0, help="输出文件存在时是否覆盖（0 跳过，1 覆盖）")
    # ====================================
    ap.add_argument("--seq-len", type=int, default=64, help="T：每视频使用的帧数（不足重复最后一帧）")
    ap.add_argument("--patch-size", type=int, default=16, help="ViT patch 大小，需整除 224")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--keep-ratio", type=float, default=0.30, help="保留比例（0~1），用于计算 K")
    g.add_argument("--k-keep",     type=int,   default=2000, help="直接指定 K")
    ap.add_argument("--hevc-n-parallel", type=int, default=6, help="HevcFeatureReader 并行度")
    ap.add_argument("--hevc-y-only",     type=int, default=1, help="Y 通道残差（1/0）")
    ap.add_argument("--flush-every",     type=int, default=100, help="每处理多少视频打印一次日志")
    # 单视频调试参数
    ap.add_argument("--video", type=str, help="单视频调试：输入视频路径（优先级高于 --list）")
    ap.add_argument("--debug-out", type=str, default="viz_residual_debug", help="单视频调试输出目录")
    ap.add_argument("--debug-frames", type=int, default=16, help="单视频调试：读取的帧数 T")
    ap.add_argument("--local_rank", type=int, default=0, help="本地 rank（DeepSpeed 自动注入）")

    args = ap.parse_args()

    # === 单视频可视化直通分支（不变） ===
    if args.video:
        vp = _maybe_swap_to_hevc(args.video)
        os.makedirs(args.debug_out, exist_ok=True)
        _debug_dump_video(vp, args.debug_out, frames=args.debug_frames, hevc_n_parallel=args.hevc_n_parallel)
        return

    videos = _load_list_file(args.list)  # 保序
    N = len(videos)
    if N == 0:
        raise RuntimeError("empty list")

    # 计算 L 与 K
    T = int(args.seq_len)
    p = int(args.patch_size)
    assert 224 % p == 0, "224 必须能被 patch_size 整除"
    hb = 224 // p
    wb = 224 // p
    L = T * hb * wb

    if args.k_keep is not None and args.k_keep >= 0:
        K = min(int(args.k_keep), L)
    else:
        keep_ratio = max(0.0, min(float(args.keep_ratio), 1.0))
        K = int(round(L * keep_ratio))
    if K <= 1:
        raise RuntimeError(f"K={K} 不安全（可能与全 0 行冲突），请设置更大的 keep-ratio 或 k-keep。")

    # ===== 新增：DeepSpeed 分片逻辑 =====
    num_local = N // WORLD_SIZE + int(RANK < (N % WORLD_SIZE))
    start = (N // WORLD_SIZE) * RANK + min(RANK, N % WORLD_SIZE)
    end = start + num_local
    local_indices = list(range(start, end))

    if RANK == 0:
        print(f"[dist] WORLD_SIZE={WORLD_SIZE}, N={N}, K={K}, slice per rank ~{N//max(1,WORLD_SIZE)} (+余数)")
    print(f"[rank {RANK}/{WORLD_SIZE}] will process indices [{start}, {end}) => {num_local} videos")
    # ====================================

    processed = 0
    for c, i in enumerate(local_indices, 1):
        vp = videos[i].strip()
        if not vp:
            sys.stderr.write(f"[WARN][rank {RANK}] empty line at {i}, skip\n")
            continue

        # ===== 新增：生成单样本输出路径 =====
        try:
            out_path = _make_out_path(
                video_path=vp,
                src=args.out_replace_src,
                dst=args.out_replace_dst,
                suffix=args.out_suffix,
            )
        except Exception as e:
            sys.stderr.write(f"[ERROR][rank {RANK}] make_out_path failed at idx {i}: {e}\n")
            continue

        if os.path.exists(out_path) and not bool(args.overwrite):
            if (c % max(1, args.flush_every)) == 0:
                print(f"[rank {RANK}] skip exists: {out_path} (c={c}/{len(local_indices)})")
            continue
        # =====================================

        vis_idx = process_one_video_mv_res(
            video_path=vp,
            seq_len=T,
            patch_size=p,
            K=K,
            hevc_n_parallel=args.hevc_n_parallel,
            hevc_y_only=bool(args.hevc_y_only),
        )

        # ===== 新增：保存到单样本文件 =====
        try:
            np.save(out_path, vis_idx.astype(np.int32), allow_pickle=False)
        except Exception as e:
            sys.stderr.write(f"[ERROR][rank {RANK}] save failed at {out_path}: {e}\n")
            continue
        # =================================

        processed += 1
        if (c % max(1, args.flush_every)) == 0:
            print(f"[rank {RANK}] processed {c}/{len(local_indices)} (global idx {i}), saved: {out_path}")

    print(f"[rank {RANK}] done. processed={processed}, assigned={len(local_indices)}")

if __name__ == "__main__":
    main()
