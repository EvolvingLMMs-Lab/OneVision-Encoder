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
from hevc_feature_decoder import HevcFeatureReader


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
        # 允许把非hevc路径自动替换为 *_hevc
        vp = _maybe_swap_to_hevc(video_path)

        # --- 用 HevcFeatureReader 与 C 端严格对齐（读取顺序与字段布局由 C 端决定） ---
        rdr = HevcFeatureReader(vp, nb_frames=seq_len, n_parallel=hevc_n_parallel)
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
        for (frame_tuple, meta) in rdr.nextFrameEx():
            if frames_collected >= T:
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

            raw_mode = bool(meta.get("raw_mode", False))
            if raw_mode:
                Hc = int(meta.get("coded_height", H))
                Wc = int(meta.get("coded_width", W))
                tight = bool(meta.get("tight_planes", True))
                # rgb carries raw YUV bytes; residual carries raw residual YUV bytes
                y_plane = _y_from_yuv_bytes(rgb, H, W, Wc, Hc, tight=tight)
                y_res   = _y_from_yuv_bytes(residual, H, W, Wc, Hc, tight=tight)
                mv_x_L0 = _reshape_mv_from_bytes(mv_x_L0, H, W)
                mv_y_L0 = _reshape_mv_from_bytes(mv_y_L0, H, W)
                ref_off_L0 = _reshape_ref_from_bytes(ref_off_L0, H, W)
                # L1/size 可按需：这里不使用，保持占位

            if meta.get("is_i_frame", False):
                fused_list[frames_collected] = np.zeros((H, W), dtype=np.float32)
            else:
                # 1) MV：四分之一分辨率上采样到 H×W；单位换算（quarter-pel -> px）在 _mv_energy_norm 内处理
                mvx = rdr._upsample_mv_to_hw(mv_x_L0.astype(np.float32))
                mvy = rdr._upsample_mv_to_hw(mv_y_L0.astype(np.float32))
                mv_norm, _mv_scale_px = _mv_energy_norm(
                    mvx, mvy, H, W, mv_unit_div=mv_unit_div, pct=mv_pct
                )  # H×W, float32 in [0,1]

                # 2) 残差：取得 Y 平面并做分位数归一化
                Y_res = y_res if raw_mode else _residual_y(residual)
                res_norm, _res_scale = _residual_energy_norm(Y_res, pct=res_pct)  # H×W

                # 3) 融合
                fused = _fuse_energy(
                    mv_norm, res_norm, mode=fuse_mode, w_mv=w_mv, w_res=w_res
                )
                fused_list[frames_collected] = fused
                last_fused = fused

            frames_collected += 1

        rdr.close()

        # 若不足 T 帧，则用最后一帧的 fused 重复补齐
        for i in range(frames_collected, T):
            fused_list[i] = last_fused.copy()

        # 兜底（极端异常下）
        for i in range(T):
            if fused_list[i] is None:
                fused_list[i] = np.zeros((H, W), dtype=np.float32)

        fused = np.stack(fused_list, axis=0)  # (T, H, W)

        # resize -> 224×224，并转成“强度”型 uint8（不需要居中）
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


def _find_zero_rows_memmap(mm: np.memmap, chunk: int = 20000):
    """返回所有全 0 行的索引列表（分块扫描，避免一次性读全量内存）"""
    N = mm.shape[0]
    todo = []
    for st in range(0, N, chunk):
        ed = min(N, st + chunk)
        blk = mm[st:ed]           # memmap 切片，逐块触盘
        # 行是否全 0：等价于 (blk != 0).any(axis=1) == False
        row_zero = ~(blk != 0).any(axis=1)
        idxs = np.nonzero(row_zero)[0] + st
        if idxs.size:
            todo.extend(idxs.tolist())
    return todo

#   --list /video_vit/train_UniViT/mp4_list.txt \
#   --out-file /video_vit/train_UniViT/visible_indices.npy \
#   --seq-len 16 --patch-size 16 --keep-ratio 0.30 \
#   --hevc-n-parallel 6 --hevc-y-only 1 --resume 1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", type=str, default= "/video_vit/train_UniViT/mp4_list_part_3.txt", help="视频列表文件（每行一个路径）")
    ap.add_argument("--out-file",  type=str, default= "/video_vit/feilong/CVPR/tools_for_hevc/mp4_list_part_3.npy", help="合并输出（.npy），仅保存 visible_indices，总形状 (N,K)")
    ap.add_argument("--seq-len", type=int, default=64, help="T：每视频使用的帧数（不足重复最后一帧）")
    ap.add_argument("--patch-size", type=int, default=16, help="ViT patch 大小，需整除 224")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--keep-ratio", type=float, default=0.30, help="保留比例（0~1），用于计算 K")
    g.add_argument("--k-keep",     type=int,   default=1568, help="直接指定 K")
    ap.add_argument("--hevc-n-parallel", type=int, default=6, help="HevcFeatureReader 并行度")
    ap.add_argument("--hevc-y-only",     type=int, default=1, help="Y 通道残差（1/0）")
    ap.add_argument("--flush-every",     type=int, default=100, help="每处理多少视频 flush 一次")
    # 断点续跑相关
    ap.add_argument("--resume",          type=int, default=1, help="若 out-file 存在则断点续跑（按全 0 行判定）")
    ap.add_argument("--overwrite",       type=int, default=0, help="忽略旧文件，重新创建并清零")
    ap.add_argument("--scan-chunk",      type=int, default=20000, help="扫描全 0 行时的分块大小")
    ap.add_argument("--prezero",         type=int, default=1, help="新建文件时是否整体清零（便于断点续跑）")
    # 单视频调试参数
    ap.add_argument("--video", type=str, help="单视频调试：输入视频路径（优先级高于 --list）")
    ap.add_argument("--debug-out", type=str, default="viz_residual_debug", help="单视频调试输出目录")
    ap.add_argument("--debug-frames", type=int, default=16, help="单视频调试：读取的帧数 T")

    args = ap.parse_args()

    # === 单视频可视化直通分支 ===
    if args.video:
        vp = _maybe_swap_to_hevc(args.video)
        os.makedirs(args.debug_out, exist_ok=True)
        _debug_dump_video(vp, args.debug_out, frames=args.debug_frames, hevc_n_parallel=args.hevc_n_parallel)
        return

    videos = _load_list_file(args.list)  # 保序
    N = len(videos)
    if N == 0:
        raise RuntimeError("empty list")

    # 计算 L 与 K（或从现有文件继承）
    T = int(args.seq_len)
    p = int(args.patch_size)
    assert 224 % p == 0, "224 必须能被 patch_size 整除"
    hb = 224 // p
    wb = 224 // p
    L = T * hb * wb

    out_file = os.path.abspath(args.out_file)
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    mm = None
    K = None

    if os.path.exists(out_file) and not args.overwrite and args.resume:
        # --- 断点续跑：沿用现有文件 ---
        mm = open_memmap(out_file, mode="r+")
        if mm.dtype != np.int32:
            raise RuntimeError(f"existing file dtype {mm.dtype}, expected int32")
        if mm.shape[0] != N:
            raise RuntimeError(f"existing file rows {mm.shape[0]} != videos {N}（列表内容/数量不一致）")
        K_file = int(mm.shape[1])
        if args.k_keep is not None:
            # 显式指定 K，需与现有文件一致
            if int(args.k_keep) != K_file:
                raise RuntimeError(f"K mismatch: existing {K_file} vs arg {args.k_keep}. 使用 --overwrite 1 重建，或去掉 --k-keep。")
            K = K_file
        else:
            # 未指定则沿用文件的 K
            K = K_file
        if K <= 1:
            raise RuntimeError("K<=1 在“全 0 行判未完成”的语义下不安全，请使用更大的 K 或改用其它占位符策略。")

        # 找出未完成（全 0 行）的索引
        todo = _find_zero_rows_memmap(mm, chunk=args.scan_chunk)
        if not todo:
            print(f"[info] nothing to do, all {N} rows already filled (N,K)=({N},{K})")
            return
        print(f"[resume] found {len(todo)}/{N} rows to process (N,K)=({N},{K})")
    else:
        # --- 新建文件 ---
        # 计算 K（保底 K>=1，但我们强制 K>=2 更安全）
        if args.k_keep is not None and args.k_keep >= 0:
            K = min(int(args.k_keep), L)
        else:
            keep_ratio = max(0.0, min(float(args.keep_ratio), 1.0))
            K = int(round(L * keep_ratio))
        if K <= 1:
            raise RuntimeError(f"K={K} 不安全（可能与全 0 行冲突），请设置更大的 keep-ratio 或 k-keep。")

        mm = open_memmap(out_file, mode="w+", dtype=np.int32, shape=(N, K))
        if args.prezero:
            mm[:] = 0    # 显式清零，保障“全 0 = 未完成”
            mm.flush()
        todo = list(range(N))
        print(f"[init] create {out_file} with shape (N,K)=({N},{K}), zero-initialized={bool(args.prezero)}")

    # --- 主循环：仅处理未完成行（全 0 行） ---
    for c, i in enumerate(todo, 1):
        vp = videos[i].strip()
        if not vp:
            sys.stderr.write(f"[WARN] empty line at {i}, fill zeros\n")
            mm[i, :] = 0
        else:
            vis_idx = process_one_video_mv_res(
                video_path=vp,
                seq_len=T,
                patch_size=p,
                K=K,
                hevc_n_parallel=args.hevc_n_parallel,
                hevc_y_only=bool(args.hevc_y_only),
            )
            mm[i, :] = vis_idx  # 若失败，这里就是全 0；下次还能继续
        if (c % args.flush_every) == 0:
            mm.flush()
            print(f"[info] processed {c}/{len(todo)} (global row {i})")

    mm.flush()
    try:
        os.fsync(mm.fp.fileno())
    except Exception as e:
        print(f"[warn] fsync failed: {e}")
    print(f"[done] saved visible_indices to {out_file} (shape={(N,K)})")

if __name__ == "__main__":
    main()




# import math

# # ==== 输入与输出配置 ====
# in_path = "/video_vit/train_UniViT/mp4_list_part_3.txt"
# out_prefix = "/video_vit/train_UniViT/mp4_list_part_3_split_"  # 输出文件名前缀
# num_parts = 4  # 想分成几份

# # ==== 读取所有非空行 ====
# with open(in_path, "r", encoding="utf-8") as f:
#     lines = [ln for ln in f if ln.strip()]

# total = len(lines)
# chunk = math.ceil(total / num_parts)
# print(f"总行数: {total}, 每份约 {chunk} 行")

# # ==== 按顺序切分并保存 ====
# for i in range(num_parts):
#     part_lines = lines[i * chunk : (i + 1) * chunk]
#     out_path = f"{out_prefix}{i:02d}.txt"
#     with open(out_path, "w", encoding="utf-8") as f:
#         f.writelines(part_lines)
#     print(f"写出 {len(part_lines)} 行 → {out_path}")

# print("✅ 文件切分完成！")

# python step3_generate_video_residual_index.py --list /video_vit/train_UniViT/mp4_list_part_15.txt --out-file /video_vit/feilong/CVPR/tools_for_hevc/part_15.npy
