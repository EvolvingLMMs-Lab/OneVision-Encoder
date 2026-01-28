#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize cv_reader output of motion_vector / residual.

Usage:
python debug_vis_mv_res.py \
  --video /path/to/xxx.mp4 \
  --num_frames 16 \
  --out_dir ./mvres_debug \
  --mv_unit_div 4 \
  --mv_pct 95 \
  --res_vis pct \
  --res_pct 99 \
  --res_gain 16 \
  --dump_stats
"""

import os
import argparse
from pathlib import Path

import numpy as np

# optional
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

try:
    from cv_reader import api as cv_api
except Exception as e:
    raise RuntimeError("cv_reader.api import failed, please confirm installed or in PYTHONPATH") from e


def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def _write_gray_u8(path: Path, arr_u8: np.ndarray):
    assert arr_u8.dtype == np.uint8 and arr_u8.ndim == 2
    if _HAS_CV2:
        cv2.imwrite(str(path), arr_u8)
    elif _HAS_PIL:
        Image.fromarray(arr_u8, mode="L").save(str(path))
    else:
        raise RuntimeError("Missing cv2/PIL, cannot save images")


def _stack_h(*imgs_u8: np.ndarray) -> np.ndarray:
    # imgs are HxW uint8, same H
    return np.concatenate(imgs_u8, axis=1)


def _upsample_nearest(src: np.ndarray, H: int, W: int) -> np.ndarray:
    """src: (Hs, Ws) -> (H, W)"""
    if _HAS_CV2:
        return cv2.resize(src, (W, H), interpolation=cv2.INTER_NEAREST)
    ys = (np.linspace(0, src.shape[0] - 1, H)).astype(np.int32)
    xs = (np.linspace(0, src.shape[1] - 1, W)).astype(np.int32)
    return src[ys[:, None], xs[None, :]]


def _residual_to_y(residual: np.ndarray) -> np.ndarray:
    """residual: (H,W) or (H,W,3)"""
    if residual.ndim == 2:
        return residual
    if residual.ndim == 3 and residual.shape[2] == 3:
        if _HAS_CV2:
            # assume BGR
            y = cv2.cvtColor(residual, cv2.COLOR_BGR2YUV)[:, :, 0]
            return y
        return residual[:, :, 0]
    r = np.squeeze(residual)
    if r.ndim == 2:
        return r
    raise ValueError(f"Unexpected residual shape: {residual.shape}")


def _norm_to_u8(x: np.ndarray, pct: float = 95.0) -> np.ndarray:
    """x float32/float64, normalize by percentile -> [0,255]"""
    x = x.astype(np.float32)
    a = float(np.percentile(x, pct))
    a = max(a, 1e-6)
    y = np.clip(x / a, 0.0, 1.0) * 255.0
    return y.astype(np.uint8)


def _norm_to_u8_gain(x: np.ndarray, gain: float = 8.0) -> np.ndarray:
    """x float32, visualize by multiplying a gain then clipping to [0,255]."""
    x = x.astype(np.float32)
    y = np.clip(x * float(gain), 0.0, 255.0)
    return y.astype(np.uint8)


def _norm_to_u8_log(x: np.ndarray, pct: float = 99.0) -> np.ndarray:
    """x float32, log1p compression then percentile normalization -> [0,255]."""
    x = x.astype(np.float32)
    x = np.log1p(np.maximum(x, 0.0))
    a = float(np.percentile(x, pct))
    a = max(a, 1e-6)
    y = np.clip(x / a, 0.0, 1.0) * 255.0
    return y.astype(np.uint8)


def _stat_line(name: str, x: np.ndarray) -> str:
    x = x.astype(np.float32)
    return (
        f"{name}: min={float(np.min(x)):.3f} max={float(np.max(x)):.3f} "
        f"mean={float(np.mean(x)):.3f} p90={float(np.percentile(x, 90)):.3f} "
        f"p95={float(np.percentile(x, 95)):.3f} p99={float(np.percentile(x, 99)):.3f}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, type=str)
    ap.add_argument("--num_frames", default=16, type=int)
    ap.add_argument("--out_dir", default="./mvres_debug", type=str)
    ap.add_argument("--mv_unit_div", default=4.0, type=float, help="MV unit scaling (commonly 4)")
    ap.add_argument("--mv_pct", default=95.0, type=float, help="MV magnitude percentile normalization")
    ap.add_argument("--res_vis", default="pct", type=str,
                    choices=["pct", "log", "gain"],
                    help="residual visualization mode: pct|log|gain")
    ap.add_argument("--res_pct", default=95.0, type=float,
                    help="residual percentile (used by pct/log modes)")
    ap.add_argument("--res_gain", default=8.0, type=float,
                    help="residual gain (used by gain mode). Example: 8/16/32")
    ap.add_argument("--dump_stats", action="store_true",
                    help="print per-frame stats for residual abs and mv mag")
    ap.add_argument("--save_raw", action="store_true",
                    help="also save raw arrays as .npy (mv_mag_hw/res_abs)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    _ensure_dir(out_dir)

    frames = cv_api.read_video(args.video, 0, int(args.num_frames))
    if not isinstance(frames, (list, tuple)) or len(frames) == 0:
        raise RuntimeError(f"cv_reader.read_video returned empty: {args.video}")

    H = int(frames[0]["height"])
    W = int(frames[0]["width"])
    codec = (frames[0].get("codec_name") or frames[0].get("codec") or "").lower()

    print(f"[INFO] video={args.video}")
    print(f"[INFO] codec={codec}, HxW={H}x{W}, got_frames={len(frames)}")
    print(f"[INFO] save to: {out_dir}")

    for i, fr in enumerate(frames):
        pict = fr.get("pict_type", "?")

        mv = np.asarray(fr.get("motion_vector"))
        res = np.asarray(fr.get("residual"))

        print(f"[{i:03d}] pict_type={pict} mv_shape={mv.shape} res_shape={res.shape} res_vis={args.res_vis} res_pct={args.res_pct} res_gain={args.res_gain}")

        # I-frames usually don't have valid mv/res (set to 0 in post-processing)
        if pict == "I" or mv.size == 0 or res.size == 0:
            # Save all-black placeholder for alignment
            mv_u8 = np.zeros((H, W), dtype=np.uint8)
            res_u8 = np.zeros((H, W), dtype=np.uint8)
        else:
            # ---- MV magnitude ----
            # Expect mv to be (Hs, Ws, 2) (quarter-res or block-grid)
            if mv.ndim != 3 or mv.shape[2] < 2:
                raise ValueError(f"Unexpected motion_vector shape: {mv.shape}")
            mvx = mv[:, :, 0].astype(np.float32) / float(args.mv_unit_div)
            mvy = mv[:, :, 1].astype(np.float32) / float(args.mv_unit_div)
            mag = np.sqrt(mvx * mvx + mvy * mvy)  # (Hs, Ws)

            mag_hw = _upsample_nearest(mag, H, W)
            mv_u8 = _norm_to_u8(mag_hw, pct=args.mv_pct)

            # ---- Residual energy (|Y-128|) ----
            y = _residual_to_y(res)
            y = y.astype(np.float32)
            res_abs = np.abs(y - 128.0)

            if args.dump_stats:
                print("    " + _stat_line("mv_mag_hw", mag_hw))
                print("    " + _stat_line("res_abs", res_abs))

            if args.res_vis == "pct":
                res_u8 = _norm_to_u8(res_abs, pct=args.res_pct)
            elif args.res_vis == "log":
                res_u8 = _norm_to_u8_log(res_abs, pct=args.res_pct)
            else:  # gain
                res_u8 = _norm_to_u8_gain(res_abs, gain=args.res_gain)

            if args.save_raw:
                np.save(out_dir / f"frame_{i:03d}_mv_mag_hw.npy", mag_hw.astype(np.float32), allow_pickle=False)
                np.save(out_dir / f"frame_{i:03d}_res_abs.npy", res_abs.astype(np.float32), allow_pickle=False)

        # save
        mv_path = out_dir / f"frame_{i:03d}_mv_mag.png"
        re_path = out_dir / f"frame_{i:03d}_res_abs.png"
        mix_path = out_dir / f"frame_{i:03d}_mv_res_concat.png"

        _write_gray_u8(mv_path, mv_u8)
        _write_gray_u8(re_path, res_u8)
        _write_gray_u8(mix_path, _stack_h(mv_u8, res_u8))

    print("[DONE] wrote pngs.")


if __name__ == "__main__":
    main()