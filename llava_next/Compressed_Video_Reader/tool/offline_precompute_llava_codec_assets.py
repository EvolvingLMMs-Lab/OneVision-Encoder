#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
offline_precompute_llava_codec_assets.py

Read a jsonl dumped from lmms-eval tasks (each line has at least: video, key),
and precompute:
  - mosaic_000.jpg ... mosaic_{num_images-1}.jpg
  - positions_thw.npy   (shape: num_images*1296, 3) int32, (t,h,w) on virtual seq_len_frames grid
  - visible_indices.npy (shape: num_images*1296,) int64, vi = t*1296 + h*36 + w
  - frame_ids.npy       (shape: seq_len_frames,) int32 sampled frame indices in original video timeline
  - meta.json           configs + debug stats + basic info

This matches the online llava_codec.py "pack_topk time_spatial" behavior.

Requirements:
  - cv_reader installed and available: from cv_reader import api as cv_api
  - opencv-python (cv2), numpy, torch, pillow
"""

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import cv2
import torch
from PIL import Image

# -------- cv_reader --------
try:
    from cv_reader import api as cv_api
    _HAS_CV_READER = True
except Exception as e:
    cv_api = None
    _HAS_CV_READER = False


# -------------------- helpers copied/adapted from online --------------------

def _get_total_frames_cv2(video_path: str) -> int:
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return 0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return max(0, n)
    except Exception:
        return 0


def _get_fps_cv2(video_path: str) -> float:
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return 0.0
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        cap.release()
        if not np.isfinite(fps):
            return 0.0
        return max(0.0, fps)
    except Exception:
        return 0.0


def _sample_frame_ids_uniform_range(start_fid: int, end_fid: int, seq_len: int) -> List[int]:
    start_fid = int(start_fid)
    end_fid = int(end_fid)
    seq_len = int(seq_len)
    if seq_len <= 0:
        return []
    if end_fid < start_fid:
        end_fid = start_fid
    return np.linspace(start_fid, end_fid, seq_len, dtype=np.int32).tolist()


def _ffprobe_sum_pkt_size(video_path: str, start_sec: float, dur_sec: float) -> int:
    """Heuristic energy proxy via packet sizes (compressed domain)."""
    try:
        import subprocess

        start_sec = max(0.0, float(start_sec))
        dur_sec = max(0.0, float(dur_sec))
        interval = f"{start_sec:.6f}%+{dur_sec:.6f}"
        cmd = [
            "ffprobe",
            "-v", "error",
            "-read_intervals", interval,
            "-select_streams", "v:0",
            "-show_packets",
            "-show_entries", "packet=size",
            "-of", "csv=p=0",
            str(video_path),
        ]
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if p.returncode != 0:
            return 0
        s = 0
        for line in p.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                s += int(line)
            except Exception:
                continue
        return int(s)
    except Exception:
        return 0


def _choose_window_start_offset(
    video_path: str,
    total_frames: int,
    fps: float,
    window_len: int,
    pick_mode: str,
) -> Tuple[int, Dict[str, Any]]:
    total_frames = int(total_frames)
    window_len = int(window_len)
    pick = (pick_mode or "none").lower().strip()

    dbg: Dict[str, Any] = {
        "pick": pick,
        "window_len": int(window_len),
        "total_frames": int(total_frames),
        "fps": float(fps),
    }

    if window_len <= 0 or total_frames <= 0:
        return 0, dbg
    if total_frames <= window_len:
        return 0, dbg

    begin = 0
    end = max(0, total_frames - window_len)
    mid = max(0, (total_frames - window_len) // 2)

    if pick in ("begin", "start"):
        return int(begin), dbg
    if pick in ("end", "tail"):
        return int(end), dbg
    if pick in ("mid", "middle"):
        return int(mid), dbg

    if pick in ("energy3", "energy_begin_mid_end"):
        fps_use = float(fps) if (fps and fps > 0) else 30.0
        dur_sec = float(window_len) / fps_use

        cand = {"begin": int(begin), "mid": int(mid), "end": int(end)}
        scores: Dict[str, int] = {}
        for name, st in cand.items():
            start_sec = float(st) / fps_use
            scores[name] = int(_ffprobe_sum_pkt_size(video_path, start_sec, dur_sec))
        dbg["pkt_size_sum"] = scores

        best = max(scores.items(), key=lambda kv: kv[1])[0] if scores else "mid"
        if scores and all(v == 0 for v in scores.values()):
            best = "mid"
        dbg["best"] = best
        return int(cand.get(best, mid)), dbg

    return 0, dbg


def _preprocess_bgr_to_square(frame_bgr: np.ndarray, out_size: int = 576) -> np.ndarray:
    """Resize longer side to out_size and pad to out_size x out_size (center). Returns BGR uint8."""
    if frame_bgr is None:
        raise ValueError("frame_bgr is None")
    H, W = frame_bgr.shape[:2]
    out_size = int(out_size)
    if H <= 0 or W <= 0:
        raise ValueError(f"Bad frame shape: {frame_bgr.shape}")

    scale = float(out_size) / float(max(H, W))
    Hn = max(1, int(round(H * scale)))
    Wn = max(1, int(round(W * scale)))

    resized = cv2.resize(frame_bgr, (Wn, Hn), interpolation=cv2.INTER_LINEAR)

    pad_top = (out_size - Hn) // 2
    pad_left = (out_size - Wn) // 2

    canvas = np.zeros((out_size, out_size, 3), dtype=np.uint8)
    canvas[pad_top:pad_top + Hn, pad_left:pad_left + Wn] = resized
    return canvas


def _residual_y_generic(residual: np.ndarray) -> np.ndarray:
    if residual.ndim == 2:
        return residual
    if residual.ndim == 3 and residual.shape[2] >= 1:
        if residual.shape[2] == 3:
            try:
                return cv2.cvtColor(residual, cv2.COLOR_BGR2YUV)[:, :, 0]
            except Exception:
                return residual[:, :, 0]
        return residual[:, :, 0]
    r = np.squeeze(residual)
    if r.ndim == 2:
        return r
    raise ValueError(f"Unexpected residual shape: {residual.shape}")


def _residual_energy_norm(res_y: np.ndarray, pct: float = 95.0, use_grad: bool = False) -> np.ndarray:
    if not use_grad:
        x = np.abs(res_y.astype(np.float32) - 128.0)
    else:
        r = res_y.astype(np.float32)
        gx = np.abs(np.roll(r, -1, axis=1) - np.roll(r, 1, axis=1))
        gy = np.abs(np.roll(r, -1, axis=0) - np.roll(r, 1, axis=0))
        x = 0.5 * (gx + gy)
    a = float(np.percentile(x, pct))
    a = max(a, 1.0)
    return np.clip(x / a, 0.0, 1.0).astype(np.float32)


def _mv_energy_norm(mv: np.ndarray, H: int, W: int, mv_unit_div: float = 4.0, pct: float = 95.0, compensate: str = "median") -> np.ndarray:
    if mv.ndim == 3 and mv.shape[2] >= 2:
        mvx_q = mv[:, :, 0]
        mvy_q = mv[:, :, 1]
    else:
        raise ValueError(f"Unexpected motion_vector shape: {mv.shape}")

    vx = mvx_q.astype(np.float32) / float(mv_unit_div)
    vy = mvy_q.astype(np.float32) / float(mv_unit_div)

    c = (compensate or "median").lower()
    if c == "median":
        vx = vx - np.median(vx)
        vy = vy - np.median(vy)
    elif c == "mean":
        vx = vx - float(np.mean(vx))
        vy = vy - float(np.mean(vy))
    elif c == "none":
        pass
    else:
        raise ValueError(f"Unknown mv compensate: {compensate}")

    mag = np.sqrt(vx * vx + vy * vy).astype(np.float32)
    a = float(np.percentile(mag, pct))
    a = max(a, 1e-6)
    norm = np.clip(mag / a, 0.0, 1.0)

    norm_u = cv2.resize(norm, (int(W), int(H)), interpolation=cv2.INTER_NEAREST)
    return norm_u.astype(np.float32)


def _resize_longer_pad_square_map(arr2d: np.ndarray, out_size: int = 576):
    H, W = arr2d.shape[:2]
    out_size = int(out_size)
    scale = float(out_size) / float(max(H, W))
    Hn = max(1, int(round(H * scale)))
    Wn = max(1, int(round(W * scale)))

    resized = cv2.resize(arr2d.astype(np.float32), (Wn, Hn), interpolation=cv2.INTER_LINEAR)

    pad_top = (out_size - Hn) // 2
    pad_left = (out_size - Wn) // 2
    out = np.zeros((out_size, out_size), dtype=np.float32)
    out[pad_top:pad_top + Hn, pad_left:pad_left + Wn] = resized

    info = dict(scale=scale, Hn=int(Hn), Wn=int(Wn), pad_top=int(pad_top), pad_left=int(pad_left), out_size=int(out_size))
    return out, info


def _valid_patch_mask_from_pad(out_size: int, patch: int, pad_top: int, pad_left: int, Hn: int, Wn: int) -> np.ndarray:
    out_size = int(out_size)
    p = int(patch)
    hb, wb = out_size // p, out_size // p

    content_top = int(pad_top)
    content_left = int(pad_left)
    content_bottom = int(pad_top + Hn)
    content_right = int(pad_left + Wn)

    mask = np.zeros((hb, wb), dtype=bool)
    for h in range(hb):
        y0, y1 = h * p, (h + 1) * p
        if y0 < content_top or y1 > content_bottom:
            continue
        for w in range(wb):
            x0, x1 = w * p, (w + 1) * p
            if x0 < content_left or x1 > content_right:
                continue
            mask[h, w] = True
    return mask


def _cv_reader_fetch_mvres_by_frame_ids(
    video_path: str,
    frame_ids: List[int],
    with_residual: bool = True,
    seek_to_frame: Optional[int] = None,
    decode_len: Optional[int] = None,
) -> List[Dict[str, Any]]:
    if not _HAS_CV_READER:
        raise RuntimeError("cv_reader not available")

    frame_ids = [int(x) for x in list(frame_ids)] if frame_ids is not None else []
    if len(frame_ids) == 0:
        return []

    pos_map: Dict[int, List[int]] = {}
    for i, fid in enumerate(frame_ids):
        pos_map.setdefault(int(fid), []).append(i)

    out: List[Any] = [None] * len(frame_ids)
    max_fid = max(frame_ids)

    def _all_done() -> bool:
        for q in pos_map.values():
            if q:
                return False
        return True

    def _cb(d: Dict[str, Any]):
        idx = int(d.get("frame_idx", -1))
        if idx in pos_map and pos_map[idx]:
            j = pos_map[idx].pop(0)
            mv = np.asarray(d["motion_vector"])
            if with_residual:
                if "residual_y" in d:
                    ry = np.asarray(d["residual_y"])
                else:
                    ry = _residual_y_generic(np.asarray(d["residual"]))
                out[j] = {"frame_idx": idx, "motion_vector": mv, "residual_y": ry}
            else:
                out[j] = {"frame_idx": idx, "motion_vector": mv}
        return (not _all_done())

    if hasattr(cv_api, "read_video_cb"):
        without_residual = 0 if with_residual else 1
        use_seek = (seek_to_frame is not None) and (int(seek_to_frame) >= 0)
        if use_seek:
            st = int(seek_to_frame)
            if decode_len is None:
                max_frames = int(max(0, max_fid - st) + 1)
            else:
                max_frames = int(max(1, int(decode_len)))
            try:
                cv_api.read_video_cb(
                    str(video_path),
                    _cb,
                    int(without_residual),
                    int(max_frames),
                    frame_ids,
                    int(st),
                    int(max_frames),
                )
            except TypeError:
                legacy_max_frames = int(max_fid) + 1
                cv_api.read_video_cb(str(video_path), _cb, int(without_residual), int(legacy_max_frames), frame_ids)
        else:
            max_frames = int(max_fid) + 1
            cv_api.read_video_cb(str(video_path), _cb, int(without_residual), int(max_frames), frame_ids)
    else:
        frames = cv_api.read_video(str(video_path), 0, -1)
        if not isinstance(frames, (list, tuple)):
            frames = list(frames)
        for i, fid in enumerate(frame_ids):
            fid2 = max(0, min(int(fid), len(frames) - 1))
            fr = frames[fid2]
            mv = np.asarray(fr["motion_vector"])
            if with_residual:
                if "residual_y" in fr:
                    ry = np.asarray(fr["residual_y"])
                else:
                    ry = _residual_y_generic(np.asarray(fr["residual"]))
                out[i] = {"frame_idx": fid2, "motion_vector": mv, "residual_y": ry}
            else:
                out[i] = {"frame_idx": fid2, "motion_vector": mv}

    last = None
    for i in range(len(out)):
        if out[i] is None:
            if last is not None:
                out[i] = last
        else:
            last = out[i]
    if last is not None:
        for i in range(len(out)):
            if out[i] is None:
                out[i] = last
    return out  # type: ignore


def _decode_frames_by_ids_square576_bgr(video_path: str, frame_ids: List[int], out_size: int = 576) -> List[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_sq: List[np.ndarray] = []
    last_good = None
    last_good_fid: Optional[int] = None

    for fid in frame_ids:
        fid_req = int(fid)
        fid_used = int(fid_req)
        if total_frames > 0:
            fid_used = max(0, min(fid_used, total_frames - 1))

        cap.set(cv2.CAP_PROP_POS_FRAMES, fid_used)
        ret, frame = cap.read()

        if (not ret) or (frame is None):
            dec = int(fid_used)
            frame = None
            while dec > 0:
                dec -= 1
                cap.set(cv2.CAP_PROP_POS_FRAMES, dec)
                r2, f2 = cap.read()
                if r2 and (f2 is not None):
                    frame = f2
                    fid_used = int(dec)
                    break

        if frame is None:
            if last_good is None:
                cap.release()
                raise RuntimeError(f"Failed to decode any frame around fid={fid_req} from {video_path}")
            frame = last_good
            if last_good_fid is not None:
                fid_used = int(last_good_fid)
        else:
            last_good = frame
            last_good_fid = int(fid_used)

        frames_sq.append(_preprocess_bgr_to_square(frame, out_size=int(out_size)))

    cap.release()
    return frames_sq


def _build_mosaics_time_spatial_np(
    frames_sq: List[np.ndarray],
    thw_packed: np.ndarray,
    num_images: int = 8,
    square_size: int = 576,
    patch_size: int = 16,
) -> List[np.ndarray]:
    """Return list of BGR uint8 mosaics (not PIL), for saving to jpg."""
    p = int(patch_size)
    hb, wb = int(square_size) // p, int(square_size) // p
    S_img = hb * wb

    thw_packed = np.asarray(thw_packed, dtype=np.int32)
    if thw_packed.shape[0] != int(num_images) * int(S_img):
        raise RuntimeError(f"thw_packed rows={thw_packed.shape[0]} != num_images*S_img ({num_images}*{S_img})")

    mosaics: List[np.ndarray] = []
    for i in range(int(num_images)):
        canvas = np.zeros((int(square_size), int(square_size), 3), dtype=np.uint8)
        chunk = thw_packed[i * S_img:(i + 1) * S_img]

        for j, (t, h, w) in enumerate(chunk.tolist()):
            rr = int(j) // int(wb)
            cc = int(j) % int(wb)
            ys = rr * int(p)
            xs = cc * int(p)

            y0 = int(h) * int(p)
            x0 = int(w) * int(p)
            patch_bgr = frames_sq[int(t)][y0:y0 + int(p), x0:x0 + int(p)]
            canvas[ys:ys + int(p), xs:xs + int(p)] = patch_bgr

        mosaics.append(canvas)
    return mosaics


def _compute_visible_indices_pack_topk_time_spatial(
    video_path: str,
    seq_len_frames: int,
    num_images: int,
    square_size: int,
    patch_size: int,
    mv_unit_div: float,
    mv_pct: float,
    res_pct: float,
    w_mv: float,
    w_res: float,
    mv_compensate: str,
    res_use_grad: bool,
    window_len: int = 0,
    window_pick: str = "none",
    keep_mid_full_frame: bool = False,
    mid_full_frame_index: int = -1,
    topk_mode: str = "global",
    time_bucket_count: int = -1,
) -> Tuple[List[int], np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Same as online:
    - uniform sample seq_len_frames within full range or optional window
    - mandatory keep full first frame patches
      - optional keep a full middle frame patches (as a whole mosaic image)
      - rest:
        - global: global topk on valid patches of remaining frames
        - time_bucket: split remaining time indices into buckets and pick topk per bucket (1 mosaic per bucket)
    - pack into num_images * 1296 thw in time_spatial dense order
    """
    if not _HAS_CV_READER:
        raise RuntimeError("cv_reader not available")

    T_all = _get_total_frames_cv2(video_path)
    if T_all <= 0:
        T_all = int(seq_len_frames)

    fps = _get_fps_cv2(video_path)

    if int(window_len) > 0 and int(T_all) > int(window_len):
        start_offset, win_dbg = _choose_window_start_offset(
            video_path=str(video_path),
            total_frames=int(T_all),
            fps=float(fps),
            window_len=int(window_len),
            pick_mode=str(window_pick),
        )
        end_offset = min(int(T_all) - 1, int(start_offset) + int(window_len) - 1)
    else:
        win_dbg = {"pick": "none", "window_len": int(window_len), "total_frames": int(T_all), "fps": float(fps)}
        start_offset = 0
        end_offset = int(T_all) - 1

    frame_ids = _sample_frame_ids_uniform_range(int(start_offset), int(end_offset), int(seq_len_frames))
    if len(frame_ids) != int(seq_len_frames):
        frame_ids = (frame_ids + [frame_ids[-1]] * int(seq_len_frames))[: int(seq_len_frames)] if frame_ids else [0] * int(seq_len_frames)

    seek_to = int(start_offset) if int(start_offset) > 0 else None
    decode_span = int(end_offset - start_offset + 1) if int(end_offset) >= int(start_offset) else None

    items = _cv_reader_fetch_mvres_by_frame_ids(
        video_path,
        frame_ids,
        with_residual=True,
        seek_to_frame=seek_to,
        decode_len=decode_span,
    )
    if not isinstance(items, (list, tuple)) or len(items) == 0:
        raise RuntimeError(f"cv_reader callback returned empty: {video_path}")

    # Infer H/W
    H0, W0 = 0, 0
    try:
        cap = cv2.VideoCapture(str(video_path))
        if cap.isOpened():
            W0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            H0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
    except Exception:
        pass
    if (H0 <= 0 or W0 <= 0) and (items[0] is not None) and ("residual_y" in items[0]):
        ry0 = np.asarray(items[0]["residual_y"])
        if ry0.ndim >= 2:
            H0, W0 = int(ry0.shape[0]), int(ry0.shape[1])

    fused_list = []
    pad_info_ref = None

    for t in range(int(seq_len_frames)):
        it = items[int(t)]
        mv = np.asarray(it["motion_vector"])
        res_y = np.asarray(it["residual_y"])

        res_norm = _residual_energy_norm(res_y, pct=float(res_pct), use_grad=bool(res_use_grad))
        mv_norm = _mv_energy_norm(
            mv,
            H=int(H0),
            W=int(W0),
            mv_unit_div=float(mv_unit_div),
            pct=float(mv_pct),
            compensate=str(mv_compensate),
        )

        denom = float(w_mv + w_res) if (w_mv + w_res) != 0 else 1.0
        fused_hw = (float(w_mv) * mv_norm + float(w_res) * res_norm) / denom
        fused_hw = np.clip(fused_hw, 0.0, 1.0).astype(np.float32)

        fused_sq, info = _resize_longer_pad_square_map(fused_hw, out_size=int(square_size))
        if pad_info_ref is None:
            pad_info_ref = info
        fused_list.append(fused_sq)

    fused_vol = np.stack(fused_list, axis=0).astype(np.float32)  # (T,576,576)

    p = int(patch_size)
    hb, wb = int(square_size) // p, int(square_size) // p
    S_total = hb * wb  # 1296

    valid_mask = _valid_patch_mask_from_pad(
        out_size=int(square_size),
        patch=int(patch_size),
        pad_top=int(pad_info_ref["pad_top"]),
        pad_left=int(pad_info_ref["pad_left"]),
        Hn=int(pad_info_ref["Hn"]),
        Wn=int(pad_info_ref["Wn"]),
    )
    S_valid = int(valid_mask.sum())

    fused_t = torch.from_numpy(fused_vol)
    scores = fused_t.reshape(int(seq_len_frames), hb, p, wb, p).sum(dim=(2, 4)).reshape(int(seq_len_frames), S_total)
    mask_flat = torch.from_numpy(valid_mask.reshape(-1))
    scores[:, ~mask_flat] = -1e6

    target = int(num_images) * int(S_total)

    # mandatory first frame
    hh0 = np.repeat(np.arange(hb, dtype=np.int32), wb)
    ww0 = np.tile(np.arange(wb, dtype=np.int32), hb)
    tt0 = np.zeros((S_total,), dtype=np.int32)
    mandatory_thw = np.stack([tt0, hh0, ww0], axis=1).astype(np.int32)  # (1296,3)

    # optional mandatory middle frame (as a whole mosaic image)
    keep_mid = bool(keep_mid_full_frame) and int(num_images) >= 2 and int(seq_len_frames) >= 2
    mid_t = int(mid_full_frame_index)
    if keep_mid:
        if mid_t < 0:
            mid_t = int(seq_len_frames) // 2
        mid_t = max(0, min(int(seq_len_frames) - 1, int(mid_t)))
        # avoid duplicating the first frame
        if mid_t == 0:
            keep_mid = False

    if keep_mid:
        tt_mid = np.full((S_total,), int(mid_t), dtype=np.int32)
        mandatory_mid_thw = np.stack([tt_mid, hh0, ww0], axis=1).astype(np.int32)  # (1296,3)
    else:
        mid_t = -1
        mandatory_mid_thw = np.zeros((0, 3), dtype=np.int32)

    if target <= 0:
        thw = np.zeros((0, 3), dtype=np.int32)
        visible_indices = np.zeros((0,), dtype=np.int64)
    else:
        K_rem = int(target) - int(S_total) - (int(S_total) if keep_mid else 0)
        if K_rem <= 0:
            thw = np.concatenate([mandatory_thw, mandatory_mid_thw], axis=0)[:target]
        else:
            scores_rem = scores.clone()
            scores_rem[0, :] = -1e6
            if keep_mid and mid_t >= 0:
                scores_rem[int(mid_t), :] = -1e6

            # Decide mode
            mode = str(topk_mode or "global").lower().strip()
            if mode not in ("global", "time_bucket"):
                mode = "global"

            rest_thw = np.zeros((0, 3), dtype=np.int32)

            if mode == "global":
                avail_rem = max(0, (int(seq_len_frames) - 1 - (1 if keep_mid else 0)) * int(S_valid))
                k_take = min(int(K_rem), int(avail_rem))
                if k_take > 0:
                    flat = scores_rem.reshape(-1)
                    _, topidx = torch.topk(flat, k=int(k_take), largest=True, sorted=True)
                    topidx_np = topidx.cpu().numpy().astype(np.int64)

                    t = (topidx_np // S_total).astype(np.int32)
                    s = (topidx_np % S_total).astype(np.int32)
                    h = (s // wb).astype(np.int32)
                    w = (s % wb).astype(np.int32)
                    rest_thw = np.stack([t, h, w], axis=1).astype(np.int32)

            else:
                # time_bucket: fill mosaics in temporal order (1 bucket ~= 1 mosaic)
                num_full = 1 + (1 if keep_mid else 0)
                num_rest_images = max(0, int(num_images) - int(num_full))

                # user override; -1 means derive from remaining mosaics
                bucket_cnt = int(time_bucket_count)
                if bucket_cnt <= 0:
                    bucket_cnt = int(num_rest_images)
                bucket_cnt = max(1, min(bucket_cnt, max(1, int(seq_len_frames) - int(num_full))))

                # build remaining time indices (exclude mandatory frames)
                excluded = {0}
                if keep_mid and mid_t >= 0:
                    excluded.add(int(mid_t))
                ts_all = [t for t in range(int(seq_len_frames)) if int(t) not in excluded]
                if len(ts_all) == 0:
                    ts_all = [0]

                # split into contiguous buckets
                buckets = np.array_split(np.asarray(ts_all, dtype=np.int32), int(bucket_cnt))

                per_bucket_take = int(S_total)  # one mosaic per bucket
                out_list: List[np.ndarray] = []
                for b in buckets:
                    if b.size == 0:
                        continue
                    # gather scores for this bucket only
                    sb = scores_rem[b.tolist(), :]  # (Tb, S_total)
                    avail_b = max(0, int(b.size) * int(S_valid))
                    k_take_b = min(int(per_bucket_take), int(avail_b))
                    if k_take_b <= 0:
                        continue

                    flat_b = sb.reshape(-1)
                    _, topidx_b = torch.topk(flat_b, k=int(k_take_b), largest=True, sorted=True)
                    topidx_np = topidx_b.cpu().numpy().astype(np.int64)

                    t_local = (topidx_np // S_total).astype(np.int32)
                    s = (topidx_np % S_total).astype(np.int32)
                    t = b[t_local].astype(np.int32)
                    h = (s // wb).astype(np.int32)
                    w = (s % wb).astype(np.int32)
                    out_list.append(np.stack([t, h, w], axis=1).astype(np.int32))

                if out_list:
                    rest_thw = np.concatenate(out_list, axis=0)

                # Trim/pad to K_rem later by force-exact logic; but try to cap at K_rem here
                if rest_thw.shape[0] > int(K_rem):
                    rest_thw = rest_thw[: int(K_rem)]

            thw = np.concatenate([mandatory_thw, mandatory_mid_thw, rest_thw], axis=0)

        # force exact target
        if thw.shape[0] < target:
            if thw.shape[0] == 0:
                thw = np.zeros((target, 3), dtype=np.int32)
            else:
                pad_n = target - thw.shape[0]
                thw = np.concatenate([thw, np.repeat(thw[-1:], pad_n, axis=0)], axis=0)
        elif thw.shape[0] > target:
            thw = thw[:target]

        thw64 = thw.astype(np.int64)
        visible_indices = thw64[:, 0] * int(S_total) + thw64[:, 1] * int(wb) + thw64[:, 2]
        visible_indices = visible_indices.astype(np.int64)

    debug = {
        "seq_len_frames": int(seq_len_frames),
        "num_images": int(num_images),
        "square_size": int(square_size),
        "patch_size": int(patch_size),
        "hb": int(hb),
        "wb": int(wb),
        "S_total": int(S_total),
        "S_valid": int(S_valid),
        "K_total": int(thw.shape[0]),
        "keep_first_full": True,
        "keep_mid_full": bool(keep_mid),
        "mid_full_t": int(mid_t),
        "topk_mode": str(topk_mode),
        "time_bucket_count": int(time_bucket_count),
        "vi_min": int(visible_indices.min()) if visible_indices.size > 0 else -1,
        "vi_max": int(visible_indices.max()) if visible_indices.size > 0 else -1,
        "T_all": int(T_all),
        "fps": float(fps),
        "window_pick": str(window_pick),
        "window_len": int(window_len),
        "start_offset": int(start_offset),
        "end_offset": int(end_offset),
        "pkt_size_sum": win_dbg.get("pkt_size_sum", None),
        "best": win_dbg.get("best", None),
        "seek_to": seek_to,
        "decode_span": decode_span,
        "frame_ids_unique": int(len(set(int(x) for x in frame_ids))),
    }
    return frame_ids, thw.astype(np.int32), visible_indices.astype(np.int64), debug


# -------------------- I/O + worker --------------------


@dataclass
class Job:
    task: str
    split: str
    doc_id: int
    n: int
    video: str
    key: str


# ------- multiprocessing helpers (must be top-level for spawn/pickle) -------
_GLOBAL_ARGS: Optional[argparse.Namespace] = None

def _init_worker(args: argparse.Namespace) -> None:
    """Initializer for multiprocessing workers (spawn-safe).

    Also reduces CPU oversubscription from OpenCV / BLAS / OpenMP inside each worker.
    """
    # Reduce oversubscription in each worker process
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    try:
        cv2.setNumThreads(int(getattr(args, "opencv_threads", 1)))
        cv2.setUseOptimized(True)
        try:
            cv2.ocl.setUseOpenCL(False)
        except Exception:
            pass
    except Exception:
        pass

    global _GLOBAL_ARGS
    _GLOBAL_ARGS = args

def _worker_process_one(job: Job) -> Tuple[str, str]:
    """Worker entry: process a single job using globally-initialized args."""
    if _GLOBAL_ARGS is None:
        raise RuntimeError("Worker args not initialized. Did you pass initializer=_init_worker?")
    return _process_one(job, _GLOBAL_ARGS)


def _atomic_write_json(path: Path, obj: Dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _safe_np_save(path: Path, arr: np.ndarray) -> None:
    """Atomically save a numpy array without NumPy auto-appending extensions."""
    tmp = Path(str(path) + ".tmp")
    with tmp.open("wb") as f:
        np.save(f, arr, allow_pickle=False)
    os.replace(str(tmp), str(path))


def _is_done(asset_dir: Path, num_images: int) -> bool:
    if not asset_dir.exists():
        return False
    need = [
        asset_dir / "meta.json",
        asset_dir / "frame_ids.npy",
        asset_dir / "visible_indices.npy",
        asset_dir / "positions_thw.npy",
    ]
    for p in need:
        if not p.exists():
            return False
    for i in range(num_images):
        if not (asset_dir / f"mosaic_{i:03d}.jpg").exists():
            return False
    return True


def _process_one(job: Job, args: argparse.Namespace) -> Tuple[str, str]:
    """
    Return (status, message).
    status in {"ok","skip","fail"}.
    """
    video_path = job.video
    key = job.key
    out_dir = Path(args.out_root) / "assets" / key
    out_dir.mkdir(parents=True, exist_ok=True)

    if (not args.overwrite) and _is_done(out_dir, args.num_images):
        return "skip", f"{key} already done"

    if not os.path.exists(video_path):
        return "fail", f"{key} missing video: {video_path}"

    t0 = time.time()

    try:
        frame_ids, thw_np, vi_np, dbg = _compute_visible_indices_pack_topk_time_spatial(
            video_path=video_path,
            seq_len_frames=args.seq_len_frames,
            num_images=args.num_images,
            square_size=args.square_size,
            patch_size=args.patch_size,
            mv_unit_div=args.mv_unit_div,
            mv_pct=args.mv_pct,
            res_pct=args.res_pct,
            w_mv=args.w_mv,
            w_res=args.w_res,
            mv_compensate=args.mv_compensate,
            res_use_grad=args.res_use_grad,
            window_len=args.window_len,
            window_pick=args.window_pick,
            keep_mid_full_frame=args.keep_mid_full_frame,
            mid_full_frame_index=args.mid_full_frame_index,
            topk_mode=args.topk_mode,
            time_bucket_count=args.time_bucket_count,
        )

        frames_sq = _decode_frames_by_ids_square576_bgr(
            video_path=video_path,
            frame_ids=frame_ids,
            out_size=args.square_size,
        )

        mosaics = _build_mosaics_time_spatial_np(
            frames_sq=frames_sq,
            thw_packed=thw_np,
            num_images=args.num_images,
            square_size=args.square_size,
            patch_size=args.patch_size,
        )

        # write jpg
        for i, img_bgr in enumerate(mosaics):
            p = out_dir / f"mosaic_{i:03d}.jpg"
            # IMPORTANT: temp path must end with .jpg so OpenCV selects the JPEG writer
            tmp = out_dir / f"mosaic_{i:03d}.tmp.jpg"
            cv2.imwrite(str(tmp), img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpg_quality)])
            os.replace(str(tmp), str(p))

        # write npy
        _safe_np_save(out_dir / "positions_thw.npy", thw_np.astype(np.int32))
        _safe_np_save(out_dir / "visible_indices.npy", vi_np.astype(np.int64))
        _safe_np_save(out_dir / "frame_ids.npy", np.asarray(frame_ids, dtype=np.int32))

        meta = {
            "version": "llava_codec_offline_precompute_v1",
            "task": job.task,
            "split": job.split,
            "doc_id": int(job.doc_id),
            "n": int(job.n),
            "video": video_path,
            "key": key,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "cfg": {
                "seq_len_frames": int(args.seq_len_frames),
                "num_images": int(args.num_images),
                "square_size": int(args.square_size),
                "patch_size": int(args.patch_size),
                "mv_unit_div": float(args.mv_unit_div),
                "mv_pct": float(args.mv_pct),
                "res_pct": float(args.res_pct),
                "w_mv": float(args.w_mv),
                "w_res": float(args.w_res),
                "mv_compensate": str(args.mv_compensate),
                "res_use_grad": bool(args.res_use_grad),
                "window_len": int(args.window_len),
                "window_pick": str(args.window_pick),
                "keep_mid_full_frame": bool(args.keep_mid_full_frame),
                "mid_full_frame_index": int(args.mid_full_frame_index),
                "topk_mode": str(args.topk_mode),
                "time_bucket_count": int(args.time_bucket_count),
            },
            "debug": dbg,
            "stats": {
                "elapsed_sec": float(time.time() - t0),
            },
        }
        _atomic_write_json(out_dir / "meta.json", meta)

        return "ok", f"{key} ok elapsed={meta['stats']['elapsed_sec']:.3f}s"

    except Exception as e:
        # write an error marker for visibility (optional)
        try:
            _atomic_write_json(out_dir / "error.json", {"error": repr(e), "video": video_path, "key": key})
        except Exception:
            pass
        return "fail", f"{key} fail err={repr(e)}"


def _load_jobs(jsonl_path: str, only_exists: bool = True) -> List[Job]:
    jobs: List[Job] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if only_exists and (not d.get("exists", True)):
                continue
            jobs.append(
                Job(
                    task=str(d.get("task", "")),
                    split=str(d.get("split", "")),
                    doc_id=int(d.get("doc_id", -1)),
                    n=int(d.get("n", 0)),
                    video=str(d["video"]),
                    key=str(d.get("key") or ""),
                )
            )
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True, help="input jsonl dumped from tasks (each line contains video,key,exists)")
    ap.add_argument("--out_root", required=True, help="output root. will write out_root/assets/<key>/...")
    ap.add_argument("--num_workers", type=int, default=8, help="multiprocessing workers")
    ap.add_argument("--chunksize", type=int, default=4, help="Pool imap chunksize (larger reduces IPC overhead)")
    ap.add_argument("--opencv_threads", type=int, default=1, help="OpenCV threads per worker (avoid oversubscription)")
    ap.add_argument("--overwrite", action="store_true", help="overwrite existing assets")
    ap.add_argument("--only_exists", action="store_true", help="skip records with exists=false (default true)", default=True)
    ap.add_argument("--max_samples", type=int, default=0, help="cap number of videos (0=all)")
    ap.add_argument("--num_shards", type=int, default=1, help="total shards")
    ap.add_argument("--shard_id", type=int, default=0, help="shard index [0..num_shards-1]")
    ap.add_argument("--log_every", type=int, default=20)

    # configs aligned with online env knobs
    ap.add_argument("--seq_len_frames", type=int, default=64)
    ap.add_argument("--num_images", type=int, default=8)
    ap.add_argument("--square_size", type=int, default=576)
    ap.add_argument("--patch_size", type=int, default=16)
    ap.add_argument(
        "--keep_mid_full_frame",
        action="store_true",
        help="keep an additional full middle frame as one whole mosaic image (mosaic_001).",
    )
    ap.add_argument(
        "--mid_full_frame_index",
        type=int,
        default=-1,
        help="which t (0..seq_len_frames-1) to use as the full middle frame; -1 means auto (seq_len_frames//2).",
    )
    ap.add_argument(
        "--topk_mode",
        type=str,
        default="global",
        choices=["global", "time_bucket"],
        help="how to select topk patches for non-mandatory frames: global=global topk; time_bucket=split time into buckets and pick per bucket (1 mosaic per bucket).",
    )
    ap.add_argument(
        "--time_bucket_count",
        type=int,
        default=-1,
        help="only for --topk_mode time_bucket: number of time buckets; -1 means derive from remaining mosaics (num_images - num_full_frames).",
    )

    ap.add_argument("--mv_unit_div", type=float, default=4.0)
    ap.add_argument("--mv_pct", type=float, default=95.0)
    ap.add_argument("--res_pct", type=float, default=95.0)
    ap.add_argument("--w_mv", type=float, default=1.0)
    ap.add_argument("--w_res", type=float, default=1.0)
    ap.add_argument("--mv_compensate", type=str, default="median", choices=["median", "mean", "none"])
    ap.add_argument("--res_use_grad", action="store_true")

    ap.add_argument("--window_len", type=int, default=0)
    ap.add_argument("--window_pick", type=str, default="none", choices=["none", "begin", "mid", "end", "energy3"])

    ap.add_argument("--jpg_quality", type=int, default=95)

    args = ap.parse_args()

    if not _HAS_CV_READER:
        print("ERROR: cv_reader is not available in this environment.", file=sys.stderr)
        sys.exit(2)

    # reduce OpenCV oversubscription on CPU nodes (main process)
    try:
        cv2.setNumThreads(int(args.opencv_threads))
        cv2.setUseOptimized(True)
        try:
            cv2.ocl.setUseOpenCL(False)
        except Exception:
            pass
    except Exception:
        pass

    jobs = _load_jobs(args.jsonl, only_exists=args.only_exists)
    if args.num_shards > 1:
        jobs = [j for i, j in enumerate(jobs) if (i % args.num_shards) == args.shard_id]

    if args.max_samples and args.max_samples > 0:
        jobs = jobs[: args.max_samples]

    Path(args.out_root).mkdir(parents=True, exist_ok=True)
    print(f"[info] total_jobs={len(jobs)} out_root={args.out_root} workers={args.num_workers} shard={args.shard_id}/{args.num_shards}")

    # multiprocessing
    if args.num_workers <= 1:
        ok = skip = fail = 0
        try:
            cv2.setNumThreads(int(args.opencv_threads))
            cv2.setUseOptimized(True)
            try:
                cv2.ocl.setUseOpenCL(False)
            except Exception:
                pass
        except Exception:
            pass
        for i, job in enumerate(jobs):
            st, msg = _process_one(job, args)
            if st == "ok":
                ok += 1
            elif st == "skip":
                skip += 1
            else:
                fail += 1
            if (i + 1) % args.log_every == 0 or (i + 1) == len(jobs):
                print(f"[prog] {i+1}/{len(jobs)} ok={ok} skip={skip} fail={fail} last={msg}")
        print(f"[done] ok={ok} skip={skip} fail={fail}")
        return

    import multiprocessing as mp
    ctx = mp.get_context("spawn")


    ok = skip = fail = 0
    t0 = time.time()

    with ctx.Pool(
        processes=int(args.num_workers),
        initializer=_init_worker,
        initargs=(args,),
    ) as pool:
        for i, (st, msg) in enumerate(pool.imap_unordered(_worker_process_one, jobs, chunksize=int(max(1, args.chunksize)))):
            if st == "ok":
                ok += 1
            elif st == "skip":
                skip += 1
            else:
                fail += 1

            if (i + 1) % args.log_every == 0 or (i + 1) == len(jobs):
                elapsed = time.time() - t0
                rate = (i + 1) / max(elapsed, 1e-6)
                print(f"[prog] {i+1}/{len(jobs)} ok={ok} skip={skip} fail={fail} rate={rate:.2f}/s last={msg}")

    print(f"[done] ok={ok} skip={skip} fail={fail} total={len(jobs)} elapsed={time.time()-t0:.2f}s")


if __name__ == "__main__":
    main()