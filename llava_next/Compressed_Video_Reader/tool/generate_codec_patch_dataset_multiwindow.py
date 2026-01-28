#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate a codec-aware patch dataset with:
  - multi-window energy selection (ffprobe pkt_size sum)
  - mv/res based scoring (cv_reader preferred)
  - output patches + patch_position in a strict ordering

Outputs per video key directory:
  - canvas_*.jpg         packed canvases (RGB) after padding-to-32 and 2x2-block ordering
  - patch_position.npy   int32 (N, 3)  [img_idx, patch_h, patch_w] aligned 1-1 with flattened token order
  - img_ptr.npy          int32 (num_images+1,) prefix sum boundaries per canvas (each canvas has hb*wb patches)
  - frame_ids.npy        int32 (seq_len,) sampled original frame ids (length == seq_len)
  - src_patch_position.npy int32 (N, 3) [frame_id, src_ph, src_pw] source location for each token (pad tokens are -1)
  - src_frame_ids.npy    int32 (N,) source frame id for each token (pad tokens are -1)
  - meta.json            misc metadata (pad sizes, window scores, etc.)
  - assets_index record  (optional; written by main process when --assets_index is set)

Constraints satisfied:
  1) P-frame per-frame patch count divisible by 4  (select 2x2 blocks)
  2) multiple GOPs allowed (sampling across video)
  3) patch_position aligns 1-1 with flatten order
  4) I-frame flatten order: each 2x2 block => consecutive 4 tokens
"""

import os
import re
import json
import math
import time
import hashlib
import argparse
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import cv2

# Pillow for reliable JPEG writing (OpenCV builds sometimes lack jpg writers)
try:
    from PIL import Image  # type: ignore
    HAS_PIL = True
except Exception:
    Image = None
    HAS_PIL = False

# ----- optional cv_reader -----
try:
    from cv_reader import api as cv_api  # type: ignore
    HAS_CV_READER = True
except Exception:
    cv_api = None
    HAS_CV_READER = False


# -----------------------------
# utilities
# -----------------------------

def sha1_8(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:8]



def load_jsonl(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def iter_jsonl(path: str):
    """Stream jsonl items without loading the whole file into memory."""
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield ln, json.loads(line)
            except Exception:
                # skip malformed lines
                continue


# -----------------------------
# jsonl schema adapters
# -----------------------------

_VIDEO_PATH_RE = re.compile(r"(/[^\s\"']+\.(?:mp4|mkv|webm|avi|mov))", re.IGNORECASE)


def extract_video_path_from_item(it: Dict[str, Any]) -> Optional[str]:
    """Best-effort extraction of a local video path from various jsonl schemas.

    Supported schemas:
      1) Simple: {"video": "/abs/path.mp4", "key": "...", "exists": true}
      2) OpenAI-like results: {"custom_id": "prefix-/abs/path.mp4", "error": null, "response": {...}}

    Returns absolute path string or None.
    """
    # 1) canonical keys
    for k in ("video", "path", "video_path"):
        v = it.get(k)
        if isinstance(v, str) and v:
            return v

    # 2) custom_id sometimes stores the path
    cid = it.get("custom_id")
    if isinstance(cid, str) and cid:
        # common pattern: "prefix-/abs/path.mp4" (split once)
        if "-/" in cid:
            maybe = cid.split("-", 1)[1]
            if isinstance(maybe, str) and maybe.startswith("/"):
                return maybe
        # fallback: regex search for an absolute path ending with known extension
        m = _VIDEO_PATH_RE.search(cid)
        if m:
            return m.group(1)

    # 3) sometimes nested in response (rare)
    resp = it.get("response")
    if isinstance(resp, dict):
        for kk in ("video", "path", "video_path"):
            v = resp.get(kk)
            if isinstance(v, str) and v:
                return v

    return None


def infer_key_from_video(video_path: str, it: Dict[str, Any]) -> str:
    """Infer a stable key for output directory."""
    k = it.get("key")
    if isinstance(k, str) and k:
        return k

    stem = Path(video_path).stem
    vid = it.get("id")
    if isinstance(vid, str) and vid:
        return f"{stem}__{sha1_8(video_path)}__{vid[:8]}"

    return f"{stem}__{sha1_8(video_path)}"

def extract_caption_from_item(it: Dict[str, Any]) -> str:
    """Extract caption / assistant text from OpenAI-like results jsonl.
    Expected schema:
      it["response"]["body"]["choices"][0]["message"]["content"]
    """
    resp = it.get("response")
    if not isinstance(resp, dict):
        return ""
    body = resp.get("body")
    if not isinstance(body, dict):
        return ""
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    ch0 = choices[0]
    if not isinstance(ch0, dict):
        return ""
    msg = ch0.get("message")
    if not isinstance(msg, dict):
        return ""
    content = msg.get("content")
    return content if isinstance(content, str) else ""

# -----------------------------
# Mirror key helper
# -----------------------------
from typing import Optional
def mirror_key_from_video(video_path: str, mirror_src_root: str, strip_ext: bool = True) -> Optional[str]:
    """Map an absolute video path to a relative directory key under `mirror_src_root`.

    Example:
      video_path=/ov2/dataset_mid_source/batch1_5millions/a/b/c.mp4
      mirror_src_root=/ov2/dataset_mid_source/batch1_5millions
      -> key=a/b/c  (if strip_ext)

    Returns None if video_path is not under mirror_src_root.
    """
    try:
        vp = Path(video_path)
        root = Path(mirror_src_root)
        # Resolve to handle redundant slashes; do not require the path to exist.
        vp_res = vp.resolve()
        root_res = root.resolve()
        try:
            rel = vp_res.relative_to(root_res)
        except Exception:
            # Fallback to string-based relpath (works even if resolve() behaves unexpectedly)
            vp_s = str(vp)
            root_s = str(root)
            if not vp_s.startswith(root_s.rstrip("/") + "/") and vp_s != root_s:
                return None
            rel = Path(os.path.relpath(vp_s, root_s))

        if strip_ext:
            rel = rel.with_suffix("")

        # Ensure a clean relative POSIX key (no leading slash)
        key = rel.as_posix().lstrip("/")
        return key if key else None
    except Exception:
        return None



def ensure_dir(p: str) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


# Helper to pack patches into full canvases in 2x2-block raster order
def pack_patches_to_canvases(
    patches: np.ndarray,
    hb: int,
    wb: int,
    patch: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pack patches into one or more full canvases.

    Packing order is raster over 2x2 blocks, and within each block the 4 patches
    are placed in the order: (0,0),(0,1),(1,0),(1,1). This guarantees that
    `image.reshape(-1)` will have consecutive 4 tokens corresponding to a 2x2 block.

    Returns:
      images_rgb: uint8 (num_images, H, W, 3)
      patch_position: int32 (N, 3) [img_idx, patch_h, patch_w] aligned 1-1 with patches
      img_ptr: int32 (num_images+1,) prefix-sum boundaries (each image has hb*wb patches)
    """
    hb = int(hb)
    wb = int(wb)
    p = int(patch)
    S_full = hb * wb
    if patches.size == 0:
        images = np.zeros((0, hb * p, wb * p, 3), dtype=np.uint8)
        patch_pos = np.zeros((0, 3), dtype=np.int32)
        img_ptr = np.zeros((1,), dtype=np.int32)
        return images, patch_pos, img_ptr

    assert patches.ndim == 4 and patches.shape[1] == p and patches.shape[2] == p and patches.shape[3] == 3
    assert patches.shape[0] % S_full == 0, f"patches must be multiple of S_full={S_full}, got {patches.shape[0]}"

    num_images = int(patches.shape[0] // S_full)
    H = hb * p
    W = wb * p
    images = np.zeros((num_images, H, W, 3), dtype=np.uint8)
    patch_pos = np.zeros((patches.shape[0], 3), dtype=np.int32)

    idx = 0
    for img_i in range(num_images):
        for bh in range(hb // 2):
            for bw in range(wb // 2):
                coords = block_to_4_patches(bh, bw)
                for (ph, pw) in coords:
                    y0 = int(ph) * p
                    x0 = int(pw) * p
                    images[img_i, y0:y0 + p, x0:x0 + p, :] = patches[idx]
                    patch_pos[idx, :] = (int(img_i), int(ph), int(pw))
                    idx += 1

    img_ptr = (np.arange(0, num_images + 1, dtype=np.int32) * int(S_full)).astype(np.int32)
    return images, patch_pos, img_ptr


# Helper to save canvases as JPG files using Pillow (prefer) or OpenCV fallback
def save_canvases_as_jpg(images_rgb: np.ndarray, out_dir: str, quality: int = 95) -> List[str]:
    """Save (num_images, H, W, 3) RGB uint8 canvases into JPEG files.

    Returns list of written filenames (basenames).
    """
    out: List[str] = []
    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    if images_rgb is None or images_rgb.size == 0:
        return out

    q = int(max(1, min(100, int(quality))))
    for i in range(int(images_rgb.shape[0])):
        fn = f"canvas_{i:03d}.jpg"
        fp = out_p / fn
        arr = images_rgb[i]

        # Prefer Pillow for JPEG reliability; fallback to OpenCV.
        if HAS_PIL and Image is not None:
            Image.fromarray(arr).save(str(fp), format="JPEG", quality=q, subsampling=0, optimize=True)
        else:
            bgr = arr[:, :, ::-1]
            ok = cv2.imwrite(str(fp), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), q])
            if not ok:
                raise RuntimeError("Failed to write JPEG. Please install pillow: pip install pillow")

        out.append(fn)

    return out



def get_total_frames_fps(video_path: str) -> Tuple[int, float, int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0, 0.0, 0, 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if not np.isfinite(fps):
        fps = 0.0
    return max(0, total), max(0.0, fps), h, w


# -----------------------------
# Patch budget auto helper
# -----------------------------
def auto_max_total_patches(
    S_full: int,
    total_frames: int,
    fps: float,
    cap_total: int = 30000,
) -> Tuple[int, Dict[str, Any]]:
    """Choose max_total_patches (multiple of S_full) under cap_total.

    Heuristic:
      - duration <= 6s  : short
      - duration >  6s  : long

    Resolution tier via S_full (patches per full canvas):
      ~1080p: ~8160, ~720p: ~3680, <=480p: <=~1500

    Additional constraint:
      - The number of packed canvases (num_images_final) will NOT exceed the
        video duration in seconds (floor), but is at least 1.
        Example: 5s video => at most 5 canvases.

    Returns: (max_total_patches, debug_dict)
    """
    S_full = int(max(1, S_full))
    cap_total = int(max(S_full, cap_total))
    fps_use = float(fps) if (fps and fps > 0) else 30.0
    duration_sec = float(total_frames) / float(fps_use) if total_frames > 0 else 0.0
    tier_time = "short" if duration_sec <= 6.0 else "long"

    # Hard time cap: num_images <= floor(duration_sec), but at least 1.
    # (max images not exceeding duration in seconds => use floor; e.g. 5.9s => 5 images max)
    max_images_by_time = max(1, int(duration_sec))

    # Resolution tier via S_full
    if S_full >= 7000:  # ~1080p
        target_num_images = 2 if tier_time == "short" else 3
        tier_res = "1080p_like"
    elif S_full >= 2500:  # ~720p
        target_num_images = 4 if tier_time == "short" else 8
        tier_res = "720p_like"
    else:  # small (320p/480p)
        target_num_images = 12 if tier_time == "short" else 24
        tier_res = "small_like"

    num_images_cap = max(1, cap_total // S_full)
    num_images_final = int(min(target_num_images, num_images_cap, max_images_by_time))

    max_total = int(num_images_final * S_full)
    dbg = {
        "cap_total": int(cap_total),
        "S_full": int(S_full),
        "fps_use": float(fps_use),
        "duration_sec": float(duration_sec),
        "tier_time": tier_time,
        "tier_res": tier_res,
        "target_num_images": int(target_num_images),
        "num_images_cap": int(num_images_cap),
        "max_images_by_time": int(max_images_by_time),
        "num_images_final": int(num_images_final),
        "max_total_patches": int(max_total),
    }
    return max_total, dbg


def ffprobe_sum_pkt_size(video_path: str, start_sec: float, dur_sec: float) -> int:
    """
    Sum packet sizes in [start_sec, start_sec + dur_sec] using ffprobe read_intervals.
    """
    try:
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
            if not line or line == "N/A":
                continue
            try:
                s += int(line)
            except Exception:
                continue
        return int(s)
    except Exception:
        return 0


def pick_windows_by_energy(
    video_path: str,
    total_frames: int,
    fps: float,
    window_len_frames: int,
    num_candidates: int,
    top_k: int,
) -> Tuple[List[Tuple[int, int, int]], Dict[str, Any]]:
    """
    Return list of selected windows: [(start_f, end_f, score_pkt_sum), ...]
    plus debug dict.
    """
    total_frames = int(total_frames)
    window_len_frames = int(window_len_frames)
    num_candidates = int(num_candidates)
    top_k = int(top_k)

    dbg: Dict[str, Any] = {
        "total_frames": total_frames,
        "fps": float(fps),
        "window_len_frames": window_len_frames,
        "num_candidates": num_candidates,
        "top_k": top_k,
        "candidates": [],
        "selected": [],
    }

    if total_frames <= 0:
        return [(0, 0, 0)], dbg

    if window_len_frames <= 0 or window_len_frames >= total_frames:
        # treat whole video as one window
        return [(0, total_frames - 1, 0)], dbg

    # candidate window starts evenly spaced in [0, total_frames - window_len]
    max_start = max(0, total_frames - window_len_frames)
    if num_candidates <= 1:
        starts = [0]
    else:
        starts = np.linspace(0, max_start, num_candidates, dtype=np.int32).tolist()

    fps_use = fps if fps and fps > 0 else 30.0
    dur_sec = float(window_len_frames) / float(fps_use)

    cand = []
    for st in starts:
        st = int(st)
        ed = int(min(total_frames - 1, st + window_len_frames - 1))
        start_sec = float(st) / float(fps_use)
        score = ffprobe_sum_pkt_size(video_path, start_sec, dur_sec)
        cand.append((st, ed, int(score)))
    cand_sorted = sorted(cand, key=lambda x: x[2], reverse=True)

    dbg["candidates"] = cand_sorted

    chosen = cand_sorted[: max(1, top_k)]
    dbg["selected"] = chosen
    return chosen, dbg


def allocate_frames_across_windows(
    windows: List[Tuple[int, int, int]],
    seq_len: int,
) -> List[int]:
    """
    Allocate seq_len frames across windows proportional to scores (fallback equal),
    sample uniformly within each window, then merge & pad/truncate to seq_len.
    """
    seq_len = int(seq_len)
    if seq_len <= 0:
        return []

    # if only one window
    if len(windows) == 1:
        st, ed, _ = windows[0]
        return np.linspace(st, ed, seq_len, dtype=np.int32).tolist()

    scores = np.array([max(0, w[2]) for w in windows], dtype=np.float64)
    if scores.sum() <= 0:
        # equal allocation
        weights = np.ones_like(scores) / float(len(scores))
    else:
        weights = scores / scores.sum()

    # initial per-window counts (at least 1)
    counts = np.maximum(1, np.floor(weights * seq_len).astype(int))
    # adjust to sum == seq_len
    while counts.sum() > seq_len:
        i = int(np.argmax(counts))
        if counts[i] > 1:
            counts[i] -= 1
        else:
            break
    while counts.sum() < seq_len:
        i = int(np.argmax(weights))
        counts[i] += 1

    frame_ids: List[int] = []
    for (st, ed, _), c in zip(windows, counts.tolist()):
        if c <= 0:
            continue
        frame_ids.extend(np.linspace(int(st), int(ed), int(c), dtype=np.int32).tolist())

    frame_ids = sorted([int(x) for x in frame_ids])

    # dedup but keep order (dedup can reduce length)
    dedup = []
    last = None
    for x in frame_ids:
        if last is None or x != last:
            dedup.append(x)
        last = x
    frame_ids = dedup

    # pad/truncate to seq_len
    if len(frame_ids) == 0:
        frame_ids = [0] * seq_len
    if len(frame_ids) < seq_len:
        frame_ids = (frame_ids + [frame_ids[-1]] * (seq_len - len(frame_ids)))[:seq_len]
    else:
        frame_ids = frame_ids[:seq_len]
    return frame_ids


def decode_frame_bgr_at(video_path: str, frame_id: int, backsearch_max: int = 32) -> Optional[np.ndarray]:
    """
    Decode one frame (best-effort).

    Notes:
      - Some corrupted streams can make backward seeking extremely slow if we
        decrement frame-by-frame for a long time. We cap the backsearch.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fid = int(frame_id)
    if total > 0:
        fid = max(0, min(fid, total - 1))

    cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
    ok, frame = cap.read()

    if (not ok) or frame is None:
        # bounded backward search
        dec = fid
        tries = 0
        bs = int(max(0, backsearch_max))
        while dec > 0 and tries < bs:
            dec -= 1
            tries += 1
            cap.set(cv2.CAP_PROP_POS_FRAMES, dec)
            ok2, f2 = cap.read()
            if ok2 and f2 is not None:
                frame = f2
                break

    cap.release()
    return frame


def decode_frames_bgr(video_path: str, frame_ids: List[int], backsearch_max: int = 32) -> List[np.ndarray]:
    """
    Decode multiple frames using a single VideoCapture (faster).

    Important:
      - When a stream is partially corrupt, OpenCV/FFmpeg may fail at some frame ids.
        Doing an unbounded decrement loop can become O(video_length) per failed frame
        and destroy throughput. We cap the backsearch and otherwise reuse last_good.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out: List[np.ndarray] = []
    last_good: Optional[np.ndarray] = None
    bs = int(max(0, backsearch_max))

    for fid0 in frame_ids:
        fid = int(fid0)
        if total > 0:
            fid = max(0, min(fid, total - 1))

        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ok, frame = cap.read()

        if (not ok) or frame is None:
            # bounded backward search
            if bs > 0:
                dec = fid
                tries = 0
                frame = None
                while dec > 0 and tries < bs:
                    dec -= 1
                    tries += 1
                    cap.set(cv2.CAP_PROP_POS_FRAMES, dec)
                    ok2, f2 = cap.read()
                    if ok2 and f2 is not None:
                        frame = f2
                        break

            # if still none, reuse last_good immediately (fast)
            if frame is None:
                if last_good is None:
                    cap.release()
                    raise RuntimeError(f"Failed to decode any frame around fid={fid0} for {video_path}")
                frame = last_good
        else:
            last_good = frame

        out.append(frame)

    cap.release()
    return out


def pad_to_multiple_of_bgr(frame_bgr: np.ndarray, base: int) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Pad bottom/right with zeros so that H and W are multiples of `base`.

    For patch-based 2x2 blocks, you typically want base = 2 * patch.

    Returns padded frame and (pad_bottom, pad_right).
    """
    base = int(max(1, base))
    H, W = frame_bgr.shape[:2]
    pad_bottom = (base - (H % base)) % base
    pad_right = (base - (W % base)) % base
    if pad_bottom == 0 and pad_right == 0:
        return frame_bgr, (0, 0)
    out = cv2.copyMakeBorder(
        frame_bgr,
        0,
        pad_bottom,
        0,
        pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )
    return out, (pad_bottom, pad_right)


def pad_to_multiple_of_32_bgr(frame_bgr: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Backward-compatible wrapper (old behavior)."""
    return pad_to_multiple_of_bgr(frame_bgr, 32)


def bgr_to_residual_y_u8(frame_bgr: np.ndarray) -> np.ndarray:
    """
    Fallback residual proxy (if cv_reader not available): use frame luminance (uint8).
    """
    yuv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YUV)
    return yuv[:, :, 0].astype(np.uint8)


def mv_res_score_map(
    mv: np.ndarray,
    res_y: np.ndarray,
    mv_unit_div: float = 4.0,
    mv_pct: float = 95.0,
    res_pct: float = 95.0,
    w_mv: float = 1.0,
    w_res: float = 1.0,
    mv_compensate: str = "none",  # {"none","median"}
    mv_dir_mode: str = "l0",      # {"l0","l1","max","sum","biw"}
    w_mv_l0: float = 1.0,
    w_mv_l1: float = 1.0,
) -> np.ndarray:
    """
    Produce fused score map in [0,1] float32 with shape (H,W).

    Inputs:
      - res_y: HxW uint8 (expected 128+residual or luminance proxy)
      - mv: motion vectors at mv-grid. Supported shapes:
          * (Hm, Wm, 2): (vx,vy) single direction (backward compatible)
          * (Hm, Wm, 4): (L0x,L0y,L1x,L1y) dual direction (HEVC/B-frame)

    mv_compensate:
      - "none": no global motion compensation
      - "median": subtract per-frame median MV (robust global motion) before magnitude

    mv_dir_mode (when mv has 4 channels):
      - "l0": use L0 only (default/backward compatible)
      - "l1": use L1 only
      - "max": use max(mag_l0, mag_l1)
      - "sum": use (mag_l0 + mag_l1)
      - "biw": use (w_mv_l0 * mag_l0 + w_mv_l1 * mag_l1)

    Notes:
      - We do not have an explicit validity mask from FFmpeg side-data.
        Zero MV can mean "missing" or "true zero". This is still a useful heuristic.
    """
    H, W = res_y.shape[:2]

    # residual energy
    x = np.abs(res_y.astype(np.float32) - 128.0)
    a = float(np.percentile(x, res_pct))
    a = max(a, 1.0)
    res_norm = np.clip(x / a, 0.0, 1.0).astype(np.float32)

    # mv energy
    mv_norm_u = np.zeros((H, W), dtype=np.float32)
    if mv.ndim == 3 and mv.shape[2] >= 2:
        # L0
        vx0 = mv[:, :, 0].astype(np.float32) / float(mv_unit_div)
        vy0 = mv[:, :, 1].astype(np.float32) / float(mv_unit_div)

        # L1 (optional)
        has_l1 = (mv.shape[2] >= 4)
        if has_l1:
            vx1 = mv[:, :, 2].astype(np.float32) / float(mv_unit_div)
            vy1 = mv[:, :, 3].astype(np.float32) / float(mv_unit_div)
        else:
            vx1 = None
            vy1 = None

        if mv_compensate == "median":
            # Robust global motion estimate; ignore all-zero vectors if possible.
            mask0 = (vx0 != 0.0) | (vy0 != 0.0)
            if np.any(mask0):
                medx0 = float(np.median(vx0[mask0]))
                medy0 = float(np.median(vy0[mask0]))
            else:
                medx0, medy0 = 0.0, 0.0
            vx0 = vx0 - medx0
            vy0 = vy0 - medy0

            if has_l1 and vx1 is not None and vy1 is not None:
                mask1 = (vx1 != 0.0) | (vy1 != 0.0)
                if np.any(mask1):
                    medx1 = float(np.median(vx1[mask1]))
                    medy1 = float(np.median(vy1[mask1]))
                else:
                    medx1, medy1 = 0.0, 0.0
                vx1 = vx1 - medx1
                vy1 = vy1 - medy1

        mag0 = np.sqrt(vx0 * vx0 + vy0 * vy0).astype(np.float32)

        if has_l1 and vx1 is not None and vy1 is not None:
            mag1 = np.sqrt(vx1 * vx1 + vy1 * vy1).astype(np.float32)

            mode = str(mv_dir_mode).lower().strip()
            if mode == "l1":
                mag = mag1
            elif mode == "max":
                mag = np.maximum(mag0, mag1)
            elif mode == "sum":
                mag = mag0 + mag1
            elif mode == "biw":
                mag = (float(w_mv_l0) * mag0 + float(w_mv_l1) * mag1)
            else:  # "l0" or unknown
                mag = mag0
        else:
            mag = mag0

        b = float(np.percentile(mag, mv_pct))
        b = max(b, 1e-6)
        mv_norm = np.clip(mag / b, 0.0, 1.0)
        mv_norm_u = cv2.resize(mv_norm, (int(W), int(H)), interpolation=cv2.INTER_NEAREST).astype(np.float32)

    denom = float(w_mv + w_res) if (w_mv + w_res) != 0 else 1.0
    fused = (float(w_mv) * mv_norm_u + float(w_res) * res_norm) / denom
    return np.clip(fused, 0.0, 1.0).astype(np.float32)


def score_map_to_patch_scores(fused_hw: np.ndarray, patch: int = 16) -> np.ndarray:
    """
    Convert (H,W) fused map to patch scores (hb,wb) by summing within each patch.
    Requires H,W divisible by patch.
    """
    H, W = fused_hw.shape[:2]
    p = int(patch)
    assert H % p == 0 and W % p == 0
    hb, wb = H // p, W // p
    # (hb, p, wb, p) sum over p dims
    x = fused_hw.reshape(hb, p, wb, p).sum(axis=(1, 3))
    return x.astype(np.float32)


def patch_scores_to_block_scores(ps: np.ndarray) -> np.ndarray:
    """
    ps: (hb,wb). hb/wb must be even.
    Return block scores: (hb//2, wb//2), sum of 2x2 patches.
    """
    hb, wb = ps.shape[:2]
    assert hb % 2 == 0 and wb % 2 == 0
    # reshape into blocks
    # (hb//2,2, wb//2,2) sum over (1,3)
    bs = ps.reshape(hb // 2, 2, wb // 2, 2).sum(axis=(1, 3))
    return bs.astype(np.float32)


def iter_blocks_in_raster(hb: int, wb: int):
    """
    Iterate 2x2 blocks in raster order of blocks.
    hb/wb are patch-grid sizes (even).
    Yields (bh, bw) in block-grid.
    """
    for bh in range(hb // 2):
        for bw in range(wb // 2):
            yield bh, bw


def block_to_4_patches(bh: int, bw: int) -> List[Tuple[int, int]]:
    """
    Convert block coord to 4 patch coords in the required contiguous order.
    """
    h0 = 2 * int(bh)
    w0 = 2 * int(bw)
    return [(h0, w0), (h0, w0 + 1), (h0 + 1, w0), (h0 + 1, w0 + 1)]


def extract_patch_rgb(frame_rgb: np.ndarray, ph: int, pw: int, patch: int = 16) -> np.ndarray:
    p = int(patch)
    y0 = int(ph) * p
    x0 = int(pw) * p
    return frame_rgb[y0:y0 + p, x0:x0 + p, :]


# -----------------------------
# cv_reader streaming fetch
# -----------------------------

def cv_reader_fetch_mvres(video_path: str, frame_ids: List[int]) -> List[Dict[str, Any]]:
    """
    Fetch mv + residual_y aligned with frame_ids (supports duplicates).
    Requires cv_reader.read_video_cb.
    """
    if not HAS_CV_READER or not hasattr(cv_api, "read_video_cb"):
        raise RuntimeError("cv_reader.read_video_cb not available")

    frame_ids = [int(x) for x in frame_ids]
    pos_map: Dict[int, List[int]] = {}
    for i, fid in enumerate(frame_ids):
        pos_map.setdefault(fid, []).append(i)

    out: List[Optional[Dict[str, Any]]] = [None] * len(frame_ids)
    max_fid = max(frame_ids)

    def all_done() -> bool:
        for q in pos_map.values():
            if q:
                return False
        return True

    def cb(d: Dict[str, Any]):
        idx = int(d.get("frame_idx", -1))
        if idx in pos_map and pos_map[idx]:
            j = pos_map[idx].pop(0)
            mv = np.asarray(d["motion_vector"])
            # prefer residual_y; fallback residual
            if "residual_y" in d:
                ry = np.asarray(d["residual_y"])
            else:
                ry = np.asarray(d["residual"])
                if ry.ndim == 3:
                    ry = cv2.cvtColor(ry, cv2.COLOR_BGR2YUV)[:, :, 0]
            out[j] = {"frame_idx": idx, "motion_vector": mv, "residual_y": ry}
        return (not all_done())

    # without_residual=0 means WITH residual (your earlier convention)
    without_residual = 0
    max_frames = int(max_fid) + 1
    cv_api.read_video_cb(str(video_path), cb, int(without_residual), int(max_frames), frame_ids)

    # fill missing with nearest previous
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

    return [x for x in out if x is not None]  # type: ignore


# -----------------------------
# main per-video worker
# -----------------------------

def process_one_video(
    video_path: str,
    key: str,
    out_root: str,
    seq_len: int,
    window_len_frames: int,
    num_candidates: int,
    top_k_windows: int,
    max_total_patches: int,
    patch: int,
    mv_unit_div: float,
    mv_pct: float,
    res_pct: float,
    w_mv: float,
    w_res: float,
    max_total_patches_cap: int = 30000,
    mv_compensate: str = "none",  # {"none","median"}
    mv_dir_mode: str = "l0",
    w_mv_l0: float = 1.0,
    w_mv_l1: float = 1.0,
    fill_to_images: bool = True,
    force_fallback_no_cv_reader: bool = False,
    sample_id: str = "",
    caption: str = "",
    decode_backsearch_max: int = 32,
    per_video_timeout_sec: int = 0,
) -> Tuple[str, str, Optional[Dict[str, Any]]]:
    """
    Returns (status, message, index_record). status in {"ok","skip","fail"}. index_record is a dict when status=="ok" else None.
    """
    # avoid OpenCV oversubscription inside multiprocessing
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass
    os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
    # Reduce FFmpeg/OpenCV log spam on corrupted streams (can become a major I/O bottleneck).
    # This affects OpenCV's internal FFmpeg backend in many builds.
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "loglevel;error")

    vp = str(video_path)
    if not vp or (not Path(vp).exists()):
        return "skip", f"{key} missing video", None

    out_dir = str(Path(out_root) / key)
    done_mark = str(Path(out_dir) / "_DONE")
    if Path(done_mark).exists():
        return "skip", f"{key} already done", None

    ensure_dir(out_dir)

    total_frames, fps, H0, W0 = get_total_frames_fps(vp)
    if total_frames <= 0:
        return "fail", f"{key} cannot read total_frames", None

    # 1) pick windows by energy
    windows, win_dbg = pick_windows_by_energy(
        vp, total_frames, fps,
        window_len_frames=window_len_frames,
        num_candidates=num_candidates,
        top_k=top_k_windows,
    )

    # 2) allocate frame ids across selected windows
    frame_ids = allocate_frames_across_windows(windows, seq_len=int(seq_len))

    # 3) decode RGB frames for selected frame ids (we need actual pixels for patches)
    frames_bgr = decode_frames_bgr(vp, frame_ids, backsearch_max=int(decode_backsearch_max))

    # Patch size and padding base.
    # For 2x2 patch blocks, we must ensure hb and wb are even:
    #   hb = H1/patch, wb = W1/patch => require H1 and W1 divisible by 2*patch.
    p = int(patch)
    pad_base = 2 * p

    # 4) pad each frame to multiples of (2*patch) so hb/wb are even (works for patch=14 -> base=28)
    frames_bgr_pad: List[np.ndarray] = []
    pad_bottom, pad_right = 0, 0
    for i, fr in enumerate(frames_bgr):
        frp, (pb, pr) = pad_to_multiple_of_bgr(fr, pad_base)
        frames_bgr_pad.append(frp)
        if i == 0:
            pad_bottom, pad_right = pb, pr

    H1, W1 = frames_bgr_pad[0].shape[:2]
    if H1 % p != 0 or W1 % p != 0:
        return "fail", f"{key} padded size not divisible by patch: H1={H1} W1={W1} patch={p}", None

    hb, wb = H1 // p, W1 // p
    if hb % 2 != 0 or wb % 2 != 0:
        return "fail", f"{key} patch grid not even after pad: hb={hb} wb={wb}", None

    # 5) fetch mv/res for scoring
    use_cv = (HAS_CV_READER and (not force_fallback_no_cv_reader))
    items: Optional[List[Dict[str, Any]]] = None
    if use_cv:
        try:
            items = cv_reader_fetch_mvres(vp, frame_ids)
        except Exception as e:
            items = None
            use_cv = False
    # Safety: some corrupted streams may cause cv_reader to miss requested frame indices.
    # If alignment breaks, fall back to RGB-luma scoring instead of failing.
    if use_cv and items is not None and len(items) != len(frames_bgr_pad):
        items = None
        use_cv = False

    # fallback scoring using frame luminance (still produces some scores)
    fused_maps: List[np.ndarray] = []
    for t in range(len(frames_bgr_pad)):
        fr = frames_bgr_pad[t]
        # to RGB
        fr_rgb = fr[:, :, ::-1]
        if use_cv and items is not None and t < len(items):
            try:
                mv = np.asarray(items[t]["motion_vector"])
                ry = np.asarray(items[t]["residual_y"])
                # pad residual_y to match padded RGB
                if ry.shape[0] != H1 or ry.shape[1] != W1:
                    ry_pad = cv2.copyMakeBorder(ry, 0, pad_bottom, 0, pad_right, borderType=cv2.BORDER_CONSTANT, value=128)
                else:
                    ry_pad = ry
                fused = mv_res_score_map(
                    mv=mv,
                    res_y=ry_pad,
                    mv_unit_div=mv_unit_div,
                    mv_pct=mv_pct,
                    res_pct=res_pct,
                    w_mv=w_mv,
                    w_res=w_res,
                    mv_compensate=mv_compensate,
                    mv_dir_mode=mv_dir_mode,
                    w_mv_l0=w_mv_l0,
                    w_mv_l1=w_mv_l1,
                )
            except Exception:
                # Any mismatch -> fall back for this frame
                ry = bgr_to_residual_y_u8(fr)
                fused = mv_res_score_map(
                    mv=np.zeros((1, 1, 2), dtype=np.float32),
                    res_y=ry,
                    mv_unit_div=mv_unit_div,
                    mv_pct=mv_pct,
                    res_pct=res_pct,
                    w_mv=0.0,
                    w_res=1.0,
                    mv_compensate="none",
                    mv_dir_mode="l0",
                    w_mv_l0=1.0,
                    w_mv_l1=1.0,
                )
        else:
            # fallback: use "residual_y proxy" only
            ry = bgr_to_residual_y_u8(fr)
            fused = mv_res_score_map(
                mv=np.zeros((1, 1, 2), dtype=np.float32),
                res_y=ry,
                mv_unit_div=mv_unit_div,
                mv_pct=mv_pct,
                res_pct=res_pct,
                w_mv=0.0,
                w_res=1.0,
                mv_compensate="none",
                mv_dir_mode="l0",
                w_mv_l0=1.0,
                w_mv_l1=1.0,
            )
        fused_maps.append(fused.astype(np.float32))

    # 6) decide token budget
    S_full = hb * wb  # patches per full image/canvas

    cap_total = int(max_total_patches_cap) if int(max_total_patches_cap) > 0 else 30000
    max_total = int(max_total_patches)
    auto_budget_dbg: Optional[Dict[str, Any]] = None

    if max_total <= 0:
        # auto budget based on resolution + duration, hard-capped by cap_total
        max_total, auto_budget_dbg = auto_max_total_patches(
            S_full=S_full, total_frames=total_frames, fps=fps, cap_total=cap_total
        )

    # enforce at least full I
    if max_total < S_full:
        max_total = int(S_full)

    # Prefer that tokens fill whole canvases: total patches must be multiple of S_full
    if bool(fill_to_images):
        num_imgs = int(math.ceil(float(max_total) / float(S_full)))
        max_total = int(num_imgs * S_full)

    # If auto budget was used, ensure we still do not exceed cap_total after rounding
    if auto_budget_dbg is not None and cap_total > 0 and max_total > cap_total:
        num_imgs_cap = max(1, int(cap_total // S_full))
        max_total = int(num_imgs_cap * S_full)

    rem = int(max_total - S_full)
    nP = max(0, len(frame_ids) - 1)

    # Convert remaining patch budget to block budget; each block contributes 4 patches.
    total_block_budget = int(rem // 4)

    # available blocks per frame
    max_blocks = int((hb // 2) * (wb // 2))

    # NOTE: We will NOT split block budget evenly per frame.
    # We will perform global topK block selection across all P-frames.

    # 7) build output token sequence
    patches_list: List[np.ndarray] = []
    src_pos_list: List[List[int]] = []       # [src_img_idx, src_patch_h, src_patch_w]
    src_frameid_list: List[int] = []         # original frame id per patch

    # helper to append one frame's patches (in a specific block order)
    def append_frame_patches(src_img_idx: int, frame_rgb: np.ndarray, blocks: List[Tuple[int, int]]):
        # NOTE: src_patch_position.npy first column stores the ORIGINAL frame_id (not the 0..seq_len-1 index).
        try:
            src_fid = int(frame_ids[int(src_img_idx)])
        except Exception:
            src_fid = -1
        for (bh, bw) in blocks:
            coords = block_to_4_patches(bh, bw)
            for (ph, pw) in coords:
                patch_rgb = extract_patch_rgb(frame_rgb, ph, pw, patch=p)
                patches_list.append(patch_rgb.astype(np.uint8))
                # src_patch_position: [frame_id, patch_h, patch_w]
                src_pos_list.append([int(src_fid), int(ph), int(pw)])
                # keep src_frame_ids.npy consistent
                src_frameid_list.append(int(src_fid))

    # I frame (src_img_idx=0): ONLY this first image is full; all subsequent are sparse (P-style)
    frame0_rgb = frames_bgr_pad[0][:, :, ::-1]
    all_blocks = list(iter_blocks_in_raster(hb, wb))
    append_frame_patches(src_img_idx=0, frame_rgb=frame0_rgb, blocks=all_blocks)

    # P frames: GLOBAL topK 2x2-block selection across all P-frames
    selected_blocks_by_frame: Dict[int, List[Tuple[int, int]]] = {}
    selected_blocks_count_by_frame: Dict[int, int] = {}

    if nP > 0 and total_block_budget > 0:
        # Build global candidate pool: each candidate is one 2x2 block at some frame t
        scores_list: List[np.ndarray] = []
        t_list: List[np.ndarray] = []
        idx_list: List[np.ndarray] = []

        bwb = int(wb // 2)
        for t in range(1, len(frames_bgr_pad)):
            ps = score_map_to_patch_scores(fused_maps[t], patch=p)  # (hb,wb)
            bs = patch_scores_to_block_scores(ps)                   # (hb//2, wb//2)
            flat = bs.reshape(-1).astype(np.float32)

            scores_list.append(flat)
            t_list.append(np.full((flat.size,), int(t), dtype=np.int32))
            idx_list.append(np.arange(flat.size, dtype=np.int32))

        scores_all = np.concatenate(scores_list, axis=0) if scores_list else np.zeros((0,), dtype=np.float32)
        t_all = np.concatenate(t_list, axis=0) if t_list else np.zeros((0,), dtype=np.int32)
        idx_all = np.concatenate(idx_list, axis=0) if idx_list else np.zeros((0,), dtype=np.int32)

        k = int(min(int(total_block_budget), int(scores_all.size)))
        if k > 0:
            top = np.argpartition(-scores_all, kth=k - 1)[:k]
            top = top[np.argsort(-scores_all[top])]

            # Group selected blocks per frame in the global score order (deterministic)
            tmp: Dict[int, List[int]] = {}
            for j in top.tolist():
                tt = int(t_all[j])
                fi = int(idx_all[j])
                tmp.setdefault(tt, []).append(fi)

            for tt, flat_ids in tmp.items():
                blocks = [(int(fid // bwb), int(fid % bwb)) for fid in flat_ids]
                selected_blocks_by_frame[int(tt)] = blocks
                selected_blocks_count_by_frame[int(tt)] = int(len(blocks))

    # Now append patches for each P-frame following the selected blocks
    for t in range(1, len(frames_bgr_pad)):
        blocks = selected_blocks_by_frame.get(int(t), [])
        if not blocks:
            continue
        fr_rgb = frames_bgr_pad[t][:, :, ::-1]
        append_frame_patches(src_img_idx=t, frame_rgb=fr_rgb, blocks=blocks)

    # 7.5) pad/truncate to exactly max_total patches so that we can fill whole canvases
    pz = np.zeros((p, p, 3), dtype=np.uint8)
    valid_n = int(len(patches_list))
    if valid_n > int(max_total):
        patches_list = patches_list[: int(max_total)]
        src_pos_list = src_pos_list[: int(max_total)]
        src_frameid_list = src_frameid_list[: int(max_total)]
        valid_n = int(max_total)

    while len(patches_list) < int(max_total):
        patches_list.append(pz)
        src_pos_list.append([-1, -1, -1])
        src_frameid_list.append(-1)

    patches = np.stack(patches_list, axis=0).astype(np.uint8) if patches_list else np.zeros((0, p, p, 3), dtype=np.uint8)
    src_patch_position = np.asarray(src_pos_list, dtype=np.int32) if src_pos_list else np.zeros((0, 3), dtype=np.int32)
    src_frame_ids = np.asarray(src_frameid_list, dtype=np.int32) if src_frameid_list else np.zeros((0,), dtype=np.int32)

    # Pack into full canvases
    images_rgb, patch_position, img_ptr_arr = pack_patches_to_canvases(patches, hb=hb, wb=wb, patch=p)
    frame_ids_arr = np.asarray(frame_ids, dtype=np.int32)

    # Save JPG canvases (do NOT save images.npy / patches.npy)
    jpg_files = save_canvases_as_jpg(images_rgb, out_dir=out_dir, quality=95)

    # Save required arrays/metadata
    np.save(str(Path(out_dir) / "patch_position.npy"), patch_position, allow_pickle=False)
    np.save(str(Path(out_dir) / "img_ptr.npy"), img_ptr_arr, allow_pickle=False)
    np.save(str(Path(out_dir) / "frame_ids.npy"), frame_ids_arr, allow_pickle=False)
    np.save(str(Path(out_dir) / "src_patch_position.npy"), src_patch_position, allow_pickle=False)  # [frame_id, patch_h, patch_w]
    np.save(str(Path(out_dir) / "src_frame_ids.npy"), src_frame_ids, allow_pickle=False)
    # np.save(str(Path(out_dir) / "position.npy"), src_patch_position, allow_pickle=False) 

    meta = {
        "video": vp,
        "key": key,
        "out_dir": str(Path(out_root) / key),
        "mirror_src_root": None,
        "total_frames": int(total_frames),
        "fps": float(fps),
        "orig_hw": [int(H0), int(W0)],
        "pad_bottom_right": [int(pad_bottom), int(pad_right)],
        "padded_hw": [int(H1), int(W1)],
        "patch": int(p),
        "pad_base": int(pad_base),
        "hb_wb": [int(hb), int(wb)],
        "seq_len": int(seq_len),
        "windows": [{"start": int(a), "end": int(b), "pkt_sum": int(c)} for (a, b, c) in windows],
        "window_debug": win_dbg,
        "w_mv": float(w_mv),
        "w_res": float(w_res),
        "mv_compensate": str(mv_compensate),
        "mv_dir_mode": str(mv_dir_mode),
        "w_mv_l0": float(w_mv_l0),
        "w_mv_l1": float(w_mv_l1),
        "use_cv_reader": bool(use_cv),
        "max_total_patches": int(max_total),
        "I_frame_patches": int(S_full),
        "total_patches_target": int(max_total),
        "total_patches_valid": int(valid_n),
        "num_images": int(images_rgb.shape[0]),
        "jpg_files": jpg_files,
        "patches_per_image": int(S_full),
        "auto_budget": auto_budget_dbg,
        "global_block_selection": {
            "total_block_budget": int(total_block_budget),
            "max_blocks_per_frame": int(max_blocks),
            "selected_blocks_total": int(sum(selected_blocks_count_by_frame.values()) if selected_blocks_count_by_frame else 0),
            "selected_blocks_per_frame": selected_blocks_count_by_frame,
        },
        "fill_to_images": bool(fill_to_images),
        # src_patch_position now stores [frame_id, patch_h, patch_w] (pad tokens are -1)
        "src_patch_position_format": "[frame_id, patch_h, patch_w] (pad tokens are -1)",
        "position_npy": "position.npy",
        "sample_id": str(sample_id) if sample_id else "",
        "caption": str(caption) if caption else "",
    }
    meta["mirror_src_root"] = os.environ.get("LLAVA_CODEC_MIRROR_SRC_ROOT")
    with open(str(Path(out_dir) / "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # A compact per-sample record for fast join in step-2 (avoid per-sample filesystem probing).
    index_rec: Dict[str, Any] = {
        "video": vp,
        "key": str(key),
        "asset_dir": str(out_dir),
        "jpg": list(jpg_files),
        "patch_position": "patch_position.npy",
        "img_ptr": "img_ptr.npy",
        "src_patch_position": "src_patch_position.npy",
        "src_frame_ids": "src_frame_ids.npy",
        "frame_ids": "frame_ids.npy",
        "position": "position.npy",
        "sample_id": str(sample_id) if sample_id else "",
        "caption": str(caption) if caption else "",
        # prefer meta-derived count; this equals the packed total (including pad patches)
        "num_patches": int(meta.get("max_total_patches", int(patches.shape[0]))),
        "num_images": int(meta.get("num_images", int(images_rgb.shape[0]))),
        "patch": int(meta.get("patch", p)),
        "padded_hw": meta.get("padded_hw"),
    }

    Path(done_mark).write_text("ok\n", encoding="utf-8")
    return "ok", f"{key} ok patches={int(patches.shape[0])} images={int(images_rgb.shape[0])} jpg={len(jpg_files)} use_cv_reader={int(use_cv)}", index_rec


# -----------------------------
# multiprocessing entry
# -----------------------------

def _worker(job: Dict[str, Any]) -> Tuple[str, str, Optional[Dict[str, Any]]]:
    key = job.get("key", "unknown")

    # Optional hard timeout per job to avoid a single corrupted stream stalling the pool.
    tmo = int(job.get("per_video_timeout_sec", 0) or 0)
    if tmo > 0:
        try:
            import signal

            def _alarm_handler(signum, frame):
                raise TimeoutError(f"timeout>{tmo}s")

            signal.signal(signal.SIGALRM, _alarm_handler)
            signal.alarm(tmo)
        except Exception:
            # If signal is unavailable, proceed without hard timeout.
            tmo = 0

    try:
        return process_one_video(**job)
    except Exception as e:
        return "fail", f"{key} err={repr(e)}", None
    finally:
        if tmo > 0:
            try:
                import signal
                signal.alarm(0)
            except Exception:
                pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True, help="jsonl lines containing at least {video, key, exists}")
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--num_workers", type=int, default=8)

    ap.add_argument("--max_lines", type=int, default=0, help="only scan first N lines of jsonl (0 => all)")
    ap.add_argument("--max_jobs", type=int, default=0, help="only process first N inferred jobs (0 => all)")
    ap.add_argument("--scan_log_every", type=int, default=5000, help="print scan progress every N lines")
    ap.add_argument("--assume_exists", action="store_true", help="skip Path.exists() check (faster on remote FS)")
    ap.add_argument("--response_ok_only", action="store_true", help="for OpenAI-like result jsonl: keep only error==null and status_code==200")

    # sampling
    ap.add_argument("--seq_len", type=int, default=64)
    ap.add_argument("--window_len_frames", type=int, default=512)
    ap.add_argument("--num_candidates", type=int, default=5)
    ap.add_argument("--top_k_windows", type=int, default=2)

    # patch/token budget
    ap.add_argument("--patch", type=int, default=16)
    ap.add_argument("--max_total_patches", type=int, default=0, help="overall patch token budget (0 => default small)")
    ap.add_argument("--max_total_patches_cap", type=int, default=30000, help="hard cap for selected patches when --max_total_patches=0 (auto budget)")

    # scoring weights
    ap.add_argument("--mv_unit_div", type=float, default=4.0)
    ap.add_argument("--mv_pct", type=float, default=95.0)
    ap.add_argument("--res_pct", type=float, default=95.0)
    ap.add_argument("--w_mv", type=float, default=1.0)
    ap.add_argument("--w_res", type=float, default=1.0)
    ap.add_argument("--mv_compensate", type=str, default="none", choices=["none", "median"], help="MV global motion compensation")
    ap.add_argument(
        "--mv_dir_mode",
        type=str,
        default="l0",
        choices=["l0", "l1", "max", "sum", "biw"],
        help="How to use dual-direction MV (L0/L1) when available (e.g., B-frames in HEVC).",
    )
    ap.add_argument("--w_mv_l0", type=float, default=1.0, help="Weight for L0 magnitude when --mv_dir_mode=biw")
    ap.add_argument("--w_mv_l1", type=float, default=1.0, help="Weight for L1 magnitude when --mv_dir_mode=biw")
    ap.add_argument("--no_fill_to_images", action="store_true", help="do not round token budget to fill full canvases")

    ap.add_argument("--chunksize", type=int, default=8)
    ap.add_argument("--force_no_cv_reader", action="store_true", help="force fallback scoring without cv_reader")

    # ---- new mirror CLI args ----
    ap.add_argument(
        "--mirror_src_root",
        type=str,
        default="",
        help="If set, mirror output directory structure relative to this source root. Example: /ov2/dataset_mid_source/batch1_5millions",
    )
    ap.add_argument(
        "--mirror_keep_ext",
        action="store_true",
        help="If set together with --mirror_src_root, keep the video file extension in the mirrored output folder name (default: strip extension).",
    )

    # ---- optional assets index for step-2 fast join ----
    ap.add_argument(
        "--assets_index",
        type=str,
        default="",
        help="If set, append a per-sample assets index jsonl line for every OK video (written by the main process).",
    )
    ap.add_argument(
        "--assets_index_append",
        action="store_true",
        help="Append to --assets_index instead of overwriting (default: append anyway; kept for clarity).",
    )
    ap.add_argument(
        "--assets_index_flush_every",
        type=int,
        default=200,
        help="Flush assets_index writer every N ok records (0 => never flush until end).",
    )
    ap.add_argument(
        "--out_train_jsonl",
        type=str,
        default="",
        help="If set, write training jsonl directly in step-1 with fields: id, key, messages, images, patch_positions, num_patches.",
    )

    ap.add_argument(
        "--decode_backsearch_max",
        type=int,
        default=32,
        help="Max frames to search backward when a target frame fails to decode (0 disables backsearch; uses last_good immediately).",
    )
    ap.add_argument(
        "--per_video_timeout_sec",
        type=int,
        default=0,
        help="Hard timeout per video in worker process (seconds). 0 disables.",
    )

    args = ap.parse_args()

    if args.mirror_src_root:
        os.environ["LLAVA_CODEC_MIRROR_SRC_ROOT"] = str(args.mirror_src_root)

    ensure_dir(args.out_root)

    print(f"[scan] jsonl={args.jsonl} (streaming) max_lines={args.max_lines} max_jobs={args.max_jobs} assume_exists={int(args.assume_exists)}")

    # Generator that yields job dicts on-the-fly
    def job_iter():
        n_missing = 0
        n_skipped_schema = 0
        n_skipped_resp = 0
        yielded = 0

        for ln, it in iter_jsonl(args.jsonl):
            if args.max_lines and ln > int(args.max_lines):
                break

            # Optional filter for OpenAI-like result jsonl
            if args.response_ok_only:
                err = it.get("error")
                if err not in (None, "null"):
                    n_skipped_resp += 1
                    continue
                resp = it.get("response")
                if isinstance(resp, dict):
                    sc = resp.get("status_code")
                    if sc is not None and int(sc) != 200:
                        n_skipped_resp += 1
                        continue

            # Some schemas provide an explicit exists; if exists==False we skip early
            if it.get("exists") is False:
                n_missing += 1
                continue

            video = extract_video_path_from_item(it)
            if not video:
                n_skipped_schema += 1
                continue

            video = str(video)
            if (not args.assume_exists) and (not Path(video).exists()):
                n_missing += 1
                continue

            key: Optional[str] = None
            if args.mirror_src_root:
                key = mirror_key_from_video(
                    video_path=str(video),
                    mirror_src_root=str(args.mirror_src_root),
                    strip_ext=(not bool(args.mirror_keep_ext)),
                )

            if not key:
                key = infer_key_from_video(video, it)

            sample_id = it.get("id")
            if not isinstance(sample_id, str):
                sample_id = ""
            caption = extract_caption_from_item(it)

            yield {
                "video_path": str(video),
                "key": str(key),
                "out_root": str(args.out_root),
                "seq_len": int(args.seq_len),
                "window_len_frames": int(args.window_len_frames),
                "num_candidates": int(args.num_candidates),
                "top_k_windows": int(args.top_k_windows),
                "max_total_patches": int(args.max_total_patches),
                "max_total_patches_cap": int(args.max_total_patches_cap),
                "patch": int(args.patch),
                "mv_unit_div": float(args.mv_unit_div),
                "mv_pct": float(args.mv_pct),
                "res_pct": float(args.res_pct),
                "w_mv": float(args.w_mv),
                "w_res": float(args.w_res),
                "mv_compensate": str(args.mv_compensate),
                "mv_dir_mode": str(args.mv_dir_mode),
                "w_mv_l0": float(args.w_mv_l0),
                "w_mv_l1": float(args.w_mv_l1),
                "fill_to_images": (not bool(args.no_fill_to_images)),
                "force_fallback_no_cv_reader": bool(args.force_no_cv_reader),
                "sample_id": str(sample_id),
                "caption": str(caption),
                "decode_backsearch_max": int(args.decode_backsearch_max),
                "per_video_timeout_sec": int(args.per_video_timeout_sec),
            }

            yielded += 1
            if args.max_jobs and yielded >= int(args.max_jobs):
                break

            if args.scan_log_every and (ln % int(args.scan_log_every) == 0):
                print(
                    f"[scan] ln={ln} yielded={yielded} missing={n_missing} schema_skip={n_skipped_schema} resp_skip={n_skipped_resp}")

        print(
            f"[scan_done] yielded={yielded} missing={n_missing} schema_skip={n_skipped_schema} resp_skip={n_skipped_resp}")

    print(
        f"[info] out_root={args.out_root} workers={args.num_workers} seq_len={args.seq_len} win_len={args.window_len_frames} "
        f"cand={args.num_candidates} topk={args.top_k_windows} patch={args.patch} max_total_patches={args.max_total_patches} "
        f"mv_compensate={args.mv_compensate} mv_dir_mode={args.mv_dir_mode} w_mv_l0={args.w_mv_l0} w_mv_l1={args.w_mv_l1} "
        f"cv_reader={int(HAS_CV_READER)} chunksize={args.chunksize} cap={args.max_total_patches_cap} "
        f"mirror_src_root={args.mirror_src_root or 'NONE'} keep_ext={int(args.mirror_keep_ext)}"
    )

    from multiprocessing import Pool

    ok = skip = fail = 0
    t0 = time.time()

    index_fp = None
    index_ok = 0
    if args.assets_index:
        idx_p = Path(args.assets_index)
        idx_p.parent.mkdir(parents=True, exist_ok=True)
        # We always append for safety with partial runs; this avoids losing previous progress.
        index_fp = open(idx_p, "a", encoding="utf-8")

    train_fp = None
    if args.out_train_jsonl:
        tp = Path(args.out_train_jsonl)
        tp.parent.mkdir(parents=True, exist_ok=True)
        train_fp = open(tp, "a", encoding="utf-8")

    with Pool(processes=int(args.num_workers)) as pool:
        for i, (st, msg, idx_rec) in enumerate(pool.imap_unordered(_worker, job_iter(), chunksize=int(args.chunksize)), start=1):
            if st == "ok":
                ok += 1
                if index_fp is not None and isinstance(idx_rec, dict) and idx_rec:
                    index_fp.write(json.dumps(idx_rec, ensure_ascii=False) + "\n")
                    index_ok += 1
                    if args.assets_index_flush_every and (index_ok % int(args.assets_index_flush_every) == 0):
                        index_fp.flush()
                if train_fp is not None and isinstance(idx_rec, dict) and idx_rec:
                    asset_dir = str(idx_rec.get("asset_dir", ""))
                    jpg_list = idx_rec.get("jpg") or []
                    images = [str(Path(asset_dir) / str(fn)) for fn in jpg_list]

                    patch_positions = [str(Path(asset_dir) / "src_patch_position.npy")]  # only one

                    n_img = max(1, len(images))
                    user_content = "".join(["<image>\n" for _ in range(n_img)])

                    out_item = {
                        # must match input results jsonl id
                        "id": str(idx_rec.get("sample_id", "")) or str(idx_rec.get("key", "")),
                        "messages": [
                            {"role": "user", "content": user_content},
                            {"role": "assistant", "content": str(idx_rec.get("caption", ""))},
                        ],
                        "images": images,
                        # codec version: one npy storing [frame_id, patch_h, patch_w] for all tokens
                        "patch_positions": patch_positions,
                        "num_patches": int(idx_rec.get("num_patches", 0)),
                    }
                    train_fp.write(json.dumps(out_item, ensure_ascii=False) + "\n")
            elif st == "skip":
                skip += 1
            else:
                fail += 1

            if i % 20 == 0 or st == "fail":
                rate = i / max(1e-6, (time.time() - t0))
                print(f"[prog] done={i} ok={ok} skip={skip} fail={fail} rate={rate:.2f}/s last={msg}")

    if index_fp is not None:
        index_fp.flush()
        index_fp.close()

    if train_fp is not None:
        train_fp.flush()
        train_fp.close()

    if args.assets_index:
        print(f"[done] ok={ok} skip={skip} fail={fail} assets_index={args.assets_index}")
    else:
        print(f"[done] ok={ok} skip={skip} fail={fail}")


if __name__ == "__main__":
    main()