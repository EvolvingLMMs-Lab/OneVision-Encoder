#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage1: square576 visidx generation
- Resize longer side to 576
- Pad to 576x576 square (center)
- Uniform sample 64 frames on original timeline
- Compute fused MV/RES energy per sampled frame
- Map fused energy to 576x576
- Patchify (16x16) -> score -> global topk over (T*valid_patches)
- Exclude padding (black border) patches from selection
- Output: visidx_thw.npy, frame_ids.npy, meta.json
    - Optionally, the first frame can be forced to keep all 576x576 patches (including padding) with --keep_first_full_frame,
      and the remaining patch budget is allocated to later frames.

Requires:
  - cv_reader (cv_reader.api as cv_api)
  - numpy, torch
Optional:
  - opencv-python (cv2) for resize (recommended)
  - ffprobe for codec check
"""

import argparse
import hashlib
import json
import subprocess
import time
import traceback
from pathlib import Path
from multiprocessing import cpu_count, get_context

import numpy as np
import torch

try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False

try:
    from cv_reader import api as cv_api
    _HAS_CV_READER = True
except Exception:
    _HAS_CV_READER = False


# -------------------- basic utils --------------------
def iter_json_array(path: str):
    """Stream-parse a top-level JSON array without loading whole file."""
    decoder = json.JSONDecoder()
    buf = ""
    with open(path, "r", encoding="utf-8") as f:
        # find '['
        while True:
            ch = f.read(1)
            if not ch:
                return
            if ch.isspace():
                continue
            if ch == "[":
                break
            raise ValueError("Expected JSON array (missing '[')")

        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            buf += chunk
            while True:
                s = buf.lstrip()
                if not s:
                    buf = ""
                    break
                if s[0] == "]":
                    return
                if s[0] == ",":
                    buf = s[1:]
                    continue
                try:
                    obj, idx = decoder.raw_decode(s)
                except json.JSONDecodeError:
                    buf = s
                    break
                yield obj
                buf = s[idx:]


def iter_dataset_items(path: str):
    p = Path(path)
    suf = p.suffix.lower()
    if suf == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                yield json.loads(ln)
    elif suf == ".json":
        yield from iter_json_array(path)
    else:
        raise ValueError(f"Unsupported dataset suffix: {suf}")


def _ffprobe_codec_name(video_path: str) -> str:
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=codec_name",
        "-of", "default=nw=1:nk=1",
        str(video_path),
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=10)
        return out.decode("utf-8", errors="ignore").strip().lower()
    except Exception:
        return ""


def _is_supported_codec(codec_name: str) -> bool:
    c = (codec_name or "").lower()
    return c in ("h264", "avc1", "hevc", "h265")


def sample_frame_ids_uniform(N: int, seq_len: int, start_offset: int = 0):
    if N <= 0:
        return []
    seq_len = int(seq_len)
    if seq_len <= 0:
        return []
    start_offset = int(start_offset)
    if start_offset < 0:
        start_offset = 0
    if start_offset >= N:
        # degenerate: fall back to the very first frame
        start_offset = 0
    # Uniformly sample within [start_offset, N-1]
    return np.linspace(start_offset, N - 1, seq_len, dtype=int).tolist()


# -------------------- energy helpers --------------------
def _residual_y_generic(residual: np.ndarray) -> np.ndarray:
    if residual.ndim == 2:
        return residual
    if residual.ndim == 3 and residual.shape[2] == 3:
        if _HAS_CV2:
            return cv2.cvtColor(residual, cv2.COLOR_BGR2YUV)[:, :, 0]
        else:
            return residual[:, :, 0]
    r = np.squeeze(residual)
    if r.ndim == 2:
        return r
    raise ValueError(f"Unexpected residual shape: {residual.shape}")


def _residual_energy_norm(res_y: np.ndarray, pct: float = 95.0, use_grad: bool = False):
    if not use_grad:
        x = np.abs(res_y.astype(np.float32) - 128.0)
    else:
        r = res_y.astype(np.float32)
        gx = np.abs(np.roll(r, -1, axis=1) - np.roll(r, 1, axis=1))
        gy = np.abs(np.roll(r, -1, axis=0) - np.roll(r, 1, axis=0))
        x = 0.5 * (gx + gy)
    a = float(np.percentile(x, pct))
    a = max(a, 1.0)
    norm = np.clip(x / a, 0.0, 1.0)
    return norm.astype(np.float32), a


def _mv_energy_norm(mvx_q: np.ndarray, mvy_q: np.ndarray, H: int, W: int,
                    mv_unit_div: float = 4.0, pct: float = 95.0,
                    compensate: str = "median"):
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

    # upsample mv-grid to HxW
    if _HAS_CV2:
        norm_u = cv2.resize(norm, (W, H), interpolation=cv2.INTER_NEAREST)
    else:
        ys = (np.linspace(0, norm.shape[0] - 1, H)).astype(np.int32)
        xs = (np.linspace(0, norm.shape[1] - 1, W)).astype(np.int32)
        norm_u = norm[ys[:, None], xs[None, :]]
    return norm_u.astype(np.float32), a


def _fuse_energy(norm_mv, norm_res, mode="weighted", w_mv=1.0, w_res=1.0):
    mode = (mode or "weighted").lower()
    if mode == "max":
        fused = np.maximum(norm_mv, norm_res)
    elif mode == "sum":
        fused = np.clip(norm_mv + norm_res, 0.0, 1.0)
    elif mode == "geomean":
        fused = np.sqrt(np.clip(norm_mv, 0.0, 1.0) * np.clip(norm_res, 0.0, 1.0))
    else:
        denom = float(w_mv + w_res) if (w_mv + w_res) != 0 else 1.0
        fused = (float(w_mv) * norm_mv + float(w_res) * norm_res) / denom
    return np.clip(fused, 0.0, 1.0).astype(np.float32)


# -------------------- square576 mapping --------------------
def _resize_longer_pad_square_map(arr2d: np.ndarray, out_size: int = 576):
    """
    arr2d: (H,W) float32
    return:
      out: (out_size,out_size)
      info: dict(scale,Hn,Wn,pad_top,pad_left,pad_bottom,pad_right)
    """
    H, W = arr2d.shape[:2]
    out_size = int(out_size)
    if H <= 0 or W <= 0:
        raise ValueError(f"Bad H,W: {H},{W}")

    scale = float(out_size) / float(max(H, W))
    Hn = max(1, int(round(H * scale)))
    Wn = max(1, int(round(W * scale)))

    if _HAS_CV2:
        resized = cv2.resize(arr2d.astype(np.float32), (Wn, Hn), interpolation=cv2.INTER_LINEAR)
    else:
        # very simple fallback
        ys = (np.linspace(0, H - 1, Hn)).astype(np.int32)
        xs = (np.linspace(0, W - 1, Wn)).astype(np.int32)
        resized = arr2d[ys[:, None], xs[None, :]].astype(np.float32)

    pad_top = (out_size - Hn) // 2
    pad_left = (out_size - Wn) // 2
    pad_bottom = out_size - Hn - pad_top
    pad_right = out_size - Wn - pad_left

    out = np.zeros((out_size, out_size), dtype=np.float32)
    out[pad_top:pad_top + Hn, pad_left:pad_left + Wn] = resized

    info = dict(
        scale=scale,
        Hn=int(Hn), Wn=int(Wn),
        pad_top=int(pad_top), pad_left=int(pad_left),
        pad_bottom=int(pad_bottom), pad_right=int(pad_right),
        out_size=int(out_size),
        H0=int(H), W0=int(W),
    )
    return out, info


def _valid_patch_mask_from_pad(out_size: int, patch: int, pad_top: int, pad_left: int, Hn: int, Wn: int):
    """
    Return valid patch mask (hb,wb) where patch is fully inside the content region (non-pad).
    """
    out_size = int(out_size)
    p = int(patch)
    hb, wb = out_size // p, out_size // p
    assert hb * p == out_size and wb * p == out_size

    content_top = int(pad_top)
    content_left = int(pad_left)
    content_bottom = int(pad_top + Hn)
    content_right = int(pad_left + Wn)

    # patch [y0:y1, x0:x1] must be fully inside content box
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


# -------------------- core compute --------------------
def compute_fused_volume_square576(
    video_path: str,
    seq_len: int = 64,
    sample_start_offset: int = 0,
    square_size: int = 576,
    patch_size: int = 16,
    mv_unit_div: float = 4.0,
    mv_pct: float = 95.0,
    res_pct: float = 95.0,
    fuse_mode: str = "weighted",
    w_mv: float = 1.0,
    w_res: float = 1.0,
    mv_compensate: str = "median",
    res_use_grad: bool = False,
):
    if not _HAS_CV_READER:
        raise RuntimeError("cv_reader not available")

    frames = cv_api.read_video(video_path, 0, -1)
    if not isinstance(frames, (list, tuple)) or len(frames) == 0:
        raise RuntimeError(f"cv_reader returned empty: {video_path}")

    T_all = len(frames)
    meta0 = frames[0]
    H0 = int(meta0["height"])
    W0 = int(meta0["width"])

    frame_ids = sample_frame_ids_uniform(T_all, int(seq_len), start_offset=int(sample_start_offset))
    fused_list = []
    pad_info_ref = None

    for fid in frame_ids:
        fr = frames[int(fid)]
        mv = np.asarray(fr["motion_vector"])          # (Hm,Wm,2) or similar
        res = np.asarray(fr["residual"])              # (H,W) or (H,W,3)

        Y_res = _residual_y_generic(res)              # (H,W)
        res_norm, _ = _residual_energy_norm(Y_res, pct=res_pct, use_grad=bool(res_use_grad))

        mv_norm, _ = _mv_energy_norm(
            mv[:, :, 0], mv[:, :, 1], H0, W0,
            mv_unit_div=mv_unit_div, pct=mv_pct, compensate=mv_compensate
        )

        fused_hw = _fuse_energy(mv_norm, res_norm, mode=fuse_mode, w_mv=w_mv, w_res=w_res)  # (H0,W0)
        fused_sq, info = _resize_longer_pad_square_map(fused_hw, out_size=int(square_size)) # (576,576)
        if pad_info_ref is None:
            pad_info_ref = info
        fused_list.append(fused_sq)

    fused_vol = np.stack(fused_list, axis=0).astype(np.float32)  # (T,576,576)

    # valid mask for excluding padding patches
    valid_mask = _valid_patch_mask_from_pad(
        out_size=int(square_size),
        patch=int(patch_size),
        pad_top=int(pad_info_ref["pad_top"]),
        pad_left=int(pad_info_ref["pad_left"]),
        Hn=int(pad_info_ref["Hn"]),
        Wn=int(pad_info_ref["Wn"]),
    )
    return fused_vol, frame_ids, (H0, W0), pad_info_ref, valid_mask


def global_topk_thw_square576(
    fused: np.ndarray,             # (T,576,576)
    valid_mask: np.ndarray,        # (36,36) bool
    patch_size: int = 16,
    keep_frames_equiv: int = 8,
    padding_policy: str = "exclude",  # "exclude" or "zero"
    keep_uniform_first: bool = False,  # if True, always keep t=0 content patches (valid_mask==True)
    keep_first_full_frame: bool = False,  # if True, keep ALL (hb*wb) patches from t=0 (including padding)
):
    T, H, W = fused.shape
    p = int(patch_size)
    assert H % p == 0 and W % p == 0
    hb, wb = H // p, W // p
    assert valid_mask.shape == (hb, wb)

    S_total = hb * wb
    S_valid = int(valid_mask.sum())

    pol = (padding_policy or "exclude").lower()
    if pol not in ("exclude", "zero"):
        raise ValueError(f"Unknown padding_policy: {padding_policy}")

    if pol == "exclude" and S_valid <= 0:
        return np.zeros((0, 3), np.int32), 0, hb, wb, S_total, S_valid

    # patch score: sum over each p×p region
    fused_t = torch.from_numpy(fused.astype(np.float32))
    scores = fused_t.reshape(T, hb, p, wb, p).sum(dim=(2, 4))   # (T,hb,wb)
    scores = scores.reshape(T, S_total)                         # (T,S_total)

    # padding handling
    mask_flat = torch.from_numpy(valid_mask.reshape(-1))
    invalid = ~mask_flat

    # If we are keeping the first full frame, we must allow padding patches to exist in the selection space.
    # We still slightly down-weight padding so it is unlikely to be selected unless necessary.
    if bool(keep_first_full_frame):
        scores[:, invalid] = -1e-4
    else:
        if pol == "exclude":
            # hard-exclude padding patches
            scores[:, invalid] = -1e6
        else:
            # allow padding patches but keep them slightly worse than any non-negative content
            # fused padding is already 0; this is a safety clamp.
            scores[:, invalid] = -1e-4

    # --- selection ---
    if pol == "exclude":
        K = int(keep_frames_equiv) * int(S_valid)
        K = min(K, T * S_valid)
    else:
        # fixed budget: full 36x36 patches per selected frame
        K = int(keep_frames_equiv) * int(S_total)
        K = min(K, T * S_total)
    if K <= 0:
        return np.zeros((0, 3), np.int32), 0, hb, wb, S_total, S_valid

    # Optionally keep the ENTIRE first frame (t=0), including padding patches.
    # Remaining budget is filled by global topk from later frames.
    if bool(keep_first_full_frame) and T > 0:
        # mandatory: all patches of t=0 in raster order (h-major then w)
        hh = np.repeat(np.arange(hb, dtype=np.int32), wb)
        ww = np.tile(np.arange(wb, dtype=np.int32), hb)
        tt = np.zeros((S_total,), dtype=np.int32)
        mandatory = np.stack([tt, hh, ww], axis=1)  # (S_total,3)

        K_total = int(keep_frames_equiv) * int(S_total)
        if K_total <= int(S_total):
            return mandatory[: int(K_total)].astype(np.int32), int(min(K_total, S_total)), int(hb), int(wb), int(S_total), int(S_valid)

        K_rem = int(K_total) - int(S_total)

        # Exclude ALL patches from t=0 for the remainder selection, so later frames fill the remaining budget.
        scores_rem = scores.clone()
        scores_rem[0, :] = -1e6

        flat = scores_rem.reshape(-1)
        _, topidx = torch.topk(flat, k=int(K_rem), largest=True, sorted=True)
        topidx_np = topidx.cpu().numpy().astype(np.int64)

        t = (topidx_np // S_total).astype(np.int32)
        s = (topidx_np % S_total).astype(np.int32)
        h = (s // wb).astype(np.int32)
        w = (s % wb).astype(np.int32)
        rest = np.stack([t, h, w], axis=1).astype(np.int32)

        thw = np.concatenate([mandatory.astype(np.int32), rest], axis=0)
        return thw, int(thw.shape[0]), int(hb), int(wb), int(S_total), int(S_valid)

    # Optionally force-include all content-region patches from the first sampled frame (t=0).
    if bool(keep_uniform_first) and T > 0:
        vm = valid_mask.astype(bool)
        mand_h, mand_w = np.where(vm)
        mandatory = np.stack(
            [np.zeros_like(mand_h), mand_h.astype(np.int32), mand_w.astype(np.int32)],
            axis=1,
        )
        K_mand = int(mandatory.shape[0])

        # If budget is smaller than mandatory, truncate mandatory and return.
        if int(K) <= K_mand:
            return mandatory[: int(K)].astype(np.int32), int(K), int(hb), int(wb), int(S_total), int(S_valid)

        K_rem = int(K) - K_mand

        # Exclude ALL patches from t=0 for the remainder selection, so later frames fill the remaining budget.
        scores_rem = scores.clone()
        scores_rem[0, :] = -1e6

        flat = scores_rem.reshape(-1)
        _, topidx = torch.topk(flat, k=int(K_rem), largest=True, sorted=True)
        topidx_np = topidx.cpu().numpy().astype(np.int64)

        t = (topidx_np // S_total).astype(np.int32)
        s = (topidx_np % S_total).astype(np.int32)
        h = (s // wb).astype(np.int32)
        w = (s % wb).astype(np.int32)
        rest = np.stack([t, h, w], axis=1).astype(np.int32)

        thw = np.concatenate([mandatory.astype(np.int32), rest], axis=0)
        return thw, int(thw.shape[0]), int(hb), int(wb), int(S_total), int(S_valid)

    # default: pure global topk (no mandatory first-frame patches)
    flat = scores.reshape(-1)  # (T*S_total)
    _, topidx = torch.topk(flat, k=int(K), largest=True, sorted=True)
    topidx_np = topidx.cpu().numpy().astype(np.int64)

    t = (topidx_np // S_total).astype(np.int32)
    s = (topidx_np % S_total).astype(np.int32)
    h = (s // wb).astype(np.int32)
    w = (s % wb).astype(np.int32)
    thw = np.stack([t, h, w], axis=1).astype(np.int32)
    return thw, int(K), int(hb), int(wb), int(S_total), int(S_valid)


# -------------------- sharding helper --------------------
def shard_dataset_to_jsonl(dataset_path: str, out_dir: str, num_shards: int = 8):
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    p = Path(dataset_path)
    stem = p.stem
    ns = int(max(1, num_shards))

    outs = []
    for sid in range(ns):
        op = out_dir_p / f"{stem}.shard{sid:02d}of{ns:02d}.jsonl"
        outs.append(open(op, "w", encoding="utf-8"))

    t0 = time.perf_counter()
    seen = 0
    wrote = [0] * ns

    try:
        for row_idx, item in enumerate(iter_dataset_items(dataset_path)):
            seen += 1
            sample_id = item.get("id")
            video_path = item.get("video")
            if (sample_id is None) or (not video_path):
                continue

            shard_key = f"{video_path}|{str(sample_id)}|{row_idx}"
            sid_hash = int(hashlib.sha1(shard_key.encode("utf-8")).hexdigest()[:8], 16)
            shard_id = sid_hash % ns

            out_item = dict(item)
            out_item["_row_idx"] = int(row_idx)
            outs[shard_id].write(json.dumps(out_item, ensure_ascii=False) + "\n")
            wrote[shard_id] += 1

            if (seen % 100000) == 0:
                elapsed = time.perf_counter() - t0
                print(f"[ShardDataset] seen={seen} rate={seen/max(1e-6,elapsed):.1f} lines/s")
    finally:
        for f in outs:
            try:
                f.flush()
                f.close()
            except Exception:
                pass

    elapsed = time.perf_counter() - t0
    print(f"[ShardDataset] DONE seen={seen} elapsed={elapsed:.1f}s")
    for sid in range(ns):
        print(f"[ShardDataset] shard {sid:02d}/{ns:02d} wrote={wrote[sid]}")


# -------------------- per-sample worker --------------------
def _worker_one_sample_safe(args):
    try:
        (
            sample_id, video_path, out_root,
            seq_len, sample_start_offset, patch_size, square_size, keep_frames_equiv, padding_policy,
            mv_unit_div, mv_pct, res_pct, fuse_mode, w_mv, w_res, mv_compensate, res_use_grad,
            keep_uniform_first,
            keep_first_full_frame,
            check_codec, allowed_codecs, sample_key,
        ) = args

        if int(check_codec) == 1:
            codec = _ffprobe_codec_name(video_path)
            if codec and (codec not in allowed_codecs):
                return ("unsupported", str(sample_id), str(video_path), codec)
            if codec and (not _is_supported_codec(codec)):
                return ("unsupported", str(sample_id), str(video_path), codec)

        out_root = Path(out_root)
        out_root.mkdir(parents=True, exist_ok=True)

        vp = Path(video_path)
        path_hash = hashlib.sha1(str(vp).encode("utf-8")).hexdigest()[:8]
        sk_hash = hashlib.sha1(str(sample_key).encode("utf-8")).hexdigest()[:8]
        out_id = f"sample_{sample_id}__{vp.stem}__{path_hash}__{sk_hash}"
        out_prefix = out_root / out_id

        fused_vol, frame_ids, (H0, W0), pad_info, valid_mask = compute_fused_volume_square576(
            str(vp),
            seq_len=int(seq_len),
            sample_start_offset=int(sample_start_offset),
            square_size=int(square_size),
            patch_size=int(patch_size),
            mv_unit_div=float(mv_unit_div),
            mv_pct=float(mv_pct),
            res_pct=float(res_pct),
            fuse_mode=str(fuse_mode),
            w_mv=float(w_mv),
            w_res=float(w_res),
            mv_compensate=str(mv_compensate),
            res_use_grad=bool(res_use_grad),
        )

        thw, K, hb, wb, S_total, S_valid = global_topk_thw_square576(
            fused_vol, valid_mask,
            patch_size=int(patch_size),
            keep_frames_equiv=int(keep_frames_equiv),
            padding_policy=str(padding_policy),
            keep_uniform_first=bool(keep_uniform_first),
            keep_first_full_frame=bool(keep_first_full_frame),
        )
        # --- Defensive check: never allow padding (black border) patches in thw ---
        # Stage1 already masks invalid patches before topk, but we keep this guard to
        # avoid any unexpected corner case.
        bad = 0
        pol = (str(padding_policy) or "exclude").lower()
        if thw.shape[0] > 0:
            try:
                sel_ok = valid_mask[thw[:, 1], thw[:, 2]]
                bad = int((~sel_ok).sum())
                if pol == "exclude" and bad > 0:
                    thw = thw[sel_ok]
                    K = int(thw.shape[0])
            except Exception:
                raise RuntimeError("valid_mask/thw check failed")

        np.save(str(out_prefix) + f"_T{seq_len}_square{square_size}_p{patch_size}_K{K}.visidx_thw.npy", thw, allow_pickle=False)
        np.save(str(out_prefix) + f"_T{seq_len}_square{square_size}_p{patch_size}_K{K}.frame_ids.npy",
                np.asarray(frame_ids, dtype=np.int32), allow_pickle=False)

        ys, xs = np.where(valid_mask)
        valid_ranges = {
            "S_valid": int(S_valid),
            "hb_total": int(hb),
            "wb_total": int(wb),
        }
        if len(ys) > 0:
            valid_ranges.update({
                "valid_h_start": int(ys.min()),
                "valid_h_end": int(ys.max() + 1),
                "valid_w_start": int(xs.min()),
                "valid_w_end": int(xs.max() + 1),
            })

        meta = {
            "sample_id": str(sample_id),
            "sample_key": str(sample_key),
            "video": str(vp),
            "T": int(seq_len),
            "frame_ids": None,  # frame_ids are saved separately
            "preprocess": "square576",
            "square_size": int(square_size),
            "patch": int(patch_size),
            "keep_frames_equiv": int(keep_frames_equiv),
            "padding_policy": str(padding_policy),
            "keep_uniform_first": bool(keep_uniform_first),
            "keep_first_full_frame": bool(keep_first_full_frame),
            "H0": int(H0),
            "W0": int(W0),
            "pad_info": pad_info,
            "hb": int(hb),
            "wb": int(wb),
            "S_total": int(S_total),
            "S_valid": int(S_valid),
            "K": int(K),
            "padding_patch_filtered": int(bad),
            "valid_ranges": valid_ranges,
        }
        with open(str(out_prefix) + f"_T{seq_len}_square{square_size}_p{patch_size}_K{K}.meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        # free
        try:
            import gc
            del fused_vol
            gc.collect()
        except Exception:
            pass

        return ("ok", str(sample_id), str(video_path), int(K))
    except Exception as e:
        return ("fail", str(args[0]), str(args[1]), repr(e), traceback.format_exc(limit=5))


def process_dataset(
    dataset_path: str,
    out_root: str,
    max_samples: int,
    num_shards: int,
    shard_id: int,
    seq_len: int,
    sample_start_offset: int,
    patch_size: int,
    square_size: int,
    keep_frames_equiv: int,
    padding_policy: str,
    keep_uniform_first: bool,
    keep_first_full_frame: bool,
    mv_unit_div: float,
    mv_pct: float,
    res_pct: float,
    fuse_mode: str,
    w_mv: float,
    w_res: float,
    mv_compensate: str,
    res_use_grad: bool,
    num_workers: int,
    maxtasks_per_child: int,
    log_every: int,
    check_codec: bool,
    allowed_codecs: str,
    fail_txt: str,
    unsupported_txt: str,
):
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    allow_set = set([c.strip().lower() for c in str(allowed_codecs).split(",") if c.strip()])

    num_workers = int(max(1, num_workers))
    num_shards = int(max(1, num_shards))
    shard_id = int(shard_id)
    if not (0 <= shard_id < num_shards):
        raise ValueError(f"Invalid shard_id={shard_id}, must be in [0,{num_shards})")

    # per-shard log paths
    fail_path = str(fail_txt) if fail_txt else str(out_root / f"stage1_failed.shard{shard_id:02d}of{num_shards:02d}.txt")
    unsup_path = str(unsupported_txt) if unsupported_txt else str(out_root / f"stage1_unsupported.shard{shard_id:02d}of{num_shards:02d}.txt")

    stats = {"seen": 0, "picked": 0, "skip": 0, "shard_skip": 0}
    ok = fail = 0
    t0 = time.perf_counter()

    def _iter_tasks():
        for item in iter_dataset_items(dataset_path):
            stats["seen"] += 1
            if int(max_samples) > 0 and stats["seen"] > int(max_samples):
                return

            # stable per-record row index (supports pre-sharded _row_idx)
            try:
                row_idx = int(item.get("_row_idx", stats["seen"] - 1))
            except Exception:
                row_idx = int(stats["seen"] - 1)

            sample_id = item.get("id")
            video_path = item.get("video")
            if (sample_id is None) or (not video_path):
                stats["skip"] += 1
                continue

            shard_key = f"{video_path}|{str(sample_id)}|{row_idx}"
            sid_hash = int(hashlib.sha1(shard_key.encode("utf-8")).hexdigest()[:8], 16)
            if (sid_hash % num_shards) != shard_id:
                stats["shard_skip"] += 1
                continue

            stats["picked"] += 1
            yield (
                str(sample_id), str(video_path), str(out_root),
                int(seq_len), int(sample_start_offset), int(patch_size), int(square_size), int(keep_frames_equiv), str(padding_policy),
                float(mv_unit_div), float(mv_pct), float(res_pct),
                str(fuse_mode), float(w_mv), float(w_res), str(mv_compensate), bool(res_use_grad),
                int(bool(keep_uniform_first)),
                int(bool(keep_first_full_frame)),
                int(bool(check_codec)), allow_set,
                shard_key,  # sample_key
            )

    ctx = get_context("spawn")
    nw = min(cpu_count(), int(num_workers))
    mt = int(maxtasks_per_child)
    mt = None if mt <= 0 else mt

    with open(fail_path, "a", encoding="utf-8") as ffail, open(unsup_path, "a", encoding="utf-8") as funsup:
        if nw <= 1:
            for i, task in enumerate(_iter_tasks(), 1):
                ret = _worker_one_sample_safe(task)
                if ret[0] == "ok":
                    ok += 1
                elif ret[0] == "unsupported":
                    funsup.write(f"{ret[1]}\t{ret[2]}\t{ret[3]}\n")
                else:
                    fail += 1
                    ffail.write(f"{ret[1]}\t{ret[2]}\t{ret[3]}\n")

                if (i % int(log_every)) == 0:
                    elapsed = time.perf_counter() - t0
                    avg = elapsed / float(max(1, ok + fail))
                    print(f"[Stage1-square576][shard {shard_id}/{num_shards}][1w] "
                          f"seen={stats['seen']} picked={stats['picked']} ok={ok} fail={fail} "
                          f"skip={stats['skip']} shard_skip={stats['shard_skip']} elapsed={elapsed:.1f}s avg={avg:.3f}s/picked")
        else:
            pool = ctx.Pool(processes=nw, maxtasksperchild=mt)
            try:
                for i, ret in enumerate(pool.imap_unordered(_worker_one_sample_safe, _iter_tasks(), chunksize=1), 1):
                    if ret[0] == "ok":
                        ok += 1
                    elif ret[0] == "unsupported":
                        funsup.write(f"{ret[1]}\t{ret[2]}\t{ret[3]}\n")
                    else:
                        fail += 1
                        ffail.write(f"{ret[1]}\t{ret[2]}\t{ret[3]}\n")

                    if (i % int(log_every)) == 0:
                        elapsed = time.perf_counter() - t0
                        avg = elapsed / float(max(1, ok + fail))
                        print(f"[Stage1-square576][shard {shard_id}/{num_shards}][{nw}w] "
                              f"seen={stats['seen']} picked={stats['picked']} ok={ok} fail={fail} "
                              f"skip={stats['skip']} shard_skip={stats['shard_skip']} elapsed={elapsed:.1f}s avg={avg:.3f}s/picked")
            finally:
                pool.close()
                pool.join()

    elapsed = time.perf_counter() - t0
    avg = elapsed / float(max(1, ok + fail))
    print(f"[Stage1-square576][shard {shard_id}/{num_shards}] DONE "
          f"seen={stats['seen']} picked={stats['picked']} ok={ok} fail={fail} "
          f"skip={stats['skip']} shard_skip={stats['shard_skip']} elapsed={elapsed:.1f}s avg={avg:.3f}s/picked")


def main():
    ap = argparse.ArgumentParser("Stage1 visidx square576")
    ap.add_argument(
        "--keep_uniform_first",
        action="store_true",
        help="Force-include all content-region patches from the first sampled frame (t=0); remaining budget is filled by topk from later frames",
    )
    ap.add_argument(
        "--keep_first_full_frame",
        action="store_true",
        help="Keep ALL 576x576 patches (including padding) from the first sampled frame (t=0, after --sample_start_offset). The remaining budget is allocated to later frames as (keep_frames_equiv-1) full-frame patch tokens (i.e., (36*36)*(keep_frames_equiv-1)).",
    )
    ap.add_argument("--dataset_path", type=str, default="", help="dataset .json (array) or .jsonl; each item has fields: id, video")
    ap.add_argument("--out_root", type=str, required=False, default="", help="output dir")
    ap.add_argument("--shard_dataset_out_dir", type=str, default="", help="If set, only shard dataset to per-shard jsonl and exit.")
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--shard_id", type=int, default=0)
    ap.add_argument("--max_samples", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=1000)

    ap.add_argument("--sequence_length", type=int, default=64)
    ap.add_argument(
        "--sample_start_offset",
        type=int,
        default=0,
        help="Uniform sampling starts from this frame index instead of 0 (useful when the beginning is black/white). The first sampled frame (t=0) will be this offset.",
    )
    ap.add_argument("--patch_size", type=int, default=16)
    ap.add_argument("--square_size", type=int, default=576)
    ap.add_argument("--keep_frames_equiv", type=int, default=8)
    ap.add_argument(
        "--padding_policy",
        type=str,
        default="exclude",
        choices=["exclude", "zero"],
        help="Padding patch handling: exclude=mask out padding patches; zero=allow padding patches with near-zero score and use full 36x36 budget",
    )

    ap.add_argument("--mv_unit_div", type=float, default=4.0)
    ap.add_argument("--mv_pct", type=float, default=95.0)
    ap.add_argument("--res_pct", type=float, default=95.0)
    ap.add_argument("--fuse_mode", type=str, default="weighted", choices=["weighted", "max", "sum", "geomean"])
    ap.add_argument("--w_mv", type=float, default=1.0)
    ap.add_argument("--w_res", type=float, default=1.0)
    ap.add_argument("--mv_compensate", type=str, default="median", choices=["none", "median", "mean"])
    ap.add_argument("--res_use_grad", action="store_true")

    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--maxtasks_per_child", type=int, default=2000)

    ap.add_argument("--check_codec", action="store_true")
    ap.add_argument("--allowed_codecs", type=str, default="h264,hevc,h265,avc1")
    ap.add_argument("--fail_txt", type=str, default="")
    ap.add_argument("--unsupported_txt", type=str, default="")

    args = ap.parse_args()

    if not _HAS_CV_READER:
        raise RuntimeError("cv_reader is not available. Please ensure preprocess image is correct.")

    if args.shard_dataset_out_dir:
        if not args.dataset_path:
            raise ValueError("--shard_dataset_out_dir requires --dataset_path")
        shard_dataset_to_jsonl(args.dataset_path, args.shard_dataset_out_dir, num_shards=int(args.num_shards))
        return

    if not args.dataset_path:
        raise ValueError("Please provide --dataset_path")
    if not args.out_root:
        raise ValueError("Please provide --out_root")

    process_dataset(
        dataset_path=args.dataset_path,
        out_root=args.out_root,
        max_samples=int(args.max_samples),
        num_shards=int(args.num_shards),
        shard_id=int(args.shard_id),
        seq_len=int(args.sequence_length),
        sample_start_offset=int(args.sample_start_offset),
        patch_size=int(args.patch_size),
        square_size=int(args.square_size),
        keep_frames_equiv=int(args.keep_frames_equiv),
        padding_policy=str(args.padding_policy),
        keep_uniform_first=bool(args.keep_uniform_first),
        keep_first_full_frame=bool(args.keep_first_full_frame),
        mv_unit_div=float(args.mv_unit_div),
        mv_pct=float(args.mv_pct),
        res_pct=float(args.res_pct),
        fuse_mode=str(args.fuse_mode),
        w_mv=float(args.w_mv),
        w_res=float(args.w_res),
        mv_compensate=str(args.mv_compensate),
        res_use_grad=bool(args.res_use_grad),
        num_workers=int(args.num_workers),
        maxtasks_per_child=int(args.maxtasks_per_child),
        log_every=int(args.log_every),
        check_codec=bool(args.check_codec),
        allowed_codecs=str(args.allowed_codecs),
        fail_txt=str(args.fail_txt),
        unsupported_txt=str(args.unsupported_txt),
    )


if __name__ == "__main__":
    main()