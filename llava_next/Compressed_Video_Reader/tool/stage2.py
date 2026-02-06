#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse
import json
import hashlib
import time
import traceback
from pathlib import Path
from multiprocessing import get_context
import numpy as np

# --- Caching for rewrite mode pack lookup ---
import os
from typing import Dict, List, Optional, Tuple

# Global cache for rewrite-mode pack lookup (to avoid per-sample expensive rglob)
_PACK_INDEX_ROOT: Optional[str] = None
_PACK_INDEX_EXACT: Dict[Tuple[str, str], str] = {}   # (stem, hash) -> pack_dir
_PACK_INDEX_STEM: Dict[str, List[str]] = {}          # stem -> [pack_dir, ...]
# --- helpers for rewrite mode ---
def _build_pack_index(pack_root: Path) -> None:
    """Build a lookup index for pack directories under pack_root.

    This avoids calling rglob() for every sample during rewrite, which is extremely slow on NFS.
    We index directories whose basename starts with:
      - sample_<sid>__<stem>__<hash>
      - video__<stem>__<hash>
    """
    global _PACK_INDEX_ROOT, _PACK_INDEX_EXACT, _PACK_INDEX_STEM

    root_str = str(pack_root)
    if _PACK_INDEX_ROOT == root_str and _PACK_INDEX_EXACT:
        return

    _PACK_INDEX_ROOT = root_str
    _PACK_INDEX_EXACT = {}
    _PACK_INDEX_STEM = {}

    # Fast directory walk
    for cur, dirs, _files in os.walk(root_str):
        for dn in dirs:
            if not (dn.startswith("sample_") or dn.startswith("video__")):
                continue

            full = os.path.join(cur, dn)
            parts = dn.split("__")
            # sample_<sid>__<stem>__<hash>
            if dn.startswith("sample_") and len(parts) >= 3:
                stem = parts[1]
                h = parts[2]
            # video__<stem>__<hash>
            elif dn.startswith("video__") and len(parts) >= 3:
                stem = parts[1]
                h = parts[2]
            else:
                continue

            key = (stem, h)
            if key not in _PACK_INDEX_EXACT:
                _PACK_INDEX_EXACT[key] = full

            _PACK_INDEX_STEM.setdefault(stem, []).append(full)

    # Make stem lists deterministic
    for k in list(_PACK_INDEX_STEM.keys()):
        _PACK_INDEX_STEM[k] = sorted(_PACK_INDEX_STEM[k])


# --- helpers for rewrite mode ---
def _safe_int(x, default=0):
    try:
        return int(x)
    except Exception:
        return default


def _video_stem_and_hash(video_path: str):
    vp = Path(str(video_path))
    stem = vp.stem
    path_hash = hashlib.sha1(str(vp).encode("utf-8")).hexdigest()[:8]
    return stem, path_hash


def resolve_pack_for_video(pack_root: Path, video_path: str, num_images: int = 8):
    """Find the on-disk pack (8 images + optional positions_thw.npy) for a given video.

    We support both video-level naming and legacy sample-level naming:
      - video__{stem}__{hash}/  (recommended)
      - sample_*__{stem}__{hash}/

    Returns (mosaics_list, positions_path_or_None, pack_dir_or_None).
    """
    stem, path_hash = _video_stem_and_hash(video_path)

    # Build cache once (rewrite mode can call this millions of times)
    _build_pack_index(pack_root)

    cand_dirs = []

    # 1) exact match by (stem, hash)
    d0 = _PACK_INDEX_EXACT.get((stem, path_hash))
    if d0 and Path(d0).is_dir():
        cand_dirs.append(Path(d0))

    # 2) fallback: ignore hash (mount prefix differences can make path_hash unstable)
    if not cand_dirs:
        for d in _PACK_INDEX_STEM.get(stem, []):
            if Path(d).is_dir():
                cand_dirs.append(Path(d))
                break

    if not cand_dirs:
        return None, None, None

    d = cand_dirs[0]
    mosaics = []
    for i in range(int(num_images)):
        p = d / f"{stem}_{i:03d}.jpg"
        if not p.is_file():
            # fallback: some earlier runs used .png or different name
            p2 = d / f"{stem}_{i:03d}.png"
            if p2.is_file():
                p = p2
            else:
                return None, None, d
        mosaics.append(str(p))

    pos_path = d / "positions_thw.npy"
    if not pos_path.is_file():
        pos_path = None

    return mosaics, (str(pos_path) if pos_path is not None else None), d


try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False


# Pad an image to multiples of 32 (bottom/right only)
def pad_to_multiple(img: np.ndarray, mult: int = 32) -> np.ndarray:
    """Pad bottom/right with zeros so H and W are divisible by mult."""
    H, W = img.shape[:2]
    mult = int(mult)
    pad_h = (mult - (H % mult)) % mult
    pad_w = (mult - (W % mult)) % mult
    if pad_h == 0 and pad_w == 0:
        return img
    return cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))


def rewrite_conversations_video_to_images(convs, num_images: int = 8):
    """Replace first <video> with <image> * num_images.

    Important: if the conversation is already image-formatted (contains <image> in the first human turn),
    we leave it unchanged to avoid duplicating <image> tokens.
    """
    if not isinstance(convs, list):
        return convs

    out = []
    replaced = False
    prefix = ("<image>\n" * int(num_images)).rstrip("\n")

    for m in convs:
        if not isinstance(m, dict):
            out.append(m)
            continue

        mm = dict(m)
        if (not replaced) and mm.get("from") == "human" and isinstance(mm.get("value"), str):
            v = mm["value"]

            # Case 1: has <video> -> replace once
            if "<video>" in v:
                mm["value"] = v.replace("<video>", prefix, 1)
                replaced = True

            # Case 2: already has <image> -> assume already in image format; do not modify
            elif "<image>" in v:
                replaced = True

            # Case 3: no <video> and no <image> -> prepend images once (legacy behavior)
            else:
                mm["value"] = prefix + "\n" + v
                replaced = True

        out.append(mm)

    return out


def iter_json_array(path: str):
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


def _shard_of_key(key: str, num_shards: int) -> int:
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return int(h[:8], 16) % int(num_shards)


def decode_frames_by_ids(video_path: str, frame_ids):
    if not _HAS_CV2:
        raise RuntimeError("cv2 is required for stage2 decoding.")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    frames = []
    for fid in frame_ids:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fid))
        ok, fr = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError(f"Failed to read frame fid={fid} from {video_path}")
        frames.append(fr)  # BGR uint8
    cap.release()
    return frames


def resize_longer_pad_square_bgr(fr_bgr: np.ndarray, out_size: int = 576):
    """Match stage1: scale = out/max(H,W), Hn/Wn = round(H*scale)/round(W*scale), then centered pad."""
    H, W = fr_bgr.shape[:2]
    out_size = int(out_size)
    scale = float(out_size) / float(max(H, W))
    Hn = max(1, int(round(H * scale)))
    Wn = max(1, int(round(W * scale)))

    fr_rs = cv2.resize(fr_bgr, (Wn, Hn), interpolation=cv2.INTER_LINEAR)
    pad_top = (out_size - Hn) // 2
    pad_left = (out_size - Wn) // 2
    pad_bottom = out_size - Hn - pad_top
    pad_right = out_size - Wn - pad_left

    out = cv2.copyMakeBorder(
        fr_rs, pad_top, pad_bottom, pad_left, pad_right,
        borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )
    info = dict(
        scale=scale, Hn=int(Hn), Wn=int(Wn),
        pad_top=int(pad_top), pad_left=int(pad_left),
        pad_bottom=int(pad_bottom), pad_right=int(pad_right),
        out_size=int(out_size), H0=int(H), W0=int(W),
    )
    return out, info


def find_stage1_meta(vis_root: Path, sample_id: str, video_path: str, T: int, patch: int):
    """Robustly find a meta.json for this sample, even if filename contains extra hash."""
    vp = Path(video_path)
    path_hash = hashlib.sha1(str(vp).encode("utf-8")).hexdigest()[:8]
    pattern = f"sample_{sample_id}__{vp.stem}__{path_hash}__*_T{int(T)}*_p{int(patch)}*_K*.meta.json"
    metas = sorted(vis_root.rglob(pattern))
    if metas:
        return metas[0]
    # fallback: looser search
    metas2 = sorted(vis_root.rglob(f"sample_{sample_id}__{vp.stem}__{path_hash}__*.meta.json"))
    return metas2[0] if metas2 else None


def _worker_one(task):
    """
    task = (it, vis_root, out_img_root, out_square, T, patch, num_images, layout,
            skip_missing, write_positions, first_full,
            export_patches)
    """
    (it, vis_root_str, out_img_root_str, square_size, T, patch, num_images, layout,
     skip_missing, write_positions, first_full,
     export_patches) = task

    try:
        sid = str(it.get("id", "unknown"))
        vid = it.get("video")
        if sid == "unknown" or not vid:
            return ("skip", None)

        vis_root = Path(vis_root_str)
        out_img_root = Path(out_img_root_str)

        meta_path = find_stage1_meta(vis_root, sid, vid, T=int(T), patch=int(patch))
        if meta_path is None:
            if skip_missing:
                return ("skip", None)
            raise RuntimeError(f"Cannot find stage1 meta for id={sid} video={vid}")

        meta = json.loads(meta_path.read_text(encoding="utf-8"))

        padding_policy = str(meta.get("padding_policy", "exclude")).lower()

        # valid (content) region in patch grid coordinates (exclude padding black borders)
        vr = meta.get("valid_ranges", None)
        if vr is None:
            if padding_policy != "zero":
                raise RuntimeError(f"meta missing valid_ranges: {meta_path}")
            # fallback defaults (unused when padding_policy==zero)
            vr = {"valid_h_start": 0, "valid_h_end": 0, "valid_w_start": 0, "valid_w_end": 0}

        vh = int(vr.get("valid_h_end", 0)) - int(vr.get("valid_h_start", 0))  # number of valid patch rows
        vw = int(vr.get("valid_w_end", 0)) - int(vr.get("valid_w_start", 0))  # number of valid patch cols
        vhs = int(vr.get("valid_h_start", 0))
        vws = int(vr.get("valid_w_start", 0))

        if padding_policy != "zero":
            if vh <= 0 or vw <= 0:
                raise RuntimeError(f"bad valid_ranges in meta: {vr}")

        # Expect square576 stage1
        sq = int(meta.get("square_size", square_size))
        if sq != int(square_size):
            # allow but warn by embedding in meta_out
            pass

        K = int(meta["K"])
        visidx_path = meta_path.with_suffix("").with_suffix(".visidx_thw.npy")  # remove .json then add .npy
        # safer replacement:
        visidx_path = Path(str(meta_path).replace(".meta.json", ".visidx_thw.npy"))
        frame_ids_path = Path(str(meta_path).replace(".meta.json", ".frame_ids.npy"))

        thw = np.load(visidx_path).astype(np.int32)
        frame_ids = np.load(frame_ids_path).astype(np.int32)

        if thw.ndim != 2 or thw.shape[1] != 3:
            raise RuntimeError(f"Bad thw shape: {thw.shape}")

        num_images = int(num_images)
        if K != thw.shape[0]:
            K = thw.shape[0]
        if K % num_images != 0:
            raise RuntimeError(f"K={K} not divisible by num_images={num_images}")
        S_img = K // num_images

        # decode + preprocess all 64 sampled frames
        frames = decode_frames_by_ids(vid, frame_ids.tolist())
        frames_sq = []
        pad_info0 = None
        for fr in frames:
            fr_sq, info = resize_longer_pad_square_bgr(fr, out_size=int(square_size))
            frames_sq.append(fr_sq)
            if pad_info0 is None:
                pad_info0 = info

        hb = int(square_size) // int(patch)
        wb = int(square_size) // int(patch)

        # output directory per sample
        vp = Path(vid)
        path_hash = hashlib.sha1(str(vp).encode("utf-8")).hexdigest()[:8]
        # keep a stable directory name
        sample_dir = out_img_root / f"sample_{sid}__{vp.stem}__{path_hash}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        export_patches = bool(export_patches)
        patch_dir = None
        patch_paths = []
        patch_positions = []
        patch_global_idx = 0
        if export_patches:
            patch_dir = sample_dir / "patches"
            patch_dir.mkdir(parents=True, exist_ok=True)
        mosaics = []
        pos_out = []

        # optional sort for determinism
        if layout in ("time_spatial", "spatial"):
            order = np.lexsort((thw[:, 2], thw[:, 1], thw[:, 0]))
            thw = thw[order]

        for i in range(num_images):
            start = i * S_img
            end = (i + 1) * S_img
            chunk = thw[start:end]

            if export_patches:
                # Export each selected patch as its own small image.
                # Filename: <imgid>_<row>_<col>_<t>_<h>_<w>.jpg
                if patch_dir is None:
                    raise RuntimeError("export_patches enabled but patch_dir is None")

                grid_n = int(int(square_size) // int(patch)) if int(patch) > 0 else 0
                if grid_n <= 0:
                    raise RuntimeError(f"bad patch={patch} or square_size={square_size}")

                # For patch_size=72 and square_size=576, grid_n==8; we name by (row,col) in an 8x8 grid.
                # If chunk has more than grid_n^2, we still export all, continuing row-major indexing.
                for j, (t, h, w) in enumerate(chunk.tolist()):
                    rr = int(j) // int(grid_n)
                    cc = int(j) % int(grid_n)

                    y0 = int(h) * int(patch)
                    x0 = int(w) * int(patch)
                    patch_bgr = frames_sq[int(t)][y0:y0+int(patch), x0:x0+int(patch)]
                    if patch_bgr.size == 0:
                        continue

                    # Filename: <imgid>_<row>_<col>_<t>_<h>_<w>.jpg
                    fn = f"{int(i)}_{int(rr)}_{int(cc)}_{int(t)}_{int(h)}_{int(w)}.jpg"
                    out_p = patch_dir / fn
                    try:
                        cv2.imwrite(str(out_p), patch_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                    except Exception:
                        cv2.imwrite(str(out_p), patch_bgr)

                    patch_paths.append(str(out_p))
                    patch_positions.append([int(t), int(h), int(w)])
                    patch_global_idx += 1

                # do not write mosaics in export_patches mode
                continue

            # ---- legacy mosaic mode (unchanged) ----
            canvas = np.zeros((int(square_size), int(square_size), 3), dtype=np.uint8)

            if bool(first_full) and i == 0:
                # put full t=0 frame
                canvas[:] = frames_sq[0]
                # Also record the positions for all patches of t=0 frame
                for hh in range(hb):
                    for ww in range(wb):
                        pos_out.append([0, int(hh), int(ww)])
            else:
                if layout == "spatial":
                    # place patches back to their (h,w)
                    for (t, h, w) in chunk.tolist():
                        y0 = int(h) * int(patch)
                        x0 = int(w) * int(patch)
                        patch_bgr = frames_sq[int(t)][y0:y0+int(patch), x0:x0+int(patch)]
                        canvas[y0:y0+int(patch), x0:x0+int(patch)] = patch_bgr
                        pos_out.append([int(t), int(h), int(w)])
                else:
                    # time_spatial (dense pack)
                    # - padding_policy=="zero": stage1 selects fixed budget of hb*wb patches per image
                    #   so we pack a FULL hb*wb grid into the square canvas.
                    # - otherwise: pack ONLY valid (content) grid into the canvas at valid offset.

                    if padding_policy == "zero":
                        if S_img != hb * wb:
                            raise RuntimeError(
                                f"padding_policy=zero expects S_img==hb*wb ({hb}*{wb}={hb*wb}), got S_img={S_img}; check stage1 K"
                            )
                        for j, (t, h, w) in enumerate(chunk.tolist()):
                            rr = j // wb
                            cc = j % wb
                            ys = rr * int(patch)
                            xs = cc * int(patch)

                            y0 = int(h) * int(patch)
                            x0 = int(w) * int(patch)
                            patch_bgr = frames_sq[int(t)][y0:y0+int(patch), x0:x0+int(patch)]

                            canvas[ys:ys+int(patch), xs:xs+int(patch)] = patch_bgr
                            pos_out.append([int(t), int(h), int(w)])

                    else:
                        # legacy: pack ONLY valid (content) patches into a vh x vw grid and place at valid offset
                        if S_img != vh * vw:
                            raise RuntimeError(
                                f"S_img={S_img} does not match valid grid vh*vw={vh*vw}; check stage1 K/valid_ranges"
                            )

                        for j, (t, h, w) in enumerate(chunk.tolist()):
                            rr = j // vw
                            cc = j % vw
                            ys = (vhs + rr) * int(patch)
                            xs = (vws + cc) * int(patch)

                            y0 = int(h) * int(patch)
                            x0 = int(w) * int(patch)
                            patch_bgr = frames_sq[int(t)][y0:y0+int(patch), x0:x0+int(patch)]

                            canvas[ys:ys+int(patch), xs:xs+int(patch)] = patch_bgr
                            pos_out.append([int(t), int(h), int(w)])

            out_img = sample_dir / f"{vp.stem}_{i:03d}.jpg"
            cv2.imwrite(str(out_img), canvas)
            mosaics.append(str(out_img))

        pos_path = None
        if export_patches:
            patch_positions_arr = np.asarray(patch_positions, dtype=np.int32)
            pos_path = (patch_dir / "positions_thw.npy") if patch_dir is not None else (sample_dir / "positions_thw.npy")
            np.save(pos_path, patch_positions_arr, allow_pickle=False)
        elif bool(write_positions):
            pos_out = np.asarray(pos_out, dtype=np.int32)
            pos_path = sample_dir / "positions_thw.npy"
            np.save(pos_path, pos_out, allow_pickle=False)

        convs = it.get("conversations", []) if export_patches else rewrite_conversations_video_to_images(it.get("conversations", []), num_images=num_images)

        out_item = {
            "id": sid,
            "conversations": convs,
            "image": (patch_paths if export_patches else mosaics),
        }
        if (export_patches or bool(write_positions)) and pos_path is not None:
            out_item["positions_thw"] = str(pos_path)

        return ("ok", json.dumps(out_item, ensure_ascii=False))

    except Exception as e:
        return ("fail", f"{it.get('id','unknown')}\t{it.get('video','')}\t{repr(e)}\n{traceback.format_exc(limit=3)}")


# --- worker for rewrite mode ---
def _worker_rewrite_one(task):
    """Rewrite one dataset item by attaching image pack paths resolved from video path."""
    (it, pack_root_str, num_images, include_positions, skip_missing_packs) = task
    try:
        sid = str(it.get("id", "unknown"))
        vid = it.get("video")
        if sid == "unknown" or not vid:
            return ("skip", None)

        pack_root = Path(pack_root_str)
        mosaics, pos_path, _ = resolve_pack_for_video(pack_root, str(vid), num_images=int(num_images))
        if mosaics is None:
            if skip_missing_packs:
                return ("skip", None)
            raise RuntimeError(f"Cannot resolve pack for video={vid}")

        convs = rewrite_conversations_video_to_images(it.get("conversations", []), num_images=int(num_images))
        out_item = {
            "id": sid,
            "conversations": convs,
            "image": mosaics,
        }
        if bool(include_positions) and pos_path is not None:
            out_item["positions_thw"] = pos_path

        return ("ok", json.dumps(out_item, ensure_ascii=False))

    except Exception as e:
        return ("fail", f"{it.get('id','unknown')}\t{it.get('video','')}\t{repr(e)}\n{traceback.format_exc(limit=3)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, default="pack", choices=["pack", "rewrite"],
                    help="pack: run stage2 to generate images; rewrite: only rewrite json/jsonl to reuse per-video packs")

    # shared
    ap.add_argument("--input_dataset", type=str, required=True, help="json/jsonl with fields: id, video, conversations")
    ap.add_argument("--out_jsonl", type=str, required=True, help="output jsonl path")
    ap.add_argument("--num_images", type=int, default=8)

    ap.add_argument("--num_workers", type=int, default=1)
    ap.add_argument("--maxtasks_per_child", type=int, default=2000)
    ap.add_argument("--max_samples", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=200)

    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--shard_id", type=int, default=0)
    ap.add_argument(
        "--force_shard_out",
        action="store_true",
        help="Always write per-shard output jsonl (adds .shardXXofYY.jsonl suffix) to avoid multi-machine write conflicts.",
    )

    ap.add_argument("--fail_txt", type=str, default="", help="write failures to this txt (default: next to out_jsonl with shard suffix)")

    # --- pack mode args (existing stage2) ---
    ap.add_argument("--visidx_root", type=str, default="", help="stage1 outputs dir (pack mode)")
    ap.add_argument("--out_image_root", type=str, default="", help="where to save 8 images per sample/video (pack mode) OR where packs already exist (rewrite mode)")
    ap.add_argument("--square_size", type=int, default=576)
    ap.add_argument("--T", type=int, default=64)
    ap.add_argument("--patch", type=int, default=16)
    ap.add_argument("--layout", type=str, default="time_spatial", choices=["time_spatial", "spatial"])
    ap.add_argument("--first_full", action="store_true", help="If set, image0 will be full t=0 frame (square576). Others are packed patches.")
    ap.add_argument("--skip_missing", action="store_true")
    ap.add_argument("--write_positions", action="store_true")
    ap.add_argument(
        "--export_patches",
        action="store_true",
        help=(
            "Pack mode: export each selected patch as an individual image under <sample_dir>/patches. "
            "Filenames: <imgid>_<row>_<col>_<t>_<h>_<w>.jpg (thw comes from stage1 visidx). "
            "When enabled, mosaics are NOT written; jsonl 'image' will point to the patch images."
        ),
    )

    # --- rewrite mode args ---
    ap.add_argument("--include_positions", action="store_true", help="rewrite mode: include positions_thw path if exists")
    ap.add_argument("--skip_missing_packs", action="store_true", help="rewrite mode: skip if pack not found")

    args = ap.parse_args()

    in_path = Path(args.input_dataset)
    out_jsonl_base = Path(args.out_jsonl)

    num_shards = int(max(1, args.num_shards))
    shard_id = int(args.shard_id)
    if not (0 <= shard_id < num_shards):
        raise ValueError(f"bad shard_id={shard_id}")

    # Write per-shard outputs to avoid concurrent write corruption on shared filesystems.
    # If num_shards>1, we always shard. If --force_shard_out is set, we shard even for 1 shard.
    out_jsonl = out_jsonl_base
    if (num_shards > 1) or bool(args.force_shard_out):
        out_jsonl = Path(str(out_jsonl_base) + f".shard{shard_id:02d}of{num_shards:02d}.jsonl")

    fail_txt = args.fail_txt
    if not fail_txt:
        fail_txt = str(Path(str(out_jsonl) + ".fail.txt"))

    # Ensure output directories exist (both for jsonl and fail txt)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    Path(fail_txt).parent.mkdir(parents=True, exist_ok=True)

    # NOTE: rewrite mode does NOT require cv2.
    if args.mode == "pack" and (not _HAS_CV2):
        raise RuntimeError("cv2 is required for pack mode.")

    # roots
    out_img_root = Path(args.out_image_root) if args.out_image_root else None
    if out_img_root is None:
        raise ValueError("--out_image_root is required")
    out_img_root.mkdir(parents=True, exist_ok=True)

    vis_root = Path(args.visidx_root) if args.visidx_root else None
    if args.mode == "pack":
        if vis_root is None:
            raise ValueError("--visidx_root is required for pack mode")

    t0 = time.perf_counter()
    seen = picked = shard_skip = 0
    ok = fail = skip = 0

    def _iter_tasks_pack():
        nonlocal seen, picked, shard_skip
        for it in iter_dataset_items(str(in_path)):
            seen += 1
            vid = it.get("video")
            sid = str(it.get("id", "unknown"))
            if not vid or sid == "unknown":
                continue

            row_idx = _safe_int(it.get("_row_idx", seen - 1), default=seen - 1)
            key = f"{vid}|{sid}|{row_idx}"
            if _shard_of_key(key, num_shards) != shard_id:
                shard_skip += 1
                continue

            picked += 1
            if int(args.max_samples) > 0 and picked > int(args.max_samples):
                return

            yield (
                it,
                str(vis_root),
                str(out_img_root),
                int(args.square_size),
                int(args.T),
                int(args.patch),
                int(args.num_images),
                str(args.layout),
                bool(args.skip_missing),
                bool(args.write_positions),
                bool(args.first_full),
                bool(args.export_patches),
            )

    def _iter_tasks_rewrite():
        nonlocal seen, picked, shard_skip
        for it in iter_dataset_items(str(in_path)):
            seen += 1
            vid = it.get("video")
            sid = str(it.get("id", "unknown"))
            if not vid or sid == "unknown":
                continue

            row_idx = _safe_int(it.get("_row_idx", seen - 1), default=seen - 1)
            key = f"{vid}|{sid}|{row_idx}"
            if _shard_of_key(key, num_shards) != shard_id:
                shard_skip += 1
                continue

            picked += 1
            if int(args.max_samples) > 0 and picked > int(args.max_samples):
                return

            yield (
                it,
                str(out_img_root),
                int(args.num_images),
                bool(args.include_positions),
                bool(args.skip_missing_packs),
            )

    ctx = get_context("spawn")
    nw = int(max(1, args.num_workers))
    mt = int(args.maxtasks_per_child)
    mt = None if mt <= 0 else mt

    iterator = _iter_tasks_pack() if args.mode == "pack" else _iter_tasks_rewrite()
    worker_fn = _worker_one if args.mode == "pack" else _worker_rewrite_one

    with open(out_jsonl, "w", encoding="utf-8") as fw, open(fail_txt, "a", encoding="utf-8") as ff:
        if nw <= 1:
            for task in iterator:
                status, payload = worker_fn(task)
                if status == "ok":
                    fw.write(payload + "\n")
                    fw.flush()
                    ok += 1
                elif status == "skip":
                    skip += 1
                else:
                    fail += 1
                    ff.write(payload + "\n")
                    ff.flush()

                if (picked % int(args.log_every)) == 0 and picked > 0:
                    elapsed = time.perf_counter() - t0
                    avg = elapsed / float(max(1, picked))
                    print(f"[Stage2-{args.mode}][shard {shard_id}/{num_shards}][1w] "
                          f"seen={seen} picked={picked} ok={ok} fail={fail} skip={skip} shard_skip={shard_skip} "
                          f"elapsed={elapsed:.1f}s avg={avg:.3f}s/picked")
        else:
            pool = ctx.Pool(processes=nw, maxtasksperchild=mt)
            try:
                for i, (status, payload) in enumerate(pool.imap_unordered(worker_fn, iterator, chunksize=1), 1):
                    if status == "ok":
                        fw.write(payload + "\n")
                        fw.flush()
                        ok += 1
                    elif status == "skip":
                        skip += 1
                    else:
                        fail += 1
                        ff.write(payload + "\n")
                        ff.flush()

                    if (i % int(args.log_every)) == 0:
                        elapsed = time.perf_counter() - t0
                        avg = elapsed / float(max(1, i))
                        print(f"[Stage2-{args.mode}][shard {shard_id}/{num_shards}][{nw}w] "
                              f"seen={seen} picked={picked} ok={ok} fail={fail} skip={skip} shard_skip={shard_skip} "
                              f"elapsed={elapsed:.1f}s avg={avg:.3f}s/picked")
            finally:
                pool.close()
                pool.join()

    elapsed = time.perf_counter() - t0
    total = ok + fail + skip
    avg = elapsed / float(max(1, total))
    print(f"[Stage2-{args.mode}] DONE total={total} ok={ok} fail={fail} skip={skip} elapsed={elapsed:.1f}s avg={avg:.3f}s/sample")
    print(f"[Stage2-{args.mode}] wrote jsonl: {out_jsonl}")
    print(f"[Stage2-{args.mode}] fail log: {fail_txt}")


if __name__ == "__main__":
    main()