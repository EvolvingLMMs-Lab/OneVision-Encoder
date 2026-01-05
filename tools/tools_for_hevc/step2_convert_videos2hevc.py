#!/usr/bin/env python3
"""
Distributed batch transcoding to H.265 (HEVC), supports rank-level sharding via DeepSpeed/torchrun etc.:
- Pass one or more list files, one path per line (absolute or relative to source_root)
- Before building tasks, perform global uniform sharding of "video entries" by RANK/WORLD_SIZE, each rank only processes its own segment, reducing redundant parsing cost
- Each rank will save:
  - This rank's "target path list" file (targets.rank{RANK}.txt): target output paths for this rank (including existing and to-be-generated targets, excluding missing source items)
- Each rank can continue using local multiprocessing pool for parallelism (parallelism is reduced by WORLD_SIZE by default)

deepspeed \
  --hostfile hosts_14 \
  --num_nodes 12 \
  --num_gpus 8 \
  --master_addr ${NODE_IP} \
  --master_port 29600 \
  step2_convert_videos2hevc.py \
  --file /video_vit/clips_square_aug_k710_ssv2/merged_list.txt \
  --source_root /video_vit/clips_square_aug_k710_ssv2 \
  --target_root /video_vit/clips_square_aug_k710_ssv2_hevc
"""

import os
import re
import glob
import argparse
import subprocess
from multiprocessing import Pool
from pathlib import Path
from typing import Iterable, List, Optional, Tuple
import time
from tqdm import tqdm

# ===== Distributed environment variables (consistent with original code) =====
RANK = int(os.environ.get("RANK", "0"))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
# ========================================

time.sleep(10 * RANK)  # Avoid log confusion when multiple processes start

# ===== Adjustable parameters (environment variables) =====
GOP_SIZE = int(os.getenv("GOP_SIZE", "16"))          # Fixed GOP=16
CRF = int(os.getenv("CRF", "23"))                    # Quality/bitrate balance (libx265) or hevc_nvenc's -cq
SKIP_AUDIO = os.getenv("SKIP_AUDIO", "1") == "1"     # Remove audio by default

# Automatically reduce parallelism per rank based on WORLD_SIZE; if NPROC is explicitly set, use NPROC
_auto_proc = max(1, (os.cpu_count() or 1) // max(1, WORLD_SIZE))
PROCESSES = 8
# =================================

def _resolve_paths(relative_src_path: str, source_root: str, target_root: str) -> Tuple[str, str]:
    rel = relative_src_path.strip()

    # 1) Parse source path (supports absolute or relative)
    if os.path.isabs(rel):
        src_path = os.path.normpath(rel)
    else:
        src_path = os.path.normpath(os.path.join(source_root, rel.lstrip('/')))

    # 2) Mirror directory structure under target_root
    if os.path.commonpath([os.path.abspath(src_path), os.path.abspath(source_root)]) == os.path.abspath(source_root):
        rel_from_src = os.path.relpath(src_path, start=source_root)
    else:
        rel_from_src = os.path.basename(src_path)

    rel_dir = os.path.dirname(rel_from_src)
    target_dir = os.path.join(target_root, rel_dir)
    os.makedirs(target_dir, exist_ok=True)

    # Unified output as .mp4 (HEVC)
    file_stem = os.path.splitext(os.path.basename(src_path))[0]
    mp4_name = file_stem + ".mp4"
    mp4_path = os.path.join(target_dir, mp4_name)
    return src_path, mp4_path

def _pick_encoder() -> str:
    pref = os.getenv("HEVC_ENCODER", "").strip()
    candidates = [c for c in [pref, "libx265", "hevc_nvenc"] if c]

    encoders_list = ""
    try:
        out = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False
        )
        encoders_list = (out.stdout or "").lower()
    except Exception:
        pass

    for enc in candidates:
        if enc.lower() in encoders_list:
            return enc
    return pref if pref else "libx265"

def _build_ffmpeg_cmd(
    src_path: str,
    dst_path: str,
    encoder: str,
    ff_log: str = "error",      # FFmpeg log level: quiet|panic|fatal|error|warning|info|verbose|debug|trace
    x265_log: str = "error"     # libx265 log level: none|error|warning|info|debug|full
) -> List[str]:
    cmd = [
        "ffmpeg",
        "-y",
        "-nostdin",
        "-hide_banner",          # Do not print banner
        "-nostats",              # Do not print progress/stat lines
        "-loglevel", ff_log,     # Control FFmpeg's own logs
        "-i", src_path,
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v", encoder,
        "-pix_fmt", "yuv420p",
    ]

    if encoder == "libx265":
        x265_params = (
            f"keyint={GOP_SIZE}:min-keyint={GOP_SIZE}:scenecut=0:"
            f"bframes=0:ref=1:repeat-headers=1:"
            f"log-level={x265_log}"  # Control libx265 logs
        )
        cmd += [
            "-preset", "fast",
            "-crf", str(CRF),
            "-g", str(GOP_SIZE),
            "-x265-params", x265_params,
        ]
    else:
        # hevc_nvenc or other hardware encoding parameters
        cmd += [
            "-preset", "p5",
            "-cq", str(CRF),
            "-g", str(GOP_SIZE),
            "-bf", "0",
            "-sc_threshold", "0",
            "-rc-lookahead", "0",
            "-forced-idr", "1",
        ]

    cmd += [
        "-tag:v", "hvc1",
        "-movflags", "+faststart",
    ]

    if SKIP_AUDIO:
        cmd += ["-an"]
    else:
        cmd += ["-c:a", "aac", "-b:a", "128k"]

    cmd += [dst_path]
    return cmd

def _append_fail(log_path: str, msg: str):
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg.rstrip("\n") + "\n")
    except Exception:
        pass

def convert_to_h265(args: Tuple[str, str, str, str]):
    src_path, dst_path, encoder, log_path = args
    if not os.path.exists(dst_path):
        cmd = _build_ffmpeg_cmd(src_path, dst_path, encoder)
        try:
            proc = subprocess.run(cmd, check=False)
            if proc.returncode != 0:
                print(f"[ffmpeg failed rc={proc.returncode}] {src_path} -> {dst_path}")
                _append_fail(
                    log_path,
                    f"ffmpeg failed (rc={proc.returncode}): {src_path} -> {dst_path}"
                )
            else:
                pass
                # print(f"[rank {RANK}] Converted: {src_path} -> {dst_path}")
        except Exception as e:
            print(f"[ffmpeg error] {src_path} -> {dst_path}: {e}")
            _append_fail(
                log_path,
                f"exception: {src_path} -> {dst_path}: {e}"
            )
    else:
        print(f"[rank {RANK}] Skipped (already exists): {dst_path}")

def _valid_line(s: str) -> bool:
    s = s.strip()
    return bool(s) and not s.startswith("#")

def _extract_field(line: str, field_index: int) -> Optional[str]:
    parts = line.strip().split()
    if not parts:
        return None
    try:
        return parts[field_index]
    except Exception:
        # If index invalid, try to find first token containing '/'
        for tok in parts:
            if "/" in tok or "\\" in tok:
                return tok
        # Otherwise fall back to token 0
        return parts[0]

def _expand_lists(list_args: List[str]) -> List[str]:
    files: List[str] = []
    for pat in list_args:
        matches = glob.glob(pat)
        if matches:
            files.extend(sorted(matches))
        else:
            # If no glob match and is existing file, add directly
            if os.path.isfile(pat):
                files.append(pat)
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for f in files:
        if f not in seen:
            deduped.append(f); seen.add(f)
    return deduped

def parse_items_from_lists(
    list_paths: List[str],
    field_index: int,
    strip_prefix: str,
) -> List[str]:
    """
    Parse list files, extract and normalize "raw entries" (with field_index and strip_prefix applied), no file existence check.
    Return deduplicated entry list in input order.
    """
    items: List[str] = []
    seen = set()
    strip_prefix = strip_prefix or ""
    for txt_path in list_paths:
        print(f"[rank {RANK}] parsing list (for items): {txt_path}")
        try:
            with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
                for ln in f:
                    if not _valid_line(ln):
                        continue
                    token = _extract_field(ln, field_index)
                    if not token:
                        continue
                    raw = token.strip()
                    if strip_prefix and raw.startswith(strip_prefix):
                        raw = raw[len(strip_prefix):]
                    if raw not in seen:
                        seen.add(raw)
                        items.append(raw)
        except FileNotFoundError:
            print(f"[rank {RANK}] list file not found, skip: {txt_path}")
    return items

def build_tasks_from_items(
    items: List[str],
    source_root: str,
    target_root: str,
    log_path: Optional[str] = None
) -> Tuple[List[Tuple[str, str, str, str]], List[str]]:
    """
    Build task and target lists based on "sharded raw entries" (only process entries for current rank).
    Returns:
      - tasks: List of tasks to process (src_path, dst_path, encoder, log_path)
      - targets_rank: All "source exists" target output paths for this rank (including existing and to-be-generated, deduplicated)
    """
    encoder = _pick_encoder()
    if not encoder:
        print("[warn] Unable to determine available HEVC encoder. Please set environment variable HEVC_ENCODER=libx265 or hevc_nvenc.")

    tasks: List[Tuple[str, str, str, str]] = []
    targets_rank: List[str] = []
    seen_dst = set()

    for idx, raw_path in enumerate(items):
        src_path, dst_path = _resolve_paths(raw_path, source_root, target_root)
        
        if idx % 1000 == 0:
            print(f"[rank {RANK}] Processing item {idx}/{len(items)}: {src_path} -> {dst_path}")
        # if dst_path in seen_dst:
        #     continue

        # if not os.path.exists(src_path):
        #     print(f"[rank {RANK}] Missing source, skip: {src_path}")
        #     if log_path:
        #         _append_fail(
        #             log_path,
        #             f"missing source: {src_path} (dst would be {dst_path})"
        #         )
        #     continue

        seen_dst.add(dst_path)
        targets_rank.append(dst_path)

        # if os.path.exists(dst_path):
        #     print(f"[rank {RANK}] Skipped (already exists): {dst_path}")
        # else:
        tasks.append((src_path, dst_path, encoder, log_path or "failed.txt"))

    return tasks, targets_rank

def shard_list(seq: List[str], world_size: int, rank: int) -> List[str]:
    if world_size <= 1:
        return seq
    return [x for i, x in enumerate(seq) if (i % world_size) == rank]

def main():
    ap = argparse.ArgumentParser(description="Distributed H.265 batch converter (DeepSpeed compatible via env RANK/WORLD_SIZE).")
    ap.add_argument("--file", required=True, help="One or more list files or glob (e.g.: worker-*.txt my.list)")
    # Support both hyphen and underscore parameter names
    ap.add_argument("--source-root", "--source_root", dest="source_root", required=True, help="Source video root directory")
    ap.add_argument("--target-root", "--target_root", dest="target_root", required=True, help="Output root directory (will mirror source directory structure, unified output .mp4)")
    ap.add_argument("--log-dir", "--log_dir", dest="log_dir", default="logs", help="Log directory (default: logs)")
    ap.add_argument("--field-index", "--field_index", dest="field_index", type=int, default=0, help="Which field per line to use as path (default 0)")
    ap.add_argument("--strip-prefix", "--strip_prefix", dest="strip_prefix", default="", help="Prefix to remove from line path head (optional)")
    ap.add_argument("--local_rank")

    args = ap.parse_args()

    # Prepare logs (separate file per rank, avoid concurrent write conflicts)
    os.makedirs(args.log_dir, exist_ok=True)
    log_path = os.path.join(args.log_dir, f"failed.rank{RANK}.txt")

    with open(args.file, "r", encoding="utf-8") as f:
        list_files = [x.strip() for x in f.readlines() if x.strip()]

    print(f"[rank {RANK}] WORLD_SIZE={WORLD_SIZE}, RANK={RANK}, LOCAL_RANK={LOCAL_RANK}")
    print(f"[rank {RANK}] Using {PROCESSES} worker processes per rank")

    items_rank = shard_list(list_files, WORLD_SIZE, RANK)
    print(f"[rank {RANK}] Items assigned to this rank: {len(items_rank)}")

    # 2) Build task and target list based on "current rank entries"
    tasks_rank, targets_rank = build_tasks_from_items(
        items=items_rank,
        source_root=args.source_root,
        target_root=args.target_root,
        log_path=log_path
    )
    print(f"[rank {RANK}] Targets in this rank (dedup & src exists): {len(targets_rank)}")
    print(f"[rank {RANK}] Tasks to process in this rank: {len(tasks_rank)}")

    # Save target path list for this rank
    targets_list_path = os.path.join(args.target_root, f"targets.rank{RANK:03d}.txt")
    try:
        with open(targets_list_path, "w", encoding="utf-8") as f:
            for p in targets_rank:
                f.write(p + "\n")
        print(f"[rank {RANK}] Saved targets list: {targets_list_path} (items={len(targets_rank)})")
    except Exception as e:
        print(f"[rank {RANK}] Failed to save targets list {targets_list_path}: {e}")

    # 3) Execute tasks for this rank
    if tasks_rank:
        with Pool(processes=PROCESSES, maxtasksperchild=64) as pool:
            pool.map(convert_to_h265, tasks_rank)

    print(f"[rank {RANK}] Done. Failed log -> {log_path}")

if __name__ == "__main__":
    main()
