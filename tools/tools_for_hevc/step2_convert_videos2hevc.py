#!/usr/bin/env python3
"""
分布式批量转码至 H.265（HEVC），支持通过 DeepSpeed/torchrun 等启动方式进行 rank 级分片：
- 传入一个或多个 list 文件，每行一个路径（绝对或相对 source_root）
- 在构建任务之前，先对“视频条目”按 RANK/WORLD_SIZE 做全局均匀分片，每个 rank 仅处理自身片段，降低重复解析成本
- 每个 rank 会保存：
  - 本 rank 的“目标路径列表”文件（targets.rank{RANK}.txt）：该 rank 对应的目标输出路径（包含已存在和将要生成的目标，剔除源缺失项）
- 每个 rank 可继续使用本地多进程池进行并行（默认会按 WORLD_SIZE 降低并行度）

deepspeed \
  --hostfile hosts_14 \
  --num_nodes 12 \
  --num_gpus 8 \
  --master_addr 172.16.5.34 \
  --master_port 29600 \
  step2_convert_videos2hevc.py \
  --file /video_vit/dataset/clips_square_aug_k710_ssv2/merged_list.txt \
  --source_root /video_vit/dataset/clips_square_aug_k710_ssv2 \
  --target_root /video_vit/dataset/clips_square_aug_k710_ssv2_hevc
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

# ===== 分布式环境变量（与原代码一致）=====
RANK = int(os.environ.get("RANK", "0"))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
# ========================================

time.sleep(10 * RANK)  # 避免多进程启动时日志混乱

# ===== 可调参数（环境变量） =====
GOP_SIZE = int(os.getenv("GOP_SIZE", "16"))          # 固定 GOP=16
CRF = int(os.getenv("CRF", "23"))                    # 画质/码率平衡（libx265）或 hevc_nvenc 的 -cq
SKIP_AUDIO = os.getenv("SKIP_AUDIO", "1") == "1"     # 默认去音频

# 根据 WORLD_SIZE 自动下调每个 rank 的并行度；若显式设置 NPROC 则以 NPROC 为准
_auto_proc = max(1, (os.cpu_count() or 1) // max(1, WORLD_SIZE))
PROCESSES = 8
# =================================

def _resolve_paths(relative_src_path: str, source_root: str, target_root: str) -> Tuple[str, str]:
    rel = relative_src_path.strip()

    # 1) 解析源路径（支持绝对或相对）
    if os.path.isabs(rel):
        src_path = os.path.normpath(rel)
    else:
        src_path = os.path.normpath(os.path.join(source_root, rel.lstrip('/')))

    # 2) 在 target_root 下镜像目录结构
    if os.path.commonpath([os.path.abspath(src_path), os.path.abspath(source_root)]) == os.path.abspath(source_root):
        rel_from_src = os.path.relpath(src_path, start=source_root)
    else:
        rel_from_src = os.path.basename(src_path)

    rel_dir = os.path.dirname(rel_from_src)
    target_dir = os.path.join(target_root, rel_dir)
    os.makedirs(target_dir, exist_ok=True)

    # 统一输出为 .mp4（HEVC）
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
    ff_log: str = "error",      # FFmpeg 的日志级别：quiet|panic|fatal|error|warning|info|verbose|debug|trace
    x265_log: str = "error"     # libx265 的日志级别：none|error|warning|info|debug|full
) -> List[str]:
    cmd = [
        "ffmpeg",
        "-y",
        "-nostdin",
        "-hide_banner",          # 不打印横幅
        "-nostats",              # 不打印进度/stat 行
        "-loglevel", ff_log,     # 控制 FFmpeg 自身日志
        "-i", src_path,
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v", encoder,
        "-pix_fmt", "yuv420p",
    ]

    if encoder == "libx265":
        x265_params = (
            f"keyint={GOP_SIZE}:min-keyint={GOP_SIZE}:scenecut=0:"
            f"bframes=0:ref=1:repeat-headers=1:"
            f"log-level={x265_log}"  # 控制 libx265 的日志
        )
        cmd += [
            "-preset", "fast",
            "-crf", str(CRF),
            "-g", str(GOP_SIZE),
            "-x265-params", x265_params,
        ]
    else:
        # hevc_nvenc 或其它硬编参数
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
        # 若索引非法，尝试找第一个包含 '/' 的 token
        for tok in parts:
            if "/" in tok or "\\" in tok:
                return tok
        # 否则退回第 0 个
        return parts[0]

def _expand_lists(list_args: List[str]) -> List[str]:
    files: List[str] = []
    for pat in list_args:
        matches = glob.glob(pat)
        if matches:
            files.extend(sorted(matches))
        else:
            # 如果没有 glob 匹配且是现有文件，就直接加
            if os.path.isfile(pat):
                files.append(pat)
    # 去重同时保序
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
    解析 list 文件，提取并规范化“原始条目”（已应用 field_index 和 strip_prefix），不做文件存在性检查。
    返回按输入顺序去重后的条目列表。
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
    基于“分片后的原始条目”构建任务与目标列表（仅对当前 rank 的条目进行处理）。
    返回：
      - tasks: 需要处理的任务列表 (src_path, dst_path, encoder, log_path)
      - targets_rank: 本 rank 的所有“源存在”的目标输出路径（含已存在与待生成，已去重）
    """
    encoder = _pick_encoder()
    if not encoder:
        print("[warn] 未能确定可用 HEVC 编码器。请设置环境变量 HEVC_ENCODER=libx265 或 hevc_nvenc。")

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
    ap.add_argument("--file", required=True, help="一个或多个 list 文件或 glob（例：worker-*.txt my.list）")
    # 同时支持短横线和下划线参数名
    ap.add_argument("--source-root", "--source_root", dest="source_root", required=True, help="源视频根目录")
    ap.add_argument("--target-root", "--target_root", dest="target_root", required=True, help="输出根目录（将镜像源目录结构，统一输出 .mp4）")
    ap.add_argument("--log-dir", "--log_dir", dest="log_dir", default="logs", help="日志目录（默认：logs）")
    ap.add_argument("--field-index", "--field_index", dest="field_index", type=int, default=0, help="每行取第几个字段作为路径（默认 0）")
    ap.add_argument("--strip-prefix", "--strip_prefix", dest="strip_prefix", default="", help="从行内路径头部去掉的前缀（可选）")
    ap.add_argument("--local_rank")

    args = ap.parse_args()

    # 准备日志（每个 rank 单独文件，避免并发写冲突）
    os.makedirs(args.log_dir, exist_ok=True)
    log_path = os.path.join(args.log_dir, f"failed.rank{RANK}.txt")

    with open(args.file, "r", encoding="utf-8") as f:
        list_files = [x.strip() for x in f.readlines() if x.strip()]

    print(f"[rank {RANK}] WORLD_SIZE={WORLD_SIZE}, RANK={RANK}, LOCAL_RANK={LOCAL_RANK}")
    print(f"[rank {RANK}] Using {PROCESSES} worker processes per rank")

    items_rank = shard_list(list_files, WORLD_SIZE, RANK)
    print(f"[rank {RANK}] Items assigned to this rank: {len(items_rank)}")

    # 2) 基于“当前 rank 的条目”构建任务与目标清单
    tasks_rank, targets_rank = build_tasks_from_items(
        items=items_rank,
        source_root=args.source_root,
        target_root=args.target_root,
        log_path=log_path
    )
    print(f"[rank {RANK}] Targets in this rank (dedup & src exists): {len(targets_rank)}")
    print(f"[rank {RANK}] Tasks to process in this rank: {len(tasks_rank)}")

    # 保存本 rank 的目标路径列表
    targets_list_path = os.path.join(args.target_root, f"targets.rank{RANK:03d}.txt")
    try:
        with open(targets_list_path, "w", encoding="utf-8") as f:
            for p in targets_rank:
                f.write(p + "\n")
        print(f"[rank {RANK}] Saved targets list: {targets_list_path} (items={len(targets_rank)})")
    except Exception as e:
        print(f"[rank {RANK}] Failed to save targets list {targets_list_path}: {e}")

    # 3) 执行本 rank 的任务
    if tasks_rank:
        with Pool(processes=PROCESSES, maxtasksperchild=64) as pool:
            pool.map(convert_to_h265, tasks_rank)

    print(f"[rank {RANK}] Done. Failed log -> {log_path}")

if __name__ == "__main__":
    main()
