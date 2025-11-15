#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多进程把 MXNet .rec 按分辨率区间（基于长边）拆分为 IndexedRecord(.idx/.rec)。

本版更改
- 每个进程独立读取并写出自己的 part 文件（无集中写入进程）。
- 文件命名改为：{prefix}_{bucket}_{part}.idx/.rec
  示例：obelics_00000_00300_part_001.idx、obelics_00000_00300_part_001.rec
- 不使用 PIL；直接解析 JPEG Header 获取宽高（零解码，快速）。
- 仅支持 JPEG（SOI=FFD8），非 JPEG 的样本记为 non_jpeg 并跳过。
- 分桶（基于长边 max(width, height)）：
  - 00000_00300: size <= 300
  - 00300_00600: 300 < size <= 600
  - 00600_01000: 600 < size <= 1000
  - 01000_10000: size > 1000（如需严格限制 <=10000，可在 classify 函数中加判断）

输出结构
  out_dir/
    obelics_00000_00300_part_001.idx
    obelics_00000_00300_part_001.rec
    obelics_00300_00600_part_001.*
    ...
    obelics_01000_10000_part_00N.*

读写约定
- 读取：Record（封装 mxnet.recordio.MXRecordIO）
- 写入：IndexedRecord（封装 mxnet.recordio.MXIndexedRecordIO）

依赖
- mxnet

用法示例
  python split_rec_by_resolution.py \
    --input_list recs.txt \
    --out_dir ./out_rec \
    --processes 8 \
    --prefix obelics

recs.txt 文件内容为绝对路径，一行一个 .rec
"""
import numpy as np
np.bool = np.bool_

import argparse
import logging
import os
import sys
from multiprocessing import get_context, Process, Queue, cpu_count
from typing import List, Optional, Tuple, Dict

from mxnet import recordio as mx_recordio


# --------- 轻量封装，满足“用 Record 读，用 IndexedRecord 写”的命名要求 ---------
class Record:
    def __init__(self, rec_path: str, flag: str = 'r'):
        self._rec = mx_recordio.MXRecordIO(rec_path, flag)

    def read(self) -> Optional[bytes]:
        # 返回单条序列化 record（二进制字符串），到末尾返回 None
        return self._rec.read()

    def close(self):
        try:
            self._rec.close()
        except Exception:
            pass


class IndexedRecord:
    def __init__(self, idx_path: str, rec_path: str, flag: str = 'w'):
        self._idxrec = mx_recordio.MXIndexedRecordIO(idx_path, rec_path, flag)
        self._next_index = 0

    def write(self, s: bytes) -> int:
        idx = self._next_index
        self._idxrec.write_idx(idx, s)
        self._next_index += 1
        return idx

    def close(self):
        try:
            self._idxrec.close()
        except Exception:
            pass

# -------------------------------------------------------------------------


BUCKETS = ("00000_00300", "00300_00600", "00600_01000", "01000_10000")


def classify_bucket_by_longer_side(width: int, height: int) -> Optional[str]:
    size = max(width, height)
    if size <= 300:
        return "00000_00300"
    elif size <= 600:
        return "00300_00600"
    elif size <= 1000:
        return "00600_01000"
    else:
        # 名称为 01000_10000，但这里按 “>1000” 归入该桶；如果你要严格限制 <=10000，可在此加判断。
        return "01000_10000"


# --------- 快速解析 JPEG 宽高（零解码，仅扫描 Header） ---------
# 参考 JPEG 规范：SOI (FFD8)，各段以 0xFF + marker 标识；SOF0/2 等段内包含高宽。
_SOF_MARKERS = {
    0xC0, 0xC1, 0xC2, 0xC3,
    0xC5, 0xC6, 0xC7,
    0xC9, 0xCA, 0xCB,
    0xCD, 0xCE, 0xCF
}
_NO_LENGTH_MARKERS = {  # 无长度字段的 marker
    0xD0, 0xD1, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7,  # RST0-7
    0xD8,  # SOI
    0xD9,  # EOI
    0x01,  # TEM
}


def parse_jpeg_size(data: bytes) -> Optional[Tuple[int, int]]:
    # 最少得有 SOI
    if len(data) < 4 or data[0] != 0xFF or data[1] != 0xD8:
        return None

    i = 2
    n = len(data)
    # 扫描到第一个 SOF 段
    while i < n:
        # 寻找下一个 0xFF
        if data[i] != 0xFF:
            i += 1
            continue
        # 跳过可能的填充 0xFF
        while i < n and data[i] == 0xFF:
            i += 1
        if i >= n:
            break
        marker = data[i]
        i += 1

        if marker in _NO_LENGTH_MARKERS:
            # 无长度字段，继续
            continue

        # 需要读取长度字段（2 字节，大端，含长度字段自身）
        if i + 1 >= n:
            break
        seg_len = (data[i] << 8) | data[i + 1]
        if seg_len < 2:
            # 畸形
            return None
        seg_start = i + 2
        seg_end = seg_start + (seg_len - 2)
        if seg_end > n:
            break

        if marker in _SOF_MARKERS:
            # SOF 段内：precision(1) + height(2) + width(2) + components(1) + ...
            if seg_len < 7:
                return None
            # 高在前，宽在后（大端）
            height = (data[seg_start + 1] << 8) | data[seg_start + 2]
            width = (data[seg_start + 3] << 8) | data[seg_start + 4]
            # 规避异常
            if width <= 0 or height <= 0:
                return None
            return (width, height)

        # 前进到下一个段
        i = seg_end

    return None
# -------------------------------------------------------------------------


def open_writers_for_part(out_dir: str, prefix: str, part_name: str) -> Dict[str, IndexedRecord]:
    """为该进程的 part 打开 4 个桶的 writer。文件名：{prefix}_{bucket}_{part}.idx/.rec"""
    writers: Dict[str, IndexedRecord] = {}
    for b in BUCKETS:
        idx_path = os.path.join(out_dir, f"{prefix}_{b}_{part_name}.idx")
        rec_path = os.path.join(out_dir, f"{prefix}_{b}_{part_name}.rec")
        writers[b] = IndexedRecord(idx_path, rec_path, 'w')
    return writers


def close_writers(writers: Dict[str, IndexedRecord]):
    for w in writers.values():
        try:
            w.close()
        except Exception:
            pass


def worker_proc(worker_id: int,
                rec_paths: List[str],
                out_dir: str,
                prefix: str,
                stats_q: Queue,
                log_every: int = 2000):
    """
    一个进程：顺序读取自己分到的 .rec 列表，并直接写入本进程的 part_* 文件中（每个桶一个文件）。
    """
    logger = logging.getLogger(f"worker[{worker_id:02d}]")
    part_name = f"part_{worker_id:03d}"

    total = 0
    bad = 0
    non_jpeg = 0
    written_per_bucket = {b: 0 for b in BUCKETS}

    # 打开 writers
    writers = open_writers_for_part(out_dir, prefix, part_name)

    for rec_path in rec_paths:
        try:
            r = Record(rec_path, 'r')
        except Exception as e:
            logger.error(f"打开 rec 失败: {rec_path} err={e}")
            continue

        idx_in_file = 0
        while True:
            s = r.read()
            if not s:
                break
            total += 1
            idx_in_file += 1
            try:
                header, img_bytes = mx_recordio.unpack(s)
                wh = parse_jpeg_size(img_bytes)
                if wh is None:
                    non_jpeg += 1
                    continue
                w, h = wh
                bucket = classify_bucket_by_longer_side(w, h)
                if bucket is None:
                    bad += 1
                    continue
                writers[bucket].write(s)
                written_per_bucket[bucket] += 1
            except Exception:
                bad += 1
                continue

            if total % log_every == 0:
                logger.info(f"已处理 {total} 条，当前文件 {rec_path} 读到第 {idx_in_file} 条")

        r.close()

    close_writers(writers)

    stats_q.put({
        "worker_id": worker_id,
        "total": total,
        "bad": bad,
        "non_jpeg": non_jpeg,
        "written": written_per_bucket,
        "part": part_name,
    })


def parse_args():
    ap = argparse.ArgumentParser(description="多进程把 MXNet .rec 按分辨率拆分到 IndexedRecord 输出（每进程写 part_***，自定义前缀）")
    ap.add_argument("--input_list", type=str, required=True,
                    help="包含绝对路径的 .rec 列表文件，每行一个")
    ap.add_argument("--out_dir", type=str, required=True,
                    help="输出目录，将生成 {prefix}_{bucket}_{part}.idx/.rec 文件")
    ap.add_argument("--processes", type=int, default=min(8, cpu_count()),
                    help="进程数量，默认 min(8, cpu_count())")
    ap.add_argument("--prefix", type=str, default="obelics",
                    help="输出文件前缀，默认 obelics。示例：obelics_00000_00300_part_001.idx")
    ap.add_argument("--log_level", type=str, default="INFO",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return ap.parse_args()


def load_rec_list(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        recs = [ln.strip() for ln in f if ln.strip()]
    return recs


def chunk_list(xs: List[str], n: int) -> List[List[str]]:
    n = max(1, n)
    # 尽量平均切分
    m = len(xs)
    base = m // n
    rem = m % n
    chunks = []
    start = 0
    for i in range(n):
        size = base + (1 if i < rem else 0)
        if size == 0:
            continue
        chunks.append(xs[start:start+size])
        start += size
    return chunks


def main():
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    rec_list = load_rec_list(args.input_list)
    if not rec_list:
        logging.error("输入列表为空")
        sys.exit(1)

    for p in rec_list:
        if not os.path.isabs(p):
            logging.warning(f"检测到非绝对路径，将按字面值处理: {p}")

    os.makedirs(args.out_dir, exist_ok=True)

    # 切分任务
    chunks = chunk_list(rec_list, args.processes)

    ctx = get_context("spawn")
    stats_q: Queue = ctx.Queue()

    procs: List[Process] = []
    for i, chunk in enumerate(chunks, start=1):
        p = ctx.Process(target=worker_proc, args=(i, chunk, args.out_dir, args.prefix, stats_q), daemon=True)
        p.start()
        procs.append(p)

    # 等待
    for p in procs:
        p.join()

    # 汇总
    total = 0
    bad = 0
    non_jpeg = 0
    written_agg = {b: 0 for b in BUCKETS}
    received = 0
    expected = len(procs)

    while received < expected:
        try:
            payload = stats_q.get(timeout=1.0)
        except Exception:
            continue
        received += 1
        total += payload.get("total", 0)
        bad += payload.get("bad", 0)
        non_jpeg += payload.get("non_jpeg", 0)
        w = payload.get("written", {})
        for b in BUCKETS:
            written_agg[b] += w.get(b, 0)
        logging.info(f"进程 {payload.get('worker_id'):02d} 完成，{payload.get('part')}, "
                     f"总计={payload.get('total')}, bad={payload.get('bad')}, non_jpeg={payload.get('non_jpeg')}, "
                     f"写入={payload.get('written')}")

    logging.info("全部完成")
    logging.info(f"读总计: {total}, 读失败(bad): {bad}, 非 JPEG: {non_jpeg}")
    for b in BUCKETS:
        logging.info(f"{b}: 写入 {written_agg[b]} 条 -> 文件前缀 {args.prefix}, 输出目录 {args.out_dir}")

if __name__ == "__main__":
    main()
