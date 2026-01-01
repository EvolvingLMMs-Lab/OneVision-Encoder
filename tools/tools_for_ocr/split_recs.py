#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-process split MXNet .rec by resolution range (based on longer side) into IndexedRecord(.idx/.rec).

Changes in this version
- Each process independently reads and writes its own part file (no centralized writing process).
- File naming changed to: {prefix}_{bucket}_{part}.idx/.rec
  Example: obelics_00000_00300_part_001.idx, obelics_00000_00300_part_001.rec
- Does not use PIL; directly parse JPEG Header to get width and height (zero decoding, fast).
- Only supports JPEG (SOI=FFD8), non-JPEG samples are marked as non_jpeg and skipped.
- Bucket classification (based on longer side max(width, height)):
  - 00000_00300: size <= 300
  - 00300_00600: 300 < size <= 600
  - 00600_01000: 600 < size <= 1000
  - 01000_10000: size > 1000 (if strict limit <=10000 is needed, add check in classify function)

Output structure
  out_dir/
    obelics_00000_00300_part_001.idx
    obelics_00000_00300_part_001.rec
    obelics_00300_00600_part_001.*
    ...
    obelics_01000_10000_part_00N.*

Read/Write conventions
- Read: Record (wrapping mxnet.recordio.MXRecordIO)
- Write: IndexedRecord (wrapping mxnet.recordio.MXIndexedRecordIO)

Dependencies
- mxnet

Usage example
  python split_rec_by_resolution.py \
    --input_list recs.txt \
    --out_dir ./out_rec \
    --processes 8 \
    --prefix obelics

recs.txt file contains absolute paths, one .rec per line
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


# --------- Lightweight wrapper to satisfy "read with Record, write with IndexedRecord" naming requirements ---------
class Record:
    def __init__(self, rec_path: str, flag: str = 'r'):
        self._rec = mx_recordio.MXRecordIO(rec_path, flag)

    def read(self) -> Optional[bytes]:
        # Return single serialized record (binary string), return None at end
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
        # Name is 01000_10000, but here ">1000" is classified into this bucket; add check here if strict <=10000 limit is needed.
        return "01000_10000"


# --------- Fast parse JPEG width/height (zero decoding, only scan Header) ---------
# Reference JPEG spec: SOI (FFD8), each segment marked with 0xFF + marker; SOF0/2 etc. segments contain height/width.
_SOF_MARKERS = {
    0xC0, 0xC1, 0xC2, 0xC3,
    0xC5, 0xC6, 0xC7,
    0xC9, 0xCA, 0xCB,
    0xCD, 0xCE, 0xCF
}
_NO_LENGTH_MARKERS = {  # Markers without length field
    0xD0, 0xD1, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7,  # RST0-7
    0xD8,  # SOI
    0xD9,  # EOI
    0x01,  # TEM
}


def parse_jpeg_size(data: bytes) -> Optional[Tuple[int, int]]:
    # Must have at least SOI
    if len(data) < 4 or data[0] != 0xFF or data[1] != 0xD8:
        return None

    i = 2
    n = len(data)
    # Scan to first SOF segment
    while i < n:
        # Find next 0xFF
        if data[i] != 0xFF:
            i += 1
            continue
        # Skip possible padding 0xFF
        while i < n and data[i] == 0xFF:
            i += 1
        if i >= n:
            break
        marker = data[i]
        i += 1

        if marker in _NO_LENGTH_MARKERS:
            # No length field, continue
            continue

        # Need to read length field (2 bytes, big-endian, includes length field itself)
        if i + 1 >= n:
            break
        seg_len = (data[i] << 8) | data[i + 1]
        if seg_len < 2:
            # Malformed
            return None
        seg_start = i + 2
        seg_end = seg_start + (seg_len - 2)
        if seg_end > n:
            break

        if marker in _SOF_MARKERS:
            # Inside SOF segment: precision(1) + height(2) + width(2) + components(1) + ...
            if seg_len < 7:
                return None
            # Height first, width second (big-endian)
            height = (data[seg_start + 1] << 8) | data[seg_start + 2]
            width = (data[seg_start + 3] << 8) | data[seg_start + 4]
            # Avoid exceptions
            if width <= 0 or height <= 0:
                return None
            return (width, height)

        # Advance to next segment
        i = seg_end

    return None
# -------------------------------------------------------------------------


def open_writers_for_part(out_dir: str, prefix: str, part_name: str) -> Dict[str, IndexedRecord]:
    """Open 4 bucket writers for this process's part. Filename: {prefix}_{bucket}_{part}.idx/.rec"""
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
    One process: sequentially read the assigned .rec list and write directly to this process's part_* files (one file per bucket).
    """
    logger = logging.getLogger(f"worker[{worker_id:02d}]")
    part_name = f"part_{worker_id:03d}"

    total = 0
    bad = 0
    non_jpeg = 0
    written_per_bucket = {b: 0 for b in BUCKETS}

    # Open writers
    writers = open_writers_for_part(out_dir, prefix, part_name)

    for rec_path in rec_paths:
        try:
            r = Record(rec_path, 'r')
        except Exception as e:
            logger.error(f"Failed to open rec: {rec_path} err={e}")
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
                logger.info(f"Processed {total} records, current file {rec_path} read to record {idx_in_file}")

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
    ap = argparse.ArgumentParser(description="Multi-process split MXNet .rec by resolution into IndexedRecord output (each process writes part_***, custom prefix)")
    ap.add_argument("--input_list", type=str, required=True,
                    help="List file containing absolute paths to .rec files, one per line")
    ap.add_argument("--out_dir", type=str, required=True,
                    help="Output directory, will generate {prefix}_{bucket}_{part}.idx/.rec files")
    ap.add_argument("--processes", type=int, default=min(8, cpu_count()),
                    help="Number of processes, default min(8, cpu_count())")
    ap.add_argument("--prefix", type=str, default="obelics",
                    help="Output file prefix, default obelics. Example: obelics_00000_00300_part_001.idx")
    ap.add_argument("--log_level", type=str, default="INFO",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return ap.parse_args()


def load_rec_list(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        recs = [ln.strip() for ln in f if ln.strip()]
    return recs


def chunk_list(xs: List[str], n: int) -> List[List[str]]:
    n = max(1, n)
    # Try to split evenly
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
        logging.error("Input list is empty")
        sys.exit(1)

    for p in rec_list:
        if not os.path.isabs(p):
            logging.warning(f"Detected non-absolute path, will process literally: {p}")

    os.makedirs(args.out_dir, exist_ok=True)

    # Split tasks
    chunks = chunk_list(rec_list, args.processes)

    ctx = get_context("spawn")
    stats_q: Queue = ctx.Queue()

    procs: List[Process] = []
    for i, chunk in enumerate(chunks, start=1):
        p = ctx.Process(target=worker_proc, args=(i, chunk, args.out_dir, args.prefix, stats_q), daemon=True)
        p.start()
        procs.append(p)

    # Wait
    for p in procs:
        p.join()

    # Aggregate
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
        logging.info(f"Process {payload.get('worker_id'):02d} completed, {payload.get('part')}, "
                     f"total={payload.get('total')}, bad={payload.get('bad')}, non_jpeg={payload.get('non_jpeg')}, "
                     f"written={payload.get('written')}")

    logging.info("All completed")
    logging.info(f"Total read: {total}, read failed(bad): {bad}, non-JPEG: {non_jpeg}")
    for b in BUCKETS:
        logging.info(f"{b}: written {written_agg[b]}  records -> file prefix {args.prefix}, output directory {args.out_dir}")

if __name__ == "__main__":
    main()
