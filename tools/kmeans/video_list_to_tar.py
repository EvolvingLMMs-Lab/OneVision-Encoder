#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import tarfile
import argparse
import logging
import time
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path
from typing import List, Dict, Any

MAX_CONCURRENCY = 32  # Task concurrency limit (no more than 4)

def read_paths(list_file: Path):
    with list_file.open('r', encoding='utf-8') as f:
        for line in f:
            p = line.strip()
            if p:
                yield p

def make_tar(paths: List[str], tar_path: str, batch_idx: int) -> Dict[str, Any]:
    """
    Subprocess execution: creates an uncompressed .tar archive.
    Returns runtime statistics for the main process to log.
    """
    t0 = time.monotonic()
    out = Path(tar_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    added = 0
    skipped = 0
    total_bytes = 0
    err_msgs: List[str] = []

    try:
        with tarfile.open(out, mode='w') as tf:
            for p in paths:
                fp = Path(p)
                try:
                    if fp.is_file():
                        try:
                            total_bytes += fp.stat().st_size
                        except Exception as se:
                            # stat failure does not block archiving
                            err_msgs.append(f"stat failed: {fp} -> {se}")

                        # Use absolute path as archive path (POSIX format)
                        # Note: Some extraction tools may remove leading slashes when unpacking
                        abs_arcname = fp.resolve().as_posix()

                        tf.add(str(fp), arcname=abs_arcname, recursive=False)
                        added += 1
                    else:
                        skipped += 1
                except Exception as e:
                    skipped += 1
                    err_msgs.append(f"add failed: {fp} -> {e}")
    except Exception as e:
        # Complete failure
        return {
            "ok": False,
            "batch_idx": batch_idx,
            "tar_path": str(out),
            "error": f"tar creation failed: {e}",
            "added": added,
            "skipped": skipped,
            "bytes": total_bytes,
            "duration": time.monotonic() - t0,
        }

    return {
        "ok": True,
        "batch_idx": batch_idx,
        "tar_path": str(out),
        "added": added,
        "skipped": skipped,
        "bytes": total_bytes,
        "duration": time.monotonic() - t0,
        "warn_samples": err_msgs[:5],
        "warn_count": len(err_msgs),
    }

def setup_logging(verbosity: int):
    # verbosity: 0=INFO, 1+=DEBUG
    level = logging.INFO if verbosity <= 0 else logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

def main():
    parser = argparse.ArgumentParser(
        description='Batch video list into .tar archives by fixed count (max 4 concurrent tasks), with logging.'
    )
    parser.add_argument('list_file', type=Path, help='Text file containing video paths (one path per line)')
    parser.add_argument('out_dir', type=Path, help='Output directory')
    parser.add_argument('-j', '--workers', type=int, default=4, help='Number of concurrent tasks (max 4, default 4)')
    parser.add_argument('-k', '--per_tar', type=int, default=100_000, help='Number of videos per .tar archive (default 100000)')
    parser.add_argument('-n', '--name_prefix', type=str, default='batch', help='Output .tar filename prefix (default batch)')
    parser.add_argument('-v', '--verbose', action='count', default=0, help='Increase logging verbosity (stackable)')

    args = parser.parse_args()

    setup_logging(args.verbose)
    log = logging.getLogger("pack")

    list_file: Path = args.list_file
    out_dir: Path = args.out_dir
    per_tar: int = max(1, args.per_tar)
    workers: int = max(1, min(MAX_CONCURRENCY, args.workers))
    name_prefix: str = args.name_prefix

    if not list_file.is_file():
        print('List file does not exist', file=sys.stderr)
        sys.exit(2)

    t_start = time.monotonic()
    futures: dict = {}
    batch: List[str] = []
    idx = 1

    submitted_batches = 0
    completed_batches = 0
    success_batches = 0
    failed_batches = 0
    total_added = 0
    total_skipped = 0
    total_bytes = 0

    log.info("Starting: list=%s output_dir=%s per_tar=%d workers=%d prefix=%s",
             list_file, out_dir, per_tar, workers, name_prefix)

    def submit_batch(paths: List[str], batch_idx: int):
        nonlocal submitted_batches
        tar_path = out_dir / f'{name_prefix}_{batch_idx:06d}.tar'
        fut = ex.submit(make_tar, paths, str(tar_path), batch_idx)
        futures[fut] = (batch_idx, str(tar_path), len(paths))
        submitted_batches += 1
        log.info("Submitted batch %06d -> %s (file_count=%d active_tasks=%d/%d)",
                 batch_idx, tar_path, len(paths), len(futures), workers)

    def consume_done(done_set):
        nonlocal completed_batches, success_batches, failed_batches
        nonlocal total_added, total_skipped, total_bytes
        for fut in done_set:
            batch_idx, tar_path, batch_len = futures.pop(fut)
            completed_batches += 1
            try:
                res = fut.result()
            except Exception as e:
                failed_batches += 1
                log.error("Completed batch %06d failed: %s", batch_idx, e)
                continue

            if not res.get("ok", False):
                failed_batches += 1
                log.error("Completed batch %06d failed: %s", batch_idx, res.get("error"))
                continue

            success_batches += 1
            added = int(res["added"])
            skipped = int(res["skipped"])
            bytes_ = int(res["bytes"])
            dur = float(res["duration"])
            total_added += added
            total_skipped += skipped
            total_bytes += bytes_

            msg = (f"Completed batch {batch_idx:06d}: OK -> {tar_path} | "
                   f"added={added} skipped={skipped} size={bytes_/1_048_576:.2f} MiB "
                   f"duration={dur:.2f}s")
            log.info(msg)

            warn_count = int(res.get("warn_count", 0))
            warn_samples = res.get("warn_samples", [])
            if warn_count > 0:
                log.warning("Batch %06d has %d warnings (sample first %d): %s",
                            batch_idx, warn_count, len(warn_samples), "; ".join(warn_samples))

            if log.isEnabledFor(logging.DEBUG):
                log.debug("Progress: completed=%d submitted=%d active=%d",
                          completed_batches, submitted_batches, len(futures))

    with ProcessPoolExecutor(max_workers=workers) as ex:
        for p in read_paths(list_file):
            batch.append(p)
            if len(batch) >= per_tar:
                submit_batch(batch, idx)
                batch = []
                idx += 1
                if len(futures) >= workers:
                    done, _ = wait(set(futures.keys()), return_when=FIRST_COMPLETED)
                    consume_done(done)

        if batch:
            submit_batch(batch, idx)
            batch = []
            if len(futures) >= workers:
                done, _ = wait(set(futures.keys()), return_when=FIRST_COMPLETED)
                consume_done(done)

        if futures:
            done, _ = wait(set(futures.keys()))
            consume_done(done)

    elapsed = time.monotonic() - t_start
    log.info("All completed: batches success=%d failed=%d total=%d | files added=%d skipped=%d | total_size=%.2f MiB | total_time=%.2fs",
             success_batches, failed_batches, submitted_batches,
             total_added, total_skipped, total_bytes/1_048_576, elapsed)

if __name__ == '__main__':
    main()