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

MAX_CONCURRENCY = 32  # 任务并发上限（不超过 4）

def read_paths(list_file: Path):
    with list_file.open('r', encoding='utf-8') as f:
        for line in f:
            p = line.strip()
            if p:
                yield p

def make_tar(paths: List[str], tar_path: str, batch_idx: int) -> Dict[str, Any]:
    """
    子进程执行：创建一个未压缩的 .tar 包。
    返回运行统计信息，供主进程打印日志。
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
                            # 统计大小失败不阻塞打包
                            err_msgs.append(f"stat失败: {fp} -> {se}")

                        # 使用绝对路径作为归档内路径（POSIX 形式）
                        # 注意：某些解压工具可能会在解包时移除前导斜杠
                        abs_arcname = fp.resolve().as_posix()

                        tf.add(str(fp), arcname=abs_arcname, recursive=False)
                        added += 1
                    else:
                        skipped += 1
                except Exception as e:
                    skipped += 1
                    err_msgs.append(f"加入失败: {fp} -> {e}")
    except Exception as e:
        # 彻底失败
        return {
            "ok": False,
            "batch_idx": batch_idx,
            "tar_path": str(out),
            "error": f"创建tar失败: {e}",
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
        description='按固定数量分批将视频列表打成 .tar 包（并发最多 4 个任务），带日志提示。'
    )
    parser.add_argument('list_file', type=Path, help='包含视频路径的文本文件（每行一个路径）')
    parser.add_argument('out_dir', type=Path, help='输出目录')
    parser.add_argument('-j', '--workers', type=int, default=4, help='并发任务数（最大 4，默认 4）')
    parser.add_argument('-k', '--per_tar', type=int, default=100_000, help='每个 .tar 包包含的视频数量（默认 100000）')
    parser.add_argument('-n', '--name_prefix', type=str, default='batch', help='输出 .tar 文件名前缀（默认 batch）')
    parser.add_argument('-v', '--verbose', action='count', default=0, help='增加日志详细程度（可叠加）')

    args = parser.parse_args()

    setup_logging(args.verbose)
    log = logging.getLogger("pack")

    list_file: Path = args.list_file
    out_dir: Path = args.out_dir
    per_tar: int = max(1, args.per_tar)
    workers: int = max(1, min(MAX_CONCURRENCY, args.workers))
    name_prefix: str = args.name_prefix

    if not list_file.is_file():
        print('列表文件不存在', file=sys.stderr)
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

    log.info("开始：list=%s 输出目录=%s 每包=%d 并发=%d 前缀=%s",
             list_file, out_dir, per_tar, workers, name_prefix)

    def submit_batch(paths: List[str], batch_idx: int):
        nonlocal submitted_batches
        tar_path = out_dir / f'{name_prefix}_{batch_idx:06d}.tar'
        fut = ex.submit(make_tar, paths, str(tar_path), batch_idx)
        futures[fut] = (batch_idx, str(tar_path), len(paths))
        submitted_batches += 1
        log.info("提交批次 %06d -> %s (文件数=%d 活动任务=%d/%d)",
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
                log.error("完成批次 %06d 失败: %s", batch_idx, e)
                continue

            if not res.get("ok", False):
                failed_batches += 1
                log.error("完成批次 %06d 失败: %s", batch_idx, res.get("error"))
                continue

            success_batches += 1
            added = int(res["added"])
            skipped = int(res["skipped"])
            bytes_ = int(res["bytes"])
            dur = float(res["duration"])
            total_added += added
            total_skipped += skipped
            total_bytes += bytes_

            msg = (f"完成批次 {batch_idx:06d}: OK -> {tar_path} | "
                   f"加入={added} 跳过={skipped} 大小={bytes_/1_048_576:.2f} MiB "
                   f"耗时={dur:.2f}s")
            log.info(msg)

            warn_count = int(res.get("warn_count", 0))
            warn_samples = res.get("warn_samples", [])
            if warn_count > 0:
                log.warning("批次 %06d 有 %d 条警告（示例前 %d 条）：%s",
                            batch_idx, warn_count, len(warn_samples), "; ".join(warn_samples))

            if log.isEnabledFor(logging.DEBUG):
                log.debug("进度：已完成=%d 已提交=%d 活动=%d",
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
    log.info("全部完成：批次 成功=%d 失败=%d 总数=%d | 文件 加入=%d 跳过=%d | 总大小=%.2f MiB | 总耗时=%.2fs",
             success_batches, failed_batches, submitted_batches,
             total_added, total_skipped, total_bytes/1_048_576, elapsed)

if __name__ == '__main__':
    main()