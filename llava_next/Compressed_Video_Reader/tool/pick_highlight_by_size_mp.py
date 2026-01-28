#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Select high-energy video segments based on ffprobe pkt_size.
"""

import argparse
import subprocess
import sys
import time
import os
from pathlib import Path
from multiprocessing import Pool

import numpy as np


# ===================== ffprobe: frame-level pkt_size =====================

def run_ffprobe_frames(video_path, ffprobe_bin="ffprobe", timeout=None):
    """
    Call ffprobe, scan only keyframes (I/IDR), get timestamps and pkt_size. Read metadata only, no pixel decoding.
    Returns:
      times: [t0, t1, ...]  (float seconds)
      sizes: [s0, s1, ...]  (int bytes)
    """
    cmd = [
        ffprobe_bin,
        "-v", "error",
        "-skip_frame", "nokey",
        "-select_streams", "v:0",
        "-show_frames",
        "-show_entries", "frame=pkt_pts_time,best_effort_timestamp_time,pkt_size",
        "-of", "csv=p=0",
        video_path,
    ]

    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=timeout)
    except subprocess.TimeoutExpired:
        sys.stderr.write(f"[WARN] ffprobe timeout on {video_path}, skip.\n")
        return [], []
    except subprocess.CalledProcessError as e:
        sys.stderr.write(f"[WARN] ffprobe failed on {video_path}: {e}\n")
        return [], []

    times, sizes = [], []
    for raw_line in out.decode("utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        parts = [p for p in line.split(",") if p != ""]
        if len(parts) < 2:
            continue

        size_str = parts[-1]

        pkt_pts = parts[0]  # May be "N/A"
        best_effort = parts[1] if len(parts) >= 3 else "N/A"

        t_str = None
        if pkt_pts != "N/A":
            t_str = pkt_pts
        elif best_effort != "N/A":
            t_str = best_effort
        else:
            continue

        try:
            t = float(t_str)
            s = int(size_str)
        except ValueError:
            continue

        times.append(t)
        sizes.append(s)

    return times, sizes


def run_ffprobe_vcodec(video_path, ffprobe_bin="ffprobe", timeout=None):
    """
    Get video stream codec_name to decide whether to use -c copy or re-encode.
    Returns codec_name (str) or "".
    """
    cmd = [
        ffprobe_bin,
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=codec_name",
        "-of", "default=nw=1:nk=1",
        video_path,
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=timeout)
        codec = out.decode("utf-8", errors="ignore").strip().lower()
        return codec
    except Exception:
        return ""



def pick_highlight_windows_multi(
    times,
    sizes,
    durations,          # e.g. [10,30,60,180]
    topk_per_dur,       # e.g. [1,1,1,1]
    smooth_sec=2.0,
    global_min_gap_sec=2.0,
    tail_margin_sec=2.0,
):
    """
    Generate multi-duration candidates with global dedup.

    Returns:
      [(start_t, dur, s_idx, e_idx), ...]
    """
    if len(times) < 2 or len(sizes) < 2:
        return []

    times = np.asarray(times, dtype=np.float64)
    sizes = np.asarray(sizes, dtype=np.float64)

    total_dur = float(times[-1] - times[0])
    if total_dur <= 0:
        return []

    # Avoid selecting segments too close to the tail, where web-crawled videos often have corruption.
    safe_end_t = float(times[-1]) - float(tail_margin_sec)

    avg_dt = total_dur / max(1, (len(times) - 1))
    score = np.log1p(sizes)

    smooth_len = max(1, int(round(smooth_sec / avg_dt)))
    smooth_len = min(smooth_len, len(score))
    kernel = np.ones(smooth_len, dtype=np.float64) / smooth_len
    smooth = np.convolve(score, kernel, mode="same")

    candidates = []
    for dur in durations:
        if dur <= 0:
            continue
        win_len = max(1, int(round(dur / avg_dt)))
        if win_len > len(smooth):
            continue
        for s_idx in range(0, len(smooth) - win_len + 1):
            e_idx = s_idx + win_len
            # End time of this window (use last frame timestamp in the window).
            end_t = float(times[min(e_idx - 1, len(times) - 1)])
            if end_t > safe_end_t:
                continue
            sc = float(smooth[s_idx:e_idx].mean())
            candidates.append((sc, int(s_idx), int(e_idx), float(dur)))

    if not candidates:
        return []

    candidates.sort(key=lambda x: x[0], reverse=True)

    def interval_of(s_idx, e_idx):
        st = float(times[s_idx])
        ed = float(times[min(e_idx - 1, len(times) - 1)])
        return st, ed

    selected = []  # (sc, s_idx, e_idx, dur)
    picked_count = {dur: 0 for dur in durations}
    dur2limit = {durations[i]: topk_per_dur[i] for i in range(len(durations))}

    for sc, s_idx, e_idx, dur in candidates:
        if picked_count.get(dur, 0) >= dur2limit.get(dur, 0):
            continue

        st, ed = interval_of(s_idx, e_idx)

        ok = True
        for _, ss, ee, _ in selected:
            st2, ed2 = interval_of(ss, ee)

            inter = min(ed, ed2) - max(st, st2)
            if inter > 0:
                ok = False
                break

            if abs(st - ed2) < global_min_gap_sec or abs(st2 - ed) < global_min_gap_sec:
                ok = False
                break

        if not ok:
            continue

        selected.append((sc, s_idx, e_idx, dur))
        picked_count[dur] += 1

        if all(picked_count[d] >= dur2limit.get(d, 0) for d in durations):
            break

    selected.sort(key=lambda x: x[1])
    out = []
    for sc, s_idx, e_idx, dur in selected:
        start_t = float(times[s_idx])
        out.append((start_t, float(dur), int(s_idx), int(e_idx)))
    return out


def process_one_video(
    video_path,
    ffprobe_bin,
    durations,
    topk_per_dur,
    smooth_sec,
    global_min_gap_sec,
    tail_margin_sec,
    min_video_dur,
    ffprobe_timeout,
):
    """
    Highlight detection for single video (multi-duration + global dedup).
    """
    times, sizes = run_ffprobe_frames(
        video_path,
        ffprobe_bin=ffprobe_bin,
        timeout=ffprobe_timeout,
    )
    if not times or not sizes:
        return []

    if times[-1] - times[0] < min_video_dur:
        return []

    windows = pick_highlight_windows_multi(
        times,
        sizes,
        durations=durations,
        topk_per_dur=topk_per_dur,
        smooth_sec=smooth_sec,
        global_min_gap_sec=global_min_gap_sec,
        tail_margin_sec=tail_margin_sec,
    )
    return windows



def dur_to_bucket_prefix(dur_sec: float) -> str:
    """Map duration seconds to your bucket naming like 0_30_s / 30_60_s / 1_2_m / 2_3_m."""
    d = float(dur_sec)
    if d <= 30:
        return "0_30_s"
    elif d <= 60:
        return "30_60_s"
    elif d <= 120:
        return "1_2_m"
    elif d <= 180:
        return "2_3_m"
    else:
        # Fallback for longer durations: bucket by minute ranges.
        m = int(d // 60)
        return f"{m}_{m+1}_m"

def make_clip_path(clip_dir, dataset_name, base_root, video_path, start, dur, force_mp4=False):
    """Generate output clip path.

    Directory layout (matching your screenshot):
      clip_dir / f"{bucket}_{dataset_name}" / rel_dir / xxx_st.._d..mp4

    Examples:
      0_30_s_youtube_v0_1/
      30_60_s_nextqa/
      1_2_m_academic_v0_1/
      2_3_m_activitynetqa/

    rel_dir is cut by base_root if provided; otherwise fall back to stripping leading '/'.
    """
    dataset_name = dataset_name if dataset_name else "unknown_dataset"
    bucket = dur_to_bucket_prefix(dur)

    rel = None
    if base_root:
        try:
            rel = os.path.relpath(video_path, base_root)
        except ValueError:
            rel = None
    if rel is None:
        rel = video_path.lstrip("/").replace("\\", "/")

    rel = rel.replace("\\", "/")
    rel_dir = os.path.dirname(rel)
    base = os.path.basename(rel)

    name, ext = os.path.splitext(base)
    # If we are re-encoding, force MP4 container/extension to avoid muxer mismatch
    # (e.g., writing H.264/AAC into .webm will fail).
    if force_mp4:
        ext = ".mp4"
    else:
        if not ext:
            ext = ".mp4"

    bucket_dir = f"{bucket}_{dataset_name}"
    out_dir = os.path.join(clip_dir, bucket_dir, rel_dir)
    os.makedirs(out_dir, exist_ok=True)

    fname = f"{name}_st{start:.2f}_d{dur:.2f}{ext}"
    return os.path.join(out_dir, fname)


def build_ffmpeg_cmd(
    ffmpeg_bin,
    video_path,
    out_path,
    st,
    dur,
    copy_mode,
    reencode_vcodec,
    reencode_preset,
    reencode_crf,
    reencode_acodec,
    reencode_abitrate,
):
    """
    copy_mode=True:  -c:v copy -c:a copy
    copy_mode=False: re-encode video (libx264), audio AAC
    """
    base = [
        ffmpeg_bin,
        "-loglevel", "error",
        "-y",
        "-ss", f"{st:.3f}",
        "-t", f"{dur:.3f}",
        "-fflags", "+discardcorrupt",
        "-err_detect", "ignore_err",
        "-i", video_path,
    ]

    if copy_mode:
        base += ["-c:v", "copy", "-c:a", "copy", out_path]
        return base

    cmd = base + [
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v", reencode_vcodec,
        "-preset", reencode_preset,
        "-crf", str(reencode_crf),
        "-c:a", reencode_acodec,
        "-b:a", str(reencode_abitrate),
        out_path,
    ]
    return cmd


def worker_task(args_tuple):
    """
    Input:
      (video_path, ffprobe_bin, ffmpeg_bin, clip_dir, dataset_name, base_root,
       durations, topk_per_dur, smooth_sec, global_min_gap_sec, tail_margin_sec,
       min_video_dur, ffprobe_timeout,
       reencode_vcodec, reencode_preset, reencode_crf, reencode_acodec, reencode_abitrate)

    Output:
      (video_path, [(st, dur, s_idx, e_idx, clip_path_or_None, mode), ...])
    """
    (video_path,
     ffprobe_bin,
     ffmpeg_bin,
     clip_dir,
     dataset_name,
     base_root,
     durations,
     topk_per_dur,
     smooth_sec,
     global_min_gap_sec,
     tail_margin_sec,
     min_video_dur,
     ffprobe_timeout,
     reencode_vcodec,
     reencode_preset,
     reencode_crf,
     reencode_acodec,
     reencode_abitrate) = args_tuple

    try:
        windows = process_one_video(
            video_path=video_path,
            ffprobe_bin=ffprobe_bin,
            durations=durations,
            topk_per_dur=topk_per_dur,
            smooth_sec=smooth_sec,
            global_min_gap_sec=global_min_gap_sec,
            tail_margin_sec=tail_margin_sec,
            min_video_dur=min_video_dur,
            ffprobe_timeout=ffprobe_timeout,
        )
    except Exception as e:
        sys.stderr.write(f"[WARN] error processing {video_path}: {e}\n")
        return video_path, []

    results = []

    if not clip_dir:
        for st, dur, s_idx, e_idx in windows:
            results.append((st, dur, s_idx, e_idx, None, "index_only"))
        return video_path, results

    vcodec = run_ffprobe_vcodec(video_path, ffprobe_bin=ffprobe_bin, timeout=ffprobe_timeout)
    copy_mode = vcodec in ("h264", "hevc")
    mode_str = "copy" if copy_mode else f"reencode({reencode_vcodec})"

    for st, dur, s_idx, e_idx in windows:
        out_path = make_clip_path(
            clip_dir=clip_dir,
            dataset_name=dataset_name,
            base_root=base_root,
            video_path=video_path,
            start=st,
            dur=dur,
            force_mp4=(not copy_mode),
        )
        cmd = build_ffmpeg_cmd(
            ffmpeg_bin=ffmpeg_bin,
            video_path=video_path,
            out_path=out_path,
            st=st,
            dur=dur,
            copy_mode=copy_mode,
            reencode_vcodec=reencode_vcodec,
            reencode_preset=reencode_preset,
            reencode_crf=reencode_crf,
            reencode_acodec=reencode_acodec,
            reencode_abitrate=reencode_abitrate,
        )

        try:
            subprocess.run(cmd, check=True)
            results.append((st, dur, s_idx, e_idx, out_path, mode_str))
        except subprocess.CalledProcessError as e:
            sys.stderr.write(
                f"[WARN] ffmpeg failed for {video_path} "
                f"(st={st:.3f}, dur={dur:.3f}, mode={mode_str}): {e}\n"
            )

    return video_path, results



def main():
    parser = argparse.ArgumentParser(
        description="Highlight selection via ffprobe pkt_size"
    )
    parser.add_argument("--list", type=str, required=True, help="Video list file")
    parser.add_argument("--out", type=str, required=True, help="Output txt path")

    parser.add_argument("--ffprobe", type=str, default="ffprobe", help="ffprobe path")
    parser.add_argument("--ffprobe-timeout", type=float, default=60.0, help="ffprobe timeout")
    parser.add_argument("--ffmpeg", type=str, default="ffmpeg", help="ffmpeg path")

    parser.add_argument("--clip-dir", type=str, default="",
                        help="Clip output dir")
    parser.add_argument("--dataset", type=str, default="",
                        help="Dataset name")
    parser.add_argument("--base-root", type=str, default="",
                        help="Base root for paths")

    parser.add_argument("--durations", type=str, default="10,30,60,180",
                        help="Segment durations in seconds")
    parser.add_argument("--topk-per-dur", type=str, default="1,1,1,1",
                        help="Max segments per duration")

    parser.add_argument("--smooth-sec", type=float, default=2.0, help="Smoothing length")
    parser.add_argument("--global-min-gap-sec", type=float, default=2.0,
                        help="Min gap between segments")

    parser.add_argument("--min-video-dur", type=float, default=10.0,
                        help="Min video duration")
    parser.add_argument("--workers", type=int, default=8, help="Workers")
    parser.add_argument("--log-every", type=int, default=1000, help="Log interval")
    parser.add_argument("--total-videos", type=int, default=0,
                        help="Total videos for ETA")

    parser.add_argument("--reencode-vcodec", type=str, default="libx264",
                        help="Video encoder")
    parser.add_argument("--reencode-preset", type=str, default="veryfast",
                        help="Encoder preset")
    parser.add_argument("--reencode-crf", type=int, default=23,
                        help="CRF value")
    parser.add_argument("--reencode-acodec", type=str, default="aac",
                        help="Audio encoder")
    parser.add_argument("--reencode-abitrate", type=str, default="128k",
                        help="Audio bitrate")

    parser.add_argument("--tail-margin-sec", type=float, default=2.0,
                        help="Tail margin")

    args = parser.parse_args()

    list_path = Path(args.list)
    out_path = Path(args.out)

    if not list_path.is_file():
        sys.stderr.write(f"[ERROR] list file not found: {list_path}\n")
        sys.exit(1)

    clip_dir = args.clip_dir if args.clip_dir else ""
    total_videos = args.total_videos if args.total_videos > 0 else None
    base_root = args.base_root.strip() if args.base_root else ""

    durations = [float(x) for x in args.durations.split(",") if x.strip() != ""]
    if not durations:
        sys.stderr.write("[ERROR] durations is empty\n")
        sys.exit(1)

    topk_list = [int(x) for x in args.topk_per_dur.split(",") if x.strip() != ""]
    if not topk_list:
        sys.stderr.write("[ERROR] topk-per-dur is empty\n")
        sys.exit(1)
    if len(topk_list) == 1:
        topk_list = topk_list * len(durations)
    if len(topk_list) != len(durations):
        sys.stderr.write("[ERROR] topk-per-dur length must match durations (or give a single number)\n")
        sys.exit(1)

    sys.stderr.write(
        f"[INFO] dataset={args.dataset or '<none>'}, base_root={base_root or '<none>'}\n"
        f"[INFO] durations={durations}, topk_per_dur={topk_list}, smooth_sec={args.smooth_sec}, "
        f"global_min_gap_sec={args.global_min_gap_sec}, tail_margin_sec={args.tail_margin_sec}\n"
        f"[INFO] workers={args.workers}, clip_dir={'<none>' if not clip_dir else clip_dir}\n"
        f"[INFO] reencode_if_not_h264_hevc: vcodec={args.reencode_vcodec}, preset={args.reencode_preset}, "
        f"crf={args.reencode_crf}, acodec={args.reencode_acodec}, abitrate={args.reencode_abitrate}\n"
    )

    def iter_video_paths(path):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                p = line.strip()
                if not p:
                    continue
                yield p

    video_iter = iter_video_paths(list_path)

    def worker_args_generator():
        for v in video_iter:
            yield (
                v,
                args.ffprobe,
                args.ffmpeg,
                clip_dir,
                args.dataset,
                base_root,
                durations,
                topk_list,
                args.smooth_sec,
                args.global_min_gap_sec,
                args.tail_margin_sec,
                args.min_video_dur,
                args.ffprobe_timeout,
                args.reencode_vcodec,
                args.reencode_preset,
                args.reencode_crf,
                args.reencode_acodec,
                args.reencode_abitrate,
            )

    start_time = time.time()
    processed = 0
    written_lines = 0

    with out_path.open("w", encoding="utf-8", buffering=1) as fout:
        with Pool(processes=args.workers) as pool:
            for video_path, window_list in pool.imap_unordered(
                worker_task, worker_args_generator(), chunksize=8
            ):
                processed += 1

                for st, dur, s_idx, e_idx, clip_path, mode in window_list:
                    if clip_path:
                        fout.write(
                            f"{video_path},{st:.3f},{dur:.3f},{s_idx},{e_idx},{clip_path},{mode}\n"
                        )
                    else:
                        fout.write(
                            f"{video_path},{st:.3f},{dur:.3f},{s_idx},{e_idx},{mode}\n"
                        )
                    written_lines += 1

                if (processed % args.log_every == 0) or (
                    total_videos is not None and processed == total_videos
                ):
                    fout.flush()

                    elapsed = time.time() - start_time
                    vps = processed / elapsed if elapsed > 0 else 0.0

                    if total_videos is not None:
                        eta = (total_videos - processed) / vps if vps > 0 else -1
                        eta_str = f"{eta/3600:.2f} h" if eta > 0 else "N/A"
                        prog_str = f"{processed/total_videos*100:.2f}%"
                    else:
                        eta_str = "N/A"
                        prog_str = "N/A"

                    sys.stderr.write(
                        f"[INFO] processed {processed}"
                        + (f"/{total_videos} ({prog_str})" if total_videos else "")
                        + f", written_lines={written_lines}, "
                        f"elapsed {elapsed/3600:.2f} h, speed {vps:.2f} vids/s, ETA {eta_str}\n"
                    )

    total_elapsed = time.time() - start_time
    sys.stderr.write(
        f"[INFO] done. results written to {out_path}\n"
        f"[INFO] total elapsed: {total_elapsed/3600:.2f} h, "
        f"avg speed: {processed/total_elapsed:.2f} vids/s, total lines written: {written_lines}\n"
    )


if __name__ == "__main__":
    main()