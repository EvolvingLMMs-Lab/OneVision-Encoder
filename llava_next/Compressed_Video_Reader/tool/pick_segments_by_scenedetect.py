#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import inspect
import os
import sys
import time
import subprocess
from pathlib import Path
from multiprocessing import Pool
from typing import List, Tuple, Dict

from scenedetect import open_video, SceneManager
from scenedetect.detectors import AdaptiveDetector, ContentDetector

Scene = Tuple[float, float]  # (start_sec, end_sec)
Seg = Tuple[float, float]    # (start_sec, dur_sec)


def dur_to_bucket_prefix(dur_sec: float) -> str:
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
        m = int(d // 60)
        return f"{m}_{m+1}_m"


def make_clip_path(
    clip_dir: str,
    dataset_name: str,
    base_root: str,
    video_path: str,
    start: float,
    dur: float,
    force_mp4: bool = False,
) -> str:
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


def run_ffprobe_vcodec(video_path: str, ffprobe_bin: str = "ffprobe", timeout: float = 30.0) -> str:
    cmd = [
        ffprobe_bin, "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=codec_name",
        "-of", "default=nw=1:nk=1",
        video_path,
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=timeout)
        return out.decode("utf-8", errors="ignore").strip().lower()
    except Exception:
        return ""


def build_ffmpeg_cmd(
    ffmpeg_bin: str,
    video_path: str,
    out_path: str,
    st: float,
    dur: float,
    copy_mode: bool,
    reencode_vcodec: str,
    reencode_preset: str,
    reencode_crf: int,
    reencode_acodec: str,
    reencode_abitrate: str,
) -> List[str]:
    base = [
        ffmpeg_bin,
        "-loglevel", "error",
        "-y",
        "-fflags", "+discardcorrupt",
        "-err_detect", "ignore_err",
        "-ss", f"{st:.3f}",
        "-t", f"{dur:.3f}",
        "-i", video_path,
    ]
    if copy_mode:
        return base + ["-c:v", "copy", "-c:a", "copy", out_path]

    return base + [
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v", reencode_vcodec,
        "-preset", reencode_preset,
        "-crf", str(reencode_crf),
        "-c:a", reencode_acodec,
        "-b:a", str(reencode_abitrate),
        out_path,
    ]


def detect_scenes(
    video_path: str,
    backend: str = "pyav",          # <- default to pyav for robustness
    detector: str = "adaptive",
    downscale: int = 2,
    adaptive_threshold: float = 3.0,
    content_threshold: float = 27.0,
) -> List[Scene]:
    # backend: "opencv" or "pyav"
    # pyav needs `pip install av`
    video = open_video(video_path, backend=backend)
    sm = SceneManager()

    if detector == "content":
        sm.add_detector(ContentDetector(threshold=content_threshold))
    else:
        sm.add_detector(AdaptiveDetector(adaptive_threshold=adaptive_threshold))

    # PySceneDetect API differs across versions. Some versions support `downscale_factor`
    # in `SceneManager.detect_scenes`, others do not. We adapt at runtime.
    kwargs = {"video": video}
    try:
        sig = inspect.signature(sm.detect_scenes)
        params = sig.parameters
        if "downscale_factor" in params:
            kwargs["downscale_factor"] = downscale
        elif "frame_skip" in params and downscale and downscale > 1:
            # Approximate downscale speed-up by skipping frames.
            kwargs["frame_skip"] = int(downscale) - 1
    except Exception:
        # If signature introspection fails, fall back to the safest call.
        pass

    sm.detect_scenes(**kwargs)
    scene_list = sm.get_scene_list()

    scenes: List[Scene] = []
    for st_tc, ed_tc in scene_list:
        st = float(st_tc.get_seconds())
        ed = float(ed_tc.get_seconds())
        if ed > st:
            scenes.append((st, ed))
    return scenes


def pick_segments_from_scenes(
    scenes: List[Scene],
    targets: List[float],
    counts: List[int],
    tolerances: List[float],
    min_gap_sec: float = 2.0,
    tail_margin_sec: float = 5.0,
    min_scene_sec: float = 0.4,
) -> List[Seg]:
    if not scenes:
        return []

    scenes = [(s, e) for (s, e) in scenes if (e - s) >= min_scene_sec]
    if not scenes:
        return []

    video_end = scenes[-1][1]
    safe_end = video_end - float(tail_margin_sec)
    if safe_end <= scenes[0][0]:
        return []

    S = [s for s, _ in scenes]
    E = [e for _, e in scenes]
    n = len(scenes)

    used: List[Tuple[float, float]] = []

    def overlaps_ok(st: float, ed: float) -> bool:
        for u0, u1 in used:
            if min(ed, u1) - max(st, u0) > 0:
                return False
            if abs(st - u1) < min_gap_sec or abs(u0 - ed) < min_gap_sec:
                return False
        return True

    specs = list(zip(targets, counts, tolerances))
    specs.sort(key=lambda x: x[0], reverse=True)

    picked: List[Seg] = []

    for target, k, tol in specs:
        if k <= 0:
            continue

        for _ in range(k):
            best = None  # (score, abs_diff, st, ed, dur)
            for i in range(n):
                st = S[i]
                if st >= safe_end:
                    break

                j = i
                while j < n and (E[j] - st) < (target - tol):
                    j += 1
                if j >= n:
                    continue

                jj = j
                while jj < n:
                    ed = E[jj]
                    dur = ed - st
                    if ed > safe_end:
                        break
                    if dur > (target + tol):
                        break

                    if overlaps_ok(st, ed):
                        cuts = max(0, jj - i)
                        cut_density = cuts / max(dur, 1e-6)
                        abs_diff = abs(dur - target)
                        cand = (cut_density, -abs_diff, st, ed, dur)
                        if best is None or cand > best:
                            best = cand
                    jj += 1

            if best is None:
                break

            _, _, st, ed, dur = best
            used.append((st, ed))
            picked.append((st, dur))

    picked.sort(key=lambda x: x[0])
    return picked


def parse_targets(spec: str) -> Tuple[List[float], List[int]]:
    targets, counts = [], []
    items = [x.strip() for x in spec.split(",") if x.strip()]
    for it in items:
        if ":" in it:
            t, c = it.split(":", 1)
            targets.append(float(t))
            counts.append(int(c))
        else:
            targets.append(float(it))
            counts.append(1)
    return targets, counts


def parse_tolerances(spec: str, targets: List[float], default_tol: float) -> List[float]:
    mp: Dict[float, float] = {}
    if spec.strip():
        items = [x.strip() for x in spec.split(",") if x.strip()]
        for it in items:
            if ":" in it:
                t, v = it.split(":", 1)
                mp[float(t)] = float(v)
    return [mp.get(t, float(default_tol)) for t in targets]


def worker_task(args_tuple):
    (
        video_path,
        list_root,
        backend,
        detector,
        downscale,
        adaptive_threshold,
        content_threshold,
        mode,
        targets,
        counts,
        tolerances,
        min_gap_sec,
        tail_margin_sec,
        min_scene_sec,
        clip_dir,
        dataset_name,
        base_root,
        ffprobe_bin,
        ffmpeg_bin,
        ffprobe_timeout,
        reencode_vcodec,
        reencode_preset,
        reencode_crf,
        reencode_acodec,
        reencode_abitrate,
    ) = args_tuple

    vp = video_path.strip()
    if not vp:
        return video_path, [], "empty"
    if not os.path.isabs(vp) and list_root:
        vp = os.path.join(list_root, vp)
    vp = os.path.normpath(vp)

    if not os.path.exists(vp):
        return vp, [], "missing_file"

    try:
        scenes = detect_scenes(
            video_path=vp,
            backend=backend,
            detector=detector,
            downscale=downscale,
            adaptive_threshold=adaptive_threshold,
            content_threshold=content_threshold,
        )
    except Exception as e:
        return vp, [], f"open_fail:{e}"

    # Mode: only dump scene boundaries.
    if mode == "dump_scenes":
        if not scenes:
            return vp, [], "no_scenes"
        # Return scenes as (start, end) tuples.
        return vp, scenes, "ok_scenes"

    segs = pick_segments_from_scenes(
        scenes=scenes,
        targets=targets,
        counts=counts,
        tolerances=tolerances,
        min_gap_sec=min_gap_sec,
        tail_margin_sec=tail_margin_sec,
        min_scene_sec=min_scene_sec,
    )
    if not segs:
        return vp, [], "no_segments"

    if not clip_dir:
        out = [(st, dur, "", "index_only") for st, dur in segs]
        return vp, out, "ok"

    vcodec = run_ffprobe_vcodec(vp, ffprobe_bin=ffprobe_bin, timeout=ffprobe_timeout)
    copy_mode = vcodec in ("h264", "hevc")
    mode_str = "copy" if copy_mode else f"reencode({reencode_vcodec})"

    out = []
    for st, dur in segs:
        out_path = make_clip_path(
            clip_dir=clip_dir,
            dataset_name=dataset_name,
            base_root=base_root,
            video_path=vp,
            start=st,
            dur=dur,
            force_mp4=(not copy_mode),
        )
        cmd = build_ffmpeg_cmd(
            ffmpeg_bin=ffmpeg_bin,
            video_path=vp,
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
            out.append((st, dur, out_path, mode_str))
        except subprocess.CalledProcessError:
            out.append((st, dur, "", f"ffmpeg_fail:{mode_str}"))

    return vp, out, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True, help="video list file (one path per line)")
    ap.add_argument("--list-root", default="", help="prepend to non-absolute paths in list")
    ap.add_argument("--out", required=True, help="output index txt")
    ap.add_argument(
        "--mode",
        choices=["pick", "dump_scenes"],
        default="pick",
        help="pick: pick segments; dump_scenes: only dump scene boundaries (cut positions)",
    )

    ap.add_argument("--clip-dir", default="", help="if set, cut clips into this directory")
    ap.add_argument("--dataset", default="", help="dataset name for bucket folder")
    ap.add_argument("--base-root", default="", help="trim this prefix when building relpath layout")

    ap.add_argument("--backend", choices=["opencv", "pyav"], default="pyav",
                    help="video decode backend; pyav is usually more robust in docker")
    ap.add_argument("--detector", choices=["adaptive", "content"], default="adaptive")
    ap.add_argument("--downscale", type=int, default=2)
    ap.add_argument("--adaptive-threshold", type=float, default=3.0)
    ap.add_argument("--content-threshold", type=float, default=27.0)

    ap.add_argument("--targets", type=str, default="60:1,30:2")
    ap.add_argument("--tolerances", type=str, default="30:6,60:10,180:20")
    ap.add_argument("--default-tol", type=float, default=10.0)

    ap.add_argument("--min-gap-sec", type=float, default=2.0)
    ap.add_argument("--tail-margin-sec", type=float, default=5.0)
    ap.add_argument("--min-scene-sec", type=float, default=0.4)

    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--log-every", type=int, default=200)

    ap.add_argument("--ffprobe", default="ffprobe")
    ap.add_argument("--ffmpeg", default="ffmpeg")
    ap.add_argument("--ffprobe-timeout", type=float, default=30.0)

    ap.add_argument("--reencode-vcodec", default="libx264")
    ap.add_argument("--reencode-preset", default="veryfast")
    ap.add_argument("--reencode-crf", type=int, default=23)
    ap.add_argument("--reencode-acodec", default="aac")
    ap.add_argument("--reencode-abitrate", default="128k")

    args = ap.parse_args()

    list_path = Path(args.list)
    if not list_path.is_file():
        sys.stderr.write(f"[ERROR] list not found: {list_path}\n")
        sys.exit(1)

    targets, counts = parse_targets(args.targets)
    tolerances = parse_tolerances(args.tolerances, targets, default_tol=args.default_tol)

    sys.stderr.write(
        f"[INFO] mode={args.mode}\n"
        f"[INFO] backend={args.backend}, detector={args.detector}, downscale={args.downscale}\n"
        f"[INFO] list_root={args.list_root or '<none>'}\n"
        f"[INFO] targets={targets}, counts={counts}, tolerances={tolerances}\n"
        f"[INFO] min_gap_sec={args.min_gap_sec}, tail_margin_sec={args.tail_margin_sec}, min_scene_sec={args.min_scene_sec}\n"
        f"[INFO] clip_dir={'<none>' if not args.clip_dir else args.clip_dir}, dataset={args.dataset or '<none>'}\n"
        f"[INFO] workers={args.workers}\n"
    )

    def iter_videos():
        with list_path.open("r", encoding="utf-8") as f:
            for line in f:
                p = line.strip()
                if p:
                    yield p

    def gen_worker_args():
        for vp in iter_videos():
            yield (
                vp,
                args.list_root.strip(),
                args.backend,
                args.detector,
                args.downscale,
                args.adaptive_threshold,
                args.content_threshold,
                args.mode,
                targets,
                counts,
                tolerances,
                args.min_gap_sec,
                args.tail_margin_sec,
                args.min_scene_sec,
                args.clip_dir.strip(),
                args.dataset.strip(),
                args.base_root.strip(),
                args.ffprobe,
                args.ffmpeg,
                args.ffprobe_timeout,
                args.reencode_vcodec,
                args.reencode_preset,
                args.reencode_crf,
                args.reencode_acodec,
                args.reencode_abitrate,
            )

    t0 = time.time()
    processed = 0
    written = 0
    ok = 0
    status_cnt: Dict[str, int] = {}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "dump_scenes":
        # video_path,scene_id,scene_start,scene_end,scene_dur,status
        with out_path.open("w", encoding="utf-8", buffering=1) as fout:
            with Pool(processes=args.workers) as pool:
                for vp, scene_list, status in pool.imap_unordered(worker_task, gen_worker_args(), chunksize=4):
                    processed += 1
                    status_cnt[status] = status_cnt.get(status, 0) + 1

                    if status == "ok_scenes":
                        ok += 1
                        for i, (st, ed) in enumerate(scene_list):
                            dur = float(ed - st)
                            st2 = float(st)
                            ed2 = float(ed)
                            status_safe = str(status).replace(",", ";")
                            fout.write(f"{vp},{i},{st2:.3f},{ed2:.3f},{dur:.3f},{status_safe}\n")
                            written += 1
                    else:
                        # Always write one row so the user can see why it failed.
                        status_safe = str(status).replace(",", ";")
                        fout.write(f"{vp},-1,0.000,0.000,0.000,{status_safe}\n")
                        written += 1

                    if processed % args.log_every == 0:
                        elapsed = time.time() - t0
                        vps = processed / max(elapsed, 1e-6)
                        top = sorted(status_cnt.items(), key=lambda x: x[1], reverse=True)[:3]
                        top_str = ", ".join([f"{k}={v}" for k, v in top]) if top else ""
                        sys.stderr.write(
                            f"[INFO] processed={processed}, ok={ok}, written={written}, "
                            f"elapsed={elapsed/60:.1f}m, speed={vps:.2f} vids/s"
                            + (f", top_status=[{top_str}]\n" if top_str else "\n")
                        )
    else:
        # video_path,st,dur,clip_path,mode,status
        with out_path.open("w", encoding="utf-8", buffering=1) as fout:
            with Pool(processes=args.workers) as pool:
                for vp, seg_list, status in pool.imap_unordered(worker_task, gen_worker_args(), chunksize=4):
                    processed += 1
                    status_cnt[status] = status_cnt.get(status, 0) + 1
                    if status == "ok":
                        ok += 1
                    for st, dur, clip_path, mode in seg_list:
                        fout.write(f"{vp},{st:.3f},{dur:.3f},{clip_path},{mode},{status}\n")
                        written += 1

                    if processed % args.log_every == 0:
                        elapsed = time.time() - t0
                        vps = processed / max(elapsed, 1e-6)
                        top = sorted(status_cnt.items(), key=lambda x: x[1], reverse=True)[:3]
                        top_str = ", ".join([f"{k}={v}" for k, v in top]) if top else ""
                        sys.stderr.write(
                            f"[INFO] processed={processed}, ok={ok}, written={written}, "
                            f"elapsed={elapsed/60:.1f}m, speed={vps:.2f} vids/s"
                            + (f", top_status=[{top_str}]\n" if top_str else "\n")
                        )

    elapsed = time.time() - t0
    sys.stderr.write(
        f"[INFO] done. out={out_path}\n"
        f"[INFO] processed={processed}, ok={ok}, written={written}, elapsed={elapsed/60:.1f}m\n"
    )


if __name__ == "__main__":
    main()