import argparse
import math
import os
import sys
import traceback

import decord
import numpy as np

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False
try:
    from PIL import Image
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

import torch
from hevc_feature_decoder import ResPipeReader

# ===== Distributed environment variables (consistent with original code) =====
RANK = int(os.environ.get("RANK", "0"))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
# ========================================


# ---------- Utilities ----------
def _resize_u8_gray(arr_u8: np.ndarray, size: int = 224) -> np.ndarray:
    if arr_u8.ndim != 2:
        raise ValueError(f"expect HxW, got shape {arr_u8.shape}")
    if _HAS_CV2:
        interp = cv2.INTER_AREA if (arr_u8.shape[0] > size or arr_u8.shape[1] > size) else cv2.INTER_LINEAR
        return cv2.resize(arr_u8, (size, size), interpolation=interp)
    elif _HAS_PIL:
        return np.array(Image.fromarray(arr_u8).resize((size, size), resample=Image.BILINEAR), dtype=np.uint8)
    else:
        H, W = arr_u8.shape
        ys = (np.linspace(0, H-1, size)).astype(np.int32)
        xs = (np.linspace(0, W-1, size)).astype(np.int32)
        return arr_u8[ys[:,None], xs[None,:]]

# def _maybe_swap_to_hevc(p: str) -> str:
#     try:
#         if not isinstance(p, str):
#             return p
#         old_seg = "/videos_frames64_kinetics_ssv2/videos_frames64_kinetics_ssv2/"
#         new_seg = "/videos_frames64_kinetics_ssv2/videos_frames64_kinetics_ssv2_hevc/"
#         if old_seg in p:
#             cand = p.replace(old_seg, new_seg)
#             if os.path.exists(cand):
#                 return cand
#     except Exception:
#         pass
#     return p

def _load_list_file(list_path: str):
    with open(list_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]
    return lines  # One video path per line

def mask_by_residual_topk(res_torch: torch.Tensor, k_keep: int, patch_size: int):
    assert res_torch.dim() == 5 and res_torch.size(1) == 1, "res must be (B,1,T,H,W)"
    B, _, T, H, W = res_torch.shape
    ph = pw = patch_size
    assert H % ph == 0 and W % pw == 0, "H/W must be divisible by patch size"
    hb, wb = H // ph, W // pw
    L = T * hb * wb

    K = int(max(0, min(k_keep, L)))
    res_abs = res_torch.abs().squeeze(1)  # (B,T,H,W)
    scores = res_abs.reshape(B, T, hb, ph, wb, pw).sum(dim=(3,5)).reshape(B, L)

    if K > 0:
        topk_idx = torch.topk(scores, k=K, dim=1, largest=True, sorted=False).indices
        visible_indices = torch.sort(topk_idx, dim=1).values
    else:
        visible_indices = torch.empty(B, 0, dtype=torch.long, device=res_torch.device)
    return visible_indices  # (B,K)

def process_one_video(
    video_path: str,
    seq_len: int,
    patch_size: int,
    K: int,
    hevc_n_parallel: int,
    hevc_y_only: bool,
) -> np.ndarray:
    """Return shape=(K,) int32 visible indices. Return all 0 on failure (for checkpoint resume marking)."""
    try:
        os.environ["UMT_HEVC_Y_ONLY"] = "1" if hevc_y_only else "0"
        prefix_fast = int(os.environ.get("HEVC_PREFIX_FAST", "1")) != 0

        # video_path = _maybe_swap_to_hevc(video_path)
        vr = decord.VideoReader(video_path, num_threads=max(4, hevc_n_parallel), ctx=decord.cpu(0))
        duration = len(vr)

        T = seq_len
        if duration >= T:
            frame_id_list = list(range(T))
        else:
            frame_id_list = list(range(duration)) + [duration - 1] * (T - duration)

        # I-frame (position relative to T segment)
        key_idx = None
        if hasattr(vr, "get_key_indices"):
            key_idx = vr.get_key_indices()
        elif hasattr(vr, "get_keyframes"):
            key_idx = vr.get_keyframes()
        if key_idx is not None:
            I_global = set(int(i) for i in np.asarray(key_idx).tolist())
        else:
            I_global = set(int(i) for i in range(0, duration, 16))
        I_pos = set([i for i, fid in enumerate(frame_id_list) if fid in I_global])

        # Read residual (Y channel), set I-frame position to 0
        Tsel = T
        residuals_y = [None] * Tsel
        H0 = W0 = None
        dtype0 = None

        def _ensure_y(arr):
            nonlocal H0, W0, dtype0
            y = arr[0] if isinstance(arr, tuple) else arr
            y = np.asarray(y)
            if y.ndim == 3:
                y = np.squeeze(y)
            if H0 is None:
                H0, W0 = int(y.shape[0]), int(y.shape[1]); dtype0 = y.dtype
            return y

        if all(fid == i for i, fid in enumerate(frame_id_list)) and prefix_fast:
            rdr = ResPipeReader(video_path, nb_frames=Tsel, n_parallel=hevc_n_parallel)
            try:
                for i, res in enumerate(rdr.next_residual()):
                    if i < Tsel:
                        if i in I_pos:
                            if H0 is None:
                                y0 = _ensure_y(res)
                                residuals_y[i] = np.zeros_like(y0, dtype=y0.dtype)
                            else:
                                residuals_y[i] = np.zeros((H0, W0), dtype=dtype0 or np.uint8)
                        else:
                            y = _ensure_y(res); residuals_y[i] = y
                    if i + 1 >= Tsel:
                        break
            finally:
                try: rdr.close()
                except Exception: pass
        else:
            rdr = ResPipeReader(video_path, nb_frames=None, n_parallel=hevc_n_parallel)
            try:
                cur_idx = 0
                wanted = set(frame_id_list)
                idx2pos = {fid: i for i, fid in enumerate(frame_id_list)}
                for res in rdr.next_residual():
                    if cur_idx in wanted:
                        pos = idx2pos[cur_idx]
                        if pos in I_pos:
                            if H0 is None:
                                y0 = _ensure_y(res)
                                H0, W0, dtype0 = y0.shape[0], y0.shape[1], y0.dtype
                            residuals_y[pos] = np.zeros((H0, W0), dtype=dtype0 or np.uint8)
                        else:
                            y = _ensure_y(res); residuals_y[pos] = y
                        if all(x is not None for x in residuals_y):
                            break
                    cur_idx += 1
            finally:
                try: rdr.close()
                except Exception: pass

        if dtype0 is None:
            dtype0 = np.uint8
            H0 = H0 or 224; W0 = W0 or 224
        for i in range(Tsel):
            if residuals_y[i] is None:
                residuals_y[i] = np.zeros((H0, W0), dtype=dtype0)

        res_stack_u8 = np.stack(residuals_y, axis=0)  # (T,H0,W0) uint8

        # resize -> 224 & convert to signed residual
        # res224_u8 = np.empty((Tsel, 224, 224), dtype=np.uint8)
        # for i in range(Tsel):
        #     print(res_stack_u8[i].shape)
            # res224_u8[i] = _resize_u8_gray(res_stack_u8[i], size=224)

        # res224_signed = res224_u8.astype(np.int16) - 128  # (T,224,224)
        res224_signed = res_stack_u8.astype(np.int16) - 128  # (T,224,224)

        # Calculate Top-K
        res_torch = torch.from_numpy(res224_signed).to(torch.int16).unsqueeze(0).unsqueeze(1)  # (1,1,T,224,224)
        with torch.no_grad():
            vis_idx = mask_by_residual_topk(res_torch, K, patch_size)  # (1,K)
        vis_idx_np = vis_idx.squeeze(0).cpu().numpy().astype(np.int32)  # (K,)
        return vis_idx_np

    except Exception:
        sys.stderr.write(f"[WARN] failed: {video_path}\n{traceback.format_exc()}\n")
        return np.zeros((K,), dtype=np.int32)  # Use all 0 as placeholder on failure

def _find_zero_rows_memmap(mm: np.memmap, chunk: int = 20000):
    """Keep original function, but no longer used in per-file save mode"""
    N = mm.shape[0]
    todo = []
    for st in range(0, N, chunk):
        ed = min(N, st + chunk)
        blk = mm[st:ed]           # memmap slice, read block by block
        # Whether row is all 0: equivalent to (blk != 0).any(axis=1) == False
        row_zero = ~(blk != 0).any(axis=1)
        idxs = np.nonzero(row_zero)[0] + st
        if idxs.size:
            todo.extend(idxs.tolist())
    return todo


def _make_out_path(video_path: str, src: str, dst: str, suffix: str = ".visidx.npy") -> str:
    """String replace based on original video path, keep hierarchy unchanged, only change prefix, and change extension to .visidx.npy"""
    if src not in video_path:
        raise ValueError(f"--out-replace-src '{src}' not in video path: {video_path}")
    replaced = video_path.replace(src, dst)
    stem, _ = os.path.splitext(replaced)
    out_path = stem + suffix
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True, help="Video list file (one path per line)")
    ap.add_argument("--seq_len", type=int, default=64, help="T: number of frames per video (repeat last frame if insufficient)")
    ap.add_argument("--patch_size", type=int, default=16, help="ViT patch size, must divide 224")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--keep_ratio", type=float, default=0.30, help="Keep ratio (0~1), used to calculate K")
    g.add_argument("--k_keep",     type=int,   default=2000, help="Directly specify K")
    ap.add_argument("--hevc_n_parallel", type=int, default=6, help="ResPipeReader parallelism")
    ap.add_argument("--hevc_y_only",     type=int, default=1, help="Y channel residual (1/0)")
    ap.add_argument("--flush_every",     type=int, default=100, help="Print log every N videos processed")
    # Per-file output related
    ap.add_argument("--out_replace_src", required=True, help="Output path replacement: substring in original path (e.g. original root directory)")
    ap.add_argument("--out_replace_dst", required=True, help="Output path replacement: replacement substring (e.g. new root directory)")
    ap.add_argument("--out_suffix", default=".visidx.npy", help="Output file suffix for each video, default .visidx.npy")
    ap.add_argument("--overwrite",  type=int, default=0, help="Whether to overwrite if output file exists (0 skip, 1 overwrite)")
    ap.add_argument("--local_rank", type=int, default=0, help="Local rank (auto-injected by DeepSpeed)")
    args = ap.parse_args()

    videos = _load_list_file(args.list)  # Preserve order     
    N = len(videos)
    if N == 0:
        raise RuntimeError("empty list")

    # Calculate L and K
    T = int(args.seq_len)
    p = int(args.patch_size)
    assert 224 % p == 0, "224 must be divisible by patch_size"
    hb = 224 // p
    wb = 224 // p
    L = T * hb * wb

    if args.k_keep is not None and args.k_keep >= 0:
        K = min(int(args.k_keep), L)
    else:
        keep_ratio = max(0.0, min(float(args.keep_ratio), 1.0))
        K = int(round(L * keep_ratio))
    if K <= 1:
        raise RuntimeError(f"K={K} is unsafe (may conflict with all-0 rows), please set larger keep-ratio or k-keep.")

    # Evenly split list to each rank
    num_local = N // WORLD_SIZE + int(RANK < (N % WORLD_SIZE))
    start = (N // WORLD_SIZE) * RANK + min(RANK, N % WORLD_SIZE)
    end = start + num_local
    local_indices = list(range(start, end))

    if RANK == 0:
        print(f"[dist] WORLD_SIZE={WORLD_SIZE}, N={N}, K={K}, slice per rank ~{N//max(1,WORLD_SIZE)} (+remainder)")
    print(f"[rank {RANK}/{WORLD_SIZE}] will process indices [{start}, {end}) => {num_local} videos")

    processed = 0
    for c, i in enumerate(local_indices, 1):
        vp = videos[i].strip()
        if not vp:
            sys.stderr.write(f"[WARN][rank {RANK}] empty line at {i}, skip\n")
            continue

        try:
            out_path = _make_out_path(
                video_path=vp,
                src=args.out_replace_src,
                dst=args.out_replace_dst,
                suffix=args.out_suffix,
            )
        except Exception as e:
            sys.stderr.write(f"[ERROR][rank {RANK}] make_out_path failed at idx {i}: {e}\n")
            continue

        # if os.path.exists(out_path) and not bool(args.overwrite):
        #     if (c % max(1, args.flush_every)) == 0:
        #         print(f"[rank {RANK}] skip exists: {out_path} (c={c}/{len(local_indices)})")
        #     continue

        vis_idx = process_one_video(
            video_path=vp,
            seq_len=T,
            patch_size=p,
            K=K,
            hevc_n_parallel=args.hevc_n_parallel,
            hevc_y_only=bool(args.hevc_y_only),
        )

        try:
            np.save(out_path, vis_idx.astype(np.int32), allow_pickle=False)
        except Exception as e:
            sys.stderr.write(f"[ERROR][rank {RANK}] save failed at {out_path}: {e}\n")
            continue

        processed += 1
        if (c % max(1, args.flush_every)) == 0:
            print(f"[rank {RANK}] processed {c}/{len(local_indices)} (global idx {i}), saved: {out_path}")

    print(f"[rank {RANK}] done. processed={processed}, assigned={len(local_indices)}")

if __name__ == "__main__":
    main()
