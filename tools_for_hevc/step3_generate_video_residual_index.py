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
except Exception:
    _HAS_CV2 = False
try:
    from PIL import Image
    _HAS_PIL = True
except Exception:
    _HAS_PIL = False

import torch
from hevc_feature_decoder import ResPipeReader

# ===== 分布式环境变量（与原代码一致）=====
RANK = int(os.environ.get("RANK", "0"))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
# ========================================


# ---------- 小工具 ----------
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
    return lines  # 每行一个视频路径

def mask_by_residual_topk(res_torch: torch.Tensor, k_keep: int, patch_size: int):
    assert res_torch.dim() == 5 and res_torch.size(1) == 1, "res 需为 (B,1,T,H,W)"
    B, _, T, H, W = res_torch.shape
    ph = pw = patch_size
    assert H % ph == 0 and W % pw == 0, "H/W 必须能被 patch 大小整除"
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
    """返回 shape=(K,) 的 int32 可见索引。失败则返回全 0（用于断点续跑标记）。"""
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

        # I 帧（相对 T 片段的位置）
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

        # 读残差（Y 通道），I 位置置 0
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

        # resize -> 224 & 转有符号残差
        # res224_u8 = np.empty((Tsel, 224, 224), dtype=np.uint8)
        # for i in range(Tsel):
        #     print(res_stack_u8[i].shape)
            # res224_u8[i] = _resize_u8_gray(res_stack_u8[i], size=224)

        # res224_signed = res224_u8.astype(np.int16) - 128  # (T,224,224)
        res224_signed = res_stack_u8.astype(np.int16) - 128  # (T,224,224)

        # 计算 Top-K
        res_torch = torch.from_numpy(res224_signed).to(torch.int16).unsqueeze(0).unsqueeze(1)  # (1,1,T,224,224)
        with torch.no_grad():
            vis_idx = mask_by_residual_topk(res_torch, K, patch_size)  # (1,K)
        vis_idx_np = vis_idx.squeeze(0).cpu().numpy().astype(np.int32)  # (K,)
        return vis_idx_np

    except Exception:
        sys.stderr.write(f"[WARN] failed: {video_path}\n{traceback.format_exc()}\n")
        return np.zeros((K,), dtype=np.int32)  # 失败用全 0 占位

def _find_zero_rows_memmap(mm: np.memmap, chunk: int = 20000):
    """保留原函数，但分文件保存模式下不再使用"""
    N = mm.shape[0]
    todo = []
    for st in range(0, N, chunk):
        ed = min(N, st + chunk)
        blk = mm[st:ed]           # memmap 切片，逐块触盘
        # 行是否全 0：等价于 (blk != 0).any(axis=1) == False
        row_zero = ~(blk != 0).any(axis=1)
        idxs = np.nonzero(row_zero)[0] + st
        if idxs.size:
            todo.extend(idxs.tolist())
    return todo


def _make_out_path(video_path: str, src: str, dst: str, suffix: str = ".visidx.npy") -> str:
    """基于原始视频路径做字符串 replace，保持层级不变，只改前缀，并将扩展改为 .visidx.npy"""
    if src not in video_path:
        raise ValueError(f"--out-replace-src '{src}' 不在视频路径中：{video_path}")
    replaced = video_path.replace(src, dst)
    stem, _ = os.path.splitext(replaced)
    out_path = stem + suffix
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True, help="视频列表文件（每行一个路径）")
    ap.add_argument("--seq_len", type=int, default=64, help="T：每视频使用的帧数（不足重复最后一帧）")
    ap.add_argument("--patch_size", type=int, default=16, help="ViT patch 大小，需整除 224")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--keep_ratio", type=float, default=0.30, help="保留比例（0~1），用于计算 K")
    g.add_argument("--k_keep",     type=int,   default=2000, help="直接指定 K")
    ap.add_argument("--hevc_n_parallel", type=int, default=6, help="ResPipeReader 并行度")
    ap.add_argument("--hevc_y_only",     type=int, default=1, help="Y 通道残差（1/0）")
    ap.add_argument("--flush_every",     type=int, default=100, help="每处理多少视频打印一次日志")
    # 分文件输出相关
    ap.add_argument("--out_replace_src", required=True, help="输出路径替换：原始路径中的子串（如原根目录）")
    ap.add_argument("--out_replace_dst", required=True, help="输出路径替换：替换成的子串（如新根目录）")
    ap.add_argument("--out_suffix", default=".visidx.npy", help="每个视频结果文件后缀，默认 .visidx.npy")
    ap.add_argument("--overwrite",  type=int, default=0, help="输出文件存在时是否覆盖（0 跳过，1 覆盖）")
    ap.add_argument("--local_rank", type=int, default=0, help="输出文件存在时是否覆盖（0 跳过，1 覆盖）")
    args = ap.parse_args()

    videos = _load_list_file(args.list)  # 保序     
    N = len(videos)
    if N == 0:
        raise RuntimeError("empty list")

    # 计算 L 与 K
    T = int(args.seq_len)
    p = int(args.patch_size)
    assert 224 % p == 0, "224 必须能被 patch_size 整除"
    hb = 224 // p
    wb = 224 // p
    L = T * hb * wb

    if args.k_keep is not None and args.k_keep >= 0:
        K = min(int(args.k_keep), L)
    else:
        keep_ratio = max(0.0, min(float(args.keep_ratio), 1.0))
        K = int(round(L * keep_ratio))
    if K <= 1:
        raise RuntimeError(f"K={K} 不安全（可能与全 0 行冲突），请设置更大的 keep-ratio 或 k-keep。")

    # 将列表均匀切给各 rank
    num_local = N // WORLD_SIZE + int(RANK < (N % WORLD_SIZE))
    start = (N // WORLD_SIZE) * RANK + min(RANK, N % WORLD_SIZE)
    end = start + num_local
    local_indices = list(range(start, end))

    if RANK == 0:
        print(f"[dist] WORLD_SIZE={WORLD_SIZE}, N={N}, K={K}, slice per rank ~{N//max(1,WORLD_SIZE)} (+余数)")
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
