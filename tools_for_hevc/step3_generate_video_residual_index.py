import os
import sys
import math
import argparse
import traceback
import numpy as np
import decord

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
from numpy.lib.format import open_memmap

# ---- 你的残差读取器 ----
from hevc_feature_decoder import ResPipeReader


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

def _maybe_swap_to_hevc(p: str) -> str:
    try:
        if not isinstance(p, str):
            return p
        old_seg = "/videos_frames64_kinetics_ssv2/videos_frames64_kinetics_ssv2/"
        new_seg = "/videos_frames64_kinetics_ssv2/videos_frames64_kinetics_ssv2_hevc/"
        if old_seg in p:
            cand = p.replace(old_seg, new_seg)
            if os.path.exists(cand):
                return cand
    except Exception:
        pass
    return p

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

        video_path = _maybe_swap_to_hevc(video_path)
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
        res224_u8 = np.empty((Tsel, 224, 224), dtype=np.uint8)
        for i in range(Tsel):
            res224_u8[i] = _resize_u8_gray(res_stack_u8[i], size=224)
        res224_signed = res224_u8.astype(np.int16) - 128  # (T,224,224)

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
    """返回所有全 0 行的索引列表（分块扫描，避免一次性读全量内存）"""
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

#   --list /video_vit/train_UniViT/mp4_list.txt \
#   --out-file /video_vit/train_UniViT/visible_indices.npy \
#   --seq-len 16 --patch-size 16 --keep-ratio 0.30 \
#   --hevc-n-parallel 6 --hevc-y-only 1 --resume 1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True, help="视频列表文件（每行一个路径）")
    ap.add_argument("--out-file", required=True, help="合并输出（.npy），仅保存 visible_indices，总形状 (N,K)")
    ap.add_argument("--seq-len", type=int, default=64, help="T：每视频使用的帧数（不足重复最后一帧）")
    ap.add_argument("--patch-size", type=int, default=16, help="ViT patch 大小，需整除 224")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--keep-ratio", type=float, default=0.30, help="保留比例（0~1），用于计算 K")
    g.add_argument("--k-keep",     type=int,   default=1568, help="直接指定 K")
    ap.add_argument("--hevc-n-parallel", type=int, default=6, help="ResPipeReader 并行度")
    ap.add_argument("--hevc-y-only",     type=int, default=1, help="Y 通道残差（1/0）")
    ap.add_argument("--flush-every",     type=int, default=100, help="每处理多少视频 flush 一次")
    # 断点续跑相关
    ap.add_argument("--resume",          type=int, default=1, help="若 out-file 存在则断点续跑（按全 0 行判定）")
    ap.add_argument("--overwrite",       type=int, default=0, help="忽略旧文件，重新创建并清零")
    ap.add_argument("--scan-chunk",      type=int, default=20000, help="扫描全 0 行时的分块大小")
    ap.add_argument("--prezero",         type=int, default=1, help="新建文件时是否整体清零（便于断点续跑）")
    args = ap.parse_args()

    videos = _load_list_file(args.list)  # 保序
    N = len(videos)
    if N == 0:
        raise RuntimeError("empty list")

    # 计算 L 与 K（或从现有文件继承）
    T = int(args.seq_len)
    p = int(args.patch_size)
    assert 224 % p == 0, "224 必须能被 patch_size 整除"
    hb = 224 // p
    wb = 224 // p
    L = T * hb * wb

    out_file = os.path.abspath(args.out_file)
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    mm = None
    K = None

    if os.path.exists(out_file) and not args.overwrite and args.resume:
        # --- 断点续跑：沿用现有文件 ---
        mm = open_memmap(out_file, mode="r+")
        if mm.dtype != np.int32:
            raise RuntimeError(f"existing file dtype {mm.dtype}, expected int32")
        if mm.shape[0] != N:
            raise RuntimeError(f"existing file rows {mm.shape[0]} != videos {N}（列表内容/数量不一致）")
        K_file = int(mm.shape[1])
        if args.k_keep is not None:
            # 显式指定 K，需与现有文件一致
            if int(args.k_keep) != K_file:
                raise RuntimeError(f"K mismatch: existing {K_file} vs arg {args.k_keep}. 使用 --overwrite 1 重建，或去掉 --k-keep。")
            K = K_file
        else:
            # 未指定则沿用文件的 K
            K = K_file
        if K <= 1:
            raise RuntimeError("K<=1 在“全 0 行判未完成”的语义下不安全，请使用更大的 K 或改用其它占位符策略。")

        # 找出未完成（全 0 行）的索引
        todo = _find_zero_rows_memmap(mm, chunk=args.scan_chunk)
        if not todo:
            print(f"[info] nothing to do, all {N} rows already filled (N,K)=({N},{K})")
            return
        print(f"[resume] found {len(todo)}/{N} rows to process (N,K)=({N},{K})")
    else:
        # --- 新建文件 ---
        # 计算 K（保底 K>=1，但我们强制 K>=2 更安全）
        if args.k_keep is not None and args.k_keep >= 0:
            K = min(int(args.k_keep), L)
        else:
            keep_ratio = max(0.0, min(float(args.keep_ratio), 1.0))
            K = int(round(L * keep_ratio))
        if K <= 1:
            raise RuntimeError(f"K={K} 不安全（可能与全 0 行冲突），请设置更大的 keep-ratio 或 k-keep。")

        mm = open_memmap(out_file, mode="w+", dtype=np.int32, shape=(N, K))
        if args.prezero:
            mm[:] = 0    # 显式清零，保障“全 0 = 未完成”
            mm.flush()
        todo = list(range(N))
        print(f"[init] create {out_file} with shape (N,K)=({N},{K}), zero-initialized={bool(args.prezero)}")

    # --- 主循环：仅处理未完成行（全 0 行） ---
    for c, i in enumerate(todo, 1):
        vp = videos[i].strip()
        if not vp:
            sys.stderr.write(f"[WARN] empty line at {i}, fill zeros\n")
            mm[i, :] = 0
        else:
            vis_idx = process_one_video(
                video_path=vp,
                seq_len=T,
                patch_size=p,
                K=K,
                hevc_n_parallel=args.hevc_n_parallel,
                hevc_y_only=bool(args.hevc_y_only),
            )
            mm[i, :] = vis_idx  # 若失败，这里就是全 0；下次还能继续
        if (c % args.flush_every) == 0:
            mm.flush()
            print(f"[info] processed {c}/{len(todo)} (global row {i})")

    mm.flush()
    try:
        os.fsync(mm.fp.fileno())
    except Exception as e:
        print(f"[warn] fsync failed: {e}")
    print(f"[done] saved visible_indices to {out_file} (shape={(N,K)})")

if __name__ == "__main__":
    main()

# import math

# # ==== 输入与输出配置 ====
# in_path = "/video_vit/train_UniViT/mp4_list_part_3.txt"
# out_prefix = "/video_vit/train_UniViT/mp4_list_part_3_split_"  # 输出文件名前缀
# num_parts = 4  # 想分成几份

# # ==== 读取所有非空行 ====
# with open(in_path, "r", encoding="utf-8") as f:
#     lines = [ln for ln in f if ln.strip()]

# total = len(lines)
# chunk = math.ceil(total / num_parts)
# print(f"总行数: {total}, 每份约 {chunk} 行")

# # ==== 按顺序切分并保存 ====
# for i in range(num_parts):
#     part_lines = lines[i * chunk : (i + 1) * chunk]
#     out_path = f"{out_prefix}{i:02d}.txt"
#     with open(out_path, "w", encoding="utf-8") as f:
#         f.writelines(part_lines)
#     print(f"写出 {len(part_lines)} 行 → {out_path}")

# print("✅ 文件切分完成！")