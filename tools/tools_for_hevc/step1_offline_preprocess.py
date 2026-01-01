import argparse
import os
from pathlib import Path

import imageio
import numpy as np
import torch

from dataloader.data_decord_video import dali_dataloader

# Environment variables (consistent with original code)
RANK = int(os.environ.get("RANK", "0"))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))

from concurrent.futures import ThreadPoolExecutor

def _process_and_write_sample(sample_cfhw, label_val, out_mp4, out_npy, mean255, std255, fps=12):
    # sample_cfhw: (C, F, H, W)
    C, F, H, W = sample_cfhw.shape
    frames = []
    for fidx in range(F):
        frame_chw = sample_cfhw[:, fidx, :, :]
        if np.issubdtype(frame_chw.dtype, np.floating):
            img = _undo_dali_norm(frame_chw, mean255, std255)
        else:
            img = np.transpose(frame_chw, (1, 2, 0)).astype(np.uint8)
        frames.append(img)

    _write_video_rgb(frames, out_mp4, fps=fps)
    np.save(out_npy, np.array(label_val, dtype=np.int32))
    return str(out_mp4), str(out_npy), label_val

def _to_numpy(x):
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.array(x)


def _undo_dali_norm(frame_chw, mean255, std255):
    # frame_chw: C x H x W (float32), output of crop_mirror_normalize
    mean = np.asarray(mean255, dtype=np.float32).reshape(-1, 1, 1)
    std = np.asarray(std255, dtype=np.float32).reshape(-1, 1, 1)
    img = frame_chw * std + mean  # back to 0..255
    img = np.transpose(img, (1, 2, 0))
    return np.clip(img, 0, 255).astype(np.uint8)


def _write_video_rgb(frames_uint8_hwc, out_path, fps=12):
    """
    Use imageio+ffmpeg to write RGB frames to mp4 (yuv420p), ensuring playability in common players.
    """
    out_path = str(out_path)
    writer = imageio.get_writer(
        out_path,
        fps=fps,
        codec="libx264",
        quality=8,
        format="FFMPEG",
        macro_block_size=None,
        ffmpeg_params=["-pix_fmt", "yuv420p"],
    )
    try:
        for f in frames_uint8_hwc:
            writer.append_data(f)
    finally:
        writer.close()


def _build_epoch_order_indices(file_list_len, batch_size, num_shards=1, shard_id=0, seed=0, rank=0):
    """
    Reproduce the sample order indices after shuffle, sharding, and drop-last for the first epoch in ExternalInputCallable train mode.
    Used to map DALI output batches back to original file_list indices for naming outputs with original filenames.
    """
    shard_size = file_list_len // num_shards
    shard_offset = shard_size * shard_id
    full_iterations = shard_size // batch_size
    sample_count = full_iterations * batch_size  # Number of samples participating in this epoch after drop-last

    rng = np.random.default_rng(seed=seed + rank)
    perm = rng.permutation(file_list_len)
    order = perm[shard_offset: shard_offset + sample_count]
    # Return 2D list sliced by batch: batch -> [idx0, idx1, ...]
    batched = [order[i * batch_size: (i + 1) * batch_size] for i in range(full_iterations)]
    return batched, full_iterations


def _bucket_dir_from_serial(base_dir: Path, serial: int) -> Path:
    """
    Split into two-level directories using the last two digits of the "serial number" (multi-level directories):
    - Tens digit as first level directory (0-9)
    - Ones digit as second level directory (0-9)
    For example serial=123 -> 2/3; serial=37 -> 3/7
    Total 10x10=100 leaf folders.
    """
    tens = (serial // 10) % 10
    ones = serial % 10
    p = base_dir / f"{tens}" / f"{ones}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def main():
    parser = argparse.ArgumentParser(description="Export augmented square videos (1s@16fps) and labels using DALI train pipeline")
    parser.add_argument("--file_list", type=str, required=True, help="Text file, one video path per line")
    parser.add_argument("--labels_npy", type=str, required=True, help="Label npy corresponding to file-list (can be int/np.int64), will be saved as int32")
    parser.add_argument("--outdir", type=str, required=True, help="Output directory, will generate *.mp4 and *_label.npy (multi-level directories, split into 100 folders by last two digits of serial number)")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--sequence_length", type=int, default=64, help="Expected 64 (script will resample/pad for non-64 frames)")
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--short_side_size", type=int, default=256)
    parser.add_argument("--dali_threads", type=int, default=2)
    parser.add_argument("--dali_py_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0, help="Seed affecting sample shuffle order (consistent with ExternalInputCallable: seed+RANK)")
    parser.add_argument("--use_original_names", action="store_true", help="Try to reproduce shuffle to use original filenames for output naming")
    parser.add_argument("--io_workers", type=int, default=4, help="Number of threads for parallel file writing (recommended 2-8)")
    parser.add_argument("--local_rank")
    args = parser.parse_args()

    # Get sharding info from environment variables (DeepSpeed/distributed environment)
    num_shards = WORLD_SIZE
    shard_id = RANK
    print(f"[Shard] Using env sharding: num_shards={num_shards}, shard_id={shard_id} (WORLD_SIZE={WORLD_SIZE}, RANK={RANK}, LOCAL_RANK={LOCAL_RANK})")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # mp4 relative path list for each GPU (relative to outdir)
    mp4_rel_paths = []

    with open(args.file_list, "r", encoding="utf-8") as f:
        file_list = [ln.strip() for ln in f if ln.strip()]
    if not file_list:
        raise SystemExit("file_list is empty")

    labels_arr = np.load(args.labels_npy)
    if len(labels_arr) < len(file_list):
        raise SystemExit(f"labels-npy shorter than file_list: {len(labels_arr)} < {len(file_list)}")

    # mean/std consistent with dali_dataloader (multiplied by 255)
    mean255 = [x * 255 for x in [0.48145466, 0.4578275, 0.40821073]]
    std255 = [x * 255 for x in [0.26862954, 0.26130258, 0.27577711]]

    # Build DALI iterator (train mode)
    dataloader = dali_dataloader(
        file_list=file_list,
        label=labels_arr,
        dali_num_threads=args.dali_threads,
        dali_py_num_workers=args.dali_py_workers,
        batch_size=args.batch_size,
        input_size=args.input_size,
        sequence_length=args.sequence_length,
        stride=1,
        mode="train",
        seed=args.seed,
        short_side_size=args.short_side_size,
        num_shards=num_shards,
        shard_id=shard_id,
    )

    # Reproduce sample order for this epoch (for naming)
    planned_batches = None
    total_planned_batches = None
    if args.use_original_names and num_shards == 1:
        planned_batches, total_planned_batches = _build_epoch_order_indices(
            file_list_len=len(file_list),
            batch_size=args.batch_size,
            num_shards=num_shards,
            shard_id=shard_id,
            seed=args.seed,
            rank=RANK,
        )

    it = iter(dataloader)

    saved = 0
    batch_idx = 0

    while True:
        
        if saved * WORLD_SIZE > 30000000:
            break

        try:
            batch = next(it)  # DALIWarper.__next__ -> {"pixel_values": videos, "labels": labels}
        except StopIteration:
            break

        videos = batch["pixel_values"]
        labels_out = batch["labels"]

        vids_np = _to_numpy(videos)  # [B, C, F, H, W]
        labs_np = _to_numpy(labels_out)  # [B] or [B,1], int64

        if vids_np.ndim != 5:
            raise RuntimeError(f"Unexpected videos shape: {vids_np.shape} (expected 5D BxCxFxHxW)")

        B, C, F, H, W = vids_np.shape

        # Calculate naming and "serial number" for each sample in this batch (for bucket directory)
        names_for_batch = []
        serials_for_batch = []
        fallback = False
        if planned_batches is not None and batch_idx < len(planned_batches):
            idxs = planned_batches[batch_idx]
            if len(idxs) == B:
                # Try to verify label consistency (fallback if inconsistent)
                try:
                    for j in range(B):
                        src_idx = idxs[j]
                        # labels_arr[src_idx] may be int or array
                        src_label = labels_arr[src_idx]
                        src_label_int = int(src_label) if np.isscalar(src_label) else int(src_label[0])
                        out_label_int = int(labs_np[j]) if np.isscalar(labs_np[j]) else int(labs_np[j][0])
                        if src_label_int != out_label_int:
                            fallback = True
                            break
                    if not fallback:
                        for j in range(B):
                            src_idx = idxs[j]
                            stem = Path(file_list[src_idx]).stem
                            names_for_batch.append(stem)
                            # Use file_list index as stable serial number
                            serials_for_batch.append(int(src_idx))
                except Exception:
                    fallback = True
            else:
                fallback = True
        else:
            fallback = True

        if fallback:
            # Name with sequential numbering, serial also uses sequential number
            names_for_batch = [f"rank_{RANK:03d}_sample_{saved + j:010d}" for j in range(B)]
            serials_for_batch = [int(saved + j) for j in range(B)]

        # Parallel file writing (split into 100 multi-level directories by last two digits of serial: tens/ones)
        io_workers = args.io_workers if args.io_workers and args.io_workers > 0 else max(2, min(8, (os.cpu_count() or 4) // 2))

        futures = []
        with ThreadPoolExecutor(max_workers=io_workers) as ex:
            for b in range(B):
                sample = vids_np[b]  # C x F x H x W (numpy view, no copy needed between threads)
                label_val = labs_np[b]
                if isinstance(label_val, np.ndarray):
                    label_val = label_val.squeeze()

                stem = names_for_batch[b]
                serial = serials_for_batch[b]
                bucket_dir = _bucket_dir_from_serial(outdir, serial)

                out_mp4 = bucket_dir / f"{stem}.mp4"
                out_npy = bucket_dir / f"{stem}_label.npy"

                futures.append(
                    ex.submit(
                        _process_and_write_sample,
                        sample, label_val, out_mp4, out_npy,
                        mean255, std255, 12
                    )
                )

            # Collect in submission order (keep print order stable; can also use as_completed for early printing)
            for fut in futures:
                out_mp4_s, out_npy_s, label_i = fut.result()
                saved += 1
                # Record mp4 path relative to outdir
                rel_mp4 = Path(out_mp4_s).relative_to(outdir).as_posix()
                mp4_rel_paths.append(rel_mp4)

                bucket_rel = str(Path(out_mp4_s).parent.relative_to(outdir))
                # print(f"[{saved}] Saved video: {out_mp4_s}, label: {out_npy_s} (label={label_i}, bucket={bucket_rel})")

        batch_idx += 1

    # Each GPU writes its mp4 relative path list to file
    list_file = outdir / f"mp4_list_rank_{RANK:03d}.txt"
    with open(list_file, "w", encoding="utf-8") as f:
        if mp4_rel_paths:
            f.write("\n".join(mp4_rel_paths) + "\n")
    print(f"Saved mp4 relative path list for rank {RANK} to: {list_file}")

    print(f"Done. Saved {saved} videos to {outdir} (bucketed by last two digits: tens/ones).")


if __name__ == "__main__":
    main()