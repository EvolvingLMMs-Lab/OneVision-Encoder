#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Video feature extraction using DALI + MetaCLIP-H/14 with checkpoint resume.
Features are saved in order matching the input video list.

Usage with DeepSpeed:
    deepspeed --num_gpus=8 extract_video_features.py \
        --input /video_vit/train_UniViT/mp4_list.txt \
        --output /output/features \
        --batch_size 32 \
        --num_frames 8 \
        --chunk_size 1000
    
    # Multi-node
    deepspeed --num_gpus=8 --num_nodes=2 --hostfile=hostfile extract_video_features.py \
        --input /video_vit/train_UniViT/mp4_list.txt \
        --output /output/features \
        --batch_size 32 \
        --num_frames 8
"""

import argparse
import os
import pickle
from pathlib import Path
from typing import Any, Dict, Sequence

import decord
import numpy as np
import torch
import open_clip
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.pipeline import pipeline_def
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy

# Get distributed info from environment (set by DeepSpeed)
rank = int(os.environ.get("RANK", "0"))
local_rank = int(os.environ.get("LOCAL_RANK", "0"))
world_size = int(os.environ.get("WORLD_SIZE", "1"))

# Set CUDA device
if torch.cuda.is_available():
    torch.cuda.set_device(local_rank)


class DALIWarper:
    """Wrapper around DALIGenericIterator."""
    
    def __init__(self, dali_iter: DALIGenericIterator, length: int = None):
        self.iter = dali_iter
        self._length = length
    
    def __next__(self):
        data = self.iter.__next__()[0]
        return data["videos"], data["labels"]
    
    def __iter__(self):
        return self
    
    def __len__(self):
        return self._length if self._length else 0


class ExternalInputCallable:
    """External input for DALI pipeline using decord."""
    
    def __init__(self, source_params: Dict[str, Any]):
        self.file_list = source_params["file_list"]
        self.sequence_length = source_params["sequence_length"]
        self.decord_threads = source_params["decord_threads"]
        self.label = np.array([-1], dtype=np.int64)
        self.fallback_video = self.file_list[0]
    
    def _read_video(self, video_path: str) -> np.ndarray:
        """Read and sample frames from video using decord."""
        vr = decord.VideoReader(video_path, num_threads=self.decord_threads, ctx=decord.cpu(0))
        duration = len(vr)
        
        if duration == 0:
            raise ValueError(f"Video has 0 frames")
        
        # Uniform sampling
        step = max(1, duration // self.sequence_length)
        offset = step // 2
        frame_ids = [min(i * step + offset, duration - 1) for i in range(self.sequence_length)]
        
        vr.seek(0)
        return vr.get_batch(frame_ids).asnumpy()
    
    def __call__(self, sample_info):
        idx = sample_info.idx_in_epoch
        if idx >= len(self.file_list):
            raise StopIteration
        
        video_path = self.file_list[idx]
        
        try:
            video_data = self._read_video(video_path)
        except Exception as e:
            print(f"[Rank {rank}] Error: {video_path} -> {e}, using fallback")
            video_data = self._read_video(self.fallback_video)
        
        return video_data, self.label


@pipeline_def()
def dali_pipeline(source_params: Dict[str, Any]):
    """DALI pipeline definition."""
    videos, labels = fn.external_source(
        source=ExternalInputCallable(source_params),
        num_outputs=2,
        batch=False,
        parallel=True,
        dtype=[types.UINT8, types.INT64],
        layout=["FHWC", "C"],
    )
    
    videos = videos.gpu()
    videos = fn.resize(
        videos,
        device="gpu",
        antialias=True,
        interp_type=types.INTERP_LINEAR,
        resize_shorter=source_params["short_side_size"],
    )
    videos = fn.crop(
        videos,
        device="gpu",
        crop=[source_params["input_size"], source_params["input_size"]],
    )
    videos = fn.crop_mirror_normalize(
        videos,
        device="gpu",
        dtype=types.FLOAT,
        output_layout="CFHW",
        mean=source_params["mean"],
        std=source_params["std"],
    )
    
    return videos.gpu(), labels.gpu()


def dali_dataloader(
    file_list: Sequence[str],
    batch_size: int,
    num_frames: int,
    input_size: int = 224,
    dali_num_threads: int = 2,
    dali_py_workers: int = 16,
    decord_threads: int = 2,
) -> DALIWarper:
    """Create DALI dataloader for video data."""
    
    # CLIP normalization
    mean = [x * 255 for x in [0.48145466, 0.4578275, 0.40821073]]
    std = [x * 255 for x in [0.26862954, 0.26130258, 0.27577711]]
    print(len(file_list), "videos for this rank")
    source_params = {
        "file_list": list(file_list),
        "batch_size": batch_size,
        "sequence_length": num_frames,
        "decord_threads": decord_threads,
        "input_size": input_size,
        "short_side_size": 256,
        "mean": mean,
        "std": std,
    }
    
    pipe = dali_pipeline(
        batch_size=batch_size,
        num_threads=dali_num_threads,
        device_id=local_rank,
        seed=42 + rank,
        py_num_workers=dali_py_workers,
        py_start_method="spawn",
        prefetch_queue_depth=1,
        source_params=source_params,
    )
    pipe.build()
    
    dali_iter = DALIGenericIterator(
        pipelines=pipe,
        output_map=["videos", "labels"],
        auto_reset=False,
        size=-1,
        last_batch_padded=False,
        last_batch_policy=LastBatchPolicy.PARTIAL,
        prepare_first_batch=False,
    )
    
    return DALIWarper(dali_iter=dali_iter, length=-(-len(file_list) // batch_size))


def load_metaclip_h14(model_path: str):
    """Load MetaCLIP-H/14 model."""
    print(f"[Rank {rank}] Loading MetaCLIP-H/14 from {model_path}")
    model = open_clip.create_model("ViT-H-14-quickgelu")
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    return model.cuda().eval()


class CheckpointManager:
    """Manage checkpoint for resume."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.checkpoint_file = output_dir / f"checkpoint_rank{rank:04d}.pkl"
    
    def load(self):
        """Load checkpoint: returns (processed_videos, chunk_id)."""
        if not self.checkpoint_file.exists():
            return 0, 0
        
        with open(self.checkpoint_file, 'rb') as f:
            state = pickle.load(f)
        
        processed_videos = state.get('processed_videos', 0)
        chunk_id = state.get('chunk_id', 0)
        print(f"[Rank {rank}] Resume: {processed_videos} videos processed, chunk {chunk_id}")
        return processed_videos, chunk_id
    
    def save(self, processed_videos: int, chunk_id: int):
        """Save checkpoint."""
        with open(self.checkpoint_file, 'wb') as f:
            pickle.dump({'processed_videos': processed_videos, 'chunk_id': chunk_id}, f)
    
    def remove(self):
        """Remove checkpoint after completion."""
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()


@torch.no_grad()
def extract_features(
    data_iter,
    model,
    output_dir: Path,
    chunk_size: int,
    num_frames: int,
):
    """Extract features with checkpoint resume."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load checkpoint
    ckpt = CheckpointManager(output_dir)
    processed_videos, start_chunk_id = ckpt.load()
    
    # Calculate starting position in current chunk
    videos_per_chunk = chunk_size
    idx_in_chunk = processed_videos % videos_per_chunk
    chunk_id = start_chunk_id
    
    if processed_videos > 0:
        print(f"[Rank {rank}] Resuming: processed={processed_videos}, "
              f"chunk={chunk_id}, idx_in_chunk={idx_in_chunk}")
    
    total_processed = processed_videos
    gpu_cache = None
    
    for batch_idx, (videos, labels) in enumerate(data_iter):
        # videos: (B, C, T, H, W) -> (B*T, C, H, W)
        B, C, T, H, W = videos.shape
        frames = videos.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        
        # Extract features
        with torch.cuda.amp.autocast(True):
            feat = model.encode_image(frames)  # (B*T, D)
        
        feat = feat.float().reshape(B, T, -1)  # (B, T, D)
        
        # Initialize cache
        if gpu_cache is None:
            D = feat.shape[-1]
            gpu_cache = torch.zeros((chunk_size, T, D), dtype=torch.float32, device="cuda")
            print(f"[Rank {rank}] Cache shape: ({chunk_size}, {T}, {D})")
            
            # If resuming mid-chunk, load existing chunk data
            if idx_in_chunk > 0:
                existing_file = output_dir / f"features_rank{rank:04d}_chunk{chunk_id:04d}.npy"
                if existing_file.exists():
                    print(f"[Rank {rank}] Loading existing chunk for resume: {existing_file}")
                    existing_data = np.load(existing_file)
                    gpu_cache[:len(existing_data)] = torch.from_numpy(existing_data).cuda()
        
        # Add to cache
        current_bs = feat.size(0)
        gpu_cache[idx_in_chunk : idx_in_chunk + current_bs] = feat
        idx_in_chunk += current_bs
        total_processed += current_bs
        
        # Save chunk when full
        if idx_in_chunk >= chunk_size:
            output_file = output_dir / f"features_rank{rank:04d}_chunk{chunk_id:04d}.npy"
            np.save(output_file, gpu_cache.cpu().numpy())
            print(f"[Rank {rank}] Saved chunk {chunk_id}: {output_file}")
            
            chunk_id += 1
            idx_in_chunk = 0
            gpu_cache.fill_(0.)
            
            # Save checkpoint every chunk
            ckpt.save(total_processed, chunk_id)
        
        # Progress (every 10 batches)
        if (batch_idx + 1) % 100 == 0:
            print(f"[Rank {rank}] Processed {total_processed} videos, batch {batch_idx + 1}")
    
    # Save last partial chunk
    if idx_in_chunk > 0:
        output_file = output_dir / f"features_rank{rank:04d}_chunk{chunk_id:04d}.npy"
        np.save(output_file, gpu_cache[:idx_in_chunk].cpu().numpy())
        print(f"[Rank {rank}] Saved final chunk {chunk_id}: {output_file} (size: {idx_in_chunk})")
    
    ckpt.remove()
    print(f"[Rank {rank}] Done! Total processed: {total_processed}")


def main(args):
    if rank == 0:
        print("=" * 60)
        print("MetaCLIP-H/14 Video Feature Extraction")
        print("=" * 60)
        print(f"Input:      {args.input}")
        print(f"Output:     {args.output}")
        print(f"Batch size: {args.batch_size}")
        print(f"Num frames: {args.num_frames}")
        print(f"World size: {world_size}")
        print(f"Rank:       {rank}")
        print(f"Local rank: {local_rank}")
        print("=" * 60)
    
    # Load full video list
    with open(args.input, 'r') as f:
        all_videos = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
    if rank == 0:
        print(f"Total videos: {len(all_videos)}")
    
    # Continuous sharding for easy merging with padding
    if world_size > 1:
        # Pad all_videos to be divisible by world_size * batch_size
        remainder = len(all_videos) % (world_size * args.batch_size)
        if remainder > 0:
            padding = [all_videos[-1]] * (world_size * args.batch_size - remainder)
            all_videos = all_videos + padding
        
        videos_per_rank = len(all_videos) // world_size
        start_idx = rank * videos_per_rank
        end_idx = start_idx + videos_per_rank
        
        my_videos = all_videos[start_idx:end_idx]
        print(f"[Rank {rank}] Processing videos [{start_idx}:{end_idx}] ({len(my_videos)} videos)")
    else:
        # Pad to ensure args.batch_size divisibility for single process case
        remainder = len(all_videos) % args.batch_size
        if remainder > 0:
            padding = [all_videos[-1]] * (args.batch_size - remainder)
            all_videos = all_videos + padding
        
        my_videos = all_videos
        print(f"[Rank {rank}] Processing all {len(my_videos)} videos")
    
    with open(f"{args.input}_padding.txt", 'w') as f:
        f.writelines(all_videos)
    # Load checkpoint
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt = CheckpointManager(output_dir)
    processed_videos, _ = ckpt.load()
    
    # Skip already processed videos
    if processed_videos > 0:
        my_videos = my_videos[processed_videos:]
        print(f"[Rank {rank}] Skipping {processed_videos} already processed videos")
        print(f"[Rank {rank}] Remaining: {len(my_videos)} videos")
    else:
        print(f"[Rank {rank}] Processing all {len(my_videos)} videos")
    
    if len(my_videos) == 0:
        print(f"[Rank {rank}] No videos to process!")
        return
    
    # Create dataloader
    loader = dali_dataloader(
        file_list=my_videos,
        batch_size=args.batch_size,
        num_frames=args.num_frames,
        input_size=224,
        dali_num_threads=args.dali_num_threads,
        dali_py_workers=args.dali_py_workers,
        decord_threads=args.decord_threads,
    )
    
    # Load model
    model = load_metaclip_h14(args.model_path)

    args.chunk_size = args.chunk_size // args.batch_size * args.batch_size  # Ensure divisibility
    # Extract features
    extract_features(
        data_iter=loader,
        model=model,
        output_dir=output_dir,
        chunk_size=args.chunk_size,
        num_frames=args.num_frames,
    )
    
    if rank == 0:
        print("\n" + "=" * 60)
        print("Extraction completed!")
        print("=" * 60)
    
    if rank == world_size - 1:
        # 计算需要删除的 padding 数量（使用之前的 remainder）
        divisor = world_size * args.batch_size if world_size > 1 else args.batch_size
        pad_count = (divisor - remainder) % divisor  # 正确的 padding 数量

        # 找到最后一个 .npy 文件
        npy_files = sorted(list(output_dir.glob("*.npy")))
        if npy_files and pad_count > 0:
            last_npy_path = npy_files[-1]
            print(f"Processing last file to remove padding: {last_npy_path}")
            
            features = np.load(str(last_npy_path))
            features_per_file = features.shape[0]

            if pad_count < features_per_file:
                cleaned_features = features[:-pad_count]
                np.save(str(last_npy_path), cleaned_features)
                print(f"Removed {pad_count} padding features from the last file")
                print(f"Final feature shape: {cleaned_features.shape}")
            elif pad_count == features_per_file:
                # 整个最后一个文件都是 padding，直接删掉文件更合理
                import os
                os.remove(str(last_npy_path))
                print(f"Removed entire last file since it was all padding ({pad_count} rows).")
            else:
                # 需要继续往前删前一个文件，当前代码未实现，给出提示
                print(f"Padding count ({pad_count}) exceeds rows in the last file ({features_per_file}). "
                      f"Please also trim previous file(s).")

        print("Feature extraction and cleanup completed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract MetaCLIP-H/14 features from videos")
    parser.add_argument("--input", required=True, help="Input video list")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--model_path", default="/vlm/xiangan/pretrain_models/metaclip/metaclip_h_14.pt")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--chunk_size", type=int, default=32 << 10)  # Save every 1024 videos
    parser.add_argument("--dali_num_threads", type=int, default=2)
    parser.add_argument("--dali_py_workers", type=int, default=8)
    parser.add_argument("--decord_threads", type=int, default=1)
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank (set by DeepSpeed)")
    
    args = parser.parse_args()
    main(args)
