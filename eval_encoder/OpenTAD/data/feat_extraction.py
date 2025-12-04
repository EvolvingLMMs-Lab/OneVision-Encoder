import argparse
import math
import os
import warnings
from pathlib import Path
from typing import List, Tuple,  Dict
import time  # <--- 引入 time 模块
import numpy as np
import torch
import torch.nn.functional as F
import torchmetrics
from dataloader.ap_dataloader_dali import get_dali_dataloader
from timm.loss import LabelSmoothingCrossEntropy
from timm.models import create_model
from timm.models.layers import trunc_normal_
from torch import distributed, nn
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import LinearLR
import sys

# Ensure custom models and layers are registered
import model_factory
warnings.filterwarnings("ignore")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Feature extraction from videos with clip-based processing")
    # Data
    parser.add_argument("--data_root", default="/video_vit/feilong/TAD-Dataset/charades", help="Root directory of video data")
    parser.add_argument("--data_csv_path", default="charades_videos.csv", help="CSV file with video paths and labels")
    parser.add_argument("--output_dir", default="/video_vit/feilong/TAD-Dataset/charades/features/llava_vit_base_cls_s4f16/", help="Output directory for .npy feature files")

    # Model
    parser.add_argument("--model_family", default="llava_vit_sampling")
    parser.add_argument("--model_name", default="llava_vit_base_ln")
    parser.add_argument("--model_weight", default="/video_vit/xiangan/checkpoint_llava_vit/continue_with_mlcd_1536_tokens_b16_mix_three_input_residual_mv_new_b16/00056000/backbone.pt")
    parser.add_argument("--num_frames", type=int, default=16,
                        help="Number of frames per chunk for model input (model processes this many frames at a time)")
    parser.add_argument("--sequence_length", type=int, default=None,
                        help="Total number of frames to load from each video. If None, uses num_frames. Set to 512 for long video processing.")
    parser.add_argument("--num_tokens", type=int, default=1568)
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--tubelet_size", type=int, default=1)
    parser.add_argument("--embedding_size", type=int, default=768)
    parser.add_argument("--target_frames", type=int, default=64,
                        help="Target number of frames to interpolate to (default: 64)")

    # Dataloader
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for DALI dataloader (use 1 for per-video processing)")
    parser.add_argument("--dali_num_threads", type=int, default=2)
    parser.add_argument("--dali_py_num_workers", type=int, default=8)
    parser.add_argument("--decord_num_threads", type=int, default=2,
                        help="Number of threads for decord video reader.")
    parser.add_argument("--short_side_size", type=int, default=256)
    parser.add_argument("--frames_token_num", type=int, default=196)

    parser.add_argument("--mean", nargs=3, type=float, default=[0.48145466, 0.4578275, 0.40821073])
    parser.add_argument("--std", nargs=3, type=float, default=[0.26862954, 0.26130258, 0.27577711])

    # Clip processing

    parser.add_argument("--clip_stride", type=int, default=4,
                        help="Stride between clips when splitting long videos. If None, uses num_frames (non-overlapping clips). "
                             "Set to 4 for 8-frame sliding window with stride=4 (50%% overlap)")

    # Misc
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--pooling_method", type=str, default="mean", 
                        choices=["mean", "cls", "all"],
                        help="Feature pooling method: 'mean' for mean pooling over tokens, 'cls' for first token only, 'all' to save all tokens")

    # 分布式相关参数
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--global_rank", type=int, default=0)

    return parser.parse_args()


def interpolate_frame_indices(frame_indices: torch.Tensor, total_frames: torch.Tensor, target_frames: int = 64) -> torch.Tensor:
    """
    将帧索引从原始视频帧数插值到目标帧数

    Args:
        frame_indices: [B, seq_len] 原始帧索引
        total_frames: [B] 每个视频的总帧数
        target_frames: 目标帧数 (默认 64)

    Returns:
        interpolated_indices: [B, seq_len] 插值后的帧索引，范围在 [0, target_frames-1]
    """
    bs, seq_len = frame_indices.shape
    device = frame_indices.device

    # 将 total_frames 转换为浮点数以进行插值计算
    total_frames_float = total_frames.float().view(bs, 1)  # [B, 1]
    frame_indices_float = frame_indices.float()  # [B, seq_len]

    # 插值公式: new_idx = (old_idx / (total_frames - 1)) * (target_frames - 1)
    # 处理 total_frames = 1 的情况
    total_frames_safe = torch.clamp(total_frames_float - 1, min=1.0)
    interpolated_indices = (frame_indices_float / total_frames_safe) * (target_frames - 1)

    # 四舍五入并转换为整数
    interpolated_indices = torch.round(interpolated_indices).long()

    # 确保索引在有效范围内
    interpolated_indices = torch.clamp(interpolated_indices, 0, target_frames - 1)

    return interpolated_indices


def get_feature(
    args: argparse.Namespace,
    videos: torch.Tensor,
    model: nn.Module,
    frame_indices: torch.Tensor = None,
    total_frames: torch.Tensor = None,
) -> torch.Tensor:
    """
    获取特征，支持视频及图片输入。

    Args:
        args: 参数配置
        videos: 视频数据 [B, C, T, H, W] 或图片数据 [B, C, H, W]
        model: 模型
        frame_indices: 视频帧索引 [B, seq_len]，用于 llava_vit_sampling
        total_frames: 每个视频的总帧数 [B]
    """
    def video_to_images(videos: torch.Tensor) -> torch.Tensor:
        """
        将视频 [B, C, T, H, W] 展开为图片序列 [B*T, C, H, W]
        """
        B, C, T, H, W = videos.shape
        images = videos.permute(0, 2, 1, 3, 4).reshape(-1, C, H, W)  # [B*T, C, H, W]
        return images

    list_vit_single_image = [
        "clip",
        "siglip",
        "siglip2",
        "dinov2",
        "dinov3",
        "metaclip",
        "llava_vit_si",
        "aimv2"
    ]
    if args.model_family in list_vit_single_image:
        # ===> 专门图片分支 <===
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            with torch.no_grad():
                # 如果是视频输入，将其转化为图片
                B, C, T, H, W = videos.shape
                if videos.dim() == 5:  # 视频分支 [B, C, T, H, W]
                    videos = video_to_images(videos)

                if videos.dim() == 4:  # 检测为图片分支 [B, C, H, W]
                    hidden_states = model(videos)
                    if isinstance(hidden_states, dict) and "visible_embeddings" in hidden_states:
                        hidden_states = hidden_states["visible_embeddings"]

                    # hidden_states = hidden_states.view(B, -1, hidden_states.size(-1))  # [B, seq_len, hidden_size]
                    hidden_states = hidden_states.reshape(B, -1, hidden_states.size(-1))  # [B, seq_len, hidden_size]
                    # ===> 新增：sin/cos 时间位置编码（2行代码）<===
                    pos = torch.arange(T, device=videos.device).unsqueeze(1) * torch.exp(torch.arange(0, args.embedding_size, 2, device=videos.device) * (-math.log(10000.0) / args.embedding_size))  # [T, D/2]
                    temporal_pos = torch.stack([torch.sin(pos), torch.cos(pos)], dim=2).flatten(1)[:, :args.embedding_size]  # [T, D]
                    hidden_states = hidden_states.view(B, T, -1, args.embedding_size) + temporal_pos.unsqueeze(0).unsqueeze(2)  # 加到每帧的 tokens 上
                    hidden_states = hidden_states.view(B, -1, args.embedding_size)  # [B, T*tokens_per_frame, D]
                    return hidden_states
                else:
                    raise ValueError("SigLIP2 only supports image input with 4 dimensions [B, C, H, W].")

    elif args.model_family == "llava_vit_sampling":
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            with torch.no_grad():
                bs, C, T, H, W = videos.shape
                device = videos.device
                frame_tokens = args.frames_token_num  # 每帧的 token 数量
                target_frames = args.target_frames  # 目标帧数，默认 64

                if frame_indices is not None and total_frames is not None:
                    # ===> 插值帧索引到 target_frames <===
                    interpolated_indices = interpolate_frame_indices(
                        frame_indices,
                        total_frames.view(-1),
                        target_frames
                    )  # [B, seq_len]
                    # ===> 创建 target_frames 帧的空白视频 <===
                    padded_videos = torch.zeros(bs, C, target_frames, H, W, device=device, dtype=videos.dtype)

                    # ===> 将原始帧放入插值后的对应位置 <===
                    seq_len = frame_indices.shape[1]

                    # 准备 scatter 的索引
                    frame_idx_expanded = interpolated_indices.view(bs, 1, seq_len, 1, 1).expand(bs, C, seq_len, H, W)

                    # 将视频帧放入对应位置
                    padded_videos.scatter_(dim=2, index=frame_idx_expanded, src=videos)

                    # ===> 计算 visible_index (基于 target_frames) <===
                    per = torch.arange(frame_tokens, device=device)
                    visible_index = (interpolated_indices.unsqueeze(-1) * frame_tokens + per).reshape(bs, -1)
                    visible_index = visible_index.clamp_max(target_frames * frame_tokens - 1)

                    enc_out = model(padded_videos, visible_index)
                    outputs = enc_out["head_output"]
                    # outputs = enc_out["visible_embeddings"]

                else:
                    raise

                return outputs

    raise ValueError(f"Unsupported model_family: {args.model_family}")


def get_model(args: argparse.Namespace) -> nn.Module:
    model = create_model(args.model_name, pretrained=False)
    if args.model_family in ["llava_vit_sampling"]:
        state_dict = torch.load(args.model_weight, map_location="cpu")
        state_dict = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
    return model



def split_video_into_clips(total_frames: int, clip_length: int, stride: int = None) -> List[List[int]]:
    """Split video frame indices into clips.
    
    Args:
        total_frames: Total number of frames in the video
        clip_length: Number of frames per clip
        stride: Stride between clips. If None, uses clip_length (non-overlapping)
    
    Returns:
        List of frame index lists for each clip
    """
    if stride is None:
        stride = clip_length
    
    if total_frames == 0:
        return []
    
    clips = []
    start = 0
    while start < total_frames:
        end = min(start + clip_length, total_frames)
        # Get frame indices for this clip
        indices = list(range(start, end))
        # Pad if necessary
        if len(indices) < clip_length and len(indices) > 0:
            indices += [indices[-1]] * (clip_length - len(indices))
        elif len(indices) == 0:
            # Edge case: if somehow we have no indices, use the last frame
            indices = [total_frames - 1] * clip_length
        clips.append(indices)
        start += stride
        
        # Stop if we've covered all frames
        if end >= total_frames:
            break
    
    return clips


def extract_features_from_dali(
    args: argparse.Namespace,
    model: nn.Module,
    device: torch.device,
    dataloader,
) -> None:
    """Extract features from videos using DALI dataloader"""
    model.to(device).eval()
    
    # Batch size for processing chunks (larger = better GPU utilization but more memory)
    CHUNK_BATCH_SIZE = 32
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.rank == 0:
        print(f"Starting feature extraction with DALI dataloader...")
        print(f"Output directory: {output_dir}")
        print(f"Processing mode: DALI standard frame sampling")
        if args.clip_stride is not None:
            print(f"Clip stride: {args.clip_stride} (sliding window with stride {args.clip_stride})")
        else:
            print(f"Clip stride: None (non-overlapping chunks)")
    
    # We need to track video names for per-video output
    # Since DALI doesn't provide video paths directly, we'll use a counter and load the CSV again
    csv_path = os.path.join(args.data_root, args.data_csv_path) if not os.path.isabs(args.data_csv_path) else args.data_csv_path
    video_names = []
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                offset_path = parts[0]
                video_name = Path(offset_path).stem
                video_names.append(video_name)
    except FileNotFoundError:
        raise FileNotFoundError(f"Data list file not found at: {csv_path}")
    
    # Shard video names to match DALI's sharding
    if args.world_size > 1:
        shard_size = len(video_names) // args.world_size
        start_idx = args.rank * shard_size
        if args.rank == args.world_size - 1:
            end_idx = len(video_names)
        else:
            end_idx = start_idx + shard_size
        video_names = video_names[start_idx:end_idx]
    
    total_processed = 0
    batch_count = 0
    
    for batch in dataloader:
        videos = batch["videos"].to(device, non_blocking=True)
        indices = batch["indices"].to(device, non_blocking=True)
        total_frames = batch["total_frames"].to(device, non_blocking=True)
        batch_size = videos.size(0)
        
        # Extract features for each video in the batch
        for i in range(batch_size):
            # Get video name first to check if feature already exists
            video_idx = batch_count * args.batch_size + i
            if video_idx < len(video_names):
                video_name = video_names[video_idx]
            else:
                # Use deterministic naming based on video index
                video_name = f"video_{video_idx}"
            
            # Check if feature file already exists
            feature_file = output_dir / f"{video_name}.npy"
            # if feature_file.exists():
            #     print(f"Rank {args.rank}: Skipping {video_name}: feature file already exists at {feature_file}")
            #     total_processed += 1
            #     continue
            
            video = videos[i:i+1]  # Keep batch dimension [1, C, T, H, W]
            video_indices = indices[i:i+1]
            video_total_frames = total_frames[i:i+1]-1
            
            # Get video dimensions
            _, C, T, H, W = video.shape
            
            # Model expects args.num_frames (default 8) frames at a time, so split T frames into chunks
            chunk_size = args.num_frames
            
            # Use split_video_into_clips to get clip frame indices with stride support
            clip_stride = args.clip_stride if args.clip_stride is not None else chunk_size
            clip_frame_indices = split_video_into_clips(T, chunk_size, stride=clip_stride)
            num_chunks = len(clip_frame_indices)
            
            # List to collect features from all chunks
            chunk_features = []
            
            for batch_start in range(0, num_chunks, CHUNK_BATCH_SIZE):
                batch_end = min(batch_start + CHUNK_BATCH_SIZE, num_chunks)
                batch_num_chunks = batch_end - batch_start
                
                # Collect chunks for this batch
                batch_video_chunks = []
                batch_chunk_indices = []
                
                for chunk_idx in range(batch_start, batch_end):
                    # Get frame indices for this clip
                    frame_list = clip_frame_indices[chunk_idx]
                    
                    # Extract frames based on the frame list [1, C, chunk_size, H, W]
                    video_chunk = video[:, :, frame_list, :, :]
                    
                    # Extract corresponding indices for this chunk
                    chunk_indices = video_indices[:, frame_list]
                    
                    batch_video_chunks.append(video_chunk)
                    batch_chunk_indices.append(chunk_indices)
                
                # Stack chunks into a single batch [batch_num_chunks, C, chunk_size, H, W]
                batched_video_chunks = torch.cat(batch_video_chunks, dim=0)
                batched_chunk_indices = torch.cat(batch_chunk_indices, dim=0)
                
                # Replicate total_frames for batch processing
                # print(video_total_frames, batch_num_chunks)
                batched_total_frames = video_total_frames.flatten().repeat(batch_num_chunks)
                
                # Extract features for this batch of chunks
                batch_chunk_feat = get_feature(
                    args, 
                    batched_video_chunks, 
                    model, 
                    frame_indices=batched_chunk_indices, 
                    total_frames=batched_total_frames
                )
                
                # # Apply pooling method per chunk
                # if batch_chunk_feat.dim() == 3:
                #     if args.pooling_method == "mean":
                #         batch_chunk_feat = batch_chunk_feat.mean(dim=1)  # [batch_num_chunks, D]
                #     elif args.pooling_method == "cls":
                #         batch_chunk_feat = batch_chunk_feat[:, 0, :]  # [batch_num_chunks, D]
                #     elif args.pooling_method == "all":
                #         pass  # Keep [batch_num_chunks, seq_len, D]
                
                # Split batch results back into individual chunks
                for j in range(batch_num_chunks):
                    chunk_features.append(batch_chunk_feat[j:j+1])
            
            # Stack all chunk features: [num_chunks, D] or [num_chunks, seq_len, D]
            if chunk_features[0].dim() == 2:
                # Pooled features: [num_chunks, D]
                feats = torch.cat(chunk_features, dim=0)  # [num_chunks, D]
            else:
                # All token features: [num_chunks, seq_len, D]
                feats = torch.cat(chunk_features, dim=0)  # [num_chunks, seq_len, D]
            
            # if "001YG" in str(feature_file):
            #     print(f"[R{args.rank}] ⚠ 检测到文件名包含 '001YG'，停止处理后续视频！文件是：{feature_file}")
            #     print(feature_file, feats.shape, total_frames)
            #     print(f"   features.dtype = {feats.dtype}")
            #     sys.exit(1)
      
            # Convert to numpy
            feats = feats.float().cpu().numpy()
            # Save features with shape [num_chunks, D] or [num_chunks, seq_len, D]
            np.save(feature_file, feats)
            total_processed += 1
        
        batch_count += 1
        
        if args.rank == 0 and batch_count % 10 == 0:
            print(f"Processed {total_processed} videos, batch {batch_count}")
    
    if args.rank == 0:
        print(f"Feature extraction completed! Processed {total_processed} videos")


def main() -> None:
    args = parse_args()

    try:
        args.rank = int(os.environ["RANK"])
        args.local_rank = int(os.environ["LOCAL_RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        distributed.init_process_group("nccl")
    except KeyError:
        args.rank = 0
        args.local_rank = 0
        args.world_size = 1
        distributed.init_process_group(backend="nccl", init_method="tcp://127.0.0.1:12584", rank=args.rank, world_size=args.world_size)
    torch.cuda.set_device(args.local_rank)
    device = torch.device(args.local_rank)
    args.global_rank = args.rank

    if args.rank == 0:
        print("Create data loaders...")

    # Set normalization based on model family
    if args.model_family in ["siglip", "siglip2"]:
        args.mean = [0.5, 0.5, 0.5]
        args.std = [0.5, 0.5, 0.5]
    if args.model_family in ["dinov2", "dinov3"]:
        args.mean = [0.485, 0.456, 0.406]
        args.std = [0.229, 0.224, 0.225]

    # Load model
    model = get_model(args)

    # Set sequence_length default if not provided
    if args.sequence_length is None:
        args.sequence_length = args.num_frames
    
    if args.rank == 0:
        print("Using DALI dataloader for standard frame sampling")
        print(f"Loading {args.sequence_length} frames per video, processing in chunks of {args.num_frames}")
    
    # Create DALI dataloader
    dataloader = get_dali_dataloader(
        data_root_path=args.data_root,
        data_csv_path=os.path.join(args.data_root, args.data_csv_path) if not os.path.isabs(args.data_csv_path) else args.data_csv_path,
        mode="val",
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        input_size=args.input_size,
        short_side_size=args.short_side_size,
        mean=args.mean,
        std=args.std,
        dali_num_threads=args.dali_num_threads,
        dali_py_num_workers=args.dali_py_num_workers,
        decord_num_threads=args.decord_num_threads,
        seed=args.seed,
        feature_extract = True,
    )
    
    try:
        # Extract features using DALI dataloader
        extract_features_from_dali(args, model, device, dataloader)

        if args.rank == 0:
            print("\n" + "=" * 60)
            print("Extraction completed!")
            print("=" * 60)
    finally:
        # Cleanup DALI resources
        try:
            if hasattr(dataloader, 'iter') and hasattr(dataloader.iter, '_pipes'):
                for pipe in dataloader.iter._pipes:
                    pipe.reset()
        except Exception as e:
            if args.rank == 0:
                print(f"Warning: Error during DALI cleanup: {e}")
        
        # Cleanup distributed process group
        try:
            if distributed.is_initialized():
                distributed.barrier()
                distributed.destroy_process_group()
        except Exception as e:
            if args.rank == 0:
                print(f"Warning: Error during distributed cleanup: {e}")


if __name__ == "__main__":
    main()
