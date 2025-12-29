import argparse
import logging
import os
import sys
import time
from typing import Any, Dict, List

import numpy as np
import torch
from timm import create_model
from torch import distributed
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from transformers import CLIPImageProcessor

import dataset
import model_factory
from dataset import DATASET_REGISTRY, Property
from onevision_encoder import OneVisionEncoderConfig, OneVisionEncoderModel
from training.checkpoint_utils import load_checkpoint, save_checkpoint
from training.fused_partial_fc_v2_multi_res import (CombinedMarginLoss,
                                                    PartialFC_V2)
from training.lr_scheduler import PolynomialLRWarmup

torch._dynamo.config.optimize_ddp = False

parser = argparse.ArgumentParser(description="Multi-dataset video training")

# General
parser.add_argument("--debug", type=int, default=0, help="Enable debug mode (0/1)")
parser.add_argument("--output", default="output", help="Output directory for logs and checkpoints")
parser.add_argument("--workers", type=int, default=2, help="Number of DataLoader workers per process")
parser.add_argument("--local_rank", type=int, default=0, help="Local rank passed by launcher; do not set manually")

# Data loading
parser.add_argument("--dataloader-type", default="dali", help="Data loader backend, e.g., 'dali' or 'torch'")
parser.add_argument("--dali_is_training", type=int, default=1, help="DALI training mode (0/1)")
parser.add_argument("--image_size", default="224", help="Input size as 'H,W' or single 'S' (S,S)")
parser.add_argument("--image_size_video", default="224", help="Video input size as 'H,W' or single 'S' (S,S)")
parser.add_argument("--input_gray", type=int, default=0, help="Treat input as grayscale (0/1)")
parser.add_argument("--num_frames", type=int, default=8, help="Number of frames per clip")
parser.add_argument("--random_diff", type=int, default=10, help="Random diff for sampling jitter")

# Multi-dataset (heads)
parser.add_argument("--list_datasets", nargs='+', type=str, default=["k710_ssv2_univit_pfs"],
                    help="Dataset registry names, one or more")
parser.add_argument("--list_batch_sizes", nargs='+', type=int, default=[32],
                    help="Per-dataset batch sizes")
parser.add_argument("--list_sample_rates", nargs='+', type=float, default=[0.1],
                    help="Per-dataset sampling rate")
parser.add_argument("--list_margins", nargs='+', type=float, default=[0.3],
                    help="Per-dataset loss margin")
parser.add_argument("--list_filters", nargs='+', type=float, default=[0.75],
                    help="Per-dataset filter ratio or threshold")
parser.add_argument("--list_lr_pfc_weights", nargs='+', type=float, default=[1.0],
                    help="Per-dataset LR scale for PFC params")
parser.add_argument("--list_loss_weights", nargs='+', type=float, default=[1.0],
                    help="Per-dataset loss weights")
parser.add_argument("--list_init_partial_fc_paths", nargs='+', type=str, default=["NULL"],
                    help="Per-dataset init path for partial-FC or 'NULL'")

# Model
parser.add_argument("--model_name", default="pretrain_encoder_small_patch16_224_v10_12_rms_unmask_with_head", help="Backbone model name")
parser.add_argument("--model_weight", default=None,
                    help="Path to pretrained weights, HuggingFace model ID, or None")
parser.add_argument("--embedding_size", type=int, default=384, help="Embedding dimension of the head")
parser.add_argument("--gradient_checkpoint", type=int, default=0, help="Enable gradient checkpointing (0/1)")
parser.add_argument("--mask", type=int, default=0, help="Enable mask-related training (0/1)")
parser.add_argument("--finetune_backbone", type=int, default=1, help="Finetune backbone parameters (0/1)")

# Optimization
parser.add_argument("--opt", default="adamw", help="Optimizer name, e.g., 'adamw'")
parser.add_argument("--lr", type=float, default=1e-3, help="Base learning rate")
parser.add_argument("--weight_decay", type=float, default=0.05, help="Weight decay for non-PFC params")
parser.add_argument("--weight_decay_pfc", type=float, default=0.05, help="Weight decay for PFC params")
parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio of total training steps")
parser.add_argument("--backward_passes_per_step", type=int, default=1, help="Gradient accumulation steps")
parser.add_argument("--repeat_pfc", type=int, default=0, help="Repeat factor for PFC ops or rebuild cycles")
parser.add_argument("--save_pfc", type=int, default=1, help="Save PFC weights in checkpoints (0/1)")

# Initialization / Resume
parser.add_argument("--init_backbone", default="NULL", help="Backbone init path or 'NULL'")

# Logging & Checkpoint
parser.add_argument("--frequent", type=int, default=10, help="Log/validation frequency in steps")
parser.add_argument("--ckpt_interval", type=int, default=2000, help="Checkpoint save interval in steps")

# Training schedule
parser.add_argument("--num_sampled_data", type=int, default=60000000, help="Total sampled examples for step calculation")

# Visualization
parser.add_argument("--visualize", type=int, default=0, help="Save input clips as GIFs (0/1)")
parser.add_argument("--vis_samples", type=int, default=2, help="Number of samples to visualize per batch")
parser.add_argument("--vis_interval", type=int, default=10, help="Visualization save interval in steps")

# Index sampling for ViT input
parser.add_argument("--total_indices", type=int, default=2048, help="Visible indices total count")
parser.add_argument("--target_num", type=int, default=2048, help="Sampled indices count")
parser.add_argument("--must_num", type=int, default=256, help="Number of indices that must be included (from front)")
parser.add_argument("--num_tokens_per_frame", type=int, default=256, help="Number of tokens per frame")

# Multi-frame training (batch size inversely proportional to frame count)
parser.add_argument("--enable_multi_frame", type=int, default=1,
                    help="Enable multi-frame training (0/1)")
parser.add_argument("--multi_frame_list", nargs='+', type=int, default=[8],
                    help="List of frame counts to use in multi-frame training")
parser.add_argument("--base_num_frames", type=int, default=8,
                    help="Base frame count for batch size calculation")

args = parser.parse_args()

rank = int(os.getenv("RANK", "0"))
local_rank = int(os.getenv("LOCAL_RANK", "0"))
world_size = int(os.getenv("WORLD_SIZE", "1"))
distributed.init_process_group(backend="nccl")

torch.cuda.set_device(local_rank)
torch.backends.cudnn.benchmark = True

os.makedirs(args.output, exist_ok=True)

if rank == 0:
    logger: logging.Logger = logging.getLogger(__name__)
    formatter = logging.Formatter(f"rank-id:{rank:03d}:%(asctime)s-%(message)s")
    file_handler = logging.FileHandler(os.path.join(args.output, f"training_{rank:03d}.logger"))
    stream_handler = logging.StreamHandler(sys.stdout)
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)
else:
    logger: logging.Logger = logging.getLogger(__name__)
    formatter = logging.Formatter(f"rank-id:{rank:03d}:%(asctime)s-%(message)s")
    file_handler = logging.FileHandler(os.path.join(args.output, f"training_{rank:03d}.logger"))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)

def unwrap_module(model):
    """Unwraps a model from DistributedDataParallel or torch.compile if it is wrapped."""
    if hasattr(model, "module"):
        return model.module
    if hasattr(model, "_orig_mod"):
        return model._orig_mod
    return model


# CLIP Specific Constants for image processor
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


def is_hf_model_dir(path):
    """Check if a path is a HuggingFace model directory (contains config.json)."""
    if not os.path.isdir(path):
        return False
    return os.path.exists(os.path.join(path, "config.json"))


def save_hf_checkpoint(output_dir, backbone, global_step, image_size=448):
    """
    Save model in HuggingFace transformers format using save_pretrained().

    Args:
        output_dir: Base output directory
        backbone: The backbone model (may be wrapped in DDP or torch.compile)
        global_step: Current training step
        image_size: Image size for the processor config
    """
    # Only save on rank 0
    if rank != 0:
        return

    # Create HuggingFace checkpoint directory
    hf_dir = os.path.join(output_dir, f"{global_step:08d}_hf")
    os.makedirs(hf_dir, exist_ok=True)

    # Unwrap the model from DDP and torch.compile
    model = unwrap_module(backbone)

    # Save using HuggingFace save_pretrained
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(hf_dir)
        logger.info(f"Saved HuggingFace model to {hf_dir}")

        # Save CLIPImageProcessor config
        processor = CLIPImageProcessor(
            size=image_size,
            crop_size=image_size,
            image_mean=CLIP_MEAN,
            image_std=CLIP_STD,
            resample=3,
            do_center_crop=True,
            do_normalize=True,
            do_resize=True,
            feature_extractor_type="CLIPFeatureExtractor"
        )
        processor.save_pretrained(hf_dir)
        logger.info(f"Saved CLIPImageProcessor to {hf_dir}")
    else:
        logger.warning(f"Model does not have save_pretrained method, skipping HF checkpoint save")


def main():
    """Main training function."""
    global_step = 0

    # image_size 保持你原逻辑
    args.image_size = [int(x) for x in args.image_size.split(",")]
    if len(args.image_size) == 1:
        args.image_size = args.image_size * 2
    args.image_size_video = [int(x) for x in args.image_size_video.split(",")]
    if len(args.image_size_video) == 1:
        args.image_size_video = args.image_size_video * 2

    if args.enable_multi_frame:
        num_frame_options = len(args.multi_frame_list)
        frame_index = rank % num_frame_options
        args.actual_num_frames = args.multi_frame_list[frame_index]
        if args.base_num_frames % args.actual_num_frames != 0:
            raise ValueError(
                f"base_num_frames ({args.base_num_frames}) must be divisible by "
                f"actual_num_frames ({args.actual_num_frames}). "
                f"Please adjust multi_frame_list or base_num_frames."
            )
        frame_scale_factor = args.base_num_frames // args.actual_num_frames
        logger.info(f"[Multi-frame] rank={rank}, frame_index={frame_index}, "
                    f"actual_num_frames={args.actual_num_frames}, "
                    f"frame_scale_factor={frame_scale_factor}")
    else:
        args.actual_num_frames = args.num_frames
        frame_scale_factor = 1
        logger.info(f"[Single-frame] Using fixed num_frames={args.actual_num_frames}")

    args.list_datasets = [DATASET_REGISTRY.get(x)() for x in args.list_datasets]
    args.num_heads = len(args.list_datasets)

    # 如果 argparse 已经做了类型转换，下面几行可以省略；保留也安全
    args.list_batch_sizes = [int(x) for x in args.list_batch_sizes]
    args.list_sample_rates = [float(x) for x in args.list_sample_rates]
    args.list_margins = [float(x) for x in args.list_margins]
    args.list_filters = [float(x) for x in args.list_filters]
    args.list_lr_pfc_weights = [float(x) for x in args.list_lr_pfc_weights]
    args.list_loss_weights = [float(x) for x in args.list_loss_weights]

    def _expand(name, v):
        if len(v) == 1:
            return v * args.num_heads
        if len(v) != args.num_heads:
            raise ValueError(f"{name}: expected 1 or {args.num_heads} values, got {len(v)}")
        return v

    args.list_batch_sizes = _expand("list_batch_sizes", args.list_batch_sizes)
    args.list_sample_rates = _expand("list_sample_rates", args.list_sample_rates)
    args.list_margins = _expand("list_margins", args.list_margins)
    args.list_filters = _expand("list_filters", args.list_filters)
    args.list_lr_pfc_weights = _expand("list_lr_pfc_weights", args.list_lr_pfc_weights)
    args.list_loss_weights = _expand("list_loss_weights", args.list_loss_weights)
    args.list_init_partial_fc_paths = _expand("list_init_partial_fc_paths", args.list_init_partial_fc_paths)

    args.list_batch_sizes_adjusted = []
    for head_id, dataset_config in enumerate(args.list_datasets):
        base_bs = args.list_batch_sizes[head_id]
        if dataset_config.dali_type == "decord":
            adjusted_bs = base_bs * frame_scale_factor
            logger.info(f"[head_id={head_id}] Video branch: base_bs={base_bs}, "
                        f"adjusted_bs={adjusted_bs} (scale={frame_scale_factor}x)")
        else:
            adjusted_bs = base_bs
            logger.info(f"[head_id={head_id}] Image branch: bs={adjusted_bs}")
        args.list_batch_sizes_adjusted.append(adjusted_bs)

    args.batch_size = sum(args.list_batch_sizes_adjusted)
    args.list_head_names = [x.name for x in args.list_datasets]
    args.total_steps = int(args.num_sampled_data / args.batch_size / world_size)

    for arg in vars(args):
        msg = f"{format(arg, '<30')}  {format(str(getattr(args, arg)))}"
        logger.info(msg)


    # Initialize models using timm's create_model
    backbone = create_model(args.model_name).cuda().train()

    if args.init_backbone != "NULL":
        assert os.path.exists(args.init_backbone)

        # Check if init_backbone is a HuggingFace model directory
        if is_hf_model_dir(args.init_backbone):
            # Load from HuggingFace pretrained directory
            backbone = OneVisionEncoderModel.from_pretrained(
                args.init_backbone,
                torch_dtype=torch.bfloat16
            ).cuda().train()
            logger.info(f"Loaded HuggingFace backbone from {args.init_backbone}")
        else:
            # Load from .pt checkpoint file
            state_dict = torch.load(args.init_backbone, "cpu")
            state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            backbone.load_state_dict(state_dict, strict=True)
            logger.info(f"Loaded backbone weights from {args.init_backbone}")

    if args.finetune_backbone:
        backbone.requires_grad_(True)
    else:
        backbone.requires_grad_(False)
        backbone_module = unwrap_module(backbone)
        if hasattr(backbone_module, "head"):
            for p in backbone_module.head.parameters():
                p.requires_grad = True
        else:
            raise RuntimeError()

    backbone_parameters = filter(lambda p: p.requires_grad, backbone.parameters())

    dict_pfc_modules = {}
    list_module_pfc = []
    parameters: List[dict] = [
        {"params": backbone_parameters},
    ]

    for head_id, _ in enumerate(range(args.num_heads)):
        head_name = args.list_head_names[head_id]
        dataset_config = args.list_datasets[head_id]
        dataset_config: Property

        if dataset_config.pfc_types[0] == "partial_fc":
            margin_loss = CombinedMarginLoss(
                64,
                1,
                0,
                args.list_margins[head_id],
                args.list_filters[head_id]
            )
            partial_fc = PartialFC_V2(
                margin_loss,
                args.embedding_size,
                dataset_config.num_classes,
                args.list_sample_rates[head_id],
                fp16=False,
            )
        else:
            raise ValueError(
                f"dataset_config.pfc_type {dataset_config.pfc_types[0]} not support!"
            )

        partial_fc.train().cuda()
        # list_module_pfc.append(torch.compile(partial_fc))
        list_module_pfc.append(partial_fc)
        dict_pfc_modules[head_name] = partial_fc

        lr_pfc = args.lr * args.list_lr_pfc_weights[head_id]
        parameters.append(
            {
                "params": partial_fc.parameters(),
                "lr": lr_pfc,
                "weight_decay": args.weight_decay_pfc,
            }
        )

        init_partial_fc = args.list_init_partial_fc_paths[head_id]
        if init_partial_fc != "NULL":
            init_partial_fc = init_partial_fc % rank
            logger.info(f"init_partial_fc: {init_partial_fc}")
            if os.path.exists(init_partial_fc):
                if init_partial_fc.endswith(".npy"):
                    _weight = torch.from_numpy(np.load(init_partial_fc)).cuda()
                    partial_fc.weight = torch.nn.Parameter(_weight)
                    logger.info(f"Loaded partial FC weights from {init_partial_fc}")
                elif init_partial_fc.endswith(".pt"):
                    _weight = torch.load(init_partial_fc, "cpu")
                    partial_fc.load_state_dict(_weight, strict=True)
                    logger.info(f"Loaded partial FC state from {init_partial_fc}")
            else:
                raise FileNotFoundError(f"Partial FC init file not found: {init_partial_fc}")

    if args.opt == "adamw":
        optimizer_cls = torch.optim.AdamW

        opt = optimizer_cls(parameters, lr=args.lr, weight_decay=args.weight_decay)
        lr_scheduler = PolynomialLRWarmup(
            opt, int(args.total_steps * args.warmup_ratio), args.total_steps, 2
        )
    else:
        raise ValueError(f"{args.opt} not support!")

    result = load_checkpoint(
        args.output,
        None,
        backbone,
        dict_pfc_modules,
        lr_scheduler,
        None,
        args.list_head_names,
    )
    if result is not None:
        global_step = result['global_step']
        logger.info(f"Resuming from step {global_step}")
    else:
        global_step = 0

    def wrap_ddp(model):
        return torch.nn.parallel.DistributedDataParallel(
            module=model,
            broadcast_buffers=False,
            device_ids=[local_rank],
            bucket_cap_mb=32,
            find_unused_parameters=True,
            static_graph=True)

    backbone_ddp = wrap_ddp(backbone)
    backbone_ddp_compiled = torch.compile(backbone_ddp)

    list_dali_dataloader = []
    list_head_names = []
    for head_id, dataset_config in enumerate(args.list_datasets):
        if dataset_config.dali_type == "decord":
            from dataloader.data_decord_video_sampling_frame import \
                get_dali_dataloader

            train_iter = get_dali_dataloader(
                data_root_path="",
                data_csv_path=dataset_config.prefixes[0],
                mode="train",
                dali_num_threads=2,
                dali_py_num_workers=4 // frame_scale_factor,
                decord_num_threads=frame_scale_factor,
                batch_size=args.list_batch_sizes_adjusted[head_id],
                input_size=args.image_size_video[0],
                sequence_length=args.actual_num_frames,
                seed=0+rank,
                shard_id=dataset_config.shard_id,
                num_shards=dataset_config.num_shards)
            logger.info(f"[head_id={head_id}] Video dataloader: batch_size={args.list_batch_sizes_adjusted[head_id]}, "
                        f"num_frames={args.actual_num_frames}")

        elif dataset_config.dali_type == "decord_residual":
            from dataloader.data_decord_llava_vit import get_dali_dataloader

            train_iter = get_dali_dataloader(
                data_root_path="",
                data_csv_path=dataset_config.prefixes[0],
                mode="train",
                dali_num_threads=2,
                dali_py_num_workers=4 // frame_scale_factor,
                decord_num_threads=frame_scale_factor,
                batch_size=args.list_batch_sizes_adjusted[head_id],
                input_size=args.image_size_video[0],
                sequence_length=64,
                seed=0+rank,
                shard_id=dataset_config.shard_id,
                num_shards=dataset_config.num_shards)

            logger.info(f"[head_id={head_id}] Video residual dataloader: batch_size={args.list_batch_sizes_adjusted[head_id]}, "
                        f"num_frames=64")

        elif dataset_config.dali_type == "origin":
            if args.debug:
                from dataloader.data_v2 import SyntheticDataIter
                train_iter = SyntheticDataIter(
                    args.list_batch_sizes_adjusted[head_id], 224, local_rank
                )
            else:
                from dataloader.data_v2 import MultiRecDALIWarper
                train_iter = MultiRecDALIWarper(
                    list_prefix=dataset_config.prefixes,
                    batch_size=args.list_batch_sizes_adjusted[head_id],
                    image_size=args.image_size,
                    workers=args.workers,
                    shard_id=dataset_config.shard_id,
                    num_shards=dataset_config.num_shards
        )
        elif dataset_config.dali_type == "ocr":
            if args.debug:
                from dataloader.data_v2_ocr import SyntheticDataIter
                train_iter = SyntheticDataIter(
                    args.list_batch_sizes_adjusted[head_id], 224, local_rank
                )
            else:
                from dataloader.data_v2_ocr import MultiRecDALIWarper
                train_iter = MultiRecDALIWarper(
                    list_prefix=dataset_config.prefixes,
                    batch_size=args.list_batch_sizes_adjusted[head_id],
                    image_size=args.image_size,
                    workers=args.workers,
                    shard_id=dataset_config.shard_id,
                    num_shards=dataset_config.num_shards
        )
        else:
            raise ValueError(
                f"dataset_config.dali_type {dataset_config.dali_type} not support!"
            )

        list_dali_dataloader.append(train_iter)
        list_head_names.append(dataset_config.name)

    if rank == 0:
        tb_writer = SummaryWriter(log_dir=f"{args.output}/tensorboard")
    else:
        tb_writer = None

    # Initialize callback for logging
    batch_end_callback = BatchEndCallBack(
        frequent=args.frequent,
        list_head_names=list_head_names,
        output=args.output,
        total_steps=args.total_steps,
        tb_writer=tb_writer,
    )
    log_args(args, logger, writer=tb_writer, save_dir=args.output, rank=rank)


    for head_id, dataset_config in enumerate(args.list_datasets):
        name = dataset_config.name if hasattr(dataset_config, "name") else f"head_{head_id}"
        prefixes = getattr(dataset_config, "prefixes", None)
        logger.info(
            f"[rank {rank}][local_rank {local_rank}] head_id={head_id} dataset={name} assigned_prefixes_num={len(prefixes) if prefixes is not None else 'N/A'}"
        )
        if prefixes is not None:
            preview_prefixes = prefixes
            logger.info(f"[rank {rank}][local_rank {local_rank}] prefixes preview: {preview_prefixes}")


    list_iter = []
    list_next_data_batch = []
    for i in range(args.num_heads):
        list_iter.append(iter(list_dali_dataloader[i]))
        list_next_data_batch.append(next(list_iter[i]))

    if global_step > args.total_steps:
        logger.info("global_step > total_steps")
        exit()

    num_samples = 0
    end_of_batch = False
    while not end_of_batch:
        list_data_batch = list_next_data_batch
        num_samples += sum(args.list_batch_sizes_adjusted) * world_size

        list_embedding = []
        list_batch_sizes = []
        for head_id, dataset_config in enumerate(args.list_datasets):

            dataset_config: Property
            if dataset_config.dali_type in ["decord"]:
                videos = list_data_batch[head_id]["videos"]       # [B, C, T, H, W]
                labels = list_data_batch[head_id]["labels"].view(-1)
                frame_indices = list_data_batch[head_id]["indices"]   # [B, seq_len]
                total_frames = list_data_batch[head_id]["total_frames"]  # [B, 1] or [B]

                bs, C, T, H, W = videos.shape
                target_frames = 64

                interpolated_indices = interpolate_frame_indices(
                    frame_indices,
                    total_frames.view(-1),
                    target_frames
                )

                seq_len = frame_indices.shape[1]
                frame_idx_expanded = interpolated_indices.view(bs, 1, seq_len, 1, 1).expand(bs, C, seq_len, H, W)

                per = torch.arange(args.num_tokens_per_frame, device="cuda")
                visible_index = (interpolated_indices.unsqueeze(-1) * args.num_tokens_per_frame + per).reshape(bs, -1)
                visible_index = visible_index.clamp_max(target_frames * args.num_tokens_per_frame - 1)

                with torch.amp.autocast(dtype=torch.bfloat16, device_type="cuda"):

                    output = backbone_ddp_compiled(videos, visible_index)
                    if hasattr(output, "pooler_output"):
                        head_embedding = output.pooler_output
                    else:
                        head_embedding  = output["head_output"]

                head_embedding = head_embedding.float()
                list_embedding.append(head_embedding)

            elif dataset_config.dali_type in ["decord_residual"]:
                head_input = list_data_batch[head_id]["videos"]  # [bs, C, 64, H, W]
                list_batch_sizes.append(head_input.size(0))
                visible_indices = list_data_batch[head_id]["video_visible_indices"]  # [bs, ?]
                visible_indices = visible_indices.long()

                bs = visible_indices.shape[0]
                dev = visible_indices.device

                # Get patch_size from backbone config
                backbone_module = unwrap_module(backbone)
                if hasattr(backbone_module, 'config'):
                    patch_size = backbone_module.config.patch_size
                elif hasattr(backbone_module, 'embeddings') and hasattr(backbone_module.embeddings, 'patch_size'):
                    patch_size = backbone_module.embeddings.patch_size
                else:
                    patch_size = 16  # default fallback

                out = visible_indices[:, :args.target_num].clone()
                n1 = int(bs * 0.5)
                n2 = int(bs * 0.875)

                idx_range = torch.arange(bs, device=dev)
                mask_residual = idx_range < n1                               # idx in [0, n1)
                mask_frame_sampling = (idx_range >= n1) & (idx_range < n2)   # idx in [n1, n2)
                mask_collage = idx_range >= n2                               # idx in [n2, bs)

                # For mask_residual: directly select first args.target_num patches
                if mask_residual.any():
                    sel_a = visible_indices[mask_residual, :args.target_num]
                    if sel_a.size(1) < args.target_num:
                        pad = sel_a[:, -1:].repeat(1, args.target_num - sel_a.size(1))
                        sel_a = torch.cat([sel_a, pad], dim=1)
                    out[mask_residual] = sel_a

                # For mask_frame_sampling: compute patch indices based on frame sampling
                SEQ = 8
                FRAMES = 64
                if mask_frame_sampling.any():
                    nB = visible_indices[mask_frame_sampling].size(0)
                    avg = FRAMES // SEQ
                    base = torch.arange(SEQ, device=dev) * avg
                    offs = torch.randint(avg, (nB, SEQ), device=dev)
                    frames = base + offs  # [nB, 8]

                    per = torch.arange(args.num_tokens_per_frame, device=dev)
                    pos = (frames.unsqueeze(-1) * args.num_tokens_per_frame + per).reshape(nB, -1)  # [nB, 8*num_tokens_per_frame]
                    sel_b = pos.to(visible_indices.dtype)

                    if sel_b.size(1) == args.target_num:
                        out[mask_frame_sampling] = sel_b
                    elif sel_b.size(1) > args.target_num:
                        out[mask_frame_sampling] = sel_b[:, :args.target_num]
                    else:
                        pad = sel_b[:, -1:].repeat(1, args.target_num - sel_b.size(1))
                        out[mask_frame_sampling] = torch.cat([sel_b, pad], dim=1)

                combined_mask = mask_residual | mask_frame_sampling
                if combined_mask.any():
                    combined_idx = torch.nonzero(combined_mask, as_tuple=False).squeeze(1)
                    combined_video = head_input[combined_idx]  # [n, C, 64, H, W]
                    combined_out = out[combined_idx]  # [n, target_num]

                    n_comb, C_vid, T_vid, H_vid, W_vid = combined_video.shape
                    Hp = H_vid // patch_size  # patches per row
                    Wp = W_vid // patch_size  # patches per col
                    patches_per_frame = Hp * Wp
                    total_patches = T_vid * patches_per_frame

                    # Convert video to patches: [n, C, T*Hp*Wp, patch_size, patch_size]
                    # First reshape to [n, C, T, Hp, patch_size, Wp, patch_size]
                    video_reshaped = combined_video.view(n_comb, C_vid, T_vid, Hp, patch_size, Wp, patch_size)
                    # Permute to [n, C, T, Hp, Wp, patch_size, patch_size]
                    video_reshaped = video_reshaped.permute(0, 1, 2, 3, 5, 4, 6)
                    # Reshape to [n, C, T*Hp*Wp, patch_size, patch_size]
                    video_patches = video_reshaped.reshape(n_comb, C_vid, total_patches, patch_size, patch_size)

                    # Select patches using combined_out (visible_indices): [n, target_num]
                    # Expand combined_out for gathering: [n, target_num, 1, 1] -> [n, C, target_num, patch_size, patch_size]
                    idx_expand = combined_out.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # [n, 1, target_num, 1, 1]
                    idx_expand = idx_expand.expand(-1, C_vid, -1, patch_size, patch_size)  # [n, C, target_num, patch_size, patch_size]
                    selected_patches = torch.gather(video_patches, 2, idx_expand)  # [n, C, target_num, patch_size, patch_size]

                    # Reshape selected patches back to video format [n, C, T', H', W']
                    # We have target_num patches, need to figure out T', Hp', Wp'
                    # For simplicity: T' = target_num // patches_per_frame, and use original Hp, Wp
                    T_new = args.target_num // patches_per_frame
                    expected_patches = T_new * patches_per_frame

                    # Handle case when target_num is not divisible by patches_per_frame
                    if expected_patches != args.target_num:
                        T_new = max(1, T_new)
                        expected_patches = T_new * patches_per_frame
                        # Truncate or pad selected_patches to match expected_patches
                        if args.target_num > expected_patches:
                            selected_patches = selected_patches[:, :, :expected_patches, :, :]
                        else:
                            # Pad with the last patch repeated
                            pad_size = expected_patches - args.target_num
                            pad_patches = selected_patches[:, :, -1:, :, :].repeat(1, 1, pad_size, 1, 1)
                            selected_patches = torch.cat([selected_patches, pad_patches], dim=2)

                    # Reshape: [n, C, expected_patches, patch_size, patch_size] -> [n, C, T_new, Hp, Wp, patch_size, patch_size]
                    # Then -> [n, C, T_new, Hp*patch_size, Wp*patch_size] = [n, C, T_new, H, W]
                    H_new = Hp * patch_size
                    W_new = Wp * patch_size

                    # First reshape to [n, C, T_new, Hp, Wp, patch_size, patch_size]
                    selected_reshaped = selected_patches.view(n_comb, C_vid, T_new, Hp, Wp, patch_size, patch_size)
                    # Permute to [n, C, T_new, Hp, patch_size, Wp, patch_size]
                    selected_reshaped = selected_reshaped.permute(0, 1, 2, 3, 5, 4, 6)
                    # Reshape to [n, C, T_new, H_new, W_new]
                    combined_head_input = selected_reshaped.reshape(n_comb, C_vid, T_new, H_new, W_new)

                    with torch.amp.autocast(dtype=torch.bfloat16, device_type="cuda"):
                        combined_head_output = backbone_ddp_compiled(combined_head_input, combined_out)
                    if hasattr(combined_head_output, "pooler_output"):
                        combined_head_output = combined_head_output.pooler_output
                    else:
                        combined_head_output = combined_head_output["head_output"]

                    combined_head_output = combined_head_output.float()


                if mask_collage.any():
                    coll_idx = torch.nonzero(mask_collage, as_tuple=False).squeeze(1)
                    nC = coll_idx.numel()
                    SEQ = 8
                    FRAMES = 64  # assume fixed 64 frames for head_subset

                    head_subset = head_input[coll_idx]  # [nC, C, 64, H, W] (must hold)

                    # 检查形状
                    if head_subset.dim() != 5 or head_subset.size(2) != FRAMES:
                        raise RuntimeError(
                            f"collage branch expects head_subset shape [nC, C, {FRAMES}, H, W], got {tuple(head_subset.shape)}"
                        )

                    nC = head_subset.size(0)
                    Cf = head_subset.size(1)
                    Hf = head_subset.size(3)
                    Wf = head_subset.size(4)
                    avg = FRAMES // SEQ  # 8
                    base = torch.arange(SEQ, device=dev) * avg
                    offs = torch.randint(avg, (nC, SEQ), device=dev)
                    frames_idx = (base.unsqueeze(0) + offs).long().clamp(max=FRAMES - 1)  # [nC, SEQ], 范围在 [0, 63]
                    idx_expand = frames_idx.view(nC, 1, SEQ, 1, 1).expand(-1, Cf, -1, Hf, Wf).to(head_subset.device)
                    sel_frames = torch.gather(head_subset, 2, idx_expand)  # [nC, Cf, SEQ, Hf, Wf]
                    sel_frames = sel_frames.permute(0, 2, 1, 3, 4)  # [nC, SEQ, Cf, Hf, Wf]
                    grid_rows = [sel_frames[:, i, :, :, :] for i in range(SEQ)]
                    grid = torch.cat(grid_rows, dim=-2)  # [nC, Cf, Hf*SEQ, Wf]
                    with torch.amp.autocast(dtype=torch.bfloat16, device_type="cuda"):
                        collage_head_output = backbone_ddp_compiled(grid)
                    if hasattr(collage_head_output, "pooler_output"):
                        collage_head_output = collage_head_output.pooler_output
                    else:
                        collage_head_output  = collage_head_output["head_output"]
                    collage_head_output = collage_head_output.float()

                D = combined_head_output.size(1)

                head_embedding_full = torch.zeros(bs, D, device=dev, dtype=torch.float32)
                if combined_mask.any():
                    head_embedding_full[combined_idx] = combined_head_output
                if mask_collage.any():
                    head_embedding_full[coll_idx] = collage_head_output

                list_embedding.append(head_embedding_full)

            elif dataset_config.dali_type in ["origin", "ocr"]:
                head_input = list_data_batch[head_id]["pixel_values"]
                list_batch_sizes.append(head_input.size(0))
                with torch.amp.autocast(dtype=torch.bfloat16, device_type="cuda"):

                    output = backbone_ddp_compiled(head_input)
                    if hasattr(output, "pooler_output"):
                        head_embedding = output.pooler_output
                    else:
                        head_embedding  = output["head_output"]
                head_embedding = head_embedding.float()

                list_embedding.append(head_embedding)
            else:
                raise ValueError(f"Unsupported DALI type: {dataset_config.dali_type}")

        list_loss = []
        list_loss_float = []

        for head_id, pfc in enumerate(list_module_pfc):
            dataset_config = args.list_datasets[head_id]
            head_embedding = list_embedding[head_id]
            head_label = list_data_batch[head_id]["labels"].long().cuda()
            label_select = dataset_config.label_select
            random_diff = dataset_config.random_diff
            loss_weight = args.list_loss_weights[head_id]
            head_label = head_label[
                :, label_select : label_select + random_diff
            ]
            head_loss = pfc(head_embedding, head_label, random_diff) * loss_weight
            list_loss.append(head_loss)
            list_loss_float.append(head_loss.item())

        is_accumulation_step = (global_step % args.backward_passes_per_step != 0)
        scaled_loss = sum(list_loss) / args.backward_passes_per_step

        if is_accumulation_step:
            with backbone_ddp_compiled.no_sync():
                scaled_loss.backward()
        else:
            scaled_loss.backward()
            clip_grad_norm_(backbone_ddp_compiled.parameters(), max_norm=5, norm_type=2)
            for pfc in list_module_pfc:
                clip_grad_norm_(pfc.parameters(), max_norm=5, norm_type=2)
            opt.step()
            opt.zero_grad()

            # fix: lr update should only happen after opt.step(), not every micro-batch
            lr_scheduler.step()

        batch_end_callback(
            global_step=global_step,
            lr_scheduler=lr_scheduler,
            list_loss_float=list_loss_float,
            batch_size=args.batch_size,
            num_samples=num_samples
        )

        global_step += 1

        for i in range(args.num_heads):
            list_next_data_batch[i] = next(list_iter[i])

        if global_step % args.ckpt_interval == 0:
            save_checkpoint(
                args.output,
                backbone,
                pfc_modules=dict_pfc_modules,
                lr_scheduler=lr_scheduler,
                amp=None,
                global_step=global_step,
                list_head_names=args.list_head_names,
                keep_num=20,
            )
            # Also save in HuggingFace format
            save_hf_checkpoint(
                args.output,
                backbone,
                global_step=global_step,
                image_size=args.image_size[0]
            )

        if global_step > args.total_steps:
            save_checkpoint(
                args.output,
                backbone,
                pfc_modules=dict_pfc_modules,
                lr_scheduler=lr_scheduler,
                amp=None,
                global_step=global_step,
                list_head_names=args.list_head_names,
                keep_num=20,
            )
            # Also save final model in HuggingFace format
            save_hf_checkpoint(
                args.output,
                backbone,
                global_step=global_step,
                image_size=args.image_size[0]
            )
            logger.info(f"Training completed at step {global_step}")
            exit()


def interpolate_frame_indices(frame_indices: torch.Tensor, total_frames: torch.Tensor, target_frames: int = 64) -> torch.Tensor:
    bs, seq_len = frame_indices.shape
    device = frame_indices.device
    total_frames_float = total_frames.float().view(bs, 1)
    frame_indices_float = frame_indices.float()
    total_frames_safe = torch.clamp(total_frames_float - 1, min=1.0)
    interpolated_indices = (frame_indices_float / total_frames_safe) * (target_frames - 1)
    interpolated_indices = torch.round(interpolated_indices).long()
    interpolated_indices = torch.clamp(interpolated_indices, 0, target_frames - 1)
    return interpolated_indices


class BatchEndCallBack(object):
    def __init__(
        self,
        frequent: int,
        list_head_names: List[str],
        output: str,
        total_steps: int,
        tb_writer = None,
    ):
        self.frequent: int = frequent
        self.list_head_names: List[str] = list_head_names
        self.output: str = output
        self.total_steps: int = total_steps

        self.num_head = len(self.list_head_names)
        self.time_start = time.time()
        self.list_loss_metric = [ScalaMetric() for _ in self.list_head_names]
        self.init = False
        self.tic = 0

        self.step_times = []
        self.max_time_history = 100

        self.total_examples = 0
        # Create TensorBoard writer if rank 0
        if rank == 0:
            self.tb_writer = tb_writer
        else:
            self.tb_writer = None
        self.logger = logging.getLogger(__name__)

    def __call__(
        self,
        global_step: int,
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
        list_loss_float: List[float],
        batch_size: int,
        num_samples=None,
    ):
        for i in range(self.num_head):
            self.list_loss_metric[i].update(list_loss_float[i])

        if global_step > 0 and global_step % self.frequent == 0:
            if self.init:
                current_time = time.time()
                time_elapsed = current_time - self.tic
                self.tic = current_time
                time_per_step = time_elapsed / self.frequent

                self.step_times.append(time_per_step)
                if len(self.step_times) > self.max_time_history:
                    self.step_times.pop(0)

                avg_time_per_step = sum(self.step_times) / len(self.step_times)

                remaining_steps = self.total_steps - global_step
                remaining_time_hours = (avg_time_per_step * remaining_steps) / 3600

                try:
                    speed: float = self.frequent * batch_size / time_elapsed
                    speed_total = speed * world_size
                except ZeroDivisionError:
                    speed = float("inf")
                    speed_total = float("inf")

                header = f"rank {speed:.2f} total {speed_total:.2f} its/s lr: {lr_scheduler.get_last_lr()[0]:.8f} "
                progress = f"step: {global_step}/{self.total_steps} ({global_step/self.total_steps*100:.2f}%) "
                time_info = f"remain: {remaining_time_hours:.2f} hours"

                loss_str_format = ""
                for head_id, name in enumerate(self.list_head_names):

                    if rank == 0 and self.tb_writer:
                        self.tb_writer.add_scalar(
                            f"loss/{name}",
                            self.list_loss_metric[head_id].avg,
                            global_step
                        )
                        self.tb_writer.add_scalar(
                            f"lr/{name}",
                            lr_scheduler.get_last_lr()[head_id + 1],
                            global_step
                        )

                        self.tb_writer.add_scalar(
                            f"samples vs. loss/{name}",
                            self.list_loss_metric[head_id].avg,
                            num_samples,
                        )

                    loss_str_format += f"\n{f'name: {name}':<50}{f'lr: {lr_scheduler.get_last_lr()[head_id + 1]:.8f}':<20}"
                    loss_str_format += f"{f'loss: {self.list_loss_metric[head_id].avg:.4f}':<20}"
                    self.list_loss_metric[head_id].reset()

                examples_info = f"samples: {num_samples}"
                msg = f"{header}{progress}{time_info} {examples_info}{loss_str_format}"

                if rank == 0:
                    logger.info(msg)
                    # Flush TensorBoard writer
                    if self.tb_writer:
                        self.tb_writer.flush()
            else:
                self.init = True
                self.tic = time.time()


class ScalaMetric(object):
    def __init__(self):
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def log_args(args, logger, writer: SummaryWriter = None, save_dir: str = None, rank: int = 0):
    if rank != 0:
        return

    args_dict: Dict[str, Any] = vars(args) if not isinstance(args, dict) else args

    sorted_items = sorted(args_dict.items(), key=lambda x: x[0])

    sep = "-" * 92
    logger.info(sep)
    logger.info("Training / Runtime Arguments")
    logger.info(sep)

    max_key_len = max(len(k) for k, _ in sorted_items) if sorted_items else 0
    col_width = max(20, max_key_len)
    for k, v in sorted_items:

        vs = str(v)
        if len(vs) > 300:
            vs = vs[:297] + "..."
        logger.info(f"{k:<{col_width}} = {vs}")
    logger.info(sep)

    # ---------- TensorBoard 记录 ----------
    if writer is not None:

        md_lines = ["| Argument | Value |", "|----------|-------|"]
        for k, v in sorted_items:
            vs = str(v).replace("|", "\\|")
            if len(vs) > 500:
                vs = vs[:497] + "..."
            md_lines.append(f"| {k} | {vs} |")
        writer.add_text("markdown_table", "\n".join(md_lines), global_step=0)

if __name__ == "__main__":
    main()
