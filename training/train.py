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

import model_factory
from dataset import DATASET_REGISTRY, Property
from training.checkpoint_utils import load_checkpoint, save_checkpoint
from training.fused_partial_fc_v2_multi_res import CombinedMarginLoss, PartialFC_V2
from training.lr_scheduler import PolynomialLRWarmup
from onevision_encoder import OneVisionEncoderModel, OneVisionEncoderConfig

# fix: removed conflicting line (was: True immediately overwritten by False)
torch._dynamo.config.optimize_ddp = False

parser = argparse.ArgumentParser(description="Multi-dataset video training")

# ---------------------------
# General / 通用
# ---------------------------
parser.add_argument("--debug", type=int, default=0, help="Enable debug mode (0/1). When 1, may reduce dataset size or add extra checks / 是否开启调试模式")
parser.add_argument("--output", default="output", help="Output directory for logs and checkpoints / 输出目录")
parser.add_argument("--workers", type=int, default=2, help="Number of DataLoader workers per process / DataLoader 进程内工作线程数")
parser.add_argument("--local_rank", type=int, default=0, help="Local rank passed by launcher; do not set manually / 启动器传入的本地进程序号，通常无需手动设置")

# ---------------------------
# Data loading / 数据加载
# ---------------------------
parser.add_argument("--dataloader-type", default="dali", help="Data loader backend, e.g., 'dali' or 'torch' / 数据加载后端")
parser.add_argument("--dali_is_training", type=int, default=1, help="DALI training mode (0/1). 1 enables training augmentations / DALI 是否处于训练模式")
parser.add_argument("--image_size", default="224", help="Input size as 'H,W' or single 'S' (interpreted as S,S) / 输入尺寸，'H,W' 或单值 'S'(等价于 S,S)")
parser.add_argument("--image_size_video", default="224", help="Input size as 'H,W' or single 'S' (interpreted as S,S) / 输入尺寸，'H,W' 或单值 'S'(等价于 S,S)")
parser.add_argument("--input_gray", type=int, default=0, help="Treat input as grayscale (0/1) / 输入按灰度处理（0/1）")
parser.add_argument("--num_frames", type=int, default=8, help="Number of frames per clip / 每个样本的帧数")
parser.add_argument("--random_diff", type=int, default=10, help="Random diff for sampling jitter across datasets / 数据集采样抖动的随机扰动")

# ---------------------------
# Multi-dataset (heads) / 多数据集（多头）
# 说明：左侧为新复数参数名，右侧为旧名别名，dest 统一为新名，向后兼容
# ---------------------------
parser.add_argument("--list_datasets", nargs='+', type=str, default=["k710_ssv2_univit_pfs"],
                    help="Dataset registry names, one or more / 数据集注册名，可多个")
parser.add_argument("--list_batch_sizes", nargs='+', type=int, default=[32],
                    help="Per-dataset batch sizes / 各数据集的 batch 大小")
parser.add_argument("--list_sample_rates", nargs='+', type=float, default=[0.1],
                    help="Per-dataset sampling rate / 各数据集采样权重")
parser.add_argument("--list_margins", nargs='+', type=float, default=[0.3],
                    help="Per-dataset loss margin / 各数据集损失 margin")
parser.add_argument("--list_filters", nargs='+', type=float, default=[0.75],
                    help="Per-dataset filter ratio or threshold / 各数据集过滤比例或阈值")
parser.add_argument("--list_lr_pfc_weights", nargs='+', type=float, default=[1.0],
                    help="Per-dataset LR scale for PFC params / 各数据集 PFC 参数学习率缩放")
parser.add_argument("--list_loss_weights", nargs='+', type=float, default=[1.0],
                    help="Per-dataset loss weights / 各数据集损失权重")
parser.add_argument("--list_init_partial_fc_paths", nargs='+', type=str, default=["NULL"],
                    help="Per-dataset init path for partial-FC or 'NULL' / 各数据集 PFC 初始化路径或 'NULL'")

# ---------------------------
# Model / 模型
# ---------------------------
parser.add_argument("--model_name", default="pretrain_encoder_small_patch16_224_v10_12_rms_unmask_with_head", help="Backbone model name / 主干模型名称")
parser.add_argument("--model_weight", default="/vlm/xiangan/VideoMLCD/checkpoints/llava_vit_s_16.py/00190000/backbone.pt",
                    help="Path to pretrained weights or None / 预训练权重路径，或 None")
parser.add_argument("--embedding_size", type=int, default=384, help="Embedding dimension of the head / 头部嵌入维度")
parser.add_argument("--gradient_checkpoint", type=int, default=0, help="Enable gradient checkpointing (0/1) / 是否启用梯度检查点（节省显存）")
parser.add_argument("--mask", type=int, default=0, help="Enable mask-related training (0/1) / 是否启用 mask 相关训练")
parser.add_argument("--finetune_backbone", type=int, default=1, help="Finetune backbone parameters (0/1) / 是否微调主干网络")

# ---------------------------
# Optimization / 优化
# ---------------------------
parser.add_argument("--opt", default="adamw", help="Optimizer name, e.g., 'adamw' / 优化器名称")
parser.add_argument("--lr", type=float, default=1e-3, help="Base learning rate / 基础学习率")
parser.add_argument("--weight_decay", type=float, default=0.05, help="Weight decay for non-PFC params / 非 PFC 参数的权重衰减")
parser.add_argument("--weight_decay_pfc", type=float, default=0.05, help="Weight decay for PFC params / PFC 参数的权重衰减")
parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio of total training steps / 训练总步数的预热比例")
parser.add_argument("--backward_passes_per_step", type=int, default=1, help="Gradient accumulation steps before optimizer step / 每次优化前累积的反传次数")
parser.add_argument("--repeat_pfc", type=int, default=0, help="Repeat factor for PFC ops or rebuild cycles / PFC 重复或重建次数（如适用）")
parser.add_argument("--save_pfc", type=int, default=1, help="Save PFC weights in checkpoints (0/1) / 是否在检查点中保存 PFC 权重")

# ---------------------------
# Initialization / Resume / 初始化与恢复
# ---------------------------
parser.add_argument("--init_backbone", default="NULL", help="Backbone init path or 'NULL' / 主干网络初始化路径，或 'NULL'")

# ---------------------------
# Logging & Checkpoint / 日志与检查点
# ---------------------------
parser.add_argument("--frequent", type=int, default=10, help="Log/validation frequency in steps / 日志与验证的步数间隔")
parser.add_argument("--ckpt_interval", type=int, default=2000, help="Checkpoint save interval in steps / 检查点保存步数间隔")

# ---------------------------
# Training schedule / 训练调度
# ---------------------------
parser.add_argument("--num_sampled_data", type=int, default=60000000, help="Total sampled examples used to compute total steps / 用于估算总步数的采样样本总量")

# ---------------------------
# Visualization / 可视化
# ---------------------------
parser.add_argument("--visualize", type=int, default=0, help="Save input clips as GIFs (0/1) / 是否将输入视频保存为 GIF")
parser.add_argument("--vis_samples", type=int, default=2, help="Number of samples to visualize per batch / 每个 batch 可视化的样本数")
parser.add_argument("--vis_interval", type=int, default=10, help="Visualization save interval in steps / 可视化保存的步数间隔")

# ---------------------------
# Index sampling for ViT input / ViT 输入的索引采样
# ---------------------------

parser.add_argument("--total_indices", type=int, default=2048, help="Visible indices total count / 可见索引总数")
parser.add_argument("--target_num", type=int, default=2048, help="Sampled indices count / 采样索引个数")
parser.add_argument("--must_num", type=int, default=256, help="Number of indices must be included (from front) / 必须包含的索引数 (前面)")
parser.add_argument("--num_tokens_per_frame", type=int, default=256, help="Number of indices must be included (from front) / 必须包含的索引数 (前面)")

# ---------------------------
# Multi-frame training / 多帧训练
# 说明：视频分支支持4帧、8帧、16帧、32帧混合训练，每个rank使用不同帧数
# batch size反比：32帧bs=base, 16帧bs=2*base, 8帧bs=4*base, 4帧bs=8*base
# ---------------------------
parser.add_argument("--enable_multi_frame", type=int, default=1,
                    help="Enable multi-frame training (0/1). When enabled, different ranks use different frame counts / 是否启用多帧混合训练")
parser.add_argument("--multi_frame_list", nargs='+', type=int, default=[8],
                    help="List of frame counts to use in multi-frame training / 多帧训练时使用的帧数列表")
parser.add_argument("--base_num_frames", type=int, default=8,
                    help="Base frame count for batch size calculation. Batch size is inversely proportional: bs = base_bs * (base_num_frames / actual_num_frames) / 用于计算batch size的基准帧数")

args = parser.parse_args()

rank = int(os.getenv("RANK", "0"))
local_rank = int(os.getenv("LOCAL_RANK", "0"))
world_size = int(os.getenv("WORLD_SIZE", "1"))
distributed.init_process_group(backend="nccl")

torch.cuda.set_device(local_rank)
torch.backends.cudnn.benchmark = True

os.makedirs(args.output, exist_ok=True)

if rank == 0:
    logger: logging.Logger = logging.getLogger(__name__)  # 模块级 logger
    formatter = logging.Formatter(f"rank-id:{rank:03d}:%(asctime)s-%(message)s")
    file_handler = logging.FileHandler(os.path.join(args.output, f"training_{rank:03d}.logger"))
    stream_handler = logging.StreamHandler(sys.stdout)
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)
else:
    logger: logging.Logger = logging.getLogger(__name__)  # 模块级 logger
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

    # =====================================================
    # Multi-frame training configuration / 多帧训练配置
    # 根据 rank % 4 确定每个 GPU 使用的帧数
    # batch size 与帧数成反比: base_bs * (base_num_frames / actual_num_frames)
    # =====================================================
    if args.enable_multi_frame:
        num_frame_options = len(args.multi_frame_list)
        frame_index = rank % num_frame_options
        args.actual_num_frames = args.multi_frame_list[frame_index]
        # 计算 batch size 缩放因子：帧数越少，batch size 越大
        # 验证 base_num_frames 必须能被 actual_num_frames 整除
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

    # 实例化数据集（使用复数参数名）
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

    # =====================================================
    # 根据多帧配置调整每个数据集的 batch size
    # 对于 video 类型 (dali_type == "decord")，应用帧数缩放
    # =====================================================
    args.list_batch_sizes_adjusted = []
    for head_id, dataset_config in enumerate(args.list_datasets):
        base_bs = args.list_batch_sizes[head_id]
        if dataset_config.dali_type == "decord":
            # 视频分支：batch size 与帧数成反比
            adjusted_bs = base_bs * frame_scale_factor
            logger.info(f"[head_id={head_id}] Video branch: base_bs={base_bs}, "
                        f"adjusted_bs={adjusted_bs} (scale={frame_scale_factor}x)")
        else:
            # 图像分支：保持原有 batch size
            adjusted_bs = base_bs
            logger.info(f"[head_id={head_id}] Image branch: bs={adjusted_bs}")
        args.list_batch_sizes_adjusted.append(adjusted_bs)

    # 其余派生量
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

    # 根据 finetune_backbone 控制哪些层参与训练：
    # - finetune_backbone = 1: 整个 backbone 可训练
    # - finetune_backbone = 0: 只有 head 可训练，其它全部冻结
    if args.finetune_backbone:
        backbone.requires_grad_(True)
    else:
        # 先全部冻结
        backbone.requires_grad_(False)
        # 仅打开 head 的梯度（假设 LlavaViTEncoder 上有 .head）
        backbone_module = unwrap_module(backbone)
        if hasattr(backbone_module, "head"):
            for p in backbone_module.head.parameters():
                p.requires_grad = True
        else:
            raise RuntimeError(
                "finetune_backbone==0 但 backbone 上没有属性 'head'，"
                "请确认使用的是 LlavaViTEncoder 并且 use_head=True。"
            )

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
    # backbone_ddp_compiled = backbone_ddp
    backbone_ddp_compiled = torch.compile(backbone_ddp)

    list_dali_dataloader = []
    list_head_names = []
    # print("开始加载数据了")
    for head_id, dataset_config in enumerate(args.list_datasets):
        if dataset_config.dali_type == "decord":
            from dataloader.data_decord_video_sampling_frame import get_dali_dataloader

            # 使用调整后的 batch size 和实际帧数
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

            # 使用调整后的 batch size 和实际帧数
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
                # print("dataset_config.prefix", dataset_config.prefixes)
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
                # print("dataset_config.prefix", dataset_config.prefixes)
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


    # -------- 这里加一段logger，输出每个rank分到的数据 --------

    for head_id, dataset_config in enumerate(args.list_datasets):
        name = dataset_config.name if hasattr(dataset_config, "name") else f"head_{head_id}"
        prefixes = getattr(dataset_config, "prefixes", None)
        logger.info(
            f"[rank {rank}][local_rank {local_rank}] head_id={head_id} dataset={name} assigned_prefixes_num={len(prefixes) if prefixes is not None else 'N/A'}"
        )
        if prefixes is not None:
            preview_prefixes = prefixes
            logger.info(f"[rank {rank}][local_rank {local_rank}] prefixes preview: {preview_prefixes}")
            # 如需全部打印，可以用:
            # for p in prefixes:
            #     logger.info(f"    {p}")

    # -----------------------------------------------------

    list_iter = []
    list_next_data_batch = []
    for i in range(args.num_heads):
        # list_dali_dataloader[i].reset()
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

                # === 插值indices到目标帧数 ===
                interpolated_indices = interpolate_frame_indices(
                    frame_indices,
                    total_frames.view(-1),
                    target_frames
                )   # [B, seq_len]

                padded_videos = torch.zeros(bs, C, target_frames, H, W, device="cuda", dtype=videos.dtype)
                # 为scatter准备indices
                seq_len = frame_indices.shape[1]
                frame_idx_expanded = interpolated_indices.view(bs, 1, seq_len, 1, 1).expand(bs, C, seq_len, H, W)
                # scatter填充各帧
                padded_videos.scatter_(dim=2, index=frame_idx_expanded, src=videos)

                # 计算可见帧token编号
                per = torch.arange(args.num_tokens_per_frame, device="cuda")
                visible_index = (interpolated_indices.unsqueeze(-1) * args.num_tokens_per_frame + per).reshape(bs, -1)
                visible_index = visible_index.clamp_max(target_frames * args.num_tokens_per_frame - 1)

                with torch.amp.autocast(dtype=torch.bfloat16, device_type="cuda"):

                    output = backbone_ddp_compiled(padded_videos, visible_index)
                    if hasattr(output, "pooler_output"):
                        head_embedding = output.pooler_output
                    else:
                        head_embedding  = output["head_output"]

                head_embedding = head_embedding.float()
                list_embedding.append(head_embedding)

            elif dataset_config.dali_type in ["decord_residual"]:
                head_input = list_data_batch[head_id]["videos"]
                list_batch_sizes.append(head_input.size(0))
                visible_indices = list_data_batch[head_id]["video_visible_indices"]  # [bs, ?]，需要至少 args.total_indices 合法列
                visible_indices = visible_indices.long()

                bs = visible_indices.shape[0]
                dev = visible_indices.device

                # 初始化 out：默认使用 visible_indices 的前 args.target_num 列的拷贝
                out = visible_indices[:, :args.target_num].clone()

                # 按 batch 固定划分：前50% residual, 中37.5% frame_sampling, 后12.5% collage
                n1 = int(bs * 0.5)
                # fix: n2 must be cumulative threshold, not standalone percentage
                # bug was: n2 = int(bs * 0.375) which gives n2=37 when bs=100
                # this caused mask_frame_sampling = (idx >= 50) & (idx < 37) to be always False
                n2 = int(bs * 0.875)  # cumulative: 50% + 37.5% = 87.5%

                idx_range = torch.arange(bs, device=dev)
                mask_residual = idx_range < n1                               # idx in [0, n1)
                mask_frame_sampling = (idx_range >= n1) & (idx_range < n2)   # idx in [n1, n2)
                mask_collage = idx_range >= n2                               # idx in [n2, bs)

                # ---------- residual（前50%）: 生成 out 行 ----------
                if mask_residual.any():
                    vis_a = visible_indices[mask_residual, :args.total_indices]
                    must = vis_a[:, :args.must_num]
                    candidates = vis_a[:, args.must_num:args.total_indices]
                    k = max(0, args.target_num - args.must_num)
                    k = min(k, candidates.size(1))
                    if k > 0:
                        scores = torch.rand(vis_a.size(0), candidates.size(1), device=dev)
                        idx = scores.topk(k, dim=1).indices
                        sampled = torch.gather(candidates, 1, idx)
                        sel_a = torch.cat([must, sampled], dim=1)
                    else:
                        sel_a = must
                    if sel_a.size(1) < args.target_num:
                        pad = sel_a[:, -1:].repeat(1, args.target_num - sel_a.size(1))
                        sel_a = torch.cat([sel_a, pad], dim=1)
                    out[mask_residual] = sel_a

                # ---------- frame_sampling（中35%）: 生成 out 行 ----------
                if mask_frame_sampling.any():
                    nB = visible_indices[mask_frame_sampling].size(0)
                    SEQ = 8
                    FRAMES = 64
                    avg = FRAMES // SEQ
                    base = torch.arange(SEQ, device=dev) * avg
                    offs = torch.randint(avg, (nB, SEQ), device=dev)
                    frames = base + offs  # [nB, 8]

                    per = torch.arange(args.must_num, device=dev)
                    pos = (frames.unsqueeze(-1) * args.must_num + per).reshape(nB, -1)  # [nB, 8*args.must_num]
                    sel_b = pos.to(visible_indices.dtype)

                    if sel_b.size(1) == args.target_num:
                        out[mask_frame_sampling] = sel_b
                    elif sel_b.size(1) > args.target_num:
                        out[mask_frame_sampling] = sel_b[:, :args.target_num]
                    else:
                        pad = sel_b[:, -1:].repeat(1, args.target_num - sel_b.size(1))
                        out[mask_frame_sampling] = torch.cat([sel_b, pad], dim=1)

                # ---------- combined: residual + frame_sampling 一起推理（有 out） ----------
                combined_mask = mask_residual | mask_frame_sampling
                if combined_mask.any():
                    combined_idx = torch.nonzero(combined_mask, as_tuple=False).squeeze(1)
                    combined_head_input = head_input[combined_idx]  # 保持原样（可能为 [n, C, H, W] 或其他）
                    combined_out = out[combined_idx]

                    with torch.amp.autocast(dtype=torch.bfloat16, device_type="cuda"):
                        combined_head_output = backbone_ddp_compiled(combined_head_input, combined_out)
                    if hasattr(combined_head_output, "pooler_output"):
                        combined_head_output = combined_head_output.pooler_output
                    else:
                        combined_head_output  = combined_head_output["head_output"]

                    combined_head_output = combined_head_output.float()

                # ---------- collage（后15%）: 从 head_input split 出帧，按 frame_sampling 策略抽帧并拼成 8行1列单张图，然后像 origin 一样单独推理 ----------
                if mask_collage.any():
                    coll_idx = torch.nonzero(mask_collage, as_tuple=False).squeeze(1)
                    nC = coll_idx.numel()
                    SEQ = 8
                    FRAMES = 64  # assume fixed 64 frames for head_subset

                    # 从 head_input 中选出需要做 collage 的样本，期望形状为 [nC, C, 64, H, W]
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

                    # 与 frame_sampling 一致的抽帧策略（在 64 帧上均匀分段后随机 offset）
                    avg = FRAMES // SEQ  # 8
                    base = torch.arange(SEQ, device=dev) * avg
                    offs = torch.randint(avg, (nC, SEQ), device=dev)
                    frames_idx = (base.unsqueeze(0) + offs).long().clamp(max=FRAMES - 1)  # [nC, SEQ], 范围在 [0, 63]

                    # 用 gather 从 head_subset 采样：gather 在 time 维 (dim=2)
                    # 为 gather 准备索引形状 [nC, Cf, SEQ, Hf, Wf]
                    idx_expand = frames_idx.view(nC, 1, SEQ, 1, 1).expand(-1, Cf, -1, Hf, Wf).to(head_subset.device)
                    sel_frames = torch.gather(head_subset, 2, idx_expand)  # [nC, Cf, SEQ, Hf, Wf]

                    # 为拼接方便，转为 [nC, SEQ, Cf, Hf, Wf]
                    sel_frames = sel_frames.permute(0, 2, 1, 3, 4)  # [nC, SEQ, Cf, Hf, Wf]

                    # 竖向拼接为 8 行 1 列图 -> [nC, Cf, Hf*SEQ, Wf]
                    grid_rows = [sel_frames[:, i, :, :, :] for i in range(SEQ)]
                    grid = torch.cat(grid_rows, dim=-2)  # [nC, Cf, Hf*SEQ, Wf]

                    # 像 origin 一样单独推理（不传 out）
                    with torch.amp.autocast(dtype=torch.bfloat16, device_type="cuda"):
                        collage_head_output = backbone_ddp_compiled(grid)
                    if hasattr(collage_head_output, "pooler_output"):
                        collage_head_output = collage_head_output.pooler_output
                    else:
                        collage_head_output  = collage_head_output["head_output"]
                    collage_head_output = collage_head_output.float()

                # ---------- 汇总 combined 与 collage 输出，按 batch 顺序放回 ----------
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
            # 中间累积步骤，避免DDP通信
            with backbone_ddp_compiled.no_sync():
                scaled_loss.backward()
        else:
            # 最后一步正常backward，会进行梯度同步
            scaled_loss.backward()

            # 只在累积完成时执行梯度裁剪和优化器更新
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
    """
    插值原始帧索引到目标帧数
    """
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
        self.list_loss_metric = [ScalaMetric() for _ in self.list_head_names]  # 只保留一个loss
        self.init = False
        self.tic = 0
        # 用于计算平均每步时间
        self.step_times = []
        self.max_time_history = 100  # 保留最近100个step的时间来平均
        # 累计样本数计数器
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
        list_loss_float: List[float],  # 只需要一个loss列表
        batch_size: int,
        num_samples=None,  # 新增参数，用于记录每个batch实际处理的样本数
    ):
        for i in range(self.num_head):
            self.list_loss_metric[i].update(list_loss_float[i])

        if global_step > 0 and global_step % self.frequent == 0:
            if self.init:
                current_time = time.time()
                time_elapsed = current_time - self.tic
                self.tic = current_time

                # 计算这个frequent间隔内每个step的平均时间
                time_per_step = time_elapsed / self.frequent

                # 保存到历史记录，用于平滑计算
                self.step_times.append(time_per_step)
                if len(self.step_times) > self.max_time_history:
                    self.step_times.pop(0)

                # 计算平均每步时间（使用最近的记录）
                avg_time_per_step = sum(self.step_times) / len(self.step_times)

                # 计算剩余步数
                remaining_steps = self.total_steps - global_step

                # 计算预计剩余时间（小时）
                remaining_time_hours = (avg_time_per_step * remaining_steps) / 3600

                try:
                    # 计算当前吞吐量
                    speed: float = self.frequent * batch_size / time_elapsed
                    speed_total = speed * world_size
                except ZeroDivisionError:
                    speed = float("inf")
                    speed_total = float("inf")

                # 使用f-string格式化输出信息
                header = f"rank {speed:.2f} total {speed_total:.2f} its/s lr: {lr_scheduler.get_last_lr()[0]:.8f} "
                progress = f"step: {global_step}/{self.total_steps} ({global_step/self.total_steps*100:.2f}%) "
                time_info = f"remain: {remaining_time_hours:.2f} hours"

                loss_str_format = ""
                for head_id, name in enumerate(self.list_head_names):
                    # Add to TensorBoard if rank 0
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

                # 添加样本数信息到日志
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

    """
    打印并记录训练参数。
    - logger: 你的 logger 实例（支持 .info）
    - writer: TensorBoard SummaryWriter（可为 None）
    - save_dir: 额外保存 JSON 的路径（可为 None）
    - rank: 仅在 rank==0 执行（分布式时避免重复）
    """
    if rank != 0:
        return

    args_dict: Dict[str, Any] = vars(args) if not isinstance(args, dict) else args
    # 排序保证可重复
    sorted_items = sorted(args_dict.items(), key=lambda x: x[0])

    # Megatron-LM 风格日志
    sep = "-" * 92
    logger.info(sep)
    logger.info("Training / Runtime Arguments")
    logger.info(sep)
    # 你原本的格式是左对齐 30，这里模仿 Megatron 可右对齐或统一宽度
    # 下面采用 name: value 风格，也可以改成两列对齐
    max_key_len = max(len(k) for k, _ in sorted_items) if sorted_items else 0
    col_width = max(20, max_key_len)
    for k, v in sorted_items:
        # 避免太长的列表完全刷屏，可以截断（可选）
        vs = str(v)
        if len(vs) > 300:
            vs = vs[:297] + "..."
        logger.info(f"{k:<{col_width}} = {vs}")
    logger.info(sep)

    # ---------- TensorBoard 记录 ----------
    if writer is not None:
        # 1) 作为 hparams（会出现在 HPARAMS 面板）
        # 需要全部是 “简单” 类型；否则转换
        # hparam_dict = {}
        # for k, v in sorted_items:
        #     sv = _sanitize_for_json(v)
        #     # hparams 里放的要是 int / float / str / bool，复杂的就转成 str
        #     if isinstance(sv, (int, float, str, bool)) or sv is None:
        #         hparam_dict[k] = sv
        #     else:
        #         hparam_dict[k] = str(sv)
        # # add_hparams 需要一个 metrics dict；没有真实指标时给个空或 dummy
        # writer.add_hparams(hparam_dict, {"_dummy_metric": 0.0})

        # 2) 作为 Markdown 表格（TEXT 面板）
        md_lines = ["| Argument | Value |", "|----------|-------|"]
        for k, v in sorted_items:
            vs = str(v).replace("|", "\\|")
            if len(vs) > 500:
                vs = vs[:497] + "..."
            md_lines.append(f"| {k} | {vs} |")
        writer.add_text("markdown_table", "\n".join(md_lines), global_step=0)

        # 3) 纯文本 JSON（TEXT 面板）
        # json_blob = json.dumps({k: _sanitize_for_json(v) for k, v in sorted_items},
        #                        indent=2, ensure_ascii=False)
        # writer.add_text("args/json_full", f"```\n{json_blob}\n```", global_step=0)


    # 可选：记录一个时间戳
    # if writer is not None:
    #     writer.add_text("args/run_timestamp", datetime.utcnow().isoformat(), global_step=0)

if __name__ == "__main__":
    main()
