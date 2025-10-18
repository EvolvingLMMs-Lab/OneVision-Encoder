import argparse
import logging
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import torch
from timm import create_model
from torch import distributed
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter

from dataset import DATASET_REGISTRY, Property
from training.checkpoint_utils import load_checkpoint, save_checkpoint
from training.fused_partial_fc_v2 import CombinedMarginLoss, PartialFC_V2
from training.lr_scheduler import PolynomialLRWarmup
import model_factory

torch._dynamo.config.optimize_ddp = False

import argparse

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
parser.add_argument("--warmup_ratio", type=float, default=0.2, help="Warmup ratio of total training steps / 训练总步数的预热比例")
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
    """Unwraps a model from DistributedDataParallel if it is wrapped."""
    if hasattr(model, "module"):
        return model.module
    else:
        return model


def main():
    """Main training function."""
    global_step = 0

    # image_size 保持你原逻辑
    args.image_size = [int(x) for x in args.image_size.split(",")]
    if len(args.image_size) == 1:
        args.image_size = args.image_size * 2

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

    # 其余派生量
    args.batch_size = sum(args.list_batch_sizes)
    args.list_head_names = [x.name for x in args.list_datasets]
    args.total_steps = int(args.num_sampled_data / args.batch_size / world_size)

    for arg in vars(args):
        msg = f"{format(arg, '<30')}  {format(str(getattr(args, arg)))}"
        logger.info(msg)

    # Initialize models
    backbone = create_model(args.model_name).cuda().train()
    if args.init_backbone != "NULL":
        assert os.path.exists(args.init_backbone)
        state_dict = torch.load(args.init_backbone, "cpu")
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        backbone.load_state_dict(state_dict, strict=True)
        logger.info(f"Loaded backbone weights from {args.init_backbone}")

    backbone.requires_grad_(bool(args.finetune_backbone))
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
                raise

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

    backbone = wrap_ddp(backbone)

    list_dali_dataloader = []
    list_head_names = []
    for head_id, dataset_config in enumerate(args.list_datasets):
        if dataset_config.dali_type == "decord":
            from dataloader.data_decord_video import dali_dataloader

            train_iter = dali_dataloader(
                file_list=dataset_config.prefixes,
                label=dataset_config.label,
                dali_num_threads=2,
                dali_py_num_workers=8,
                batch_size=args.list_batch_sizes[head_id],
                input_size=args.image_size[0],
                sequence_length=args.num_frames,
                stride=8,
                seed=0+rank,
                short_side_size=256 / 224 * args.image_size[0],
                shard_id=dataset_config.shard_id,
                num_shards=dataset_config.num_shards)
        elif dataset_config.dali_type == "origin":

            from dataloader.data_v2 import MultiRecDALIWarper
            print("dataset_config.prefix", dataset_config.prefixes)
            train_iter = MultiRecDALIWarper(
                list_prefix=dataset_config.prefixes,
                batch_size=args.list_batch_sizes[head_id],
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
        num_samples += sum(args.list_batch_sizes) * world_size

        list_embedding = []
        list_batch_sizes = []
        for head_id, dataset_config in enumerate(args.list_datasets):

            dataset_config: Property
            if dataset_config.dali_type in ["decord", "origin"]:
                head_input = list_data_batch[head_id]["pixel_values"]
                list_batch_sizes.append(head_input.size(0))
                with torch.amp.autocast(dtype=torch.bfloat16, device_type="cuda"):
                    head_embedding = backbone(head_input)["head_output"]
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

        if is_accumulation_step and isinstance(backbone, torch.nn.parallel.DistributedDataParallel):
            # 中间累积步骤，避免DDP通信
            with backbone.no_sync():
                scaled_loss.backward()
        else:
            # 最后一步正常backward，会进行梯度同步
            scaled_loss.backward()

            # 只在累积完成时执行梯度裁剪和优化器更新
            if global_step % args.backward_passes_per_step == 0:
                clip_grad_norm_(backbone.parameters(), max_norm=5, norm_type=2)
                for pfc in list_module_pfc:
                    clip_grad_norm_(pfc.parameters(), max_norm=5, norm_type=2)
                opt.step()
                opt.zero_grad()

        # 学习率更新
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
            exit()
            logger.info(f"Training completed at step {global_step}")

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


def save_video_as_gif(tensor, path, fps=10, mean=None, std=None):
    import imageio
    """
    Save a video tensor as a GIF file with proper denormalization.
    
    Args:
        tensor: Tensor of shape [3, num_frames, height, width]
        path: Path to save the GIF
        fps: Frames per second for the GIF
        mean: Mean values used for normalization [R, G, B]
        std: Standard deviation values used for normalization [R, G, B]
    """


    # Make sure the directory exists
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    
    # Set default ImageNet mean and std if not provided
    if mean is None:
        mean = [x * 255 for x in [0.48145466, 0.4578275, 0.40821073]]
    if std is None:
        std = [x * 255 for x in [0.26862954, 0.26130258, 0.27577711]]
    
    # Convert mean and std to tensors with proper shape for broadcasting
    mean_tensor = torch.tensor(mean).view(3, 1, 1).to(tensor.device)
    std_tensor = torch.tensor(std).view(3, 1, 1).to(tensor.device)
    
    # Convert to numpy array and ensure it's in the right format for imageio
    frames = []
    for i in range(tensor.shape[1]):  # Iterate through frames
        # Properly denormalize: pixel = pixel * std + mean
        frame = tensor[:, i] * std_tensor + mean_tensor
        
        # Convert from [3, H, W] to [H, W, 3]
        frame = frame.permute(1, 2, 0).cpu().detach().numpy()
        
        # Clip values to valid range and convert to uint8
        frame = np.clip(frame, 0, 255).astype(np.uint8)
        frames.append(frame)
    
    # Save as GIF
    imageio.mimsave(path, frames, fps=fps)
    return path


def _sanitize_for_json(v):
    """尽量把值转成 JSON 可序列化类型；不行就转成字符串。"""
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, (list, tuple)):
        return [_sanitize_for_json(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _sanitize_for_json(val) for k, val in v.items()}
    # 其它（例如 Namespace、Path、自定义对象等）
    return str(v)

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
