import argparse
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from timm import create_model
from torch import distributed
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter

import model_factory
from dataset import DATASET_REGISTRY
from training.lr_scheduler import PolynomialLRWarmup

torch._dynamo.config.optimize_ddp = False

parser = argparse.ArgumentParser(description="Video Distillation Training Script")
parser.add_argument("--backward_passes_per_step", type=int, default=1)
parser.add_argument("--debug", type=int, default=0)

parser.add_argument("--list_batch_size", nargs='+', default=["128"])
parser.add_argument("--list_dataset", nargs='+', default=["distill_mlcd_coyo_laion"])

parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument("--image_size", default="224")
parser.add_argument("--num_sampled_data", type=int, default=1_000_000_000)
parser.add_argument("--output", default="output")
parser.add_argument("--output_decoder", default="output_decoder")

parser.add_argument("--init_encoder", default="")
parser.add_argument("--init_decoder", default="")

parser.add_argument("--frequent", type=int, default=10)
parser.add_argument("--warmup_ratio", type=float, default=0.1)
parser.add_argument("--weight_decay", type=float, default=0.05)
parser.add_argument("--workers", type=int, default=2)
parser.add_argument("--ckpt_interval", type=int, default=1000)

parser.add_argument("--mask_ratio", type=float, default=0.5,
                    help="Ratio of patches to mask during training")
parser.add_argument("--finetune_backbone", type=int, default=1)
parser.add_argument("--num_frames", type=int, default=8)

# Add visualization arguments
parser.add_argument("--visualize", type=int, default=0,
                    help="Save input videos as GIFs for visualization")
parser.add_argument("--vis_samples", type=int, default=2,
                    help="Number of samples to visualize per batch")
parser.add_argument("--vis_interval", type=int, default=10,
                    help="How often to save visualizations")

# Add the model name arguments
parser.add_argument("--model_name_encoder", default="pretrain_encoder_small_patch16_224_v10_08_rms",
                    help="Model name for the encoder architecture")
parser.add_argument("--model_name_decoder", default="mlcd_decoder_small_patch16_224_v10_08_rms",
                    help="Model name for the decoder architecture")
parser.add_argument("--model_name_teacher", default="mlcd_vit_s_16_512px",
                    choices=[
                        "mlcd_vit_b_16_512px",
                        "mlcd_vit_s_16_512px"],
                    help="Model name for the teacher architecture")

# 注：虽然模型名中包含512px分辨率，但由于RoPE机制，这些模型向下兼容较低分辨率，结果质量不会有显著差异
TEACHER_MODEL_PATH = {
    "mlcd_vit_b_16_512px": "/video_vit/pretrain_models/deepglint/mlcd/mlcd_vit_b_16_512px.pt",
    "mlcd_vit_s_16_512px": "/video_vit/pretrain_models/deepglint/mlcd/mlcd_vit_s_16_512px.pt",
}

args = parser.parse_args()

rank = int(os.getenv("RANK", "0"))
local_rank = int(os.getenv("LOCAL_RANK", "0"))
world_size = int(os.getenv("WORLD_SIZE", "1"))
distributed.init_process_group(backend="nccl")

torch.cuda.set_device(local_rank)
torch.backends.cudnn.benchmark = True

os.makedirs(args.output, exist_ok=True)

if rank == 0:
    log = logging.getLogger()
    formatter = logging.Formatter(f"rank-id:{rank:03d}:%(asctime)s-%(message)s")
    file_handler = logging.FileHandler(os.path.join(args.output, f"training_{rank:03d}.log"))
    stream_handler = logging.StreamHandler(sys.stdout)
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    log.addHandler(file_handler)
    log.addHandler(stream_handler)
    log.setLevel(logging.INFO)
else:
    log = logging.getLogger()
    formatter = logging.Formatter(f"rank-id:{rank:03d}:%(asctime)s-%(message)s")
    file_handler = logging.FileHandler(os.path.join(args.output, f"training_{rank:03d}.log"))
    file_handler.setFormatter(formatter)
    log.addHandler(file_handler)
    log.setLevel(logging.INFO)


def unwrap_module(model):
    """Unwraps a model from DistributedDataParallel if it is wrapped."""
    if hasattr(model, "module"):
        return model.module
    else:
        return model


def save_checkpoint(output_dir, encoder, decoder, optimizer, lr_scheduler, global_step, keep_num=5):
    """Save model checkpoint with encoder and decoder models."""
    if rank != 0:
        return

    os.makedirs(output_dir, exist_ok=True)
    
    # Save encoder
    encoder_path = os.path.join(output_dir, f"encoder_checkpoint_{global_step}.pt")
    encoder_state_dict = unwrap_module(encoder).state_dict()
    torch.save(encoder_state_dict, encoder_path)
    
    # Save decoder
    decoder_path = os.path.join(output_dir, f"decoder_checkpoint_{global_step}.pt")
    decoder_state_dict = unwrap_module(decoder).state_dict()
    torch.save(decoder_state_dict, decoder_path)
    
    # Save optimizer and scheduler state
    optim_path = os.path.join(output_dir, f"optimizer_{global_step}.pt")
    checkpoint = {
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'global_step': global_step,
    }
    torch.save(checkpoint, optim_path)
    
    # Keep only the most recent checkpoints
    log.info(f"Saved checkpoint at step {global_step}")
    
    # Clean up old checkpoints to maintain only keep_num
    all_checkpoints = []
    for file in os.listdir(output_dir):
        if file.startswith("encoder_checkpoint_") and file.endswith(".pt"):
            step = int(file.split("_")[-1].split(".")[0])
            all_checkpoints.append(step)
    
    if len(all_checkpoints) > keep_num:
        all_checkpoints.sort()
        for step in all_checkpoints[:-keep_num]:
            for prefix in ["encoder_checkpoint_", "decoder_checkpoint_", "optimizer_"]:
                file_path = os.path.join(output_dir, f"{prefix}{step}.pt")
                if os.path.exists(file_path):
                    os.remove(file_path)


def load_checkpoint(output_dir, encoder, decoder, optimizer, lr_scheduler):
    """Load model checkpoint for both encoder and decoder."""
    # Find the latest checkpoint
    latest_step = -1
    for file in os.listdir(output_dir):
        if file.startswith("encoder_checkpoint_") and file.endswith(".pt"):
            step = int(file.split("_")[-1].split(".")[0])
            if step > latest_step:
                latest_step = step

    if latest_step == -1:
        log.info("No checkpoint found, starting from scratch")
        return None
    
    # Load encoder
    encoder_path = os.path.join(output_dir, f"encoder_checkpoint_{latest_step}.pt")
    if os.path.exists(encoder_path):
        encoder_state_dict = torch.load(encoder_path, map_location="cpu")
        unwrap_module(encoder).load_state_dict(encoder_state_dict)
        log.info(f"Loaded encoder checkpoint from step {latest_step}")
    
    # Load decoder
    decoder_path = os.path.join(output_dir, f"decoder_checkpoint_{latest_step}.pt")
    if os.path.exists(decoder_path):
        decoder_state_dict = torch.load(decoder_path, map_location="cpu")
        unwrap_module(decoder).load_state_dict(decoder_state_dict)
        log.info(f"Loaded decoder checkpoint from step {latest_step}")
    
    # Load optimizer and scheduler state
    optim_path = os.path.join(output_dir, f"optimizer_{latest_step}.pt")
    if os.path.exists(optim_path) and optimizer is not None:
        checkpoint = torch.load(optim_path, map_location="cpu")
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        log.info(f"Loaded optimizer and scheduler from step {latest_step}")
    
    return {'global_step': latest_step}


def main():
    """Main training function."""
    global_step = 0
    log = logging.getLogger()
    args.image_size = [int(x) for x in args.image_size.split(",")]
    if len(args.image_size) == 1:
        args.image_size = args.image_size * 2

    args.list_dataset = [
        DATASET_REGISTRY.get(x)() for x in args.list_dataset]

    args.num_head = len(args.list_dataset)
    args.list_batch_size = [int(x) for x in args.list_batch_size]
    args.batch_size = sum(args.list_batch_size)
    args.list_head_name = [x.name for x in args.list_dataset]
    args.total_steps = int(args.num_sampled_data / args.batch_size / world_size)

    for arg in vars(args):
        msg = f"{format(arg, '<30')}  {format(str(getattr(args, arg)))}"
        log.info(msg)

    # Initialize models
    llava_vit_encoder = create_model(args.model_name_encoder).cuda().train()
    llava_vit_decoder = create_model(args.model_name_decoder).cuda().train()
    # llava_vit_encoder = torch.compile(llava_vit_encoder)
    # llava_vit_decoder = torch.compile(llava_vit_decoder)

    # Initialize teacher model and load pre-trained weights
    llava_vit_teacher = create_model("mlcd_rope2d_vit_s_16").cuda().eval()
    log.info(f"Loading teacher model from {TEACHER_MODEL_PATH[args.model_name_teacher]}")
    state_dict = torch.load(TEACHER_MODEL_PATH[args.model_name_teacher], "cpu")
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    llava_vit_teacher.load_state_dict(state_dict, strict=True)

    # Freeze teacher model parameters
    for param in llava_vit_teacher.parameters():
        param.requires_grad = False

    # Load initial weights for encoder and decoder if specified
    if args.init_encoder:
        log.info(f"Initializing encoder from {args.init_encoder}")
        state_dict = torch.load(args.init_encoder, "cpu")
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        llava_vit_encoder.load_state_dict(state_dict, strict=False)

    if args.init_decoder:
        log.info(f"Initializing decoder from {args.init_decoder}")
        state_dict = torch.load(args.init_decoder, "cpu")
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        llava_vit_decoder.load_state_dict(state_dict, strict=False)

    # Set up optimizer
    parameters = [
        {"params": llava_vit_encoder.parameters()},
        {"params": llava_vit_decoder.parameters()}
    ]

    optimizer_cls = torch.optim.AdamW
    optimizer = optimizer_cls(parameters, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = PolynomialLRWarmup(
        optimizer, int(args.total_steps * args.warmup_ratio), args.total_steps, 2
    )

    # Try to load checkpoint if exists
    checkpoint_result = load_checkpoint(args.output, llava_vit_encoder, llava_vit_decoder, optimizer, lr_scheduler)
    if checkpoint_result is not None:
        global_step = checkpoint_result['global_step']
        log.info(f"Resuming from step {global_step}")

    # Wrap models with DDP
    def wrap_ddp(model):
        return torch.nn.parallel.DistributedDataParallel(
            module=model, broadcast_buffers=False, device_ids=[local_rank],
            bucket_cap_mb=32, find_unused_parameters=True, static_graph=True)

    llava_vit_encoder_ddp = wrap_ddp(llava_vit_encoder)
    llava_vit_decoder_ddp = wrap_ddp(llava_vit_decoder)

    # Define loss function
    mse_loss = torch.nn.MSELoss().cuda()

    # Set up data loaders
    list_dali_dataloader = []
    list_head_name = []

    for head_id, dataset_config in enumerate(args.list_dataset):
        if args.debug:

            from dataloader.data_v2_video import SyntheticDataIter
            train_iter = SyntheticDataIter(args.batch_size, args.image_size[0], local_rank)
        elif dataset_config.dali_type == "decord":

            from dataloader.data_decord_video import dali_dataloader
            num_workers = 4
            train_iter = dali_dataloader(
                file_list=dataset_config.prefix,
                dali_num_threads=2,
                dali_py_num_workers=num_workers,
                batch_size=args.list_batch_size[head_id],
                input_size=args.image_size[0],
                sequence_length=args.num_frames,
                stride=8,
                seed=0 + rank,
                short_side_size=int(256 / 224 * args.image_size[0]),
                shard_id=dataset_config.shard_id,
                num_shards=dataset_config.num_shards,
            )
        elif dataset_config.dali_type == "origin":

            from dataloader.data_v2 import MultiRecDALIWarper
            print("dataset_config.prefix", dataset_config.prefix)
            train_iter = MultiRecDALIWarper(
                list_prefix=dataset_config.prefix,
                batch_size=args.list_batch_size[head_id],
                image_size=args.image_size,
                workers=args.workers,
                shard_id=dataset_config.shard_id,
                num_shards=dataset_config.num_shards
        )
        else:
            raise NotImplementedError(f"Dataloader type {dataset_config.dali_type} not implemented")

        list_dali_dataloader.append(train_iter)
        list_head_name.append(dataset_config.name)

    if rank == 0:
        tb_writer = SummaryWriter(log_dir=f"{args.output}/tensorboard")
    else:
        tb_writer = None
    # Initialize callback for logging
    batch_end_callback = BatchEndCallBack(
        frequent=args.frequent,
        list_head_name=list_head_name,
        output=args.output,
        total_steps=args.total_steps,
        tb_writer=tb_writer,
    )
    log_args(args, log, writer=tb_writer, save_dir=args.output, rank=rank)

    # Prepare for training loop
    list_iter = []
    list_next_data_batch = []
    for i in range(args.num_head):
        list_iter.append(iter(list_dali_dataloader[i]))
        try:
            list_next_data_batch.append(next(list_iter[i]))
        except StopIteration:
            list_dali_dataloader[i].reset()
            list_iter[i] = iter(list_dali_dataloader[i])
            list_next_data_batch.append(next(list_iter[i]))

    if global_step >= args.total_steps:
        log.info("Training already completed (global_step >= total_steps)")
        return

    num_samples = 0
    # Main training loop
    while global_step < args.total_steps:
        list_data_batch = list_next_data_batch
        list_data = [x[0].cuda() for x in list_data_batch]
        list_loss = []
        list_loss_float = []
        num_samples += sum(args.list_batch_size) * world_size

        for head_id, dataset_config in enumerate(args.list_dataset):
            head_input = list_data[head_id]
            
            with torch.amp.autocast(dtype=torch.bfloat16, device_type="cuda"):
                # Run encoder with masking

                enc_out = llava_vit_encoder_ddp(head_input, mask_ratio=args.mask_ratio)

                # Extract encoder outputs
                visible_embeddings = enc_out["visible_embeddings"]     # (B, N_vis, D)

                # Run decoder
                dec_out = llava_vit_decoder_ddp(visible_embeddings)

                decoded_full = dec_out["decoded_full"]  # Full sequence of decoded tokens

            # Get teacher model output
            with torch.no_grad():
                # Get teacher embeddings and reshape back
                teacher_output = llava_vit_teacher(head_input)[:, 1:, :]  # Skip CLS token

                # Shift teacher output by removing the first frame for proper supervision
                # Only compute loss on the first (n-1) frames
            with torch.amp.autocast(dtype=torch.bfloat16, device_type="cuda"):
                loss = mse_loss(decoded_full, teacher_output)

            list_loss.append(loss)
            list_loss_float.append(loss.float())

        # Compute total loss and backward pass
        total_loss = sum(list_loss)
        optimizer.zero_grad()
        total_loss.backward()

        # Apply gradient clipping and step optimizer
        clip_grad_norm_(llava_vit_encoder.parameters(), max_norm=1, norm_type=2)
        clip_grad_norm_(llava_vit_decoder.parameters(), max_norm=1, norm_type=2)
        optimizer.step()
        lr_scheduler.step()

        # Update progress and metrics
        batch_end_callback(
            global_step=global_step,
            lr_scheduler=lr_scheduler,
            list_loss_float=list_loss_float,
            batch_size=args.batch_size,
            num_samples=num_samples
        )

        global_step += 1

        # Save checkpoint periodically
        if global_step % args.ckpt_interval == 0:
            save_checkpoint(
                args.output,
                llava_vit_encoder_ddp,
                llava_vit_decoder_ddp,
                optimizer,
                lr_scheduler,
                global_step
            )

        # Get next batch
        for i in range(args.num_head):
            try:
                list_next_data_batch[i] = next(list_iter[i])
            except StopIteration:
                list_dali_dataloader[i].reset()
                list_iter[i] = iter(list_dali_dataloader[i])
                list_next_data_batch[i] = next(list_iter[i])

    # Save final checkpoint
    save_checkpoint(
        args.output,
        llava_vit_encoder_ddp,
        llava_vit_decoder_ddp,
        optimizer,
        lr_scheduler,
        global_step
    )
    log.info(f"Training completed at step {global_step}")


class BatchEndCallBack(object):
    def __init__(
        self,
        frequent: int,
        list_head_name: List[str],
        output: str,
        total_steps: int,
        tb_writer = None,
    ):
        self.frequent: int = frequent
        self.list_head_name: List[str] = list_head_name
        self.output: str = output
        self.total_steps: int = total_steps

        self.num_head = len(self.list_head_name)
        self.time_start = time.time()
        self.list_loss_metric = [ScalaMetric() for _ in self.list_head_name]  # 只保留一个loss
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
                for head_id, name in enumerate(self.list_head_name):
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
                examples_info = f"samples: {self.total_examples}"
                msg = f"{header}{progress}{time_info} {examples_info}{loss_str_format}"

                if rank == 0:
                    logging.info(msg)
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
    - logger: 你的 log 实例（支持 .info）
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