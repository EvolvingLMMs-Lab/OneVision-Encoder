import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import argparse
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from timm import create_model
import numpy as np
import torch
from torch import distributed
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter

from dataset import DATASET_REGISTRY, Property
import model_factory
from training.fused_partial_fc_v2 import  PartialFC_V2, CombinedMarginLoss
from training.lr_scheduler import PolynomialLRWarmup

# --- helper: robustly resolve per-head pfc_type from dataset_config ---
def _resolve_pfc_type(cfg, head_idx: int = 0):
    """
    Normalize pfc_type access for different config styles:
    - cfg.pfc_type: str or callable -> str
    - cfg.pfc_types: List[str] or str
    - dict-like: keys 'pfc_type' or 'pfc_types'
    Returns: str or None
    """
    # Direct attribute 'pfc_type'
    try:
        v = getattr(cfg, "pfc_type")
        if callable(v):
            v = v()
        if v is not None:
            return v
    except Exception:
        pass
    # Plural attribute 'pfc_types'
    try:
        vs = getattr(cfg, "pfc_types")
        if callable(vs):
            vs = vs()
        if isinstance(vs, (list, tuple)):
            if len(vs) == 0:
                return None
            return vs[head_idx] if head_idx < len(vs) else vs[0]
        if isinstance(vs, str):
            return vs
    except Exception:
        pass
    # Dict-like access
    try:
        if isinstance(cfg, dict):
            v = cfg.get("pfc_type", None)
            if v is None:
                v = cfg.get("pfc_types", None)
            if isinstance(v, (list, tuple)):
                return v[head_idx] if head_idx < len(v) else (v[0] if v else None)
            return v
        if hasattr(cfg, "get"):
            v = cfg.get("pfc_type", None)
            if v is None:
                v = cfg.get("pfc_types", None)
            if isinstance(v, (list, tuple)):
                return v[head_idx] if head_idx < len(v) else (v[0] if v else None)
            return v
    except Exception:
        pass
    return None

# --- helper: robustly resolve num_classes from dataset_config ---
def _resolve_num_classes(cfg):
    """
    Normalize num_classes access for different config styles:
    - cfg.num_class (singular) or cfg.num_classes (plural)
    - dict-like: 'num_class' or 'num_classes'
    Returns: int or raises ValueError if not found.
    """
    # Attribute access first (singular then plural)
    for key in ("num_class", "num_classes"):
        try:
            v = getattr(cfg, key)
            if callable(v):
                v = v()
            if v is not None:
                return int(v)
        except Exception:
            pass
    # Dict-like fallback
    try:
        if isinstance(cfg, dict):
            for key in ("num_class", "num_classes"):
                if key in cfg and cfg[key] is not None:
                    return int(cfg[key])
        if hasattr(cfg, "get"):
            for key in ("num_class", "num_classes"):
                v = cfg.get(key, None)
                if v is not None:
                    return int(v)
    except Exception:
        pass
    raise ValueError("Could not resolve 'num_classes' from dataset_config; expected attribute 'num_class' or 'num_classes'.")

# --- helper: robustly resolve file list prefix from dataset_config ---
def _resolve_prefix(cfg, head_idx: int = 0):
    """
    Normalize access to dataset file list(s):
    - cfg.prefix: str
    - cfg.prefixes: List[str] or str
    - dict-like: 'prefix' or 'prefixes'
    Returns: str or None
    """
    # Attribute: singular
    try:
        v = getattr(cfg, "prefix")
        if callable(v):
            v = v()
        if isinstance(v, str) and len(v) > 0:
            return v
    except Exception:
        pass
    # Attribute: plural
    try:
        vs = getattr(cfg, "prefixes")
        if callable(vs):
            vs = vs()
        if isinstance(vs, (list, tuple)):
            if len(vs) == 0:
                return None
            return vs[head_idx] if head_idx < len(vs) else vs[0]
        if isinstance(vs, str) and len(vs) > 0:
            return vs
    except Exception:
        pass
    # Dict-like
    try:
        if isinstance(cfg, dict):
            if "prefix" in cfg and isinstance(cfg["prefix"], str) and len(cfg["prefix"]) > 0:
                return cfg["prefix"]
            if "prefixes" in cfg:
                vs = cfg["prefixes"]
                if isinstance(vs, (list, tuple)) and len(vs) > 0:
                    return vs[head_idx] if head_idx < len(vs) else vs[0]
                if isinstance(vs, str) and len(vs) > 0:
                    return vs
        if hasattr(cfg, "get"):
            v = cfg.get("prefix", None)
            if isinstance(v, str) and len(v) > 0:
                return v
            vs = cfg.get("prefixes", None)
            if isinstance(vs, (list, tuple)) and len(vs) > 0:
                return vs[head_idx] if head_idx < len(vs) else vs[0]
            if isinstance(vs, str) and len(vs) > 0:
                return vs
    except Exception:
        pass
    return None

# --- helper: robustly resolve dali_type from dataset_config ---
def _resolve_dali_type(cfg, head_idx: int = 0, default: str = "decord"):
    """
    Normalize dali_type access for different config styles:
    - cfg.dali_type: str or callable
    - cfg.dali_types: List[str] or str
    - dict-like: 'dali_type' / 'dali_types'
    Returns lower-cased string; falls back to `default` if not present.
    """
    # Attribute: singular
    try:
        v = getattr(cfg, "dali_type")
        if callable(v):
            v = v()
        if isinstance(v, str) and v:
            return v.lower()
    except Exception:
        pass
    # Attribute: plural
    try:
        vs = getattr(cfg, "dali_types")
        if callable(vs):
            vs = vs()
        if isinstance(vs, (list, tuple)) and len(vs) > 0:
            return str(vs[head_idx] if head_idx < len(vs) else vs[0]).lower()
        if isinstance(vs, str) and vs:
            return vs.lower()
    except Exception:
        pass
    # Dict-like
    try:
        if isinstance(cfg, dict):
            v = cfg.get("dali_type") or cfg.get("dali_types")
            if isinstance(v, (list, tuple)) and len(v) > 0:
                return str(v[head_idx] if head_idx < len(v) else v[0]).lower()
            if isinstance(v, str) and v:
                return v.lower()
        if hasattr(cfg, "get"):
            v = cfg.get("dali_type", None)
            if isinstance(v, str) and v:
                return v.lower()
            vs = cfg.get("dali_types", None)
            if isinstance(vs, (list, tuple)) and len(vs) > 0:
                return str(vs[head_idx] if head_idx < len(vs) else vs[0]).lower()
            if isinstance(vs, str) and vs:
                return vs.lower()
    except Exception:
        pass
    return default.lower()

torch._dynamo.config.optimize_ddp = False

parser = argparse.ArgumentParser(description="")
parser.add_argument("--backward_passes_per_step", type=int, default=1)
parser.add_argument("--debug", type=int, default=0)
parser.add_argument("--dataloader-type", default="dali")
parser.add_argument("--dali_is_training", default=1, type=int)
parser.add_argument("--list_batch_size", nargs='+', default=["128"])
parser.add_argument("--list_dataset", nargs='+', default=["laion400m"])
parser.add_argument("--list_filter", nargs='+', default=["0"])
parser.add_argument("--list_margin", nargs='+', default=["0.4"])
parser.add_argument("--list_sample_rate", nargs='+', default=["1"])
parser.add_argument("--list_lr_pfc_weight", nargs='+', default=["1"])

parser.add_argument("--embedding_size", type=int, default=512)
parser.add_argument("--gradient_checkpoint", type=int, default=0)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument("--input_gray", type=int, default=0)
parser.add_argument("--image_size", default="224")
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--num_epochs", type=float, default=3)
parser.add_argument("--num_sampled_data", type=int, default=1000000)
parser.add_argument("--model_name", default="ViT-B/16")
parser.add_argument("--model_weight", default=None)
parser.add_argument("--opt", default="adamw")
parser.add_argument("--output", default="output")
parser.add_argument(
    "--random_diff",
    type=int,
    default=10,
)
parser.add_argument("--init_dir", default="NULL")
parser.add_argument("--init_mode", default="NULL")

parser.add_argument("--init_backbone", default="NULL")
parser.add_argument("--list_init_partial_fc", nargs='+', default=("NULL", ))

parser.add_argument("--repeat_pfc", type=int, default=0)
parser.add_argument("--save_pfc", type=int, default=1)
parser.add_argument("--frequent", type=int, default=100)
parser.add_argument("--warmup_ratio", type=float, default=0.2)
parser.add_argument("--weight_decay", type=float, default=0.05)
parser.add_argument("--weight_decay_pfc", type=float, default=0.05)
parser.add_argument("--workers", type=int, default=2)
parser.add_argument("--ckpt_interval", type=int, default=200)

parser.add_argument("--mask", type=int, default=0)

parser.add_argument("--finetune_backbone", type=int, default=1)
parser.add_argument("--num_frames", type=int, default=16)
# parser.add_argument("--finetune_pixel_shuffle_encoder", type=int, default=0)
# parser.add_argument("--finetune_partial_fc", type=int, default=1)

# Add to parser arguments
parser.add_argument("--visualize", type=int, default=0, help="Save input videos as GIFs for visualization")
parser.add_argument("--vis_samples", type=int, default=2, help="Number of samples to visualize per batch")
parser.add_argument("--vis_interval", type=int, default=10, help="How often to save visualizations")

# --- mask debug CLI flags ---
parser.add_argument("--mask_debug_only", type=int, default=0,
                    help="If 1, bypass PartialFC/labels and only run forward to verify I/P mask pipeline.")
parser.add_argument("--max_debug_steps", type=int, default=50,
                    help="When mask_debug_only=1, stop after this many steps.")
parser.add_argument("--mask_log_topk", type=int, default=2,
                    help="How many samples to log mask ratios for (per step) in debug mode.")

parser.add_argument("--list_loss_weight", nargs='+', default=["1"])
parser.add_argument("--force_dali_type", type=str, default=None,
                    help="Override dataset_config dali_type for all heads; e.g., decord_ip / decord. CLI has precedence over env FORCE_DALI_TYPE.")

args = parser.parse_args()

# --- Normalize list-like arguments and cast numerics as needed ---
def _as_list(v):
    # Normalize argparse outputs: if user didn't pass and default is a string, wrap into a list
    return v if isinstance(v, (list, tuple)) else [v]

# Normalize list-like args early to avoid iterating over characters of a string (e.g., "0.4" -> ["0.4"])
args.list_dataset        = _as_list(args.list_dataset)
args.list_batch_size     = [int(x) for x in _as_list(args.list_batch_size)]
args.list_sample_rate    = [float(x) for x in _as_list(args.list_sample_rate)]
args.list_margin         = [float(x) for x in _as_list(args.list_margin)]
args.list_filter         = [float(x) for x in _as_list(args.list_filter)]
args.list_lr_pfc_weight  = [float(x) for x in _as_list(args.list_lr_pfc_weight)]
args.list_loss_weight    = [float(x) for x in _as_list(args.list_loss_weight)]
args.list_init_partial_fc = _as_list(args.list_init_partial_fc)

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


def save_checkpoint(output_dir, model, optimizer, lr_scheduler, global_step, keep_num=5):
    """Save model checkpoint with encoder and decoder models."""
    if rank != 0:
        return

    os.makedirs(output_dir, exist_ok=True)
    
    # Save encoder
    model_path = os.path.join(output_dir, f"model_checkpoint_{global_step}.pt")
    model_state_dict = unwrap_module(model).state_dict()
    torch.save(model_state_dict, model_path)
    
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
        if file.startswith("model_checkpoint_") and file.endswith(".pt"):
            step = int(file.split("_")[-1].split(".")[0])
            all_checkpoints.append(step)
    
    if len(all_checkpoints) > keep_num:
        all_checkpoints.sort()
        for step in all_checkpoints[:-keep_num]:
            for prefix in ["model_checkpoint_", "optimizer_"]:
                file_path = os.path.join(output_dir, f"{prefix}{step}.pt")
                if os.path.exists(file_path):
                    os.remove(file_path)


def load_checkpoint(output_dir, model, optimizer, lr_scheduler):
    """Load model checkpoint for both encoder and decoder."""
    # Find the latest checkpoint
    latest_step = -1
    for file in os.listdir(output_dir):
        if file.startswith("model_checkpoint_") and file.endswith(".pt"):
            step = int(file.split("_")[-1].split(".")[0])
            if step > latest_step:
                latest_step = step
    
    if latest_step == -1:
        log.info("No checkpoint found, starting from scratch")
        return None
    
    # Load model
    model_path = os.path.join(output_dir, f"model_checkpoint_{latest_step}.pt")
    if os.path.exists(model_path):
        model_state_dict = torch.load(model_path, map_location="cpu")
        unwrap_module(model).load_state_dict(model_state_dict)
        log.info(f"Loaded encoder checkpoint from step {latest_step}")
    
    
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

    # Build dataset objects from normalized names
    args.list_dataset = [DATASET_REGISTRY.get(x)() for x in args.list_dataset]
    args.num_head     = len(args.list_dataset)
    args.batch_size   = sum(args.list_batch_size)
    args.list_head_name = [x.name for x in args.list_dataset]
    args.total_steps  = int(args.num_sampled_data / args.batch_size / world_size)

    # Broadcast singletons per-head if needed
    def _broadcast_per_head(v, n):
        return v if len(v) == n else (v * n if len(v) == 1 else v)
    args.list_sample_rate   = _broadcast_per_head(args.list_sample_rate,   args.num_head)
    args.list_margin        = _broadcast_per_head(args.list_margin,        args.num_head)
    args.list_filter        = _broadcast_per_head(args.list_filter,        args.num_head)
    args.list_lr_pfc_weight = _broadcast_per_head(args.list_lr_pfc_weight, args.num_head)
    args.list_loss_weight   = _broadcast_per_head(args.list_loss_weight,   args.num_head)

    for arg in vars(args):
        msg = f"{format(arg, '<30')}  {format(str(getattr(args, arg)))}"
        log.info(msg)

    # Initialize models
    backbone = create_model(args.model_name).cuda().train()
    if args.init_backbone != "NULL":
        assert os.path.exists(args.init_backbone)
        state_dict = torch.load(args.init_backbone, "cpu")
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        backbone.load_state_dict(state_dict, strict=False)
    
    backbone.requires_grad_(bool(args.finetune_backbone))
    backbone_parameters = filter(lambda p: p.requires_grad, backbone.parameters())

    dict_pfc_modules = {}
    list_module_pfc = []
    parameters: List[dict] = [
        {"params": backbone_parameters},
    ]

    for head_id, _ in enumerate(range(args.num_head)):
        head_name = args.list_head_name[head_id]
        dataset_config = args.list_dataset[head_id]
        dataset_config: Property
        pfc_type = _resolve_pfc_type(dataset_config, head_id)

        if int(args.mask_debug_only) == 1:
            # Skip building PartialFC in mask-only debug mode
            log.info(f"[mask-debug] skipping PFC construction for head='{head_name}'")
            continue

        if pfc_type != "partial_fc":
            log.warning(f"[head:{head_name}] pfc_type={pfc_type} not 'partial_fc'; defaulting to partial_fc")

        margin_loss = CombinedMarginLoss(
            64, 1, 0, args.list_margin[head_id], args.list_filter[head_id]
        )

        # Resolve number of classes robustly (supports num_class / num_classes / dict/Property)
        num_classes = _resolve_num_classes(dataset_config)

        partial_fc = PartialFC_V2(
            margin_loss,
            args.embedding_size,
            num_classes,
            args.list_sample_rate[head_id],
            fp16=False,
        )

        partial_fc.train().cuda()
        list_module_pfc.append(partial_fc)
        dict_pfc_modules[head_name] = partial_fc

        lr_pfc = args.lr * args.list_lr_pfc_weight[head_id]
        parameters.append(
            {
                "params": partial_fc.parameters(),
                "lr": lr_pfc,
                "weight_decay": args.weight_decay_pfc,
            }
        )

        init_partial_fc = args.list_init_partial_fc[head_id]
        if init_partial_fc != "NULL":
            init_partial_fc = init_partial_fc % rank
            log.info(f"init_partial_fc: {init_partial_fc}")
            if os.path.exists(init_partial_fc):
                if init_partial_fc.endswith(".npy"):
                    _weight = torch.from_numpy(np.load(init_partial_fc)).cuda()
                    partial_fc.weight = torch.nn.Parameter(_weight)
                elif init_partial_fc.endswith(".pt"):
                    _weight = torch.load(init_partial_fc, "cpu")
                    partial_fc.load_state_dict(_weight, strict=True)
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
        backbone,
        opt,
        lr_scheduler,
    )
    if result is not None:
        global_step = result['global_step']
        log.info(f"Resuming from step {global_step}")
    else:
        global_step = 0

    def wrap_ddp(model):
        return torch.nn.parallel.DistributedDataParallel(
            module=model, broadcast_buffers=False, device_ids=[local_rank],
            bucket_cap_mb=32, find_unused_parameters=True, static_graph=True)
    
    backbone = wrap_ddp(backbone)

    list_dali_dataloader = []
    list_head_name = []
    list_head_dali_type = []  # keep the resolved dali type per head for later use
    for head_id, dataset_config in enumerate(args.list_dataset):
        # Resolve dali_type with optional global override (CLI first, then ENV)
        force_cli = (getattr(args, "force_dali_type", None) or os.environ.get("FORCE_DALI_TYPE") or "").strip().lower()
        dali_type = _resolve_dali_type(dataset_config, head_idx=head_id, default="decord_ip")
        if force_cli:
            dali_type = force_cli
        # Persist back so downstream code that inspects dataset_config.dali_type sees the same value
        try:
            setattr(dataset_config, "dali_type", dali_type)
        except Exception:
            pass
        file_list = _resolve_prefix(dataset_config, head_idx=head_id)
        if file_list is None:
            raise ValueError(f"[head:{getattr(dataset_config, 'name', head_id)}] Cannot resolve dataset file list: expected 'prefix' or 'prefixes'")
        head_name = getattr(dataset_config, 'name', f'h{head_id}')
        logging.getLogger().info(f"[data] head={head_name} dali_type={dali_type} file_list={file_list}")
        list_head_dali_type.append(dali_type)

        # Map a few common aliases to concrete loaders
        def _looks_like_mxrec_path(p: str) -> bool:
            """Heuristic: return True if `p` is a .rec/.idx file or a dir containing such files."""
            try:
                if os.path.isdir(p):
                    ls = os.listdir(p)
                    has_rec = any(x.endswith(".rec") for x in ls)
                    has_idx = any(x.endswith(".idx") for x in ls)
                    return has_rec and has_idx
                return p.endswith(".rec") or p.endswith(".idx")
            except Exception:
                return False

        if dali_type in ["decord", "decord_v1"]:
            from dataloader.data_decord_video_v1 import dali_dataloader
        elif dali_type in ["decord_ip", "decord_ip_v1", "ip", "ip_v1"]:
            # IP/HEVC features path (video)
            from dataloader.data_decord_video_IP_v1 import dali_dataloader
        elif dali_type in ["mxrec", "origin", "laion", "coyo"] or _looks_like_mxrec_path(file_list):
            # MXNet RecordIO image shards (e.g., COYO/LAION). Try image loader if present.
            try:
                from dataloader.data_mxrec_image_v1 import dali_dataloader  # <- expected image loader
                logging.getLogger().info(f"[data] head={head_name} using MXNet-REC loader for {file_list}")
            except ImportError as e:
                raise ValueError(
                    f"[data] Detected MXNet RecordIO dataset at '{file_list}' but image loader 'dataloader/data_mxrec_image_v1.py' was not found.\n"
                    f"Please add an MXNet-REC DALI loader (fn.readers.mxnet(index_path=[...], path=[...])) or switch this head to a video dataset.\n"
                    f"Temporarily, you can run only the video head by dropping the COYO/LAION head from --list_dataset.") from e
        else:
            raise ValueError(f"dataset_config.dali_type {dali_type} not support!")

        train_iter = dali_dataloader(
            file_list=file_list,
            dali_num_threads=2,
            dali_py_num_workers=4,
            batch_size=args.list_batch_size[head_id],
            input_size=args.image_size[0],
            sequence_length=args.num_frames,
            stride=8,
            seed=0+rank,
            short_side_size=256 / 224 * args.image_size[0],
            shard_id=dataset_config.shard_id,
            num_shards=dataset_config.num_shards
        )

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

    list_iter = []
    list_next_data_batch = []
    for i in range(args.num_head):
        # list_dali_dataloader[i].reset()
        list_iter.append(iter(list_dali_dataloader[i]))
        list_next_data_batch.append(next(list_iter[i]))

    if global_step > args.total_steps:
        log.info("global_step > total_steps")
        exit()

    num_samples = 0
    end_of_batch = False
    while not end_of_batch:
        list_data_batch = list_next_data_batch
        num_samples += sum(args.list_batch_size) * world_size

        list_embedding = []
        list_batch_size = []
        for head_id, dataset_config in enumerate(args.list_dataset):
            dataset_config: Property
            dali_t = list_head_dali_type[head_id]
            # Debug: on the very first step, log available keys from the loader
            if rank == 0 and global_step == 0:
                try:
                    logging.getLogger().info(f"[data] head={list_head_name[head_id]} batch_keys={list(list_data_batch[head_id].keys())}")
                except Exception:
                    pass

            if dali_t in ["decord", "decord_v1", "decord_ip", "decord_ip_v1", "ip", "ip_v1"]:
                head_input = list_data_batch[head_id]["pixel_values"]
                # print("head_input", head_input.shape)
                list_batch_size.append(head_input.size(0))

                # --- mask debug: log res_zero_masks ratio for a few samples ---
                if int(args.mask_debug_only) == 1 and head_id == 0 and rank == 0:
                    try:
                        rz = list_data_batch[head_id].get("res_zero_masks", None)
                        if rz is not None:
                            # rz: (B, 1, T, H, W) or (B, T, H, W)
                            if rz.dim() == 5:
                                B, C1, T, H, W = rz.shape
                                rz_view = rz[:, 0]  # (B,T,H,W)
                            elif rz.dim() == 4:
                                B, T, H, W = rz.shape
                                rz_view = rz
                            else:
                                rz_view = None
                            if rz_view is not None:
                                topk = min(int(args.mask_log_topk), rz_view.size(0))
                                # Per-sample, per-frame masked fraction (1 means fully masked if mask==1 denotes masked)
                                # We log only first frame for brevity
                                ratios = (rz_view[:, 0].float().mean(dim=(1, 2))).detach().cpu().numpy()  # (B,)
                                ratios_str = ", ".join([f"{ratios[i]:.3f}" for i in range(topk)])
                                logging.getLogger().info(f"[mask] ratios(first-frame, first {topk} samples): {ratios_str}")
                        else:
                            logging.getLogger().warning("[mask] res_zero_masks not found in batch keys")
                    except Exception as _e:
                        logging.getLogger().warning(f"[mask] ratio log failed: {_e}")

                with torch.amp.autocast(dtype=torch.bfloat16, device_type="cuda"):
                    head_embedding = backbone(head_input)["head_output"]
                head_embedding = head_embedding.float()
            else:
                raise ValueError(f"Unsupported dali_type at train step: {dali_t}")

            list_embedding.append(head_embedding)

        list_loss = []
        list_loss_float = []

        if int(args.mask_debug_only) == 1:
            # Synthesize a zero scalar loss that depends on embeddings so backward() is valid
            if len(list_embedding) == 0:
                dummy = torch.zeros((), device=head_input.device)
            else:
                dummy = sum([emb.mean() * 0 for emb in list_embedding])
            list_loss_float = [0.0 for _ in range(args.num_head)]
            is_accumulation_step = (global_step % args.backward_passes_per_step != 0)
            scaled_loss = dummy  # zero tensor
        else:
            for head_id, pfc in enumerate(list_module_pfc):
                dataset_config = args.list_dataset[head_id]
                head_embedding = list_embedding[head_id]
                # --- robust label handling (original code block unchanged) ---
                head_label = list_data_batch[head_id]["labels"]
                if not isinstance(head_label, torch.Tensor):
                    head_label = torch.as_tensor(head_label)
                head_label = head_label.long().cuda(non_blocking=True)

                label_select = getattr(dataset_config, "label_select", 0)
                random_diff = int(getattr(dataset_config, "random_diff", 1))
                random_diff = max(1, random_diff)

                loss_weight = args.list_loss_weight[head_id]

                if dataset_config.dali_type == "det":
                    full_label_num = head_label.shape[-1]
                    head_label_2d = head_label.view(-1, full_label_num)[:, label_select : label_select + random_diff]
                else:
                    if head_label.ndim == 1:
                        head_label_2d = head_label.unsqueeze(1)
                    else:
                        head_label_2d = head_label
                    if head_label_2d.ndim == 2 and head_label_2d.size(1) > 1:
                        head_label_2d = head_label_2d[:, label_select : label_select + random_diff]

                num_cls = getattr(pfc, 'num_classes', None) or getattr(pfc, 'class_num', None)
                if num_cls is None:
                    try:
                        num_cls = _resolve_num_classes(dataset_config)
                    except Exception:
                        pass
                if isinstance(num_cls, int):
                    if dataset_config.dali_type == "det":
                        head_label_2d = head_label_2d.clamp(0, num_cls - 1)
                    else:
                        if head_label_2d.ndim == 2:
                            head_label_2d = head_label_2d.clamp(0, num_cls - 1)
                        else:
                            head_label = head_label.clamp(0, num_cls - 1)

                if rank == 0 and global_step == 0:
                    try:
                        if dataset_config.dali_type == "det" or head_label_2d.ndim == 2:
                            _flat = head_label_2d.reshape(-1)
                        else:
                            _flat = head_label.view(-1)
                        _min = int(_flat.min().item())
                        _max = int(_flat.max().item())
                        logging.getLogger().info(f"[label_range] head={list_head_name[head_id]} min={_min} max={_max} num_cls={num_cls}")
                    except Exception:
                        pass

                try:
                    _num_cls = num_cls if isinstance(num_cls, int) else (
                        getattr(pfc, 'num_classes', None) or getattr(pfc, 'class_num', None)
                    )
                    if isinstance(_num_cls, torch.Tensor):
                        _num_cls = int(_num_cls.item())

                    def _sanitize_1d(lbl_1d: torch.Tensor) -> torch.Tensor:
                        lbl_1d = lbl_1d.view(-1).contiguous().long().cuda(non_blocking=True)
                        if isinstance(_num_cls, int):
                            invalid = (lbl_1d < 0) | (lbl_1d >= _num_cls)
                            if invalid.any() and rank == 0 and global_step < 5:
                                logging.getLogger().warning(
                                    f"[labels] found {int(invalid.sum().item())} invalid id(s); num_classes={_num_cls}. They will be clamped to range.")
                            lbl_1d = lbl_1d.clamp(0, _num_cls - 1)
                        return lbl_1d

                    def _sanitize_2d(lbl_2d: torch.Tensor, cfg_random_diff: int):
                        if lbl_2d.ndim == 1:
                            lbl_2d = lbl_2d.unsqueeze(1)
                        lbl_2d = lbl_2d.contiguous().long().cuda(non_blocking=True)
                        K = int(lbl_2d.size(1)) if lbl_2d.ndim == 2 else 1
                        if K == 0:
                            lbl_2d = lbl_2d.view(-1, 1)
                            K = 1
                        rd = int(max(1, cfg_random_diff))
                        rd = min(rd, K)
                        lbl_2d = lbl_2d[:, :rd]
                        if isinstance(_num_cls, int):
                            invalid = (lbl_2d < 0) | (lbl_2d >= _num_cls)
                            if invalid.any() and rank == 0 and global_step < 5:
                                logging.getLogger().warning(
                                    f"[labels-2d] found {int(invalid.sum().item())} invalid id(s) in 2D labels; num_classes={_num_cls}. They will be clamped.")
                            lbl_2d = lbl_2d.clamp(0, _num_cls - 1)
                        return lbl_2d, rd

                    force_1d = os.environ.get("PFC_FORCE_1D", "0") == "1"

                    if head_id == 0 and not force_1d:
                        if (dataset_config.dali_type == "det") or (head_label_2d.ndim == 2):
                            labels2d, rd = _sanitize_2d(head_label_2d, random_diff)
                            if rank == 0 and global_step == 0:
                                logging.getLogger().info(
                                    f"[pfc-call] head0 2D path: shape={tuple(labels2d.shape)} random_diff={rd} num_classes={_num_cls}")
                            head_loss = pfc(head_embedding, labels2d, rd) * loss_weight
                        else:
                            labels1d = _sanitize_1d(head_label)
                            if rank == 0 and global_step == 0:
                                logging.getLogger().info(
                                    f"[pfc-call] head0 1D path: shape={tuple(labels1d.shape)} num_classes={_num_cls}")
                            head_loss = pfc(head_embedding, labels1d) * loss_weight
                    else:
                        if (dataset_config.dali_type == "det") or (head_label_2d.ndim == 2):
                            _lbl = head_label_2d[:, 0] if head_label_2d.ndim == 2 else head_label
                        else:
                            _lbl = head_label
                        labels1d = _sanitize_1d(_lbl)
                        if rank == 0 and global_step == 0:
                            logging.getLogger().info(
                                f"[pfc-call] head{head_id} 1D path: shape={tuple(labels1d.shape)} num_classes={_num_cls}")
                        head_loss = pfc(head_embedding, labels1d) * loss_weight

                except RuntimeError as e:
                    if rank == 0:
                        try:
                            _dbg = {
                                "emb": tuple(head_embedding.shape),
                                "label_ndim": int(head_label.ndim),
                                "label_shape": tuple(head_label.shape),
                                "label2d_ndim": int(head_label_2d.ndim) if isinstance(head_label_2d, torch.Tensor) else None,
                                "label2d_shape": tuple(head_label_2d.shape) if isinstance(head_label_2d, torch.Tensor) else None,
                                "random_diff": int(random_diff),
                                "num_classes": int(_num_cls) if isinstance(_num_cls, int) else None,
                            }
                            def _minmax(t):
                                try:
                                    return int(t.min().item()), int(t.max().item())
                                except Exception:
                                    return None
                            _dbg["label_minmax"] = _minmax(head_label)
                            if isinstance(head_label_2d, torch.Tensor):
                                _dbg["label2d_minmax"] = _minmax(head_label_2d)
                            logging.getLogger().error(f"[pfc-exception] {e}\ncontext={_dbg}")
                        except Exception:
                            pass
                    raise

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

        for i in range(args.num_head):
            list_next_data_batch[i] = next(list_iter[i])

        if int(args.mask_debug_only) == 1 and global_step >= int(args.max_debug_steps):
            if rank == 0:
                logging.getLogger().info(f"[mask-debug] reached max_debug_steps={args.max_debug_steps}, exiting.")
            exit(0)

        if global_step % args.ckpt_interval == 0:
            save_checkpoint(
                args.output,
                backbone,
                opt,
                lr_scheduler,
                global_step,
                keep_num=10,
            )

        if global_step > args.total_steps:
            save_checkpoint(
                args.output,
                backbone,
                opt,
                lr_scheduler,
                global_step,
                keep_num=10,
            )
            exit()
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