import argparse
import math
import os
import time
import warnings
from typing import dict

import torch
import torch.nn.functional as F
import torchmetrics
from timm.loss import LabelSmoothingCrossEntropy
from timm.models import create_model
from timm.models.layers import trunc_normal_
from torch import distributed, nn
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import LinearLR
from transformers import AutoModel

# Ensure custom models and layers are registered
import model_factory
from dataloader.ap_dataloader_dali import get_dali_dataloader
from model_factory.layers import Siglip2MultiheadAttentionPoolingHead

warnings.filterwarnings("ignore")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Attentive probing with SigLIP2 head (Meta style)")
    # Data
    parser.add_argument("--data_root", default="/data_3/data_attentive_probe")
    parser.add_argument("--train_data_csv_path", default="ssv2_train_new.csv")
    parser.add_argument("--val_data_csv_path", default="ssv2_val_new.csv")
    parser.add_argument("--dataset", default="ssv2")

    # Model
    parser.add_argument("--model_family", default="chunk_wise_sampling")
    parser.add_argument("--model_name", default="ov_encoder_large")
    parser.add_argument("--model_weight", default="NULL")
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--num_tokens", type=int, default=1568)
    parser.add_argument("--input_size", type=int, default=224)
    parser.add_argument("--tubelet_size", type=int, default=1)
    parser.add_argument("--embedding_size", type=int, default=768)
    parser.add_argument("--num_classes", type=int, default=0)
    parser.add_argument(
        "--target_frames", type=int, default=64, help="Target number of frames to interpolate to (default: 64)"
    )

    # Train
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--default_epoch", type=int, default=10)
    parser.add_argument("--default_weight_decay", type=float, default=0)
    parser.add_argument("--default_min_lr", type=float, default=1e-7)
    parser.add_argument("--default_lr_list", type=float, nargs="+", default=[1e-4])
    parser.add_argument("--clip_grad", type=float, default=5.0)
    parser.add_argument("--smoothing", type=float, default=0.1)
    parser.add_argument("--print_freq", type=int, default=10)
    parser.add_argument("--eval_freq", type=int, default=1)
    parser.add_argument("--frames_token_num", type=int, default=196)

    # Dataloader
    parser.add_argument("--dali_num_threads", type=int, default=2)
    parser.add_argument("--dali_py_num_workers", type=int, default=4)
    # ===> New: decord thread number parameter <===
    parser.add_argument("--decord_num_threads", type=int, default=2, help="Number of threads for decord video reader.")
    parser.add_argument("--short_side_size", type=int, default=256)

    parser.add_argument("--mean", nargs=3, type=float, default=[0.48145466, 0.4578275, 0.40821073])
    parser.add_argument("--std", nargs=3, type=float, default=[0.26862954, 0.26130258, 0.27577711])

    # Misc
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--save_report", default="fewshot_video_report/ActionRecognition")

    # Distributed training parameters
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--global_rank", type=int, default=0)

    # New: temporal and spatial crop parameters (defaults match DALI defaults)
    parser.add_argument("--num_temporal_crops", type=int, default=1, help="Number of temporal crops for evaluation")
    parser.add_argument("--num_spatial_crops", type=int, default=1, help="Number of spatial crops for evaluation")

    parser.add_argument("--probe_size", default=1, type=int)

    return parser.parse_args()


def interpolate_frame_indices(
    frame_indices: torch.Tensor, total_frames: torch.Tensor, target_frames: int = 64
) -> torch.Tensor:
    """
    Interpolate frame indices from original video frame count to target frame count

    Args:
        frame_indices: [B, seq_len] original frame indices
        total_frames: [B] total frames per video
        target_frames: target frame count (default 64)

    Returns:
        interpolated_indices: [B, seq_len] interpolated frame indices, range [0, target_frames-1]
    """
    bs, seq_len = frame_indices.shape
    device = frame_indices.device

    # Convert total_frames to float for interpolation calculation
    total_frames_float = total_frames.float().view(bs, 1)  # [B, 1]
    frame_indices_float = frame_indices.float()  # [B, seq_len]

    # Interpolation formula: new_idx = (old_idx / (total_frames - 1)) * (target_frames - 1)
    # Handle total_frames = 1 case
    total_frames_safe = torch.clamp(total_frames_float - 1, min=1.0)
    interpolated_indices = (frame_indices_float / total_frames_safe) * (target_frames - 1)

    # Round and convert to integer
    interpolated_indices = torch.round(interpolated_indices).long()

    # Ensure indices are in valid range
    interpolated_indices = torch.clamp(interpolated_indices, 0, target_frames - 1)

    return interpolated_indices


def get_feature(
    args: argparse.Namespace,
    videos: torch.Tensor,
    model: nn.Module,
    frame_indices: torch.Tensor = None,
    total_frames: torch.Tensor = None,
    is_training: bool = False,
) -> torch.Tensor:
    """
    Extract features, supporting both video and image input.

    Args:
        args: argument configuration
        videos: video data [B, C, T, H, W] or image data [B, C, H, W]
        model: model
        frame_indices: video frame indices [B, seq_len], used for ov_encoder_large sampling
        total_frames: total frames per video [B]
        is_training: whether in training mode
    """

    def video_to_images(videos: torch.Tensor) -> torch.Tensor:
        """
        Unfold video [B, C, T, H, W] into image sequence [B*T, C, H, W]
        """
        B, C, T, H, W = videos.shape
        images = videos.permute(0, 2, 1, 3, 4).reshape(-1, C, H, W)  # [B*T, C, H, W]
        return images

    list_vit_single_image = ["clip", "siglip", "siglip2", "dinov2", "dinov3", "metaclip", "aimv2"]
    if args.model_family in list_vit_single_image:
        # ===> Image-specific branch <===
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            with torch.no_grad():
                # If video input, convert to images
                B, C, T, H, W = videos.shape
                if videos.dim() == 5:  # Video branch [B, C, T, H, W]
                    videos = video_to_images(videos)

                if videos.dim() == 4:  # Detected as image branch [B, C, H, W]
                    hidden_states = model(videos)
                    if isinstance(hidden_states, dict) and "visible_embeddings" in hidden_states:
                        hidden_states = hidden_states["visible_embeddings"]

                    # hidden_states = hidden_states.view(B, -1, hidden_states.size(-1))  # [B, seq_len, hidden_size]
                    hidden_states = hidden_states.reshape(B, -1, hidden_states.size(-1))  # [B, seq_len, hidden_size]
                    # ===> New: sin/cos temporal positional encoding (2 lines of code) <===
                    pos = torch.arange(T, device=videos.device).unsqueeze(1) * torch.exp(
                        torch.arange(0, args.embedding_size, 2, device=videos.device)
                        * (-math.log(10000.0) / args.embedding_size)
                    )  # [T, D/2]
                    temporal_pos = torch.stack([torch.sin(pos), torch.cos(pos)], dim=2).flatten(1)[
                        :, : args.embedding_size
                    ]  # [T, D]
                    hidden_states = hidden_states.view(B, T, -1, args.embedding_size) + temporal_pos.unsqueeze(
                        0
                    ).unsqueeze(2)  # Add to tokens of each frame
                    hidden_states = hidden_states.view(B, -1, args.embedding_size)  # [B, T*tokens_per_frame, D]
                    return hidden_states
                else:
                    raise ValueError("SigLIP2 only supports image input with 4 dimensions [B, C, H, W].")

    elif args.model_family == "chunk_wise_sampling":
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            with torch.no_grad():
                bs, C, T, H, W = videos.shape
                device = videos.device
                frame_tokens = args.frames_token_num  # Number of tokens per frame
                target_frames = args.target_frames  # Target frame count, default 64

                if frame_indices is not None and total_frames is not None:
                    # ===> Interpolate frame indices to target_frames <===
                    interpolated_indices = interpolate_frame_indices(
                        frame_indices, total_frames.view(-1), target_frames
                    )
                    # ===> Calculate visible_index (based on target_frames) <===
                    per = torch.arange(frame_tokens, device=device)
                    visible_index = (interpolated_indices.unsqueeze(-1) * frame_tokens + per).reshape(bs, -1)
                    visible_index = visible_index.clamp_max(target_frames * frame_tokens - 1)

                    # ===> Compute patch_positions for RoPE with temporal scaling to [0, 64) <===
                    # Calculate spatial grid dimensions (assume square patches)
                    patches_per_side = int(math.sqrt(frame_tokens))  # e.g., 14 for 196 tokens
                    seq_len = frame_indices.shape[1]  # Number of frames sampled

                    # Temporal positions: use interpolated_indices (already in [0, target_frames-1])
                    # Shape: [bs, seq_len] -> expand to [bs, seq_len * frame_tokens]
                    t_positions = interpolated_indices.unsqueeze(-1).expand(-1, -1, frame_tokens).reshape(bs, -1)

                    # Spatial positions: h and w within each frame
                    # per is [0, 1, 2, ..., frame_tokens-1]
                    h_per_patch = per // patches_per_side  # [0,0,...,0,1,1,...,1,...,patches_per_side-1]
                    w_per_patch = per % patches_per_side  # [0,1,...,patches_per_side-1,0,1,...,patches_per_side-1,...]

                    # Expand spatial positions for all frames and batches
                    # Shape: [frame_tokens] -> [bs, seq_len * frame_tokens]
                    h_positions = h_per_patch.unsqueeze(0).unsqueeze(0).expand(bs, seq_len, -1).reshape(bs, -1)
                    w_positions = w_per_patch.unsqueeze(0).unsqueeze(0).expand(bs, seq_len, -1).reshape(bs, -1)

                    # Stack to create patch_positions: [bs, seq_len * frame_tokens, 3]
                    patch_positions = torch.stack([t_positions, h_positions, w_positions], dim=-1)

                    enc_out = model(videos, patch_positions, visible_index)
                    if hasattr(enc_out, "last_hidden_state"):
                        outputs = enc_out.last_hidden_state
                    else:
                        outputs = enc_out["visible_embeddings"]

                else:
                    raise

                return outputs

    raise ValueError(f"Unsupported model_family: {args.model_family}")


class ClassificationHead(nn.Module):
    def __init__(self, hidden_dim: int, num_classes: int, init_scale: float = 1e-3, probe_size=1) -> None:
        super().__init__()
        self.pool = Siglip2MultiheadAttentionPoolingHead(
            hidden_size=hidden_dim,
            num_attention_heads=max(1, hidden_dim // 64),
            intermediate_size=hidden_dim * 4,
        )
        # self.pool = Siglip2TransformerAttentionPoolingHead(
        #     hidden_size=hidden_dim,
        #     num_attention_heads=max(1, hidden_dim // 64),
        #     num_layers=probe_size,
        #     norm_cls=nn.LayerNorm
        # )
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.apply(self._init_weights)
        self.fc.weight.data.mul_(init_scale)
        self.fc.bias.data.mul_(init_scale)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        x = self.pool(feats)
        x = self.norm(x)
        x = self.fc(x)
        return x


def train_one_experiment(
    args: argparse.Namespace,
    lr: float,
    device: torch.device,
    base_model: nn.Module,
    loader_train,
    loader_val,
) -> tuple[float, float]:
    base_model.to(device).eval()
    head = ClassificationHead(hidden_dim=args.embedding_size, num_classes=args.num_classes, probe_size=args.probe_size)
    head.to(device)
    head = torch.nn.parallel.DistributedDataParallel(head, device_ids=[args.local_rank])
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=lr, eps=1e-8, betas=(0.9, 0.999), weight_decay=args.default_weight_decay
    )
    steps_per_epoch = len(loader_train)
    total_iters = steps_per_epoch * args.default_epoch
    if total_iters <= 0:
        raise ValueError("Total iters is 0. Check dataloader and epochs.")
    scheduler = None
    if args.default_min_lr < lr:
        scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=args.default_min_lr / lr, total_iters=total_iters)
    criterion = (
        LabelSmoothingCrossEntropy(smoothing=args.smoothing).to(device)
        if args.smoothing > 0.0
        else nn.CrossEntropyLoss().to(device)
    )
    train_metrics = torchmetrics.MetricCollection(
        {
            "loss": torchmetrics.aggregation.MeanMetric(),
            "lr": torchmetrics.aggregation.MeanMetric(),
            "grad_norm": torchmetrics.aggregation.MeanMetric(),
        }
    ).to(device)
    best = {"acc1": 0.0, "acc5": 0.0}

    start_time = time.time()

    for epoch in range(args.default_epoch):
        head.train()
        train_metrics.reset()
        for i, batch in enumerate(loader_train):
            # ===> Unpack data from batch dictionary (including total_frames) <===
            videos = batch["videos"].to(device, non_blocking=True)
            labels = batch["labels"].view(-1).to(device, non_blocking=True)
            indices = batch["indices"].to(device, non_blocking=True)  # [B, seq_len]
            total_frames = batch["total_frames"].to(device, non_blocking=True)  # [B, 1]

            with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
                feats = get_feature(
                    args, videos, base_model, frame_indices=indices, total_frames=total_frames, is_training=True
                )
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                logits = head(feats)
                loss = criterion(logits, labels)
            loss_value = float(loss.item())
            if not math.isfinite(loss_value):
                if args.rank == 0:
                    print(f"Non-finite loss {loss_value}, aborting.")
                return 0.0, 0.0
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = clip_grad_norm_(head.parameters(), max_norm=args.clip_grad)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            train_metrics["loss"].update(loss_value)
            train_metrics["lr"].update(optimizer.param_groups[0]["lr"])
            train_metrics["grad_norm"].update(float(grad_norm))

            if (i + 1) % args.print_freq == 0:
                metrics_computed = train_metrics.compute()

                if args.rank == 0:
                    elapsed_time = time.time() - start_time
                    samples_processed = args.print_freq * args.batch_size * args.world_size
                    samples_per_sec = samples_processed / elapsed_time

                    print(
                        f"Epoch: [{epoch}][{i + 1}/{steps_per_epoch}]  "
                        f"Speed: {samples_per_sec:.2f} samples/s  "
                        f"Loss: {metrics_computed['loss']:.4f}  "
                        f"LR: {metrics_computed['lr']:.6f}  "
                        f"Grad Norm: {metrics_computed['grad_norm']:.4f}"
                    )

                start_time = time.time()
                train_metrics.reset()

        if hasattr(loader_train, "reset"):
            loader_train.reset()

        if epoch % args.eval_freq == 0 or epoch == args.default_epoch - 1:
            stats = evaluate(args, head, device, base_model, loader_val)
            if hasattr(loader_val, "reset"):
                loader_val.reset()
            if stats["acc1"] > best["acc1"]:
                best = stats
            if args.rank == 0:
                print(
                    f"[Val][Epoch {epoch}] acc1={stats['acc1']:.4f} acc5={stats['acc5']:.4f} | Best acc1={best['acc1']:.4f}"
                )

    return best["acc1"], best["acc5"]


@torch.no_grad()
def evaluate(
    args: argparse.Namespace,
    head: nn.Module,
    device: torch.device,
    base_model: nn.Module,
    loader_val,
) -> dict[str, float]:
    head.eval()
    val_metrics = torchmetrics.MetricCollection(
        {
            "acc1": torchmetrics.Accuracy(task="multiclass", num_classes=args.num_classes, top_k=1),
            "acc5": torchmetrics.Accuracy(task="multiclass", num_classes=args.num_classes, top_k=5),
        }
    ).to(device)

    num_crops = args.num_temporal_crops * args.num_spatial_crops

    all_logits, all_targets = [], []
    steps_val = len(loader_val)
    for i, batch in enumerate(loader_val):
        videos = batch["videos"].to(device, non_blocking=True)  # [B*N, C, T, H, W]
        labels = batch["labels"].view(-1).to(device, non_blocking=True)  # [B*N]
        indices = batch["indices"].to(device, non_blocking=True)
        total_frames = batch["total_frames"].to(device, non_blocking=True)

        B = videos.shape[0] // num_crops
        # Reshape to [B, num_crops, ...]
        videos = videos.view(B, num_crops, *videos.shape[1:])
        labels = labels.view(B, num_crops)[:, 0]  # [B], labels are the same for the same video
        indices = indices.view(B, num_crops, *indices.shape[1:])
        total_frames = total_frames.view(B, num_crops)[:, 0]

        logits_per_crop = []
        for crop_id in range(num_crops):
            feats = get_feature(
                args,
                videos[:, crop_id],
                base_model,
                frame_indices=indices[:, crop_id],
                total_frames=total_frames,
                is_training=False,
            )
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                logits = head(feats)  # [B, num_classes]
                logits_per_crop.append(logits)
        # [num_crops, B, num_classes] -> [B, num_crops, num_classes]
        logits_all = torch.stack(logits_per_crop, dim=1)
        # Average over crop dimension (can use softmax then average / directly average logits)
        logits_mean = logits_all.mean(dim=1)  # [B, num_classes]
        # Collect results
        all_logits.append(logits_mean)
        all_targets.append(labels)

        if (i + 1) % args.print_freq == 0 and args.rank == 0:
            print(f"Eval: [{i + 1}/{steps_val}]")

    all_logits = torch.cat(all_logits, dim=0)  # [total_B, num_classes]
    all_targets = torch.cat(all_targets, dim=0)  # [total_B]

    val_metrics.update(all_logits, all_targets)
    computed_metrics = val_metrics.compute()
    if args.rank == 0:
        print(
            f"* Final Acc@1: {computed_metrics['acc1'] * 100:.1f} | Final Acc@5: {computed_metrics['acc5'] * 100:.1f}"
        )
    return {k: v.item() * 100 for k, v in computed_metrics.items()}


def get_model(args: argparse.Namespace) -> nn.Module:
    if args.model_name == "ov_encoder_large":
        model = AutoModel.from_pretrained(
            "lmms-lab-encoder/onevision-encoder-large", trust_remote_code=True, attn_implementation="flash_attention_2"
        )
        model = torch.compile(model)
        return model

    model = create_model(args.model_name, pretrained=False)
    if args.model_family in ["chunk_wise_sampling"]:
        state_dict = torch.load(args.model_weight, map_location="cpu")
        state_dict = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=True)
    return model


def main() -> None:
    args = parse_args()
    nb_classes_map = {
        "charadesego": 157,
        "CharadesEgo_v1_only3rd": 157,
        "Drone_Action": 13,
        "epic_noun": 300,
        "hmdb51": 51,
        "k400": 400,
        "k700": 700,
        "mit": 339,
        "rareact": 149,
        "ucf101": 101,
        "CharadesEgo_v1_only1st": 157,
        "diving48": 48,
        "epic_verb": 97,
        "k600": 600,
        "k710": 710,
        "perception_test": 63,
        "ssv2": 174,
        "SSV2": 174,
    }
    args.num_classes = nb_classes_map[args.dataset]

    if args.dataset == "ssv2":
        args.train_data_root_path = os.path.join(args.data_root, "ssv2")
        args.val_data_root_path = os.path.join(args.data_root, "ssv2")
        args.train_data_csv_path = "ssv2_train_new.csv"
        args.val_data_csv_path = "ssv2_val_new.csv"
    if args.dataset == "diving48":
        args.train_data_root_path = os.path.join(args.data_root, "diving48")
        args.val_data_root_path = os.path.join(args.data_root, "diving48")
        args.train_data_csv_path = "diving48_train_new.csv"
        args.val_data_csv_path = "diving48_val_new.csv"
    if args.dataset == "epic_verb":
        args.train_data_root_path = os.path.join(args.data_root, "epic_verb")
        args.val_data_root_path = os.path.join(args.data_root, "epic_verb")
        args.train_data_csv_path = "train_new.csv"
        args.val_data_csv_path = "val_new.csv"
    if args.dataset == "epic_noun":
        args.train_data_root_path = os.path.join(args.data_root, "epic_noun")
        args.val_data_root_path = os.path.join(args.data_root, "epic_noun")
        args.train_data_csv_path = "train_new.csv"
        args.val_data_csv_path = "val_new.csv"
    if args.dataset == "perception_test":
        args.train_data_root_path = os.path.join(args.data_root, "perception_test")
        args.val_data_root_path = os.path.join(args.data_root, "perception_test")
        args.train_data_csv_path = "train_new.csv"
        args.val_data_csv_path = "val_new.csv"
    if args.dataset == "hmdb51":
        args.train_data_root_path = os.path.join(args.data_root, "hmdb51")
        args.val_data_root_path = os.path.join(args.data_root, "hmdb51")
        args.train_data_csv_path = "train_new.csv"
        args.val_data_csv_path = "val_new.csv"
    if args.dataset == "k400":
        args.train_data_root_path = os.path.join(args.data_root, "k400")
        args.val_data_root_path = os.path.join(args.data_root, "k400")
        args.train_data_csv_path = "train_new.csv"
        args.val_data_csv_path = "val_new.csv"
    if args.dataset == "charadesego":
        args.train_data_root_path = os.path.join(args.data_root, "charadesego")
        args.val_data_root_path = os.path.join(args.data_root, "charadesego")
        args.train_data_csv_path = "train_new.csv"
        args.val_data_csv_path = "val_new.csv"
    try:
        args.rank = int(os.environ["RANK"])
        args.local_rank = int(os.environ["LOCAL_RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        distributed.init_process_group("nccl")
    except KeyError:
        args.rank = 0
        args.local_rank = 0
        args.world_size = 1
        distributed.init_process_group(
            backend="nccl", init_method="tcp://127.0.0.1:12584", rank=args.rank, world_size=args.world_size
        )
    torch.cuda.set_device(args.local_rank)
    device = torch.device(args.local_rank)
    args.global_rank = args.rank

    if args.rank == 0:
        print("Create data loaders...")

    if args.model_family in ["siglip", "siglip2"]:
        args.mean = [0.5, 0.5, 0.5]
        args.std = [0.5, 0.5, 0.5]
    if args.model_family in ["dinov2", "dinov3"]:
        args.mean = [0.485, 0.456, 0.406]
        args.std = [0.229, 0.224, 0.225]

    train_loader = get_dali_dataloader(
        data_root_path=args.train_data_root_path,
        data_csv_path=os.path.join(args.train_data_root_path, args.train_data_csv_path),
        mode="train",
        batch_size=args.batch_size,
        sequence_length=args.num_frames,
        input_size=args.input_size,
        short_side_size=args.short_side_size,
        mean=args.mean,
        std=args.std,
        dali_num_threads=args.dali_num_threads,
        dali_py_num_workers=args.dali_py_num_workers,
        decord_num_threads=args.decord_num_threads,
        seed=args.seed,
        # Training does not need num_temporal_crops/num_spatial_crops (only used for evaluation)
    )
    val_loader = get_dali_dataloader(
        data_root_path=args.val_data_root_path,
        data_csv_path=os.path.join(args.val_data_root_path, args.val_data_csv_path),
        mode="val",
        batch_size=args.batch_size,
        sequence_length=args.num_frames,
        input_size=args.input_size,
        short_side_size=args.short_side_size,
        mean=args.mean,
        std=args.std,
        dali_num_threads=args.dali_num_threads,
        dali_py_num_workers=args.dali_py_num_workers,
        decord_num_threads=args.decord_num_threads,
        seed=1024,
        # num_temporal_crops=args.num_temporal_crops,   # New!
        # num_spatial_crops=args.num_spatial_crops     # New!
    )
    if args.rank == 0:
        print("Data loaders ready.")

    lrs = args.default_lr_list if isinstance(args.default_lr_list, list) else [args.default_lr_list]
    best_lr, best_top1, best_top5 = 0.0, 0.0, 0.0
    for lr in lrs:
        base_model = get_model(args)
        # base_model = torch.compile(base_model)
        acc1, acc5 = train_one_experiment(args, lr, device, base_model, train_loader, val_loader)
        if acc1 > best_top1:
            best_lr, best_top1, best_top5 = lr, acc1, acc5

    if args.rank == 0:
        print(f"best_lr: {best_lr} max_acc_top1: {best_top1} max_acc_top5: {best_top5}")

        save_path = os.path.join(args.save_report, f"report_attentive_probe_{os.path.basename(args.model_weight)}.txt")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "a+") as f:
            f.write(f"{args.dataset} {best_top1}\n")
        print(f"Saved report to {save_path}")


if __name__ == "__main__":
    main()
