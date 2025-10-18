import warnings

warnings.filterwarnings('ignore')

import argparse
import json
import math
import os
import warnings
from functools import partial

import torch
import torch.nn.functional as F
from ac_ap_dataloader_dali import dali_dataloader
from all_utils import (MetricLogger, load_finetune_checkpoint,
                       setup_for_distributed, setup_seed)
from timm.loss import LabelSmoothingCrossEntropy
from timm.models import create_model
from timm.models.layers import DropPath, trunc_normal_
from timm.utils import accuracy
from torch import distributed, inf, nn
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import LinearLR

import model_factory


def get_args():
    parser = argparse.ArgumentParser('Extract features using the videomae model', add_help=False)
    parser.add_argument('--train_data_root_path', default="/video_vit/fewshot_video/ActionRecognition")
    parser.add_argument('--train_data_csv_path', default="/video_vit/fewshot_video/ActionRecognition")
    parser.add_argument('--val_data_root_path', default='/video_vit/eval_data/val/')
    parser.add_argument('--val_data_csv_path', default='/video_vit/eval_data/annotation/')
    parser.add_argument('--save_report', default="fewshot_video_report/ActionRecognition")

    parser.add_argument('--dataset', default="ssv2")
    parser.add_argument('--num_shots', default=10, type=int)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--num_step", default=8, type=int)
    parser.add_argument('--multi_views', default=False)

    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument('--model_family', default='')
    parser.add_argument('--model_name', default='')
    parser.add_argument('--ckpt_path', default='')
    parser.add_argument("--num_frames", default=8, type=int)
    parser.add_argument("--input_size", default=224, type=int)
    parser.add_argument("--tubelet_size", default=1, type=int)
    parser.add_argument("--embedding_size", default=768, type=int)

    # default
    parser.add_argument('--model_key', default='model|module', type=str)
    parser.add_argument('--mean', nargs=3, default=[0.485, 0.456, 0.406], type=float)
    parser.add_argument('--std', nargs=3, default=[0.229, 0.224, 0.225], type=float)
    parser.add_argument('--dali_num_threads', default=8, type=int)
    parser.add_argument('--dali_py_num_workers', default=14, type=int)
    parser.add_argument('--short_side_size', default=256, type=int)
    parser.add_argument('--use_rgb', default=False)
    parser.add_argument('--smoothing', default=0.1, type=float)

    # default
    parser.add_argument('--default_warmup_epochs', default=0, type=int)
    parser.add_argument('--default_epoch', default=20, type=int)
    parser.add_argument('--default_attentive_head', default=16, type=int)
    parser.add_argument('--default_attentive_out_dim', default=768, type=int)
    parser.add_argument('--default_weight_decay', default=0.01, type=float)
    parser.add_argument('--default_min_lr', default=1e-7, type=float)
    parser.add_argument('--default_lr_list', default=[1e-4], type=float)
    parser.add_argument('--default_start_warmup_value', default=0.0, type=float)
    parser.add_argument('--clip_grad', default=5.0, type=float)
    parser.add_argument('--print_freq', default=10, type=int)
    parser.add_argument('--eval_freq', default=10, type=int)

    parser.add_argument('--using_normlize', action='store_true')
    return parser.parse_args()



def get_feature(videos, processor, forward_base_model):
    # base model export feature
    if args.model_family == 'pe' or args.model_family == 'ijepa':
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = [] 
            for frame_idx in range(videos.shape[2]):
                frame = videos[:, :,frame_idx,  :, :]
                with torch.no_grad():
                    output = forward_base_model(frame)
                outputs.append(output)
            outputs = torch.cat(outputs, dim=1)
    elif args.model_family == 'clip':
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = [] 
            for frame_idx in range(videos.shape[2]):
                frame = videos[:, :,frame_idx,  :, :]
                inputs = processor(images = frame, return_tensors='pt').to(device)
                with torch.no_grad():  
                    output = forward_base_model(**inputs) 
                outputs.append(output.last_hidden_state[1:])
            outputs = torch.cat(outputs, dim=1)    
    elif args.model_family == 'siglip':
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = [] 
            for frame_idx in range(videos.shape[2]):
                frame = videos[:, :,frame_idx,  :, :]
                inputs = processor(images = frame, padding='max_length', return_tensors='pt')['pixel_values']
                inputs = inputs.to(dtype=torch.bfloat16, device=device)
                with torch.no_grad(): 
                    output = forward_base_model.module.vision_model(inputs)  
                outputs.append(output.last_hidden_state[1:])
            outputs = torch.cat(outputs, dim=1)
    elif args.model_family == 'dino_v2':
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = [] 
            for frame_idx in range(videos.shape[2]):
                frame = videos[:, :,frame_idx,  :, :]
                inputs = processor(images = frame, return_tensors='pt')
                with torch.no_grad():  
                    output = forward_base_model(**inputs)  
                outputs.append(output.last_hidden_state[1:])
            outputs = torch.cat(outputs, dim=1)
    elif args.model_family == 'dino_v3':
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = []
            for frame_idx in range(videos.shape[2]):
                frame = videos[:, :,frame_idx,  :, :]
                inputs = processor(images = frame, return_tensors='pt')
                with torch.no_grad():
                    output = forward_base_model(**inputs)
                outputs.append(output.last_hidden_state[:, 5:, :])
            outputs = torch.cat(outputs, dim=1)
    elif args.model_family == "languagebind":
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                output = forward_base_model.module.vision_model(pixel_values=videos)
        output=output.last_hidden_state[1:]
    elif args.model_family == "rice":
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = []
            for frame_idx in range(videos.shape[2]):
                frame = videos[:, :,frame_idx,  :, :]
                with torch.no_grad():
                    output = forward_base_model(frame)
                outputs.append(output.last_hidden_state[:, 1:, :])
            outputs = torch.cat(outputs, dim=1)

    elif args.model_family == "llava_vit":
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            with torch.no_grad():
                enc_out = forward_base_model(videos, mask_ratio=0.5)
                outputs = enc_out["visible_embeddings"]

    elif args.model_family == "llava_vit_si":
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = []
            temporal_table = None
            T = videos.shape[2]
            for frame_idx in range(T):
                frame = videos[:, :, frame_idx, :, :]

                with torch.no_grad():
                    enc_out = forward_base_model(frame)

                feats = enc_out["visible_embeddings"]

                # --- NEW: 加时间位置编码（正弦），首帧时一次性构建 ---
                if temporal_table is None:
                    B, N_patch, D = feats.shape
                    device = feats.device
                    # 构建 [T, D] 正弦时间编码
                    position = torch.arange(T, device=device).float().unsqueeze(1)
                    div_term = torch.exp(torch.arange(0, D, 2, device=device).float() * (-math.log(10000.0) / D))
                    temporal_table = torch.zeros(T, D, device=device)
                    temporal_table[:, 0::2] = torch.sin(position * div_term)
                    temporal_table[:, 1::2] = torch.cos(position * div_term)
                    # temporal_table: [T, D]

                # 取当前帧的编码，加到该帧所有 patch 上
                feats = feats + temporal_table[frame_idx].view(1, 1, -1)
                # --- NEW END ---

                outputs.append(feats)

            outputs = torch.cat(outputs, dim=1)  # [B, T*N_patch, D]


    elif args.model_family in ["ov_1_5_vit", "mlcd_base", "mlcd", "mlcd_torch"]:
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = []
            temporal_table = None
            T = videos.shape[2]
            for frame_idx in range(T):
                frame = videos[:, :, frame_idx, :, :]

                with torch.no_grad():
                    output = forward_base_model(frame)

                if args.model_family == "ov_1_5_vit":
                    feats = output.last_hidden_state[:, 1:, :]  # [B, N_patch, D]
                else:
                    feats = output[:, 1:, :]  # [B, N_patch, D]

                # --- NEW: 加时间位置编码（正弦），首帧时一次性构建 ---
                if temporal_table is None:
                    B, N_patch, D = feats.shape
                    device = feats.device
                    # 构建 [T, D] 正弦时间编码
                    position = torch.arange(T, device=device).float().unsqueeze(1)
                    div_term = torch.exp(torch.arange(0, D, 2, device=device).float() * (-math.log(10000.0) / D))
                    temporal_table = torch.zeros(T, D, device=device)
                    temporal_table[:, 0::2] = torch.sin(position * div_term)
                    temporal_table[:, 1::2] = torch.cos(position * div_term)
                    # temporal_table: [T, D]

                # 取当前帧的编码，加到该帧所有 patch 上
                feats = feats + temporal_table[frame_idx].view(1, 1, -1)
                # --- NEW END ---

                outputs.append(feats)

            outputs = torch.cat(outputs, dim=1)  # [B, T*N_patch, D]
    return outputs


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0, attn_head_dim=None, out_dim=None):
        super().__init__()
        if out_dim is None:
            out_dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * num_heads
        assert all_head_dim == dim
        self.scale = qk_scale or head_dim ** -0.5
        self.head_dim = head_dim
        self.q = nn.Linear(dim, all_head_dim, bias=False)
        self.k = nn.Linear(dim, all_head_dim, bias=False)
        self.v = nn.Linear(dim, all_head_dim, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.k_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.k_bias = None
            self.v_bias = None
        self.attn_drop_value = attn_drop
        self.proj = nn.Linear(all_head_dim, out_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, k=None, v=None, attn_mask=None, is_causal=False):
        B, Nq, C = x.shape
        if k is None:
            k = x
        if v is None:
            v = k
        q = F.linear(x, self.q.weight, self.q_bias)
        k = F.linear(k, self.k.weight, self.k_bias)
        v = F.linear(v, self.v.weight, self.v_bias)
        q = q.view(B, Nq, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, k.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, v.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        default_scale = self.head_dim ** -0.5
        if abs(self.scale - default_scale) > 1e-12:
            q = q * (self.scale * (self.head_dim ** 0.5))
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attn_drop_value if self.training else 0.0,
            is_causal=is_causal
        )
        out = out.transpose(1, 2).contiguous().view(B, Nq, -1)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class AttentiveBlock(nn.Module):
    
    def __init__(self, dim, num_heads, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, attn_head_dim=None, out_dim=None):
        super().__init__()
        self.norm1_q = norm_layer(dim)
        self.norm1_k = norm_layer(dim)
        self.norm1_v = norm_layer(dim)
        self.cross_attn = CrossAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
            proj_drop=drop, attn_head_dim=attn_head_dim, out_dim=out_dim)
        
        if drop_path > 0.:
            print(f"Use DropPath in projector: {drop_path}")
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x_q, x_kv, pos_q, pos_k, bool_masked_pos, rel_pos_bias=None):
        x_q = self.norm1_q(x_q + pos_q)
        x_k = self.norm1_k(x_kv + pos_k)
        x_v = self.norm1_v(x_kv)
        x = self.cross_attn(x_q, k=x_k, v=x_v)
        return x


class AttentionPoolingBlock(AttentiveBlock):
    def forward(self, x):
        x_q = x.mean(1, keepdim=True)
        x_kv, pos_q, pos_k = x, 0, 0
        x = super().forward(x_q, x_kv, pos_q, pos_k, bool_masked_pos=None, rel_pos_bias=None)
        x = x.squeeze(1)
        return x


class CustomModel(nn.Module):
    def __init__(self, attentive_probe_model,
                 attentive_dim, num_classes,
                 init_scale=0.001):
        super(CustomModel, self).__init__()
        self.attentive_probe_model = attentive_probe_model
        self.fc_norm = nn.LayerNorm(attentive_dim)
        self.head = nn.Linear(attentive_dim, num_classes)
        self.apply(self._init_weights)
        self.head.weight.data.mul_(init_scale)
        self.head.bias.data.mul_(init_scale)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, features):
        x = self.attentive_probe_model(features)
        x = self.fc_norm(x)
        x = self.head(x)
        return x


def train_AdamW(
    args,
    lr,
    cur_ap_model,
    cur_device,
    forward_base_model,
    data_loader_train,
    data_loader_val,
    processor=None
):

    cur_ap_model.to(cur_device)
    forward_base_model.to(cur_device).eval()

    optimizer = torch.optim.AdamW(
        cur_ap_model.parameters(),
        lr=lr,
        eps=1e-8,
        betas=(0.9, 0.999),
        weight_decay=args.default_weight_decay,
    )

    min_lr = getattr(args, "default_min_lr", 0.0)
    total_epochs = args.default_epoch
    steps_per_epoch = args.num_train_steps_per_epoch
    total_iters = total_epochs * steps_per_epoch

    if total_iters <= 0:
        raise ValueError("total_iters 计算为 0，请确认 args.default_epoch 与 steps_per_epoch 设置正确。")

    if min_lr < lr:
        scheduler = LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=min_lr / lr,
            total_iters=total_iters
        )
    else:
        scheduler = None  # 不降学习率

    if args.smoothing > 0.0:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing).to(cur_device)
    else:
        criterion = torch.nn.CrossEntropyLoss().to(cur_device)

    use_bf16 = torch.cuda.is_available()
    last_val_stats = {}
    global_step = 0

    for epoch in range(total_epochs):
        cur_ap_model.train()
        metric_logger = MetricLogger(delimiter="  ")
        header = f"Epoch: [{epoch}]"

        for data_iter_step, (videos, labels) in enumerate(
            metric_logger.log_every(
                data_loader_train,
                print_freq=args.print_freq,
                header=header,
                world_size=args.world_size,
                batch_size=args.batch_size
            )
        ):
            videos = videos.to(cur_device, non_blocking=True)
            labels = labels.to(cur_device, non_blocking=True).view(-1)

            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=use_bf16, dtype=torch.bfloat16):
                    feats = get_feature(videos, processor, forward_base_model)
                    if args.using_normlize:
                        feats = F.normalize(feats, dim=-1)

            with torch.cuda.amp.autocast(enabled=use_bf16, dtype=torch.bfloat16):
                preds = cur_ap_model(feats)
                loss = criterion(preds, labels)

            loss_value = float(loss.item())
            if not math.isfinite(loss_value):
                print(f"Loss is {loss_value}, stopping training.")
                return 0.0, 0.0

            optimizer.zero_grad()
            loss.backward()
            grad_norm = clip_grad_norm_(cur_ap_model.parameters(), max_norm=args.clip_grad)
            optimizer.step()

            if scheduler is not None:
                scheduler.step()   # 按 iteration 更新

            metric_logger.update(loss=loss_value)
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            metric_logger.update(grad_norm=float(grad_norm))

            global_step += 1

        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        if hasattr(data_loader_train, "reset"):
            data_loader_train.reset()

        # 验证
        if epoch % args.eval_freq == 0 or epoch == total_epochs - 1:
            val_stats = validation_one_epoch(
                args=args,
                model=cur_ap_model,
                cur_device=cur_device,
                forward_base_model=forward_base_model,
                data_loader_val=data_loader_val,
                processor=processor
            )
            last_val_stats = val_stats
            if hasattr(data_loader_val, "reset"):
                data_loader_val.reset()
            print(f"[Val][Epoch {epoch}] acc1={val_stats.get('acc1', 0):.4f} acc5={val_stats.get('acc5', 0):.4f}")

    if "acc1" not in last_val_stats:
        last_val_stats = {"acc1": 0.0, "acc5": 0.0}
    return last_val_stats["acc1"], last_val_stats["acc5"]


@torch.no_grad()
def validation_one_epoch(
    args, model, cur_device,
    forward_base_model, data_loader_val, processor = None):

    metric_logger_val = MetricLogger(delimiter="  ")
    header = 'Val:'
    model.eval()
    for (videos, target) in metric_logger_val.log_every(data_loader_val, args.print_freq, header,
                                                        args.world_size, args.batch_size):
        videos = videos.to(cur_device, non_blocking = True)
        target = target.to(cur_device, non_blocking = True)
        target = target.view(-1)
        outputs = get_feature(videos, processor, forward_base_model)
        if args.using_normlize:
            outputs = F.normalize(outputs, dim=-1)

        # attentive_probing
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            pred = model(outputs)

        acc1, acc5 = accuracy(pred, target, topk=(1, 5))
        batch_size = videos.shape[0]
        metric_logger_val.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger_val.meters['acc5'].update(acc5.item(), n=batch_size)

    # gather the stats from all processes
    metric_logger_val.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f}'.format(
        top1=metric_logger_val.acc1,
        top5=metric_logger_val.acc5))
    return {k: meter.global_avg for k, meter in metric_logger_val.meters.items()}


def find_peak(args, lr, cur_device, base_model, data_loader_train, data_loader_val):

    # base model export feature
    if isinstance(base_model, tuple):
        base_model, processor = base_model
    else:
        processor = None

    # if args.model_family != "dino_v3" and args.model_family != "ov_1_5_vit" and args.model_family != "rice":
    if args.model_family not in ["dino_v3", "ov_1_5_vit", "rice", "llava_vit", "mlcd_torch"]:
        print("have load ckpt")
        base_model = load_finetune_checkpoint(args, base_model)

    base_model.to(cur_device).eval()
    attentive_probe_model = AttentionPoolingBlock(
                                        dim=args.embedding_size,
                                        num_heads=args.default_attentive_head,
                                        qkv_bias=True,
                                        qk_scale=None,
                                        drop=0.0,
                                        attn_drop=0.0,
                                        drop_path=0.0,
                                        norm_layer=partial(nn.LayerNorm, eps=1e-5),
                                        out_dim=args.default_attentive_out_dim)
    ap_model = CustomModel(attentive_probe_model,
                        attentive_dim=args.default_attentive_out_dim,
                        num_classes=args.num_classes)
    print("create attentive probe model end!")
    cur_ap_model = ap_model.to(cur_device)
    cur_ap_model = torch.nn.parallel.DistributedDataParallel(cur_ap_model, device_ids=[args.local_rank])
    cur_ap_model.train()

    acc_top1, acc_top5 = train_AdamW(args, lr, cur_ap_model, cur_device, base_model, data_loader_train, data_loader_val, processor)
    return acc_top1, acc_top5


def mkdir_os(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_model(args):
    if args.model_family == "umt":
        import video_models.umt
        base_model = create_model(
            args.model_name,
            img_size=args.input_size,
            pretrained=False,
            num_classes=args.num_classes,
            all_frames=args.num_frames,
            tubelet_size=args.tubelet_size,
            use_mean_pooling=True)
        base_model.forward= base_model.forward_features_attentive_probe
    elif args.model_family == "videomae_v1":
        import video_models.videomae_v1
        base_model = create_model(
            args.model_name,
            img_size=args.input_size,
            pretrained=False,
            num_classes=args.num_classes,
            all_frames=args.num_frames,
            tubelet_size=args.tubelet_size,
            use_mean_pooling=True)
        base_model.forward= base_model.forward_features_attentive_probe
    elif args.model_family == "videomae_v2":
        import video_models.videomae_v2
        base_model = create_model(
            args.model_name,
            img_size=args.input_size,
            pretrained=False,
            num_classes=args.num_classes,
            all_frames=args.num_frames,
            tubelet_size=args.tubelet_size,
            use_mean_pooling=True)
        base_model.forward= base_model.forward_features_attentive_probe
    elif args.model_family == "vswift":
        import video_models.vswift
        base_model = create_model(
            args.model_name,
            img_size=args.input_size,
            pretrained=False,
            num_classes=args.num_classes,
            all_frames=args.num_frames,
            tubelet_size=args.tubelet_size,
            use_mean_pooling=True)
        base_model.forward= base_model.forward_features_attentive_probe
    elif args.model_family == "ov2":
        import video_models.ov2
        base_model = create_model(
            args.model_name,
            img_size=args.input_size,
            pretrained=False,
            num_classes=args.num_classes,
            all_frames=args.num_frames,
            tubelet_size=args.tubelet_size,
            use_mean_pooling=True)
    elif args.model_family == "viclip":
        import video_models.viclip
        base_model = create_model(
            args.model_name,
            input_resolution=args.input_size,
            pretrained=False,
            kernel_size=args.tubelet_size,
            center=True, 
            num_frames=args.num_frames,
            drop_path=0.0, 
            checkpoint_num=0,
            dropout=0.0)
        base_model.forward= base_model.forward_features_attentive_probe
    elif args.model_family == "internvideo_v1":
        import video_models.internvidev1
        base_model = create_model(
            args.model_name,
            img_size=args.input_size,
            pretrained=False,
            num_classes=args.num_classes,
            all_frames=args.num_frames,
            tubelet_size=args.tubelet_size,
            use_mean_pooling=True)
        base_model.forward= base_model.forward_features_attentive_probe
    elif args.model_family == "internvideo_v2":
        import video_models.internvideo_v2
        base_model = create_model(
            args.model_name,
            pretrained=False,
            num_classes=args.num_classes,
            num_frames=args.num_frames,
            tubelet_size=args.tubelet_size,
            drop_path_rate=0.1,
            checkpoint_num=0,
        )
        base_model.forward= base_model.forward_features_attentive_probe
    elif args.model_family == "vjepa":
        if args.model_name=="vit_large":
            from video_models.vjepa.modeling_finetune import vit_large
            base_model = vit_large(
                img_size=args.input_size,
                pretrained=False,
                kernel_size=args.tubelet_size,
                frames_per_clip=args.num_frames,
                uniform_power=True,
                use_sdpa=True,
                use_SiLU=False,
                tight_SiLU=False)
        else:
            from video_models.vjepa.modeling_finetune import vit_huge
            base_model = vit_huge(
                img_size=args.input_size,
                pretrained=False,
                kernel_size=args.tubelet_size,
                frames_per_clip=args.num_frames,
                uniform_power=True,
                use_sdpa=True,
                use_SiLU=False,
                tight_SiLU=False)
    elif args.model_family == 'univit':
        import video_models.video_mlcd
        print("load create univit")
        base_model = create_model(
            args.model_name,
            img_size=224,
            num_classes=512
        )
    elif args.model_family == 'cvpr':
        import video_models.cvpr
        print("load create cvpr_model")
        base_model = create_model(
            args.model_name,
            # img_size=224,
            # num_classes=512
        )

    elif args.model_family == "rice":
        from modeling_rice_base import MLCDVisionModel
        base_model = MLCDVisionModel.from_pretrained("/vlm/xiangan/pretrain_models/deepglint/rice-vit-large-patch14-560-v1")

    elif args.model_family == "ov_1_5_vit":
       from transformers import MLCDVisionModel
       base_model = MLCDVisionModel.from_pretrained(args.ckpt_path).cuda()

    elif args.model_family in ['llava_vit', 'llava_vit_si']:
        base_model = create_model(
            args.model_name,
            pretrained=False,)
        state_dict = torch.load(args.ckpt_path, map_location='cpu')
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        base_model.load_state_dict(state_dict, strict=True)

    elif args.model_family == "mlcd_torch":
        base_model = create_model(
            args.model_name,
            pretrained=False,)
        state_dict = torch.load(args.ckpt_path, map_location='cpu')
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        base_model.load_state_dict(state_dict, strict=True)


    elif args.model_family == 'mlcd':
        if args.model_name=="vit-bigG-patch14-448":
            base_model = MLCDVisionModel.from_pretrained(args.ckpt_path)
        elif args.model_name=="vit-bigG-patch14-224":
            base_model = MLCDVisionModel.from_pretrained(args.ckpt_path)
        elif args.model_name=="vit-large-patch14-336":
            base_model = CLIPVisionModel.from_pretrained(args.ckpt_path)

    elif args.model_family == 'mlcd_base':
        pretrained_cfg = {
                "ckpt_path": args.ckpt_path,
        }
        base_model = create_model(
            args.model_name,
            pretrained=True,
            pretrained_cfg=pretrained_cfg)

    elif args.model_family == 'languagebind':
        from languagebind import (LanguageBindVideo,
                                  LanguageBindVideoProcessor,
                                  LanguageBindVideoTokenizer)
        base_model = LanguageBindVideo.from_pretrained(args.ckpt_path)
        tokenizer = LanguageBindVideoTokenizer.from_pretrained(args.ckpt_path)
        processor = LanguageBindVideoProcessor(args.model_name.config, tokenizer)
        return base_model, processor
    elif args.model_family == 'pe':
        import core.vision_encoder.pe as pe
        import core.vision_encoder.transforms as transforms
        base_model = pe.CLIP.from_config(args.ckpt_path, pretrained=True)  # Downloads from HF
        processor = transforms.get_image_transform(base_model.image_size)
        return base_model, processor
    elif args.model_family == "ijepa":
        from transformers import AutoModel, AutoProcessor
        base_model = AutoModel.from_pretrained(args.ckpt_path)
        processor = AutoProcessor.from_pretrained(args.ckpt_path)
        return base_model, processor
    elif args.model_family == "siglip":
        from transformers import AutoModel, AutoProcessor, AutoTokenizer
        base_model = AutoModel.from_pretrained(args.ckpt_path, torch_dtype=torch.float32)
        processor = AutoProcessor.from_pretrained(args.ckpt_path, torch_dtype=torch.float32)
        return base_model, processor
    elif args.model_family == "dino":
        from transformers import AutoImageProcessor, AutoModel
        base_model = AutoModel.from_pretrained(args.ckpt_path, torch_dtype=torch.float32)
        processor = AutoImageProcessor.from_pretrained(args.ckpt_path, torch_dtype=torch.float32)
        return base_model, processor
    elif args.model_family == "dino_v3":
        from transformers import AutoImageProcessor, AutoModel
        base_model = AutoModel.from_pretrained(args.ckpt_path, torch_dtype=torch.float32)
        processor = AutoImageProcessor.from_pretrained(args.ckpt_path, torch_dtype=torch.float32)
        return base_model, processor
    elif args.model_family == "clip":
        from transformers import AutoProcessor, CLIPVisionModelWithProjection
        base_model = CLIPVisionModelWithProjection.from_pretrained(args.ckpt_path, torch_dtype=torch.float32)
        processor = AutoProcessor.from_pretrained(args.ckpt_path, torch_dtype=torch.float32)
        return base_model, processor
    else:
        raise RuntimeError
    return base_model

if __name__ == '__main__':
    args = get_args()
    # check num_classes
    nb_classes_map = {
        'CharadesEgo': 157,
        'CharadesEgo_v1_only3rd': 157,
        'Drone_Action': 13,
        'epic_noun': 300,
        'hmdb51': 51,
        'k400': 400,
        'k700': 700,
        'mit': 339,  
        'RareAct': 149,
        'ucf101': 101,
        'CharadesEgo_v1_only1st': 157,
        'diving48': 48,
        'epic_verb': 97,
        'k600': 600,
        'k710': 710,
        'perception_test': 63,
        'ssv2': 174,
        'SSV2': 174,
        }

    args.num_classes = nb_classes_map[args.dataset]

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
            backend="nccl",
            init_method="tcp://127.0.0.1:12584",
            rank=args.rank,
            world_size=args.world_size,
        )

    torch.cuda.set_device(args.local_rank)
    device = torch.device(args.local_rank)

    args.global_rank = args.rank
    setup_for_distributed(args.global_rank == 0)
    setup_seed(seed=args.seed, cuda_deterministic=False)

    if args.model_family == "siglip" or args.model_family == "dino_v2" or args.model_family == "clip" or args.model_family == "dino_v3":
        from ac_ap_dataloader_dali_no_norm import dali_dataloader
    print("create data loader start")
    data_loader_train = dali_dataloader(args.train_data_root_path,
                                        args.train_data_csv_path,
                                        args.dataset,
                                        dali_num_threads=args.dali_num_threads,
                                        dali_py_num_workers=args.dali_py_num_workers,
                                        batch_size=args.batch_size,
                                        input_size=args.input_size,
                                        short_side_size=args.short_side_size,
                                        sequence_length=args.num_frames,
                                        use_rgb=args.use_rgb,
                                        mean=args.mean,
                                        std=args.std,
                                        mode="train",
                                        seed=args.seed,
                                        num_shots=args.num_shots)
    args.total_batch_size = args.world_size * args.batch_size
    args.num_train_steps_per_epoch = len(data_loader_train)

    data_loader_val = dali_dataloader(args.val_data_root_path,
                                        args.val_data_csv_path,
                                        args.dataset,
                                        dali_num_threads=4,
                                        dali_py_num_workers=8,
                                        batch_size=args.batch_size,
                                        input_size=args.input_size,
                                        short_side_size=args.short_side_size,
                                        sequence_length=args.num_frames,
                                        use_rgb=args.use_rgb,
                                        mean=args.mean,
                                        std=args.std,
                                        mode="val",
                                        seed=1024,
                                        num_shots=args.num_shots)
    args.num_val_steps_per_epoch = len(data_loader_val)
    print("create data loader end")
    best_lr, max_acc_top1, max_acc_top5 = 0, 0, 0
    
    if isinstance(args.default_lr_list, float):
        args.default_lr_list = [args.default_lr_list]
    for lr in args.default_lr_list:
        base_model = get_model(args)

        acc_top1, acc_top5 = find_peak(args, lr, device,
                                       base_model, data_loader_train, data_loader_val)
        if max_acc_top1 < acc_top1:
            best_lr, max_acc_top1, max_acc_top5 = lr, acc_top1, acc_top5

    print("best_lr: ", best_lr, "max_acc_top1: ", max_acc_top1, "max_acc_top5: ", max_acc_top5)
    if args.global_rank == 0:

        args.save_path = os.path.join(args.save_report, "report_attentive_probe_{}_{}.txt".format(args.ckpt_path.split("/")[-1], args.num_shots))
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
        print("successfully save")
        with open(args.save_path, "a+") as writer:
            writer.write(str(args.dataset)+" "+str(max_acc_top1))
            writer.write("\n")
