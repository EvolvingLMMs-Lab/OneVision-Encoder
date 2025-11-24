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
from ac_ap_dataloader_dali_ip import dali_dataloader
from ac_ap_dataloader_dali_ip_mv import dali_dataloader as dali_dataloader_mv
#
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
from model_factory.layers import Siglip2MultiheadAttentionPoolingHead


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
    parser.add_argument('--using_mv', default=0, type=int)
    parser.add_argument('--num_target', default=1568, type=int)
    return parser.parse_args()



def get_feature(videos, res_zero_masks, processor, forward_base_model):
    # base model export feature
    if args.model_family == 'pe':
        pass

    elif args.model_family == "llava_vit":
        def mask_by_residual_topk(res: torch.Tensor, k_keep: int, patch_size=16):
            """
            先固定拿第一帧的 patch（数量 = hb*wb，典型为 196），
            再从剩余帧里按残差 Top-K 选，补齐到 k_keep（即 args.num_target）。

            Args:
                res:  (B, 1, T, H, W) —— 建议第一帧(I帧)残差已置 0
                k_keep: int            —— 目标总可见块数（含第一帧）
                patch_size: int        —— patch 尺寸，默认 16

            Returns:
                visible_indices: LongTensor (B, K) —— 选中的线性 patch 索引（升序）
            """
            assert res.dim() == 5 and res.size(1) == 1, "res 需为 (B,1,T,H,W)"
            B, _, T, H, W = res.shape
            ph, pw = patch_size, patch_size

            hb, wb = H // ph, W // pw           # 每帧的 patch 网格
            per_frame = hb * wb                 # 每帧 patch 数（典型为 196）
            L = T * per_frame                   # 总 patch 数

            # 总 K 边界
            K = int(max(0, min(k_keep, L)))

            # 计算每个 patch 的残差得分（|.| 在 patch 内求和） -> (B, T, hb, wb) -> (B, L)
            res_abs = res.abs().squeeze(1)                                   # (B,T,H,W)
            scores = res_abs.reshape(B, T, hb, ph, wb, pw).sum(dim=(3, 5))   # (B,T,hb,wb)
            scores = scores.reshape(B, L)                                    # (B, L)

            # 1) 先拿第一帧的索引（线性 0..per_frame-1）
            take_from_first = min(K, per_frame)
            if take_from_first > 0:
                first_frame_idx = torch.arange(per_frame, device=res.device, dtype=torch.long)
                first_frame_idx = first_frame_idx[:take_from_first]          # 若 K < per_frame，只取前 K 个
                first_frame_idx = first_frame_idx.unsqueeze(0).expand(B, -1) # (B, take_from_first)
            else:
                first_frame_idx = torch.empty(B, 0, dtype=torch.long, device=res.device)

            # 2) 剩余从后续帧中按 Top-K 选
            remain_k = K - take_from_first
            if remain_k > 0:
                # 剩余区域是线性 [per_frame, L)
                remain_scores = scores[:, per_frame:]                        # (B, L - per_frame)
                topk_idx_rem = torch.topk(remain_scores, k=remain_k, dim=1, largest=True, sorted=False).indices
                topk_idx_rem = topk_idx_rem + per_frame                      # 还原到全局线性下标
            else:
                topk_idx_rem = torch.empty(B, 0, dtype=torch.long, device=res.device)

            # 3) 合并并升序
            visible_indices = torch.cat([first_frame_idx, topk_idx_rem], dim=1)  # (B, K)
            if K > 0:
                visible_indices, _ = torch.sort(visible_indices, dim=1)

            return visible_indices

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            with torch.no_grad():
                visible_indices = mask_by_residual_topk(res_zero_masks, k_keep=args.num_target, patch_size=16)
                # print(visible_indices.max())
                # print(visible_indices.sum())
                # print(visible_indices)
                enc_out = forward_base_model(videos, visible_indices, mask_ratio=None)
                outputs = enc_out["visible_embeddings"]
        return outputs
 

    elif args.model_family == "llava_vit_mv":
        def mask_by_residual_topk(res: torch.Tensor, k_keep: int, patch_size: int = 16, max_index: int | None = None):
            """
            更简单的 Top-K 选择：直接在所有 patch（可选地限定前 max_index 个线性索引）上按残差取前 k_keep 个。

            Args:
                res:        (B, 1, T, H, W)
                k_keep:     目标总可见块数
                patch_size: patch 大小，默认 16
                max_index:  如果给定，只在线性索引 [0, max_index) 范围内选

            Returns:
                visible_indices: LongTensor (B, K')，K' = min(k_keep, 候选数)，每行升序
            """
            assert res.dim() == 5 and res.size(1) == 1, "res 需为 (B,1,T,H,W)"
            B, _, T, H, W = res.shape
            ph = pw = patch_size
            hb, wb = H // ph, W // pw
            per_frame = hb * wb
            L = T * per_frame

            K = int(max(0, min(k_keep, L)))
            if K == 0:
                return torch.empty(B, 0, dtype=torch.long, device=res.device)

            # 计算每个 patch 的 residual score
            scores = res.abs().squeeze(1)  # (B, T, H, W)
            scores = scores.reshape(B, T, hb, ph, wb, pw).sum(dim=(3, 5)).reshape(B, L)  # (B, L)

            # 限定候选区间（若提供）
            if max_index is not None:
                max_index = max(0, int(max_index))
                max_index = min(max_index, L)
                if max_index == 0:
                    return torch.empty(B, 0, dtype=torch.long, device=res.device)
                cand = scores[:, :max_index]
            else:
                cand = scores

            cand_len = cand.size(1)
            if K >= cand_len:
                idx = torch.arange(cand_len, device=res.device, dtype=torch.long)
                return idx.unsqueeze(0).expand(B, -1).clone()

            topk_idx = torch.topk(cand, k=K, dim=1, largest=True, sorted=False).indices  # (B, K)
            visible_indices, _ = torch.sort(topk_idx, dim=1)
            return visible_indices
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            with torch.no_grad():
                visible_indices = mask_by_residual_topk(res_zero_masks, k_keep=args.num_target, patch_size=16)
                enc_out = forward_base_model(videos, visible_indices, mask_ratio=None)
                outputs = enc_out["visible_embeddings"]
        return outputs


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
    last_val_stats = {"acc1": 0.0, "acc5": 0.0}

    for epoch in range(total_epochs):
        cur_ap_model.train()
        metric_logger = MetricLogger(delimiter="  ")
        header = f"Epoch: [{epoch}]"

        for data_iter_step, (videos, res_zero_masks, labels) in enumerate(
            metric_logger.log_every(
                data_loader_train,
                print_freq=args.print_freq,
                header=header,
                world_size=args.world_size,
                batch_size=args.batch_size
            )
        ):

            videos = videos.to(cur_device, non_blocking=True)
            res_zero_masks = res_zero_masks.to(cur_device, non_blocking=True)
            labels = labels.to(cur_device, non_blocking=True).view(-1)

            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=use_bf16, dtype=torch.bfloat16):
                    feats = get_feature(videos, res_zero_masks, processor, forward_base_model)
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
            # last_val_stats = val_stats
            if hasattr(data_loader_val, "reset"):
                data_loader_val.reset()
            print(f"[Val][Epoch {epoch}] acc1={val_stats.get('acc1', 0):.4f} acc5={val_stats.get('acc5', 0):.4f}")

            if val_stats.get("acc1", 0) > last_val_stats.get("acc1", 0):
                last_val_stats = val_stats

            print(f"Best so far: acc1={last_val_stats.get('acc1', 0):.4f} acc5={last_val_stats.get('acc5', 0):.4f}")

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
    for (videos, res_zero_masks, target) in metric_logger_val.log_every(data_loader_val, args.print_freq, header,
                                                        args.world_size, args.batch_size):
        videos = videos.to(cur_device, non_blocking = True)
        res_zero_masks = res_zero_masks.to(cur_device, non_blocking = True)
        target = target.to(cur_device, non_blocking = True)
        target = target.view(-1)
        outputs = get_feature(videos, res_zero_masks, processor, forward_base_model)
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

    if args.model_family != "dino_v3" and args.model_family != "ov_1_5_vit" and args.model_family != "rice":
        print("have load ckpt")
        base_model = load_finetune_checkpoint(args, base_model)

    base_model.to(cur_device).eval()

    attentive_probe_model = Siglip2MultiheadAttentionPoolingHead(
        hidden_size=args.embedding_size,
        num_attention_heads=args.embedding_size // 64,
        intermediate_size=args.embedding_size * 4,
    )

    ap_model = CustomModel(
        attentive_probe_model,
        attentive_dim=args.embedding_size,
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

    elif args.model_family in ['llava_vit', 'llava_vit_mv']:
        base_model = create_model(
            args.model_name,
            pretrained=True,
            ckpt_path=args.ckpt_path)
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

    args.rank = int(os.environ["RANK"])
    args.local_rank = int(os.environ["LOCAL_RANK"])
    args.world_size = int(os.environ["WORLD_SIZE"])
    distributed.init_process_group("nccl")

    torch.cuda.set_device(args.local_rank)
    device = torch.device(args.local_rank)

    args.global_rank = args.rank
    setup_for_distributed(args.global_rank == 0)
    setup_seed(seed=args.seed, cuda_deterministic=False)

    if args.using_mv:
        dali_dataloader = dali_dataloader_mv

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
