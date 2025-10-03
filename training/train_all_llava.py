import argparse
import logging
import os
import sys
import time
import json
from typing import List

import numpy as np
import torch
from torch import distributed
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter

import json
from safetensors.torch import load_file
from dataset import DATASET_REGISTRY, Property
from model_factory import MODEL_REGISTRY
from training.checkpoint_utils import load_checkpoint, save_checkpoint
from training.fused_partial_fc_v2 import CombinedMarginLoss, PartialFC_V2
from training.lr_scheduler import PolynomialLRWarmup
from training.parallel_softmax import ParallelCrossEntropy
from training.unmask_loss import UnmaskFC_V1
from training.mask_loss import MaskFC_V1
# from training.modeling_rice import Qwen2VLVisionConfig, Qwen2VisionTransformerPretrainedModel

from transformers import MLCDVisionModel

torch._dynamo.config.optimize_ddp = False

parser = argparse.ArgumentParser(description="")
parser.add_argument("--backward_passes_per_step", type=int, default=1)
parser.add_argument("--debug", type=int, default=0)
parser.add_argument("--dataloader-type", default="dali")
parser.add_argument("--dali_is_training", default=1, type=int)
parser.add_argument("--list_batch_size", nargs='+', default="128")
parser.add_argument("--list_dataset", nargs='+', default="laion400m")
parser.add_argument("--list_filter", nargs='+', default="0")
parser.add_argument("--list_margin", nargs='+', default="0.4")
parser.add_argument("--list_sample_rate", nargs='+', default="1")
parser.add_argument("--list_lr_pfc_weight", nargs='+', default="1")

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
parser.add_argument("--model_decoder_name", default="ViT-B/16")
parser.add_argument("--model_weight", default=None)
parser.add_argument("--teacher_model_path", type=str, default="/vlm/yinxie/code/checkpoints/rice-vit-large-patch14-560-w8b")
parser.add_argument("--opt", default="adamw")
parser.add_argument("--output", default="output")
parser.add_argument("--output_decoder", default="output_decoder")
parser.add_argument(
    "--random_diff",
    type=int,
    default=10,
)
parser.add_argument("--init_dir", default="NULL")
parser.add_argument("--init_mode", default="NULL")

parser.add_argument("--init_backbone", default="")
parser.add_argument("--init_decoder_backbone", default="")
parser.add_argument("--list_init_partial_fc", nargs='+', default="NULL")

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
    if hasattr(model, "module"):
        return model.module
    else:
        return model

def main():
    global_step = 0
    log = logging.getLogger()
    args.image_size = [int(x) for x in args.image_size.split(",")]
    if len(args.image_size) == 1:
        args.image_size = args.image_size * 2

    args.list_dataset = [
        DATASET_REGISTRY.get(x)() for x in args.list_dataset]

    # print(args.list_dataset)
    args.num_head = len(args.list_dataset)

    args.list_sample_rate = [float(x) for x in args.list_sample_rate]
    if len(args.list_sample_rate) == 1:
        args.list_sample_rate = args.list_sample_rate * args.num_head

    args.list_margin = [float(x) for x in args.list_margin]
    if len(args.list_margin) == 1:
        args.list_margin = args.list_margin * args.num_head

    args.list_filter = [float(x) for x in args.list_filter]
    if len(args.list_filter) == 1:
        args.list_filter = args.list_filter * args.num_head

    args.list_lr_pfc_weight = [float(x) for x in args.list_lr_pfc_weight]
    if len(args.list_lr_pfc_weight) == 1:
        args.list_lr_pfc_weight = args.list_lr_pfc_weight * args.num_head

    args.list_batch_size = [int(x) for x in args.list_batch_size]
    args.batch_size = sum(args.list_batch_size)

    args.list_head_name = [x.name for x in args.list_dataset]
    args.total_steps = int(args.num_sampled_data / args.batch_size / world_size)

    for arg in vars(args):
        msg = f"{format(arg, '<30')}  {format(str(getattr(args, arg)))}"
        log.info(msg)

    # student backbone
    # video rope
    backbone = MODEL_REGISTRY.get(args.model_name)().train().cuda()
    decoder_backbone = MODEL_REGISTRY.get(args.model_decoder_name)().train().cuda()
    # teacher backbone
    teacher_backbone = MLCDVisionModel.from_pretrained(args.teacher_model_path).cuda()
    # teacher_backbone = RiceEncoder(model_path=args.teacher_model_path).cuda()

    if args.init_backbone == "NULL":
        pass
    else:
        assert os.path.exists(args.init_backbone)
        state_dict = torch.load(args.init_backbone, "cpu")
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        # True 改为 False
        backbone.load_state_dict(state_dict, strict=False)
    
    if args.init_decoder_backbone == "NULL":
        pass
    else:
        assert os.path.exists(args.init_decoder_backbone)
        state_dict = torch.load(args.init_decoder_backbone, "cpu")
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        decoder_backbone.load_state_dict(state_dict, strict=False)

    backbone.requires_grad_(bool(args.finetune_backbone))
    # backbone = torch.compile(backbone)
    backbone_parameters = filter(lambda p: p.requires_grad, backbone.parameters())

    decoder_backbone.requires_grad_(bool(args.finetune_backbone))
    # decoder_backbone = torch.compile(decoder_backbone)
    decoder_backbone_parameters = filter(lambda p: p.requires_grad, decoder_backbone.parameters())


    dict_pfc_modules = {} # 给我加个脚注
    # dict_pfc_modules: dict[str, PartialFC_V2] = {}

    list_module_pfc: List[PartialFC_V2] = []
    parameters: List[dict] = [
        {"params": backbone_parameters},
        {"params": decoder_backbone_parameters}
    ]

    # print("len(args.num_head)", args.num_head)
    for head_id, _ in enumerate(range(args.num_head)):
        head_name = args.list_head_name[head_id]
        dataset_config = args.list_dataset[head_id]
        dataset_config: Property
        # import pdb; pdb.set_trace()
        # print("dataset_config.num_class", dataset_config.num_class)
        # FIXME
        
        if hasattr(dataset_config, "num_class"):  # 先检查有没有这个属性
            if isinstance(dataset_config.num_class, list):
        # if isinstance(dataset_config.num_class, list):
                list_this_head_pfc = []
                for num_class in dataset_config.num_class:
                    # print("num_class zhuanyong", num_class)
                    assert dataset_config.pfc_type == "partial_fc"
                    margin_loss = CombinedMarginLoss(
                        64, 1, 0, args.list_margin[head_id], args.list_filter[head_id]
                    )
                    partial_fc = PartialFC_V2(
                        margin_loss,
                        args.embedding_size,
                        num_class,
                        args.list_sample_rate[head_id],
                        fp16=False,
                    )
                    partial_fc.train().cuda()
                    list_this_head_pfc.append(partial_fc)
                    # Define the learning rate
                    lr_pfc = args.lr * args.list_lr_pfc_weight[head_id]
                    # Append the parameters with the determined learning rate
                    parameters.append(
                        {
                            "params": partial_fc.parameters(),
                            "lr": lr_pfc,
                            "weight_decay": args.weight_decay_pfc,
                        }
                    )
                    
                dict_pfc_modules[head_name] = list_this_head_pfc
                list_module_pfc.append(list_this_head_pfc)
        elif isinstance(dataset_config.pfc_type, list):
            list_this_head_fc = []
            for pfc_type in dataset_config.pfc_type:
                assert pfc_type in ["partial_fc", "parallel_softmax", "unmask", "mask"]
                if pfc_type == "unmask":
                    llava_fc = UnmaskFC_V1()
                    llava_fc.train().cuda()
                    list_this_head_fc.append((llava_fc, "unmasked_embeddings")) 
                    lr_pfc = args.lr * args.list_lr_pfc_weight[head_id]
                    parameters.append(
                        {
                            "params": llava_fc.parameters(),
                            "lr": lr_pfc,
                            "weight_decay": args.weight_decay_pfc,
                        }
                    )
                elif pfc_type == "mask":
                    llava_fc = MaskFC_V1()
                    llava_fc.train().cuda()
                    list_this_head_fc.append((llava_fc, "masked_embeddings")) 
                    lr_pfc = args.lr * args.list_lr_pfc_weight[head_id]
                    parameters.append(
                        {
                            "params": llava_fc.parameters(),
                            "lr": lr_pfc,
                            "weight_decay": args.weight_decay_pfc,
                        }
                    )
            list_module_pfc.append(list_this_head_fc)
            # list_module_pfc.append(llava_fc)
            dict_pfc_modules[head_name] = list_this_head_fc
        else:
            if dataset_config.pfc_type == "partial_fc":
                margin_loss = CombinedMarginLoss(
                    64, 1, 0, args.list_margin[head_id], args.list_filter[head_id]
                )
                # print('dataset_config.num_class', dataset_config.num_class)
                partial_fc = PartialFC_V2(
                    margin_loss,
                    args.embedding_size,
                    dataset_config.num_class,
                    args.list_sample_rate[head_id],
                    fp16=False,
                )
            elif dataset_config.pfc_type == "parallel_softmax":

                partial_fc = ParallelCrossEntropy(
                    args.embedding_size,
                    dataset_config.num_class,
                )

            partial_fc.train().cuda()
            list_module_pfc.append(partial_fc)
            dict_pfc_modules[head_name] = partial_fc

            # Define the learning rate
            lr_pfc = args.lr * args.list_lr_pfc_weight[head_id]
            # Append the parameters with the determined learning rate
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

    if args.opt == "sgd":
        opt = torch.optim.SGD(parameters, lr=args.lr, weight_decay=args.weight_decay)
        lr_scheduler = PolynomialLRWarmup(
            opt, int(args.total_steps * args.warmup_ratio), args.total_steps, 2
        )

    elif args.opt == "adamw":
        optimizer_cls = torch.optim.AdamW

        opt = optimizer_cls(parameters, lr=args.lr, weight_decay=args.weight_decay)
        lr_scheduler = PolynomialLRWarmup(
            opt, int(args.total_steps * args.warmup_ratio), args.total_steps, 2
        )
    elif args.opt == "lamb":
        import torch_optimizer as optim  # 需要安装 torch_optimizer 包
        optimizer_cls = optim.Lamb
        opt = optimizer_cls(parameters, lr=args.lr, weight_decay=args.weight_decay)
        lr_scheduler = PolynomialLRWarmup(
            opt, int(args.total_steps * args.warmup_ratio), args.total_steps, 2
        )
    else:
        raise ValueError(f"{args.opt} not support!")

    # TODO 需要加上加载decoder的代码
    result = load_checkpoint(
        args.output,
        None,
        backbone,
        list_module_pfc,
        lr_scheduler,
        None,
        args.list_head_name,
    )

    if result is not None:
        global_step = result["global_step"]
        log.info(f"load checkpoint from {global_step}")
    else:
        global_step = 0


    if args.finetune_backbone:
        backbone = torch.nn.parallel.DistributedDataParallel(
            module=backbone,
            broadcast_buffers=False,
            device_ids=[local_rank],
            bucket_cap_mb=32,
            find_unused_parameters=True,
            static_graph=True,
        )
        decoder_backbone = torch.nn.parallel.DistributedDataParallel(
            module=decoder_backbone,
            broadcast_buffers=False,
            device_ids=[local_rank],
            bucket_cap_mb=32,
            find_unused_parameters=True,
            static_graph=True,
        )
        teacher_backbone = torch.nn.parallel.DistributedDataParallel(
            module=teacher_backbone,
            broadcast_buffers=False,
            device_ids=[local_rank],
            bucket_cap_mb=32,
            find_unused_parameters=True,
            static_graph=True,
        )
        for k, v in dict_pfc_modules.items():
            if isinstance(v, list):  
                new_list = []
                for module in v:
                    if isinstance(module, tuple):  
                        # 保留 (module, tag) 里的 tag
                        ddp_module = torch.nn.parallel.DistributedDataParallel(
                            module=module[0],
                            broadcast_buffers=False,
                            device_ids=[local_rank],
                            bucket_cap_mb=32,
                            find_unused_parameters=True,
                            static_graph=True,
                        )
                        new_list.append((ddp_module, module[1]))
                    else:
                        ddp_module = torch.nn.parallel.DistributedDataParallel(
                            module=module,
                            device_ids=[local_rank],
                            broadcast_buffers=False,
                            find_unused_parameters=True,
                            static_graph=True,
                        )
                        new_list.append(ddp_module)
                dict_pfc_modules[k] = new_list
            else:
                if isinstance(v, tuple):
                    ddp_module = torch.nn.parallel.DistributedDataParallel(
                        module=v[0],
                        device_ids=[local_rank],
                        broadcast_buffers=False,
                        find_unused_parameters=True,
                        static_graph=True,
                    )
                    dict_pfc_modules[k] = (ddp_module, v[1])  # 保留 tag
                else:
                    dict_pfc_modules[k] = torch.nn.parallel.DistributedDataParallel(
                        module=v,
                        device_ids=[local_rank],
                        broadcast_buffers=False,
                        find_unused_parameters=True,
                        static_graph=True,
                    )


    list_dali_dataloader = []
    list_head_name = []
    # print(len(args.list_dataset))
    for head_id, dataset_config in enumerate(args.list_dataset):

        if args.debug:
            from dataloader.data_v2_video import SyntheticDataIter
            train_iter = SyntheticDataIter(args.batch_size, args.image_size[0], local_rank)


        elif dataset_config.dali_type == "parallel_rec":
            from dataloader.data_v2_parallel_rec import ParallelDALIIterator
            print("dataset_config.prefix", dataset_config.prefix)
            train_iter = ParallelDALIIterator(
                list_prefix=dataset_config.prefix,
                batch_size=args.list_batch_size[head_id],
                image_size=args.image_size,
                workers=args.workers,
                shard_id=dataset_config.shard_id,
                num_shards=dataset_config.num_shards
        )

        elif dataset_config.dali_type == "ocr":
            from dataloader.data_v2_ocr import MultiRecDALIWarper
            train_iter = MultiRecDALIWarper(
                list_prefix=dataset_config.prefix,
                batch_size=args.list_batch_size[head_id],
                image_size=args.image_size,
                workers=args.workers,
                shard_id=dataset_config.shard_id,
                num_shards=dataset_config.num_shards,
            )

        elif dataset_config.dali_type == "video":
            # print("用的video数据集")
            from dataloader.data_v2_video import MultiRecDALIWarper
            train_iter = MultiRecDALIWarper(
                list_prefix=dataset_config.prefix,
                batch_size=args.list_batch_size[head_id],
                image_size=args.image_size,
                workers=args.workers,
                shard_id=dataset_config.shard_id,
                num_shards=dataset_config.num_shards,
            )

        elif dataset_config.dali_type == "decord":
            from dataloader.data_decord_video_IP_v1 import dali_dataloader
            # print("用的video decord数据集")
            # num_workers = args.list_batch_size[head_id] // 2
            # num_workers = max(1, num_workers)
            # num_workers = min(num_workers, 4)
            num_workers = 4


            train_iter = dali_dataloader(
                file_list=dataset_config.prefix,
                dali_num_threads=2,
                dali_py_num_workers=num_workers,
                batch_size=args.list_batch_size[head_id],
                input_size=args.image_size[0],
                sequence_length=args.num_frames,
                stride=8,
                seed=0+rank,
                short_side_size=256 / 224 * args.image_size[0],
                shard_id=dataset_config.shard_id,
                num_shards=dataset_config.num_shards,
            )

        elif dataset_config.dali_type == "decord_torch":
            from dataloader.data_decord_torch import create_video_dataloader
            num_workers = args.list_batch_size[head_id] // 2

            num_workers = max(1, num_workers)
            num_workers = min(num_workers, 8)

            train_iter = create_video_dataloader(
                file_list=dataset_config.prefix,
                batch_size=args.list_batch_size[head_id],
                num_workers=args.workers,
                input_size=args.image_size[0],
                sequence_length=args.num_frames,
                seed=0+rank,
                num_shard=dataset_config.num_shards,
                shard_id=dataset_config.shard_id,
            )
        list_dali_dataloader.append(train_iter)

        list_head_name.append(dataset_config.name)
    

    batch_end_callback = BatchEndCallBack(
        frequent=args.frequent,
        list_head_name=list_head_name,
        output=args.output,
        total_steps=args.total_steps,
    )
    tensorboard_logger = TensorboardLogger(args.output)

    list_iter = []
    list_next_data_batch = []
    for i in range(args.num_head):
        # list_dali_dataloader[i].reset()
        list_iter.append(iter(list_dali_dataloader[i]))
        list_next_data_batch.append(next(list_iter[i]))

    if global_step > args.total_steps:
        log.info("global_step > total_steps")
        exit()

    end_of_batch = False
    while not end_of_batch:
        list_data_batch = list_next_data_batch
        # list_data:(b, 3, 16, h, w), list_i:(b, num_i_frames) list_P:(b, num_p_frames)
        list_data, list_I, list_P = combine(list_data_batch)

        # print(list_I, list_P)
        list_output = []
        list_batch_size = []

        # print(list_data[head_id].shape)


        for head_id, dataset_config in enumerate(args.list_dataset):
            head_input = list_data[head_id]
            # teacher_head_input = list_teacher_data[head_id]
            
            dataset_config: Property
            if dataset_config.dali_type in ["decord", "decord_torch"]:
                head_input = list_data[head_id]
                list_batch_size.append(head_input.size(0))
                
                # import pdb; pdb.set_trace()
                with torch.amp.autocast(dtype=torch.bfloat16, device_type="cuda"):
                    # backbone_time = time.time()
                    output, ids_restore = backbone(head_input, list_I[head_id], list_P[head_id])
                    # decoder_time = time.time()
                    output["masked_embeddings"], mask_pos = decoder_backbone(output["masked_embeddings"], ids_restore=ids_restore)
                
                with torch.no_grad():
                    with torch.amp.autocast(dtype=torch.bfloat16, device_type="cuda"):
                        # import pdb; pdb.set_trace()
                        # teacher_time = time.time()
                        if len(head_input.shape) == 5:
                            b, c, t, h, w = head_input.shape
                            head_input = head_input.reshape(b*t, c, h, w)
                        teacher_initial_output = teacher_backbone(head_input).last_hidden_state[:, 1:, :]
                        patches_per_frame = teacher_initial_output.shape[1]
                        teacher_initial_output = teacher_initial_output.reshape(b, -1, teacher_initial_output.shape[-1])
                        teacher_output = get_output(teacher_initial_output, b, patches_per_frame, list_I[head_id], list_P[head_id])
                        # teacher_output = teacher_backbone(teacher_head_input, list_I[i], list_P[i])
                        # import pdb; pdb.set_trace()
                        B, _, D = teacher_output["masked_embeddings"].shape
                        # end_time = time.time()
                        teacher_output["masked_embeddings"] = teacher_output["masked_embeddings"][mask_pos].view(B, -1, D)


                # print("output.shape", output["head_embeddings"][0].shape)
                if isinstance(output, torch.Tensor):
                    output = output.float()
                    teacher_output = teacher_output.float()

            elif dataset_config.dali_type in ["parallel_rec"]:
                head_input = list_data[head_id]
                list_batch_size.append(head_input.size(0))
                with torch.amp.autocast(dtype=torch.bfloat16, device_type="cuda"):
                    # print("head_input", head_input.shape)
                    output = backbone(head_input)
                    # print("output.shape", output.shape)
                output = output.float()

            else:
                head_input = list_data[head_id]
                list_batch_size.append(head_input.size(0))
                with torch.amp.autocast(dtype=torch.bfloat16, device_type="cuda"):
                    output = backbone(head_input)
                    # print("output.shape", output)
                if isinstance(output, torch.Tensor):
                    output = output.float()
            
            list_output.append(output)


        list_loss = []
        list_loss_float = []
        list_loss_unmask = []
        list_loss_mask = []
        list_loss_float_unmask = []
        list_loss_float_mask = []


        
        # print("list_module_pfc", len(list_module_pfc))
        for head_id, pfc in enumerate(list_module_pfc):
            dataset_config = args.list_dataset[head_id]
            head_backbone_output = list_output[head_id]
            # head_label = list_label[head_id].clone()
            
            if isinstance(pfc, list) and isinstance(head_backbone_output, dict):
                unmask_embedding = head_backbone_output["unmasked_embeddings"]
                mask_embedding = head_backbone_output["masked_embeddings"]
                teacher_unmask_embedding = teacher_output["unmasked_embeddings"]
                teacher_mask_embedding = teacher_output["masked_embeddings"]
                # print(head_backbone_output["x_without_class"].shape)

                for i in range(len(pfc)):
                    llava_fc, pfc_type = pfc[i]
                    with torch.amp.autocast(dtype=torch.bfloat16, device_type="cuda"):
                        if pfc_type == "unmasked_embeddings":
                            head_embedding = unmask_embedding
                            teacher_head_embedding = teacher_unmask_embedding
                            head_loss = llava_fc(head_embedding, teacher_head_embedding)
                            list_loss_float_unmask.append(head_loss.float())
                            list_loss_unmask.append(head_loss)
                            # print("unmasked_embeddings_head_loss", head_loss.item())

                        elif pfc_type == "masked_embeddings":
                            head_embedding = mask_embedding
                            teacher_head_embedding = teacher_mask_embedding
                            # teacher_mask_embedding.reshape()
                            head_loss = llava_fc(head_embedding, teacher_head_embedding)
                            list_loss_float_mask.append(head_loss.float())
                            list_loss_mask.append(head_loss)
                            # print("masked_embeddings_head_loss", head_loss.item())

                    
                    list_loss.append(head_loss)
                    list_loss_float.append(head_loss.float())
                    # list_norm.append(compute_norm(list_head_norms[i]))
                    # list_prob.append(sorted_probs)
                    


        loss_unmask = torch.stack(list_loss_float).mean()
        loss_mask   = torch.stack(list_loss_mask).mean()   

        total_loss = loss_unmask + loss_mask

        opt.zero_grad()
        total_loss.backward()
        if global_step % args.backward_passes_per_step == 0:
            clip_grad_norm_(backbone.parameters(), max_norm=5, norm_type=2)
            # import pdb; pdb.set_trace()
            for pfc in list_module_pfc:
                if isinstance(pfc, list):
                    for x, pfc_type in pfc:
                        clip_grad_norm_(x.parameters(), max_norm=5, norm_type=2)
                else:
                    clip_grad_norm_(pfc.parameters(), max_norm=5, norm_type=2)
            opt.step()
            opt.zero_grad()
        lr_scheduler.step()

        batch_end_callback(
            global_step, lr_scheduler, list_loss_float_mask, list_loss_float_unmask,args.batch_size
        )

        global_step += 1

        for i in range(args.num_head):
            list_next_data_batch[i] = next(list_iter[i])


        for i in range(args.num_head):
            try:
                list_next_data_batch[i] = next(list_iter[i])
            except StopIteration:

                list_dali_dataloader[i].reset()
                list_iter[i] = iter(list_dali_dataloader[i])
                list_next_data_batch[i] = next(list_iter[i])


        if global_step % args.ckpt_interval == 0:
            save_checkpoint(
                args.output,
                backbone,
                list_module_pfc,
                lr_scheduler,
                None,
                global_step,
                args.list_head_name,
                keep_num=200,
            )
            save_checkpoint(
                args.output_decoder,
                decoder_backbone,
                list_module_pfc,
                lr_scheduler,
                None,
                global_step,
                args.list_head_name,
                keep_num=200,
            )

        if global_step > args.total_steps:
            save_checkpoint(
                args.output,
                backbone,
                list_module_pfc,
                lr_scheduler,
                None,
                global_step,
                args.list_head_name,
                keep_num=200,
            )
            save_checkpoint(
                args.output_decoder,
                decoder_backbone,
                list_module_pfc,
                lr_scheduler,
                None,
                global_step,
                args.list_head_name,
                keep_num=200,
            )
            exit()


# def compute_cosine_with_

def compute_norm(list_embeddings):
    norms = list_embeddings.norm(p=2, dim=1)
    return norms.mean().item()

def get_output(hidden_states, B, patches_per_frame, list_I, list_P):
    all_I, all_P = [], []
    # print("get_output list_I", list_I)
    # print("patches_per_frame", patches_per_frame)
    for b in range(B):
        I_patch_idx = []
        for i in list_I[b].tolist():   # 当前 batch 的 I 帧索引
            start = i * patches_per_frame
            end = (i + 1) * patches_per_frame
            I_patch_idx.extend(range(start, end))

        P_patch_idx = []
        for p in list_P[b].tolist():   # 当前 batch 的 P 帧索引
            start = p * patches_per_frame
            end = (p + 1) * patches_per_frame
            P_patch_idx.extend(range(start, end))

        I_patch_idx = torch.tensor(I_patch_idx, dtype=torch.long, device=hidden_states.device)
        P_patch_idx = torch.tensor(P_patch_idx, dtype=torch.long, device=hidden_states.device)

        all_I.append(hidden_states[b, I_patch_idx, :])  # (num_I_patches, D)
        all_P.append(hidden_states[b, P_patch_idx, :])  # (num_P_patches, D)

    # 按 batch 打包回来
    unmask_hidden_states = torch.stack(all_I, dim=0)  # (B, num_I_patches, D)
    mask_hidden_states   = torch.stack(all_P, dim=0)  # (B, num_P_patches, D)
    # end_time = time.time()
    # print("end_time", end_time-start_time)
    return {
        "unmasked_embeddings": unmask_hidden_states,
        "masked_embeddings": mask_hidden_states,
    }


def visualize_batch(batch_img, img, global_step, output_dir, vis_samples=2, device=None):
    """
    Visualize a batch of images and their corresponding processed video frames.
    
    Args:
        batch_img: Original batch of grid images [batch_size, 3, H, W]
        img: Processed video tensors [batch_size, 3, num_frames, height, width]
        global_step: Current training step
        output_dir: Base directory for saving visualizations
        vis_samples: Number of samples to visualize
        device: Device where tensors are stored
    """
    import os
    from pathlib import Path

    import torchvision
    
    try:
        # Create visualization directory
        vis_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        
        # Use ImageNet mean and std for denormalization
        mean = [x * 255 for x in [0.48145466, 0.4578275, 0.40821073]]
        std = [x * 255 for x in [0.26862954, 0.26130258, 0.27577711]]
        
        # Convert to tensors for easier manipulation
        mean_tensor = torch.tensor(mean).view(1, 3, 1, 1).to(batch_img.device if device is None else device)
        std_tensor = torch.tensor(std).view(1, 3, 1, 1).to(batch_img.device if device is None else device)
        
        # Save original grid images with proper denormalization
        for i in range(min(vis_samples, batch_img.shape[0])):
            # Denormalize: pixel = pixel * std + mean
            denorm_img = batch_img[i:i+1] * std_tensor + mean_tensor
            # Clip values to valid range for the image
            denorm_img = torch.clamp(denorm_img / 255.0, 0.0, 1.0)
            
            grid_path = os.path.join(vis_dir, f"step_{global_step}_grid_sample_{i}.png")
            torchvision.utils.save_image(denorm_img, grid_path)
            logging.debug(f"Saved grid visualization to {grid_path}")
        
        # Save video sequences as GIFs
        for i in range(min(vis_samples, img.shape[0])):
            video_tensor = img[i]  # [3, num_frames, cell_size, cell_size]
            save_path = os.path.join(vis_dir, f"step_{global_step}_video_sample_{i}.gif")
            saved_path = save_video_as_gif(video_tensor, save_path, mean=mean, std=std)
            logging.info(f"Saved video visualization to {saved_path}")
            
        return vis_dir
    except Exception as e:
        logging.error(f"Error during visualization: {e}")
        return None


def save_video_as_gif(tensor, path, fps=10, mean=None, std=None):
    """
    Save a video tensor as a GIF file with proper denormalization.
    
    Args:
        tensor: Tensor of shape [3, num_frames, height, width]
        path: Path to save the GIF
        fps: Frames per second for the GIF
        mean: Mean values used for normalization [R, G, B]
        std: Standard deviation values used for normalization [R, G, B]
    """
    import os
    from pathlib import Path

    import imageio

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


class TensorboardLogger(object):
    def __init__(self, output: str):
        self.output = output
        self.summary = None
        if rank == 0:
            self.summary = SummaryWriter(os.path.join(output, "tensorboard"))

    def flush(self):
        """Flush TensorBoard writer periodically"""
        if rank == 0:
            self.summary.flush()
    
    def add_scalar(self, tag: str, value: float, global_step: int, new_style: bool = True):
        """Add scalar value to TensorBoard"""
        if rank == 0:
            self.summary.add_scalar(tag, value, global_step)

    def add_histogram(self, tag: str, values: torch.Tensor, global_step: int):
        """Add histogram to TensorBoard"""
        if rank == 0:
            self.summary.add_histogram(tag, values, global_step,)
    
    def add_embedding(self, tag: str, embedding: torch.Tensor, global_step: int):
        """Add embedding to TensorBoard"""
        if rank == 0:
            self.summary.add_embedding(embedding, global_step=global_step, tag=tag)

    def log_backbone_lr(self, lr_value: float, global_step: int):
        """Log backbone learning rate"""
        if rank == 0:
            self.summary.add_scalar("backbone_lr", lr_value, global_step, new_style=True)
    
    def log_head_metrics(self, head_name: str, loss_value: float, lr_value: float, global_step: int):
        """Log head-specific loss and learning rate"""
        if rank == 0:
            self.summary.add_scalar(f"loss_{head_name}", loss_value, global_step, new_style=True)
            self.summary.add_scalar(f"lr_{head_name}", lr_value, global_step, new_style=True)


class BatchEndCallBack(object):
    def __init__(
        self,
        frequent: int,
        list_head_name: List[str],
        output: str,
        total_steps: int,
    ):
        self.frequent: int = frequent
        self.list_head_name: List[str] = list_head_name
        self.output: str = output
        self.total_steps: int = total_steps

        self.num_head = len(self.list_head_name)
        self.time_start = time.time()
        self.list_loss_metric = [[ScalaMetric(), ScalaMetric()] for x in self.list_head_name]
        self.init = False
        self.tic = 0
        self.summary = None
        # if rank == 0:
        #     self.summary = SummaryWriter(os.path.join(output, "tensorboard"))
        
        # 用于计算平均每步时间
        self.step_times = []
        self.max_time_history = 100  # 保留最近100个step的时间来平均

    def __call__(
        self,
        global_step: int,
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
        list_loss_float1: List[float],
        list_loss_float2: List[float],
        batch_size: int,
        # avg_norm,
        # prob,
    ):
        for i in range(self.num_head):
            self.list_loss_metric[i][0].update(list_loss_float1[i])
            self.list_loss_metric[i][1].update(list_loss_float2[i])

        if global_step > 0 and global_step % self.frequent == 0:
            # if rank == 0:
            #     self.summary.add_scalar(
            #         "backbone_lr",
            #         lr_scheduler.get_last_lr()[0],
            #         global_step,
            #         new_style=True,
            #     )
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

                loss_str_format = ""
                for head_id, name in enumerate(self.list_head_name):
                    # if rank == 0:
                    #     self.summary.add_scalar(
                    #         f"loss_{name}",
                    #         self.list_loss_metric[head_id].avg,
                    #         global_step,
                    #         new_style=True,
                    #     )
                    #     self.summary.add_scalar(
                    #         f"lr_{name}",
                    #         lr_scheduler.get_last_lr()[head_id + 1],
                    #         global_step,
                    #         new_style=True,
                    #     )
                    _ = "\n"
                    _ += format(f"name: {self.list_head_name[head_id]}", "<50")
                    _ += format(
                        f"lr: {lr_scheduler.get_last_lr()[head_id + 1] :.8f}", "<20"
                    )
                    _ += format(
                        f"mask loss: {self.list_loss_metric[head_id][0].avg :.4f}", "<20"
                    )
                    _ += format(
                        f"unmask loss: {self.list_loss_metric[head_id][1].avg :.4f}", "<20"
                    )

                    loss_str_format += _
                    self.list_loss_metric[head_id][0].reset()
                    self.list_loss_metric[head_id][1].reset()

                msg = (
                    "rank %.2f total %.2f its/s lr: %.8f step: %d/%d (%.2f%%) remain: %.2f hours %s "
                    % (
                        speed,
                        speed_total,
                        lr_scheduler.get_last_lr()[0],
                        global_step,
                        self.total_steps,
                        (global_step / self.total_steps) * 100,
                        remaining_time_hours,
                        loss_str_format,
                        # avg_norm,
                        # prob[0],
                        # prob[1],
                        # prob[2],
                    )
                )

                if rank == 0:
                    logging.info(msg)
                    # self.summary.flush()
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


def combine(data_batches):
    list_data = [x[0].cuda() for x in data_batches]
    # list_techer_data = [x[1].cuda() for x in data_batches]
    list_I = [x[1] for x in data_batches]
    list_P = [x[2] for x in data_batches]
    return list_data, list_I, list_P


def reshape_grid_to_seqs(image_batch, num_frames, grid_shape=(4, 4), cell_size=224):
    """
    Reshape a batch of grid images to video sequences
    
    Args:
        image_batch: Tensor of shape [batch_size, 3, H, W] containing grid images
        num_frames: Number of frames to extract for each video
        grid_shape: Tuple (rows, cols) specifying the grid layout
        cell_size: Size of each cell in pixels
        
    Returns:
        Tensor of shape [batch_size, 3, num_frames, cell_size, cell_size]
    """
    batch_size = image_batch.shape[0]
    rows, cols = grid_shape
    cells_per_image = rows * cols
    stride = cells_per_image // num_frames
    
    # First reshape to separate out the rows and columns
    # [batch_size, 3, rows*cell_size, cols*cell_size] -> [batch_size, 3, rows, cell_size, cols, cell_size]
    reshaped = image_batch.view(batch_size, 3, rows, cell_size, cols, cell_size)
    
    # Permute to get [batch_size, 3, rows, cols, cell_size, cell_size]
    # This keeps channels (3) at dimension 1
    permuted = reshaped.permute(0, 1, 2, 4, 3, 5)
    
    # Reshape to [batch_size, 3, rows*cols, cell_size, cell_size]
    frames = permuted.reshape(batch_size, 3, rows*cols, cell_size, cell_size)
    
    # Select every stride-th frame to get num_frames
    # Result: [batch_size, 3, num_frames, cell_size, cell_size]
    frames = frames[:, :, ::stride, :, :]
    
    return frames


def format_value(value, indent_level=0):
    """Format values with special handling for lists and Property objects."""
    indent = ' ' * 4 * indent_level
    
    if isinstance(value, list):
        if not value:
            return "[]"
        
        result = "[\n"
        for item in value:
            formatted_item = format_value(item, indent_level + 1)
            result += f"{indent}    {formatted_item},\n"
        result += f"{indent}]"
        return result
    elif hasattr(value, '__repr__') and not isinstance(value, (str, int, float, bool, type(None))):
        # For objects with custom __repr__ methods like Property
        # Strip leading/trailing whitespace and adjust indentation
        repr_str = value.__repr__().strip()
        lines = repr_str.split('\n')
        formatted_lines = [lines[0]]
        for line in lines[1:]:
            formatted_lines.append(f"{indent}{line}")
        return '\n'.join(formatted_lines)
    
    return str(value)


def log_args(args, log):
    """Log arguments with improved formatting for nested structures."""
    for arg in vars(args):
        arg_name = format(arg, '<30')
        value = getattr(args, arg)
        
        if isinstance(value, list):
            msg = f"{arg_name} {format_value(value)}"
            log.info(msg)
        else:
            msg = f"{arg_name} {str(value)}"
            log.info(msg)


if __name__ == "__main__":
    main()
