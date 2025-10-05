import os
import sys

sys.dont_write_bytecode = True
path = os.path.join(os.path.dirname(__file__), "../../..")
if path not in sys.path:
    sys.path.insert(0, path)

import tqdm
import argparse
import json
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from mmengine.config import Config
from opentad.cores.test_engine import gather_ddp_results
from opentad.models import build_detector
from opentad.models.utils.post_processing import build_classifier
from opentad.datasets import build_dataset, build_dataloader
from opentad.utils import update_workdir, set_seed, create_folder, setup_logger
from opentad.datasets.base import SlidingWindowDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Test a Temporal Action Detector")
    parser.add_argument("config", metavar="FILE", type=str, help="path to config file")
    parser.add_argument("--checkpoint", type=str, default="none", help="the checkpoint path")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--id", type=int, default=0, help="repeat experiment id")
    parser.add_argument("--pre_nms_topk", type=int, default=50000, help="max predictions before nms")
    parser.add_argument("--max_seg_num", type=int, default=8000, help="max predictions per video")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)

    # override to test configuration
    cfg.dataset.test.subset_name = "test"

    # DDP init
    args.local_rank = int(os.environ["LOCAL_RANK"])
    args.world_size = int(os.environ["WORLD_SIZE"])
    args.rank = int(os.environ["RANK"])
    print(f"Distributed init (rank {args.rank}/{args.world_size}, local rank {args.local_rank})")
    dist.init_process_group("nccl", rank=args.rank, world_size=args.world_size)
    torch.cuda.set_device(args.local_rank)

    # set random seed, create work_dir
    set_seed(args.seed)
    cfg = update_workdir(cfg, args.id, torch.cuda.device_count())
    if args.rank == 0:
        create_folder(cfg.work_dir)

    # setup logger
    logger = setup_logger("Test", save_dir=cfg.work_dir, distributed_rank=args.rank)
    logger.info(f"Using torch version: {torch.__version__}, CUDA version: {torch.version.cuda}")
    logger.info(f"Config: \n{cfg.pretty_text}")

    # build dataset
    test_dataset = build_dataset(cfg.dataset.test, default_args=dict(logger=logger))
    test_loader = build_dataloader(
        test_dataset,
        rank=args.rank,
        world_size=args.world_size,
        shuffle=False,
        drop_last=False,
        **cfg.solver.test,
    )

    # build model
    model = build_detector(cfg.model)

    # DDP
    model = model.to(args.local_rank)
    model = DistributedDataParallel(model, device_ids=[args.local_rank])
    logger.info(f"Using DDP with {torch.cuda.device_count()} GPUS...")

    # load checkpoint: args -> config -> best
    if args.checkpoint != "none":
        checkpoint_path = args.checkpoint
    elif "test_epoch" in cfg.inference.keys():
        checkpoint_path = os.path.join(cfg.work_dir, f"checkpoint/epoch_{cfg.inference.test_epoch}.pth")
    else:
        checkpoint_path = os.path.join(cfg.work_dir, "checkpoint/best.pth")
    logger.info("Loading checkpoint from: {}".format(checkpoint_path))
    device = f"cuda:{args.rank % torch.cuda.device_count()}"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    logger.info("Checkpoint is epoch {}.".format(checkpoint["epoch"]))

    # Model EMA
    use_ema = getattr(cfg.solver, "ema", False)
    if use_ema:
        model.load_state_dict(checkpoint["state_dict_ema"])
        logger.info("Using Model EMA...")
    else:
        model.load_state_dict(checkpoint["state_dict"])

    # AMP: automatic mixed precision
    use_amp = getattr(cfg.solver, "amp", False)
    if use_amp:
        logger.info("Using Automatic Mixed Precision...")

    # test the detector
    logger.info("Testing Starts...\n")
    """Inference and Evaluation the model"""

    cfg.inference["folder"] = os.path.join(cfg.work_dir, "outputs")
    if cfg.inference.save_raw_prediction:
        create_folder(cfg.inference["folder"])

    # external classifier
    if "external_cls" in cfg.post_processing:
        if cfg.post_processing.external_cls != None:
            external_cls = build_classifier(cfg.post_processing.external_cls)
    else:
        external_cls = test_loader.dataset.class_map

    # whether the testing dataset is sliding window
    cfg.post_processing.sliding_window = isinstance(test_loader.dataset, SlidingWindowDataset)

    model.eval()

    # mAP model forward
    result_dict = {}
    post_processing_cfg = cfg.post_processing.copy()
    post_processing_cfg.pre_nms_topk = args.pre_nms_topk
    post_processing_cfg.nms.max_seg_num = args.max_seg_num
    for data_dict in tqdm.tqdm(test_loader, disable=(args.rank != 0)):
        with torch.cuda.amp.autocast(dtype=torch.float16, enabled=use_amp):
            with torch.no_grad():
                results = model(
                    **data_dict,
                    return_loss=False,
                    infer_cfg=cfg.inference,
                    post_cfg=post_processing_cfg,
                    ext_cls=external_cls,
                )

        # update the result dict
        for k, v in results.items():
            if k in result_dict.keys():
                result_dict[k].extend(v)
            else:
                result_dict[k] = v

    result_dict = gather_ddp_results(args.world_size, result_dict, post_processing_cfg)

    # convert the output to epic format
    result_dict_converted = {}
    for k, v in result_dict.items():
        tmp_result = []
        for result in v:
            tmp_result.append(
                dict(
                    interaction=int(result["label"].replace("id_", "")),
                    score=result["score"],
                    segment=result["segment"],
                )
            )
        result_dict_converted[k] = tmp_result

    if args.rank == 0:
        result_dict_submit = dict(
            version="0.1",
            challenge="audio_based_interaction_detection",
            sls_pt=2,
            sls_tl=3,
            sls_td=4,
            t_mod=0,
            results=result_dict_converted,
        )
        result_submit_path = os.path.join(cfg.work_dir, "test.json")
        with open(result_submit_path, "w") as out:
            json.dump(result_dict_submit, out)
        # os.system(f"cd {cfg_verb.work_dir} && zip -j my-submission.zip test.json")
        logger.info(f"Testing finished, results saved to {cfg.work_dir}")
        return


if __name__ == "__main__":
    main()
