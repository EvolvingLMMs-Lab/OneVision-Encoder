import os
import sys

sys.dont_write_bytecode = True
path = os.path.join(os.path.dirname(__file__), "../../..")
if path not in sys.path:
    sys.path.insert(0, path)

import argparse
import torch
import json
import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from mmengine.config import Config, DictAction
from opentad.models import build_detector
from opentad.datasets import build_dataset, build_dataloader
from opentad.utils import update_workdir, set_seed, create_folder, setup_logger
from opentad.datasets.base import SlidingWindowDataset
from opentad.models.utils.post_processing import build_classifier
from opentad.cores.test_engine import gather_ddp_results


def parse_args():
    parser = argparse.ArgumentParser(description="Test a Temporal Action Detector")
    parser.add_argument("config", metavar="FILE", type=str, help="path to config file")
    parser.add_argument("--checkpoint", type=str, default="none", help="the checkpoint path")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--id", type=int, default=0, help="repeat experiment id")
    parser.add_argument("--map_sigma", type=float, default=2.0, help="nms sigma for mAP")
    parser.add_argument("--recall_sigma", type=float, default=0.7, help="nms sigma for recall")
    parser.add_argument("--cfg-options", nargs="+", action=DictAction, help="override settings")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

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
    model = DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)
    logger.info(f"Using DDP with total {args.world_size} GPUS...")

    if cfg.inference.load_from_raw_predictions:  # if load with saved predictions, no need to load checkpoint
        logger.info(f"Loading from raw predictions: {cfg.inference.fuse_list}")
    else:  # load checkpoint: args -> config -> best
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
    map_result_dict = {}
    mAP_post_processing_cfg = cfg.post_processing.copy()
    mAP_post_processing_cfg.nms.sigma = args.map_sigma
    for data_dict in tqdm.tqdm(test_loader, disable=(args.rank != 0)):
        with torch.cuda.amp.autocast(dtype=torch.float16, enabled=use_amp):
            with torch.no_grad():
                results = model(
                    **data_dict,
                    return_loss=False,
                    infer_cfg=cfg.inference,
                    post_cfg=mAP_post_processing_cfg,
                    ext_cls=external_cls,
                )

        # update the result dict
        for k, v in results.items():
            if k in map_result_dict.keys():
                map_result_dict[k].extend(v)
            else:
                map_result_dict[k] = v

    map_result_dict = gather_ddp_results(args.world_size, map_result_dict, mAP_post_processing_cfg)

    # recall model forward
    recall_result_dict = {}
    recall_post_processing_cfg = cfg.post_processing.copy()
    recall_post_processing_cfg.nms.sigma = args.recall_sigma
    for data_dict in tqdm.tqdm(test_loader, disable=(args.rank != 0)):
        with torch.cuda.amp.autocast(dtype=torch.float16, enabled=use_amp):
            with torch.no_grad():
                results = model(
                    **data_dict,
                    return_loss=False,
                    infer_cfg=cfg.inference,
                    post_cfg=recall_post_processing_cfg,
                    ext_cls=external_cls,
                )

        # update the result dict
        for k, v in results.items():
            if k in recall_result_dict.keys():
                recall_result_dict[k].extend(v)
            else:
                recall_result_dict[k] = v

    recall_result_dict = gather_ddp_results(args.world_size, recall_result_dict, recall_post_processing_cfg)

    if args.rank == 0:
        result_eval = dict(
            version="1.0",
            challenge="ego4d_moment_queries",
            detect_results=map_result_dict,
            retrieve_results=recall_result_dict,
        )
        result_path = os.path.join(cfg.work_dir, "result_detection.json")
        with open(result_path, "w") as out:
            json.dump(result_eval, out)
        logger.info(f"Testing finished, results saved to {result_path}")


if __name__ == "__main__":
    main()
