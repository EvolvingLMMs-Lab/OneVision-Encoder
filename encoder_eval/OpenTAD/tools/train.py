import os
import sys

sys.dont_write_bytecode = True
path = os.path.join(os.path.dirname(__file__), "..")
if path not in sys.path:
    sys.path.insert(0, path)

import argparse
import torch
import torch.distributed as dist
from torch.distributed.algorithms.ddp_comm_hooks import default as comm_hooks
from torch.nn.parallel import DistributedDataParallel
from torch.cuda.amp import GradScaler
from mmengine.config import Config, DictAction
from opentad.models import build_detector
from opentad.datasets import build_dataset, build_dataloader
from opentad.cores import train_one_epoch, val_one_epoch, eval_one_epoch, build_optimizer, build_scheduler
from opentad.utils import (
    set_seed,
    update_workdir,
    create_folder,
    save_config,
    setup_logger,
    ModelEma,
    save_checkpoint,
    save_best_checkpoint,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Temporal Action Detector")
    parser.add_argument("config", metavar="FILE", type=str, help="path to config file")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--id", type=int, default=0, help="repeat experiment id")
    parser.add_argument("--resume", type=str, default=None, help="resume from a checkpoint")
    parser.add_argument("--not_eval", action="store_true", help="whether not to eval, only do inference")
    parser.add_argument("--disable_deterministic", action="store_true", help="disable deterministic for faster speed")
    parser.add_argument("--cfg-options", nargs="+", action=DictAction, help="override settings")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    # print(cfg)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # DDP init
    args.local_rank = int(os.environ["LOCAL_RANK"])
    args.world_size = int(os.environ["WORLD_SIZE"])
    args.rank = int(os.environ["RANK"])
    print(f"Distributed init (rank {args.rank}/{args.world_size}, local rank {args.local_rank})")
    dist.init_process_group("nccl", rank=args.rank, world_size=args.world_size)
    torch.cuda.set_device(args.local_rank)

    # set random seed, create work_dir, and save config
    set_seed(args.seed, args.disable_deterministic)
    cfg = update_workdir(cfg, args.id, args.world_size)
    if args.rank == 0:
        create_folder(cfg.work_dir)
        save_config(args.config, cfg.work_dir)

    # setup logger
    logger = setup_logger("Train", save_dir=cfg.work_dir, distributed_rank=args.rank)
    logger.info(f"Using torch version: {torch.__version__}, CUDA version: {torch.version.cuda}")
    logger.info(f"Config: \n{cfg.pretty_text}")

    # build dataset
    # print("train_dataset is:", cfg.dataset.train)
    train_dataset = build_dataset(cfg.dataset.train, default_args=dict(logger=logger))
    train_loader = build_dataloader(
        train_dataset,
        rank=args.rank,
        world_size=args.world_size,
        shuffle=True,
        drop_last=True,
        **cfg.solver.train,
    )
    # print("val_dataset is:", cfg.dataset.val)
    val_dataset = build_dataset(cfg.dataset.val, default_args=dict(logger=logger))
    val_loader = build_dataloader(
        val_dataset,
        rank=args.rank,
        world_size=args.world_size,
        shuffle=False,
        drop_last=False,
        **cfg.solver.val,
    )

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
    use_static_graph = getattr(cfg.solver, "static_graph", False)
    model = model.to(args.local_rank)
    model = DistributedDataParallel(
        model,
        device_ids=[args.local_rank],
        output_device=args.local_rank,
        find_unused_parameters=False if use_static_graph else True,
        static_graph=use_static_graph,  # default is False, should be true when use activation checkpointing in E2E
    )
    logger.info(f"Using DDP with total {args.world_size} GPUS...")

    # FP16 compression
    use_fp16_compress = getattr(cfg.solver, "fp16_compress", False)
    if use_fp16_compress:
        logger.info("Using FP16 compression ...")
        model.register_comm_hook(state=None, hook=comm_hooks.fp16_compress_hook)

    # Model EMA
    use_ema = getattr(cfg.solver, "ema", False)
    if use_ema:
        logger.info("Using Model EMA...")
        model_ema = ModelEma(model)
    else:
        model_ema = None

    # AMP: automatic mixed precision
    use_amp = getattr(cfg.solver, "amp", False)
    if use_amp:
        logger.info("Using Automatic Mixed Precision...")
        scaler = GradScaler()
    else:
        scaler = None

    # build optimizer and scheduler
    optimizer = build_optimizer(cfg.optimizer, model, logger)
    scheduler, max_epoch = build_scheduler(cfg.scheduler, optimizer, len(train_loader))

    # override the max_epoch
    max_epoch = cfg.workflow.get("end_epoch", max_epoch)

    # resume: reset epoch, load checkpoint / best rmse
    if args.resume != None:
        logger.info("Resume training from: {}".format(args.resume))
        device = f"cuda:{args.local_rank}"
        checkpoint = torch.load(args.resume, map_location=device)
        resume_epoch = checkpoint["epoch"]
        logger.info("Resume epoch is {}".format(resume_epoch))
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        if model_ema != None:
            model_ema.module.load_state_dict(checkpoint["state_dict_ema"])

        del checkpoint  #  save memory if the model is very large such as ViT-g
        torch.cuda.empty_cache()
    else:
        resume_epoch = -1

    # train the detector
    logger.info("Training Starts...\n")
    val_loss_best = 1e6
    val_start_epoch = cfg.workflow.get("val_start_epoch", 0)
    for epoch in range(resume_epoch + 1, max_epoch):
        train_loader.sampler.set_epoch(epoch)

        # train for one epoch
        train_one_epoch(
            train_loader,
            model,
            optimizer,
            scheduler,
            epoch,
            logger,
            model_ema=model_ema,
            clip_grad_l2norm=cfg.solver.clip_grad_norm,
            logging_interval=cfg.workflow.logging_interval,
            scaler=scaler,
        )

        # save checkpoint
        if (epoch == max_epoch - 1) or ((epoch + 1) % cfg.workflow.checkpoint_interval == 0):
            if args.rank == 0:
                save_checkpoint(model, model_ema, optimizer, scheduler, epoch, work_dir=cfg.work_dir)

        # val for one epoch
        if epoch >= val_start_epoch:
            if (cfg.workflow.val_loss_interval > 0) and ((epoch + 1) % cfg.workflow.val_loss_interval == 0):
                val_loss = val_one_epoch(
                    val_loader,
                    model,
                    logger,
                    args.rank,
                    epoch,
                    model_ema=model_ema,
                    use_amp=use_amp,
                )

                # save the best checkpoint
                if val_loss < val_loss_best:
                    logger.info(f"New best epoch {epoch}")
                    val_loss_best = val_loss
                    if args.rank == 0:
                        save_best_checkpoint(model, model_ema, epoch, work_dir=cfg.work_dir)

        # eval for one epoch
        if epoch >= val_start_epoch:
            if (cfg.workflow.val_eval_interval > 0) and ((epoch + 1) % cfg.workflow.val_eval_interval == 0):
                eval_one_epoch(
                    test_loader,
                    model,
                    cfg,
                    logger,
                    args.rank,
                    model_ema=model_ema,
                    use_amp=use_amp,
                    world_size=args.world_size,
                    not_eval=args.not_eval,
                )
    logger.info("Training Over...\n")


if __name__ == "__main__":
    main()
