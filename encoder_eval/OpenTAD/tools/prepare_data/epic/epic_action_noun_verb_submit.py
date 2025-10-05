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
from mmengine.config import Config, DictAction
from opentad.cores.test_engine import gather_ddp_results
from opentad.models import build_detector
from opentad.models.utils.post_processing import build_classifier, batched_nms
from opentad.datasets import build_dataset, build_dataloader
from opentad.evaluations import build_evaluator
from opentad.utils import update_workdir, set_seed, create_folder, setup_logger
from opentad.datasets.base import SlidingWindowDataset
from opentad.models.utils.post_processing import convert_to_seconds, batched_nms


def parse_args():
    parser = argparse.ArgumentParser(description="Test a Temporal Action Detector")
    parser.add_argument("config_noun", metavar="FILE", type=str, help="path to noun config file")
    parser.add_argument("config_verb", metavar="FILE", type=str, help="path to verb config file")
    parser.add_argument("ckpt_noun", type=str, default="none", help="the noun checkpoint path")
    parser.add_argument("ckpt_verb", type=str, default="none", help="the verb checkpoint path")
    parser.add_argument("--submit", action="store_true", help="generate the challenge result")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--id", type=int, default=0, help="repeat experiment id")
    parser.add_argument("--pre_nms_topk", type=int, default=50000, help="max predictions before nms")
    parser.add_argument("--max_seg_num", type=int, default=30000, help="max predictions per video")
    parser.add_argument("--not_eval", action="store_true", help="whether to not to eval, only do inference")
    parser.add_argument("--cfg-options", nargs="+", action=DictAction, help="override settings")
    args = parser.parse_args()
    return args


def post_processing(noun_pred, verb_pred, metas, post_cfg):
    noun_proposals, noun_scores = noun_pred
    verb_proposals, verb_scores = verb_pred

    pre_nms_thresh = getattr(post_cfg, "pre_nms_thresh", 0.001)
    pre_nms_topk = getattr(post_cfg, "pre_nms_topk", 2000)
    num_classes_noun, num_classes_verb = noun_scores[0].shape[-1], verb_scores[0].shape[-1]
    assert num_classes_noun == 293
    assert num_classes_verb == 97
    max_noun_per_feature = 10
    max_verb_per_feature = 10

    results = {}
    for i in range(len(metas)):
        # segment = (noun_seg + verb_seg) / 2
        noun_segment, verb_segment = noun_proposals[i].detach().cpu(), verb_proposals[i].detach().cpu()  # [N,2]
        fused_segment = (noun_segment + verb_segment) / 2

        noun_score = noun_scores[i].detach().cpu().float()  # [N,293]
        noun_score, noun_idx = torch.topk(noun_score, max_noun_per_feature, 1)  # [N,10]
        noun_score = noun_score.unsqueeze(1)  # [N,1,10]

        verb_score = verb_scores[i].detach().cpu().float()  # [N,97]
        verb_score, verb_idx = torch.topk(verb_score, max_verb_per_feature, 1)  # [N,10]
        verb_score = verb_score.unsqueeze(2)  # [N,10,1]

        pred_prob = (noun_score * verb_score).sqrt().flatten()  # [N*10*10]
        verb_idx = verb_idx.unsqueeze(2).repeat(1, 1, max_noun_per_feature).flatten(1, 2)  # [N,10*10]
        noun_idx = noun_idx.unsqueeze(1).repeat(1, max_verb_per_feature, 1).flatten(1, 2)  # [N,10*10]
        pred_idx = verb_idx * num_classes_noun + noun_idx  # [N,10*10]
        pred_idx = (
            pred_idx
            + torch.arange(0, pred_idx.size(0), device=pred_idx.device)[:, None] * num_classes_noun * num_classes_verb
        )
        pred_idx = pred_idx.flatten()

        # 1. Keep seg with confidence score > a threshold
        keep_idxs1 = pred_prob > pre_nms_thresh
        pred_prob = pred_prob[keep_idxs1]
        topk_idxs = keep_idxs1.nonzero(as_tuple=True)[0]
        topk_idxs = pred_idx[topk_idxs]

        """
        # scores: verb first, noun second
        noun_score = noun_scores[i].detach().cpu().unsqueeze(1)  # [N,1,293]
        verb_score = verb_scores[i].detach().cpu().unsqueeze(2)  # [N,97,1]
        pred_prob = (noun_score * verb_score).sqrt().flatten()  # [N*97*293]

        # 1. Keep seg with confidence score > a threshold
        keep_idxs1 = pred_prob > pre_nms_thresh
        pred_prob = pred_prob[keep_idxs1]
        topk_idxs = keep_idxs1.nonzero(as_tuple=True)[0]
        """

        # 2. Keep top k top scoring boxes only
        num_topk = min(pre_nms_topk, topk_idxs.size(0))
        pred_prob, idxs = pred_prob.sort(descending=True)
        pred_prob = pred_prob[:num_topk].clone()
        topk_idxs = topk_idxs[idxs[:num_topk]].clone()

        # 3. gather predicted proposals
        pt_idxs = torch.div(topk_idxs, num_classes_noun * num_classes_verb, rounding_mode="floor")
        cls_idxs = torch.fmod(topk_idxs, num_classes_noun * num_classes_verb)
        cls_idxs_verb = torch.div(cls_idxs, num_classes_noun, rounding_mode="floor")
        cls_idxs_noun = torch.fmod(cls_idxs, num_classes_noun)

        segments = fused_segment[pt_idxs]
        scores = pred_prob
        noun_labels = cls_idxs_noun
        verb_labels = cls_idxs_verb
        labels = verb_labels * num_classes_noun + noun_labels  # convert to action label

        # if not sliding window, do nms # todo
        if post_cfg.sliding_window == False and post_cfg.nms is not None:
            segments, scores, labels = batched_nms(segments, scores, labels, **post_cfg.nms)

        video_id = metas[i]["video_name"]

        # convert segments to seconds
        segments = convert_to_seconds(segments, metas[i])

        results_per_video = []
        for segment, label, score in zip(segments, labels, scores):
            # convert to python scalars
            results_per_video.append(
                dict(
                    segment=[round(seg.item(), 2) for seg in segment],
                    label=int(label.item()),
                    score=round(score.item(), 4),
                )
            )

        if video_id in results.keys():
            results[video_id].extend(results_per_video)
        else:
            results[video_id] = results_per_video

    return results


def gather_ddp_results(world_size, result_dict, post_cfg):
    gather_dict_list = [None for _ in range(world_size)]
    dist.all_gather_object(gather_dict_list, result_dict)
    result_dict = {}
    for i in range(world_size):  # update the result dict
        for k, v in gather_dict_list[i].items():
            if k in result_dict.keys():
                result_dict[k].extend(v)
            else:
                result_dict[k] = v

    # do nms for sliding window, if needed
    if post_cfg.sliding_window == True and post_cfg.nms is not None:
        # assert sliding_window=True
        tmp_result_dict = {}
        for k, v in result_dict.items():
            segments = torch.Tensor([data["segment"] for data in v])
            scores = torch.Tensor([data["score"] for data in v])
            labels = torch.Tensor([int(data["label"]) for data in v])

            segments, scores, labels = batched_nms(segments, scores, labels, **post_cfg.nms)

            results_per_video = []
            for segment, label, score in zip(segments, labels, scores):
                # convert to python scalars
                results_per_video.append(
                    dict(
                        segment=[round(seg.item(), 2) for seg in segment],
                        label=int(label.item()),
                        score=round(score.item(), 4),
                    )
                )
            tmp_result_dict[k] = results_per_video
        result_dict = tmp_result_dict
    return result_dict


def eval_one_epoch(
    test_loader_noun,
    test_loader_verb,
    model_noun,
    model_verb,
    cfg_noun,
    cfg_verb,
    logger,
    rank,
    submit=False,
    use_amp_noun=None,
    use_amp_verb=None,
    pre_nms_topk=50000,
    max_seg_num=30000,
    world_size=0,
    not_eval=False,
):
    """Inference and Evaluation the model"""

    # noun external classifier
    if "external_cls" in cfg_noun.post_processing:
        if cfg_noun.post_processing.external_cls != None:
            ext_cls_noun = build_classifier(cfg_noun.post_processing.external_cls)
    else:
        ext_cls_noun = test_loader_noun.dataset.class_map

    # verb external classifier
    if "external_cls" in cfg_verb.post_processing:
        if cfg_verb.post_processing.external_cls != None:
            ext_cls_verb = build_classifier(cfg_verb.post_processing.external_cls)
    else:
        ext_cls_verb = test_loader_verb.dataset.class_map

    # whether the testing dataset is sliding window
    cfg_verb.post_processing.sliding_window = isinstance(test_loader_verb.dataset, SlidingWindowDataset)

    # override the pre_nms_topk and max_seg_num
    cfg_verb.post_processing.pre_nms_topk = pre_nms_topk
    cfg_verb.post_processing.nms.max_seg_num = max_seg_num

    # model forward
    model_noun.eval()
    model_verb.eval()
    result_dict = {}
    for data_dict_noun, data_dict_verb in tqdm.tqdm(zip(test_loader_noun, test_loader_verb), disable=(rank != 0)):
        # inference noun
        with torch.cuda.amp.autocast(dtype=torch.float16, enabled=use_amp_noun):
            with torch.no_grad():
                results_noun = model_noun(**data_dict_noun, infer_cfg=cfg_noun.inference)

        # inference verb
        with torch.cuda.amp.autocast(dtype=torch.float16, enabled=use_amp_verb):
            with torch.no_grad():
                results_verb = model_verb(**data_dict_verb, infer_cfg=cfg_verb.inference)

        # post processing
        results = post_processing(
            results_noun,
            results_verb,
            data_dict_verb["metas"],
            cfg_verb.post_processing,
        )

        # update the result dict
        for k, v in results.items():
            if k in result_dict.keys():
                result_dict[k].extend(v)
            else:
                result_dict[k] = v

    result_dict = gather_ddp_results(world_size, result_dict, cfg_verb.post_processing)

    # convert the action label to noun and verb
    num_classes_noun = len(ext_cls_noun)
    result_dict_converted = {}
    for k, v in result_dict.items():
        tmp_result = []
        for result in v:
            action_label = result["label"]
            verb_label = ext_cls_verb[action_label // num_classes_noun].replace("id_", "")
            noun_label = ext_cls_noun[action_label % num_classes_noun].replace("id_", "")
            tmp_result.append(
                dict(
                    segment=result["segment"],
                    score=result["score"],
                    noun=int(noun_label),
                    verb=int(verb_label),
                    action=str(int(verb_label)) + "," + str(int(noun_label)),
                )
            )
        result_dict_converted[k] = tmp_result

    if rank == 0:
        result_eval = dict(results=result_dict_converted)
        if cfg_verb.post_processing.save_dict:
            result_path = os.path.join(cfg_verb.work_dir, "result_epic_detection.json")
            with open(result_path, "w") as out:
                json.dump(result_eval, out)

        if submit:  # submit the result to the server
            result_dict_submit = dict(
                version="0.2",
                challenge="action_detection",
                sls_pt=2,
                sls_tl=3,
                sls_td=4,
                results=result_dict_converted,
            )
            result_submit_path = os.path.join(cfg_verb.work_dir, "test.json")
            with open(result_submit_path, "w") as out:
                json.dump(result_dict_submit, out)
            # os.system(f"cd {cfg_verb.work_dir} && zip -j my-submission.zip test.json")
            logger.info(f"Testing finished, results saved to {cfg_verb.work_dir}")
            return

        if not not_eval:
            # build evaluator
            for task in ["noun", "verb", "action"]:
                evaluator = build_evaluator(
                    dict(
                        type="mAP_EPIC",
                        subset="val",
                        tiou_thresholds=[0.1, 0.2, 0.3, 0.4, 0.5],
                        prediction_filename=result_eval,
                        ground_truth_filename="data/epic_kitchens-100/annotations/epic_kitchens_full.json",
                        task=task,
                    )
                )
                # evaluate and output
                logger.info(f"Task {task} evaluation starts...")
                metrics_dict = evaluator.evaluate()
                evaluator.logging(logger)


def main():
    args = parse_args()

    # load config
    cfg_noun = Config.fromfile(args.config_noun)
    cfg_verb = Config.fromfile(args.config_verb)
    if args.cfg_options is not None:
        cfg_verb.merge_from_dict(args.cfg_options)

    # override to test configuration
    if args.submit:
        cfg_noun.dataset.test.subset_name = "test"
        cfg_verb.dataset.test.subset_name = "test"

    # DDP init
    args.local_rank = int(os.environ["LOCAL_RANK"])
    args.world_size = int(os.environ["WORLD_SIZE"])
    args.rank = int(os.environ["RANK"])
    print(f"Distributed init (rank {args.rank}/{args.world_size}, local rank {args.local_rank})")
    dist.init_process_group("nccl", rank=args.rank, world_size=args.world_size)
    torch.cuda.set_device(args.local_rank)

    # set random seed, create work_dir
    set_seed(args.seed)
    cfg_verb.work_dir = cfg_verb.work_dir + "_action"  # create new work_dir for action
    cfg_verb.work_dir = cfg_verb.work_dir.replace("_verb_action", "_action")
    cfg_verb.work_dir = cfg_verb.work_dir + "_submit" if args.submit else cfg_verb.work_dir
    cfg_verb = update_workdir(cfg_verb, args.id, torch.cuda.device_count())
    if args.rank == 0:
        create_folder(cfg_verb.work_dir)

    # setup logger
    logger = setup_logger("Test", save_dir=cfg_verb.work_dir, distributed_rank=args.rank)
    logger.info(f"Using torch version: {torch.__version__}, CUDA version: {torch.version.cuda}")
    logger.info(f"Noun Config: \n{cfg_noun.pretty_text}")
    logger.info(f"Verb Config: \n{cfg_verb.pretty_text}")

    # build dataset
    logger.info("Building test noun dataset...")
    test_dataset_noun = build_dataset(cfg_noun.dataset.test, default_args=dict(logger=logger))
    test_loader_noun = build_dataloader(
        test_dataset_noun,
        rank=args.rank,
        world_size=args.world_size,
        shuffle=False,
        drop_last=False,
        **cfg_noun.solver.test,
    )
    logger.info("Building test verb dataset...")
    test_dataset_verb = build_dataset(cfg_verb.dataset.test, default_args=dict(logger=logger))
    test_loader_verb = build_dataloader(
        test_dataset_verb,
        rank=args.rank,
        world_size=args.world_size,
        shuffle=False,
        drop_last=False,
        **cfg_verb.solver.test,
    )

    # build model
    model_noun = build_detector(cfg_noun.model)
    model_verb = build_detector(cfg_verb.model)

    # override the forward function
    model_noun.forward = model_noun.forward_test
    model_verb.forward = model_verb.forward_test

    # DDP
    model_noun = model_noun.to(args.local_rank)
    model_verb = model_verb.to(args.local_rank)
    model_noun = DistributedDataParallel(model_noun, device_ids=[args.local_rank], output_device=args.local_rank)
    model_verb = DistributedDataParallel(model_verb, device_ids=[args.local_rank], output_device=args.local_rank)
    logger.info(f"Using DDP with {torch.cuda.device_count()} GPUS...")

    # load checkpoint
    device = f"cuda:{args.rank % torch.cuda.device_count()}"

    logger.info("Loading noun checkpoint from: {}".format(args.ckpt_noun))
    checkpoint_noun = torch.load(args.ckpt_noun, map_location=device)
    noun_use_ema = getattr(cfg_noun.solver, "ema", False)
    state_dict_noun = checkpoint_noun["state_dict_ema"] if noun_use_ema else checkpoint_noun["state_dict"]
    model_noun.load_state_dict(state_dict_noun)

    logger.info("Loading verb checkpoint from: {}".format(args.ckpt_verb))
    checkpoint_verb = torch.load(args.ckpt_verb, map_location=device)
    verb_use_ema = getattr(cfg_verb.solver, "ema", False)
    state_dict_verb = checkpoint_verb["state_dict_ema"] if verb_use_ema else checkpoint_verb["state_dict"]
    model_verb.load_state_dict(state_dict_verb)

    # AMP: automatic mixed precision
    use_amp_noun = getattr(cfg_noun.solver, "amp", False)
    use_amp_verb = getattr(cfg_verb.solver, "amp", False)
    if use_amp_noun:
        logger.info("Using Automatic Mixed Precision on Noun...")
    if use_amp_verb:
        logger.info("Using Automatic Mixed Precision on Verb...")

    logger.info(f"Working directory: {cfg_verb.work_dir}")
    eval_one_epoch(
        test_loader_noun,
        test_loader_verb,
        model_noun,
        model_verb,
        cfg_noun,
        cfg_verb,
        logger,
        args.rank,
        submit=args.submit,
        use_amp_noun=use_amp_noun,
        use_amp_verb=use_amp_verb,
        pre_nms_topk=args.pre_nms_topk,
        max_seg_num=args.max_seg_num,
        world_size=args.world_size,
        not_eval=args.not_eval,
    )


if __name__ == "__main__":
    main()
