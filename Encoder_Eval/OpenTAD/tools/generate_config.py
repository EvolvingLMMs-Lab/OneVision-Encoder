from pathlib import Path
import os, argparse

def safe_recreate_file(path: str | Path, content: str):
    """
    如果目标文件已存在——> 先删除，再重建并写入 content
    """
    p = Path(path)
    if p.exists():          # 检查文件是否存在
        p.unlink()          # 删除文件；若想捕获异常可用 try/except
    p.write_text(content, encoding="utf-8")   # 重建并写入

def args():
    arg = argparse.ArgumentParser()
    arg.add_argument("--dataset", default="charades")
    arg.add_argument("--feature_path", default="")
    arg.add_argument("--embedding_size", default="")
    arg.add_argument("--data_path", default="")
    arg.add_argument("--exp_save_path", default="")
    arg.add_argument("--model_config_path", default="")
    arg.add_argument("--data_config_path", default="")
    arg = arg.parse_args()
    return arg

def main():
    arg = args()
    data_config_path = arg.data_config_path.replace("..", 'configs')
    for p in (arg.model_config_path, data_config_path):
        Path(p).parent.mkdir(parents=True, exist_ok=True)
    print(arg.dataset)
    if arg.dataset == 'charades':
        model_config = '''
_base_ = [
    "{}",
    "../_base_/models/actionformer.py",  # model config
]

model = dict(
    projection=dict(
        in_channels={},
        arch=(2, 2, 7),
        attn_cfg=dict(n_mha_win_size=-1),
        use_abs_pe=True,
        max_seq_len=512,
        input_pdrop=0.3,
    ),
    neck=dict(num_levels=8),
    rpn_head=dict(
        num_classes=157,
        prior_generator=dict(
            type="PointGenerator",
            strides=[1, 2, 4, 8, 16, 32, 64, 128],
            regression_range=[(0, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 128), (128, 256), (256, 10000)],
        ),
    ),
)

solver = dict(
    train=dict(batch_size=16, num_workers=4),
    val=dict(batch_size=16, num_workers=4),
    test=dict(batch_size=16, num_workers=4),
    clip_grad_norm=1,
    ema=True,
)

optimizer = dict(type="AdamW", lr=1e-4, weight_decay=0.05, paramwise=True)
scheduler = dict(type="LinearWarmupCosineAnnealingLR", warmup_epoch=5, max_epoch=20)

inference = dict(load_from_raw_predictions=False, save_raw_prediction=False)
post_processing = dict(
    pre_nms_topk=2000,
    pre_nms_thresh=0.001,
    nms=dict(
        use_soft_nms=True,
        sigma=0.3,
        max_seg_num=1000,
        min_score=0.001,
        multiclass=True,
        voting_thresh=0.7,  #  set 0 to disable
    ),
    save_dict=False,
)

workflow = dict(
    logging_interval=200,
    checkpoint_interval=1,
    val_loss_interval=1,
    val_eval_interval=1,
    val_start_epoch=8,
    end_epoch=50,
)

work_dir = "{}"
'''.format(arg.data_config_path, arg.embedding_size, arg.exp_save_path)
        # 写入（覆盖）
        # Path(arg.model_config_path).write_text(model_config, encoding="utf-8")
        # print(f"[INFO] model_config 写入 → {os.path.abspath(arg.model_config_path)}")
        # with open(arg.model_config_path, 'w', encoding='utf-8') as f:
        #     f.write(model_config)

        data_config='''
annotation_path = "{}/charades/annotations/charades.json"
class_map = "{}/charades/annotations/category_idx.txt"
data_path = "{}/"
block_list = data_path + "missing_files.txt"

window_size = 512

dataset = dict(
    train=dict(
        type="EpicKitchensPaddingDataset",
        ann_file=annotation_path,
        subset_name="training",
        block_list=block_list,
        class_map=class_map,
        data_path=data_path,
        filter_gt=False,
        # dataloader setting
        fps=30,
        feature_stride=4,
        sample_stride=1,
        offset_frames=2,
        pipeline=[
            dict(type="LoadFeats", feat_format="npy"),
            dict(type="ConvertToTensor", keys=["feats", "gt_segments", "gt_labels"]),
            dict(type="RandomTrunc", trunc_len=window_size, trunc_thresh=0.5, crop_ratio=[0.9, 1.0]),
            dict(type="Rearrange", keys=["feats"], ops="t c -> c t"),
            dict(type="Collect", inputs="feats", keys=["masks", "gt_segments", "gt_labels"]),
        ],
    ),
    val=dict(
        type="EpicKitchensSlidingDataset",
        ann_file=annotation_path,
        subset_name="testing",
        block_list=block_list,
        class_map=class_map,
        data_path=data_path,
        filter_gt=False,
        # dataloader setting
        window_size=window_size,
        fps=30,
        feature_stride=4,
        sample_stride=1,
        offset_frames=2,
        window_overlap_ratio=0.25,
        pipeline=[
            dict(type="LoadFeats", feat_format="npy"),
            dict(type="ConvertToTensor", keys=["feats", "gt_segments", "gt_labels"]),
            dict(type="SlidingWindowTrunc", with_mask=True),
            dict(type="Rearrange", keys=["feats"], ops="t c -> c t"),
            dict(type="Collect", inputs="feats", keys=["masks", "gt_segments", "gt_labels"]),
        ],
    ),
    test=dict(
        type="EpicKitchensSlidingDataset",
        ann_file=annotation_path,
        subset_name="testing",
        block_list=block_list,
        class_map=class_map,
        data_path=data_path,
        filter_gt=False,
        test_mode=True,
        # dataloader setting
        window_size=window_size,
        fps=30,
        feature_stride=4,
        sample_stride=1,
        offset_frames=2,
        window_overlap_ratio=0.5,
        pipeline=[
            dict(type="LoadFeats", feat_format="npy"),
            dict(type="ConvertToTensor", keys=["feats"]),
            dict(type="SlidingWindowTrunc", with_mask=True),
            dict(type="Rearrange", keys=["feats"], ops="t c -> c t"),
            dict(type="Collect", inputs="feats", keys=["masks"]),
        ],
    ),
)


evaluation = dict(
    type="mAP",
    subset="testing",
    tiou_thresholds=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    ground_truth_filename=annotation_path,
)
'''.format(arg.data_path, arg.data_path, arg.feature_path)
        
        # Path(arg.data_config_path).write_text(data_config, encoding="utf-8")
        # print(f"[INFO] data_config 写入 → {os.path.abspath(arg.data_config_path)}")
        # with open(arg.data_config_path, 'w', encoding='utf-8') as f:
        #     f.write(data_config)

        safe_recreate_file(arg.model_config_path, model_config)
        print(f"[INFO] model_config → {os.path.abspath(arg.model_config_path)}")
        safe_recreate_file(data_config_path,  data_config)
        print(f"[INFO] data_config  → {os.path.abspath(data_config_path)}")

    elif arg.dataset == 'thumos':
        model_config = '''
_base_ = [
    "{}",  # dataset config
    "../_base_/models/actionformer.py",  # model config
]

model = dict(projection=dict(in_channels={}, input_pdrop=0.2))

solver = dict(
    train=dict(batch_size=16, num_workers=2),
    val=dict(batch_size=16, num_workers=1),
    test=dict(batch_size=16, num_workers=1),
    clip_grad_norm=1,
    ema=True,
)

optimizer = dict(type="AdamW", lr=1e-4, weight_decay=0.05, paramwise=True)
scheduler = dict(type="LinearWarmupCosineAnnealingLR", warmup_epoch=5, max_epoch=35)

inference = dict(load_from_raw_predictions=False, save_raw_prediction=False)
post_processing = dict(
    nms=dict(
        use_soft_nms=True,
        sigma=0.5,
        max_seg_num=2000,
        iou_threshold=0.1,  # does not matter when use soft nms
        min_score=0.001,
        multiclass=True,
        voting_thresh=0.7,  #  set 0 to disable
    ),
    save_dict=False,
)

workflow = dict(
    logging_interval=20,
    checkpoint_interval=1,
    val_loss_interval=1,
    val_eval_interval=1,
    val_start_epoch=30,
)

work_dir = "{}"
'''.format(arg.data_config_path, arg.embedding_size, arg.exp_save_path)
        # 写入（覆盖）
        # with open(arg.model_config_path, 'w', encoding='utf-8') as f:
        #     f.write(model_config)
        data_config='''
annotation_path = "{}/thumos-14/annotations/thumos_14_anno.json"
class_map = "{}/thumos-14/annotations/category_idx.txt"
data_path = "{}/"
block_list = data_path + "missing_files.txt"

window_size = 2304

dataset = dict(
    train=dict(
        type="ThumosPaddingDataset",
        ann_file=annotation_path,
        subset_name="training",
        block_list=block_list,
        class_map=class_map,
        data_path=data_path,
        filter_gt=False,
        # thumos dataloader setting
        feature_stride=4,
        sample_stride=1,  # 1x4=4
        offset_frames=8,
        pipeline=[
            dict(type="LoadFeats", feat_format="npy"),
            dict(type="ConvertToTensor", keys=["feats", "gt_segments", "gt_labels"]),
            dict(type="RandomTrunc", trunc_len=window_size, trunc_thresh=0.75, crop_ratio=[0.9, 1.0]),
            dict(type="Rearrange", keys=["feats"], ops="t c -> c t"),
            dict(type="Collect", inputs="feats", keys=["masks", "gt_segments", "gt_labels"]),
        ],
    ),
    val=dict(
        type="ThumosSlidingDataset",
        ann_file=annotation_path,
        subset_name="validation",
        block_list=block_list,
        class_map=class_map,
        data_path=data_path,
        filter_gt=False,
        # thumos dataloader setting
        feature_stride=4,
        sample_stride=1,  # 1x4=4
        window_size=window_size,
        offset_frames=8,
        window_overlap_ratio=0.25,
        pipeline=[
            dict(type="LoadFeats", feat_format="npy"),
            dict(type="ConvertToTensor", keys=["feats", "gt_segments", "gt_labels"]),
            dict(type="SlidingWindowTrunc", with_mask=True),
            dict(type="Rearrange", keys=["feats"], ops="t c -> c t"),
            dict(type="Collect", inputs="feats", keys=["masks", "gt_segments", "gt_labels"]),
        ],
    ),
    test=dict(
        type="ThumosSlidingDataset",
        ann_file=annotation_path,
        subset_name="validation",
        block_list=block_list,
        class_map=class_map,
        data_path=data_path,
        filter_gt=False,
        test_mode=True,
        # thumos dataloader setting
        feature_stride=4,
        sample_stride=1,  # 1x4=4
        window_size=window_size,
        offset_frames=8,
        window_overlap_ratio=0.5,
        pipeline=[
            dict(type="LoadFeats", feat_format="npy"),
            dict(type="ConvertToTensor", keys=["feats"]),
            dict(type="SlidingWindowTrunc", with_mask=True),
            dict(type="Rearrange", keys=["feats"], ops="t c -> c t"),
            dict(type="Collect", inputs="feats", keys=["masks"]),
        ],
    ),
)


evaluation = dict(
    type="mAP",
    subset="validation",
    tiou_thresholds=[0.3, 0.4, 0.5, 0.6, 0.7],
    ground_truth_filename=annotation_path,
)
'''.format(arg.data_path, arg.data_path, arg.feature_path)
        # with open(arg.data_config_path, 'w', encoding='utf-8') as f:
        #     f.write(data_config)

        safe_recreate_file(arg.model_config_path, model_config)
        print(f"[INFO] model_config → {os.path.abspath(arg.model_config_path)}")
        safe_recreate_file(data_config_path,  data_config)
        print(f"[INFO] data_config  → {os.path.abspath(data_config_path)}")

    elif arg.dataset =='fineaction':
        model_config='''
_base_ = [
    "{}",  # dataset config
    "../_base_/models/actionformer.py",  # model config
]

model = dict(
    projection=dict(
        in_channels={},
        out_channels=256,
        attn_cfg=dict(n_mha_win_size=[7, 7, 7, 7, 7, -1]),
        use_abs_pe=True,
        max_seq_len=192,
    ),
    neck=dict(in_channels=256, out_channels=256),
    rpn_head=dict(
        in_channels=256,
        feat_channels=256,
        num_classes=1,
        label_smoothing=0.1,
        loss_weight=2.0,
        loss_normalizer=200,
    ),
)

solver = dict(
    train=dict(batch_size=16, num_workers=4),
    val=dict(batch_size=16, num_workers=4),
    test=dict(batch_size=16, num_workers=4),
    clip_grad_norm=1,
    ema=True,
)

optimizer = dict(type="AdamW", lr=1e-3, weight_decay=0.05, paramwise=True)
scheduler = dict(type="LinearWarmupCosineAnnealingLR", warmup_epoch=5, max_epoch=20)

inference = dict(load_from_raw_predictions=False, save_raw_prediction=False)
post_processing = dict(
    nms=dict(
        use_soft_nms=True,
        sigma=0.75,
        max_seg_num=100,
        iou_threshold=0,  # does not matter when use soft nms
        min_score=0.001,
        multiclass=False,
        voting_thresh=0.9,  #  set 0 to disable
    ),
    external_cls=dict(
        type="StandardClassifier",
        path="./data/fineaction/classifiers/new_swinB_1x1x256_views2x3_max_label_avg_prob.json",
        topk=2,
    ),
    save_dict=False,
)

workflow = dict(
    logging_interval=200,
    checkpoint_interval=1,
    val_loss_interval=1,
    val_eval_interval=1,
    val_start_epoch=10,
)

work_dir = "{}"


'''.format(arg.data_config_path, arg.embedding_size, arg.exp_save_path)
        # with open(arg.model_config_path, 'w', encoding='utf-8') as f:
        #     f.write(model_config)
        data_config='''
dataset_type = "AnetResizeDataset"
annotation_path = "{}/fineaction/annotations/annotations_gt.json"
class_map = "{}/fineaction/annotations/category_idx.txt"
data_path = "{}/"
block_list = data_path + "missing_files.txt"
resize_length = 192

dataset = dict(
    train=dict(
        type=dataset_type,
        ann_file=annotation_path,
        subset_name="training",
        block_list=block_list,
        class_map=class_map,
        data_path=data_path,
        filter_gt=True,
        resize_length=resize_length,
        class_agnostic=True,
        pipeline=[
            dict(type="LoadFeats", feat_format="npy"),
            dict(type="ConvertToTensor", keys=["feats", "gt_segments", "gt_labels"]),
            dict(type="ResizeFeat", tool="torchvision_align"),
            dict(type="RandomTrunc", trunc_len=resize_length, trunc_thresh=0.5, crop_ratio=[0.9, 1.0]),
            dict(type="Rearrange", keys=["feats"], ops="t c -> c t"),
            dict(type="Collect", inputs="feats", keys=["masks", "gt_segments", "gt_labels"]),
        ],
    ),
    val=dict(
        type=dataset_type,
        ann_file=annotation_path,
        subset_name="validation",
        block_list=block_list,
        class_map=class_map,
        data_path=data_path,
        filter_gt=True,
        resize_length=resize_length,
        class_agnostic=True,
        pipeline=[
            dict(type="LoadFeats", feat_format="npy"),
            dict(type="ConvertToTensor", keys=["feats", "gt_segments", "gt_labels"]),
            dict(type="ResizeFeat", tool="torchvision_align"),
            dict(type="Rearrange", keys=["feats"], ops="t c -> c t"),
            dict(type="Collect", inputs="feats", keys=["masks", "gt_segments", "gt_labels"]),
        ],
    ),
    test=dict(
        type=dataset_type,
        ann_file=annotation_path,
        subset_name="validation",
        block_list=block_list,
        class_map=class_map,
        data_path=data_path,
        filter_gt=False,
        test_mode=True,
        resize_length=resize_length,
        class_agnostic=True,
        pipeline=[
            dict(type="LoadFeats", feat_format="npy"),
            dict(type="ConvertToTensor", keys=["feats"]),
            dict(type="ResizeFeat", tool="torchvision_align"),
            dict(type="Rearrange", keys=["feats"], ops="t c -> c t"),
            dict(type="Collect", inputs="feats", keys=["masks"]),
        ],
    ),
)

evaluation = dict(
    type="mAP",
    subset="validation",
    tiou_thresholds=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
    ground_truth_filename=annotation_path,
)
'''.format(arg.data_path, arg.data_path, arg.feature_path)
        # with open(arg.data_config_path, 'w', encoding='utf-8') as f:
        #     f.write(data_config)
        safe_recreate_file(arg.model_config_path, model_config)
        print(f"[INFO] model_config → {os.path.abspath(arg.model_config_path)}")
        safe_recreate_file(data_config_path,  data_config)
        print(f"[INFO] data_config  → {os.path.abspath(data_config_path)}")

    elif arg.dataset == 'epic_kitchen':
        model_config='''
_base_ = [
    "{}",  # dataset config
    "../_base_/models/actionformer.py",  # model config
]

model = dict(
    projection=dict(
        in_channels={},
        attn_cfg=dict(n_mha_win_size=9),
    ),
    rpn_head=dict(
        num_classes=293,  # total 300, but 7 classes are empty
        prior_generator=dict(
            strides=[1, 2, 4, 8, 16, 32],
            regression_range=[(0, 4), (2, 8), (4, 16), (8, 32), (16, 64), (32, 10000)],
        ),
        label_smoothing=0.1,
        loss_normalizer=250,
    ),
)

solver = dict(
    train=dict(batch_size=2, num_workers=2),
    val=dict(batch_size=1, num_workers=1),
    test=dict(batch_size=1, num_workers=1),
    clip_grad_norm=1,
    ema=True,
)

optimizer = dict(type="AdamW", lr=1e-4, weight_decay=0.05, paramwise=True)
scheduler = dict(type="LinearWarmupCosineAnnealingLR", warmup_epoch=5, max_epoch=20)

inference = dict(load_from_raw_predictions=False, save_raw_prediction=False)
post_processing = dict(
    pre_nms_topk=5000,
    nms=dict(
        use_soft_nms=True,
        sigma=0.4,
        max_seg_num=2000,
        iou_threshold=0,  # does not matter when use soft nms
        min_score=0.001,
        multiclass=True,
        voting_thresh=0.75,  #  set 0 to disable
    ),
    save_dict=False,
)

workflow = dict(
    logging_interval=200,
    checkpoint_interval=1,
    val_loss_interval=1,
    val_eval_interval=1,
    val_start_epoch=15,
)

work_dir = "{}"

'''.format(arg.data_config_path, arg.embedding_size, arg.exp_save_path)
        # with open(arg.model_config_path, 'w', encoding='utf-8') as f:
        #     f.write(model_config)
        data_config='''
annotation_path = "{}/epic_kitchens-100/annotations/epic_kitchens_verb.json"
class_map = "{}/epic_kitchens-100/annotations/category_idx_verb.txt"
data_path = (
    "{}"
)
block_list = data_path + "missing_files.txt"

window_size = 4608

dataset = dict(
    train=dict(
        type="EpicKitchensPaddingDataset",
        ann_file=annotation_path,
        subset_name="train",
        block_list=block_list,
        class_map=class_map,
        data_path=data_path,
        filter_gt=True,
        # epic-kitchens dataloader setting
        fps=30,
        feature_stride=8,
        sample_stride=1,
        offset_frames=4,
        pipeline=[
            dict(type="LoadFeats", feat_format="npy"),
            dict(type="ConvertToTensor", keys=["feats", "gt_segments", "gt_labels"]),
            dict(type="RandomTrunc", trunc_len=window_size, trunc_thresh=0.3, crop_ratio=[0.9, 1.0]),
            dict(type="Rearrange", keys=["feats"], ops="t c -> c t"),
            dict(type="Collect", inputs="feats", keys=["masks", "gt_segments", "gt_labels"]),
        ],
    ),
    val=dict(
        type="EpicKitchensSlidingDataset",
        ann_file=annotation_path,
        subset_name="val",
        block_list=block_list,
        class_map=class_map,
        data_path=data_path,
        filter_gt=False,
        # dataloader setting
        window_size=window_size,
        fps=30,
        feature_stride=8,
        sample_stride=1,
        offset_frames=4,
        window_overlap_ratio=0.25,
        pipeline=[
            dict(type="LoadFeats", feat_format="npy"),
            dict(type="ConvertToTensor", keys=["feats", "gt_segments", "gt_labels"]),
            dict(type="SlidingWindowTrunc", with_mask=True),
            dict(type="Rearrange", keys=["feats"], ops="t c -> c t"),
            dict(type="Collect", inputs="feats", keys=["masks", "gt_segments", "gt_labels"]),
        ],
    ),
    test=dict(
        type="EpicKitchensSlidingDataset",
        ann_file=annotation_path,
        subset_name="val",
        block_list=block_list,
        class_map=class_map,
        data_path=data_path,
        filter_gt=False,
        test_mode=True,
        # epic-kitchens dataloader setting
        window_size=window_size,
        fps=30,
        feature_stride=8,
        sample_stride=1,
        offset_frames=4,
        window_overlap_ratio=0.5,
        pipeline=[
            dict(type="LoadFeats", feat_format="npy"),
            dict(type="ConvertToTensor", keys=["feats"]),
            dict(type="SlidingWindowTrunc", with_mask=True),
            dict(type="Rearrange", keys=["feats"], ops="t c -> c t"),
            dict(type="Collect", inputs="feats", keys=["masks"]),
        ],
    ),
)


evaluation = dict(
    type="mAP",
    subset="val",
    tiou_thresholds=[0.1, 0.2, 0.3, 0.4, 0.5],
    ground_truth_filename=annotation_path,
)

'''.format(arg.data_path, arg.data_path, arg.feature_path)
        # with open(arg.data_config_path, 'w', encoding='utf-8') as f:
        #     f.write(data_config)
        safe_recreate_file(arg.model_config_path, model_config)
        print(f"[INFO] model_config → {os.path.abspath(arg.model_config_path)}")
        safe_recreate_file(data_config_path,  data_config)
        print(f"[INFO] data_config  → {os.path.abspath(data_config_path)}")

    elif arg.dataset == 'hacs':
        model_config='''
_base_ = [
    "{}",  # dataset config
    "../_base_/models/actionformer.py",  # model config
]

dataset = dict(
    train=dict(class_agnostic=True),
    val=dict(class_agnostic=True),
    test=dict(class_agnostic=True),
)

model = dict(
    projection=dict(
        in_channels={},
        out_channels=256,
        attn_cfg=dict(n_mha_win_size=[13, 13, 13, 13, 13, -1]),
        max_seq_len=960,
        use_abs_pe=True,
        input_pdrop=0.1,
    ),
    neck=dict(in_channels=256, out_channels=256),
    rpn_head=dict(
        in_channels=256,
        feat_channels=256,
        num_classes=1,
        label_smoothing=0.1,
        loss_normalizer=400,
    ),
)
solver = dict(
    train=dict(batch_size=16, num_workers=4),
    val=dict(batch_size=16, num_workers=4),
    test=dict(batch_size=16, num_workers=4),
    clip_grad_norm=1.0,
    ema=True,
)

optimizer = dict(type="AdamW", lr=1e-3, weight_decay=0.03, paramwise=True)
scheduler = dict(type="LinearWarmupCosineAnnealingLR", warmup_epoch=5, max_epoch=15)

inference = dict(load_from_raw_predictions=False, save_raw_prediction=False)
post_processing = dict(
    nms=dict(
        use_soft_nms=True,
        sigma=0.75,
        max_seg_num=100,
        min_score=0.001,
        multiclass=True,
        voting_thresh=0.7,  #  set 0 to disable
    ),
    external_cls=dict(
        type="TCANetHACSClassifier",
        path="data/hacs-1.1.1/classifiers/validation94.32.json",
        topk=3,
    ),
    save_dict=False,
)

workflow = dict(
    logging_interval=200,
    checkpoint_interval=1,
    val_loss_interval=1,
    val_eval_interval=1,
    val_start_epoch=8,
)

work_dir = "{}"


'''.format(arg.data_config_path, arg.embedding_size, arg.exp_save_path)
        # with open(arg.model_config_path, 'w', encoding='utf-8') as f:
        #     f.write(model_config)
        data_config='''
dataset_type = "AnetPaddingDataset"
annotation_path = "{}/hacs-1.1.1/annotations/HACS_segments_v1.1.1.json"
class_map = "{}/hacs-1.1.1/annotations/category_idx.txt"
data_path = "{}/"
block_list = data_path + "missing_files.txt"

pad_len = 960

dataset = dict(
    train=dict(
        type=dataset_type,
        ann_file=annotation_path,
        subset_name="training",
        block_list=block_list,
        class_map=class_map,
        data_path=data_path,
        filter_gt=True,
        feature_stride=8,
        sample_stride=1,  # 1x8=8
        offset_frames=16,
        fps=15,
        pipeline=[
            dict(type="LoadFeats", feat_format="npy"),
            dict(type="ConvertToTensor", keys=["feats", "gt_segments", "gt_labels"]),
            dict(type="RandomTrunc", trunc_len=pad_len, trunc_thresh=0.5, crop_ratio=[0.9, 1.0]),
            dict(type="Rearrange", keys=["feats"], ops="t c-> c t"),
            dict(type="Collect", inputs="feats", keys=["masks", "gt_segments", "gt_labels"]),
        ],
    ),
    val=dict(
        type=dataset_type,
        ann_file=annotation_path,
        subset_name="validation",
        block_list=block_list,
        class_map=class_map,
        data_path=data_path,
        filter_gt=False,
        feature_stride=8,
        sample_stride=1,  # 1x8=8
        offset_frames=16,
        fps=15,
        pipeline=[
            dict(type="LoadFeats", feat_format="npy"),
            dict(type="ConvertToTensor", keys=["feats", "gt_segments", "gt_labels"]),
            dict(type="Padding", length=pad_len),
            dict(type="Rearrange", keys=["feats"], ops="t c-> c t"),
            dict(type="Collect", inputs="feats", keys=["masks", "gt_segments", "gt_labels"]),
        ],
    ),
    test=dict(
        type=dataset_type,
        ann_file=annotation_path,
        subset_name="validation",
        block_list=block_list,
        class_map=class_map,
        data_path=data_path,
        filter_gt=False,
        test_mode=True,
        feature_stride=8,
        sample_stride=1,  # 1x8=8
        offset_frames=16,
        fps=15,
        pipeline=[
            dict(type="LoadFeats", feat_format="npy"),
            dict(type="ConvertToTensor", keys=["feats"]),
            dict(type="Padding", length=pad_len),
            dict(type="Rearrange", keys=["feats"], ops="t c-> c t"),
            dict(type="Collect", inputs="feats", keys=["masks"]),
        ],
    ),
)


evaluation = dict(
    type="mAP",
    subset="validation",
    tiou_thresholds=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
    ground_truth_filename=annotation_path,
)

'''.format(arg.data_path, arg.data_path, arg.feature_path)
        # with open(arg.data_config_path, 'w', encoding='utf-8') as f:
        #     f.write(data_config)
        safe_recreate_file(arg.model_config_path, model_config)
        print(f"[INFO] model_config → {os.path.abspath(arg.model_config_path)}")
        safe_recreate_file(data_config_path,  data_config)
        print(f"[INFO] data_config  → {os.path.abspath(data_config_path)}")

    elif arg.dataset == 'ego4d':
        model_config='''
_base_ = [
    "{}",  # dataset config
    "../_base_/models/actionformer.py",  # model config
]

model = dict(
    projection=dict(
        in_channels={},
        arch=(2, 2, 9),
        use_abs_pe=True,
        max_seq_len=2048,
        conv_cfg=dict(proj_pdrop=0.1),
        attn_cfg=dict(n_mha_win_size=[17, 17, 17, 17, 17, 17, 17, -1, -1, -1]),
        input_pdrop=0.2,
    ),
    neck=dict(type="FPNIdentity", num_levels=10),
    rpn_head=dict(
        num_classes=110,
        prior_generator=dict(
            strides=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
            regression_range=[
                (0, 4),
                (2, 8),
                (4, 16),
                (8, 32),
                (16, 64),
                (32, 128),
                (64, 256),
                (128, 512),
                (256, 1024),
                (512, 10000),
            ],
        ),
        filter_similar_gt=False,
    ),
)

solver = dict(
    train=dict(batch_size=2, num_workers=2),
    val=dict(batch_size=2, num_workers=2),
    test=dict(batch_size=2, num_workers=2),
    clip_grad_norm=1,
    ema=True,
)

optimizer = dict(type="AdamW", lr=1e-4, weight_decay=0.05, paramwise=True)
scheduler = dict(type="LinearWarmupCosineAnnealingLR", warmup_epoch=5, max_epoch=25)

inference = dict(load_from_raw_predictions=False, save_raw_prediction=False)
post_processing = dict(
    pre_nms_topk=5000,
    nms=dict(
        use_soft_nms=True,
        sigma=2.0,
        max_seg_num=2000,
        min_score=0.001,
        multiclass=True,
        voting_thresh=0.95,  #  set 0 to disable
    ),
    save_dict=False,
)

workflow = dict(
    logging_interval=200,
    checkpoint_interval=1,
    val_loss_interval=-1,
    val_eval_interval=1,
    val_start_epoch=12,
    end_epoch=20,
)

work_dir = "{}"


'''.format(arg.data_config_path, arg.embedding_size, arg.exp_save_path)
        # with open(arg.model_config_path, 'w', encoding='utf-8') as f:
        #     f.write(model_config)
        data_config='''
annotation_path = "{}/ego4d/annotations/ego4d_v2_220429.json"
class_map = "{}/ego4d/annotations/category_idx.txt"
data_path = "{}/"

window_size = 2048
dataset = dict(
    train=dict(
        type="Ego4DPaddingDataset",
        ann_file=annotation_path,
        subset_name="train",
        block_list=None,
        class_map=class_map,
        data_path=data_path,
        filter_gt=False,
        # dataloader setting
        feature_stride=8,
        sample_stride=1,
        offset_frames=4,
        pipeline=[
            dict(type="LoadFeats", feat_format="npy"),
            dict(type="ConvertToTensor", keys=["feats", "gt_segments", "gt_labels"]),
            dict(type="RandomTrunc", trunc_len=window_size, trunc_thresh=0.3, crop_ratio=[0.9, 1.0]),
            dict(type="Rearrange", keys=["feats"], ops="t c -> c t"),
            dict(type="Collect", inputs="feats", keys=["masks", "gt_segments", "gt_labels"]),
        ],
    ),
    val=dict(
        type="Ego4DSlidingDataset",
        ann_file=annotation_path,
        subset_name="val",
        block_list=None,
        class_map=class_map,
        data_path=data_path,
        filter_gt=False,
        # dataloader setting
        window_size=window_size,
        feature_stride=8,
        sample_stride=1,
        offset_frames=4,
        window_overlap_ratio=0,
        pipeline=[
            dict(type="LoadFeats", feat_format="npy"),
            dict(type="ConvertToTensor", keys=["feats", "gt_segments", "gt_labels"]),
            dict(type="SlidingWindowTrunc", with_mask=True),
            dict(type="Rearrange", keys=["feats"], ops="t c -> c t"),
            dict(type="Collect", inputs="feats", keys=["masks", "gt_segments", "gt_labels"]),
        ],
    ),
    test=dict(
        type="Ego4DSlidingDataset",
        ann_file=annotation_path,
        subset_name="val",
        block_list=None,
        class_map=class_map,
        data_path=data_path,
        filter_gt=False,
        test_mode=True,
        # dataloader setting
        window_size=window_size,
        feature_stride=8,
        sample_stride=1,
        offset_frames=4,
        window_overlap_ratio=0,
        pipeline=[
            dict(type="LoadFeats", feat_format="npy"),
            dict(type="ConvertToTensor", keys=["feats"]),
            dict(type="SlidingWindowTrunc", with_mask=True),
            dict(type="Rearrange", keys=["feats"], ops="t c -> c t"),
            dict(type="Collect", inputs="feats", keys=["masks"]),
        ],
    ),
)


evaluation = dict(
    type="mAP",
    subset="val",
    tiou_thresholds=[0.1, 0.2, 0.3, 0.4, 0.5],
    ground_truth_filename=annotation_path,
    top_k=[1, 5],
)
'''.format(arg.data_path, arg.data_path, arg.feature_path)
        # with open(arg.data_config_path, 'w', encoding='utf-8') as f:
        #     f.write(data_config)
        safe_recreate_file(arg.model_config_path, model_config)
        print(f"[INFO] model_config → {os.path.abspath(arg.model_config_path)}")
        safe_recreate_file(data_config_path,  data_config)
        print(f"[INFO] data_config  → {os.path.abspath(data_config_path)}")

if __name__ == "__main__":
    main()
