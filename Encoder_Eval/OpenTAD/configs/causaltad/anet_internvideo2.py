_base_ = [
    "../_base_/datasets/activitynet-1.3/features_tsp_resize_trunc.py",  # dataset config
    "../_base_/models/causaltad.py",  # model config
]

resize_length = 192
data_path = "data/activitynet-1.3/features/activitynet_6b/"
dataset = dict(
    train=dict(
        data_path=data_path,
        block_list=None,
        pipeline=[
            dict(type="LoadFeats", feat_format="pt", prefix="v_", suffix="_spatial_pool_feature_6"),
            dict(type="ConvertToTensor", keys=["feats", "gt_segments", "gt_labels"]),
            dict(type="ResizeFeat", tool="torchvision_align"),
            dict(type="RandomTrunc", trunc_len=resize_length, trunc_thresh=0.5, crop_ratio=[0.9, 1.0]),
            dict(type="Rearrange", keys=["feats"], ops="t c -> c t"),
            dict(type="Collect", inputs="feats", keys=["masks", "gt_segments", "gt_labels"]),
        ],
    ),
    val=dict(
        data_path=data_path,
        block_list=None,
        pipeline=[
            dict(type="LoadFeats", feat_format="pt", prefix="v_", suffix="_spatial_pool_feature_6"),
            dict(type="ConvertToTensor", keys=["feats", "gt_segments", "gt_labels"]),
            dict(type="ResizeFeat", tool="torchvision_align"),
            dict(type="Rearrange", keys=["feats"], ops="t c -> c t"),
            dict(type="Collect", inputs="feats", keys=["masks", "gt_segments", "gt_labels"]),
        ],
    ),
    test=dict(
        data_path=data_path,
        block_list=None,
        pipeline=[
            dict(type="LoadFeats", feat_format="pt", prefix="v_", suffix="_spatial_pool_feature_6"),
            dict(type="ConvertToTensor", keys=["feats"]),
            dict(type="ResizeFeat", tool="torchvision_align"),
            dict(type="Rearrange", keys=["feats"], ops="t c -> c t"),
            dict(type="Collect", inputs="feats", keys=["masks"]),
        ],
    ),
)


model = dict(
    projection=dict(
        in_channels=3200,
        out_channels=256,
        use_abs_pe=True,
        max_seq_len=192,
        input_pdrop=0.2,
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
    amp=True,
)

optimizer = dict(type="AdamW", lr=1e-3, weight_decay=0.05, paramwise=True)
scheduler = dict(type="LinearWarmupCosineAnnealingLR", warmup_epoch=5, max_epoch=15)

inference = dict(load_from_raw_predictions=False, save_raw_prediction=False)
post_processing = dict(
    nms=dict(
        use_soft_nms=True,
        sigma=0.75,
        max_seg_num=100,
        min_score=0.001,
        multiclass=False,
        voting_thresh=0.9,  #  set 0 to disable
    ),
    external_cls=dict(
        type="StandardClassifier",
        path="data/activitynet-1.3/classifiers/anet_UMTv2_6B_k710+K40_f16_frozenTuning.json_converted.json",
        topk=2,
    ),
    save_dict=False,
)

workflow = dict(
    logging_interval=200,
    checkpoint_interval=1,
    val_loss_interval=-1,
    val_eval_interval=1,
    val_start_epoch=7,
)

work_dir = "exps/anet/causal_internvideo2_6b"
