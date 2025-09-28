_base_ = [
    "../_base_/datasets/charades/features_i3d_pad.py",  # dataset config
    "../_base_/models/temporalmaxer.py",  # model config
]

model = dict(
    projection=dict(in_channels=2048, drop_out=0.2),
    neck=dict(norm_cfg=dict(type="GN", num_groups=4, affine=False)),
)

solver = dict(
    train=dict(batch_size=2, num_workers=2),
    val=dict(batch_size=1, num_workers=1),
    test=dict(batch_size=1, num_workers=1),
    clip_grad_norm=1,
    ema=True,
)

optimizer = dict(type="AdamW", lr=1e-4, weight_decay=0.05, paramwise=True)
scheduler = dict(type="LinearWarmupCosineAnnealingLR", warmup_epoch=5, max_epoch=65)

inference = dict(load_from_raw_predictions=False, save_raw_prediction=False)
post_processing = dict(
    pre_nms_topk=2000,
    pre_nms_thresh=0.001,
    nms=dict(
        use_soft_nms=True,
        sigma=0.5,
        max_seg_num=200,
        iou_threshold=0,  # does not matter when use soft nms
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
    val_start_epoch=39,
    end_epoch=50,
)

work_dir = "exps/thumos/temporalmaxer_i3d"
