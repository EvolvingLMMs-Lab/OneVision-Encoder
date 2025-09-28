_base_ = [
    "../_base_/datasets/hacs-1.1.1/features_slowfast_pad.py",  # dataset config
    "../_base_/models/tridet.py",  # model config
]

model = dict(
    projection=dict(
        in_channels=2304,
        out_channels=1024,
        sgp_win_size=[3, 3, 3, 3, 3, 3],
        sgp_mlp_dim=1024,
        use_abs_pe=True,
        max_seq_len=960,
        k=1.2,
        init_conv_vars=0.1,
        input_noise=0.2,
    ),
    neck=dict(in_channels=1024, out_channels=1024),
    rpn_head=dict(
        in_channels=1024,
        feat_channels=1024,
        kernel_size=5,
        boundary_kernel_size=1,
        num_classes=200,
        label_smoothing=0.1,
        loss_normalizer=400,
        iou_weight_power=1,
        num_bins=14,
    ),
)
solver = dict(
    train=dict(batch_size=16, num_workers=4),
    val=dict(batch_size=16, num_workers=4),
    test=dict(batch_size=16, num_workers=4),
    clip_grad_norm=0.5,
    ema=True,
)

optimizer = dict(type="AdamW", lr=1e-3, weight_decay=0.03, paramwise=True)
scheduler = dict(type="LinearWarmupCosineAnnealingLR", warmup_epoch=7, max_epoch=11, eta_min=5e-4)

inference = dict(load_from_raw_predictions=False, save_raw_prediction=False)
post_processing = dict(
    nms=dict(
        use_soft_nms=True,
        sigma=0.75,
        max_seg_num=250,
        iou_threshold=0,  # does not matter when use soft nms
        min_score=0.001,
        multiclass=True,
        voting_thresh=0.95,  #  set 0 to disable
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

work_dir = "exps/hacs/tridet_slowfast"
