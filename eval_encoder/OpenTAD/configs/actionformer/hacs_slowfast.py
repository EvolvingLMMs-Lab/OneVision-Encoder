_base_ = [
    "../_base_/datasets/hacs-1.1.1/features_slowfast_pad.py",  # dataset config
    "../_base_/models/actionformer.py",  # model config
]

dataset = dict(
    train=dict(class_agnostic=True),
    val=dict(class_agnostic=True),
    test=dict(class_agnostic=True),
)

model = dict(
    projection=dict(
        in_channels=2304,
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

work_dir = "exps/hacs/actionformer_slowfast"
