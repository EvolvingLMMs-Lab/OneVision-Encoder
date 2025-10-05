_base_ = [
    "../_base_/datasets/ego4d_mq/features_internvideo_train_trunc_test_sw_2048_s8.py",  # dataset config
    "../_base_/models/actionformer.py",  # model config
]

model = dict(
    projection=dict(
        in_channels=1024,
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

work_dir = "exps/ego4d/actionformer_internvideo"
