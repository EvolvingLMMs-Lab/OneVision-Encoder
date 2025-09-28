
_base_ = [
    "../_base_/datasets/charades/feature_RoPE2d_ViT_L_14_1024videomae_v1_normal.py",
    "../_base_/models/actionformer.py",  # model config
]

model = dict(
    projection=dict(
        in_channels=768,
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

work_dir = "exps/charades/actionformer_RoPE2d_ViT_L_14_1024videomae_v1_normal"
