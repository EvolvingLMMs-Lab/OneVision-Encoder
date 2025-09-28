_base_ = [
    "../_base_/datasets/ego4d_mq/features_internvideo_train_trunc_test_sw_2048_s8.py",  # dataset config
    "../_base_/models/causaltad.py",  # model config
]

window_size = 2506
data_path = "data/ego4d/features/internvideo2_1b_mq_ft_img224_stride8_len16_interval1_ego4d/"
dataset = dict(
    train=dict(
        data_path=data_path,
        pipeline=[
            dict(type="LoadFeats", feat_format="npy"),
            dict(type="ConvertToTensor", keys=["feats", "gt_segments", "gt_labels"]),
            dict(type="RandomTrunc", trunc_len=window_size, trunc_thresh=0.3, crop_ratio=[0.9, 1.0]),
            dict(type="Rearrange", keys=["feats"], ops="t c -> c t"),
            dict(type="Collect", inputs="feats", keys=["masks", "gt_segments", "gt_labels"]),
        ],
    ),
    val=dict(data_path=data_path, window_size=window_size),
    test=dict(data_path=data_path, window_size=window_size),
)

model = dict(
    projection=dict(
        in_channels=1408,
        arch=(2, 2, 9),
        use_abs_pe=True,
        max_seq_len=window_size,
        input_pdrop=0.2,
    ),
    neck=dict(num_levels=10),
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
        loss_weight=0.2,
    ),
)

solver = dict(
    train=dict(batch_size=2, num_workers=2),
    val=dict(batch_size=2, num_workers=2),
    test=dict(batch_size=2, num_workers=2),
    clip_grad_norm=1,
    ema=True,
    amp=True,
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
    val_start_epoch=11,
    end_epoch=20,
)

work_dir = "exps/ego4d/causal_internvideo2"
