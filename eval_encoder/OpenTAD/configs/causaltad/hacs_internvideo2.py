_base_ = [
    "../_base_/datasets/hacs-1.1.1/features_slowfast_pad.py",  # dataset config
    "../_base_/models/causaltad.py",  # model config
]

block_list = [
    "-LzyV1PtJXE",
    "6okHpDA7caA",
    "8P9hAN-teOU",
    "AcOgvJ6U0T8",
    "AkMSIaZyX00",
    "Cm2j1EhVkHc",
    "EEvcgmd8kzg",
    "HjunnoyAinU",
    "Ht2gV7oaqbo",
    "Jbu3hE_CQaw",
    "Lp1oWVjxm4I",
    "New9JV1dKSU",
    "PcltZ1RZmZ0",
    "Q_QRFa5r3s0",
    "S4ZC3rz0q5c",
    "ShwMX7iMdCw",
    "V9uNF5W9KjM",
    "ZrhHEvR84AE",
    "d0ViiZ_QsLo",
    "jsuwmH5Y7OM",
    "mAE0CQURjj8",
    "mllZ0ycwvTs",
    "mnhMpLONbtY",
    "oUMmneMSfC0",
    "tqBKTZxSxwQ",
    "vA4STJJyyxU",
    "xaAjiyc4VmM",
    "y41wrOt1K1M",
]  # missing video in HACS, not used in evaluation

trunc_len = 2304
data_path = "data/hacs-1.1.1/features/hacs_6b/"
dataset = dict(
    train=dict(
        data_path=data_path,
        block_list=None,
        feature_stride=8,
        offset_frames=8,
        fps=30,
        pipeline=[
            dict(type="LoadFeats", feat_format="pt", prefix="v_", suffix="_spatial_pool_feature_6"),
            dict(type="ConvertToTensor", keys=["feats", "gt_segments", "gt_labels"]),
            dict(type="RandomTrunc", trunc_len=trunc_len, trunc_thresh=0.5, crop_ratio=[0.9, 1.0]),
            dict(type="Rearrange", keys=["feats"], ops="t c-> c t"),
            dict(type="Collect", inputs="feats", keys=["masks", "gt_segments", "gt_labels"]),
        ],
    ),
    val=dict(
        data_path=data_path,
        block_list=block_list,
        feature_stride=8,
        offset_frames=8,
        fps=30,
        pipeline=[
            dict(type="LoadFeats", feat_format="pt", prefix="v_", suffix="_spatial_pool_feature_6"),
            dict(type="ConvertToTensor", keys=["feats", "gt_segments", "gt_labels"]),
            dict(type="Padding", length=trunc_len),
            dict(type="Rearrange", keys=["feats"], ops="t c-> c t"),
            dict(type="Collect", inputs="feats", keys=["masks", "gt_segments", "gt_labels"]),
        ],
    ),
    test=dict(
        data_path=data_path,
        block_list=block_list,
        feature_stride=8,
        offset_frames=8,
        fps=30,
        pipeline=[
            dict(type="LoadFeats", feat_format="pt", prefix="v_", suffix="_spatial_pool_feature_6"),
            dict(type="ConvertToTensor", keys=["feats"]),
            dict(type="Padding", length=trunc_len),
            dict(type="Rearrange", keys=["feats"], ops="t c-> c t"),
            dict(type="Collect", inputs="feats", keys=["masks"]),
        ],
    ),
)

model = dict(
    projection=dict(
        in_channels=3200,
        out_channels=256,
        use_abs_pe=True,
        max_seq_len=2304,
        input_pdrop=0.1,
    ),
    neck=dict(in_channels=256, out_channels=256),
    rpn_head=dict(
        in_channels=256,
        feat_channels=256,
        num_classes=200,
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
    amp=True,
)

optimizer = dict(type="AdamW", lr=1e-3, weight_decay=0.05, paramwise=True)
scheduler = dict(type="LinearWarmupCosineAnnealingLR", warmup_epoch=5, max_epoch=25)

inference = dict(load_from_raw_predictions=False, save_raw_prediction=False)
post_processing = dict(
    nms=dict(
        use_soft_nms=True,
        sigma=0.75,
        max_seg_num=100,
        min_score=0.001,
        multiclass=False,
        voting_thresh=0.7,  #  set 0 to disable
    ),
    external_cls=dict(
        type="StandardClassifier",
        path="data/hacs-1.1.1/classifiers/hacs_UMTv2_6B_k710+K40_f16_frozenTuning.json_converted.json",
        topk=2,
    ),
    save_dict=False,
)

workflow = dict(
    logging_interval=200,
    checkpoint_interval=1,
    val_loss_interval=-1,
    val_eval_interval=1,
    val_start_epoch=8,
)

work_dir = "exps/hacs/causal_internvideo2_6b"
