_base_ = ["e2e_anet_videomae_s_192x4_160_adapter.py"]

resize_length = 192
scale_factor = 4
chunk_num = resize_length * scale_factor // 16  # 768/16=48 chunks, since videomae takes 16 frames as input
dataset = dict(
    train=dict(
        class_agnostic=False,
        resize_length=resize_length,
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4", prefix="v_"),
            dict(type="mmaction.DecordInit", num_threads=4),
            dict(type="LoadFrames", num_clips=1, method="resize", scale_factor=scale_factor),  # load 192x4=768 frames
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, 256)),
            dict(type="mmaction.RandomResizedCrop"),
            dict(type="mmaction.Resize", scale=(224, 224), keep_ratio=False),
            dict(type="mmaction.Flip", flip_ratio=0.5),
            dict(type="mmaction.ImgAug", transforms="default"),
            dict(type="mmaction.ColorJitter"),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            dict(type="ConvertToTensor", keys=["imgs", "gt_segments", "gt_labels"]),
            dict(type="Collect", inputs="imgs", keys=["masks", "gt_segments", "gt_labels"]),
        ],
    ),
    val=dict(
        class_agnostic=False,
        resize_length=resize_length,
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4", prefix="v_"),
            dict(type="mmaction.DecordInit", num_threads=4),
            dict(type="LoadFrames", num_clips=1, method="resize", scale_factor=scale_factor),  # load 192x4=768 frames
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, 224)),
            dict(type="mmaction.CenterCrop", crop_size=224),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            dict(type="ConvertToTensor", keys=["imgs", "gt_segments", "gt_labels"]),
            dict(type="Collect", inputs="imgs", keys=["masks", "gt_segments", "gt_labels"]),
        ],
    ),
    test=dict(
        class_agnostic=False,
        resize_length=resize_length,
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4", prefix="v_"),
            dict(type="mmaction.DecordInit", num_threads=4),
            dict(type="LoadFrames", num_clips=1, method="resize", scale_factor=scale_factor),  # load 192x4=768 frames
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, 224)),
            dict(type="mmaction.CenterCrop", crop_size=224),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            dict(type="ConvertToTensor", keys=["imgs"]),
            dict(type="Collect", inputs="imgs", keys=["masks"]),
        ],
    ),
)

model = dict(
    backbone=dict(
        backbone=dict(
            embed_dims=1024,
            depth=24,
            num_heads=16,
            adapter_index=list(range(24)),
        ),
        custom=dict(pretrain="pretrained/vit-large-p16_videomae-k400-pre_16x4x1_kinetics-400_20221013-229dbb03.pth"),
    ),
    projection=dict(in_channels=1024, out_channels=512),
    neck=dict(in_channels=512, out_channels=512),
    rpn_head=dict(
        in_channels=512,
        feat_channels=512,
        num_classes=200,
        label_smoothing=0.1,
        loss_weight=1.0,
        loss_normalizer=200,
    ),
)

optimizer = dict(
    type="AdamW",
    lr=1e-3,
    weight_decay=0.05,
    paramwise=True,
    backbone=dict(
        lr=0,
        weight_decay=0,
        custom=[dict(name="adapter", lr=8e-5, weight_decay=0.05)],
        exclude=["backbone"],
    ),
)
scheduler = dict(type="LinearWarmupCosineAnnealingLR", warmup_epoch=5, max_epoch=20)

post_processing = dict(
    _delete_=True,
    nms=dict(
        use_soft_nms=True,
        sigma=0.75,
        max_seg_num=100,
        min_score=0.001,
        multiclass=True,
        voting_thresh=0.9,  #  set 0 to disable
    ),
    save_dict=False,
)

workflow = dict(val_start_epoch=10, end_epoch=15)

work_dir = "exps/anet/adatad/e2e_actionformer_videomae_l_192x4_224_adapter_cls_e20_lr8e-5"
