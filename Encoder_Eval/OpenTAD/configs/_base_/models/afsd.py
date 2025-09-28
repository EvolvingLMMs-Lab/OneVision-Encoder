model = dict(
    type="AFSD",
    neck=dict(
        type="AFSDNeck",
        in_channels=2048,
        out_channels=512,
        frame_num=768,  # 96*8
        layer_num=6,
    ),
    rpn_head=dict(
        type="AFSDCoarseHead",
        in_channels=512,
        out_channels=512,
        frame_num=768,  # 96*8
        fpn_strides=[4, 8, 16, 32, 64, 128],
        num_classes=2,
        layer_num=6,
        feat_t=768 // 8,
    ),
    roi_head=dict(
        type="AFSDRefineHead",
        in_channels=512,
        num_classes=2,
        # for loss
        overlap_thresh=0.6,
        loc_weight=1.0,
        loc_bounded=True,
        use_smooth_l1=True,
    ),
)
