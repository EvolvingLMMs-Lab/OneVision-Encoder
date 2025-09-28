model = dict(
    type="TemporalMaxer",
    projection=dict(
        type="TemporalMaxerProj",
        in_channels=2048,
        out_channels=512,
        arch=(2, 0, 5),  # feature projection layers, downsampling layer
        conv_cfg=dict(kernel_size=3),
        norm_cfg=dict(type="LN"),
    ),
    neck=dict(
        type="FPNIdentity",
        in_channels=512,
        out_channels=512,
        num_levels=6,
    ),
    rpn_head=dict(
        type="TemporalMaxerHead",
        num_classes=20,
        in_channels=512,
        feat_channels=512,
        num_convs=2,
        cls_prior_prob=0.01,
        prior_generator=dict(
            type="PointGenerator",
            strides=[1, 2, 4, 8, 16, 32],
            regression_range=[(0, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 10000)],
        ),
        loss_normalizer=100,
        loss_normalizer_momentum=0.9,
        loss=dict(
            cls_loss=dict(type="FocalLoss"),
            reg_loss=dict(type="DIOULoss"),
        ),
        assigner=dict(
            type="AnchorFreeSimOTAAssigner",
            iou_weight=2,
            cls_weight=1.0,
            center_radius=1.5,
            keep_percent=1.0,
            confuse_weight=0.0,
        ),
    ),
)
