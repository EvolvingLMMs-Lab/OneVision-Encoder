tscale = 256
model = dict(
    type="VSGN",
    projection=dict(
        type="VSGNPyramidProj",
        in_channels=2048,
        out_channels=256,
        pyramid_levels=[2, 4, 8, 16, 32],
    ),
    neck=dict(
        type="VSGNFPN",
        in_channels=256,
        out_channels=256,
        num_levels=5,
    ),
    rpn_head=dict(
        type="VSGNRPNHead",
        in_channels=256,
        num_layers=4,
        num_classes=1,
        iou_thr=0.6,
        anchor_generator=dict(
            pyramid_levels=[2, 4, 8, 16, 32],
            tscale=tscale,
            anchor_scale=[7, 7.5],
        ),
        tem_head=dict(
            type="TemporalEvaluationHead",
            in_channels=256,
            num_classes=3,
            loss=dict(pos_thresh=0.5, gt_type=["startness", "endness", "actionness"]),
        ),
        loss_cls=dict(type="BalancedCEloss"),  # CE for THUMOS, Sigmoid for ANET
        loss_loc=dict(type="GIoULoss"),  # this is only a placeholder
    ),
    roi_head=dict(
        type="VSGNRoIHead",
        in_channels=256,
        iou_thr=0.7,
        roi_extractor=dict(type="CornerExtractor", beta=8.0, base_stride=2, tscale=tscale),
        loss_loc=dict(type="GIoULoss"),  # this is only a placeholder
    ),
)
