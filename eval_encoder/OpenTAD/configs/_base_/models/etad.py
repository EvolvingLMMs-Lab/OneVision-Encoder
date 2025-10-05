model = dict(
    type="ETAD",
    projection=dict(
        type="ConvSingleProj",
        in_channels=400,
        out_channels=256,
        num_convs=1,
        conv_cfg=dict(groups=4),
        norm_cfg=dict(type="GN", num_groups=16),
    ),
    neck=dict(
        type="LSTMNeck",
        in_channels=256,
        out_channels=256,
        conv_cfg=dict(groups=4),
        norm_cfg=dict(type="GN", num_groups=16),
    ),
    rpn_head=dict(
        type="TemporalEvaluationHead",  # tem
        in_channels=256,
        num_classes=2,
        shared=True,
        conv_cfg=dict(groups=4),
        loss=dict(pos_thresh=0.5, gt_type=["startness", "endness"]),
    ),
    roi_head=dict(
        type="ETADRoIHead",
        stages=dict(
            number=3,
            loss_weight=[1, 1, 1],
            pos_iou_thresh=[0.7, 0.8, 0.9],
        ),
        proposal_generator=dict(
            type="ProposalMapSampling",
            tscale=128,
            dscale=128,
            sampling_ratio=0.06,
            strategy="random",
        ),
        proposal_roi_extractor=dict(
            type="ROIAlignExtractor",
            roi_size=16,
            extend_ratio=0.5,
            base_stride=1,
        ),
        proposal_head=dict(
            type="ETADHead",
            in_channels=256,
            roi_size=16,
            feat_channels=512,
            fcs_num=3,
            fcs_channels=128,
            loss=dict(
                cls_weight=1.0,
                reg_weight=5.0,
                boundary_weight=10.0,
            ),
        ),
    ),
)
