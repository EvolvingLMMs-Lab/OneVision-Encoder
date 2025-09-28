model = dict(
    type="GTAD",
    projection=dict(
        type="ConvSingleProj",
        num_convs=1,
        in_channels=400,
        out_channels=256,
        conv_cfg=dict(groups=4),
    ),
    neck=dict(
        type="GCNeXt",
        in_channels=256,
        out_channels=256,
        k=3,
        groups=32,
    ),
    rpn_head=dict(
        type="GCNextTemporalEvaluationHead",
        in_channels=256,
        num_classes=2,
        loss=dict(pos_thresh=0.5, gt_type=["startness", "endness"]),
    ),
    roi_head=dict(
        type="StandardProposalMapHead",
        proposal_generator=dict(type="DenseProposalMap", tscale=100, dscale=100),
        proposal_roi_extractor=dict(
            type="GTADExtractor",
            in_channels=256,
            out_channels=512,
            tscale=100,
            dscale=100,
        ),
        proposal_head=dict(
            type="PEMHead",  # FC_head
            in_channels=512,
            feat_channels=128,
            num_convs=3,
            num_classes=2,
            kernel_size=1,
            loss=dict(
                cls_loss=dict(type="BalancedBCELoss", pos_thresh=0.9),
                reg_loss=dict(type="BalancedL2Loss", high_thresh=0.7, low_thresh=0.3, weight=5.0),
            ),
        ),
    ),
)
