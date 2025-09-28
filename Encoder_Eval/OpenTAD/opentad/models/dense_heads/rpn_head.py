from ..builder import HEADS
from .anchor_head import AnchorHead


@HEADS.register_module()
class RPNHead(AnchorHead):
    def __init__(
        self,
        in_channels,
        feat_channels,
        num_convs=3,
        prior_generator=None,
        loss=None,
        cls_prior_prob=0.01,
    ):
        super().__init__(
            num_classes=1,
            in_channels=in_channels,
            feat_channels=feat_channels,
            num_convs=num_convs,
            prior_generator=prior_generator,
            loss=loss,
            cls_prior_prob=cls_prior_prob,
        )
