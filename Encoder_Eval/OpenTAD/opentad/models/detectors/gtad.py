from ..builder import DETECTORS
from . import BMN


@DETECTORS.register_module()
class GTAD(BMN):
    def __init__(
        self,
        projection,
        neck,
        rpn_head,
        roi_head,
        backbone=None,
    ):
        super(BMN, self).__init__(
            backbone=backbone,
            neck=neck,
            projection=projection,
            rpn_head=rpn_head,
            roi_head=roi_head,
        )
