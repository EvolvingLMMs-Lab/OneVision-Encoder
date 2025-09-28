import torch
import torch.nn as nn

from ..builder import DETECTORS
from .two_stage import TwoStageDetector
from ..utils.post_processing import boundary_choose, batched_nms, convert_to_seconds


@DETECTORS.register_module()
class ETAD(TwoStageDetector):
    def __init__(
        self,
        projection,
        neck,
        rpn_head,
        roi_head,
        backbone=None,
    ):
        super().__init__(
            backbone=backbone,
            neck=neck,
            projection=projection,
            rpn_head=rpn_head,
            roi_head=roi_head,
        )

        self.reset_params()

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward_test(self, inputs, masks, metas=None, infer_cfg=None):
        if self.with_backbone:
            x = self.backbone(inputs, masks)
        else:
            x = inputs

        if self.with_projection:
            x, masks = self.projection(x, masks)

        if self.with_neck:
            x, masks = self.neck(x, masks)

        # tem score
        tem_score = self.rpn_head.forward_test(x, masks).sigmoid()  # [B,2,T]

        # iou score
        proposals, pred_iou = self.roi_head.forward_test(x)  # [D,T,2]

        # pack all
        predictions = tem_score, pred_iou, proposals
        return predictions

    def post_processing(self, predictions, metas, post_cfg, ext_cls, **kwargs):
        assert ext_cls != None  #  BMN is a proposal generation network

        tscale = self.roi_head.proposal_generator.tscale

        tem_score, pred_iou, proposals = predictions
        pred_iou = pred_iou[..., 0] * pred_iou[..., 1]

        start_mask = boundary_choose(tem_score[:, 0, :])
        start_mask[:, 0] = True
        end_mask = boundary_choose(tem_score[:, 1, :])
        end_mask[:, -1] = True

        results = {}
        for i in range(len(metas)):  # processing each video
            start_idx = proposals[i][:, 0].int().clip(0, tscale - 1)
            end_idx = proposals[i][:, 1].int().clip(0, tscale - 1)
            idx = (start_mask[i][start_idx] == 1) & (end_mask[i][end_idx] == 1)

            segments = proposals[i][idx].detach().cpu()
            scores = pred_iou[i][idx].detach().cpu()
            # scores = (tem_score[i, 0][start_idx[idx]] * tem_score[i, 1][end_idx[idx]] * pred_iou[i][idx]).detach().cpu()

            labels = torch.zeros_like(scores)  # pseudo labels

            segments, scores, labels = batched_nms(segments, scores, labels, **post_cfg.nms)

            video_id = metas[i]["video_name"]

            # convert segments to seconds
            segments = convert_to_seconds(segments, metas[i])

            # merge with external classifier
            segments, labels, scores = ext_cls(video_id, segments, scores)

            results_per_video = []
            for segment, label, score in zip(segments, labels, scores):
                # convert to python scalars
                results_per_video.append(
                    dict(
                        segment=[round(seg.item(), 2) for seg in segment],
                        label=label,
                        score=round(score.item(), 4),
                    )
                )

            if video_id in results.keys():
                results[video_id].extend(results_per_video)
            else:
                results[video_id] = results_per_video
        return results
