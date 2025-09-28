import torch
import torch.nn as nn
from .two_stage import TwoStageDetector
from ..builder import DETECTORS
from ..utils.post_processing import batched_nms, convert_to_seconds


@DETECTORS.register_module()
class VSGN(TwoStageDetector):
    def __init__(
        self,
        backbone=None,
        projection=None,
        neck=None,
        rpn_head=None,
        roi_head=None,
    ):
        super(VSGN, self).__init__(
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

    def forward_train(self, inputs, masks, metas, gt_segments, gt_labels, **kwargs):
        losses = dict()

        if self.with_backbone:
            x = self.backbone(inputs, masks)
        else:
            x = inputs

        if self.with_projection:
            x, masks = self.projection(x, masks)

        if self.with_neck:
            x, masks = self.neck(x, masks)

        rpn_losses, rpn_proposals = self.rpn_head.forward_train(
            x,
            masks,
            gt_segments=gt_segments,
            gt_labels=gt_labels,
            **kwargs,
        )
        losses.update(rpn_losses)

        roi_losses = self.roi_head.forward_train(
            x,
            rpn_proposals,
            gt_segments=gt_segments,
            gt_labels=gt_labels,
            **kwargs,
        )
        losses.update(roi_losses)

        losses["cost"] = sum(_value for _key, _value in losses.items())
        return losses

    def forward_test(self, inputs, masks, metas=None, infer_cfg=None, **kwargs):
        if self.with_backbone:
            x = self.backbone(inputs, masks)
        else:
            x = inputs

        if self.with_projection:
            x, masks = self.projection(x, masks)

        if self.with_neck:
            x, masks = self.neck(x, masks)

        tem_logits, rpn_proposals, rpn_scores = self.rpn_head.forward_test(x, masks, **kwargs)

        roi_proposals = self.roi_head.forward_test(x, rpn_proposals, **kwargs)

        # pack all
        predictions = tem_logits, rpn_scores, roi_proposals
        return predictions

    def post_processing(self, predictions, metas, post_cfg, ext_cls, **kwargs):
        tem_logits, rpn_scores, roi_proposals = predictions
        # tem_logits [B,3,T], rpn_scores [B,N,num_class], roi_proposals [B,N,2]

        pre_nms_thresh = 0.001
        pre_nms_topk = 2000
        num_classes = rpn_scores.shape[-1]
        tscale = tem_logits.shape[-1]
        start_scores, end_scores, _ = tem_logits.sigmoid().detach().cpu().unbind(dim=1)  # [B,T]

        results = {}

        for i in range(len(metas)):  # processing each video
            segments = roi_proposals[i].detach().cpu()  # [N,2]
            segments = segments.clip(min=0, max=tscale - 1)

            start_score_l = start_scores[i][torch.floor(segments[:, 0]).int()]
            start_score_r = start_scores[i][torch.ceil(segments[:, 0]).int()]
            start_score = (start_score_l + start_score_r) * 0.5

            end_score_l = end_scores[i][torch.floor(segments[:, 1]).int()]
            end_score_r = end_scores[i][torch.ceil(segments[:, 1]).int()]
            end_score = (end_score_l + end_score_r) * 0.5

            scores = rpn_scores[i].detach().cpu() * (start_score * end_score).unsqueeze(-1)  # [N,class]

            if num_classes == 1:
                scores = scores.squeeze(-1)
                labels = torch.zeros(scores.shape[0]).contiguous()
            else:
                # VSGN has the background class, so we remove it
                scores = scores[:, 1:]

                pred_prob = scores.flatten()  # [N*class]

                # Apply filtering to make NMS faster following detectron2
                # 1. Keep seg with confidence score > a threshold
                keep_idxs1 = pred_prob > pre_nms_thresh
                pred_prob = pred_prob[keep_idxs1]
                topk_idxs = keep_idxs1.nonzero(as_tuple=True)[0]

                # 2. Keep top k top scoring boxes only
                num_topk = min(pre_nms_topk, topk_idxs.size(0))
                pred_prob, idxs = pred_prob.sort(descending=True)
                pred_prob = pred_prob[:num_topk].clone()
                topk_idxs = topk_idxs[idxs[:num_topk]].clone()

                # 3. gather predicted proposals
                # num_classes=21, so we should minus 1
                pt_idxs = torch.div(topk_idxs, num_classes - 1, rounding_mode="floor")
                cls_idxs = torch.fmod(topk_idxs, num_classes - 1)

                segments = segments[pt_idxs]
                scores = pred_prob
                labels = cls_idxs

            # if not sliding window, do nms
            if post_cfg.sliding_window == False and post_cfg.nms is not None:
                segments, scores, labels = batched_nms(segments, scores, labels, **post_cfg.nms)

            video_id = metas[i]["video_name"]

            # convert segments to seconds
            segments = convert_to_seconds(segments, metas[i])

            # merge with external classifier
            if isinstance(ext_cls, list):  # own classification results
                labels = [ext_cls[label.item()] for label in labels]
            else:
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
