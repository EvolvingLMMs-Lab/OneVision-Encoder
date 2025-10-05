import torch

from .base import BaseDetector
from ..builder import DETECTORS, build_transformer, build_backbone, build_projection, build_neck
from ..utils.bbox_tools import proposal_cw_to_se
from ..utils.post_processing import batched_nms, convert_to_seconds


@DETECTORS.register_module()
class DETR(BaseDetector):
    def __init__(
        self,
        projection,
        transformer,
        neck=None,
        backbone=None,
    ):
        super().__init__()

        if backbone is not None:
            self.backbone = build_backbone(backbone)

        if projection is not None:
            self.projection = build_projection(projection)

        if neck is not None:
            self.neck = build_neck(neck)

        self.transformer = build_transformer(transformer)

    @property
    def with_backbone(self):
        """bool: whether the detector has backbone"""
        return hasattr(self, "backbone") and self.backbone is not None

    @property
    def with_projection(self):
        """bool: whether the detector has projection"""
        return hasattr(self, "projection") and self.projection is not None

    @property
    def with_neck(self):
        """bool: whether the detector has neck"""
        return hasattr(self, "neck") and self.neck is not None

    def forward_train(self, inputs, masks, metas, gt_segments, gt_labels, **kwargs):
        if self.with_backbone:
            x = self.backbone(inputs)
        else:
            x = inputs

        if self.with_projection:
            x, masks = self.projection(x, masks)

        if self.with_neck:
            x, masks = self.neck(x, masks)

        # padding masks is the opposite of valid masks
        padding_masks = ~masks

        losses = dict()
        transformer_loss = self.transformer.forward_train(
            x,
            padding_masks,
            gt_segments=gt_segments,
            gt_labels=gt_labels,
            **kwargs,
        )
        losses.update(transformer_loss)

        # only key has loss will be record
        losses["cost"] = sum(_value for _key, _value in losses.items())
        return losses

    def forward_test(self, inputs, masks, metas=None, infer_cfg=None, **kwargs):
        if self.with_backbone:
            x = self.backbone(inputs)
        else:
            x = inputs

        if self.with_projection:
            x, masks = self.projection(x, masks)

        if self.with_neck:
            x, masks = self.neck(x, masks)

        # padding masks is the opposite of valid masks
        padding_masks = ~masks
        output = self.transformer.forward_test(x, padding_masks, **kwargs)

        predictions = output["pred_logits"], output["pred_boxes"], masks
        return predictions

    @torch.no_grad()
    def post_processing(self, predictions, metas, post_cfg, ext_cls, **kwargs):
        pred_logits, pred_boxes, masks = predictions  # [B,K,2], [B,K,num_classes] before softmax

        # For each box we assign the best class or the second best if the best on is `no_object`.
        # batch_scores, batch_labels = F.softmax(pred_logits, dim=-1)[:, :, :-1].max(-1)
        batch_scores = pred_logits.sigmoid()
        batch_proposals = proposal_cw_to_se(pred_boxes) * torch.sum(masks, dim=1)[:, None, None]  # cw -> sw, 0~tscale

        pre_nms_thresh = 0.001
        pre_nms_topk = 200
        num_classes = batch_scores.shape[-1]

        results = {}
        for i in range(len(metas)):  # processing each video
            segments = batch_proposals[i].detach().cpu()  # [N,2]
            scores = batch_scores[i].detach().cpu()  # [N,class]
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
            pt_idxs = torch.div(topk_idxs, num_classes, rounding_mode="floor")
            cls_idxs = torch.fmod(topk_idxs, num_classes)

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
