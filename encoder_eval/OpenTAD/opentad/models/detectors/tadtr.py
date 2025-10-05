import torch

from ..builder import DETECTORS
from .deformable_detr import DeformableDETR
from ..utils.bbox_tools import proposal_cw_to_se
from ..utils.post_processing import batched_nms, convert_to_seconds


@DETECTORS.register_module()
class TadTR(DeformableDETR):
    def __init__(
        self,
        projection,
        transformer,
        neck=None,
        backbone=None,
    ):
        super(TadTR, self).__init__(
            projection=projection,
            transformer=transformer,
            neck=neck,
            backbone=backbone,
        )
        self.with_act_reg = transformer.with_act_reg

    @torch.no_grad()
    def post_processing(self, predictions, metas, post_cfg, ext_cls, **kwargs):
        output, masks = predictions
        pred_logits = output["pred_logits"]  #  [B,K,num_classes], before sigmoid
        pred_boxes = output["pred_boxes"]  # [B,K,2]

        pre_nms_topk = getattr(post_cfg, "pre_nms_topk", 200)
        bs, _, num_classes = pred_logits.shape

        # Select top-k confidence boxes for inference
        if self.with_act_reg:
            prob = pred_logits.sigmoid() * output["pred_actionness"]
        else:
            prob = pred_logits.sigmoid()

        pre_nms_topk = min(pre_nms_topk, prob.view(bs, -1).shape[1])
        topk_values, topk_indexes = torch.topk(prob.view(bs, -1), pre_nms_topk, dim=1)
        batch_scores = topk_values
        topk_boxes = torch.div(topk_indexes, num_classes, rounding_mode="floor")
        batch_labels = torch.fmod(topk_indexes, num_classes)

        batch_proposals = proposal_cw_to_se(pred_boxes) * masks.shape[-1]  # cw -> sw, 0~tscale
        batch_proposals = torch.gather(batch_proposals, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 2))

        results = {}
        for i in range(len(metas)):  # processing each video
            segments = batch_proposals[i].detach().cpu()  # [N,2]
            scores = batch_scores[i].detach().cpu()  # [N,class]
            labels = batch_labels[i].detach().cpu()  # [N]

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

    def get_optim_groups(self, cfg):
        def match_name_keywords(n, name_keywords):
            out = False
            for b in name_keywords:
                if b in n:
                    out = True
                    break
            return out

        lr_linear_proj_names = ["reference_points", "sampling_offsets"]

        param_dicts = [
            # non-backbone, non-offset
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not n.startswith("backbone")
                    and not match_name_keywords(n, lr_linear_proj_names)
                    and p.requires_grad
                ],
                "lr": cfg.lr,
                "initial_lr": cfg.lr,
            },
            # offset
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not n.startswith("backbone") and match_name_keywords(n, lr_linear_proj_names) and p.requires_grad
                ],
                "lr": cfg.lr * 0.1,
                "initial_lr": cfg.lr * 0.1,
            },
        ]
        return param_dicts
