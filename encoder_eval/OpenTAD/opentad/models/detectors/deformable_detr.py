import torch
import torch.nn as nn

from ..builder import DETECTORS
from .detr import DETR
from ..bricks import AffineDropPath
from ..utils.bbox_tools import proposal_cw_to_se
from ..utils.post_processing import batched_nms, convert_to_seconds


@DETECTORS.register_module()
class DeformableDETR(DETR):
    def __init__(
        self,
        projection,
        transformer,
        neck=None,
        backbone=None,
    ):
        super(DeformableDETR, self).__init__(
            projection=projection,
            transformer=transformer,
            neck=neck,
            backbone=backbone,
        )

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
        if isinstance(masks, list):
            padding_masks = [~mask for mask in masks]
        elif isinstance(masks, torch.Tensor):
            padding_masks = ~masks
        else:
            raise TypeError("masks should be either list or torch.Tensor")

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
        if isinstance(masks, list):
            padding_masks = [~mask for mask in masks]
        elif isinstance(masks, torch.Tensor):
            padding_masks = ~masks
        else:
            raise TypeError("masks should be either list or torch.Tensor")

        output = self.transformer.forward_test(x, padding_masks, **kwargs)

        predictions = output, masks[0]
        return predictions

    @torch.no_grad()
    def post_processing(self, predictions, metas, post_cfg, ext_cls, **kwargs):
        output, masks = predictions
        pred_logits = output["pred_logits"]  #  [B,K,num_classes], before sigmoid
        pred_boxes = output["pred_boxes"]  # [B,K,2]

        pre_nms_topk = getattr(post_cfg, "pre_nms_topk", 200)
        bs, _, num_classes = pred_logits.shape

        # Select top-k confidence boxes for inference
        prob = pred_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(bs, -1), pre_nms_topk, dim=1)
        batch_scores = topk_values
        topk_boxes = torch.div(topk_indexes, num_classes, rounding_mode="floor")
        batch_labels = torch.fmod(topk_indexes, num_classes)

        batch_proposals = proposal_cw_to_se(pred_boxes) * torch.sum(masks, dim=1)[:, None, None]  # cw -> sw, 0~tscale
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
        # separate out all parameters that with / without weight decay
        # see https://github.com/karpathy/minGPT/blob/master/mingpt/model.py#L134
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear, nn.Conv1d)
        blacklist_weight_modules = (nn.LayerNorm, nn.GroupNorm)

        # loop over all modules / params
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

                # exclude the backbone parameters
                if fpn.startswith("backbone"):
                    continue

                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)
                elif pn.endswith("scale") and isinstance(m, AffineDropPath):
                    # corner case of our scale layer
                    no_decay.add(fpn)
                elif pn.endswith("in_proj_weight") and isinstance(m, nn.MultiheadAttention):
                    decay.add(fpn)
                elif pn.endswith("level_embeds"):
                    # corner case for position encoding
                    no_decay.add(fpn)
                elif pn.endswith("weight") and ("tgt_embed" in pn):
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters() if not pn.startswith("backbone")}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (str(param_dict.keys() - union_params),)

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": cfg["weight_decay"]},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        return optim_groups
