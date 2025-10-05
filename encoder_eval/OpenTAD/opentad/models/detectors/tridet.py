import torch
import torch.nn as nn

from ..builder import DETECTORS
from .single_stage import SingleStageDetector
from ..bricks import Scale, AffineDropPath
from ..utils.post_processing import batched_nms, convert_to_seconds


@DETECTORS.register_module()
class TriDet(SingleStageDetector):
    def __init__(
        self,
        projection,
        rpn_head,
        neck=None,
        backbone=None,
    ):
        super(TriDet, self).__init__(
            backbone=backbone,
            neck=neck,
            projection=projection,
            rpn_head=rpn_head,
        )

        self.max_seq_len = projection.max_seq_len
        assert len(projection.sgp_win_size) == len(rpn_head.prior_generator.strides)

        max_div_factor = 1
        for s, w in zip(rpn_head.prior_generator.strides, projection.sgp_win_size):
            stride = s * w if w > 1 else s
            if max_div_factor < stride:
                max_div_factor = stride
        self.max_div_factor = max_div_factor

    def pad_data(self, inputs, masks):
        feat_len = inputs.shape[-1]
        if feat_len <= self.max_seq_len:
            max_len = self.max_seq_len
        else:
            max_len = feat_len
            # pad the input to the next divisible size
            stride = self.max_div_factor
            max_len = (max_len + (stride - 1)) // stride * stride

        padding_size = [0, max_len - feat_len]
        inputs = torch.nn.functional.pad(inputs, padding_size, value=0)
        pad_masks = torch.zeros((inputs.shape[0], max_len), device=masks.device).bool()
        pad_masks[:, :feat_len] = masks
        return inputs, pad_masks

    def forward_train(self, inputs, masks, metas, gt_segments, gt_labels, **kwargs):
        losses = dict()
        if self.with_backbone:
            x = self.backbone(inputs)
        else:
            x = inputs

        # pad the features and unsqueeze the mask
        if not self.training:
            x, masks = self.pad_data(x, masks)

        if self.with_projection:
            x, masks = self.projection(x, masks)

        if self.with_neck:
            x, masks = self.neck(x, masks)

        loc_losses = self.rpn_head.forward_train(
            x,
            masks,
            gt_segments=gt_segments,
            gt_labels=gt_labels,
            **kwargs,
        )
        losses.update(loc_losses)

        # only key has loss will be record
        losses["cost"] = sum(_value for _key, _value in losses.items())
        return losses

    def forward_test(self, inputs, masks, metas=None, infer_cfg=None, **kwargs):
        if self.with_backbone:
            x = self.backbone(inputs)
        else:
            x = inputs

        x, masks = self.pad_data(x, masks)

        if self.with_projection:
            x, masks = self.projection(x, masks)

        if self.with_neck:
            x, masks = self.neck(x, masks)

        points, rpn_reg, rpn_scores = self.rpn_head.forward_test(x, masks, **kwargs)
        predictions = points, rpn_reg, rpn_scores
        return predictions

    def post_processing(self, predictions, metas, post_cfg, ext_cls, **kwargs):
        points, rpn_reg, rpn_scores = predictions  # [N, 4], [B, num_classes, N, 2], [B, N, num_classes]

        pre_nms_thresh = 0.001
        pre_nms_topk = 2000
        num_classes = rpn_scores.shape[-1]
        points = points.cpu()

        results = {}
        for i in range(len(metas)):  # processing each video
            scores = rpn_scores[i].detach().cpu()  # [N]
            reg = rpn_reg[i].detach().cpu()  # [num_classes, N, 2]

            if num_classes == 1:
                segments = self.rpn_head.get_proposals(points, reg.squeeze(0)).detach().cpu()  # [N, 2]
                scores = scores.squeeze(-1)
                labels = torch.zeros(scores.shape[0]).contiguous()
            else:
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

                segments = self.rpn_head.get_proposals(points[pt_idxs], reg[cls_idxs, pt_idxs]).detach().cpu()  # [N, 2]
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
                elif pn.endswith("scale") and isinstance(m, (Scale, AffineDropPath)):
                    # corner case of our scale layer
                    no_decay.add(fpn)
                elif pn.endswith("rel_pe"):
                    # corner case for relative position encoding
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
