import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ..builder import DETECTORS
from .two_stage import TwoStageDetector
from ..utils.post_processing import batched_nms, convert_to_seconds


@DETECTORS.register_module()
class AFSD(TwoStageDetector):
    def __init__(
        self,
        neck,
        rpn_head,
        roi_head,
        projection=None,
        backbone=None,
    ):
        super(AFSD, self).__init__(
            backbone=backbone,
            projection=projection,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
        )

        self.num_classes = self.roi_head.num_classes
        self.clip_len = rpn_head.frame_num
        self.scale_factor = rpn_head.frame_num // rpn_head.feat_t
        self.loss = MultiSegmentLoss(
            self.num_classes,
            self.clip_len,
            overlap_thresh=roi_head.overlap_thresh,
            negpos_ratio=1.0,
            loc_weight=roi_head.loc_weight,
            loc_bounded=roi_head.loc_bounded,
            use_smooth_l1=roi_head.use_smooth_l1,
        )
        self.calc_bce_loss = BCELoss()

    def forward_train(self, inputs, masks, metas, gt_segments, gt_labels, **kwargs):
        if self.with_backbone:
            x = self.backbone(inputs, masks)  # [B,C,256,96,96] -> [B,C,256,3,3]
        else:
            x = inputs

        if self.with_projection:
            x, masks = self.projection(x, masks)

        # AFSD Neck
        pyramid_feats, frame_level_feat = self.neck(x, masks)

        # AFSD Coarse Head
        (
            loc,
            conf,
            priors,
            loc_feats,
            conf_feats,
            segments,
            frame_segments,
        ) = self.rpn_head.forward_train(pyramid_feats, **kwargs)

        # AFSD Refine Head
        (
            start,
            end,
            prop_loc,
            prop_conf,
            center,
            start_loc_prop,
            end_loc_prop,
            start_conf_prop,
            end_conf_prop,
        ) = self.roi_head.forward_train(
            frame_level_feat,
            loc_feats,
            conf_feats,
            segments,
            frame_segments,
            **kwargs,
        )

        # compute loss
        gt_segments = [bboxes / masks.shape[1] for bboxes in gt_segments]  # normalize gt_segments to [0,1]
        gt_labels = [labels + 1 for labels in gt_labels]  # change 0 to 1, since softmax

        losses = self.loss(
            loc,
            conf,
            prop_loc,
            prop_conf,
            center,
            priors,
            gt_segments,
            gt_labels,
        )

        losses.update(
            self.compute_start_end_loss(
                start,
                end,
                start_loc_prop,
                end_loc_prop,
                start_conf_prop,
                end_conf_prop,
                gt_segments,
            )
        )

        # only key has loss will be record
        losses["cost"] = sum(_value for _key, _value in losses.items())
        return losses

    def compute_start_end_loss(
        self,
        pred_start,
        pred_end,
        start_loc_prop,
        end_loc_prop,
        start_conf_prop,
        end_conf_prop,
        gt_segments,
    ):
        with torch.no_grad():
            scores = []
            for gt_segment in gt_segments:
                start = np.zeros([self.clip_len])
                end = np.zeros([self.clip_len])
                for seg in gt_segment:
                    s, e = seg.cpu().numpy() * self.clip_len
                    d = max((e - s) / 10.0, 2.0)
                    start_s = np.clip(int(round(s - d / 2)), 0, self.clip_len - 1)
                    start_e = np.clip(int(round(s + d / 2)), 0, self.clip_len - 1) + 1
                    start[start_s:start_e] = 1
                    end_s = np.clip(int(round(e - d / 2)), 0, self.clip_len - 1)
                    end_e = np.clip(int(round(e + d / 2)), 0, self.clip_len - 1) + 1
                    end[end_s:end_e] = 1

                scores.append(torch.from_numpy(np.stack([start, end], axis=0)))
            scores = torch.stack(scores, dim=0).to(pred_start.device).float()

        loss_start, loss_end = self.calc_bce_loss(pred_start, pred_end, scores)
        scores_ = F.interpolate(scores, scale_factor=1.0 / self.scale_factor)
        loss_start_loc_prop, loss_end_loc_prop = self.calc_bce_loss(start_loc_prop, end_loc_prop, scores_)
        loss_start_conf_prop, loss_end_conf_prop = self.calc_bce_loss(start_conf_prop, end_conf_prop, scores_)
        loss_start = loss_start + 0.1 * (loss_start_loc_prop + loss_start_conf_prop)
        loss_end = loss_end + 0.1 * (loss_end_loc_prop + loss_end_conf_prop)
        return dict(loss_start=loss_start, loss_end=loss_end)

    def forward_test(self, inputs, masks, metas=None, infer_cfg=None):
        if self.with_backbone:
            x = self.backbone(inputs, masks)  # [B,C,256,96,96]->
        else:
            x = inputs

        if self.with_projection:
            x, masks = self.projection(x, masks)

        # AFSD Neck
        pyramid_feats, frame_level_feat = self.neck(x, masks)

        # AFSD Coarse Head
        loc, conf, priors, loc_feats, conf_feats, segments, frame_segments = self.rpn_head.forward_train(pyramid_feats)

        # AFSD Refine Head
        _, _, prop_loc, prop_conf, center, _, _, _, _ = self.roi_head.forward_train(
            frame_level_feat, loc_feats, conf_feats, segments, frame_segments
        )

        # pack all
        predictions = loc, conf, priors, prop_loc, prop_conf, center
        return predictions

    def post_processing(self, predictions, metas, post_cfg, ext_cls, **kwargs):
        locs, confs, priors, prop_locs, prop_confs, centers = predictions

        pre_nms_thresh = getattr(post_cfg, "pre_nms_thresh", 0.001)
        pre_nms_topk = getattr(post_cfg, "pre_nms_topk", 2000)

        results = {}
        for i in range(len(metas)):  # processing each video
            loc = locs[i]  # [T,2]
            conf = nn.Softmax(dim=-1)(confs[i])  # [N,num_class]
            prop_loc = prop_locs[i]  # [K,2]
            prop_conf = nn.Softmax(dim=-1)(prop_confs[i])  # [N,num_class]
            center = centers[i].sigmoid()  # [N,1]

            pre_loc_w = loc[:, :1] + loc[:, 1:]
            loc = 0.5 * pre_loc_w * prop_loc + loc
            decoded_segments = torch.cat(
                [
                    priors[:, :1] * self.clip_len - loc[:, :1],
                    priors[:, :1] * self.clip_len + loc[:, 1:],
                ],
                dim=-1,
            )
            decoded_segments.clamp_(min=0, max=self.clip_len)  # [N,2]
            decoded_segments = decoded_segments.cpu()

            conf = (conf + prop_conf) / 2.0
            conf = conf * center
            conf = conf.view(-1, self.num_classes).transpose(1, 0)
            conf_scores = conf.clone()

            # class 0 is background, we remove it
            pred_prob = conf_scores[1:].transpose(0, 1).cpu()  # [N,num_class-1]
            num_classes = pred_prob.shape[1]
            if num_classes == 1:  # anet
                scores = pred_prob.squeeze(-1)
                labels = torch.zeros(scores.shape[0]).to(scores.device)
            else:  # thumos
                pred_prob = pred_prob.flatten()  # [N*num_class-1]

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

                decoded_segments = decoded_segments[pt_idxs]
                scores = pred_prob
                labels = cls_idxs

            # convert segments from 0~frame_num to 0~feature_grid
            if metas[i]["fps"] == -1:  # resize setting, like in anet / hacs
                segments = decoded_segments.cpu() / self.clip_len * metas[i]["resize_length"]
            else:  # sliding window / padding setting, like in thumos / ego4d
                segments = decoded_segments.cpu() / self.clip_len * metas[i]["window_size"]

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


class MultiSegmentLoss(nn.Module):
    def __init__(
        self,
        num_classes,
        clip_len,
        overlap_thresh,
        negpos_ratio,
        loc_weight=1.0,
        cls_weight=1.0,
        loc_bounded=True,
        use_smooth_l1=False,
    ):
        super(MultiSegmentLoss, self).__init__()
        self.num_classes = num_classes
        self.clip_len = clip_len
        self.overlap_thresh = overlap_thresh
        self.negpos_ratio = negpos_ratio
        self.focal_loss = FocalLoss_Ori(num_classes, balance_index=0, size_average=False, alpha=0.25)
        self.loc_weight = loc_weight
        self.cls_weight = cls_weight
        self.loc_bounded = loc_bounded
        self.use_smooth_l1 = use_smooth_l1

    def forward(self, loc_data, conf_data, prop_loc_data, prop_conf_data, center_data, priors, gt_segments, gt_labels):
        """
        :param predictions: a tuple containing loc, conf and priors
        :param targets: ground truth segments and labels
        :return: loc loss and conf loss
        """
        num_batch = loc_data.size(0)
        num_priors = priors.size(0)
        num_classes = self.num_classes
        clip_length = self.clip_len

        loss_l_list = []
        loss_c_list = []
        loss_ct_list = []
        loss_prop_l_list = []
        loss_prop_c_list = []

        for idx in range(num_batch):
            loc_t = torch.Tensor(num_priors, 2).to(loc_data.device)
            prop_loc_t = torch.Tensor(num_priors, 2).to(loc_data.device)

            loc_p = loc_data[idx]
            conf_p = conf_data[idx]
            prop_loc_p = prop_loc_data[idx]
            prop_conf_p = prop_conf_data[idx]
            center_p = center_data[idx]

            with torch.no_grad():
                # match priors and ground truth segments
                truths = gt_segments[idx]
                labels = gt_labels[idx].long()

                K = priors.size(0)
                N = truths.size(0)
                center = priors[:, 0].unsqueeze(1).expand(K, N)
                left = (center - truths[:, 0].unsqueeze(0).expand(K, N)) * clip_length
                right = (truths[:, 1].unsqueeze(0).expand(K, N) - center) * clip_length
                area = left + right
                maxn = clip_length * 2
                area[left < 0] = maxn
                area[right < 0] = maxn

                if self.loc_bounded:
                    max_dis = torch.max(left, right)
                    prior_lb, prior_rb = gen_bounds(priors)
                    l_bound = prior_lb.expand(K, N)
                    r_bound = prior_rb.expand(K, N)
                    area[max_dis <= l_bound] = maxn
                    area[max_dis > r_bound] = maxn

                best_truth_area, best_truth_idx = area.min(1)
                loc_t[:, 0] = (priors[:, 0] - truths[best_truth_idx, 0]) * clip_length
                loc_t[:, 1] = (truths[best_truth_idx, 1] - priors[:, 0]) * clip_length
                conf_t = labels[best_truth_idx]
                conf_t[best_truth_area >= maxn] = 0

                iou = iou_loss(loc_p, loc_t, loss_type="iou")  # [num_priors]
                prop_conf_t = conf_t.clone()

                if self.loc_bounded:
                    if (conf_t > 0).sum() > 0:
                        max_iou = iou[conf_t > 0].max(0)[0]
                    else:
                        max_iou = 2.0
                    prop_conf_t[iou < min(self.overlap_thresh, max_iou)] = 0
                else:
                    prop_conf_t[iou < self.overlap_thresh] = 0

                prop_w = loc_p[:, 0] + loc_p[:, 1]
                prop_loc_t[:, 0] = (loc_t[:, 0] - loc_p[:, 0]) / (0.5 * prop_w)
                prop_loc_t[:, 1] = (loc_t[:, 1] - loc_p[:, 1]) / (0.5 * prop_w)

            pos = conf_t > 0  # [num_priors]
            pos_idx = pos.unsqueeze(-1).expand_as(loc_p)  # [num_priors, 2]
            gt_loc_t = loc_t.clone()
            loc_p = loc_p[pos_idx].view(-1, 2)
            loc_target = loc_t[pos_idx].view(-1, 2)
            if loc_p.numel() > 0:
                loss_l = iou_loss(loc_p, loc_target, loss_type="giou", reduction="sum")
            else:
                loss_l = loc_p.sum()

            prop_pos = prop_conf_t > 0
            prop_pos_idx = prop_pos.unsqueeze(-1).expand_as(prop_loc_p)  # [num_priors, 2]
            prop_loc_p_pos = prop_loc_p[prop_pos_idx].view(-1, 2)
            prop_loc_t_pos = prop_loc_t[prop_pos_idx].view(-1, 2)

            if prop_loc_p_pos.numel() > 0:
                if self.use_smooth_l1:
                    loss_prop_l = F.smooth_l1_loss(prop_loc_p_pos, prop_loc_t_pos, reduction="sum")
                else:
                    loss_prop_l = F.l1_loss(prop_loc_p_pos, prop_loc_t_pos, reduction="sum")
            else:
                loss_prop_l = prop_loc_p_pos.sum()

            prop_pre_loc = loc_p
            cur_loc_t = gt_loc_t[pos_idx].view(-1, 2)
            prop_loc_p = prop_loc_p[pos_idx].view(-1, 2)
            center_p = center_p[pos.unsqueeze(-1)].view(-1)
            if prop_pre_loc.numel() > 0:
                prop_pre_w = (prop_pre_loc[:, 0] + prop_pre_loc[:, 1]).unsqueeze(-1)
                cur_loc_p = 0.5 * prop_pre_w * prop_loc_p + prop_pre_loc
                ious = iou_loss(cur_loc_p, cur_loc_t, loss_type="iou").clamp_(min=0)
                loss_ct = F.binary_cross_entropy_with_logits(center_p, ious, reduction="sum")
            else:
                loss_ct = prop_pre_loc.sum()

            # softmax focal loss
            conf_p = conf_p.view(-1, num_classes)
            targets_conf = conf_t.view(-1, 1)
            conf_p = F.softmax(conf_p, dim=1)
            loss_c = self.focal_loss(conf_p, targets_conf)

            prop_conf_p = prop_conf_p.view(-1, num_classes)
            prop_conf_p = F.softmax(prop_conf_p, dim=1)
            loss_prop_c = self.focal_loss(prop_conf_p, prop_conf_t)

            N = max(pos.sum(), 1)
            PN = max(prop_pos.sum(), 1)
            loss_l /= N
            loss_c /= N
            loss_prop_l /= PN
            loss_prop_c /= PN
            loss_ct /= N

            loss_l_list.append(loss_l)
            loss_c_list.append(loss_c)
            loss_prop_l_list.append(loss_prop_l)
            loss_prop_c_list.append(loss_prop_c)
            loss_ct_list.append(loss_ct)

        losses = dict(
            loss_loc=sum(loss_l_list) / num_batch * self.loc_weight,
            loss_cls=sum(loss_c_list) / num_batch * self.cls_weight,
            loss_iou=sum(loss_ct_list) / num_batch * self.cls_weight,
            loss_prop_loc=sum(loss_prop_l_list) / num_batch * self.loc_weight,
            loss_prop_cls=sum(loss_prop_c_list) / num_batch * self.cls_weight,
        )

        return losses


class FocalLoss_Ori(nn.Module):
    """
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(self, num_class, alpha=0.25, gamma=2, balance_index=-1, size_average=True):
        super(FocalLoss_Ori, self).__init__()
        self.num_class = num_class
        self.alpha = alpha
        self.gamma = gamma
        self.size_average = size_average
        self.eps = 1e-6

        if isinstance(self.alpha, (list, tuple)):
            assert len(self.alpha) == self.num_class
            self.alpha = torch.Tensor(list(self.alpha))
        elif isinstance(self.alpha, (float, int)):
            assert 0 < self.alpha < 1.0, "alpha should be in `(0,1)`)"
            assert balance_index > -1
            alpha = torch.ones((self.num_class))
            alpha *= 1 - self.alpha
            alpha[balance_index] = self.alpha
            self.alpha = alpha
        elif isinstance(self.alpha, torch.Tensor):
            self.alpha = self.alpha
        else:
            raise TypeError("Not support alpha type, expect `int|float|list|tuple|torch.Tensor`")

    def forward(self, logit, target):
        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.transpose(1, 2).contiguous()  # [N,C,d1*d2..] -> [N,d1*d2..,C]
            logit = logit.view(-1, logit.size(-1))  # [N,d1*d2..,C]-> [N*d1*d2..,C]
        target = target.view(-1, 1)  # [N,d1,d2,...]->[N*d1*d2*...,1]

        # ----------memory saving way--------
        pt = logit.gather(1, target).view(-1) + self.eps
        logpt = pt.log()

        if self.alpha.device != logpt.device:
            self.alpha = self.alpha.to(logpt.device)

        alpha_class = self.alpha.gather(0, target.view(-1))
        logpt = alpha_class * logpt
        loss = -1 * torch.pow(torch.sub(1.0, pt), self.gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.bce_loss = torch.nn.BCELoss(reduction="mean")

    def forward(self, start, end, scores):
        start = torch.tanh(start).mean(-1)
        end = torch.tanh(end).mean(-1)
        loss_start = self.bce_loss(start.view(-1), scores[:, 0].contiguous().view(-1).to(start.device))
        loss_end = self.bce_loss(end.view(-1), scores[:, 1].contiguous().view(-1).to(end.device))
        return loss_start, loss_end


def iou_loss(pred, target, weight=None, loss_type="giou", reduction="none"):
    """
    iou: A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    """
    pred_left = pred[:, 0]
    pred_right = pred[:, 1]
    target_left = target[:, 0]
    target_right = target[:, 1]

    pred_area = pred_left + pred_right
    target_area = target_left + target_right

    eps = torch.finfo(torch.float32).eps

    inter = torch.min(pred_left, target_left) + torch.min(pred_right, target_right)
    area_union = target_area + pred_area - inter
    ious = inter / area_union.clamp(min=eps)

    if loss_type == "linear_iou":
        loss = 1.0 - ious
    elif loss_type == "giou":
        ac_uion = torch.max(pred_left, target_left) + torch.max(pred_right, target_right)
        gious = ious - (ac_uion - area_union) / ac_uion.clamp(min=eps)
        loss = 1.0 - gious
    else:
        loss = ious

    if weight is not None:
        loss = loss * weight.view(loss.size())
    if reduction == "sum":
        loss = loss.sum()
    elif reduction == "mean":
        loss = loss.mean()
    return loss


def gen_bounds(priors):
    bounds = [[0, 30], [15, 60], [30, 120], [60, 240], [96, 768], [256, 768]]

    K = priors.size(0)
    prior_lb = priors[:, 1].clone()
    prior_rb = priors[:, 1].clone()
    for i in range(K):
        prior_lb[i] = bounds[int(prior_lb[i])][0]
        prior_rb[i] = bounds[int(prior_rb[i])][1]
    prior_lb = prior_lb.unsqueeze(1)
    prior_rb = prior_rb.unsqueeze(1)
    return prior_lb, prior_rb
