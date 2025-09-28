import math
import torch
import torch.nn as nn
from ..builder import HEADS, build_prior_generator, build_loss
from ..bricks import ConvModule
from ..utils.bbox_tools import compute_delta, delta_to_pred
from ..utils.post_processing import batched_nms


@HEADS.register_module()
class AnchorHead(nn.Module):
    def __init__(
        self,
        num_classes,
        in_channels,
        feat_channels,
        num_convs=3,
        prior_generator=None,
        loss=None,
        cls_prior_prob=0.01,
    ):
        super(AnchorHead, self).__init__()

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.num_convs = num_convs
        self.cls_prior_prob = cls_prior_prob

        self.scales = prior_generator.scales
        self.strides = prior_generator.strides

        # anchor generator
        self.prior_generator = build_prior_generator(prior_generator)

        # build layers
        self._init_layers()

        # loss
        self.assigner = build_loss(loss.assigner)
        self.sampler = build_loss(loss.sampler)
        self.cls_loss = build_loss(loss.cls_loss)
        self.reg_loss = build_loss(loss.reg_loss)

    def _init_layers(self):
        self.rpn_convs = nn.ModuleList([])
        for i in range(self.num_convs):
            self.rpn_convs.append(
                ConvModule(
                    self.in_channels if i == 0 else self.feat_channels,
                    self.feat_channels,
                    kernel_size=3,
                    padding=1,
                    act_cfg=dict(type="relu"),
                )
            )

        # regression
        self.rpn_reg = nn.Conv1d(self.feat_channels, len(self.scales) * 2, kernel_size=1)

        # classification (no sigmoid in layers)
        self.rpn_cls = nn.Conv1d(self.feat_channels, len(self.scales) * 1, kernel_size=1)

        # use prior in model initialization to improve stability
        # this will overwrite other weight init
        if self.cls_prior_prob > 0:
            bias_value = -(math.log((1 - self.cls_prior_prob) / self.cls_prior_prob))
            torch.nn.init.constant_(self.rpn_cls.bias, bias_value)

    def forward_train(self, feat_list, mask_list, gt_segments, gt_labels, **kwargs):
        cls_pred = []
        reg_pred = []

        for feat, mask in zip(feat_list, mask_list):
            for i in range(self.num_convs):
                feat, mask = self.rpn_convs[i](feat, mask)

            cls_pred.append(self.rpn_cls(feat) * mask.unsqueeze(1).float())  # todo
            reg_pred.append(self.rpn_reg(feat) * mask.unsqueeze(1).float())  # todo

        anchors = self.prior_generator(feat_list)  # List: [k,2] 0~1

        # loss
        losses = self.losses(cls_pred, reg_pred, mask_list, anchors, gt_segments, gt_labels)

        # get proposals
        proposals, _ = self._get_nms_proposal_list(anchors, mask_list, cls_pred, reg_pred)
        return losses, proposals

    def forward_test(self, feat_list, mask_list):
        cls_pred = []
        reg_pred = []

        for feat, mask in zip(feat_list, mask_list):
            for i in range(self.num_convs):
                feat, mask = self.rpn_convs[i](feat, mask)

            cls_pred.append(self.rpn_cls(feat) * mask.unsqueeze(1).float())
            reg_pred.append(self.rpn_reg(feat) * mask.unsqueeze(1).float())

        anchors = self.prior_generator(feat_list)  # List: [k,2] 0~1

        # get proposals
        proposals, scores = self._get_nms_proposal_list(anchors, mask_list, cls_pred, reg_pred)
        return proposals, scores

    def losses(self, cls_pred, reg_pred, mask_list, anchors, gt_segments, gt_labels):
        bs, num_scales = cls_pred[0].shape[:2]

        cls_pred = torch.cat(cls_pred, dim=-1).permute(0, 2, 1).reshape(bs, -1)  # [B,K]
        reg_pred = torch.cat(reg_pred, dim=-1).permute(0, 2, 1).reshape(bs, -1, 2)  # [B,K,2]
        masks = torch.cat(mask_list, dim=-1).unsqueeze(-1).repeat(1, 1, num_scales).reshape(bs, -1)  # [B,K]

        # get gt targets and positive negative mask
        gt_cls, gt_reg, pos_idxs_list, neg_idxs_list = self.prepare_targets(anchors, masks, gt_segments, gt_labels)

        num_pos = sum([len(pos_idxs) for pos_idxs in pos_idxs_list])
        num_neg = sum([len(neg_idxs) for neg_idxs in neg_idxs_list])

        # classification loss
        sampled_cls_pred = []
        for pred, mask, pos_idxs, neg_idxs in zip(cls_pred, masks, pos_idxs_list, neg_idxs_list):
            sampled_cls_pred.append(pred[mask][pos_idxs + neg_idxs])
        sampled_cls_pred = torch.cat(sampled_cls_pred, dim=0)

        loss_cls = self.cls_loss(sampled_cls_pred, gt_cls.float())
        loss_cls /= num_pos + num_neg

        # regression loss
        sampled_reg_pred = []
        for pred, mask, pos_idxs in zip(reg_pred, masks, pos_idxs_list):
            sampled_reg_pred.append(pred[mask][pos_idxs])
        sampled_reg_pred = torch.cat(sampled_reg_pred, dim=0)

        if num_pos == 0:  # not have positive sample
            # do not have positive samples in regression loss
            loss_reg = torch.Tensor([0]).sum().to(reg_pred.device)
        else:
            loss_reg = self.reg_loss(sampled_reg_pred, gt_reg)
            loss_reg /= num_pos

        losses = {"rpn_cls": loss_cls, "rpn_reg": loss_reg}
        return losses

    @torch.no_grad()
    def prepare_targets(self, anchors, masks, gt_segments, gt_labels):
        # prepare gts: assign the gt_segment to each anchor
        anchors = torch.cat(anchors, dim=0)  # [B,K,2]

        gt_cls_list, gt_reg_list, pos_idxs_list, neg_idxs_list = [], [], [], []
        for i, (gt_segment, gt_label) in enumerate(zip(gt_segments, gt_labels)):
            if len(gt_segment) == 0:  # make a pseudo gt_segment
                gt_segment = torch.tensor([[0, 0]], dtype=torch.float32, device=anchors.device)
                gt_label = torch.zeros(self.num_classes, device=anchors.device).to(torch.int64)
            else:
                gt_label = torch.ones((gt_segment.shape[0], 1), device=anchors.device).to(torch.int64)  # binary label

            # assign GT for valid positions
            _, assigned_gt_idxs, assigned_labels = self.assigner.assign(
                anchors[masks[i]],
                gt_segment,
                gt_label,
            )

            # sample positive and negative anchors
            pos_idxs, neg_idxs = self.sampler.sample(assigned_gt_idxs)
            pos_idxs_list.append(pos_idxs)
            neg_idxs_list.append(neg_idxs)

            # classification target: pos_mask + neg_mask
            gt_cls = assigned_labels[pos_idxs + neg_idxs].squeeze(-1)
            gt_cls_list.append(gt_cls)

            # regression target: pos_mask
            gt_reg = compute_delta(anchors[masks[i]][pos_idxs], gt_segment[assigned_gt_idxs[pos_idxs] - 1])
            gt_reg_list.append(gt_reg)

        gt_cls_concat = torch.cat(gt_cls_list, dim=0)  #  [B*(pos+neg)]
        gt_reg_concat = torch.cat(gt_reg_list, dim=0)  #  [B*(pos),2]
        return gt_cls_concat, gt_reg_concat, pos_idxs_list, neg_idxs_list

    @torch.no_grad()
    def _get_nms_proposal_list(self, anchors, mask_list, cls_pred, reg_pred):
        bs = cls_pred[0].shape[0]
        device = cls_pred[0].device
        pre_nms_topk = 2000
        post_nms_topk = 1000
        nms_thresh = 0.7

        # for each feature map, apply delta_to_pred() and select top-k anchors before nms
        topk_proposals, topk_scores, topk_masks, level_ids = [], [], [], []
        batch_idx = torch.arange(bs, device=device)
        for l, (anchor_i, logits_i, reg_i, mask_i) in enumerate(zip(anchors, cls_pred, reg_pred, mask_list)):
            # 1. get valid anchors
            mask_i = mask_i.unsqueeze(-1).repeat(1, 1, len(self.scales)).flatten(1)  # [bs,T*len(scales)]

            # 2. apply delta_to_pred() to get proposals
            reg_i = reg_i.permute(0, 2, 1).reshape(bs, -1, 2)
            scores_i = logits_i.permute(0, 2, 1).reshape(bs, -1).sigmoid()  # [bs, T*len(scales)]
            proposals_i = delta_to_pred(anchor_i, reg_i)  # [bs, T*len(scales), 2]

            # 3. select top-k anchor for each level and each video
            num_proposals_i = min(proposals_i.shape[1], pre_nms_topk)
            topk_scores_i, topk_idx = scores_i.topk(num_proposals_i, dim=1)
            topk_proposals_i = proposals_i[batch_idx[:, None], topk_idx]  # [bs,topk,2]
            topk_masks_i = mask_i[batch_idx[:, None], topk_idx]  # [bs,topk]

            topk_proposals.append(topk_proposals_i)
            topk_scores.append(topk_scores_i)
            topk_masks.append(topk_masks_i)
            level_ids.append(torch.full((num_proposals_i,), l, dtype=torch.int64, device=device))

        # concat all levels together
        topk_proposals = torch.cat(topk_proposals, dim=1)
        topk_scores = torch.cat(topk_scores, dim=1)
        topk_masks = torch.cat(topk_masks, dim=1)
        level_ids = torch.cat(level_ids, dim=0)  # we have recorded the level id

        # NMS on each level, and choose topk results.
        nms_proposals, nms_scores = [], []
        for i in range(bs):
            # select valid proposals
            valid = topk_masks[i]

            # NMS on each feature map
            new_proposals, new_scores, _ = batched_nms(
                topk_proposals[i][valid],
                topk_scores[i][valid],
                level_ids[valid],
                iou_threshold=nms_thresh,
                max_seg_num=post_nms_topk,
                use_soft_nms=False,
                multiclass=True,
            )

            nms_proposals.append(new_proposals.to(device))
            nms_scores.append(new_scores.to(device))
        return nms_proposals, nms_scores

    # @torch.no_grad()
    def _get_proposal_list(self, anchors, mask_list, cls_pred, reg_pred):
        bs = cls_pred[0].shape[0]

        reg_pred = torch.cat(reg_pred, dim=-1).permute(0, 2, 1).reshape(bs, -1, 2)  # [B,T*len(scales),2]
        cls_pred = torch.cat(cls_pred, dim=-1).permute(0, 2, 1).reshape(bs, -1)  # [B,T*len(scales)]

        anchors = torch.cat(anchors, dim=0).unsqueeze(0)  # [1,T*len(scales),2]
        proposals = delta_to_pred(anchors, reg_pred.detach())  # [B,K,2]
        masks = torch.cat(mask_list, dim=1).unsqueeze(-1).repeat(1, 1, len(self.scales))  # [B,T,len(scales)]

        new_proposals, new_scores = [], []
        for proposal, logits, mask in zip(proposals, cls_pred, masks):
            new_proposals.append(proposal[mask.view(-1)])
            new_scores.append(logits[mask.view(-1)].detach().sigmoid())
        return new_proposals, new_scores
