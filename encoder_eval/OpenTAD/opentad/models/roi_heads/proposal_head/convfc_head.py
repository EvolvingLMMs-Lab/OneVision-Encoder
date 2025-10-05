import math
import torch
import torch.nn as nn
from ...builder import HEADS, build_loss
from ...utils.bbox_tools import delta_to_pred, compute_delta
from ...utils.misc import convert_gt_to_one_hot


@HEADS.register_module()
class ConvFCHead(nn.Module):
    r"""General proposal head, with shared conv and fc layers and two separated branches.
                                            /-> cls fcs
    roi feats -> shared convs -> shared fcs
                                            \-> reg fcs
    """

    def __init__(
        self,
        in_channels,
        roi_size,
        num_classes,
        shared_convs_num=0,  # shared convs
        shared_convs_channel=128,
        shared_fcs_num=1,  # shared fcs
        shared_fcs_channel=512,
        head_fcs_num=3,  # specific fcs for each feature
        head_fcs_channel=128,
        cls_prior_prob=0.01,
        # reg_class_agnostic=True,
        loss=None,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.roi_size = roi_size
        self.num_classes = num_classes
        self.cls_prior_prob = cls_prior_prob

        # shared layers setting
        self.shared_convs_num = shared_convs_num
        self.shared_convs_channel = shared_convs_channel
        self.shared_fcs_num = shared_fcs_num
        self.shared_fcs_channel = shared_fcs_channel

        # head layers setting
        self.head_fcs_num = head_fcs_num
        self.head_fcs_channel = head_fcs_channel

        # initialize layers
        self._init_layers()

        # loss
        self.multiply_with_iou = loss.multiply_with_iou if "multiply_with_iou" in loss else False
        self.assigner = build_loss(loss.assigner)
        self.sampler = build_loss(loss.sampler)
        self.cls_loss = build_loss(loss.cls_loss)
        self.reg_loss = build_loss(loss.reg_loss)

    def _init_layers(self):
        """Initialize layers of the head."""
        self._init_shared()
        self._init_heads()

    def _init_shared(self):
        """Initialize shared conv fc layers"""
        assert (self.shared_convs_num + self.shared_fcs_num) > 0
        self.shared_conv_fc = ConvFC(
            in_channels=self.in_channels,
            roi_size=self.roi_size,
            convs_num=self.shared_convs_num,
            convs_channel=self.shared_convs_channel,
            fcs_num=self.shared_fcs_num,
            fcs_channel=self.shared_fcs_channel,
        )

    def _init_heads(self):
        """Initialize classification head and regression head"""
        self.cls_head = FCHead(
            in_channels=self.shared_fcs_channel,
            out_channels=self.num_classes,
            fcs_num=self.head_fcs_num,
            fcs_channel=self.head_fcs_channel,
        )

        self.reg_head = FCHead(
            in_channels=self.shared_fcs_channel,
            out_channels=2,
            fcs_num=self.head_fcs_num,
            fcs_channel=self.head_fcs_channel,
        )

        # use prior in model initialization to improve stability
        # this will overwrite other weight init
        if self.cls_prior_prob > 0:
            bias_value = -(math.log((1 - self.cls_prior_prob) / self.cls_prior_prob))
            torch.nn.init.constant_(self.cls_head.head.bias, bias_value)

    def forward_train(self, proposal_feats, proposal_list, gt_segments, gt_labels):
        proposal_feats = self.shared_conv_fc(proposal_feats)  # [sum(K),C]
        cls_pred = self.cls_head(proposal_feats)  # [sum(K),num_classes], logits
        reg_pred = self.reg_head(proposal_feats)  # [sum(K),2]

        refined_proposals_list, _ = self.refine_proposals(proposal_list, reg_pred, cls_pred)

        loss = self.losses(cls_pred, reg_pred, proposal_list, gt_segments, gt_labels)
        return loss, refined_proposals_list

    def forward_test(self, proposal_feats, proposal_list):
        proposal_feats = self.shared_conv_fc(proposal_feats)  # [sum(K),C]
        cls_pred = self.cls_head(proposal_feats)  # [sum(K),num_classes], logits
        reg_pred = self.reg_head(proposal_feats)  # [sum(K),2]

        refined_proposals_list, proposal_score_list = self.refine_proposals(proposal_list, reg_pred, cls_pred)
        return refined_proposals_list, proposal_score_list

    @torch.no_grad()
    def prepare_targets(self, gt_segments, gt_labels, proposal_list):
        # convert gt to one hot encoding for multi class
        gt_segments, gt_labels = convert_gt_to_one_hot(gt_segments, gt_labels, self.num_classes)

        gt_cls_list, gt_reg_list, gt_iou_list, pos_idxs_list, neg_idxs_list = [], [], [], [], []
        for i, (gt_segment, gt_label, proposal) in enumerate(zip(gt_segments, gt_labels, proposal_list)):
            if len(gt_segment) == 0:  # make a pseudo gt_segment
                gt_segment = torch.tensor([[0, 0]], dtype=torch.float32, device=proposal.device)
                gt_label = torch.zeros(self.num_classes, device=proposal.device).to(torch.int64)

            # assign GT
            ious, assigned_gt_idxs, assigned_labels = self.assigner.assign(proposal, gt_segment, gt_label)

            # sample positive and negative anchors
            pos_idxs, neg_idxs = self.sampler.sample(assigned_gt_idxs)
            pos_idxs_list.append(pos_idxs)
            neg_idxs_list.append(neg_idxs)

            # classification target: pos_mask + neg_mask
            gt_cls = assigned_labels.float().clamp(min=0, max=1)[pos_idxs + neg_idxs]
            gt_iou = ious[pos_idxs + neg_idxs]
            gt_cls_list.append(gt_cls)
            gt_iou_list.append(gt_iou)

            # regression target: pos_mask
            gt_reg = compute_delta(proposal[pos_idxs], gt_segment[assigned_gt_idxs[pos_idxs] - 1])
            gt_reg_list.append(gt_reg)

        gt_cls_concat = torch.cat(gt_cls_list, dim=0)  #  [B*(pos+neg),num_classes]
        gt_reg_concat = torch.cat(gt_reg_list, dim=0)  #  [B*(pos),2]
        gt_iou_concat = torch.cat(gt_iou_list, dim=0)  #  [B*(pos+neg)]
        return gt_cls_concat, gt_reg_concat, gt_iou_concat, pos_idxs_list, neg_idxs_list

    def losses(self, cls_pred, reg_pred, proposal_list, gt_segments, gt_labels):
        # get gt targets and positive negative mask
        gt_cls, gt_reg, gt_iou, pos_idxs_list, neg_idxs_list = self.prepare_targets(
            gt_segments, gt_labels, proposal_list
        )

        num_pos = sum([len(pos_idxs) for pos_idxs in pos_idxs_list])
        num_neg = sum([len(neg_idxs) for neg_idxs in neg_idxs_list])

        # classification loss
        if self.multiply_with_iou:
            gt_cls *= gt_iou.unsqueeze(-1)

        sampled_cls_pred = []
        cls_pred = torch.split(cls_pred, [proposal.shape[0] for proposal in proposal_list], dim=0)
        for pred, pos_idxs, neg_idxs in zip(cls_pred, pos_idxs_list, neg_idxs_list):
            sampled_cls_pred.append(pred[pos_idxs + neg_idxs])
        sampled_cls_pred = torch.cat(sampled_cls_pred, dim=0)

        loss_cls = self.cls_loss(sampled_cls_pred, gt_cls.float())
        loss_cls /= num_pos + num_neg

        # regression loss
        sampled_reg_pred = []
        reg_pred = torch.split(reg_pred, [proposal.shape[0] for proposal in proposal_list], dim=0)
        for pred, pos_idxs in zip(reg_pred, pos_idxs_list):
            sampled_reg_pred.append(pred[pos_idxs])
        sampled_reg_pred = torch.cat(sampled_reg_pred, dim=0)

        if num_pos == 0:  # not have positive sample
            # do not have positive samples in regression loss
            loss_reg = torch.Tensor([0]).sum().to(sampled_reg_pred.device)
        else:
            loss_reg = self.reg_loss(sampled_reg_pred, gt_reg)
            loss_reg /= num_pos

        losses = {"cls_loss": loss_cls, "reg_loss": loss_reg}
        return losses

    @torch.no_grad()
    def refine_proposals(self, proposal_list, reg_pred, score_pred):
        new_proposal_list = []
        new_score_list = []
        cur = 0
        for proposal in proposal_list:
            N = proposal.shape[0]
            new_proposal_list.append(delta_to_pred(proposal, reg_pred[cur : cur + N]))
            new_score_list.append(score_pred[cur : cur + N].sigmoid())
            cur += N
        return new_proposal_list, new_score_list


class ConvFC(nn.Module):
    def __init__(self, in_channels=256, roi_size=32, convs_num=4, convs_channel=128, fcs_num=1, fcs_channel=512):
        super(ConvFC, self).__init__()

        self.convs_num = convs_num
        self.fcs_num = fcs_num
        last_layer_dim = in_channels

        # convs
        self.convs = nn.Sequential()
        if convs_num > 0:
            for i in range(convs_num):
                in_channel = last_layer_dim if i == 0 else convs_channel
                self.convs.append(
                    nn.Sequential(
                        nn.Conv1d(in_channel, convs_channel, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(inplace=True),
                    )
                )
            last_layer_dim = convs_channel

        # fc
        self.fcs = nn.Sequential()
        if fcs_num > 0:
            for i in range(fcs_num):
                in_channel = last_layer_dim * roi_size if i == 0 else fcs_channel
                self.fcs.append(
                    nn.Sequential(
                        nn.Linear(in_channel, fcs_channel),
                        nn.ReLU(inplace=True),
                    )
                )

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, roi_ft):
        # roi_ft [sum(K),C,res]
        if self.convs_num > 0:
            roi_ft = self.convs(roi_ft)  # [sum(K),C,res]

        if self.fcs_num > 0:
            roi_ft = roi_ft.flatten(1)
            roi_ft = self.fcs(roi_ft)  # [sum(K),C]
        return roi_ft


class FCHead(nn.Module):
    def __init__(self, in_channels, out_channels, fcs_num=3, fcs_channel=128):
        super(FCHead, self).__init__()
        assert fcs_num > 0

        self.fcs = nn.Sequential()
        for i in range(fcs_num):
            in_channel = in_channels if i == 0 else fcs_channel
            self.fcs.append(
                nn.Sequential(
                    nn.Linear(in_channel, fcs_channel),
                    nn.ReLU(inplace=True),
                )
            )
        self.head = nn.Linear(fcs_channel, out_channels)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fcs(x)
        x = self.head(x)
        return x
