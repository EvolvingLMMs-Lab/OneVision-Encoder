import torch
import torch.nn as nn
from .layers import MLP, inverse_sigmoid
from ..builder import TRANSFORMERS
from .deformable_detr_transformer import DeformableDETRTransformer
from ..utils.bbox_tools import proposal_se_to_cw
from ..roi_heads.roi_extractors.align1d.align import Align1DLayer


@TRANSFORMERS.register_module()
class TadTRTransformer(DeformableDETRTransformer):
    def __init__(
        self,
        num_proposals,
        num_classes,
        position_embedding=None,
        encoder=None,
        decoder=None,
        aux_loss=True,
        loss=None,
        with_act_reg=True,
        roi_size=16,
        roi_extend_ratio=0.25,
    ):
        super(TadTRTransformer, self).__init__(
            two_stage_num_proposals=num_proposals,
            num_classes=num_classes,
            position_embedding=position_embedding,
            encoder=encoder,
            decoder=decoder,
            aux_loss=aux_loss,
            loss=loss,
            with_box_refine=True,
            as_two_stage=False,
        )

        self.with_act_reg = with_act_reg
        if self.with_act_reg:  # RoI alignment
            hidden_dim = self.encoder.embed_dim
            self.roi_extend_ratio = roi_extend_ratio
            self.roi_extractor = Align1DLayer(roi_size)
            self.actionness_pred = nn.Sequential(
                nn.Linear(roi_size * hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid(),
            )

    @staticmethod
    def _to_roi_align_format(rois, T, roi_extend_ratio=1):
        """Convert RoIs to RoIAlign format.
        Params:
            RoIs: normalized segments coordinates, shape (batch_size, num_segments, 4)
            T: length of the video feature sequence
        """
        # transform to absolute axis
        B, N = rois.shape[:2]
        rois_center = rois[:, :, 0:1]
        rois_size = rois[:, :, 1:2] * (roi_extend_ratio * 2 + 1)
        rois_abs = torch.cat((rois_center - rois_size / 2, rois_center + rois_size / 2), dim=2) * T
        # expand the RoIs
        rois_abs = torch.clamp(rois_abs, min=0, max=T)  # (N, T, 2)
        # add batch index
        batch_ind = torch.arange(0, B).view((B, 1, 1)).to(rois_abs.device)
        batch_ind = batch_ind.repeat(1, N, 1)
        rois_abs = torch.cat((batch_ind.float(), rois_abs), dim=2)
        # NOTE: stop gradient here to stabilize training
        return rois_abs.view((B * N, 3)).detach()

    def forward_train(self, x, masks, gt_segments=None, gt_labels=None, is_training=True, **kwargs):
        # The input of TadTR's transformer is single scale feature
        # x: [bs, c, t], masks: [bs, t], padding is 1.

        # Here we set masks to be all False
        masks = torch.zeros_like(masks, dtype=torch.bool)

        feat = x.permute(0, 2, 1)  # [bs, c, t] -> [bs, t, c]
        pos_embed = self.position_embedding(masks) + self.level_embeds[0].view(1, 1, -1)  # [bs, t, c]

        lengths = torch.as_tensor([feat.shape[1]], dtype=torch.long, device=feat.device)
        level_start_index = lengths.new_zeros((1,))

        valid_ratios = self.get_valid_ratio(masks)[:, None]  # [bs, 1]
        reference_points = self.get_reference_points(lengths, valid_ratios, device=feat.device)  # [bs, t, 1]

        memory = self.encoder(
            query=feat,
            key=None,
            value=None,
            query_pos=pos_embed,
            query_key_padding_mask=masks,
            spatial_shapes=lengths,
            reference_points=reference_points,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            **kwargs,
        )

        bs, _, c = memory.shape
        query_pos, query = torch.split(self.query_embedding.weight, c, dim=1)
        query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
        query = query.unsqueeze(0).expand(bs, -1, -1)
        reference_points = self.reference_points(query_pos).sigmoid()  # override the reference points

        # decoder
        inter_states, inter_references_out = self.decoder(
            query=query,  # bs, num_queries, embed_dims
            key=memory,  # bs, num_tokens, embed_dims
            value=memory,  # bs, num_tokens, embed_dims
            query_pos=query_pos,
            key_padding_mask=masks,  # bs, num_tokens
            reference_points=reference_points,  # num_queries, 1
            spatial_shapes=lengths,  # nlvl
            level_start_index=level_start_index,  # nlvl
            valid_ratios=valid_ratios,
            **kwargs,
        )

        #  Calculate output coordinates and classes.
        outputs_classes = []
        outputs_coords = []
        for lvl in range(inter_states.shape[0]):
            reference = reference_points if lvl == 0 else inter_references_out[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](inter_states[lvl])
            tmp = self.bbox_embed[lvl](inter_states[lvl])
            if reference.shape[-1] == 2:
                tmp += reference
            else:
                assert reference.shape[-1] == 1
                tmp[..., 0] += reference.squeeze(-1)
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)  # tensor shape: [num_decoder_layers, bs, num_query, num_class]
        outputs_coord = torch.stack(outputs_coords)  # tensor shape: [num_decoder_layers, bs, num_query, 2]

        output = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]}

        if self.with_act_reg:
            # perform RoIAlign
            B, N = outputs_coord[-1].shape[:2]
            rois = self._to_roi_align_format(outputs_coord[-1], memory.shape[1], roi_extend_ratio=self.roi_extend_ratio)
            roi_features = self.roi_extractor(memory.permute(0, 2, 1), rois).view((B, N, -1))
            pred_actionness = self.actionness_pred(roi_features)
            output["pred_actionness"] = pred_actionness

        if is_training:
            if self.aux_loss:
                output["aux_outputs"] = [
                    {"pred_logits": a, "pred_boxes": b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
                ]
            return self.losses(output, masks, gt_segments, gt_labels)
        else:
            return output

    @torch.no_grad()
    def prepare_targets(self, masks, gt_segments, gt_labels):
        gt_segments = [proposal_se_to_cw(bboxes / masks.shape[-1]) for bboxes in gt_segments]  # normalize gt_segments
        gt_labels = [labels.long() for labels in gt_labels]
        return gt_segments, gt_labels
