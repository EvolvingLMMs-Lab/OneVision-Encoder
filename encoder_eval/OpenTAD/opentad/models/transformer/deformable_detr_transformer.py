import math
import copy
import torch
import torch.nn as nn
from .layers import MultiScaleDeformableAttention, MLP, inverse_sigmoid, get_sine_pos_embed
from ..builder import TRANSFORMERS, build_transformer, build_loss
from ..utils.bbox_tools import proposal_se_to_cw


@TRANSFORMERS.register_module()
class DeformableDETRTransformer(nn.Module):
    def __init__(
        self,
        two_stage_num_proposals,
        num_classes,
        position_embedding=None,
        encoder=None,
        decoder=None,
        aux_loss=True,
        loss=None,
        with_box_refine=False,
        as_two_stage=False,
    ):
        super().__init__()

        self.position_embedding = build_transformer(position_embedding)
        self.encoder = build_transformer(encoder)
        self.decoder = build_transformer(decoder)
        self.level_embeds = nn.Parameter(torch.Tensor(self.encoder.num_feature_levels, self.encoder.embed_dim))

        self.as_two_stage = as_two_stage
        self.with_box_refine = with_box_refine
        self.two_stage_num_proposals = two_stage_num_proposals
        self.num_classes = num_classes

        if self.as_two_stage:
            self.enc_output = nn.Linear(self.encoder.embed_dim, self.encoder.embed_dim)
            self.enc_output_norm = nn.LayerNorm(self.encoder.embed_dim)
            self.pos_trans = nn.Linear(self.encoder.embed_dim, self.encoder.embed_dim * 2)
            self.pos_trans_norm = nn.LayerNorm(self.encoder.embed_dim * 2)
        else:
            self.reference_points = nn.Linear(self.encoder.embed_dim, 1)
            self.query_embedding = nn.Embedding(two_stage_num_proposals, self.encoder.embed_dim * 2)

        self.init_weights()

        # define classification head and box head
        self.class_embed = nn.Linear(self.decoder.embed_dim, num_classes)
        self.bbox_embed = MLP(self.decoder.embed_dim, self.decoder.embed_dim, 2, num_layers=3)

        # init parameters for heads
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)

        # If two-stage, the last class_embed and bbox_embed is for region proposal generation
        # Decoder layers share the same heads without box refinement, while use the different
        # heads when box refinement is used.
        num_pred = self.decoder.num_layers + 1 if as_two_stage else self.decoder.num_layers
        if self.with_box_refine:
            self.class_embed = nn.ModuleList([copy.deepcopy(self.class_embed) for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([copy.deepcopy(self.bbox_embed) for _ in range(num_pred)])
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[1], -2.0)
            self.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[1], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.decoder.bbox_embed = None

        # hack implementation for two-stage. The last class_embed and bbox_embed is for region proposal generation
        if self.as_two_stage:
            self.decoder.class_embed = self.class_embed
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[1], 0.0)

        # loss function
        self.criterion = build_loss(loss)
        self.aux_loss = aux_loss

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention):
                m.init_weights()
        if not self.as_two_stage:
            nn.init.xavier_normal_(self.reference_points.weight.data, gain=1.0)
            nn.init.constant_(self.reference_points.bias.data, 0.0)
        nn.init.normal_(self.level_embeds)

    def get_valid_ratio(self, mask):
        """Get the valid ratios of feature maps of all levels."""
        _, T = mask.shape
        valid_T = torch.sum(~mask, 1)
        valid_ratio = valid_T.float() / T
        return valid_ratio  # [0~1]

    @staticmethod
    def get_reference_points(lengths, valid_ratios, device):
        """Get the reference points used in decoder.

        Args:
            lengths (Tensor): The shape of all feature maps, has shape (num_level).
            device (obj:`device`): The device where reference_points should be.

        Returns:
            Tensor: reference points used in decoder, has shape (bs, num_keys, num_levels).
        """
        reference_points_list = []
        for lvl, T in enumerate(lengths):
            ref = torch.linspace(0.5, T - 0.5, T, dtype=torch.float32, device=device)
            ref = ref[None, ...] / (valid_ratios[:, None, lvl] * T)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)  # [B,T]
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]  # [B,T,num_level]
        return reference_points

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, lengths, base_width=0.05):
        proposals = []
        _cur = 0
        for lvl, T in enumerate(lengths):
            valid_T = torch.sum(~memory_padding_mask[:, _cur : (_cur + T)], 1)  # [B]
            grid = torch.linspace(0.5, T - 0.5, T, dtype=torch.float32, device=memory.device)
            grid = grid.view(1, T).repeat(memory.shape[0], 1) / valid_T.unsqueeze(-1)  # [B,N_]
            width = torch.ones_like(grid) * base_width * (2.0**lvl)
            proposal = torch.stack((grid, width), -1)  # [B,N_,2]
            proposals.append(proposal)
            _cur += T

        output_proposals = torch.cat(proposals, 1)  # [B,N,2]
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))  # unsigmoid
        output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float("inf"))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float("inf"))

        output_memory = memory
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals

    def forward_train(self, feat_list, mask_list, gt_segments=None, gt_labels=None, is_training=True, **kwargs):
        # x: [bs, c, t]
        # masks: [bs, t], padding is 1.

        # generate positional embeddings for each level
        pos_embed = []
        for i, mask in enumerate(mask_list):
            pos_embed.append(self.position_embedding(mask) + self.level_embeds[i].view(1, 1, -1))

        feat_flatten = torch.cat(feat_list, dim=-1).permute(0, 2, 1)  # [bs, c, t] -> [bs, t, c]
        mask_flatten = torch.cat(mask_list, dim=1)  # [bs, t]
        pos_embed_flatten = torch.cat(pos_embed, dim=1)  # [bs, t, c]
        lengths = torch.as_tensor([feat.shape[-1] for feat in feat_list], dtype=torch.long, device=feat_flatten.device)
        level_start_index = torch.cat((lengths.new_zeros((1,)), lengths.cumsum(0)[:-1]))

        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in mask_list], 1)
        reference_points = self.get_reference_points(lengths, valid_ratios, device=feat_flatten.device)

        memory = self.encoder(
            query=feat_flatten,
            key=None,
            value=None,
            query_pos=pos_embed_flatten,
            query_key_padding_mask=mask_flatten,
            spatial_shapes=lengths,
            reference_points=reference_points,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            **kwargs,
        )

        bs, _, c = memory.shape
        if self.as_two_stage:
            output_memory, output_proposals = self.gen_encoder_output_proposals(memory, mask_flatten, lengths)
            enc_outputs_class = self.class_embed[-1](output_memory)
            enc_outputs_coord = (self.bbox_embed[-1](output_memory) + output_proposals).sigmoid()
            topk_proposals = torch.topk(enc_outputs_class[..., 0], self.two_stage_num_proposals, dim=1)[1]

            reference_points = torch.gather(enc_outputs_coord, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 2)).detach()
            pos_trans_out = self.pos_trans_norm(self.pos_trans(get_sine_pos_embed(reference_points)))
            query_pos, query = torch.split(pos_trans_out, c, dim=2)
        else:
            query_pos, query = torch.split(self.query_embedding.weight, c, dim=1)
            query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
            query = query.unsqueeze(0).expand(bs, -1, -1)
            reference_points = self.reference_points(query_pos).sigmoid()

        # decoder
        inter_states, inter_references_out = self.decoder(
            query=query,  # bs, num_queries, embed_dims
            key=memory,  # bs, num_tokens, embed_dims
            value=memory,  # bs, num_tokens, embed_dims
            query_pos=query_pos,
            key_padding_mask=mask_flatten,  # bs, num_tokens
            reference_points=reference_points,  # num_queries, 2
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

        if self.as_two_stage:
            output["enc_outputs"] = {"pred_logits": enc_outputs_class, "pred_boxes": enc_outputs_coord}

        if is_training:
            if self.aux_loss:
                output["aux_outputs"] = [
                    {"pred_logits": a, "pred_boxes": b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
                ]
            return self.losses(output, mask_list[0], gt_segments, gt_labels)
        else:
            return output

    def forward_test(self, x, masks, **kwargs):
        return self.forward_train(x, masks, is_training=False, **kwargs)

    def losses(self, outputs, masks, gt_segments, gt_labels):
        gt_segments, gt_labels = self.prepare_targets(masks, gt_segments, gt_labels)
        loss_dict = self.criterion(outputs, gt_segments, gt_labels)
        return loss_dict

    @torch.no_grad()
    def prepare_targets(self, masks, gt_segments, gt_labels):
        gt_segments = [
            bboxes / (~mask).float().sum() for bboxes, mask in zip(gt_segments, masks)
        ]  # normalize gt_segments
        gt_segments = [proposal_se_to_cw(bboxes) for bboxes in gt_segments]
        gt_labels = [labels.long() for labels in gt_labels]
        return gt_segments, gt_labels
