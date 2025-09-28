import torch
import torch.nn as nn
from ..builder import TRANSFORMERS, build_transformer, build_loss
from ..utils.bbox_tools import proposal_se_to_cw


@TRANSFORMERS.register_module()
class DETRTransformer(nn.Module):
    def __init__(
        self, num_queries, position_embedding=None, encoder=None, decoder=None, head=None, aux_loss=True, loss=None
    ):
        super().__init__()

        self.position_embedding = build_transformer(position_embedding)
        self.encoder = build_transformer(encoder)
        self.decoder = build_transformer(decoder)
        self.head = build_transformer(head)

        self.query_embed = nn.Embedding(num_queries, self.encoder.embed_dim)

        self.criterion = build_loss(loss)
        self.aux_loss = aux_loss

        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward_train(self, x, masks, gt_segments=None, gt_labels=None, is_training=True, **kwargs):
        # x: [bs, c, t]
        # masks: [bs, t], padding is 1.
        x = x.permute(0, 2, 1)  # [bs, c, t] -> [bs, t, c]
        pos_embed = self.position_embedding(masks)  # [bs, t, c]

        query_embed = self.query_embed.weight.unsqueeze(0).repeat(x.shape[0], 1, 1)
        # query_embed [bs, num_query, dim]
        memory = self.encoder(
            query=x,
            key=None,
            value=None,
            query_pos=pos_embed,
            query_key_padding_mask=masks,  # mask shape [bs, t]
        )  # [bs, t, c]
        target = torch.zeros_like(query_embed)
        decoder_output = self.decoder(
            query=target,
            key=memory,
            value=memory,
            key_pos=pos_embed,
            query_pos=query_embed,
            key_padding_mask=masks,
        )  # [num_layers, bs, num_query, c]

        outputs_class, outputs_coord = self.head(decoder_output)
        output = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]}

        if is_training:
            if self.aux_loss:
                output["aux_outputs"] = [
                    {"pred_logits": a, "pred_boxes": b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
                ]
            return self.losses(output, masks, gt_segments, gt_labels)
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
