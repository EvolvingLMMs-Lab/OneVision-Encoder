import torch
import torch.nn as nn

from ..builder import DETECTORS
from .bmn import BMN
from ..utils.post_processing import boundary_choose, batched_nms, convert_to_seconds


@DETECTORS.register_module()
class TSI(BMN):
    def __init__(
        self,
        projection,
        rpn_head,
        roi_head,
        neck=None,
        backbone=None,
    ):
        super().__init__(
            backbone=backbone,
            neck=neck,
            projection=projection,
            rpn_head=rpn_head,
            roi_head=roi_head,
        )

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d, nn.Conv2d, nn.Conv3d)):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def post_processing(self, predictions, metas, post_cfg, ext_cls, **kwargs):
        assert ext_cls != None  #  BMN is a proposal generation network

        score_type = getattr(post_cfg, "score_type", "iou")  # otherwise, "iou*s*e"
        proposal_post = getattr(post_cfg, "proposal", False)

        (pred_local_s, pred_local_e, pred_global_s, pred_global_e), pred_iou_map = predictions

        pred_iou_map = pred_iou_map.sigmoid()
        pred_s = torch.sqrt(pred_local_s.sigmoid() * pred_global_s.sigmoid())
        pred_e = torch.sqrt(pred_local_e.sigmoid() * pred_global_e.sigmoid())

        dscale = self.roi_head.proposal_generator.dscale
        tscale = self.roi_head.proposal_generator.tscale

        # here is a nice matrix implementation for post processing
        ds = torch.arange(0, dscale).to(pred_iou_map.device)
        ts = torch.arange(0, tscale).to(pred_iou_map.device)
        ds_mesh, ts_mesh = torch.meshgrid(ds, ts, indexing="ij")
        start_end_index = torch.stack((ts_mesh, ts_mesh + ds_mesh), dim=-1)
        valid_mask = start_end_index[:, :, 1] < tscale
        start_end_index = start_end_index.clamp(max=tscale - 1).float()

        start_mask = boundary_choose(pred_s)
        start_mask[:, 0] = True
        end_mask = boundary_choose(pred_e)
        end_mask[:, -1] = True
        start_end_map = start_mask.unsqueeze(2) * end_mask.unsqueeze(1)
        pred_iou_map = pred_iou_map[:, 0, :, :] * pred_iou_map[:, 1, :, :]

        results = {}
        for i in range(len(metas)):  # processing each video
            start_end_mask = start_end_map[i][
                start_end_index[:, :, 0].view(-1).long(),
                start_end_index[:, :, 1].view(-1).long(),
            ]
            start_end_mask = start_end_mask.reshape(dscale, tscale) * valid_mask

            segments_start = start_end_index[start_end_mask][:, 0]
            segments_end = start_end_index[start_end_mask][:, 1] + 1
            segments = torch.stack((segments_start, segments_end), dim=-1).detach().cpu()

            # score
            scores_iou = pred_iou_map[i][start_end_mask].detach().cpu()
            if score_type == "iou":
                scores = scores_iou
            elif score_type == "iou*s*e":
                score_start = pred_s[i, start_end_index[start_end_mask][:, 0].long()].detach().cpu()
                score_end = pred_e[i, start_end_index[start_end_mask][:, 1].long()].detach().cpu()
                scores = score_start * score_end * scores_iou
            else:
                raise f"score type should be iou or iou*s*e, but get {score_type}"

            # nms
            labels = torch.zeros_like(scores_iou)  # pseudo labels
            if proposal_post:
                segments = segments / tscale  # convert to 0~1, since recall need a different NMS
                segments, scores, labels = batched_nms(segments, scores, labels, **post_cfg.nms)
                segments = segments * tscale  # convert back
            else:
                segments, scores, labels = batched_nms(segments, scores, labels, **post_cfg.nms)

            video_id = metas[i]["video_name"]

            # convert segments to seconds
            segments = convert_to_seconds(segments, metas[i])

            # merge with external classifier
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
