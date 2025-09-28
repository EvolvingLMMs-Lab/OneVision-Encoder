import torch
import math


def compute_delta(bboxes_init, gt_segments, wc=2.0, wl=2.0):
    assert bboxes_init.size(0) == gt_segments.size(0)
    assert bboxes_init.size(-1) == gt_segments.size(-1) == 2

    # wc, wl = (10.0, 5.0)
    # wc, wl = (1.0, 1.0)

    init_c = (bboxes_init[..., 0] + bboxes_init[..., 1]) * 0.5
    init_w = bboxes_init[..., 1] - bboxes_init[..., 0]

    gt_c = (gt_segments[..., 0] + gt_segments[..., 1]) * 0.5
    gt_w = gt_segments[..., 1] - gt_segments[..., 0]

    dc = (gt_c - init_c) / init_w * wc
    dw = torch.log(gt_w / init_w) * wl

    deltas = torch.stack([dc, dw], dim=-1)
    return deltas


def delta_to_pred(bboxes_init, pred_delta, wc=2.0, wl=2.0):
    assert pred_delta.shape[-2] == bboxes_init.shape[-2]

    # wc, wl = (10.0, 5.0)
    # wc, wl = (1.0, 1.0)

    dc = pred_delta[..., 0] / wc
    # Prevent sending too large values into torch.exp()
    dw = torch.clamp(pred_delta[..., 1] / wl, max=math.log(1000.0 / 16))

    init_c = (bboxes_init[..., 0] + bboxes_init[..., 1]) * 0.5
    init_w = bboxes_init[..., 1] - bboxes_init[..., 0]

    pred_c = dc * init_w + init_c
    pred_w = torch.exp(dw) * init_w

    pred_bboxes = torch.stack([pred_c - 0.5 * pred_w, pred_c + 0.5 * pred_w], dim=-1)

    pred_bboxes = pred_bboxes.clamp(min=0)
    return pred_bboxes


def proposal_cw_to_se(x):
    c, w = x.unbind(-1)
    s = c - 0.5 * w
    e = c + 0.5 * w
    return torch.stack([s, e], dim=-1)


def proposal_se_to_cw(x):
    s, e = x.unbind(-1)
    c = (s + e) * 0.5
    w = e - s
    return torch.stack([c, w], dim=-1)
