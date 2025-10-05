import torch


def convert_gt_to_one_hot(gt_segments, gt_labels, num_classes):
    """convert the gt from class index to one hot encoding. this is for multi class case."""

    gt_segments_unique, gt_labels_onehot = [], []
    for gt_segment, gt_label in zip(gt_segments, gt_labels):
        if len(gt_segment) > 0:
            bbox_unique, inverse_indices = torch.unique(gt_segment, dim=0, return_inverse=True)
            label_unique = []
            for i in range(bbox_unique.shape[0]):
                label = torch.nn.functional.one_hot(
                    gt_label[inverse_indices == i].long(),
                    num_classes=num_classes,
                )
                label_unique.append(label.sum(dim=0).to(gt_label.device))
            label_unique = torch.stack(label_unique)
        else:
            bbox_unique, label_unique = [], []
        gt_segments_unique.append(bbox_unique)  # [K]
        gt_labels_onehot.append(label_unique)  # [K,num_classes]
    # gt_segments is the unique gt_segments
    # gt_labels is the one hot encoding for multi class
    return gt_segments_unique, gt_labels_onehot
