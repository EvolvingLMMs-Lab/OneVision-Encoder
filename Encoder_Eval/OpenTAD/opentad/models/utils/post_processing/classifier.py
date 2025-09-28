import json
import numpy as np
import torch
from mmengine.registry import Registry

CLASSIFIERS = Registry("models")


def build_classifier(cfg):
    """Build external classifier."""
    return CLASSIFIERS.build(cfg)


@CLASSIFIERS.register_module()
class CUHKANETClassifier:
    def __init__(self, path, topk=1):
        super().__init__()

        with open(path, "r") as f:
            cuhk_data = json.load(f)
        self.cuhk_data_score = cuhk_data["results"]
        self.cuhk_data_action = np.array(cuhk_data["class"])
        self.topk = topk

    def __call__(self, video_id, segments, scores):
        assert len(segments) == len(scores)

        # sort video classification
        cuhk_score = np.array(self.cuhk_data_score[video_id])
        cuhk_classes = self.cuhk_data_action[np.argsort(-cuhk_score)]
        cuhk_score = cuhk_score[np.argsort(-cuhk_score)]

        new_segments = []
        new_labels = []
        new_scores = []
        # for segment, score in zip(segments, scores):
        for k in range(self.topk):
            new_segments.append(segments)
            new_labels.extend([cuhk_classes[k]] * len(segments))
            new_scores.append(scores * cuhk_score[k])

        new_segments = torch.cat(new_segments)
        new_scores = torch.cat(new_scores)
        return new_segments, new_labels, new_scores


@CLASSIFIERS.register_module()
class UntrimmedNetTHUMOSClassifier:
    def __init__(self, path, topk=1):
        super().__init__()

        self.thumos_class = {
            7: "BaseballPitch",
            9: "BasketballDunk",
            12: "Billiards",
            21: "CleanAndJerk",
            22: "CliffDiving",
            23: "CricketBowling",
            24: "CricketShot",
            26: "Diving",
            31: "FrisbeeCatch",
            33: "GolfSwing",
            36: "HammerThrow",
            40: "HighJump",
            45: "JavelinThrow",
            51: "LongJump",
            68: "PoleVault",
            79: "Shotput",
            85: "SoccerPenalty",
            92: "TennisSwing",
            93: "ThrowDiscus",
            97: "VolleyballSpiking",
        }

        self.cls_data = np.load(path)
        self.thu_label_id = np.array(list(self.thumos_class.keys())) - 1  # get thumos class id
        self.topk = topk

    def __call__(self, video_id, segments, scores):
        assert len(segments) == len(scores)

        # sort video classification
        video_cls = self.cls_data[int(video_id[-4:]) - 1][self.thu_label_id]  # order by video list, output 20
        video_cls_rank = sorted((e, i) for i, e in enumerate(video_cls))
        unet_classes = [self.thu_label_id[video_cls_rank[-k - 1][1]] + 1 for k in range(self.topk)]
        unet_scores = [video_cls_rank[-k - 1][0] for k in range(self.topk)]

        new_segments = []
        new_labels = []
        new_scores = []
        # for segment, score in zip(segments, scores):
        for k in range(self.topk):
            new_segments.append(segments)
            new_labels.extend([self.thumos_class[int(unet_classes[k])]] * len(segments))
            new_scores.append(scores * unet_scores[k])

        new_segments = torch.cat(new_segments)
        new_scores = torch.cat(new_scores)
        return new_segments, new_labels, new_scores


@CLASSIFIERS.register_module()
class TCANetHACSClassifier:
    def __init__(self, path, topk=1):
        super().__init__()

        with open(path, "r") as f:
            cls_data = json.load(f)
        self.cls_data_score = cls_data["results"]
        self.cls_data_action = cls_data["class"]
        self.topk = topk

    def __call__(self, video_id, segments, scores):
        assert len(segments) == len(scores)

        # sort video classification
        cls_score = np.array(self.cls_data_score[video_id][0])
        cls_score = np.exp(cls_score) / np.sum(np.exp(cls_score)) * 2.0
        cls_data_action = np.array(self.cls_data_action)
        cls_classes = cls_data_action[np.argsort(-cls_score)]
        cls_score = cls_score[np.argsort(-cls_score)]

        new_segments = []
        new_labels = []
        new_scores = []

        for k in range(self.topk):
            new_segments.append(segments)
            new_labels.extend([cls_classes[k]] * len(segments))
            new_scores.append(scores * cls_score[k])

        new_segments = torch.cat(new_segments)
        new_scores = torch.cat(new_scores)
        return new_segments, new_labels, new_scores


@CLASSIFIERS.register_module()
class StandardClassifier:
    def __init__(self, path, topk=1, apply_softmax=False):
        super().__init__()

        with open(path, "r") as f:
            cls_data = json.load(f)
        self.cls_data_score = cls_data["results"]
        self.cls_data_label = np.array(cls_data["class"]) if "class" in cls_data else np.array(cls_data["classes"])
        self.apply_softmax = apply_softmax
        self.topk = topk

    def __call__(self, video_id, segments, scores):
        assert len(segments) == len(scores)
        cls_score = np.array(self.cls_data_score[video_id])

        if self.apply_softmax:  # do softmax
            cls_score = np.exp(cls_score) / np.sum(np.exp(cls_score))

        # sort video classification scores
        topk_cls_idx = np.argsort(cls_score)[::-1][: self.topk]
        topk_cls_score = cls_score[topk_cls_idx]
        topk_cls_label = self.cls_data_label[topk_cls_idx]

        new_segments = []
        new_labels = []
        new_scores = []

        for k in range(self.topk):
            new_segments.append(segments)
            new_labels.extend([topk_cls_label[k]] * len(segments))
            new_scores.append(np.sqrt(scores * topk_cls_score[k]))  # default is sqrt

        new_segments = torch.cat(new_segments)
        new_scores = torch.cat(new_scores)
        return new_segments, new_labels, new_scores


@CLASSIFIERS.register_module()
class PseudoClassifier:
    def __init__(self, pseudo_label=""):
        super().__init__()

        self.pseudo_label = pseudo_label

    def __call__(self, video_id, segments, scores):
        assert len(segments) == len(scores)

        labels = [self.pseudo_label for _ in range(len(segments))]

        return segments, labels, scores
