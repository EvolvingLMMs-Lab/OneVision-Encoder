import numpy as np
np.bool = np.bool_  # 解决numpy 1.24版本的bool类型问题

import os
if "LINEAR_PROBE_ROOT" not in os.environ:
    os.environ["LINEAR_PROBE_ROOT"] = "/eval_data/linear_probe_image/"

import argparse
import importlib
import os
import random

import clip
import numpy as np
import torch
from torch.utils.data import DataLoader

from linear_probe import search
from model_factory import MODEL_REGISTRY

# 基于图片中显示的文件列表，定义支持的数据集
SUPPORTED_DATASETS = [
    "birdsnap","caltech101", "cifar10", "cifar100", "clevr", "coco", 
    "country211", "dtd", "eurosat", "fer2013", "fgvc_aircraft", 
    "flickr30k", "flowers", "food101", "gtsrb", "hateful_memes_raw", 
    "hateful_memes", "imagenet", "kinetics700", "kitti", "mnist", 
    "patchcamelyon", "pets", "resisc45", "sst2", "stanford_car", 
    "stl10", "sun397", "ucf101", "voc2007"
]

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="ucf101")
parser.add_argument("--model", default="CLIP-ViT-B/32")
parser.add_argument("--workers", type=int, default=8)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--debug", type=int, default=0)
parser.add_argument("--input_size", type=int, default=224,)
args = parser.parse_args()


def setup_seed(seed, cuda_deterministic=True):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if cuda_deterministic:  # slower, more reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:  # faster, less reproducible
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def get_dataset_loaders(dataset_name, transform):
    """从dataloader_rec目录加载数据集，并获取相关参数"""
    
    # 断言检查请求的数据集是否支持
    assert dataset_name in SUPPORTED_DATASETS, f"Dataset '{dataset_name}' is not supported. Supported datasets: {SUPPORTED_DATASETS}"
    
    # 从dataloader_rec目录导入对应模块
    try:
        module = importlib.import_module(f"dataloader_rec.{dataset_name}")
        
        # 获取classes (必需)
        if hasattr(module, "num_classes"):
            classes = module.num_classes
        else:
            raise AttributeError(f"Module 'dataloader_rec.{dataset_name}' does not define 'num_classes'")
        
        # 获取metric (可选，默认为"acc")
        metric = getattr(module, "metric", "acc")
        
        # 获取数据集
        dataset_train = module.get_loader_train(transform, None, None, 2333)[0]
        dataset_test = module.get_loader_test(transform, None, None, 2333)[0]
        
        return dataset_train, dataset_test, classes, metric
        
    except (ImportError, AttributeError) as e:
        raise Exception(f"Failed to load dataset '{dataset_name}' from dataloader_rec: {e}")


@torch.no_grad()
def get_feat(dataset, model, workers=4):
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=workers)
    cache_x = None
    cache_y = None

    idx = 0
    for images, labels in dataloader:
        images = images.cuda()

        with torch.cuda.amp.autocast():
            if hasattr(model, "forward_wo_proj"):
                feat = model.forward_wo_proj(images)
            elif hasattr(model, "encode_image"):
                feat = model.encode_image(images)
            else:
                feat = model(images)
        if hasattr(feat, "pooler_output"):
            feat = feat.pooler_output

        feat = feat.float()

        if cache_x is None:
            cache_x = torch.zeros(len(dataset), feat.size(1), dtype=feat.dtype).cuda()
            if len(labels.size()) > 1:
                cache_y = torch.zeros(
                    len(dataset), labels.size(1), dtype=labels.dtype
                ).cuda()
            else:
                cache_y = torch.zeros(len(dataset), dtype=labels.dtype).cuda()

        cache_x[idx : idx + feat.size(0)] = feat
        cache_y[idx : idx + feat.size(0)] = labels
        idx += feat.size(0)
    return cache_x, cache_y


if __name__ == "__main__":
    setup_seed(2048)

    # 加载模型和transform
    if args.model[:4] == "CLIP":
        model_name = args.model.split("CLIP-")[-1]
        if len(args.model.split(",")) == 1:
            model, transform = clip.load(args.model.split("CLIP-")[-1], "cpu")
            model = model.cuda().eval()
        else:
            model, transform = clip.load(model_name.split(",")[0], "cpu")
            model = model.visual
            path_weight = model_name.split(",")[-1]
            state_dict = torch.load(path_weight, "cpu")
            state_dict = {
                k.replace("_orig_mod.", ""): v for k, v in state_dict.items()
            }
            model.load_state_dict(state_dict, strict=True)
            model = model.cuda().eval()
    elif args.model[:8] == "OPENCLIP":
        import open_clip

        result = args.model.split("OPENCLIP-")[1].split(",")
        if len(result) == 2:
            name, pretrained = result
            model, _, transform = open_clip.create_model_and_transforms(
                name, pretrained
            )
        else:
            model, _, transform = open_clip.create_model_and_transforms(result[0])
        model = model.cuda().eval()
    else:
        name, weight = args.model.split(",")
        model = MODEL_REGISTRY.get(name)()
        state_dict = torch.load(weight, "cpu")
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=True)

        # 从CLIP加载 transform
        from clip.clip import _transform
        transform = _transform(args.input_size)

    model = model.cuda().eval()
    # 使用统一的数据集加载函数，同时获取classes和metric
    dataset_train, dataset_test, classes, metric = get_dataset_loaders(args.dataset, transform)
    
    # 计算特征
    x_train, y_train = get_feat(dataset_train, model, args.workers)
    x_test, y_test = get_feat(dataset_test, model, args.workers)

    y_train = y_train.contiguous()
    y_test = y_test.contiguous()

    # check feat
    sum_x_train = torch.sum(x_train, dim=1)
    sum_x_test = torch.sum(x_test, dim=1)
    if torch.sum(sum_x_train == 0) > 1:
        print(0, end=",")
        exit()

    assert torch.sum(y_train > classes) == 0
    assert torch.sum(y_test > classes) == 0

    result, _ = search(x_train, y_train, x_test, y_test, args.debug, classes, metric)
    print(result, end=",")
