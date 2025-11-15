# CUDA_VISIBLE_DEVICES=0 python kmeans_gemm.py --input /train_tmp/list_train_0000 --class_center /train_tmp/center_2000000_with_in1k_v2.npy
# CUDA_VISIBLE_DEVICES=1 python kmeans_gemm.py --input /train_tmp/list_train_0001 --class_center /train_tmp/center_2000000_with_in1k_v2.npy
import argparse
import os

import numpy as np
import torch
from torch import distributed
from torch.nn.functional import normalize

local_rank = int(os.getenv("LOCAL_RANK", "0"))
world_size = int(os.getenv("WORLD_SIZE", "1"))
# distributed.init_process_group(backend="nccl")
torch.cuda.set_device(local_rank)
rank = int(os.getenv("RANK", "0"))
torch.backends.cudnn.benchmark = True


def split_with_rank_worldsize(list_like, rank, world_size):
    num_local: int = len(list_like) // world_size + int(rank < len(list_like) % world_size)
    class_start: int = len(list_like) // world_size * rank + min(rank, len(list_like) % world_size)
    return num_local, class_start


def collect_input_paths(input_path: str):
    """
    收集待处理的 .npy 文件路径：
    - 若为目录：收集目录下所有 .npy（不递归），按文件名排序
    - 若为文件：视为列表文件，逐行读取 .npy 路径（相对路径以列表文件所在目录为基准），过滤不存在的路径，整体排序
    """
    if os.path.isdir(input_path):
        files = [
            os.path.join(input_path, fn)
            for fn in os.listdir(input_path)
            if fn.lower().endswith(".npy") and os.path.isfile(os.path.join(input_path, fn))
        ]
        return sorted(files)

    if os.path.isfile(input_path):
        # 视为列表文件
        base_dir = os.path.dirname(os.path.abspath(input_path))
        paths = []
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                p = line.strip()
                if not p:
                    continue
                if not os.path.isabs(p):
                    p = os.path.normpath(os.path.join(base_dir, p))
                if os.path.isfile(p) and p.lower().endswith(".npy"):
                    paths.append(p)
        return sorted(paths)

    raise FileNotFoundError(f"输入路径无效：{input_path}")


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", help="可以是目录（含 .npy）或包含 .npy 路径的列表文件")
    parser.add_argument("--class_center", "-c", required=True, help="class center (.npy)")
    parser.add_argument("--drop_last", type=int, default=0)
    parser.add_argument("--topk", type=int, default=10)
    args = parser.parse_args()

    # 收集特征文件路径并按 rank 切片
    lines = collect_input_paths(args.input)
    if len(lines) == 0:
        raise FileNotFoundError(f"未在输入中找到任何 .npy 文件：{args.input}")

    num_local, start = split_with_rank_worldsize(lines, rank, world_size)
    lines = lines[start: start + num_local]

    center = np.load(args.class_center)
    center = torch.from_numpy(center).cuda().float()
    center = normalize(center)

    for feat_path in lines:
        feat = np.load(feat_path)
        feat = torch.from_numpy(feat).cuda().float()

        if args.drop_last:
            feat = feat[:, :-1]

        feat = normalize(feat)
        start = 0
        batch_size = 128

        # cached_score = torch.zeros(feat.size(0)).cuda()
        cached_label = torch.zeros(feat.size(0), args.topk).cuda().long()

        while start < feat.size(0):
            end = min(start + batch_size, feat.size(0))
            bs_score = torch.einsum("ik, jk -> ij", feat[start: end], center)
            score, index = torch.topk(bs_score, k=args.topk, dim=1)
            # cached_score[start: end] = score.reshape(-1)
            cached_label[start: end] = index

            start += batch_size
        # np.save(f"{feat_path}_score_{os.path.basename(args.class_center)}".replace(".npy", ""), cached_score.cpu().numpy())
        np.save(f"{feat_path}_label_{os.path.basename(args.class_center)}".replace(".npy", ""), cached_label.cpu().numpy())


if __name__ == "__main__":
    main()
