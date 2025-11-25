#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#! conda create -n faiss python=3.9
#! conda install -c pytorch -c conda-forge faiss-gpu=1.7.3 cudatoolkit=11.8 numpy scipy

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import print_function

import argparse
import os
import sys
import time

import faiss
import numpy as np


def iter_npy_paths(src):
    """
    生成器：从目录、列表文件或单个 .npy 路径中产出待加载的 .npy 文件路径。
    - 目录：收集该目录下所有 .npy（不递归），按文件名排序
    - 列表文件：逐行读取，相对路径相对于列表文件所在目录，忽略空行与不存在的路径
    - 单个 .npy 文件：直接返回该路径
    """
    if os.path.isdir(src):
        for fn in sorted(os.listdir(src)):
            if fn.lower().endswith(".npy"):
                yield os.path.join(src, fn)
        return

    if os.path.isfile(src):
        if src.lower().endswith(".npy"):
            yield src
            return

        # 视为列表文件
        base = os.path.dirname(os.path.abspath(src))
        collected = []
        with open(src, "r", encoding="utf-8") as f:
            for line in f:
                p = line.strip()
                if not p or p.startswith("#"):
                    continue
                if not os.path.isabs(p):
                    p = os.path.normpath(os.path.join(base, p))
                if os.path.isfile(p) and p.lower().endswith(".npy"):
                    collected.append(p)
                else:
                    print(f"[warn] 列表中路径不存在或不是 .npy，已忽略: {p}", file=sys.stderr)
        for p in sorted(collected):
            yield p
        return

    print(f"[error] --input 路径无效：{src}", file=sys.stderr)


def load_and_concat(paths, drop_last=False):
    """
    加载多个 .npy 并在样本维拼接。
    - 数组会 reshape 为 [num_samples, -1] 保证拼接一致
    - drop_last=True 时，会对每个数组执行 arr = arr[:, :-1]
    - 统一对文件路径排序，保证稳定顺序
    """
    paths = sorted(list(paths))
    if not paths:
        raise FileNotFoundError("未收集到任何 .npy 文件")

    arrays = []
    for idx, p in enumerate(paths):
        print(f"[load] ({idx+1}/{len(paths)}) {p}")
        arr = np.load(p)
        if arr.ndim == 0:
            raise ValueError(f"文件 {p} 加载为标量，无法使用")
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        else:
            # 将后续维度展平，使其成为二维 [num_samples, feat_dim]
            arr = arr.reshape(arr.shape[0], -1)

        if drop_last and arr.shape[1] > 0:
            arr = arr[:, :-1]

        arrays.append(arr.astype(np.float32, copy=False))

    if len(arrays) == 1:
        return arrays[0]
    return np.concatenate(arrays, axis=0)


def read_feat(input_path, drop_last=False):
    """
    读取特征入口：支持目录、列表文件或单个 .npy 文件。
    不做任何归一化处理（归一化由主流程根据 args 控制）。
    """
    paths = iter_npy_paths(input_path)
    x = load_and_concat(paths, drop_last=bool(drop_last))
    return x


def l2_row_normalize(a, eps=1e-12):
    """
    行向量 L2 归一化，返回新的数组（float32）。
    防止除零，使用 eps。
    """
    # a: (n, d)
    norms = np.linalg.norm(a, axis=1, keepdims=True)
    return (a / (norms + eps)).astype(np.float32, copy=False)


def train_kmeans(x, k, ngpu, niter=20):
    """
    在一个或多个 GPU 上运行 KMeans（Faiss）
    x: float32 [n, d]
    k: 聚类数
    """
    d = x.shape[1]
    clus = faiss.Clustering(d, k)
    clus.verbose = True
    clus.niter = niter

    # 不子采样
    clus.max_points_per_centroid = 10_000_000

    res = [faiss.StandardGpuResources() for _ in range(ngpu)]

    flat_config = []
    for i in range(ngpu):
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = False
        cfg.device = i
        flat_config.append(cfg)

    if ngpu == 1:
        index = faiss.GpuIndexFlatL2(res[0], d, flat_config[0])
    else:
        indexes = [faiss.GpuIndexFlatL2(res[i], d, flat_config[i]) for i in range(ngpu)]
        index = faiss.IndexReplicas()
        for sub_index in indexes:
            index.addIndex(sub_index)

    # 训练
    clus.train(x, index)
    centroids = faiss.vector_float_to_array(clus.centroids)
    return centroids.reshape(k, d)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Numpy 输入：目录 / 列表文件 / 单个 .npy 文件")
    parser.add_argument("--num_classes", type=int, required=True, help="聚类数 (k)")
    parser.add_argument("--output", required=True, help="输出 .npy 文件路径（保存质心，不做归一化，除非指定 --l2norm）")
    parser.add_argument("--drop_last", type=int, default=0, help="是否丢弃最后一维（1 丢弃，0 保留）")
    parser.add_argument("--ngpu", type=int, default=8, help="使用的 GPU 数量")
    parser.add_argument(
        "--l2norm",
        action="store_true",
        help=(
            "是否对输入特征做 L2 归一化（行归一化）。"
            "如果指定，则会在聚类前对输入做 unit-norm，并在训练结束后对质心也做 unit-norm 后保存。"
            "适用于希望以余弦/内积相似度为目标的场景。"
        ),
    )
    args = parser.parse_args()

    ngpu = args.ngpu

    print("[info] 读取特征...")
    x = read_feat(args.input, args.drop_last)
    x = x.reshape(x.shape[0], -1).astype("float32", copy=False)

    print(f"[info] 特征维度: n={x.shape[0]}, d={x.shape[1]}")

    if args.l2norm:
        print("[info] 对输入特征进行 L2 归一化（unit-norm）...")
        x = l2_row_normalize(x)

    print("[info] 运行 KMeans...")
    t0 = time.time()
    centroids = train_kmeans(x, args.num_classes, ngpu)
    t1 = time.time()
    print("total runtime: %.3f s" % (t1 - t0))

    # 如果输入被归一化，我们也对质心做行归一化再保存，这样保存的质心可以直接用于基于内积/余弦的检索。
    if args.l2norm:
        print("[info] 对质心做 L2 归一化后保存（因为 --l2norm 被指定）...")
        centroids = l2_row_normalize(centroids)

    print(f"[info] 保存质心到: {args.output}")
    np.save(args.output, centroids)
    print("[done]")
