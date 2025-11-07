from functools import partial
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision.datasets import STL10, Kitti

from .utils import (DistributedSampler, dataset_root, rank, worker_init_fn,
                    world_size)

root = "/mnt/linear_probe_dataset/rec"
root = f"{os.environ['LINEAR_PROBE_ROOT']}"

num_example_train_val = 5000
num_example_train = 1000
num_example_val = 4000
num_example_test = 8000
num_classes = 10


def get_loader_train(
    transform, batch_size, num_workers, seed
) -> Tuple[Dataset, DataLoader]:
    dataset = STL10(root, download=False, split='train', transform=transform)
    dataset_train, dataset_val= random_split(
        dataset,
        lengths=[num_example_train, num_example_val],
        generator=torch.Generator().manual_seed(seed))
    return (dataset_train, None)


def get_loader_test(
    transform, batch_size, num_workers, seed
) -> Tuple[Dataset, DataLoader]:
    dataset_test = STL10(root, download=False, split='test', transform=transform)
    return (dataset_test, None)
