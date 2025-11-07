from functools import partial
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision.datasets import CIFAR10

from .utils import DistributedSampler
from .utils import rank, world_size
from .utils import worker_init_fn
from .utils import dataset_root

root = f"{os.environ['LINEAR_PROBE_ROOT']}"
num_example_train = 50000
num_example_test = 10000
num_classes = 10


def get_loader_train(
    transform, batch_size, num_workers, seed
) -> Tuple[Dataset, DataLoader]:
    dataset_train = CIFAR10(root, download=False, train=True, transform=transform)
    return (dataset_train, None)


def get_loader_test(
    transform, batch_size, num_workers, seed
) -> Tuple[Dataset, DataLoader]:
    dataset_test = CIFAR10(root, download=False, train=False, transform=transform)
    return (dataset_test, None)


  
