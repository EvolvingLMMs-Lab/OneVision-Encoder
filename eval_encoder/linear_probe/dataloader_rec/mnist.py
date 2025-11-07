import os
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import MNIST

from .utils import dataset_root

root = f"{os.environ['LINEAR_PROBE_ROOT']}"

num_example_train_val = 60000
num_example_train = 50000
num_example_val = 10000
num_example_test = 10000
num_classes = 10

def get_loader_train(
    transform, batch_size, num_workers, seed
) -> Tuple[Dataset, DataLoader]:
    dataset_train = MNIST(root, download=False, train=True, transform=transform)
    return (dataset_train, None)

def get_loader_test(
    transform, batch_size, num_workers, seed
) -> Tuple[Dataset, DataLoader]:
    dataset_test = MNIST(root, download=False, train=False, transform=transform)
    return (dataset_test, None)
