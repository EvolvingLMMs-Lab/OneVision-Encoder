#!/usr/bin/env python
# coding=utf-8
'''
Author: Kaicheng Yang
LastEditors: Kaicheng Yang
Date: 2022-08-25 11:53:49
LastEditTime: 2022-08-27 16:16:53
'''

import os
import pathlib
from typing import Any, Callable, Optional, Tuple

import h5py
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .utils import dataset_root

root = '/mnt/linear_probe_dataset/rec'
root = f"{os.environ['LINEAR_PROBE_ROOT']}"


num_example_train_val = 294912
num_example_train = 262144
num_example_val = 32768
num_example_test = 32768
num_classes = 2


class PatchCamelyon(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        ) -> None:
        self.transform = transform
        self.target_transform = target_transform
        self._split = split
        self._data_folder = os.path.join(root, 'PatchCamelyon')
        if not self._check_exists():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        image_path = os.path.join(self._data_folder, self._split, self._split + '_data.h5')
        label_path = os.path.join(self._data_folder, self._split, self._split + '_label.h5')

        self._image_files = np.array(h5py.File(image_path, 'r')['x'])
        self._image_labels = np.array(h5py.File(label_path, 'r')['y']).reshape(-1)

    def __len__(self) -> int:
        return len(self._image_files)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        label = self._image_labels[idx]
        image = Image.fromarray(self._image_files[idx])

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def _check_exists(self) -> bool:
        return pathlib.Path(self._data_folder).exists() and pathlib.Path(self._data_folder).is_dir()

    def extra_repr(self) -> str:
        return f"split = {self._split}"




def get_loader_train(
    transform, batch_size, num_workers, seed
) -> Tuple[Dataset, DataLoader]:
    dataset_train = PatchCamelyon(root, download = True, split = 'train', transform = transform)
    return (dataset_train, None)



def get_loader_trainval(
    transform, batch_size, num_workers, seed
) -> Tuple[Dataset, DataLoader]:
    dataset_train = PatchCamelyon(root, download = True, split = 'train', transform = transform)
    dataset_val = PatchCamelyon(root, download = True, split = 'valid', transform = transform)
    dataset_trainval = dataset_train + dataset_val
    return (dataset_trainval, None)


def get_loader_val(
    transform, batch_size, num_workers, seed
) -> Tuple[Dataset, DataLoader]:
    dataset_val = PatchCamelyon(root, download = True, split = 'valid', transform = transform)
    return (dataset_val, None)


def get_loader_test(
    transform, batch_size, num_workers, seed
) -> Tuple[Dataset, DataLoader]:
    dataset_test = PatchCamelyon(root, download = True, split = 'test', transform = transform)
    return (dataset_test, None)
