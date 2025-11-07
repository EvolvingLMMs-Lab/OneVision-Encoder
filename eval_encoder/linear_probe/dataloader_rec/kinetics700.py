

import os
import pathlib
from functools import partial
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from .utils import (DistributedSampler, dataset_root, rank, worker_init_fn,
                    world_size)

root = '/mnt/linear_probe_dataset/rec'
root = f"{os.environ['LINEAR_PROBE_ROOT']}/kinetics700"
num_example_train_val = 530779
num_example_train = 430779
num_example_val = 100000
num_example_test = 33944
num_classes = 700

class Kinetices700(Dataset):
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
        self._data_folder = os.path.join(root, 'Kinetics700')
        if not self._check_exists():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")
        
        image_path = os.path.join(self._data_folder, self._split)
        label_path = os.path.join(self._data_folder, self._split + '.npy')

        image_file_list = os.listdir(image_path)
        image_label_map = np.load(label_path, allow_pickle = True).tolist()

        self._image_files, self._image_labels = [], []
        for i in range(len(image_file_list)):
            image_path_ = os.path.join(image_path, image_file_list[i])
            image_label_ = image_label_map[image_file_list[i]]
            self._image_files.append(image_path_)
            self._image_labels.append(image_label_)

    def __len__(self) -> int:
        return len(self._image_files)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        label = self._image_labels[idx]
        image = Image.open(self._image_files[idx]).convert('RGB')
        
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
    dataset_train_val = Kinetices700(root, download = False, split = 'train', transform = transform)

    dataset_train, dataset_val = random_split(
        dataset_train_val,
        lengths=[num_example_train, num_example_val],
        generator=torch.Generator().manual_seed(seed))

    train_sampler = DistributedSampler(
        dataset_train, num_replicas = world_size,
        rank = rank, shuffle = True, seed = seed)

    init_fn = partial(
        worker_init_fn, num_workers = num_workers,
        rank = rank, seed = seed)

    loader_train = DataLoader(
        dataset_train,
        batch_size = batch_size,
        num_workers = num_workers,
        worker_init_fn = init_fn,
        sampler = train_sampler,
        pin_memory = True,
        drop_last = False,
    )

    return (dataset_train, loader_train)



def get_loader_trainval(
    transform, batch_size, num_workers, seed
) -> Tuple[Dataset, DataLoader]:
    dataset_train_val = Kinetices700(root, download = False, split = 'train', transform = transform)

    trainval_sampler = DistributedSampler(
        dataset_train_val, num_replicas = world_size,
        rank = rank, shuffle = True, seed = seed)

    init_fn = partial(
        worker_init_fn, num_workers = num_workers,
        rank = rank, seed = seed)

    loader_trainval = DataLoader(
        dataset_train_val,
        batch_size = batch_size,
        num_workers = num_workers,
        worker_init_fn = init_fn,
        sampler = trainval_sampler,
        pin_memory = True,
        drop_last = False,
    )

    return (dataset_train_val, loader_trainval)


def get_loader_val(
    transform, batch_size, num_workers, seed
) -> Tuple[Dataset, DataLoader]:
    dataset_train_val = Kinetices700(root, download = False, split = 'train', transform = transform)

    dataset_train, dataset_val = random_split(
        dataset_train_val,
        lengths=[num_example_train, num_example_val],
        generator=torch.Generator().manual_seed(seed))
    
    val_sampler = DistributedSampler(
        dataset_val, num_replicas = world_size,
        rank = rank, shuffle = True, seed = seed)

    init_fn = partial(
        worker_init_fn, num_workers = num_workers,
        rank = rank, seed=seed)

    loader_val = DataLoader(
        dataset_val,
        batch_size = batch_size,
        num_workers = num_workers,
        worker_init_fn = init_fn,
        sampler = val_sampler,
        pin_memory = True,
        drop_last = False,
    )

    return (dataset_val, loader_val)


def get_loader_test(
    transform, batch_size, num_workers, seed
) -> Tuple[Dataset, DataLoader]:
    dataset_test = Kinetices700(root, download = False, split = 'val', transform = transform)

    test_sampler = DistributedSampler(
        dataset_test, num_replicas = world_size,
        rank = rank, shuffle = True, seed = seed)

    init_fn = partial(
        worker_init_fn, num_workers = num_workers,
        rank = rank, seed = seed)

    loader_test = DataLoader(
        dataset_test,
        batch_size = batch_size,
        num_workers = num_workers,
        worker_init_fn = init_fn,
        sampler = test_sampler,
        pin_memory = True,
        drop_last = False,
    )

    return (dataset_test, loader_test)


def get_loader_trainval_1000(
    transform, batch_size, num_workers, seed
) -> Tuple[Dataset, DataLoader]:
    raise NotImplementedError


def get_loader_train_800(
    transform, batch_size, num_workers, seed
) -> Tuple[Dataset, DataLoader]:
    raise NotImplementedError


def get_loader_val_200(
    transform, batch_size, num_workers, seed
) -> Tuple[Dataset, DataLoader]:
    raise NotImplementedError


def avg_acc1_acc5(
    transform, batch_size, num_workers, seed
) -> Tuple[Dataset, DataLoader]:
    raise NotImplementedError