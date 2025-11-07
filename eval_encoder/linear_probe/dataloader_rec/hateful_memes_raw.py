import json
import os
import pathlib
from typing import Any, Callable, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split

from .utils import dataset_root

# root = os.getenv("DATASET_ROOT", "/home/face/cache")
root = '/mnt/linear_probe_dataset/rec'
root = f"{os.environ['LINEAR_PROBE_ROOT']}/hateful_memes"
num_example_train_val = 8500
num_example_train = 8000
num_example_val = 500
num_example_test = 500
num_classes = 2

class HatefulMemes(Dataset):
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
        self._data_folder = os.path.join(root, 'hateful_memes')
        if not self._check_exists():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        self.image_path = os.path.join(self._data_folder, 'img')
        self._image_files, self._labels = [], []
        self._labels: List[Optional[int]]
        self.annotation_file = os.path.join(self._data_folder, self._split + ".jsonl")

        data = open(self.annotation_file)
        for line in data.readlines():
            data = json.loads(line)
            image_str = "0" + str(data['id']) if len(str(data['id'])) == 4 else str(data['id'])
            self._image_files.append(os.path.join(self.image_path, image_str + '.png'))
            self._labels.append(data['label'])

    def __len__(self) -> int:
        return len(self._image_files)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        image_file = self._image_files[idx]
        label = self._labels[idx]
        image = Image.open(image_file).convert("RGB")

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
    dataset_trainval = HatefulMemes(root, download = False, split = 'train', transform = transform)
    dataset_train, dataset_val = random_split(
        dataset_trainval,
        lengths=[num_example_train, num_example_val],
        generator=torch.Generator().manual_seed(seed))
    return (dataset_train, None)



def get_loader_trainval(
    transform, batch_size, num_workers, seed
) -> Tuple[Dataset, DataLoader]:
    dataset_trainval = HatefulMemes(root, download = False, split = 'train', transform = transform)
    return (dataset_trainval, None)


def get_loader_val(
    transform, batch_size, num_workers, seed
) -> Tuple[Dataset, DataLoader]:
    dataset_trainval = HatefulMemes(root, download = False, split = 'train', transform = transform)
    dataset_train, dataset_val = random_split(
        dataset_trainval,
        lengths=[num_example_train, num_example_val],
        generator=torch.Generator().manual_seed(seed))
    return (dataset_val, None)


def get_loader_test(
    transform, batch_size, num_workers, seed
) -> Tuple[Dataset, DataLoader]:
    dataset_test = HatefulMemes(root, download = False, split = 'dev', transform = transform)
    return (dataset_test, None)



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

def roc_auc_score(
    transform, batch_size, num_workers, seed
) -> Tuple[Dataset, DataLoader]:
    raise NotImplementedError