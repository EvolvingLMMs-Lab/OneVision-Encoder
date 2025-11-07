import io
import os
from functools import partial
from typing import Tuple

import mxnet as mx
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split

from .utils import DistributedSampler, rank, worker_init_fn, world_size

root = "/mnt/linear_probe_dataset/rec/stanford_cars"
root = f"{os.environ['LINEAR_PROBE_ROOT']}/stanford_cars"
num_example_train = 8144
num_example_test = 8041
num_classes = 196


class MXnetDataset(Dataset):
    def __init__(self, split, transform):
        super(MXnetDataset, self).__init__()
        self.transform = transform
        path_imgrec = os.path.join(root, f'{split}.rec')
        path_imgidx = os.path.join(root, f'{split}.idx')

        self.indexed_recordio = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
        self.rec_keys = np.array(list(self.indexed_recordio.keys))

    def __getitem__(self, index):
        idx = self.rec_keys[index]
        header, buffer = mx.recordio.unpack(self.indexed_recordio.read_idx(idx))
        label = header.label
        sample = Image.open(io.BytesIO(buffer))
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label

    def __len__(self):
        return len(self.rec_keys)


def get_loader_train(
    transform, batch_size, num_workers, seed
) -> Tuple[Dataset, DataLoader]:
    dataset_train = MXnetDataset(
        split='train', transform=transform)

    train_sampler = DistributedSampler(
        dataset_train, num_replicas=world_size,
        rank=rank, shuffle=True, seed=seed)
    init_fn = partial(
        worker_init_fn, num_workers=num_workers,
        rank=rank, seed=seed)
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=batch_size,
        num_workers=num_workers,
        worker_init_fn=init_fn,
        sampler=train_sampler,
        pin_memory=True,
        drop_last=True,
    )
    return (dataset_train, dataloader_train)



def get_loader_test(
    transform, batch_size, num_workers, seed
) -> Tuple[Dataset, DataLoader]:
    dataset_test = MXnetDataset(
        split='test', transform=transform)
    test_sampler = DistributedSampler(
        dataset_test, num_replicas=world_size,
        rank=rank, shuffle=True, seed=seed)
    init_fn = partial(
        worker_init_fn, num_workers=num_workers,
        rank=rank, seed=seed)
    dataloader_test = DataLoader(
        dataset_test,
        batch_size=batch_size,
        num_workers=num_workers,
        worker_init_fn=init_fn,
        sampler=test_sampler,
        pin_memory=True,
        drop_last=True,
    )
    return (dataset_test, dataloader_test)


if __name__ == "__main__":
    dataset_train_val = MXnetDataset(
        split='train', transform = None)
    print(dataset_train_val)