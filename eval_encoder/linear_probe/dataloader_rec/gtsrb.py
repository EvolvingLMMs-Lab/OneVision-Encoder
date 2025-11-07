import io
import os
import random
from functools import partial
from typing import Tuple

import mxnet as mx
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split

root = f"{os.environ['LINEAR_PROBE_ROOT']}/gtsrb"
num_classes = 43


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

    return (dataset_train, None)



def get_loader_test(
    transform, batch_size, num_workers, seed
) -> Tuple[Dataset, DataLoader]:
    dataset_test = MXnetDataset(
        split='test', transform=transform)
    return (dataset_test, None)


if __name__ == "__main__":
    dataset_train_val = MXnetDataset(
        split='train', transform = None)
    print(dataset_train_val)