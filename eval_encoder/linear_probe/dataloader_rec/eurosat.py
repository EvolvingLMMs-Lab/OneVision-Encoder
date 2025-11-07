import io
import os
from functools import partial
from typing import Tuple

import mxnet as mx
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split

root = f"{os.environ['LINEAR_PROBE_ROOT']}/eurosat"
num_example_train = 10000
num_example_test = 5000
num_example_other = 12000
num_classes = 10


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
    dataset = MXnetDataset(
        split='train', transform=transform)

    dataset_train, dataset_test, _ = random_split(
        dataset,
        lengths=[num_example_train,
                 num_example_test,
                 num_example_other],
        generator=torch.Generator().manual_seed(seed))
    return (dataset_train, None)



def get_loader_test(
    transform, batch_size, num_workers, seed
) -> Tuple[Dataset, DataLoader]:
    dataset = MXnetDataset(
        split='train', transform=transform)
    dataset_train, dataset_test, _ = random_split(
        dataset,
        lengths=[num_example_train,
                 num_example_test,
                 num_example_other],
        generator=torch.Generator().manual_seed(seed))
    return (dataset_test, None)


if __name__ == "__main__":
    with open("/mnt/linear_probe_dataset/rec/eurosat/train.idx", 'r') as f:
        data = f.readlines()
    print(len(data))