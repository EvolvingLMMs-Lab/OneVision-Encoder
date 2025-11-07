import io
import os
from functools import partial
from typing import Tuple

import mxnet as mx
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split

root = "/mnt/linear_probe_dataset/rec/SUN397"
root= f"{os.environ['LINEAR_PROBE_ROOT']}/SUN397"
num_example_train = 19850
num_example_test = 19850
num_example_others = 68934
num_classes = 397


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
                 num_example_others],
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
                 num_example_others],
        generator=torch.Generator().manual_seed(seed))
    return (dataset_test, None)


if __name__ == "__main__":
    dataset_train_val = MXnetDataset(
        split='train', transform = None)
    print(dataset_train_val)