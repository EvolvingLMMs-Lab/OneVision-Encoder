import io

import mxnet as mx
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class MXnetDataset(Dataset):
    def __init__(self, prefix, transform):
        super().__init__()
        self.transform = transform
        path_imgrec = f'{prefix}.rec'
        path_imgidx = f'{prefix}.idx'

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
