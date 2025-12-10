import os

import nvidia.dali
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import torch
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIClassificationIterator
import random


class MultiRecDALIWarper(object):
    def __init__(
        self, list_prefix, batch_size, image_size, workers, shard_id, num_shards
    ):
        self.list_prefix = list_prefix
        self.batch_size = batch_size
        self.image_size = image_size
        self.workers = workers
        self.shard_id = shard_id
        self.num_shards = num_shards
        self.idx_rec = None
        self.dali_iter = None
        self.reset()

    def __next__(self):
        try:
            return next(self.dali_iter)
        except StopIteration:
            self.idx_rec += 1

            if self.idx_rec < len(self.list_prefix):
                del self.dali_iter
                nvidia.dali.backend.ReleaseUnusedMemory()
                self.dali_iter = dali_dataloader(
                    self.list_prefix[self.idx_rec],
                    self.batch_size,
                    self.image_size,
                    self.workers,
                    True,
                    seed=random.randint(0, 8096),
                    shard_id=self.shard_id,
                    num_shards=self.num_shards,
                )

                nvidia.dali.backend.ReleaseUnusedMemory()
                return next(self.dali_iter)
            else:
                self.reset()
                return next(self.dali_iter)

    def __iter__(self):
        return self

    def reset(self):
        self.idx_rec = 0
        self.dali_iter = dali_dataloader(
            self.list_prefix[0],
            self.batch_size,
            self.image_size,
            self.workers,
            True,
            seed=random.randint(0, 8096),
            shard_id=self.shard_id,
            num_shards=self.num_shards,
        )


@torch.no_grad()
class SyntheticDataIter(object):
    def __init__(self, batch_size, image_size, local_rank):
        data = torch.randint(
            low=0,
            high=255,
            size=(batch_size, 3, image_size, image_size),
            dtype=torch.float32,
            device=local_rank,
        )
        data[:, 0, :, :] -= 123.0
        data[:, 1, :, :] -= 116.0
        data[:, 2, :, :] -= 103.0
        data *= 0.01
        label = torch.zeros(size=(batch_size, 1), dtype=torch.long, device=local_rank)

        self.tensor_data = data
        self.tensor_label = label

    def __next__(self):
        return self.tensor_data, self.tensor_label

    def __iter__(self):
        return self

    def reset(self):
        return


class DALIWarperV2(object):
    def __init__(self, dali_iter, label_select=None):
        self.iter = dali_iter
        self.label_select = label_select

    def __next__(self):
        data_dict = self.iter.__next__()[0]
        tensor_data = data_dict["data"].cuda()
        tensor_label = data_dict["label"].long().cuda()

        if self.label_select is None:
            return {
                "pixel_values": tensor_data,
                "labels": tensor_label
            }
        else:
            if tensor_label.size(1) > 1:
                tensor_label: torch.Tensor = tensor_label[:, int(self.label_select)]
            else:
                tensor_label: torch.Tensor = tensor_label[:, 0]
            return {
                "pixel_values": tensor_data,
                "labels": tensor_label
            }
            # return tensor_data, tensor_label

    def __iter__(self):
        return self

    def __len__(self):
        return self.iter.__len__()

    def reset(self):
        self.iter.reset()


# rank = int(os.environ["RANK"])
# local_rank = int(os.environ["LOCAL_RANK"])
# world_size = int(os.environ["WORLD_SIZE"])

local_rank = int(os.environ.get("LOCAL_RANK", "0"))

def dali_dataloader(
    prefix,
    batch_size,
    image_size,
    workers=4,
    is_training=True,
    mean=[x * 255 for x in [0.48145466, 0.4578275, 0.40821073]],
    std=[x * 255 for x in [0.26862954, 0.26130258, 0.27577711]],
    label_select=None,
    seed=1437,
    num_shards=None,
    shard_id=None,
    max_side=336
):

    if num_shards is None:
        num_shards = int(os.environ.get("WORLD_SIZE", "1"))
    if shard_id is None:
        shard_id = int(os.environ.get("RANK", "0"))

    if isinstance(prefix, list):
        rec_file = [f"{x}.rec" for x in prefix]
        idx_file = [f"{x}.idx" for x in prefix]
    else:
        rec_file = f"{prefix}.rec"
        idx_file = f"{prefix}.idx"

    pipe = Pipeline(
        batch_size=batch_size,
        num_threads=workers,
        device_id=local_rank % 8,
        prefetch_queue_depth=3,
        seed=seed,
    )
    device_memory_padding = 211025920
    host_memory_padding = 140544512
    with pipe:
        jpegs, labels = fn.readers.mxnet(
            path=rec_file,
            index_path=idx_file,
            initial_fill=16384,
            num_shards=num_shards,
            shard_id=shard_id,
            random_shuffle=True,
            pad_last_batch=False,
            name=f"train_{prefix}",
            stick_to_shard=True,
        )

        images = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB)
        images = fn.resize(
            images,
            device="gpu",
            size=image_size,
            mode="not_smaller",
            interp_type=types.INTERP_TRIANGULAR,
        )
        images = fn.crop_mirror_normalize(
            images.gpu(), dtype=types.FLOAT, output_layout="CHW", crop=(image_size[0], image_size[1]),
            mean=mean, std=std, mirror=False,)

        pipe.set_outputs(images, labels)
    pipe.build()

    dataloader = DALIWarperV2(
        DALIClassificationIterator(pipelines=[pipe], reader_name=f"train_{prefix}"),
        label_select=label_select,
    )
    return dataloader


if __name__ == "__main__":
    import cv2
    import numpy as np

    loader =  dali_dataloader(
    "/data_4/coyo_ocr_v0/train_00",
    4,
    [336, 336],
    workers=4,
    is_training=True,
    mean=[0, 0, 0],
    std=[1, 1, 1],
    label_select=None,
    seed=1437,
    num_shards=None,
    shard_id=None,
    max_side=336)

    image_list = []
    big_img = np.zeros((3360, 3360, 3), dtype=np.uint8)
    for image, label in loader:

        print(label)
        image = image[0].permute(1, 2, 0).cpu().numpy()
        image_list.append(image)
        if len(image_list) == 100:
            break

    for i in range(100):
        row = i // 10
        col = i % 10
        big_img[row * 336:(row + 1) * 336, col * 336:(col + 1) * 336] = image_list[i]
    cv2.imwrite("output_big_image.jpg", big_img[:, :, ::-1])
