# Standard library imports
import os
import random

# Third-party library imports
import torch
import nvidia.dali
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIClassificationIterator


rank = int(os.getenv("RANK", "0"))
local_rank = int(os.getenv("LOCAL_RANK", "0"))
world_size = int(os.getenv("WORLD_SIZE", "1"))


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
                    num_shards=self.num_shards,
                    shard_id=self.shard_id,
                    dali_aug=False,
                    grid_rows=4,
                    grid_cols=4)

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
            num_shards=self.num_shards,
            shard_id=self.shard_id,
            dali_aug=False,
            grid_rows=4,
            grid_cols=4
        )


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
        label = torch.zeros(size=(batch_size, 10), dtype=torch.long, device=local_rank)

        self.tensor_data = data
        self.tensor_label = label

    def __next__(self):
        with torch.no_grad():
            return self.tensor_data, self.tensor_label

    def __iter__(self):
        return self

    def reset(self):
        return


class DALIWarperV2(object):
    def __init__(self, dali_iter):
        self.iter = dali_iter

    def __next__(self):
        data_dict = self.iter.__next__()[0]
        tensor_data = data_dict["data"].cuda()
        tensor_label = data_dict["label"].long().cuda()
        return tensor_data, tensor_label

    def __iter__(self):
        return self

    def __len__(self):
        return self.iter.__len__()

    def reset(self):
        self.iter.reset()


def dali_random_gaussian_blur(img, window_size):
    img = fn.gaussian_blur(img, window_size=window_size * 2 + 1)
    return img

def dali_random_gray(img, prob_gray):
    saturate = fn.random.coin_flip(probability=1 - prob_gray)
    saturate = fn.cast(saturate, dtype=types.FLOAT)
    img = fn.hsv(img, saturation=saturate)
    return img

def dali_random_hsv(img, hue, saturation):
    img = fn.hsv(img, hue=hue, saturation=saturation)
    return img

def dali_random_brightness_contrast(img, brightness, contrast):
    img = fn.brightness_contrast(img, brightness=brightness, contrast=contrast)
    return img

def multiplexing(condition, true_case, false_case):
    neg_condition = condition ^ True
    return condition * true_case + neg_condition * false_case


def dali_dataloader(
    prefix,
    batch_size,
    image_size,
    workers,
    is_training=True,
    mean=None,
    std=None,
    seed=1437,
    num_shards=None,
    shard_id=None,
    dali_aug=False,
    grid_rows=4,
    grid_cols=4,
    random_shuffle=None
):
    if mean is None:
        mean = [x * 255 for x in [0.48145466, 0.4578275, 0.40821073]]
    if std is None:
        std = [x * 255 for x in [0.26862954, 0.26130258, 0.27577711]]

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

    if random_shuffle is None:
        if is_training:
            random_shuffle = True
        else:
            random_shuffle = False

    pipe = Pipeline(
        batch_size=batch_size,
        num_threads=workers,
        device_id=local_rank % 8,
        prefetch_queue_depth=3,
        seed=seed,
        exec_async=False,  # Synchronous execution, may be more stable but slower
        exec_pipelined=False  # Disable pipelined execution
    )

    with pipe:
        
        jpegs, labels = fn.readers.mxnet(
            path=rec_file,
            index_path=idx_file,
            initial_fill=16384,
            num_shards=num_shards,
            shard_id=shard_id,
            random_shuffle=random_shuffle,
            pad_last_batch=False,
            prefetch_queue_depth=4,
            name="train",
            stick_to_shard=True,
        )
        if is_training:
            images = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB)
            target_height, target_width = image_size[0] * grid_rows, image_size[1] * grid_cols
            
            images = fn.resize(
                images,
                device="gpu",
                resize_x=target_width,
                resize_y=target_height,
                # interp_type=types.INTERP_BICUBIC,
                # antialias=True
                interp_type=types.INTERP_TRIANGULAR,
            )
            
            if dali_aug:
                condition_blur = fn.random.coin_flip(probability=0.5)
                window_size_blur = fn.random.uniform(range=(1, 2), dtype=types.INT32)
                condition_flip = fn.random.coin_flip(probability=0.5)
                condition_hsv = fn.random.coin_flip(probability=0.5)
                hsv_hue = fn.random.uniform(range=(0., 20.), dtype=types.FLOAT)
                hsv_saturation = fn.random.uniform(range=(1., 1.2), dtype=types.FLOAT)
                
                condition_brightness_contrast = fn.random.coin_flip(probability=0.5)
                contrast = fn.random.uniform(range=(0.75, 1.25))
                brightness = fn.random.uniform(range=(0.75, 1.25))
                
                
                images = multiplexing(condition_blur, dali_random_gaussian_blur(images, window_size_blur), images)
                images = multiplexing(condition_hsv, dali_random_hsv(images, hsv_hue, hsv_saturation), images)
                images = dali_random_gray(images, 0.1)
                images = multiplexing(condition_brightness_contrast, dali_random_brightness_contrast(images, brightness, contrast), images)
                images = fn.cast(images, dtype=types.UINT8)
                
                images = fn.transpose(images, perm=(2, 0, 1)) # w * h * 3 ==> 3 * w * h
                images = fn.reshape(images, shape=(-1, 3, grid_cols, image_size[0], grid_rows, image_size[1]))
                
                images = fn.transpose(images, perm=(0, 2, 4, 1, 3, 5))
                images = fn.reshape(images, shape=(-1, 3, image_size[0], image_size[1])) # 16 * 3 * w * h
                images = fn.transpose(images, perm=(0, 2, 3, 1)) # 16 * w * h * 3
            mirror = 0
        else:
            images = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB)
            images = fn.resize(
                images,
                device="gpu",
                size=int(256 / 224 * image_size[0]),
                mode="not_smaller",
                interp_type=types.INTERP_TRIANGULAR,
            )
            mirror = False
            
        if dali_aug:
            # random crop
            crop_pos_x = random.uniform(0, 0.1)
            crop_pos_y = random.uniform(0, 0.1)
            crop_h = random.uniform(0.8, 1)
            
            crop_posx = int(crop_pos_x * image_size[0])
            crop_posy = int(crop_pos_y * image_size[1])
            crop_h = int(crop_h * image_size[0])
            crop_w = crop_h
            images = images[:, crop_posy:crop_posy + crop_h, crop_posx:crop_posx + crop_w,:]
            # resize
            images = fn.resize(
                images,
                resize_x=image_size[1],
                resize_y=image_size[1],
                interp_type=types.INTERP_TRIANGULAR,
            )
            images = fn.transpose(images, perm=(3, 0, 1, 2)) # 16 * w * h * 3 --> 3 * 16 * w * h
            images = fn.reshape(images, shape=(-1, 3, grid_rows, grid_cols, image_size[0], image_size[1]))
            images = fn.transpose(images, perm=(0, 1, 2, 4, 3, 5))
            images = fn.reshape(images, shape=(3, image_size[0] * grid_cols, image_size[1] * grid_rows))
        else:
            images = fn.crop_mirror_normalize(
                images.gpu(),
                dtype=types.FLOAT,
                output_layout="CHW",
                mean=mean,
                std=std,
                mirror=mirror,
            )
        
        pipe.set_outputs(images, labels)
    pipe.build()

    dataloader = DALIWarperV2(
        DALIClassificationIterator(pipelines=[pipe], reader_name="train"),
    )
    return dataloader
