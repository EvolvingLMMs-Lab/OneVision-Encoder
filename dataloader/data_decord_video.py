import os

import decord
import numpy as np
import nvidia.dali.fn as fn
import nvidia.dali.types as types

from nvidia.dali.auto_aug import rand_augment
from nvidia.dali.pipeline import pipeline_def
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy

rank = int(os.environ.get("RANK", "0"))
local_rank = int(os.environ.get("LOCAL_RANK", "0"))
world_size = int(os.environ.get("WORLD_SIZE", "1"))

class DALIWarper(object):
    def __init__(self, dali_iter, step_data_num, mode="train"):
        self.iter = dali_iter
        self.step_data_num = step_data_num
        assert(mode in ["train", "val", "test"])
        self.mode = mode

    def __next__(self):
        try:
            data_dict = self.iter.__next__()[0]
            videos = data_dict["videos"]
            labels = data_dict["labels"]
            return {
                "pixel_values": videos,
                "labels": labels
            }
        except StopIteration:
            self.iter.reset()
            return self.__next__()

    def __iter__(self):
        return self

    def __len__(self):
        return self.step_data_num

    def reset(self):
        self.iter.reset()


class ExternalInputCallable:
    def __init__(
        self, source_params):

        self.file_list = source_params.get("file_list", None)
        self.label = source_params.get("label")
        if self.file_list is None:
            raise ValueError("file_list is None")

        self.num_shards = source_params.get("num_shards", world_size)
        self.shard_id = source_params.get("shard_id", rank)

        self.batch_size = source_params.get("batch_size", 32)
        self.input_size = source_params.get("input_size", 224)

        self.short_side_size = source_params.get("short_side_size", 256)
        self.sequence_length = source_params.get("sequence_length", 16)
        self.stride = source_params.get("stride", 8)
        self.use_sparse_sampling = source_params.get("use_sparse_sampling", False)
        self.use_rgb = source_params.get("use_rgb", True)
        self.use_flip = source_params.get("use_flip", True)
        self.seed = source_params.get("seed", 0)
        self.reprob = source_params.get("reprob", 0.0)

        self.perm = None
        self.last_seen_epoch = None
        self.replace_example_info = self.file_list[0]

        self.mode = "train"

        # If the dataset size is not divisible by number of shards, the trailing samples will be omitted.
        self.shard_size = len(self.file_list) // self.num_shards
        self.shard_offset = self.shard_size * self.shard_id
        # drop last batch
        self.full_iterations = self.shard_size // self.batch_size


    def sparse_sampling_get_frameid_data(
            self,
            video_path,
            sequence_length,
            test_info):

        decord_vr = decord.VideoReader(video_path, num_threads=1, ctx=decord.cpu(0))
        duration = len(decord_vr)

        if self.mode in ["train", "val"]:
            average_duration = duration // sequence_length
            all_index = []

            if average_duration > 0:
                if self.mode == 'val':
                    all_index = list(
                        np.multiply(list(range(sequence_length)), average_duration) +
                        np.ones(sequence_length, dtype=int) * (average_duration // 2)
                    )
                elif self.mode == 'train':
                    all_index = list(
                        np.multiply(list(range(sequence_length)), average_duration) +
                        np.random.randint(average_duration, size=sequence_length)
                    )
                else:
                    raise ValueError("mode should be train or val")

            elif duration > sequence_length:
                if self.mode == 'val':
                    all_index = list(range(sequence_length))
                elif self.mode == 'train':
                    all_index = list(
                        np.sort(np.random.randint(duration, size=sequence_length))
                    )
                else:
                    raise ValueError("mode should be train or val")

            else:
                all_index = [0] * (sequence_length - duration) + list(range(duration))

            frame_id_list = list(np.array(all_index))
            
            decord_vr.seek(0)
            video_data = decord_vr.get_batch(frame_id_list).asnumpy()
            
            # if self.use_rgb:
            #     video_data = video_data[:, :, :, ::-1]
                
            return video_data
        else:
            chunk_nb, split_nb, video_idx = test_info
            tick = duration / float(sequence_length)
            all_index = []
            for t_seg in range(self.test_tta_num_segment):
                tmp_index = [
                    int(t_seg * tick / self.test_tta_num_segment + tick * x)
                    for x in range(sequence_length)
                ]
                all_index.extend(tmp_index)
            all_index = list(np.sort(np.array(all_index)))
            cur_index = all_index[chunk_nb::self.test_tta_num_segment]     
            decord_vr.seek(0)
            video_data = decord_vr.get_batch(cur_index).asnumpy()     
            if self.use_rgb:
                video_data = video_data[:,:,:,::-1]

            vf, vh, vw, vc = video_data.shape
            short_side_size = min(vh, vw)
            long_side_size = max(vh, vw)
            spatial_step = 1.0 * (long_side_size - short_side_size) / (self.test_tta_num_crop - 1)
            spatial_start = int(split_nb * spatial_step)
            if vh >= vw:
                video_data = video_data[:, spatial_start:spatial_start + short_side_size, :, :]
            else:
                video_data = video_data[:, :, spatial_start:spatial_start + short_side_size, :]
            return video_data


    def __call__(self, sample_info):
        # sample_info
        # idx_in_epoch – 0-based index of the sample within epoch
        # idx_in_batch – 0-based index of the sample within batch
        # iteration – number of current batch within epoch
        # epoch_idx – number of current epoch
        if sample_info.iteration >= self.full_iterations:
            # Indicate end of the epoch
            raise StopIteration

        if self.last_seen_epoch != sample_info.epoch_idx:
            self.last_seen_epoch = sample_info.epoch_idx
            cur_seed = self.seed + sample_info.epoch_idx
            self.perm = np.random.default_rng(seed=cur_seed).permutation(len(self.file_list))

        sample_idx = self.perm[sample_info.idx_in_epoch + self.shard_offset]

        example_info = self.file_list[sample_idx]
        video_label = self.label[sample_idx]
        test_info = None

        video_path = example_info

        try:
            video_data = self.sparse_sampling_get_frameid_data(video_path, self.sequence_length, test_info)
        except:
            print("Error: ", video_path)
            video_path = self.replace_example_info
            video_label = self.label[0]
            video_data = self.sparse_sampling_get_frameid_data(video_path, self.sequence_length, test_info)

        if self.mode == "test":
            chunk_nb, split_nb, video_idx = test_info
            return (
                    video_data,
                    np.int64([int(video_label)]),
                    np.int64([int(chunk_nb)]),
                    np.int64([int(split_nb)]),
                    np.int64([int(video_idx)])
                    )
        else:
            if isinstance(video_label, int):
                video_label = np.int64(np.array([video_label]))
            elif isinstance(video_label, np.ndarray):
                video_label = video_label.astype(np.int64)
            return video_data, video_label


@pipeline_def()
def dali_pipeline(mode, source_params):

    if mode == "train":
        videos, labels = fn.external_source(
            source      = ExternalInputCallable(source_params),
            num_outputs = 2,
            batch       = False,
            parallel    = True,
            dtype       = [types.UINT8, types.INT64],
            layout      = ["FHWC", "C"]
        )
        
        videos = videos.gpu()
        # videos = rand_augment.rand_augment(videos, n=4, m=7, fill_value=128, monotonic_mag=True)
        videos = fn.random_resized_crop(
            videos,
            random_area         = (0.50, 1.0),
            random_aspect_ratio = (0.75, 1.3333),
            size                = [source_params['input_size'], source_params['input_size']],
            num_attempts        = 10,
            antialias           = True,
            interp_type         = types.INTERP_LINEAR
        )
        if source_params['reprob'] > 0:
            erase_probability = fn.random.coin_flip(dtype=types.BOOL, probability=source_params['reprob'])
            if erase_probability:
                mask = videos * 0
                # anchor=(y0, x0, y1, x1, …);shape=(h0, w0, h1, w1, …)
                mask = fn.erase(
                    mask,
                    device     = "gpu",
                    axis_names = "HW",
                    fill_value = 255,
                    anchor     = fn.random.uniform(range=(0, source_params['input_size']), shape=(2,)),
                    shape      = fn.random.uniform(range=(20, 90), shape=(2,))
                )
                noise = fn.random.normal(videos, device="gpu", dtype=types.INT8)
                videos = (videos & (255 - mask)) | (noise & mask)
            else:
                # align dali-types
                mask = videos * 0
                videos = (videos & (255 + mask))
                
        if source_params['use_flip']:
            videos = fn.flip(
                videos,
                device     = "gpu",
                horizontal = fn.random.coin_flip(probability=0.5)
            )

        videos = fn.crop_mirror_normalize(
            videos,
            device        = "gpu",
            dtype         = types.FLOAT,
            output_layout = "CFHW",
            mean          = source_params['mean'],
            std           = source_params['std']
        )
        labels = labels.gpu()
        return videos, labels
        
    elif mode == "val":
        videos, labels = fn.external_source(
            source      = ExternalInputCallable(mode, source_params),
            num_outputs = 2,
            batch       = False,
            parallel    = True,
            dtype       = [types.UINT8, types.INT64],
            layout      = ["FHWC", "C"]
        )
        
        videos = videos.gpu()
        videos = fn.resize(
            videos,
            device         = "gpu",
            antialias      = True,
            interp_type    = types.INTERP_LINEAR,
            resize_shorter = source_params['short_side_size']
        )
        videos = fn.crop(
            videos,
            device = "gpu",
            crop   = [source_params['input_size'], source_params['input_size']]
        )
        videos = fn.crop_mirror_normalize(
            videos,
            device        = "gpu",
            dtype         = types.FLOAT,
            output_layout = "CFHW",
            mean          = source_params['mean'],
            std           = source_params['std']
        )
        
        labels = labels.gpu()
        return videos, labels
        
    else:
        videos, labels, chunk_nb, split_nb, video_idx = fn.external_source(
            source      = ExternalInputCallable(mode, source_params),
            num_outputs = 5,
            batch       = False,
            parallel    = True,
            dtype       = [types.UINT8, types.INT64, types.INT64, types.INT64, types.INT64],
            layout      = ["FHWC", "C", "C", "C", "C"]
        )
        
        videos = videos.gpu()
        videos = fn.resize(
            videos,
            device      = "gpu",
            antialias   = True,
            interp_type = types.INTERP_LINEAR,
            resize_y    = source_params['input_size'],
            resize_x    = source_params['input_size']
        )
        videos = fn.crop_mirror_normalize(
            videos,
            device        = "gpu",
            dtype         = types.FLOAT,
            output_layout = "CFHW",
            mean          = source_params['mean'],
            std           = source_params['std']
        )
        
        labels = labels.gpu()
        chunk_nb = chunk_nb.gpu()
        split_nb = split_nb.gpu()
        video_idx = video_idx.gpu()
        
        return videos, labels, chunk_nb, split_nb, video_idx


def dali_dataloader(
    file_list,
    label,
    dali_num_threads,
    dali_py_num_workers,
    batch_size,
    input_size         = 224,
    sequence_length    = 16,
    stride             = 8,
    mode               = "train",
    seed               = 0,
    short_side_size    = 256,
    num_shards         = None,
    shard_id           = None,
):

    if isinstance(input_size, list):
        input_size = input_size[0]

    mean = [x * 255 for x in [0.48145466, 0.4578275, 0.40821073]]
    std = [x * 255 for x in [0.26862954, 0.26130258, 0.27577711]]

    source_params = {
        # Basic parameters
        "batch_size":           batch_size,
        "seed":                 seed + rank,
        "num_shards":           num_shards, 
        "shard_id":             shard_id,
        "file_list":            file_list,
        "label":                label,
        
        # Size parameters
        "input_size":           input_size,
        "short_side_size":      short_side_size,
        "sequence_length":      sequence_length,
        "stride":               stride,
        
        # Feature flags
        "use_sparse_sampling":  True,
        "use_rgb":              True,
        "use_flip":             True,
        
        # Augmentation parameters
        "mean":                 mean,
        "std":                  std,
        "reprob":               0,


    }
        
    pipe = dali_pipeline(
        batch_size           = batch_size,
        num_threads          = dali_num_threads,
        device_id            = local_rank,
        seed                 = seed + rank,
        py_num_workers       = dali_py_num_workers,
        py_start_method      = 'spawn',
        prefetch_queue_depth = 2,
        mode                 = mode,
        source_params        = source_params
    )
    pipe.build()

    # Define output mapping based on mode
    output_map = ['videos', 'labels']
    if mode == "test":
        output_map.extend(['chunk_nb', 'split_nb', 'sample_idx'])

    dataloader = DALIWarper(
        dali_iter = DALIGenericIterator(
            pipelines           = pipe,
            output_map          = output_map,
            auto_reset          = True,
            size                = -1,
            last_batch_padded   = False,
            last_batch_policy   = LastBatchPolicy.FILL,
            prepare_first_batch = False
        ),
        step_data_num = len(file_list) // world_size // batch_size,
        mode          = mode
    )
    return dataloader


# --------- Paste everything below at the end of your file --------------

import argparse
from pathlib import Path
import numpy as np
from PIL import Image

try:
    import torch
except ImportError:
    torch = None


def _to_numpy(x):
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.array(x)


def _undo_dali_norm(frame_chw, mean255, std255):
    # frame_chw: C x H x W (float32), output of crop_mirror_normalize
    mean = np.asarray(mean255, dtype=np.float32).reshape(-1, 1, 1)
    std = np.asarray(std255, dtype=np.float32).reshape(-1, 1, 1)
    img = frame_chw * std + mean  # back to 0..255
    img = np.transpose(img, (1, 2, 0))
    return np.clip(img, 0, 255).astype(np.uint8)


def _save_image(img_hwc_uint8, path):
    Image.fromarray(img_hwc_uint8).save(str(path))


def main():
    parser = argparse.ArgumentParser(description="Quick DALI dataloader visual check")
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--file-list", type=str, default="/video_vit/train_UniViT/mp4_list.txt")
    parser.add_argument("--outdir", type=str, default="./dali_debug_out", help="Output dir to save frames")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=8)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--short-side-size", type=int, default=256)
    parser.add_argument("--dali-threads", type=int, default=2)
    parser.add_argument("--dali-py-workers", type=int, default=2)
    parser.add_argument("--num-samples", type=int, default=10, help="Save concatenated frames for N samples")
    # NOTE: only 'train' mode is wired correctly in the provided code (val/test ctor differs)
    parser.add_argument("--mode", type=str, default="train", choices=["train"], help="Use 'train' for this check")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    with open(args.file_list, "r", encoding="utf-8") as f:
        file_list = [ln.strip() for ln in f if ln.strip()]
    if not file_list:
        raise SystemExit("file_list is empty")

    labels = np.load("/video_vit/train_UniViT/list_merged.npy")

    # Important: set num_shards=1, shard_id=0 to avoid None handling issues in source_params
    dataloader = dali_dataloader(
        file_list=file_list,
        label=labels,
        dali_num_threads=args.dali_threads,
        dali_py_num_workers=args.dali_py_workers,
        batch_size=args.batch_size,
        input_size=args.input_size,
        sequence_length=args.sequence_length,
        stride=1,
        mode=args.mode,
        seed=0,
        short_side_size=args.short_side_size,
        num_shards=1,
        shard_id=0,
    )

    it = iter(dataloader)
    saved = 0
    target = args.num_samples

    # Mean/std used inside dali_dataloader (scaled to 0..255)
    mean255 = [x * 255 for x in [0.48145466, 0.4578275, 0.40821073]]
    std255 = [x * 255 for x in [0.26862954, 0.26130258, 0.27577711]]

    while saved < target:
        videos, labels_out = next(it)
        print(labels_out)
        vids_np = _to_numpy(videos)
        print(f"DALI batch shape: {vids_np.shape}, dtype={vids_np.dtype}")

        # Expect shape: [B, C, F, H, W]
        if vids_np.ndim != 5:
            raise RuntimeError(f"Unexpected videos shape: {vids_np.shape} (expected 5D BxCxFxHxW)")

        B, C, F, H, W = vids_np.shape
        for b in range(B):
            if saved >= target:
                break
            sample = vids_np[b]  # C x F x H x W

            # Collect frames (no channel swap), then concatenate horizontally
            frames = []
            for fidx in range(F):
                frame_chw = sample[:, fidx, :, :]  # C x H x W
                if np.issubdtype(frame_chw.dtype, np.floating):
                    img = _undo_dali_norm(frame_chw, mean255, std255)  # H x W x C, uint8
                else:
                    img = np.transpose(frame_chw, (1, 2, 0)).astype(np.uint8)
                frames.append(img)

            big_img = np.concatenate(frames, axis=1)  # H x (F*W) x C
            out_path = outdir / f"sample_{saved:03d}_concat_F{F}_HxW{H}x{W}.png"
            _save_image(big_img, out_path)
            print(f"[{saved+1}/{target}] Saved {out_path}")
            saved += 1

    print(f"Done. Saved {saved} concatenated samples to {outdir}")


if __name__ == "__main__":
    main()
