import os
import numpy as np
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
import decord
from nvidia.dali.pipeline import pipeline_def


class DALIWarper(object):
    def __init__(self, dali_iter, step_data_num):
        self.iter = dali_iter
        self.step_data_num = step_data_num

    def __next__(self):
        data_dict = self.iter.__next__()[0]
        videos = data_dict["videos"]
        labels = data_dict["labels"]
        # print(videos.max())
        return videos, labels

    def __iter__(self):
        return self

    def __len__(self):
        return self.step_data_num

    def reset(self):
        self.iter.reset()


class ExternalInputCallable:
    def __init__(self, mode, source_params):

        self.mode = mode

        self.file_list = source_params['file_list']
        self.num_shards = source_params['num_shards']
        self.shard_id = source_params['shard_id']
        self.batch_size = source_params['batch_size']
        self.sequence_length = source_params['sequence_length']
        self.use_rgb = source_params['use_rgb']
        self.seed = source_params['seed']

        # If the dataset size is not divisible by number of shards, the trailing samples will be omitted.
        self.shard_size = len(self.file_list) // self.num_shards
        self.shard_offset = self.shard_size * self.shard_id
        # drop last batch
        self.full_iterations = self.shard_size // self.batch_size
        # so that we don't have to recompute the `self.perm` for every sample
        self.perm = None
        self.last_seen_epoch = None
        self.replace_example_info = self.file_list[0]


    def get_frame_id_list(self, video_path, sequence_length):
        decord_vr = decord.VideoReader(video_path, num_threads=1, ctx=decord.cpu(0))
        duration = len(decord_vr)

        average_duration = duration // sequence_length
        all_index = []
        if average_duration > 0:
            if self.mode == 'train':
                all_index = list(
                    np.multiply(list(range(sequence_length)), average_duration) +
                    np.random.randint(average_duration, size = sequence_length))
            else:
                all_index = list(
                    np.multiply(list(range(sequence_length)), average_duration) +
                    np.ones(sequence_length, dtype = int) * (average_duration // 2))
        elif duration > sequence_length:
            if self.mode == 'train':
                all_index = list(np.sort(np.random.randint(duration, 
                                                            size = sequence_length)))
            else:
                all_index = list(range(sequence_length))
        else:
            all_index = [0] * (sequence_length - duration) + list(range(duration))
        frame_id_list = list(np.array(all_index))
        
        decord_vr.seek(0)
        video_data = decord_vr.get_batch(frame_id_list).asnumpy()            
        if self.use_rgb:
            video_data = video_data[:,:,:,::-1]
        return video_data
        

    def __call__(self, sample_info):
        #sample_info
        #idx_in_epoch – 0-based index of the sample within epoch
        #idx_in_batch – 0-based index of the sample within batch
        #iteration – number of current batch within epoch
        #epoch_idx – number of current epoch
        if sample_info.iteration >= self.full_iterations:
            # Indicate end of the epoch
            raise StopIteration
        if self.last_seen_epoch != sample_info.epoch_idx:
            self.last_seen_epoch = sample_info.epoch_idx
            cur_seed = self.seed + sample_info.epoch_idx
            self.perm = np.random.default_rng(seed=cur_seed).permutation(len(self.file_list))
            
        sample_idx = self.perm[sample_info.idx_in_epoch + self.shard_offset]

        example_info = self.file_list[sample_idx]
        video_path, video_label = example_info
        # print(video_path, video_label)
        try:
            video_data = self.get_frame_id_list(video_path, self.sequence_length)
        except:
            # print("error", video_path)
            # print(video_path)
            video_path, video_label = self.replace_example_info
            # print(video_path, video_label)
            video_data = self.get_frame_id_list(video_path, self.sequence_length)
        return video_data, np.int64([int(video_label)])

@pipeline_def(enable_conditionals=True)
def dali_pipeline(mode, source_params):
    
    short_side_size = source_params['short_side_size']
    input_size = source_params['input_size']
    mean = source_params['mean']
    std = source_params['std']

    if mode == "train":
        videos, labels = fn.external_source(
            source = ExternalInputCallable(mode, source_params),
            num_outputs = 2,
            batch = False,
            parallel = True,
            dtype = [types.UINT8, types.INT64],
            layout = ["FHWC", "C"]
        )
        videos = videos.gpu()
        videos = fn.resize(videos, resize_shorter=short_side_size, antialias=True, 
                            interp_type=types.INTERP_LINEAR, device="gpu")
        videos = fn.random_resized_crop(videos, size=[input_size, input_size], num_attempts=50, 
                                        random_area=[0.9, 1.0], device="gpu")
        
        brightness_contrast_probability = fn.random.coin_flip(dtype=types.BOOL, probability=0.8)
        if brightness_contrast_probability:
            videos = fn.brightness_contrast(videos, contrast=fn.random.uniform(range=(0.6, 1.4)),
                                            brightness=fn.random.uniform(range=(-0.125, 0.125)), device="gpu")
        saturation_probability = fn.random.coin_flip(dtype=types.BOOL, probability=0.8)
        if saturation_probability:
            videos = fn.saturation(videos, saturation=fn.random.uniform(range=[0.6, 1.4]), device="gpu")
        hue_probability = fn.random.coin_flip(dtype=types.BOOL, probability=0.8)
        if hue_probability:
            videos = fn.hue(videos, hue=fn.random.uniform(range=[-0.2, 0.2]), device="gpu")
        color_space_probability = fn.random.coin_flip(dtype=types.BOOL, probability=0.1)
        if color_space_probability:
            videos = fn.color_space_conversion(videos, image_type=types.RGB, output_type=types.BGR, device="gpu")
        videos = fn.crop_mirror_normalize(videos, dtype=types.FLOAT, output_layout = "CFHW",
                                        mean=[0.0,0.0,0.0], std=[1.0,1.0,1.0], device="gpu")
        # videos = videos.permute(0, 4, 1, 2, 3).gpu()

        return videos, labels
    else:
        videos, labels = fn.external_source(
            source = ExternalInputCallable(mode, source_params),
            num_outputs = 2,
            batch = False,
            parallel = True,
            dtype = [types.UINT8, types.INT64],
            layout = ["FHWC", "C"]
        )
        videos = videos.gpu()
        videos = fn.resize(videos, resize_shorter=input_size, antialias=True, 
                            interp_type=types.INTERP_LINEAR, device="gpu")
        videos = fn.crop(videos, crop=[input_size, input_size], device="gpu")
        videos = fn.crop_mirror_normalize(videos, dtype=types.FLOAT, output_layout = "CFHW",
                                        mean=[0.0,0.0,0.0], std=[1.0,1.0,1.0], device="gpu")

        labels = labels.gpu()
        return videos, labels

def dali_dataloader(data_root_path,
                    data_csv_path,
                    data_set,
                    dali_num_threads = 4,
                    dali_py_num_workers = 8,
                    batch_size = 32,
                    input_size = 224,
                    short_side_size = 239,
                    sequence_length = 16,
                    use_rgb = False,
                    mean = [0.0, 0.0, 0.0],
                    std = [1.0, 1.0, 1.0],
                    mode = "val",
                    num_shots = None,
                    seed = 0):

    if num_shots is not None:
        if mode == "train":
            txt_file_name = "{}_{}_{}.txt".format(data_set, mode, "fewshot{}".format(num_shots))
        else:
            txt_file_name = "{}_{}.txt".format(data_set, mode)
        file_list = []
        with open(os.path.join(data_csv_path, data_set, txt_file_name), 'r') as file:
            reader = file.readlines()
            if mode == "train":
                for line in reader:
                    offset_viedo_path, video_label = line.strip().split(',')
                    video_path = os.path.join(data_root_path, data_set, offset_viedo_path)
                    # print(video_path)
                    file_list.append([video_path, int(video_label)])
            else:
                for line in reader:
                    offset_viedo_path, video_label = line.strip().split(',')
                    video_path = os.path.join(data_root_path, data_set, offset_viedo_path)
                    # print(video_path)
                    file_list.append([video_path, int(video_label)])
    else:
        raise NotImplementedError("This function is not implemented yet")
            

    rank = int(os.getenv("RANK", 0))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))

    source_params = {
        "num_shards": world_size,
        "shard_id": rank,
        "file_list": file_list,
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "seed": seed + rank,
        "use_rgb": use_rgb,
        "input_size": input_size,
        "short_side_size": short_side_size,
        "mean": mean,
        "std": std,
    }


    pipe = dali_pipeline(
        batch_size = batch_size,
        num_threads = dali_num_threads,
        device_id = local_rank,
        seed = seed + rank,
        py_num_workers = dali_py_num_workers,
        py_start_method = 'spawn',
        prefetch_queue_depth = 1,
        mode = mode,
        source_params = source_params
    )
    pipe.build()


    dataloader = DALIWarper(
        dali_iter = DALIGenericIterator(pipelines=pipe,
            output_map=['videos', 'labels'],
            auto_reset=False,
            size=-1,
            last_batch_padded=False,
            last_batch_policy=LastBatchPolicy.FILL,
            prepare_first_batch=False),
        step_data_num = len(file_list) // world_size // batch_size,
    )
    
    return dataloader