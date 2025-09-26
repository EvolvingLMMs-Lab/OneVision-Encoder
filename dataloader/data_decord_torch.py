import os
import random
import torch
import decord
import numpy as np
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torchvision import transforms
from typing import List, Tuple

# Global variables
rank = int(os.environ.get("RANK", "0"))
local_rank = int(os.environ.get("LOCAL_RANK", "0"))
world_size = int(os.environ.get("WORLD_SIZE", "1"))


class VideoDataset(Dataset):
    """Dataset for video classification training."""
    
    def __init__(
        self,
        file_list: List[Tuple[str, int]],
        input_size: int = 224,
        sequence_length: int = 16,
        use_rgb: bool = True,
        use_flip: bool = True,
        reprob: float = 0.0,
        seed: int = 0,
        ):
        self.file_list = file_list
        self.input_size = input_size
        self.sequence_length = sequence_length
        self.use_rgb = use_rgb
        self.use_flip = use_flip
        self.reprob = reprob
        self.seed = seed
        
        # Default mean and std values
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]) * 255
        self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711]) * 255
        
        # Save a valid item for replacement in case of errors
        self.replace_example_info = self.file_list[0]
        
        # Set up transforms for training
        self.transform = transforms.Compose([
            transforms.Lambda(lambda x: torch.from_numpy(x.astype(np.float32)).permute(3, 0, 1, 2)),  # FHWC -> CFHW
            transforms.RandomResizedCrop(
                size=(self.input_size, self.input_size),
                scale=(0.5, 1.0),
                ratio=(0.75, 1.3333),
                antialias=True
            ),
            transforms.RandomHorizontalFlip(p=0.5) if self.use_flip else transforms.Lambda(lambda x: x),
            transforms.Lambda(lambda x: (x - self.mean.view(-1, 1, 1, 1)) / self.std.view(-1, 1, 1, 1)),
        ])

    def _sample_frames(self, video_path):
        """Sample frames from video for training."""
        try:
            decord.bridge.set_bridge('torch')
            decord_vr = decord.VideoReader(video_path, num_threads=4)
            duration = len(decord_vr)
            
            average_duration = duration // self.sequence_length
            all_index = []
            
            if average_duration > 0:
                all_index = list(
                    np.multiply(list(range(self.sequence_length)), average_duration) +
                    np.random.randint(average_duration, size=self.sequence_length)
                )
            elif duration > self.sequence_length:
                all_index = list(
                    np.sort(np.random.randint(duration, size=self.sequence_length))
                )
            else:
                all_index = [0] * (self.sequence_length - duration) + list(range(duration))
            
            frame_id_list = list(np.array(all_index))
            decord_vr.seek(0)
            video_data = decord_vr.get_batch(frame_id_list).numpy()
            
            if not self.use_rgb:  # Convert BGR to RGB if needed
                video_data = video_data[:, :, :, ::-1]
                
            return video_data
                
        except Exception as e:
            print(f"Error processing {video_path}: {e}")
            return None
    
    def __len__(self) -> int:
        return len(self.file_list)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        video_path, video_label = self.file_list[idx]
        
        try:
            video_data = self._sample_frames(video_path)
            if video_data is None:
                video_path, video_label = self.replace_example_info
                video_data = self._sample_frames(video_path)
        except Exception:
            print(f"Error: {video_path}")
            video_path, video_label = self.replace_example_info
            video_data = self._sample_frames(video_path)
        
        video_tensor = self.transform(video_data)
        
        if isinstance(video_label, int):
            label_tensor = torch.tensor(video_label, dtype=torch.long)
        elif isinstance(video_label, np.ndarray):
            label_tensor = torch.from_numpy(video_label).long()
        else:
            label_tensor = torch.tensor(video_label, dtype=torch.long)
            
        return video_tensor, label_tensor


class DALIWarper(object):
    def __init__(self, dali_iter, step_data_num, mode="train", auto_reset=True):
        self.iter = dali_iter
        self.step_data_num = step_data_num
        assert(mode in ["train", "val", "test"])
        self.mode = mode
        self.auto_reset = auto_reset

    def __next__(self):
        try:
            videos, labels = self.iter.__next__()
            videos = videos.cuda()
            labels = labels.cuda()
            return videos, labels
        except StopIteration:
            if self.auto_reset:
                self.iter.reset()
                return self.__next__()
            else:
                raise StopIteration

    def __iter__(self):
        return self

    def __len__(self):
        return self.step_data_num

    def reset(self):
        self.iter.reset()


def create_video_dataloader(
    file_list: List[Tuple[str, int]],
    batch_size: int = 32,
    num_workers: int = 4,
    input_size: int = 224,
    sequence_length: int = 16,
    seed: int = 0,
    num_shard: int = 1,
    shard_id: int = 0,
    pin_memory: bool = True,
    reprob: float = 0.0,
    use_dali_warper: bool = False,
    dali_mode: str = "train",
    auto_reset: bool = True,
):
    """Create a PyTorch dataloader for video training data."""
    
    dataset = VideoDataset(
        file_list=file_list,
        input_size=input_size,
        sequence_length=sequence_length,
        seed=seed,
        reprob=reprob,
    )

    sampler = DistributedSampler(
        dataset,
        num_replicas=num_shard,
        rank=shard_id,
        shuffle=True,
        seed=seed
    )
    # shuffle = False  # Sampler will handle shuffling

    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        # shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2,
    )
    
    # 如果需要，使用 DALIWarper 封装 DataLoader
    if use_dali_warper:
        step_data_num = len(dataloader)
        dataloader = DALIWarper(
            dataloader, 
            step_data_num=step_data_num, 
            mode=dali_mode,
            auto_reset=auto_reset
        )
    
    return dataloader
