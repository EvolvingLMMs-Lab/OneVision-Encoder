import random

import nvidia.dali.backend
import torch

from .data_v2 import dali_dataloader


class ParallelDALIIterator(object):
    """
    A wrapper class for running multiple DALI data loaders in parallel
    and combining their outputs into a single batch.
    
    When one iterator reaches StopIteration, only that iterator is reset.
    """
    def __init__(
        self,
        list_prefix,    # List of rec file prefixes
        batch_size,     # Total batch size to be divided among loaders
        image_size,     # Image size for all loaders
        workers,        # Number of worker threads
        shard_id,       # Shard ID for distributed training
        num_shards,     # Number of shards for distributed training
        is_training=True,
        mean=[x * 255 for x in [0.48145466, 0.4578275, 0.40821073]],
        std=[x * 255 for x in [0.26862954, 0.26130258, 0.27577711]],
    ):
        """
        Args:
            list_prefix: List of prefixes for each data loader
            batch_size: Total batch size (must be divisible by len(list_prefix))
            image_size: Image size for all data loaders
            workers: Number of worker threads
            shard_id: Shard ID for distributed training
            num_shards: Number of shards for distributed training
            is_training: Whether in training mode
            mean: Mean values for normalization
            std: Standard deviation values for normalization
        """
        self.list_prefix = list_prefix
        
        # Check if batch_size is divisible by the number of prefixes
        assert batch_size % len(list_prefix) == 0, f"batch_size ({batch_size}) must be divisible by the number of prefixes ({len(list_prefix)})"
        
        # Calculate per-loader batch size
        self.per_loader_batch_size = batch_size // len(list_prefix)
        self.total_batch_size = batch_size
        
        self.image_size = image_size
        self.workers = workers
        self.shard_id = shard_id
        self.num_shards = num_shards
        self.is_training = is_training
        self.mean = mean
        self.std = std
        
        self.dali_iters = []
        
        # Initialize all iterators
        self.create_all_iterators()
    
    def create_all_iterators(self):
        """Create all DALI iterators"""
        # Clear existing iterators
        for iter_ in self.dali_iters:
            del iter_
        
        self.dali_iters = []
        nvidia.dali.backend.ReleaseUnusedMemory()
        
        # Create a new iterator for each prefix
        for prefix in self.list_prefix:
            iter_ = dali_dataloader(
                prefix=prefix,
                batch_size=self.per_loader_batch_size,  # Use the divided batch size
                image_size=self.image_size,
                workers=self.workers,
                is_training=self.is_training,
                mean=self.mean,
                std=self.std,
                seed=random.randint(0, 8096),
                shard_id=self.shard_id,
                num_shards=self.num_shards
            )
            self.dali_iters.append(iter_)

    def __iter__(self):
        return self
    
    def __next__(self):
        all_data = []
        all_labels = []
        
        # Iterate over each data loader
        for i, iter_ in enumerate(self.dali_iters):
            try:
                # Get the next batch from this iterator
                data, label = next(iter_)
                all_data.append(data)
                all_labels.append(label)
            except StopIteration:
                # Just reset this specific iterator

                self.dali_iters[i].reset()
                # Try again with the reset iterator
                data, label = next(self.dali_iters[i])
                all_data.append(data)
                all_labels.append(label)
        
        # Combine data from all iterators
        return torch.cat(all_data, dim=0), torch.cat(all_labels, dim=0)
    
    def reset(self):
        """Reset all iterators"""
        self.create_all_iterators()
