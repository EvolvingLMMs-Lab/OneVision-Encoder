import os
import numpy as np
import random
import shutil
import torch
import torch.distributed as dist


def set_seed(seed, disable_deterministic=False):
    """Set randon seed for pytorch and numpy"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if disable_deterministic:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    else:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=True)


def update_workdir(cfg, exp_id, gpu_num):
    cfg.work_dir = os.path.join(cfg.work_dir, f"gpu{gpu_num}_id{exp_id}/")
    return cfg


def create_folder(folder_path):
    dir_name = os.path.expanduser(folder_path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name, mode=0o777, exist_ok=True)


def save_config(cfg, folder_path):
    shutil.copy2(cfg, folder_path)


def reduce_loss(loss_dict):
    # reduce loss when distributed training, only for logging
    for loss_name, loss_value in loss_dict.items():
        loss_value = loss_value.data.clone()
        dist.all_reduce(loss_value.div_(dist.get_world_size()))
        loss_dict[loss_name] = loss_value
    return loss_dict


class AverageMeter(object):
    """Computes and stores the average and current value.
    Used to compute dataset stats from mini-batches
    """

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = 0.0

    def initialize(self, val, n):
        self.val = val
        self.avg = val
        self.sum = val * n
        self.count = n
        self.initialized = True

    def update(self, val, n=1):
        if not self.initialized:
            self.initialize(val, n)
        else:
            self.add(val, n)

    def add(self, val, n):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
