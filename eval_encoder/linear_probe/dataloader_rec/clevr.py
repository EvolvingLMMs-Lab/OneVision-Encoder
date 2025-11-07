from typing import Tuple
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import CLEVRClassification
from .utils import dataset_root

root = f"{os.environ['LINEAR_PROBE_ROOT']}"
num_example_train = 2000
num_example_test = 500
num_classes = 8


def get_loader_train(
    transform, batch_size, num_workers, seed
) -> Tuple[Dataset, DataLoader]:
    dataset_all = CLEVRClassification(root, download = False, split = 'train', transform = transform)
    dataset_train, other = random_split(
        dataset_all,
        lengths=[num_example_train, 68000],
        generator=torch.Generator().manual_seed(seed))
    return (dataset_train, None)


def get_loader_test(
    transform, batch_size, num_workers, seed
) -> Tuple[Dataset, DataLoader]:
    dataset_all = CLEVRClassification(root, download = False, split = 'val', transform = transform)
    dataset_test, other = random_split(
        dataset_all,
        lengths = [num_example_test, 14500],
        generator = torch.Generator().manual_seed(seed))
    return (dataset_test, None)

