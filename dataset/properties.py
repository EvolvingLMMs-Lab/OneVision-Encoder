"""Dataset property definitions.

Defines the Property class for configuring dataset parameters.
"""
import os
from typing import List, Optional

import numpy as np
from easydict import EasyDict

rank = int(os.getenv("RANK", "0"))
local_rank = int(os.getenv("LOCAL_RANK", "0"))
world_size = int(os.getenv("WORLD_SIZE", "1"))


class Property(EasyDict):
    """Dataset property configuration class extending EasyDict."""

    def __init__(
        self,
        name: str,
        prefixes: List[str],
        num_classes: int,
        num_examples: int,
        label_start: int = 0,
        label_select: int = 0,
        num_shards: int = world_size,
        shard_id: int = rank,
        dali_type: str = "origin",
        random_diff: int = 10,
        pfc_types: tuple = ("partial_fc",),
        mp4_list_path: Optional[str] = None,
        label: Optional[np.ndarray] = None,
        label_list_path: Optional[str] = None,
    ):
        super(Property, self).__init__()

        self.name = name
        self.prefixes = prefixes
        self.num_classes = num_classes
        self.num_examples = num_examples

        self.label_start = label_start
        self.label_select = label_select

        self.num_shards = num_shards
        self.shard_id = shard_id

        self.dali_type = dali_type
        self.random_diff = random_diff

        self.pfc_types = list(pfc_types)

        self.mp4_list_path = mp4_list_path
        self.label_list_path = label_list_path
        self.label = label

        valid_dali_types = [
            "origin",
            "ocr",
            "decord",
            "decord_residual",
        ]
        if self.dali_type not in valid_dali_types:
            raise ValueError(f"dali_type must be one of {valid_dali_types}")

        valid_pfc_types = ["partial_fc"]
        for pfc in self.pfc_types:
            if pfc not in valid_pfc_types:
                raise ValueError(f"pfc_types must be a subset of {valid_pfc_types}")

    def __repr__(self) -> str:
        msg = "\n"
        for k, v in self.items():
            k = f"{k}:"
            k = format(k, "<30")

            if hasattr(v, "__len__") and not isinstance(v, (str, bytes)) and len(v) > 100:
                continue

            msg += f"{k} {v}\n"
        return msg

    def __str__(self) -> str:
        return self.__repr__()
