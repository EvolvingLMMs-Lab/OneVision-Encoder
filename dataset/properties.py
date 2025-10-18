"""
Dataset property definitions.
数据集属性定义。

This module defines the Property class for configuring dataset parameters
and contains predefined dataset configurations.
本模块定义了用于配置数据集参数的Property类，并包含预定义的数据集配置。
"""
import os
from typing import List, Optional

import numpy as np
from easydict import EasyDict

# Get distributed training environment variables
# 获取分布式训练环境变量
rank = int(os.getenv("RANK", "0"))              # Global rank of current process / 当前进程的全局排名
local_rank = int(os.getenv("LOCAL_RANK", "0"))  # Local rank within node / 节点内的本地排名
world_size = int(os.getenv("WORLD_SIZE", "1"))  # Total number of processes / 进程总数


class Property(EasyDict):
    """
    Dataset property configuration class.
    数据集属性配置类。

    Extends EasyDict to provide a convenient way to define and access dataset properties.
    扩展EasyDict，提供便捷的方式定义和访问数据集属性。
    """

    def __init__(
        self,
        # Required / 必填
        name: str,                      # Dataset name / 数据集名称
        prefixes: List[str],            # Data path prefixes / 数据路径前缀（多个）
        num_classes: int,               # Total number of classes / 类别总数
        num_examples: int,              # Total number of examples / 样本总数

        # Optional / 可选
        label_start: int = 0,           # Label starting offset / 标签起始偏移
        label_select: int = 0,          # Label selection index / 标签选择索引

        num_shards: int = world_size,   # Number of data shards / 数据分片数量
        shard_id: int = rank,           # Current process shard ID / 当前进程的分片ID

        dali_type: str = "origin",      # Data loader type / 数据加载器类型
        random_diff: int = 10,          # Random difference parameter / 随机差异参数

        pfc_types: tuple = ("partial_fc",),  # PFC variants / PFC 类型（可多选）

        # For video datasets / 视频数据集相关
        mp4_list_path: Optional[str] = None,    # Path to mp4 list file / mp4列表文件路径
        label: Optional[np.ndarray] = None,      # Label array / 标签数组
        label_list_path: Optional[str] = None,  # Path to label list file / 标签列表文件路径
    ):
        """
        Initialize dataset property object.
        初始化数据集属性对象。
        """
        super(Property, self).__init__()

        # Keep assignment order in a logical and readable sequence
        # 以下赋值顺序与参数顺序保持一致，便于阅读与 __repr__ 输出
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

        # Cast tuple to list for mutability in runtime usage
        # 运行期如需修改，转为 list 更灵活
        self.pfc_types = list(pfc_types)

        self.mp4_list_path = mp4_list_path
        self.label_list_path = label_list_path
        self.label = label

        # Validate dali_type to ensure it's a supported type
        # 验证 dali_type 以确保它是受支持的类型
        valid_dali_types = [
            "origin",
            # "ocr",
            # "multi_res",
            # "video",
            # "parallel_rec",
            "decord",
            # "decord_torch",
        ]
        if self.dali_type not in valid_dali_types:
            raise ValueError(f"dali_type must be one of {valid_dali_types}")

        # Validate pfc_types
        # 验证 pfc_types
        valid_pfc_types = [
            "partial_fc",
            # "parallel_softmax",
            # "mask",
            # "unmask"
        ]
        for pfc in self.pfc_types:
            if pfc not in valid_pfc_types:
                raise ValueError(f"pfc_types must be a subset of {valid_pfc_types}")

    def __repr__(self) -> str:
        """
        Return a formatted string representation of properties.
        返回属性的格式化字符串表示。
        """
        msg = "\n"
        for k, v in self.items():
            k = f"{k}:"
            k = format(k, "<30")

            if hasattr(v, "__len__") and not isinstance(v, (str, bytes)) and len(v) > 100:
                continue

            msg += f"{k} {v}\n"
        return msg

    def __str__(self) -> str:
        """
        String representation consistent with __repr__.
        与 __repr__ 保持一致的字符串表示。
        """
        return self.__repr__()
