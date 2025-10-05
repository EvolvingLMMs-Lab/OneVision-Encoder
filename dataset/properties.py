"""
Dataset property definitions.
数据集属性定义。

This module defines the Property class for configuring dataset parameters
and contains predefined dataset configurations.
本模块定义了用于配置数据集参数的Property类，并包含预定义的数据集配置。
"""
import os
from typing import List

from easydict import EasyDict

# Get distributed training environment variables
# 获取分布式训练环境变量
rank = int(os.getenv("RANK", "0"))        # Global rank of current process / 当前进程的全局排名
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
        # num_class: int,          # Total number of classes / 类别总数
        num_example: int,          # Total number of examples / 样本总数
        prefix: List[str],             # Data path prefix / 数据路径前缀
        name: str,               # Dataset name / 数据集名称
        # label_select: int = 0,   # Label selection index / 标签选择索引
        # label_start: int = 0,    # Label starting offset / 标签起始偏移
        num_shards: int = world_size,  # Number of data shards / 数据分片数量
        shard_id: int = rank,    # Current process shard ID / 当前进程的分片ID
        dali_type: str = "origin",  # Data loader type / 数据加载器类型
        # random_diff: int = 10,   # Random difference parameter / 随机差异参数
        pfc_type: list = ["partial_fc", ],
        
        # For video datasets
        list_mp4: str = None,
        # list_label: str = None,
        
        # for dense pfc
        # frame_scales: list = None
    ):
        """
        Initialize dataset property object.
        初始化数据集属性对象。
        """
        super(Property, self).__init__()
        self.name = name
        self.prefix = prefix
        self.num_example = num_example
        self.num_shards = num_shards
        self.shard_id = shard_id
        self.dali_type = dali_type
        self.pfc_type = pfc_type
        self.list_mp4 = list_mp4

        # Validate dali_type to ensure it's a supported type
        # 验证dali_type以确保它是受支持的类型
        valid_dali_types = [
            "origin",
            "ocr",
            "multi_res",
            "video",
            "parallel_rec",
            "decord",
            "decord_torch",
            ]
        if self.dali_type not in valid_dali_types:
            raise ValueError(f"dali_type must be one of {valid_dali_types}")

        valid_pfc_types = [
            "partial_fc",
            "parallel_softmax",
            "mask",
            "unmask"
        ]
        for pfc in self.pfc_type:
            if pfc not in valid_pfc_types:
                raise ValueError(f"pfc_type must be one of {valid_pfc_types}")
        # if self.pfc_type not in valid_pfc_types:
        #     raise ValueError(f"pfc_type must be one of {valid_pfc_types}")

    def __repr__(self) -> str:
        """
        Return a formatted string representation of properties.
        返回属性的格式化字符串表示。
        """
        msg = "\n"
        for k, v in self.items():
            k = f"{k}:"
            k = format(k, "<30")

            if hasattr(v, "__len__") and len(v) > 100:
                continue

            msg += f"{k} {v}\n"
        return msg

    def __str__(self) -> str:
        """
        String representation consistent with __repr__.
        与__repr__保持一致的字符串表示。
        """
        return self.__repr__()
