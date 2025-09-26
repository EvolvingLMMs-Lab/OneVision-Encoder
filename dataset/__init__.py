"""
Dataset initialization module.
数据集初始化模块。

This module automatically imports all dataset implementations 
that follow the naming convention 'dataset_*.py'.
本模块自动导入所有遵循'dataset_*.py'命名约定的数据集实现。
"""
import importlib
import os
from .properties import Property

# Import the dataset registry for registering dataset implementations
# 导入数据集注册表，用于注册数据集实现
from .registry import DATASET_REGISTRY

# Get the current directory path to locate dataset modules
# 获取当前目录路径，用于定位数据集模块
current_directory = os.path.dirname(__file__)

# Dynamically import all dataset modules in the current directory
# 动态导入当前目录中的所有数据集模块
for filename in os.listdir(current_directory):
    if filename.startswith('dataset_') and filename.endswith('.py'):
        module_name = filename[:-3]  # Remove .py extension / 移除.py扩展名
        try:
            # Attempt to import the module so it can register itself
            # 尝试导入模块，使其能够自行注册
            importlib.import_module(f".{module_name}", __package__)
        except Exception as e:
            # Log any import errors for debugging
            # 记录任何导入错误以便调试
            print(f"Error importing module {module_name}: {e}")
