"""Dataset initialization module.

Automatically imports all dataset implementations following the 'dataset_*.py' naming convention.
"""
import importlib
import os
from .properties import Property

from .registry import DATASET_REGISTRY

current_directory = os.path.dirname(__file__)

for filename in os.listdir(current_directory):
    if filename.startswith('dataset_') and filename.endswith('.py'):
        module_name = filename[:-3]
        try:
            importlib.import_module(f".{module_name}", __package__)
        except Exception as e:
            print(f"Error importing module {module_name}: {e}")
