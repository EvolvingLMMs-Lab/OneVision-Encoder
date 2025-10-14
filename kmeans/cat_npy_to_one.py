import numpy as np
import glob
from pprint import pprint
import sys

def merge_npy_files(file_list, output_file, axis=0):
    """
    Merge multiple .npy files along the specified axis.
    
    Args:
        file_list: List of paths to .npy files
        output_file: Path to save the merged result
        axis: Axis along which to concatenate (default: 0)
    """
    npy_list = []

    for file_path in file_list:
        npy_array = np.load(file_path)
        npy_list.append(npy_array)

    merged_array = np.concatenate(npy_list, axis=axis)
    np.save(output_file, merged_array)

# 读取list_chunk文件列表
with open(f'{sys.argv[1]}', 'r') as file:
    file_list = [line.strip() for line in file]

# 检查是否提供了axis参数
axis = 0  # 默认值
if len(sys.argv) > 2:
    axis = int(sys.argv[2])

# 合并npy文件并保存结果
merge_npy_files(file_list, f'{sys.argv[1]}_merged', axis=axis)
