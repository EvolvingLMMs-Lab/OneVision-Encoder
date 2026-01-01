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

# Read list_chunk file list
with open(f'{sys.argv[1]}', 'r') as file:
    file_list = [line.strip() for line in file]

# Check if axis parameter is provided
axis = 0  # default value
if len(sys.argv) > 2:
    axis = int(sys.argv[2])

# Merge npy files and save result
merge_npy_files(file_list, f'{sys.argv[1]}_merged', axis=axis)
