import numpy as np
import os
import multiprocessing as mp
from tqdm import tqdm
from sklearn.preprocessing import normalize

def process_npy_file(args):
    """Process a single NPY file, input data shape should be [num_features, num_frames, dims]"""
    file_path, dims, frames_out, output_dir = args
    try:
        data = np.load(file_path)
        if not isinstance(data, np.ndarray):
            print(f"File {file_path} is not a numpy array after loading, skipping")
            return file_path, False

        if data.ndim != 3:
            print(f"File {file_path} has shape {data.shape}, not 3D [num_features, num_frames, dims], skipping")
            return file_path, False

        num_features, num_frames, D = data.shape

        # Select the number of dimensions to use
        if dims is None:
            use_dims = D
        else:
            if dims <= 0:
                print(f"File {file_path}: --dims={dims} is invalid, must be positive, skipping")
                return file_path, False
            if dims > D:
                print(f"File {file_path}: dims({dims}) greater than feature dimension({D}), will truncate to {D}")
                use_dims = D
            else:
                use_dims = dims

        # Extract the first use_dims dimensions
        processed = data[:, :, :use_dims]  # [N, T, use_dims]

        # Frame-wise normalization (L2 normalize the last dimension feature vector)
        flat = processed.reshape(-1, use_dims)  # [(N*T), use_dims]
        flat = normalize(flat, axis=1)
        processed = flat.reshape(num_features, num_frames, use_dims)

        # Validate frames_out (target output frame count)
        if not isinstance(frames_out, int) or frames_out <= 0:
            print(f"File {file_path}: frames_out={frames_out} is invalid, must be a positive integer, skipping")
            return file_path, False
        if frames_out > num_frames:
            print(f"File {file_path}: frames_out({frames_out}) must be less than num_frames({num_frames}), skipping")
            return file_path, False

        # Require divisibility, otherwise error
        if num_frames % frames_out != 0:
            print(f"File {file_path}: num_frames({num_frames}) is not divisible by frames_out({frames_out}), skipping")
            return file_path, False

        # Calculate step and uniformly sample frames by step (equivalent to slice [::step]), ensure output frame count is frames_out
        step = num_frames // frames_out
        processed = processed[:, ::step, :]  # [N, frames_out, use_dims]
        if processed.shape[1] != frames_out:
            print(f"File {file_path}: frame count after sampling({processed.shape[1]}) != expected({frames_out}), skipping")
            return file_path, False

        # Output 2D array: (N, frames_out * use_dims)
        out2d = processed.reshape(-1, frames_out * use_dims)
        # Normalize again (optional)
        out2d = normalize(out2d, axis=1)

        # Save to output directory
        os.makedirs(output_dir, exist_ok=True)
        base_filename = os.path.basename(file_path)
        if base_filename.lower().endswith(".npy"):
            base_filename = base_filename[:-4]
        output_path = os.path.join(output_dir, f"processed_{base_filename}_frames{frames_out}.npy")

        np.save(output_path, out2d)
        return file_path, True

    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        return file_path, False

def collect_npy_files_from_dir(input_dir, recursive=False):
    files = []
    if recursive:
        for root, _, filenames in os.walk(input_dir):
            for fn in filenames:
                if fn.lower().endswith(".npy"):
                    files.append(os.path.join(root, fn))
    else:
        for fn in os.listdir(input_dir):
            fp = os.path.join(input_dir, fn)
            if os.path.isfile(fp) and fn.lower().endswith(".npy"):
                files.append(fp)
    return files

def collect_npy_files(input_path, recursive=False):
    """input_path can be a directory or list file"""
    if os.path.isdir(input_path):
        return collect_npy_files_from_dir(input_path, recursive=recursive)
    if os.path.isfile(input_path):
        # Treat as list file
        files = []
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                p = line.strip()
                if not p:
                    continue
                if not p.lower().endswith(".npy"):
                    continue
                if not os.path.isabs(p):
                    # Relative paths are based on the list file's directory
                    p = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(input_path)), p))
                if os.path.isfile(p):
                    files.append(p)
                else:
                    print(f"Path in list does not exist, ignored: {p}")
        return files
    print(f"Input path is neither a directory nor a file: {input_path}")
    return []

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Process NPY feature files (shape [num_features, num_frames, dims]); supports directory or list file input; uniformly sample frames based on target output frame count")
    parser.add_argument("--input", type=str, required=True, help="Input source: directory (containing NPY) or txt list file containing NPY file paths")
    parser.add_argument("--recursive", action="store_true", help="If --input is a directory, whether to recursively search subdirectories for NPY files")
    parser.add_argument("--num_processes", type=int, default=64, help="Number of processes to use (max 64, default is the smaller of CPU core count and 64)")
    parser.add_argument("--dims", type=int, default=512, help="Number of feature dimensions to extract (truncate to first dims along last dimension; default keeps all)")
    parser.add_argument("--frames_out", type=int, default=8, help="Target output frame count, requires 0 < frames_out < num_frames and num_frames divisible by frames_out")
    parser.add_argument("--output_suffix", type=str, default=None, help="Output directory suffix (default: _processed_dim_{dims or 'all'}_frames_{frames_out})")
    args = parser.parse_args()

    # Collect files
    npy_files = collect_npy_files(args.input, recursive=args.recursive)
    if not npy_files:
        print("No NPY files found")
        return
    npy_files = sorted(npy_files)
    print("Starting to process NPY files...")
    print(f"NPY files to process: {len(npy_files)}")
    print(f"dims={'all' if args.dims is None else args.dims}, frames_out={args.frames_out}")

    # Output directory (append suffix to input directory or list file's parent directory)
    if args.output_suffix is None:
        suffix = f"_processed_dim_{args.dims if args.dims is not None else 'all'}_frames_{args.frames_out}"
    else:
        suffix = args.output_suffix

    if os.path.isdir(args.input):
        base_dir = args.input.rstrip('/\\')
        output_dir = base_dir + suffix
    else:
        # List file's parent directory + suffix
        parent_dir = os.path.dirname(os.path.abspath(args.input)).rstrip('/\\')
        output_dir = parent_dir + suffix

    print(f"Output directory: {output_dir}")

    # Process count limit max 64
    cpu_cnt = mp.cpu_count()
    if args.num_processes is None or args.num_processes <= 0:
        num_processes = min(cpu_cnt, 64)
    else:
        num_processes = min(args.num_processes, 64)
    print(f"Using {num_processes} processes for parallel processing (CPU:{cpu_cnt}, limit:64)")

    # Package arguments
    process_args = [(file_path, args.dims, args.frames_out, output_dir) for file_path in npy_files]

    # Parallel processing
    with mp.Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap(process_npy_file, process_args),
            total=len(npy_files),
            desc="Processing NPY files"
        ))

    # Statistics
    success_files = [fp for fp, ok in results if ok]
    print(f"Successfully processed {len(success_files)}/{len(npy_files)} files")
    print("Processing complete!")

if __name__ == "__main__":
    main()