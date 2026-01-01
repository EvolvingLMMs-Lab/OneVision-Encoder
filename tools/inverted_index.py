import numpy as np
import pickle
import argparse
from collections import defaultdict
from pathlib import Path
from multiprocessing import Pool

def process_block(args):
    """
    Single process handles a portion of labels, returns partial inverted index
    """
    block_labels, block_start = args
    local_index = defaultdict(list)
    for i, row in enumerate(block_labels):
        row_idx = block_start + i
        row = np.unique(row)
        for label in row:
            label_key = str(label) if not isinstance(label, (int, str, float)) else label
            local_index[label_key].append(row_idx)
    return dict(local_index)

def merge_indices(indices_list):
    """
    Merge multiple inverted indices
    """
    merged = defaultdict(list)
    for idx in indices_list:
        for label, rows in idx.items():
            merged[label].extend(rows)
    return dict(merged)

def create_inverted_index_parallel(input_file, label_start=None, label_end=None, num_workers=16):
    try:
        labels = np.load(input_file)
        print(f"Successfully loaded file {input_file}, shape: {labels.shape}")

        if label_start is not None or label_end is not None:
            start_idx = 0 if label_start is None else label_start
            end_idx = labels.shape[1] if label_end is None else label_end
            print(f"Processing label range: [{start_idx}:{end_idx}]")
            labels = labels[:, start_idx:end_idx]
            print(f"Shape after selection: {labels.shape}")

        N = labels.shape[0]
        block_size = (N + num_workers - 1) // num_workers  # Round up

        # Prepare blocks
        blocks = []
        for i in range(num_workers):
            s = i * block_size
            e = min((i + 1) * block_size, N)
            if s < e:
                blocks.append((labels[s:e], s))

        print(f"Starting process pool, total {num_workers} blocks")
        with Pool(num_workers) as pool:
            indices_list = list(pool.map(process_block, blocks))

        print("Merging results...")
        final_index = merge_indices(indices_list)
        return final_index

    except Exception as e:
        print(f"Error processing file: {e}")
        return None

def save_index(index, output_prefix):
    pickle_path = f"{output_prefix}.pkl"
    with open(pickle_path, 'wb') as f:
        pickle.dump(index, f)
    print(f"Inverted index saved in pickle format: {pickle_path}")

    # (Optional) Save as text or JSON, uncomment to use
    # txt_path = f"{output_prefix}.txt"
    # with open(txt_path, 'w', encoding='utf-8') as f:
    #     for label, rows in sorted(index.items()):
    #         f.write(f"{label}: {rows}\n")
    # print(f"Inverted index saved in text format: {txt_path}")

    # json_path = f"{output_prefix}.json"
    # with open(json_path, 'w', encoding='utf-8') as f:
    #     json.dump(index, f, ensure_ascii=False, indent=2)
    # print(f"Inverted index saved in JSON format: {json_path}")

def main():
    parser = argparse.ArgumentParser(description='Create inverted index from npy file (16 processes)')
    parser.add_argument('input_file', help='Input npy file path')
    parser.add_argument('--output', '-o', default='inverted_index', help='Output file prefix (default: inverted_index)')
    parser.add_argument('--label-range', '-r', default=None, help='Label column range, format "start,end", e.g. "0,10"')
    parser.add_argument('--workers', '-w', type=int, default=16, help='Number of processes (default 16)')
    args = parser.parse_args()

    label_start = None
    label_end = None
    if args.label_range:
        try:
            parts = args.label_range.split(',')
            if len(parts) == 2:
                if parts[0]:
                    label_start = int(parts[0])
                if parts[1]:
                    label_end = int(parts[1])
        except ValueError:
            print(f"Invalid label range format '{args.label_range}', using all labels")

    inverted_index = create_inverted_index_parallel(
        args.input_file, label_start, label_end, num_workers=args.workers
    )

    if inverted_index:
        print(f"Found {len(inverted_index)} unique labels")
        label_stats = [(label, len(rows)) for label, rows in inverted_index.items()]
        label_stats.sort(key=lambda x: x[1], reverse=True)

        print("\nTop 10 most frequent labels:")
        for label, count in label_stats[:10]:
            print(f"Label {label}: appears in {count} rows")

        save_index(inverted_index, args.output)

if __name__ == "__main__":
    main()
