import numpy as np
import pickle
import argparse
from collections import defaultdict
from pathlib import Path
from multiprocessing import Pool

def process_block(args):
    """
    单个进程处理一部分 labels，返回部分倒排索引
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
    合并多个倒排索引
    """
    merged = defaultdict(list)
    for idx in indices_list:
        for label, rows in idx.items():
            merged[label].extend(rows)
    return dict(merged)

def create_inverted_index_parallel(input_file, label_start=None, label_end=None, num_workers=16):
    try:
        labels = np.load(input_file)
        print(f"成功加载文件 {input_file}，形状: {labels.shape}")

        if label_start is not None or label_end is not None:
            start_idx = 0 if label_start is None else label_start
            end_idx = labels.shape[1] if label_end is None else label_end
            print(f"处理标签范围: [{start_idx}:{end_idx}]")
            labels = labels[:, start_idx:end_idx]
            print(f"选择后的形状: {labels.shape}")

        N = labels.shape[0]
        block_size = (N + num_workers - 1) // num_workers  # 向上取整

        # 准备分块
        blocks = []
        for i in range(num_workers):
            s = i * block_size
            e = min((i + 1) * block_size, N)
            if s < e:
                blocks.append((labels[s:e], s))

        print(f"启动进程池，共 {num_workers} 块")
        with Pool(num_workers) as pool:
            indices_list = list(pool.map(process_block, blocks))

        print("合并结果中……")
        final_index = merge_indices(indices_list)
        return final_index

    except Exception as e:
        print(f"处理文件时出错: {e}")
        return None

def save_index(index, output_prefix):
    pickle_path = f"{output_prefix}.pkl"
    with open(pickle_path, 'wb') as f:
        pickle.dump(index, f)
    print(f"倒排索引已保存为pickle格式: {pickle_path}")

    # (可选) 保存为文本、JSON，可以解注释
    # txt_path = f"{output_prefix}.txt"
    # with open(txt_path, 'w', encoding='utf-8') as f:
    #     for label, rows in sorted(index.items()):
    #         f.write(f"{label}: {rows}\n")
    # print(f"倒排索引已保存为文本格式: {txt_path}")

    # json_path = f"{output_prefix}.json"
    # with open(json_path, 'w', encoding='utf-8') as f:
    #     json.dump(index, f, ensure_ascii=False, indent=2)
    # print(f"倒排索引已保存为JSON格式: {json_path}")

def main():
    parser = argparse.ArgumentParser(description='从npy文件创建倒排索引(16进程)')
    parser.add_argument('input_file', help='输入的npy文件路径')
    parser.add_argument('--output', '-o', default='inverted_index', help='输出文件前缀 (默认: inverted_index)')
    parser.add_argument('--label-range', '-r', default=None, help='标签列的范围，格式"起始,结束"，例如"0,10"')
    parser.add_argument('--workers', '-w', type=int, default=16, help='进程数 (默认16)')
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
            print(f"标签范围格式无效 '{args.label_range}'，使用全部标签")

    inverted_index = create_inverted_index_parallel(
        args.input_file, label_start, label_end, num_workers=args.workers
    )

    if inverted_index:
        print(f"共找到 {len(inverted_index)} 个唯一标签")
        label_stats = [(label, len(rows)) for label, rows in inverted_index.items()]
        label_stats.sort(key=lambda x: x[1], reverse=True)

        print("\n出现次数最多的10个标签:")
        for label, count in label_stats[:10]:
            print(f"标签 {label}: 出现在 {count} 行")

        save_index(inverted_index, args.output)

if __name__ == "__main__":
    main()
