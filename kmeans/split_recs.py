import numpy as np
np.bool = np.bool_
import mxnet as mx
from mxnet import recordio
import os
import glob
from multiprocessing import Process, Queue
import time

def split_rec_file(rec_path, idx_path, output_prefix, process_id):
    """
    将一个 MXNet RecordIO 文件分成两部分
    
    Args:
        rec_path: .rec 文件路径
        idx_path: .idx 文件路径
        output_prefix: 输出文件的前缀
        process_id: 进程ID，用于日志标识
    """
    print(f"[Process {process_id}] Starting: {os.path.basename(rec_path)}")
    start_time = time.time()
    
    try:
        # 读取索引文件获取总记录数
        with open(idx_path, 'r') as f:
            lines = f.readlines()
        total_records = len(lines)
        
        print(f"[Process {process_id}] Total records: {total_records:,}")
        
        # 计算分割点（分成两半）
        split_point = total_records // 2
        
        # 创建输出文件
        part1_rec = output_prefix + "_part1.rec"
        part1_idx = output_prefix + "_part1.idx"
        part2_rec = output_prefix + "_part2.rec"
        part2_idx = output_prefix + "_part2.idx"
        
        # 打开原始 rec 文件（使用非索引方式，顺序读取 - 快速）
        record = recordio.MXRecordIO(rec_path, 'r')
        
        # 创建两个写入器（使用索引方式 - 自动生成索引）
        writer1 = recordio.MXIndexedRecordIO(part1_idx, part1_rec, 'w')
        writer2 = recordio.MXIndexedRecordIO(part2_idx, part2_rec, 'w')
        
        # 读取并分割数据
        i = 0
        
        while True:
            try:
                # 顺序读取记录（快速）
                item = record.read()
                if item is None:
                    break
                
                if i < split_point:
                    # 写入第一部分
                    writer1.write_idx(i, item)
                else:
                    # 写入第二部分（索引从0开始）
                    writer2.write_idx(i - split_point, item)
                
                i += 1
                
                # 显示进度（每10万条）
                if i % 100000 == 0:
                    elapsed = time.time() - start_time
                    progress = i * 100 / total_records
                    records_per_sec = i / elapsed
                    eta = (total_records - i) / records_per_sec if records_per_sec > 0 else 0
                    print(f"[Process {process_id}] Progress: {i:,}/{total_records:,} ({progress:.1f}%) - "
                          f"Speed: {records_per_sec:.0f} rec/s, Elapsed: {elapsed/60:.1f}min, ETA: {eta/60:.1f}min")
                    
            except Exception as e:
                print(f"[Process {process_id}] Error reading record {i}: {e}")
                break
        
        # 关闭所有文件
        record.close()
        writer1.close()
        writer2.close()
        
        elapsed = time.time() - start_time
        print(f"[Process {process_id}] ✓ COMPLETED in {elapsed/3600:.2f} hours ({elapsed/60:.1f} minutes)")
        print(f"[Process {process_id}]   Part 1: {os.path.basename(part1_rec)} ({split_point:,} records)")
        print(f"[Process {process_id}]   Part 2: {os.path.basename(part2_rec)} ({i - split_point:,} records)")
        
        return True
        
    except Exception as e:
        print(f"[Process {process_id}] ✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def worker(task_queue, process_id):
    """
    工作进程函数
    """
    while True:
        task = task_queue.get()
        if task is None:  # 结束信号
            break
        
        rec_file, idx_file, output_prefix = task
        split_rec_file(rec_file, idx_file, output_prefix, process_id)

def main():
    print("=" * 80)
    print("MXNet RecordIO File Splitter - Parallel Version (Fast)")
    print("=" * 80)
    print()
    
    # 数据目录（根据你的实际路径修改）
    data_dir = "./"
    
    # 查找所有的 .rec 文件
    rec_files = sorted(glob.glob(os.path.join(data_dir, "coyo700m_22.rec")))
    
    print(f"Found {len(rec_files)} rec files:")
    for i, rec_file in enumerate(rec_files, 1):
        file_size_gb = os.path.getsize(rec_file) / (1024**3)
        print(f"  {i}. {os.path.basename(rec_file)} ({file_size_gb:.1f} GB)")
    print()
    
    # 准备任务列表
    tasks = []
    for rec_file in rec_files:
        idx_file = rec_file.replace('.rec', '.idx')
        
        if not os.path.exists(idx_file):
            print(f"Warning: Index file not found for {rec_file}, skipping...")
            continue
        
        output_prefix = rec_file.replace('.rec', '')
        tasks.append((rec_file, idx_file, output_prefix))
    
    if len(tasks) != 1:
        print(f"Warning: Expected 8 rec files, found {len(tasks)}")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    print(f"Starting parallel processing with 8 processes...")
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print()
    
    start_time = time.time()
    
    # 创建任务队列
    task_queue = Queue()
    
    # 将任务放入队列
    for task in tasks:
        task_queue.put(task)
    
    # 添加结束信号
    for _ in range(1):
        task_queue.put(None)
    
    # 创建并启动1个进程
    processes = []
    for i in range(1):
        p = Process(target=worker, args=(task_queue, i+1))
        p.start()
        processes.append(p)
    
    # 等待所有进程完成
    for p in processes:
        p.join()
    
    total_time = time.time() - start_time
    
    print()
    print("=" * 80)
    print(f"All tasks completed!")
    print(f"Total time: {total_time/3600:.2f} hours ({total_time/60:.1f} minutes)")
    print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

if __name__ == "__main__":
    main()
