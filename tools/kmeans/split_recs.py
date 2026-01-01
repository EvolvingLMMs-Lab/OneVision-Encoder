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
    Split a MXNet RecordIO file into two parts
    
    Args:
        rec_path: .rec file path
        idx_path: .idx file path
        output_prefix: Prefix for output files
        process_id: Process ID for log identification
    """
    print(f"[Process {process_id}] Starting: {os.path.basename(rec_path)}")
    start_time = time.time()
    
    try:
        # Read index file to get total record count
        with open(idx_path, 'r') as f:
            lines = f.readlines()
        total_records = len(lines)
        
        print(f"[Process {process_id}] Total records: {total_records:,}")
        
        # Calculate split point (split into two halves)
        split_point = total_records // 2
        
        # Create output files
        part1_rec = output_prefix + "_part1.rec"
        part1_idx = output_prefix + "_part1.idx"
        part2_rec = output_prefix + "_part2.rec"
        part2_idx = output_prefix + "_part2.idx"
        
        # Open original rec file (using non-indexed method, sequential read - fast)
        record = recordio.MXRecordIO(rec_path, 'r')
        
        # Create two writers (using indexed method - automatically generates index)
        writer1 = recordio.MXIndexedRecordIO(part1_idx, part1_rec, 'w')
        writer2 = recordio.MXIndexedRecordIO(part2_idx, part2_rec, 'w')
        
        # Read and split data
        i = 0
        
        while True:
            try:
                # Sequential record reading (fast)
                item = record.read()
                if item is None:
                    break
                
                if i < split_point:
                    # Write to first part
                    writer1.write_idx(i, item)
                else:
                    # Write to second part (index starts from 0)
                    writer2.write_idx(i - split_point, item)
                
                i += 1
                
                # Show progress (every 100,000 records)
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
        
        # Close all files
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
    Worker process function
    """
    while True:
        task = task_queue.get()
        if task is None:  # End signal
            break
        
        rec_file, idx_file, output_prefix = task
        split_rec_file(rec_file, idx_file, output_prefix, process_id)

def main():
    print("=" * 80)
    print("MXNet RecordIO File Splitter - Parallel Version (Fast)")
    print("=" * 80)
    print()
    
    # Data directory (modify according to your actual path)
    data_dir = "./"
    
    # Find all .rec files
    rec_files = sorted(glob.glob(os.path.join(data_dir, "coyo700m_22.rec")))
    
    print(f"Found {len(rec_files)} rec files:")
    for i, rec_file in enumerate(rec_files, 1):
        file_size_gb = os.path.getsize(rec_file) / (1024**3)
        print(f"  {i}. {os.path.basename(rec_file)} ({file_size_gb:.1f} GB)")
    print()
    
    # Prepare task list
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
    
    # Create task queue
    task_queue = Queue()
    
    # Put tasks into queue
    for task in tasks:
        task_queue.put(task)
    
    # Add end signals
    for _ in range(1):
        task_queue.put(None)
    
    # Create and start 1 process
    processes = []
    for i in range(1):
        p = Process(target=worker, args=(task_queue, i+1))
        p.start()
        processes.append(p)
    
    # Wait for all processes to complete
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
