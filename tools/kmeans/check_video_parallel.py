#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Video validation with checkpoint resume.
Usage: python validate_videos.py --input mp4_list.txt --output output_dir
"""

import argparse
import json
import multiprocessing as mp
import os
import pickle
import time
from pathlib import Path
from typing import List, Tuple

import decord
from decord import VideoReader, cpu


def validate_video(video_path: str, decord_threads: int = 2) -> Tuple[str, bool, str]:
    """Validate single video. Returns (path, is_valid, error_msg)."""
    if not os.path.isfile(video_path):
        return video_path, False, "FileNotFound"
    
    if os.path.getsize(video_path) == 0:
        return video_path, False, "EmptyFile"
    
    try:
        vr = VideoReader(video_path, num_threads=decord_threads, ctx=cpu(0))
        if len(vr) == 0:
            return video_path, False, "NoFrames"
        
        # Test decode a few frames
        n = len(vr)
        test_indices = [0, n//4, n//2, 3*n//4, n-1]
        frames = vr.get_batch(test_indices).asnumpy()
        
        if frames.shape[0] != len(test_indices):
            return video_path, False, "DecodeError"
        
        return video_path, True, ""
    
    except Exception as e:
        error_type = "CodecError" if "codec" in str(e).lower() else "DecodeError"
        return video_path, False, error_type


def worker(args: Tuple[List[str], int]) -> List[Tuple[str, bool, str]]:
    """Worker function for multiprocessing."""
    videos, threads = args
    return [validate_video(v, threads) for v in videos]


class VideoValidator:
    """Video validation manager with checkpointing."""
    
    def __init__(self, input_file: str, output_dir: str, batch_size: int = 100000,
                 num_processes: int = 32, decord_threads: int = 2):
        self.input_file = input_file
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        self.num_processes = num_processes
        self.decord_threads = decord_threads
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_file = self.output_dir / "checkpoint.pkl"
        self.valid_file = self.output_dir / "valid_videos.txt"
        self.invalid_file = self.output_dir / "invalid_videos.txt"
        self.error_log = self.output_dir / "errors.jsonl"
    
    def load_videos(self) -> List[str]:
        """Load video list from input file."""
        with open(self.input_file, 'r') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
    def load_checkpoint(self):
        """Load checkpoint if exists."""
        if not self.checkpoint_file.exists():
            return None, [], []
        
        with open(self.checkpoint_file, 'rb') as f:
            state = pickle.load(f)
        
        print(f"✓ Checkpoint found: {state['processed']}/{state['total']} processed")
        if input("Resume? (y/n): ").lower() == 'y':
            return state['processed'], state['valid'], state['invalid']
        return None, [], []
    
    def save_checkpoint(self, processed: int, total: int, valid: List[str], invalid: List[str]):
        """Save checkpoint."""
        with open(self.checkpoint_file, 'wb') as f:
            pickle.dump({'processed': processed, 'total': total, 'valid': valid, 'invalid': invalid}, f)
        
        with open(self.valid_file, 'w') as f:
            f.write('\n'.join(valid) + '\n')
        
        with open(self.invalid_file, 'w') as f:
            f.write('\n'.join(invalid) + '\n')
    
    def process_batch(self, videos: List[str]) -> List[Tuple[str, bool, str]]:
        """Process batch with multiprocessing."""
        chunk_size = max(1, len(videos) // self.num_processes)
        chunks = [(videos[i:i+chunk_size], self.decord_threads) 
                  for i in range(0, len(videos), chunk_size)]
        
        with mp.Pool(self.num_processes) as pool:
            results = pool.map(worker, chunks)
        
        return [item for chunk in results for item in chunk]
    
    def run(self):
        """Run validation."""
        print("="*60)
        print("Video Validation")
        print(f"Input: {self.input_file}")
        print(f"Output: {self.output_dir}")
        print(f"Batch: {self.batch_size:,} | Processes: {self.num_processes} | Threads: {self.decord_threads}")
        print("="*60)
        
        # Load videos
        all_videos = self.load_videos()
        print(f"Total videos: {len(all_videos):,}")
        
        # Load checkpoint
        start_idx, valid_list, invalid_list = self.load_checkpoint()
        if start_idx is None:
            start_idx, valid_list, invalid_list = 0, [], []
        
        remaining = all_videos[start_idx:]
        if not remaining:
            print("✓ All done!")
            return
        
        print(f"Starting from video #{start_idx+1}")
        print("-"*60)
        
        start_time = time.time()
        
        try:
            for i in range(0, len(remaining), self.batch_size):
                batch = remaining[i:i+self.batch_size]
                batch_num = (start_idx + i) // self.batch_size + 1
                
                print(f"\n[Batch {batch_num}] Videos {start_idx+i+1} to {start_idx+i+len(batch)}")
                
                t0 = time.time()
                results = self.process_batch(batch)
                
                # Update lists
                for path, is_valid, error in results:
                    if is_valid:
                        valid_list.append(path)
                    else:
                        invalid_list.append(path)
                        with open(self.error_log, 'a') as f:
                            f.write(json.dumps({'path': path, 'error': error}) + '\n')
                
                # Stats
                valid_cnt = sum(1 for _, v, _ in results if v)
                elapsed = time.time() - t0
                processed = start_idx + i + len(batch)
                progress = processed / len(all_videos) * 100
                
                print(f"✓ {elapsed:.1f}s | Valid: {valid_cnt}/{len(batch)} ({valid_cnt/len(batch)*100:.1f}%)")
                print(f"  Progress: {processed:,}/{len(all_videos):,} ({progress:.1f}%)")
                print(f"  Speed: {len(batch)/elapsed:.1f} videos/s")
                
                # Save checkpoint
                self.save_checkpoint(processed, len(all_videos), valid_list, invalid_list)
                
                # ETA
                total_elapsed = time.time() - start_time
                eta = (total_elapsed / processed) * (len(all_videos) - processed)
                print(f"  ETA: {eta/3600:.1f}h")
        
        except KeyboardInterrupt:
            print("\n\n⚠ Interrupted! Checkpoint saved.")
            return
        
        # Done
        total_time = time.time() - start_time
        print("\n" + "="*60)
        print("✓ DONE!")
        print(f"Valid: {len(valid_list):,} ({len(valid_list)/len(all_videos)*100:.1f}%)")
        print(f"Invalid: {len(invalid_list):,} ({len(invalid_list)/len(all_videos)*100:.1f}%)")
        print(f"Time: {total_time/3600:.1f}h | Speed: {len(all_videos)/total_time:.1f} videos/s")
        print(f"\nOutputs:")
        print(f"  - {self.valid_file}")
        print(f"  - {self.invalid_file}")
        print(f"  - {self.error_log}")
        print("="*60)
        
        # Cleanup
        self.checkpoint_file.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Video validation with checkpoint")
    parser.add_argument('--input', required=True, help='Input video list file')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--batch-size', type=int, default=100000, help='Batch size (default: 100000)')
    parser.add_argument('--num-processes', type=int, default=32, help='Processes (default: 32)')
    parser.add_argument('--decord-threads', type=int, default=2, help='Decord threads (default: 2)')
    
    args = parser.parse_args()
    
    validator = VideoValidator(
        input_file=args.input,
        output_dir=args.output,
        batch_size=args.batch_size,
        num_processes=args.num_processes,
        decord_threads=args.decord_threads
    )
    
    validator.run()


if __name__ == '__main__':
    main()
