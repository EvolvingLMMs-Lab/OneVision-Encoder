# Feature Extraction Performance Tuning Guide

## Quick Reference

### Default Optimizations (Auto-Enabled)
These optimizations are automatically enabled based on your hardware:

✅ **Batch Size: 64** (increased from 20)
✅ **Video Decoder Threads: 64** (increased from 8)
✅ **Prefetch Queue: 8 clips** (new feature)
✅ **Mixed Precision (FP16)**: Auto-enabled on V100+ GPUs
✅ **TF32**: Auto-enabled on A100+ GPUs
✅ **torch.compile**: Auto-enabled if PyTorch 2.0+

### Performance Tuning Parameters

If you encounter issues or want to fine-tune, modify these constants in `feat_extraction.py`:

#### 1. Batch Size (Line ~577)
```python
BATCH_CLIPS = 64  # Default: 64, Range: 16-128
```
- **Larger values** = Better GPU utilization, more GPU memory
- **Smaller values** = Less GPU memory, may fix OOM errors
- Recommended: 64 for A100 40GB, 32 for smaller GPUs

#### 2. CPU Threads (Line ~578)
```python
NUM_CPU = 64  # Default: 64, Range: 8-128
```
- **More threads** = Faster video decoding
- Should match your available CPU threads
- Recommended: 32-64 for typical servers

#### 3. Prefetch Queue Size (Line ~622)
```python
prefetcher = ClipPrefetcher(vr, clip_starts, transform, prefetch_size=8)
```
- **Larger values** = More overlap, more memory
- **Smaller values** = Less memory, less overlap
- Recommended: 4-16 depending on RAM

### Troubleshooting

#### Out of Memory (OOM) Error
```python
# Reduce batch size
BATCH_CLIPS = 32  # or 48
```

#### Slow Video Decoding
```python
# Increase CPU threads (if you have more cores)
NUM_CPU = 96  # or 128
```

#### Mixed Precision Errors
```python
# In extract_feature(), force FP32:
use_amp = False  # Line ~562
```

#### Compilation Errors
The code will automatically fall back if torch.compile fails.
To disable manually, comment out lines ~564-570.

### Expected Performance

#### Before Optimization
- **Time**: ~100s per video (226 clips)
- **Breakdown**: 84s forward, 7.3s decode, 8.7s preprocess

#### After Optimization
- **Expected Time**: ~10-15s per video (6-10x faster)
- **Breakdown**: Most time in forward pass (optimized), decode/preprocess overlapped

#### Per-Optimization Impact
| Optimization | Expected Speedup |
|-------------|------------------|
| Batch size (20→64) | 3.2x |
| Mixed precision | 1.5-2x |
| Threading + prefetch | 1.5x |
| TF32 + non-blocking | 1.2x |
| **Total** | **8-15x** |

### Validation Commands

Check your improvements with timing output:

```bash
# Look for these in output:
[R0] Enabled TF32 for faster matrix operations on Ampere GPU
[R0] Model compiled with torch.compile for faster inference
[R0] {video}.mp4 耗时 15.2s | 初始化 2.1s | 前向 12.8s | 保存 0.01s | clips=226
```

### Hardware-Specific Recommendations

#### 8x A100 (40GB)
```python
BATCH_CLIPS = 64
NUM_CPU = 64
prefetch_size = 8
# All optimizations enabled automatically
```

#### 8x V100 (16GB)
```python
BATCH_CLIPS = 32  # Reduce for 16GB
NUM_CPU = 64
prefetch_size = 6
# FP16 enabled, TF32 not available
```

#### 4x A100 (80GB)
```python
BATCH_CLIPS = 96  # Can increase for 80GB
NUM_CPU = 64
prefetch_size = 12
# All optimizations enabled automatically
```

### Advanced: Custom Model Optimization

If using a custom model, ensure compatibility:

1. **Mixed Precision**: Model should support FP16
2. **torch.compile**: Model should be torch.compile-compatible
3. **Batch Processing**: Model should handle batched inputs

Disable specific features if incompatible:
```python
use_amp = False  # Disable FP16
# Comment out torch.compile block (lines 564-570)
```

### Monitoring Performance

Track these metrics during execution:

1. **GPU Utilization**: Should be 90-100% (use `nvidia-smi`)
2. **CPU Utilization**: Should use all allocated threads
3. **Memory**: GPU memory should be near capacity (indicates full batches)
4. **Time per Video**: Should decrease significantly

### Support

If performance is still slower than expected:
1. Check GPU utilization with `nvidia-smi -l 1`
2. Check CPU usage with `htop` or `top`
3. Verify batch size isn't causing OOM
4. Ensure prefetching is working (no gaps in GPU utilization)
5. Check video file formats (some formats decode slower)

For issues or questions, refer to `OPTIMIZATION_SUMMARY.md` for detailed technical information.
