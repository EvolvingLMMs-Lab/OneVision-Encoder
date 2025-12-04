# Feature Extraction and Verification Tool

这个工具用于从packing和非packing的HuggingFace ViT模型中提取倒数第二层特征，并自动进行一致性验证。

This tool extracts second-to-last layer features from both packing and non-packing HuggingFace ViT models and automatically performs consistency verification.

## Files

- `extract_features.py` - 特征提取和验证工具（二合一）/ Feature extraction and verification tool (combined)
- `verify_features.py` - 独立验证工具（可选）/ Standalone verification tool (optional)

## Requirements

```bash
pip install torch torchvision numpy pillow opencv-python transformers
```

## Usage

### Extract Features and Verify Consistency / 提取特征并验证一致性

**推荐用法 / Recommended Usage:**

```bash
python model_factory/extract_features.py \
    --hf_model_path /path/to/hf_model \
    --packing_model_path /path/to/packing_model \
    --image1 1.jpg \
    --image2 2.jpg \
    --video1 1.mp4 \
    --video2 2.mp4 \
    --num_frames 8 \
    --output_dir ./features \
    --threshold 0.99
```

此命令会：
1. 从两个模型提取特征
2. 保存特征到文件
3. 自动验证一致性
4. 输出详细的对比报告

This command will:
1. Extract features from both models
2. Save features to files
3. Automatically verify consistency
4. Output detailed comparison report

**参数说明 / Parameters:**

- `--hf_model_path`: 非packing HF模型路径 / Path to non-packing HF model
- `--packing_model_path`: packing模型路径 / Path to packing model  
- `--image1`, `--image2`: 两张图片路径 / Paths to two images
- `--video1`, `--video2`: 两个视频路径 / Paths to two videos
- `--num_frames`: 视频采样帧数 (默认: 8) / Number of frames to sample (default: 8)
- `--output_dir`: 输出目录 (默认: ./features) / Output directory (default: ./features)
- `--threshold`: 相似度阈值 (默认: 0.99) / Similarity threshold (default: 0.99)

**输入要求 / Input Requirements:**

- **图片 / Images**: 使用原始分辨率，不进行缩放 / Native resolution, no resizing
- **视频 / Videos**: 均匀采样8帧，使用原始分辨率 / Uniformly sample 8 frames at native resolution
- **预处理 / Preprocessing**: 应用CLIP归一化 (mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])

**输出 / Output:**

程序会在输出目录中创建以下文件，并在终端输出验证结果：

The program creates the following files in the output directory and outputs verification results to the console:

```
features/
├── features_hf.npz         # 非packing模型的特征 / Features from non-packing model
├── features_packing.npz    # packing模型的特征 / Features from packing model
└── metadata.json           # 元数据信息 / Metadata information
```

### (Optional) Standalone Verification / （可选）独立验证

如果你已经有保存的特征文件，可以使用独立验证工具：

If you already have saved feature files, you can use the standalone verification tool:

```bash
python model_factory/verify_features.py --features_dir ./features --threshold 0.99
```

## Output Example / 输出示例

运行 `extract_features.py` 后，你会看到类似以下的输出：

After running `extract_features.py`, you will see output similar to the following:

```
=== Loading Models ===
Loading HF model from: /path/to/hf_model
✅ HF model loaded
Loading Packing model from: /path/to/packing_model
✅ Packing model loaded

=== Loading Inputs ===
Loading image: 1.jpg
  Resolution: 1920x1080
Loading image: 2.jpg
  Resolution: 1024x768
Loading video: 1.mp4
  Total frames: 240, FPS: 30.0, Resolution: 1920x1080
  Sampling frames: [0, 34, 68, 102, 137, 171, 205, 239]
Loading video: 2.mp4
  Total frames: 180, FPS: 25.0, Resolution: 1280x720
  Sampling frames: [0, 25, 51, 77, 102, 128, 154, 179]

=== Extracting features from HF model ===
Processing image 1...
  Feature shape: (1, 1377, 1024)
Processing image 2...
  Feature shape: (1, 768, 1024)
Processing video 1...
  Feature shape: (1, 1088, 1024)
Processing video 2...
  Feature shape: (1, 640, 1024)

=== Extracting features from Packing model ===
Processing image 1...
  Feature shape: (1377, 1024)
Processing image 2...
  Feature shape: (768, 1024)
Processing video 1...
  Feature shape: (1088, 1024)
Processing video 2...
  Feature shape: (640, 1024)

=== Saving Features ===
✅ HF features saved to: ./features/features_hf.npz
✅ Packing features saved to: ./features/features_packing.npz
✅ Metadata saved to: ./features/metadata.json

================================================================================
Feature Consistency Verification
================================================================================
Similarity threshold: 0.99

--- image1 ---
HF shape:      (1, 1377, 1024)
Packing shape: (1377, 1024)
Max Diff:        0.000123
Mean Diff:       0.000045
Min Cosine Sim:  0.99987654
Mean Cosine Sim: 0.99995432
Max Cosine Sim:  1.00000000
✅ image1: PASS (min cosine > 0.99)

--- image2 ---
HF shape:      (1, 768, 1024)
Packing shape: (768, 1024)
Max Diff:        0.000098
Mean Diff:       0.000034
Min Cosine Sim:  0.99991234
Mean Cosine Sim: 0.99996789
Max Cosine Sim:  1.00000000
✅ image2: PASS (min cosine > 0.99)

--- video1 ---
HF shape:      (1, 1088, 1024)
Packing shape: (1088, 1024)
Max Diff:        0.000145
Mean Diff:       0.000052
Min Cosine Sim:  0.99985432
Mean Cosine Sim: 0.99994321
Max Cosine Sim:  1.00000000
✅ video1: PASS (min cosine > 0.99)

--- video2 ---
HF shape:      (1, 640, 1024)
Packing shape: (640, 1024)
Max Diff:        0.000112
Mean Diff:       0.000041
Min Cosine Sim:  0.99989876
Mean Cosine Sim: 0.99995678
Max Cosine Sim:  1.00000000
✅ video2: PASS (min cosine > 0.99)

================================================================================
Verification Summary
================================================================================
Total comparisons: 4
Passed:            4
Failed:            0

✅ All features match! Models are consistent.

=== Feature Extraction and Verification Complete ===
All outputs saved to: ./features
```

## Technical Details / 技术细节

### Feature Extraction / 特征提取

1. **图片处理 / Image Processing:**
   - 加载原始分辨率图片 / Load image at native resolution
   - 应用CLIP归一化 / Apply CLIP normalization
   - 转换为bfloat16精度 / Convert to bfloat16 precision
   - 从`output.hidden_states[-2]`提取特征 / Extract features from `output.hidden_states[-2]`

2. **视频处理 / Video Processing:**
   - 均匀采样指定帧数 / Uniformly sample specified number of frames
   - 保持原始分辨率 / Maintain native resolution
   - 应用CLIP归一化 / Apply CLIP normalization
   - 对于HF模型：创建64帧填充视频和visible_indices / For HF model: Create 64-frame padded video with visible_indices
   - 对于packing模型：使用插值的时间位置 / For packing model: Use interpolated temporal positions
   - 从`output.hidden_states[-2]`提取特征 / Extract features from `output.hidden_states[-2]`

### Feature Format / 特征格式

保存的特征文件 (`features_hf.npz`, `features_packing.npz`) 包含：

The saved feature files (`features_hf.npz`, `features_packing.npz`) contain:

```python
{
    'image1': np.ndarray,  # Shape: [batch, seq_len, hidden_dim] or [seq_len, hidden_dim]
    'image2': np.ndarray,  # Shape: [batch, seq_len, hidden_dim] or [seq_len, hidden_dim]
    'video1': np.ndarray,  # Shape: [batch, seq_len, hidden_dim] or [seq_len, hidden_dim]
    'video2': np.ndarray,  # Shape: [batch, seq_len, hidden_dim] or [seq_len, hidden_dim]
}
```

### Metadata Format / 元数据格式

`metadata.json` 包含：/ `metadata.json` contains:

```json
{
  "hf_model_path": "...",
  "packing_model_path": "...",
  "num_frames": 8,
  "device": "cuda",
  "inputs": {
    "image1": {
      "path": "1.jpg",
      "width": 1920,
      "height": 1080,
      "shape": [1, 3, 1080, 1920]
    },
    "video1": {
      "path": "1.mp4",
      "width": 1920,
      "height": 1080,
      "total_frames": 240,
      "sampled_frames": 8,
      "frame_indices": [0, 34, 68, 102, 137, 171, 205, 239],
      "fps": 30.0,
      "shape": [1, 3, 8, 1080, 1920]
    }
  },
  "features": {
    "hf": {
      "image1": [1, 1377, 1024],
      "video1": [1, 1088, 1024]
    },
    "packing": {
      "image1": [1377, 1024],
      "video1": [1088, 1024]
    }
  }
}
```

## Example Workflow / 示例工作流程

```bash
# 1. 提取特征并验证（一步完成）/ Extract features and verify (all in one step)
python model_factory/extract_features.py \
    --hf_model_path ./models/hf_llava_vit_so400m_patch14_siglip_384 \
    --packing_model_path ./models/packing_llava_vit_so400m_patch14_siglip_384 \
    --image1 ./test_data/1.jpg \
    --image2 ./test_data/2.jpg \
    --video1 ./test_data/1.mp4 \
    --video2 ./test_data/2.mp4 \
    --num_frames 8 \
    --output_dir ./extracted_features \
    --threshold 0.99

# 程序会自动提取特征、保存文件、并验证一致性
# The program will automatically extract features, save files, and verify consistency

# 2. （可选）如果需要重新验证已保存的特征 / (Optional) Re-verify saved features if needed
python model_factory/verify_features.py \
    --features_dir ./extracted_features \
    --threshold 0.99

# 3. 在Python中加载和分析特征 / Load and analyze features in Python
python
>>> import numpy as np
>>> features_hf = np.load('./extracted_features/features_hf.npz')
>>> features_packing = np.load('./extracted_features/features_packing.npz')
>>> print(features_hf['image1'].shape)
(1, 1377, 1024)
>>> print(features_packing['image1'].shape)
(1377, 1024)
```

## Notes / 注意事项

1. **显存要求 / GPU Memory**: 处理高分辨率图片和视频需要足够的显存 / Processing high-resolution images and videos requires sufficient GPU memory

2. **视频格式 / Video Format**: 支持OpenCV可以读取的所有视频格式 / Supports all video formats readable by OpenCV (mp4, avi, mov, etc.)

3. **特征形状 / Feature Shape**: 
   - HF模型输出形状：`[batch, seq_len, hidden_dim]` / HF model output shape: `[batch, seq_len, hidden_dim]`
   - Packing模型输出形状：`[seq_len, hidden_dim]` / Packing model output shape: `[seq_len, hidden_dim]`
   - 这是正常的，验证工具会自动处理 / This is normal, verification tool handles it automatically

4. **相似度阈值 / Similarity Threshold**: 余弦相似度 > 0.99 被认为是一致的 / Cosine similarity > 0.99 is considered consistent

## Troubleshooting / 故障排除

**问题 / Problem**: `CUDA out of memory`
**解决方案 / Solution**: 
- 降低图片/视频分辨率 / Reduce image/video resolution
- 减少视频帧数 / Reduce number of frames
- 使用CPU: 设置`CUDA_VISIBLE_DEVICES=""`

**问题 / Problem**: 视频无法打开 / Cannot open video
**解决方案 / Solution**: 
- 检查视频文件是否存在 / Check if video file exists
- 确保OpenCV支持该视频编码 / Ensure OpenCV supports the video codec
- 尝试转换为常见格式如mp4 / Try converting to common format like mp4

**问题 / Problem**: 特征不匹配 / Features don't match
**解决方案 / Solution**:
- 检查两个模型是否从同一个源权重转换 / Check if both models converted from same source weights
- 检查模型版本是否一致 / Check if model versions are consistent
- 查看详细的相似度指标找出差异 / Review detailed similarity metrics to identify differences
