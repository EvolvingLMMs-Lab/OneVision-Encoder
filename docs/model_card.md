# Model Card: OneVision Encoder Large

## Model Overview

**OneVision Encoder Large** is a vision transformer that resolves the fundamental trade-off in video understanding: processing more frames captures richer temporal information but increases computation quadratically. Using principles from HEVC video compression, it implements codec-style patch selection that identifies temporally-salient regions—areas with motion, object interactions, or semantic changes—and processes only these informative patches.

### Model Details

| Property | Value |
|----------|-------|
| **Model Type** | Vision Transformer (ViT) |
| **Architecture** | HEVC-Style Vision Transformer |
| **Hidden Size** | 1024 |
| **Intermediate Size** | 4096 |
| **Number of Layers** | 24 |
| **Number of Attention Heads** | 16 |
| **Patch Size** | 14 |
| **Image Resolution** | 448×448 (pre-trained) |
| **Video Resolution** | 224×224 with 256 tokens per frame |
| **Positional Encoding** | 3D RoPE (4:6:6 split for T:H:W) |
| **Normalization** | Layer Normalization |
| **Activation Function** | GELU |
| **Attention Implementation** | Flash Attention 2 |
| **License** | Apache 2.0 |

## Key Features

- **Unified Vision Foundation**: A single base model for consistent understanding of images, videos, and OCR.
- **Codec-Style Patch Selection**: Instead of sampling sparse frames densely (all patches from few frames), OneVision Encoder samples dense frames sparsely (important patches from many frames).
- **3D Rotary Position Embedding**: Uses a 4:6:6 split for temporal, height, and width dimensions to capture spatiotemporal relationships.
- **Global Contrastive Learning**: Trained with a 2M concept bank for better-separated semantic clusters.
- **Native Resolution Support**: Supports native resolution input without tiling or cropping.

## Unified Input Processing

OneVision Encoder uses a unified architecture to process three types of visual inputs—images, video chunks (uniform frame sampling), and codec-style sparse patches—through the same Vision Transformer backbone. The key insight is that all inputs are converted to a sequence of patch tokens with 3D position encodings, enabling a single model to handle diverse visual modalities.

### Image Processing

For single image input, the ViT processes data in the standard 4D tensor format `[B, C, H, W]`:

```
Input: [B, C, H, W] → e.g., [1, 3, 448, 448]
                           ↓
                   Patch Embedding (Conv2d with kernel=14, stride=14)
                           ↓
              Flatten: [B, num_patches, hidden_size]
                       e.g., [1, 1024, 1024] for 448×448 image
                           ↓
                   3D RoPE Position Encoding (T=1, H=32, W=32)
                           ↓
                   Transformer Encoder (24 layers)
                           ↓
              Output: [B, num_patches, hidden_size]
```

**Key points:**
- Images are internally treated as single-frame videos with `T=1`
- Position encoding uses the same 3D RoPE with temporal dimension fixed at 1
- All patches are processed (no masking), resulting in `(H/patch_size) × (W/patch_size)` tokens

### Video Chunk Sampling

For video input with uniform frame sampling, the ViT processes 5D tensor format `[B, C, T, H, W]`:

```
Input: [B, C, T, H, W] → e.g., [1, 3, 16, 224, 224]
                             ↓
                    Patch Embedding (per-frame Conv2d)
                             ↓
               Flatten: [B, T × H_patches × W_patches, hidden_size]
                        e.g., [1, 16 × 16 × 16, 1024] = [1, 4096, 1024]
                             ↓
                    Build visible_indices for temporal mapping
                             ↓
                    3D RoPE Position Encoding with frame positions
                             ↓
                    Transformer Encoder (24 layers)
                             ↓
               Output: [B, num_visible_patches, hidden_size]
```

**The `visible_indices` mechanism:**

The `visible_indices` tensor maps actual frame positions to a virtual temporal grid (e.g., 64 virtual frames), enabling proper temporal position encoding even with sparse frame sampling:

```python
# Example: 16 frames sampled from a video, mapped to 64 virtual frame positions
num_frames = 16          # Actual number of sampled frames
frame_tokens = 256       # Patches per frame (16×16 for 224×224 with patch_size=14)
target_frames = 64       # Virtual temporal grid size (model's RoPE temporal dimension)

# Map 16 actual frames to positions in the 64-frame virtual grid
frame_pos = torch.linspace(0, target_frames - 1, num_frames).long()
# frame_pos = [0, 4, 8, 12, 17, 21, 25, 29, 34, 38, 42, 46, 51, 55, 59, 63]

# Build visible_indices: each frame's patches get position encoding based on frame_pos
visible_indices = (frame_pos.unsqueeze(-1) * frame_tokens + 
                   torch.arange(frame_tokens)).reshape(1, -1)
# Shape: [1, 4096] (16 frames × 256 patches)
```

This enables the model to understand temporal relationships even when frames are not densely sampled.

### Codec-Style Input

Codec-style input is the most sophisticated processing mode, inspired by HEVC video compression. Instead of processing all patches from all frames, it selectively processes only temporally-salient patches identified through motion and residual analysis.

```
Input Video: 64 frames
        ↓
┌───────────────────────────────────────────────┐
│  HEVC Feature Extraction                      │
│  ├── Motion Vectors (MV): quarter-pel motion  │
│  └── Residuals: prediction error signals      │
└───────────────────────────────────────────────┘
        ↓
┌───────────────────────────────────────────────┐
│  Temporal Saliency Detection                  │
│  ├── MV Energy: camera-compensated motion mag │
│  ├── Residual Energy: prediction error mag    │
│  └── Fused Energy: weighted combination       │
└───────────────────────────────────────────────┘
        ↓
┌───────────────────────────────────────────────┐
│  Top-K Patch Selection                        │
│  ├── Score each patch by fused energy         │
│  ├── Select K most salient patches            │
│  └── Build sparse visible_indices             │
└───────────────────────────────────────────────┘
        ↓
┌───────────────────────────────────────────────┐
│  ViT Processing with Sparse visible_indices   │
│  ├── Input: [B, C, T, H, W] full video        │
│  ├── visible_indices: [B, K] selected patches │
│  └── Output: [B, K, hidden_size]              │
└───────────────────────────────────────────────┘
```

**Detailed Codec Processing Pipeline:**

1. **Motion Vector Analysis**: Extract motion vectors from HEVC codec at quarter-pixel precision. Apply camera motion compensation (median, similarity, or affine model) to isolate object motion from camera movement.

2. **Residual Analysis**: Extract prediction residuals that capture texture changes and fine-grained motion not captured by block-based motion compensation.

3. **Energy Fusion**: Combine MV energy and residual energy with configurable weights to produce a unified saliency map.

4. **Top-K Selection**: Rank all patches (across all frames) by their saliency scores and select the top K patches. This achieves 75-98% compression while retaining critical temporal dynamics.

5. **Sparse Processing**: The selected patches are processed by the ViT with proper 3D position encodings, enabling the model to understand the spatiotemporal context of each selected patch.

**Example codec-style inference:**

```python
# Codec-style: select 2048 most salient patches from 64 frames
# (equivalent to 8 full frames worth of tokens)
K_keep = 2048  # 256 patches/frame × 8 frames equivalent

# visible_indices are computed by the codec saliency detection
# Each index points to a specific (frame, h, w) position in the patch grid
visible_indices = compute_codec_visible_indices(
    video_path,
    K=K_keep,
    mv_compensate="similarity",  # Camera motion compensation
    patch_size=14
)

# Process with the model
outputs = model(video, visible_indices=visible_indices)
# Output: [B, 2048, 1024] - features for 2048 selected patches
```

### Comparison of Input Modes

| Mode | Input Shape | visible_indices | Output Shape | Use Case |
|------|-------------|-----------------|--------------|----------|
| **Image** | `[B, 3, H, W]` | All patches | `[B, (H/14)×(W/14), 1024]` | Single image understanding |
| **Video Chunk** | `[B, 3, T, H, W]` | Frame-mapped | `[B, T×(H/14)×(W/14), 1024]` | Uniform temporal sampling |
| **Codec-Style** | `[B, 3, T, H, W]` | Top-K salient | `[B, K, 1024]` | Efficient dense temporal |

### 3D RoPE Position Encoding

All three input modes share the same 3D Rotary Position Embedding (RoPE) with a 4:6:6 split:

- **Temporal (T)**: 4/16 of head dimension → captures frame ordering
- **Height (H)**: 6/16 of head dimension → captures vertical position  
- **Width (W)**: 6/16 of head dimension → captures horizontal position

```python
# 3D position encoding construction
head_dim = hidden_size // num_heads  # 1024 // 16 = 64
half = head_dim // 2  # 32

# Split dimensions with 4:6:6 ratio (4+6+6 = 16 units total)
unit = half // 16  # 32 // 16 = 2
t_size = 4 * unit  # 4 * 2 = 8 dims for temporal
h_size = 6 * unit  # 6 * 2 = 12 dims for height
w_size = 6 * unit  # 6 * 2 = 12 dims for width
# Total: 8 + 12 + 12 = 32 = half of head_dim

# Compute frequencies for each dimension
freqs = concat([
    freq_temporal[t_ids],   # Based on frame index
    freq_height[h_ids],     # Based on patch row
    freq_width[w_ids]       # Based on patch column
])
```

This unified position encoding allows the model to maintain consistent spatial and temporal understanding across all input modalities.

## Intended Use

### Primary Use Cases

- **Video Understanding**: Action recognition, video captioning, video question answering
- **Image Understanding**: Document understanding (DocVQA), chart understanding (ChartQA), OCR tasks
- **Vision-Language Models**: As the vision encoder backbone for multimodal large language models

### Downstream Tasks

- Video benchmarks: MVBench, VideoMME, Perception Test
- Image understanding: DocVQA, ChartQA, OCRBench
- Action recognition: SSv2, UCF101, Kinetics

## Quick Start

```python
from transformers import AutoModel, AutoImageProcessor
from PIL import Image
import torch

# Load model and preprocessor
model = AutoModel.from_pretrained(
    "lmms-lab-encoder/onevision-encoder-large",
    trust_remote_code=True,
    attn_implementation="flash_attention_2"
).to("cuda").eval()

preprocessor = AutoImageProcessor.from_pretrained(
    "lmms-lab-encoder/onevision-encoder-large",
    trust_remote_code=True
)

# Image inference: [B, C, H, W]
image = Image.open("path/to/your/image.jpg")
pixel_values = preprocessor(images=image, return_tensors="pt")["pixel_values"].to("cuda")
with torch.no_grad():
    outputs = model(pixel_values)
    # outputs.last_hidden_state: [B, num_patches, hidden_size]
    # outputs.pooler_output: [B, hidden_size]

# Video inference: [B, C, T, H, W] with visible_indices
num_frames, frame_tokens, target_frames = 16, 256, 64
frames = [Image.open(f"path/to/frame_{i}.jpg") for i in range(num_frames)]
video_pixel_values = preprocessor(images=frames, return_tensors="pt")["pixel_values"]
video = video_pixel_values.unsqueeze(0).permute(0, 2, 1, 3, 4).to("cuda")

# Build visible_indices for temporal sampling
frame_pos = torch.linspace(0, target_frames - 1, num_frames).long().cuda()
visible_indices = (frame_pos.unsqueeze(-1) * frame_tokens + torch.arange(frame_tokens).cuda()).reshape(1, -1)

with torch.no_grad():
    outputs = model(video, visible_indices=visible_indices)
```


## Evaluation Results

### Attentive Probe Results

Performance evaluated using Attentive Probe evaluation with single clip input, trained for 10 epochs across 8 action recognition datasets.

### LMM Probe Results

Training on a mixed dataset of 740K samples from LLaVA-OneVision and 800K samples from LLaVA-Video SFT. The training pipeline proceeds directly to Stage 2 fine-tuning with native-resolution strategy.

## Limitations

- The model is pre-trained at specific resolutions (448×448 for images, 224×224 for video)
- Performance may vary on domains significantly different from training data
- Video processing requires proper temporal sampling configuration

## Citation

```bibtex
@misc{onevision-encoder,
  title={OneVision Encoder: HEVC-Style Vision Transformer},
  author={EvolvingLMMs-Lab},
  year={2024},
  url={https://github.com/EvolvingLMMs-Lab/OneVision-Encoder}
}
```

## Contact

For questions and issues, please open an issue on the [GitHub repository](https://github.com/Evolvinglmms-lab/OneVision-Encoder).

## Multi-Modal Training Strategy

OneVision Encoder uses a unified training approach that simultaneously processes images, video codec-style patches, video frame sampling, and video collage within the same batch. This multi-modal training enables the model to learn robust representations across different input modalities.

### Training Batch Composition

Within each training batch, samples are divided into different processing modes:

```
                           Training Batch (bs=16)
    ┌─────────────────────────────────────────────────────────────────────┐
    │                                                                     │
    │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐     │
    │  │   Image Head    │  │   Video Head    │  │   OCR Head      │     │
    │  │   (origin)      │  │ (decord_residual│  │   (ocr)         │     │
    │  │                 │  │                 │  │                 │     │
    │  │  [B, 3, H, W]   │  │ Split by mode:  │  │  [B, 3, H, W]   │     │
    │  │                 │  │  • Codec 50%    │  │                 │     │
    │  │                 │  │  • Sampling 37.5│  │                 │     │
    │  │                 │  │  • Collage 12.5%│  │                 │     │
    │  └─────────────────┘  └─────────────────┘  └─────────────────┘     │
    │                                                                     │
    └─────────────────────────────────────────────────────────────────────┘
```

### Video Processing Modes

For video inputs, the batch is further split into three processing modes:

| Mode | Batch % | Description | Input → Output |
|------|---------|-------------|----------------|
| **Codec-Style** | 50% | Select top-K salient patches based on HEVC residual | `[n, 3, 64, 224, 224]` → `[n, 3, 8, 224, 224]` |
| **Frame Sampling** | 37.5% | Uniform temporal sampling, 1 frame per bin | `[n, 3, 64, 224, 224]` → `[n, 3, 8, 224, 224]` |
| **Collage** | 12.5% | 8 frames concatenated into tall image | `[n, 3, 64, 224, 224]` → `[n, 3, 1792, 224]` |

### Processing Pipeline

```
                        Video Input: [bs, 3, 64, 224, 224]
                                       │
                    ┌──────────────────┼──────────────────┐
                    │                  │                  │
                    ▼                  ▼                  ▼
           ┌───────────────┐  ┌───────────────┐  ┌───────────────┐
           │  Codec-Style  │  │Frame Sampling │  │   Collage     │
           │   (50%)       │  │   (37.5%)     │  │   (12.5%)     │
           └───────┬───────┘  └───────┬───────┘  └───────┬───────┘
                   │                  │                  │
                   ▼                  ▼                  ▼
           ┌───────────────┐  ┌───────────────┐  ┌───────────────┐
           │ Patchify      │  │ Sample frames │  │ Sample frames │
           │ [n,3,16384,p²]│  │ from 8 bins   │  │ from 8 bins   │
           └───────┬───────┘  └───────┬───────┘  └───────┬───────┘
                   │                  │                  │
                   ▼                  ▼                  ▼
           ┌───────────────┐  ┌───────────────┐  ┌───────────────┐
           │ Select top-K  │  │ Build indices │  │ Concat frames │
           │ by vis_idx    │  │ for 8 frames  │  │ vertically    │
           └───────┬───────┘  └───────┬───────┘  └───────┬───────┘
                   │                  │                  │
                   ▼                  ▼                  ▼
           ┌───────────────┐  ┌───────────────┐  ┌───────────────┐
           │ Unpatchify    │  │               │  │               │
           │[n,3,8,224,224]│  │[n,3,8,224,224]│  │[n,3,1792,224] │
           └───────┬───────┘  └───────┬───────┘  └───────┬───────┘
                   │                  │                  │
                   └──────────────────┼──────────────────┘
                                      │
                                      ▼
                              ┌───────────────┐
                              │  ViT Backbone │
                              │  with RoPE    │
                              └───────┬───────┘
                                      │
                                      ▼
                              [bs, hidden_size]
```

### 1. Codec-Style Processing (50% of batch)

This mode uses HEVC-extracted saliency information to select the most informative patches:

```python
# Example: bs=16, first 8 samples use codec-style
# visible_indices contains pre-computed salient patch indices from HEVC analysis

# Step 1: Use pre-computed visible_indices (sorted by saliency)
out[mask_residual] = visible_indices[mask_residual, :target_num]  # [8, 2048]

# Step 2: Patchify full video
# [8, 3, 64, 224, 224] → [8, 3, 16384, 14, 14] (64 frames × 256 patches/frame)
patches = video.view(n, C, T, Hp, patch_size, Wp, patch_size)
                .permute(0, 1, 2, 3, 5, 4, 6)
                .reshape(n, C, T * Hp * Wp, patch_size, patch_size)

# Step 3: Select top-K patches by visible_indices
selected = torch.gather(patches, 2, idx)  # [8, 3, 2048, 14, 14]

# Step 4: Unpatchify back to video format
# 2048 patches = 8 frames × 256 patches/frame
combined_head_input = selected.view(n, C, 8, Hp, Wp, patch_size, patch_size)
                              .permute(0, 1, 2, 3, 5, 4, 6)
                              .reshape(n, C, 8, H, W)  # [8, 3, 8, 224, 224]
```

### 2. Frame Sampling Processing (37.5% of batch)

This mode uniformly samples frames from temporal bins:

```python
# Example: samples 8-13 in batch use frame sampling
# Divide 64 frames into 8 bins of 8 frames each, sample 1 from each bin

# Step 1: Sample frame indices
# bins: [0-7], [8-15], [16-23], [24-31], [32-39], [40-47], [48-55], [56-63]
frames = torch.arange(8) * 8 + torch.randint(8, (nB, 8))  # [6, 8]

# Step 2: Build patch indices for all patches in selected frames
# Each frame has 256 patches
out[mask_frame_sampling] = (frames.unsqueeze(-1) * 256 + 
                            torch.arange(256)).reshape(nB, -1)  # [6, 2048]

# Step 3: Same patchify → select → unpatchify as codec-style
# Result: [6, 3, 8, 224, 224]
```

### 3. Collage Processing (12.5% of batch)

This mode concatenates sampled frames into a single tall image:

```python
# Example: samples 14-15 in batch use collage
# Sample 8 frames and concatenate vertically

# Step 1: Sample 8 frames (same bin-based sampling)
frames_idx = base + offsets  # [2, 8], values in [0, 63]

# Step 2: Gather selected frames
sel_frames = torch.gather(video, 2, idx_expand)  # [2, 3, 8, 224, 224]

# Step 3: Concatenate frames vertically
sel_frames = sel_frames.permute(0, 2, 1, 3, 4)  # [2, 8, 3, 224, 224]
grid = torch.cat([sel_frames[:, i] for i in range(8)], dim=-2)  # [2, 3, 1792, 224]

# Result: Processed as a tall image (1792 = 224 × 8)
```

### Benefits of Multi-Modal Training

1. **Unified Architecture**: Same ViT backbone handles all modalities through different input preprocessing
2. **Complementary Learning**:
   - Codec-style: Learns to focus on temporally salient regions
   - Frame sampling: Learns uniform temporal understanding
   - Collage: Learns spatial arrangement of temporal information
3. **Robust Representations**: Exposure to diverse input formats improves generalization
4. **Efficient Training**: Single forward pass processes all modalities together

### Position Encoding Consistency

All video modes use the same 3D RoPE position encoding:

```python
# visible_indices maps selected patches to positions in a 64-frame virtual grid
# This enables consistent temporal position encoding across all modes

# Codec-style: patches scattered across 64 frames
# Frame sampling: 8 complete frames with gaps
# Collage: treated as single image (T=1)

# The model learns to handle all patterns through the unified RoPE mechanism
```
