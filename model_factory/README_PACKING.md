# LlavaViT Packing Model

> **Documentation for**: `model_factory/vit_preview_v0_packing_hf.py`  
> **Conversion Script**: `model_factory/convert_llava_vit_packing_to_hf.py`

This document describes how to use the `LlavaViTPackingModel`, a vision transformer designed for efficient variable-length sequence processing with FlashAttention support. The output of this model is **consistent with `vit_preview_v0_hf.py`** when using the same weights.

## Overview

The packing model is similar to Qwen2VL's vision encoder, accepting inputs in `[seq_len, patch_dim]` format instead of traditional `[batch, channels, height, width]`. This allows for efficient batch processing of images and videos with different sizes through sequence packing.

## Requirements

- **FlashAttention 2** is **mandatory** for this model:
  ```bash
  pip install flash-attn --no-build-isolation
  ```
- CUDA-compatible GPU
- PyTorch with CUDA support

---

## Understanding `grid_thw`

The `grid_thw` tensor specifies the **patch grid dimensions** for each image/video in the batch:

| Component | Meaning | Example |
|-----------|---------|---------|
| `t` | Number of temporal frames | `1` for images, `8` for 8-frame video |
| `h` | Number of patches in height | `height_pixels // patch_size` |
| `w` | Number of patches in width | `width_pixels // patch_size` |

**Examples:**

```python
# Single 224x224 image with patch_size=16
# h_patches = w_patches = 224 // 16 = 14
grid_thw = torch.tensor([[1, 14, 14]])  # t=1, h=14, w=14
# seq_len = 1 * 14 * 14 = 196 patches

# Single 448x448 image with patch_size=14
# h_patches = w_patches = 448 // 14 = 32
grid_thw = torch.tensor([[1, 32, 32]])  # t=1, h=32, w=32
# seq_len = 1 * 32 * 32 = 1024 patches

# 8-frame video at 224x224 with patch_size=16
grid_thw = torch.tensor([[8, 14, 14]])  # t=8, h=14, w=14
# seq_len = 8 * 14 * 14 = 1568 patches

# Batch of 2 images with different sizes
grid_thw = torch.tensor([
    [1, 14, 14],  # 224x224 image
    [1, 32, 32],  # 448x448 image
])
# total_seq_len = 196 + 1024 = 1220 patches
```

---

## Image Input

### Single Image Processing

```python
import torch
from PIL import Image
import requests

def image_to_packing_input(image: Image.Image, patch_size: int = 16):
    """Convert a PIL Image to packing model input format.
    
    Args:
        image: PIL Image (will be resized to patch-aligned size)
        patch_size: Size of each patch
        
    Returns:
        hidden_states: (seq_len, patch_dim) tensor
        grid_thw: (1, 3) tensor with [t, h, w] patches
    """
    import torchvision.transforms as T
    
    # Resize to patch-aligned size
    w, h = image.size
    new_h = (h // patch_size) * patch_size
    new_w = (w // patch_size) * patch_size
    image = image.resize((new_w, new_h))
    
    # Convert to tensor: (C, H, W)
    transform = T.Compose([
        T.ToTensor(),  # (C, H, W), values in [0, 1]
        T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                   std=[0.26862954, 0.26130258, 0.27577711]),
    ])
    pixel_tensor = transform(image)  # (3, H, W)
    
    # Calculate patch dimensions
    channels = pixel_tensor.shape[0]
    h_patches = new_h // patch_size
    w_patches = new_w // patch_size
    t_frames = 1  # Images have t=1
    
    # Reshape to patches: (C, H, W) -> (h_patches, w_patches, C, patch_size, patch_size)
    patches = pixel_tensor.view(
        channels, h_patches, patch_size, w_patches, patch_size
    )
    patches = patches.permute(1, 3, 0, 2, 4).contiguous()  # (h, w, C, pH, pW)
    
    # Flatten to (seq_len, patch_dim)
    seq_len = t_frames * h_patches * w_patches
    patch_dim = patch_size * patch_size * channels
    hidden_states = patches.view(seq_len, patch_dim)
    
    # Create grid_thw: [t, h, w]
    grid_thw = torch.tensor([[t_frames, h_patches, w_patches]], dtype=torch.long)
    
    return hidden_states, grid_thw


# Example usage
url = "https://images.cocodataset.org/val2017/000000039769.jpg"
response = requests.get(url, stream=True, timeout=10)
image = Image.open(response.raw)

hidden_states, grid_thw = image_to_packing_input(image, patch_size=16)
hidden_states = hidden_states.cuda().bfloat16()
grid_thw = grid_thw.cuda()

print(f"hidden_states shape: {hidden_states.shape}")  # (seq_len, 768)
print(f"grid_thw: {grid_thw}")  # [[1, h, w]]

with torch.no_grad():
    outputs = model(hidden_states=hidden_states, grid_thw=grid_thw)
print(f"Output shape: {outputs.last_hidden_state.shape}")
```

### Batch of Multiple Images

```python
def batch_images_to_packing_input(images: list, patch_size: int = 16):
    """Convert multiple images to a packed batch.
    
    Args:
        images: List of PIL Images (can be different sizes)
        patch_size: Size of each patch
        
    Returns:
        hidden_states: (total_seq_len, patch_dim) tensor
        grid_thw: (num_images, 3) tensor
    """
    all_hidden_states = []
    all_grid_thw = []
    
    for img in images:
        hs, grid = image_to_packing_input(img, patch_size)
        all_hidden_states.append(hs)
        all_grid_thw.append(grid)
    
    # Concatenate all patches along sequence dimension
    hidden_states = torch.cat(all_hidden_states, dim=0)  # (total_seq_len, patch_dim)
    grid_thw = torch.cat(all_grid_thw, dim=0)  # (num_images, 3)
    
    return hidden_states, grid_thw


# Example: Batch of 3 images with different sizes
images = [
    Image.new('RGB', (224, 224), color='red'),    # 14x14 = 196 patches
    Image.new('RGB', (224, 224), color='blue'),   # 14x14 = 196 patches
    Image.new('RGB', (448, 448), color='green'),  # 28x28 = 784 patches
]

hidden_states, grid_thw = batch_images_to_packing_input(images, patch_size=16)
hidden_states = hidden_states.cuda().bfloat16()
grid_thw = grid_thw.cuda()

print(f"Total seq_len: {hidden_states.shape[0]}")  # 196 + 196 + 784 = 1176
print(f"grid_thw:\n{grid_thw}")
# [[1, 14, 14],
#  [1, 14, 14],
#  [1, 28, 28]]

with torch.no_grad():
    outputs = model(hidden_states=hidden_states, grid_thw=grid_thw)
    
# outputs.last_hidden_state: (1176, hidden_size)
# outputs.pooler_output: (3, hidden_size) - one per image
```

---

## Video Input

For video input, set `t > 1` in `grid_thw` to specify the number of frames.

### Single Video Processing (8 Frames)

```python
import torch
import numpy as np

def video_to_packing_input(frames: list, patch_size: int = 16):
    """Convert video frames to packing model input format.
    
    Args:
        frames: List of PIL Images (video frames), all same size
        patch_size: Size of each patch
        
    Returns:
        hidden_states: (seq_len, patch_dim) tensor where seq_len = t * h * w
        grid_thw: (1, 3) tensor with [t, h, w] patches
    """
    import torchvision.transforms as T
    
    t_frames = len(frames)
    assert t_frames > 0, "Must provide at least one frame"
    
    # Assume all frames are the same size
    w, h = frames[0].size
    new_h = (h // patch_size) * patch_size
    new_w = (w // patch_size) * patch_size
    
    h_patches = new_h // patch_size
    w_patches = new_w // patch_size
    
    # Transform each frame
    transform = T.Compose([
        T.Resize((new_h, new_w)),
        T.ToTensor(),
        T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                   std=[0.26862954, 0.26130258, 0.27577711]),
    ])
    
    all_frame_patches = []
    
    for frame in frames:
        pixel_tensor = transform(frame)  # (3, H, W)
        channels = pixel_tensor.shape[0]
        
        # Reshape to patches: (C, H, W) -> (h_patches, w_patches, C, patch_size, patch_size)
        patches = pixel_tensor.view(
            channels, h_patches, patch_size, w_patches, patch_size
        )
        patches = patches.permute(1, 3, 0, 2, 4).contiguous()  # (h, w, C, pH, pW)
        
        # Flatten spatial dims: (h * w, patch_dim)
        patch_dim = patch_size * patch_size * channels
        frame_patches = patches.view(h_patches * w_patches, patch_dim)
        all_frame_patches.append(frame_patches)
    
    # Stack all frames: (t * h * w, patch_dim)
    hidden_states = torch.cat(all_frame_patches, dim=0)
    
    # Create grid_thw: [t, h, w]
    grid_thw = torch.tensor([[t_frames, h_patches, w_patches]], dtype=torch.long)
    
    return hidden_states, grid_thw


# Example: 8-frame video at 224x224 with patch_size=16
# Create synthetic video frames
video_frames = [
    Image.new('RGB', (224, 224), color=(i * 30, i * 20, i * 10)) 
    for i in range(8)
]

hidden_states, grid_thw = video_to_packing_input(video_frames, patch_size=16)
hidden_states = hidden_states.cuda().bfloat16()
grid_thw = grid_thw.cuda()

print(f"hidden_states shape: {hidden_states.shape}")  # (1568, 768)
print(f"grid_thw: {grid_thw}")  # [[8, 14, 14]]
# seq_len = 8 * 14 * 14 = 1568 patches

with torch.no_grad():
    outputs = model(hidden_states=hidden_states, grid_thw=grid_thw)

print(f"Output shape: {outputs.last_hidden_state.shape}")  # (1568, hidden_size)
print(f"Pooler output shape: {outputs.pooler_output.shape}")  # (1, hidden_size)
```

### Video with Different Frame Counts

```python
# Process videos with different number of frames
def batch_videos_to_packing_input(videos: list, patch_size: int = 16):
    """Convert multiple videos to a packed batch.
    
    Args:
        videos: List of videos, each video is a list of PIL Images
        patch_size: Size of each patch
        
    Returns:
        hidden_states: (total_seq_len, patch_dim) tensor
        grid_thw: (num_videos, 3) tensor
    """
    all_hidden_states = []
    all_grid_thw = []
    
    for video_frames in videos:
        hs, grid = video_to_packing_input(video_frames, patch_size)
        all_hidden_states.append(hs)
        all_grid_thw.append(grid)
    
    hidden_states = torch.cat(all_hidden_states, dim=0)
    grid_thw = torch.cat(all_grid_thw, dim=0)
    
    return hidden_states, grid_thw


# Example: Batch of 2 videos
video1 = [Image.new('RGB', (224, 224), 'red') for _ in range(8)]    # 8 frames
video2 = [Image.new('RGB', (224, 224), 'blue') for _ in range(4)]   # 4 frames

videos = [video1, video2]
hidden_states, grid_thw = batch_videos_to_packing_input(videos, patch_size=16)

print(f"grid_thw:\n{grid_thw}")
# [[8, 14, 14],   # 8 frames: 8 * 14 * 14 = 1568 patches
#  [4, 14, 14]]   # 4 frames: 4 * 14 * 14 = 784 patches
# Total: 1568 + 784 = 2352 patches
```

### Mixed Batch (Images + Videos)

```python
# You can mix images and videos in the same batch!
# Images have t=1, videos have t>1

# Image: 448x448
image = Image.new('RGB', (448, 448), 'green')
img_hs, img_grid = image_to_packing_input(image, patch_size=16)
# img_grid = [[1, 28, 28]]  # 784 patches

# Video: 8 frames at 224x224
video = [Image.new('RGB', (224, 224), 'blue') for _ in range(8)]
vid_hs, vid_grid = video_to_packing_input(video, patch_size=16)
# vid_grid = [[8, 14, 14]]  # 1568 patches

# Combine into batch
hidden_states = torch.cat([img_hs, vid_hs], dim=0).cuda().bfloat16()
grid_thw = torch.cat([img_grid, vid_grid], dim=0).cuda()

print(f"Mixed batch grid_thw:\n{grid_thw}")
# [[1, 28, 28],   # Image: 784 patches
#  [8, 14, 14]]   # Video: 1568 patches
# Total: 2352 patches

with torch.no_grad():
    outputs = model(hidden_states=hidden_states, grid_thw=grid_thw)
# pooler_output: (2, hidden_size) - one for image, one for video
```

---

## Quick Start (Basic Usage)

```python
import torch
from model_factory.vit_preview_v0_packing_hf import (
    LlavaViTPackingModel,
    LlavaViTPackingConfig,
)

# Create model configuration
config = LlavaViTPackingConfig(
    patch_size=16,           # Size of each patch
    hidden_size=768,         # Hidden dimension
    num_attention_heads=12,  # Number of attention heads
    num_hidden_layers=12,    # Number of transformer layers
    num_channels=3,          # RGB channels
)

# Create model
model = LlavaViTPackingModel(config)
model = model.cuda().bfloat16()
model.eval()

# Prepare input for a 224x224 image
patch_size = 16
in_channels = 3
h_patches, w_patches = 14, 14  # 224 // 16 = 14
t_frames = 1  # 1 for images

patch_dim = patch_size * patch_size * in_channels  # 768
seq_len = t_frames * h_patches * w_patches  # 196

# Input format: (seq_len, patch_dim)
hidden_states = torch.randn(seq_len, patch_dim, dtype=torch.bfloat16, device='cuda')
grid_thw = torch.tensor([[t_frames, h_patches, w_patches]], dtype=torch.long, device='cuda')

# Forward pass
with torch.no_grad():
    outputs = model(hidden_states=hidden_states, grid_thw=grid_thw)
    
print(f"Output shape: {outputs.last_hidden_state.shape}")  # (196, 768)
```

---

## Model Variants

Available model variants through `timm`:

```python
import timm

# Small model (6 layers, 384 hidden size, patch_size=16)
model = timm.create_model("hf_llava_vit_packing_small_ln", pretrained=False)

# Base model (12 layers, 768 hidden size, patch_size=16)
model = timm.create_model("hf_llava_vit_packing_base_ln", pretrained=False)

# Large model (24 layers, 1024 hidden size, patch_size=14)
model = timm.create_model("hf_llava_vit_packing_large_ln", pretrained=False)

# Huge model (32 layers, 1280 hidden size, patch_size=14)
model = timm.create_model("hf_llava_vit_packing_huge_ln", pretrained=False)

# Giant model (40 layers, 1536 hidden size, patch_size=14)
model = timm.create_model("hf_llava_vit_packing_giant_ln", pretrained=False)
```

## Weight Conversion

To convert weights from `vit_preview_v0_hf` to the packing model:

```bash
python model_factory/convert_llava_vit_packing_to_hf.py \
    llava_vit_large_ln \
    /path/to/backbone.pt \
    --output_dir /path/to/output
```

The converted model produces **identical outputs** to `vit_preview_v0_hf.py` when given the same input.

---

## Input/Output Format Summary

### Input

| Parameter | Shape | Description |
|-----------|-------|-------------|
| `hidden_states` | `(seq_len, patch_dim)` | Flattened pixel patches. `patch_dim = patch_size * patch_size * 3` |
| `grid_thw` | `(batch_size, 3)` | Grid dimensions `[t, h, w]` for each image/video |

### Output

| Field | Shape | Description |
|-------|-------|-------------|
| `last_hidden_state` | `(seq_len, hidden_size)` | Encoded patch representations |
| `pooler_output` | `(batch_size, hidden_size)` | Pooled representation per image/video (if `use_head=True`) |
| `hidden_states` | tuple | All layer hidden states (if `output_hidden_states=True`) |

### Sequence Length Calculation

```
seq_len = sum(t_i * h_i * w_i) for all images/videos in batch
```

### Common Configurations

| Input | Size | patch_size | grid_thw | seq_len |
|-------|------|------------|----------|---------|
| Image | 224×224 | 16 | `[1, 14, 14]` | 196 |
| Image | 448×448 | 16 | `[1, 28, 28]` | 784 |
| Image | 448×448 | 14 | `[1, 32, 32]` | 1024 |
| Video 8f | 224×224 | 16 | `[8, 14, 14]` | 1568 |
| Video 8f | 448×448 | 14 | `[8, 32, 32]` | 8192 |

---

## Key Differences from Standard ViT

1. **Input Format**: Uses `[seq_len, patch_dim]` instead of `[B, C, H, W]`
2. **FlashAttention Required**: Mandatory for efficient variable-length attention
3. **Packing**: Multiple images/videos can be packed into a single sequence
4. **RoPE**: Uses 3D rotary position embeddings (temporal + 2D spatial)
5. **grid_thw**: Explicit patch grid specification allows variable-size inputs
6. **Video Support**: Native support for video input via `t > 1` in grid_thw

## Notes

- The model uses LayerNorm (not RMSNorm) by default
- FlashAttention 2 is **required** - model will raise `ImportError` if not available
- For best performance, use `bfloat16` precision
- The pooler uses multi-head attention pooling (similar to SigLIP2)
- **Output consistency**: Results match `vit_preview_v0_hf.py` when using converted weights
