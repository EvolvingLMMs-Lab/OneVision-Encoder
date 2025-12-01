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

For video input, the source model (`vit_preview_v0`) uses a **64-frame temporal context** with uniform frame sampling. When processing 8 frames, they are interpolated to specific positions within this 64-frame context:

- **8 frames** → interpolated to positions `[0, 9, 18, 27, 36, 45, 54, 63]` in 64-frame context

This ensures temporal RoPE (Rotary Position Embedding) frequencies match between the source model and the packing model, which is **essential for output consistency**.

> **Important**: All video processing with the packing model should use interpolated temporal positions via `compute_patch_positions_with_interpolated_temporal` to match the source model's behavior.

---

## Using `patch_positions` with Interpolated Temporal Positions

To achieve output consistency with the source model, you must use `compute_patch_positions_with_interpolated_temporal` to compute patch positions with interpolated temporal values.

### The `compute_patch_positions_with_interpolated_temporal` Function

This function computes patch positions where the temporal positions are based on interpolated frame indices:

```python
def compute_patch_positions_with_interpolated_temporal(
    interpolated_indices: torch.Tensor,
    h_patches: int,
    w_patches: int,
    device: torch.device
) -> torch.Tensor:
    """
    Compute patch positions with interpolated temporal positions for RoPE.
    
    Args:
        interpolated_indices: [B, num_frames] Interpolated frame indices in 64-frame context
        h_patches: Number of patches in height dimension
        w_patches: Number of patches in width dimension
        device: Target device
    
    Returns:
        patch_positions: Tensor of shape (total_patches, 3) with [t, h, w] positions
    """
```

### Complete 8-Frame Video Example with Interpolated Positions

This is the recommended approach when you need output consistency with the source model:

```python
import torch
from PIL import Image
import torchvision.transforms as T
from model_factory.vit_preview_v0_packing_hf import LlavaViTPackingModel
from model_factory.convert_llava_vit_packing_to_hf import (
    interpolate_frame_indices,
    compute_patch_positions_with_interpolated_temporal,
)

# ============================================================
# Configuration (8 frames, 224x224, patch_size=16)
# ============================================================
device = torch.device('cuda')
num_frames = 8
frame_size = 224
patch_size = 16
target_frames = 64  # Source model's temporal context
h_patches = frame_size // patch_size  # 14
w_patches = frame_size // patch_size  # 14

# ============================================================
# Step 1: Compute Interpolated Frame Indices
# ============================================================
# For 8 frames uniformly sampled, we compute their positions in 64-frame context
frame_indices = torch.arange(num_frames).unsqueeze(0).to(device)  # [[0, 1, 2, 3, 4, 5, 6, 7]]
total_frames_tensor = torch.tensor([num_frames]).to(device)  # [8]

interpolated_indices = interpolate_frame_indices(
    frame_indices, total_frames_tensor, target_frames
)
# Result: [[0, 9, 18, 27, 36, 45, 54, 63]]
# These are the positions of 8 frames uniformly distributed in 64-frame context

print(f"Original frame indices: {frame_indices[0].tolist()}")
# [0, 1, 2, 3, 4, 5, 6, 7]

print(f"Interpolated indices (in 64-frame context): {interpolated_indices[0].tolist()}")
# [0, 9, 18, 27, 36, 45, 54, 63]

# ============================================================
# Step 2: Compute patch_positions with Interpolated Temporal Values
# ============================================================
patch_positions = compute_patch_positions_with_interpolated_temporal(
    interpolated_indices, h_patches, w_patches, device=device
)
# Shape: (num_frames * h_patches * w_patches, 3) = (8 * 14 * 14, 3) = (1568, 3)
# Each row: [t_interpolated, h, w]

print(f"patch_positions shape: {patch_positions.shape}")
# torch.Size([1568, 3])

# Visualize first few positions (first frame at t=0)
print(f"First 5 positions: {patch_positions[:5].tolist()}")
# [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3], [0, 0, 4]]

# Visualize positions at frame boundary (t=0 -> t=9)
print(f"Last position of frame 0: {patch_positions[195].tolist()}")  # [0, 13, 13]
print(f"First position of frame 1: {patch_positions[196].tolist()}")  # [9, 0, 0]  <- t=9

# ============================================================
# Step 3: Prepare Video Frames and Hidden States
# ============================================================
# Load or create video frames
transform = T.Compose([
    T.Resize((frame_size, frame_size)),
    T.ToTensor(),
    T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]),
])

# Example: Create synthetic video frames
video_frames = [Image.new('RGB', (frame_size, frame_size), color=(i*30, i*20, i*10)) 
                for i in range(num_frames)]

# Convert frames to patches
all_patches = []
for frame in video_frames:
    pixel_tensor = transform(frame)  # (3, 224, 224)
    channels = pixel_tensor.shape[0]
    patches = pixel_tensor.view(channels, h_patches, patch_size, w_patches, patch_size)
    patches = patches.permute(1, 3, 0, 2, 4).contiguous()  # (h, w, C, pH, pW)
    frame_patches = patches.view(h_patches * w_patches, patch_size * patch_size * channels)
    all_patches.append(frame_patches)

hidden_states = torch.cat(all_patches, dim=0).to(device=device, dtype=torch.bfloat16)  # (1568, 768)

# ============================================================
# Step 4: Create grid_thw and Forward Pass
# ============================================================
# grid_thw uses actual frame count (8), not the interpolated context (64)
grid_thw = torch.tensor([[num_frames, h_patches, w_patches]], dtype=torch.long, device=device)
# [[8, 14, 14]]

print(f"hidden_states shape: {hidden_states.shape}")  # torch.Size([1568, 768])
print(f"grid_thw: {grid_thw.tolist()}")  # [[8, 14, 14]]
print(f"patch_positions shape: {patch_positions.shape}")  # torch.Size([1568, 3])

# Load model and run forward pass
# Replace with your converted model path (see "Weight Conversion" section below)
model = LlavaViTPackingModel.from_pretrained("/path/to/your_model_hf_packing", torch_dtype=torch.bfloat16)
model = model.to(device).eval()

with torch.no_grad():
    outputs = model(
        hidden_states=hidden_states,
        grid_thw=grid_thw,
        patch_positions=patch_positions,  # <-- Critical: use interpolated positions
    )

print(f"Output shape: {outputs.last_hidden_state.shape}")  # torch.Size([1568, hidden_size])
print(f"Pooler output shape: {outputs.pooler_output.shape}")  # torch.Size([1, hidden_size])
```

### Understanding the Interpolation Formula

The `interpolate_frame_indices` function computes interpolated positions using:

```python
# Interpolation formula:
# new_idx = (old_idx / (total_frames - 1)) * (target_frames - 1)

# For 8 frames -> 64-frame context:
# Frame 0: (0 / 7) * 63 = 0
# Frame 1: (1 / 7) * 63 = 9
# Frame 2: (2 / 7) * 63 = 18
# Frame 3: (3 / 7) * 63 = 27
# Frame 4: (4 / 7) * 63 = 36
# Frame 5: (5 / 7) * 63 = 45
# Frame 6: (6 / 7) * 63 = 54
# Frame 7: (7 / 7) * 63 = 63
```

### Summary: When to Use Each Approach

| Use Case | Function | Example |
|----------|----------|---------|
| Images (single frame) | `compute_patch_positions_from_grid_thw` | Standard image processing |
| Videos (source model consistency) | `compute_patch_positions_with_interpolated_temporal` | When output must match source model |

---

## Mixed Batch: Image + 8-Frame Video Packing with `patch_positions`

This example demonstrates how to pack one image and one 8-frame video together in a single batch, with proper `patch_positions` for both. This is the recommended approach for mixed image-video processing with source model consistency.

```python
import torch
from PIL import Image
import torchvision.transforms as T
from model_factory.vit_preview_v0_packing_hf import (
    LlavaViTPackingModel,
    compute_patch_positions_from_grid_thw,
)
from model_factory.convert_llava_vit_packing_to_hf import (
    interpolate_frame_indices,
    compute_patch_positions_with_interpolated_temporal,
)

# ============================================================
# Configuration
# ============================================================
device = torch.device('cuda')
patch_size = 16
image_size = 224  # 224x224 image
num_frames = 8
target_frames = 64  # Source model's temporal context

h_patches = image_size // patch_size  # 14
w_patches = image_size // patch_size  # 14

# ============================================================
# Step 1: Prepare Image Input
# ============================================================
transform = T.Compose([
    T.Resize((image_size, image_size)),
    T.ToTensor(),
    T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]),
])

# Load or create image
image = Image.new('RGB', (image_size, image_size), color='green')
image_tensor = transform(image)  # (3, 224, 224)

# Convert image to patches
channels = image_tensor.shape[0]
image_patches = image_tensor.view(channels, h_patches, patch_size, w_patches, patch_size)
image_patches = image_patches.permute(1, 3, 0, 2, 4).contiguous()  # (h, w, C, pH, pW)
image_hidden_states = image_patches.view(h_patches * w_patches, patch_size * patch_size * channels)
# Shape: (196, 768)

# Compute patch_positions for image using compute_patch_positions_from_grid_thw
image_grid_thw = torch.tensor([[1, h_patches, w_patches]], dtype=torch.long, device=device)
image_patch_positions = compute_patch_positions_from_grid_thw(image_grid_thw)
# Shape: (196, 3), values: [[0, 0, 0], [0, 0, 1], ..., [0, 13, 13]]

# ============================================================
# Step 2: Prepare Video Input with Interpolated Temporal Positions
# ============================================================
# Create or load video frames
video_frames = [Image.new('RGB', (image_size, image_size), color=(i*30, i*20, i*10)) 
                for i in range(num_frames)]

# Convert video frames to patches
all_video_patches = []
for frame in video_frames:
    frame_tensor = transform(frame)  # (3, 224, 224)
    patches = frame_tensor.view(channels, h_patches, patch_size, w_patches, patch_size)
    patches = patches.permute(1, 3, 0, 2, 4).contiguous()
    frame_patches = patches.view(h_patches * w_patches, patch_size * patch_size * channels)
    all_video_patches.append(frame_patches)

video_hidden_states = torch.cat(all_video_patches, dim=0)
# Shape: (1568, 768)

# Compute interpolated frame indices for 64-frame context
# 8 frames → interpolated to positions [0, 9, 18, 27, 36, 45, 54, 63] in 64-frame context
frame_indices = torch.arange(num_frames).unsqueeze(0).to(device)  # [[0, 1, 2, 3, 4, 5, 6, 7]]
total_frames_tensor = torch.tensor([num_frames]).to(device)  # [8]
interpolated_indices = interpolate_frame_indices(frame_indices, total_frames_tensor, target_frames)
# Result: [[0, 9, 18, 27, 36, 45, 54, 63]]

# Compute patch_positions with interpolated temporal positions
video_patch_positions = compute_patch_positions_with_interpolated_temporal(
    interpolated_indices, h_patches, w_patches, device
)
# Shape: (1568, 3)

video_grid_thw = torch.tensor([[num_frames, h_patches, w_patches]], dtype=torch.long, device=device)

# ============================================================
# Step 3: Combine Image and Video into Packed Batch
# ============================================================
# Concatenate hidden_states
packed_hidden_states = torch.cat([
    image_hidden_states,  # (196, 768)
    video_hidden_states,  # (1568, 768)
], dim=0).to(device=device, dtype=torch.bfloat16)
# Total shape: (1764, 768)

# Concatenate grid_thw
packed_grid_thw = torch.cat([
    image_grid_thw,  # [[1, 14, 14]]
    video_grid_thw,  # [[8, 14, 14]]
], dim=0)
# Shape: (2, 3)

# Concatenate patch_positions
packed_patch_positions = torch.cat([
    image_patch_positions,  # (196, 3)
    video_patch_positions,  # (1568, 3)
], dim=0)
# Shape: (1764, 3)

print(f"packed_hidden_states shape: {packed_hidden_states.shape}")  # (1764, 768)
print(f"packed_grid_thw: {packed_grid_thw.tolist()}")  # [[1, 14, 14], [8, 14, 14]]
print(f"packed_patch_positions shape: {packed_patch_positions.shape}")  # (1764, 3)

# ============================================================
# Step 4: Forward Pass with Packing Model
# ============================================================
model = LlavaViTPackingModel.from_pretrained("/path/to/your_model_hf_packing", torch_dtype=torch.bfloat16)
model = model.to(device).eval()

with torch.no_grad():
    outputs = model(
        hidden_states=packed_hidden_states,
        grid_thw=packed_grid_thw,
        patch_positions=packed_patch_positions,
    )

print(f"Output shape: {outputs.last_hidden_state.shape}")  # (1764, hidden_size)
print(f"Pooler output shape: {outputs.pooler_output.shape}")  # (2, hidden_size)
# pooler_output[0]: image pooled representation
# pooler_output[1]: video pooled representation
```

---

## Verifying Consistency with Source Model (Separate Inference)

This example demonstrates how to verify that the packing model produces consistent outputs with the source model (`vit_preview_v0`) by running separate inference for image and video inputs.

```python
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
import timm
from model_factory.vit_preview_v0_packing_hf import (
    LlavaViTPackingModel,
    compute_patch_positions_from_grid_thw,
)
from model_factory.convert_llava_vit_packing_to_hf import (
    interpolate_frame_indices,
    compute_patch_positions_with_interpolated_temporal,
)

# ============================================================
# Load Both Models
# ============================================================
device = torch.device('cuda')

# Load source model (vit_preview_v0)
src_model = timm.create_model("llava_vit_large_ln", pretrained=False)
src_checkpoint = torch.load("/path/to/backbone.pt", map_location='cpu')
src_state_dict = src_checkpoint.get("model", src_checkpoint.get("state_dict", src_checkpoint))
src_model.load_state_dict(src_state_dict, strict=False)
src_model = src_model.to(device, dtype=torch.bfloat16).eval()

# Load packing model
packing_model = LlavaViTPackingModel.from_pretrained(
    "/path/to/your_model_hf_packing", 
    torch_dtype=torch.bfloat16
)
packing_model = packing_model.to(device).eval()

# ============================================================
# Configuration
# ============================================================
patch_size = 16
image_size = 224
num_frames = 8
target_frames = 64
h_patches = image_size // patch_size  # 14
w_patches = image_size // patch_size  # 14

transform = T.Compose([
    T.Resize((image_size, image_size)),
    T.ToTensor(),
    T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]),
])

# ============================================================
# Test 1: Image Consistency
# ============================================================
print("=" * 60)
print("Test 1: Image Consistency")
print("=" * 60)

# Prepare image
image = Image.new('RGB', (image_size, image_size), color='red')
image_tensor = transform(image).unsqueeze(0).to(device, dtype=torch.bfloat16)  # (1, 3, 224, 224)

with torch.no_grad():
    # Source model forward (image input: B, C, H, W)
    src_out = src_model(image_tensor)
    src_image_feat = src_out['visible_embeddings']  # (1, 196, hidden_size)
    
    # Packing model forward
    channels = 3
    patches = image_tensor.view(1, channels, h_patches, patch_size, w_patches, patch_size)
    patches = patches.permute(0, 2, 4, 1, 3, 5).contiguous()
    hidden_states = patches.view(h_patches * w_patches, patch_size * patch_size * channels)
    
    grid_thw = torch.tensor([[1, h_patches, w_patches]], dtype=torch.long, device=device)
    patch_positions = compute_patch_positions_from_grid_thw(grid_thw)
    
    packing_out = packing_model(
        hidden_states=hidden_states,
        grid_thw=grid_thw,
        patch_positions=patch_positions,
    )
    packing_image_feat = packing_out.last_hidden_state  # (196, hidden_size)

# Compare outputs
src_flat = src_image_feat.flatten(0, -2).float()
packing_flat = packing_image_feat.float()

cos_sim = F.cosine_similarity(src_flat, packing_flat, dim=-1)
print(f"Image Min Cosine Similarity: {cos_sim.min().item():.8f}")
print(f"Image Mean Cosine Similarity: {cos_sim.mean().item():.8f}")
if cos_sim.min().item() > 0.99:
    print("✅ Image Consistency: PASS")
else:
    print("❌ Image Consistency: FAIL")

# ============================================================
# Test 2: Video Consistency (8 frames)
# ============================================================
print("\n" + "=" * 60)
print("Test 2: Video Consistency (8 frames)")
print("=" * 60)

# Create video frames
video_frames = [Image.new('RGB', (image_size, image_size), color=(i*30, i*20, i*10)) 
                for i in range(num_frames)]
video_tensors = torch.stack([transform(f) for f in video_frames], dim=0)  # (8, 3, 224, 224)
video_tensor = video_tensors.unsqueeze(0).permute(0, 2, 1, 3, 4).to(device, dtype=torch.bfloat16)
# Shape: (1, 3, 8, 224, 224) -> (B, C, T, H, W)

# Compute interpolated frame indices
# 8 frames → interpolated to positions [0, 9, 18, 27, 36, 45, 54, 63] in 64-frame context
frame_indices = torch.arange(num_frames).unsqueeze(0).to(device)
total_frames_tensor = torch.tensor([num_frames]).to(device)
interpolated_indices = interpolate_frame_indices(frame_indices, total_frames_tensor, target_frames)
print(f"Interpolated indices: {interpolated_indices[0].tolist()}")

with torch.no_grad():
    # Source model forward (requires 64-frame padded video with visible_index)
    bs = 1
    channels = 3
    frame_tokens = h_patches * w_patches
    
    # Create 64-frame padded video
    padded_video = torch.zeros(bs, channels, target_frames, image_size, image_size,
                               device=device, dtype=video_tensor.dtype)
    
    # Scatter original frames into interpolated positions
    frame_idx_expanded = interpolated_indices.view(bs, 1, num_frames, 1, 1).expand(
        bs, channels, num_frames, image_size, image_size
    )
    padded_video.scatter_(dim=2, index=frame_idx_expanded, src=video_tensor)
    
    # Compute visible_index
    per = torch.arange(frame_tokens, device=device)
    visible_index = (interpolated_indices.unsqueeze(-1) * frame_tokens + per).reshape(bs, -1)
    visible_index = visible_index.clamp_max(target_frames * frame_tokens - 1)
    
    src_video_out = src_model(padded_video, visible_indices=visible_index, mask_ratio=None)
    src_video_feat = src_video_out['visible_embeddings']  # (1, 1568, hidden_size)
    
    # Packing model forward
    patches = video_tensor.view(bs, channels, num_frames, h_patches, patch_size, w_patches, patch_size)
    patches = patches.permute(0, 2, 3, 5, 1, 4, 6).contiguous()
    hidden_states = patches.view(num_frames * h_patches * w_patches, patch_size * patch_size * channels)
    
    grid_thw = torch.tensor([[num_frames, h_patches, w_patches]], dtype=torch.long, device=device)
    patch_positions = compute_patch_positions_with_interpolated_temporal(
        interpolated_indices, h_patches, w_patches, device
    )
    
    packing_video_out = packing_model(
        hidden_states=hidden_states,
        grid_thw=grid_thw,
        patch_positions=patch_positions,
    )
    packing_video_feat = packing_video_out.last_hidden_state  # (1568, hidden_size)

# Compare outputs
src_video_flat = src_video_feat.flatten(0, -2).float()
packing_video_flat = packing_video_feat.float()

cos_sim_video = F.cosine_similarity(src_video_flat, packing_video_flat, dim=-1)
print(f"Video Min Cosine Similarity: {cos_sim_video.min().item():.8f}")
print(f"Video Mean Cosine Similarity: {cos_sim_video.mean().item():.8f}")
if cos_sim_video.min().item() > 0.99:
    print("✅ Video Consistency: PASS")
else:
    print("❌ Video Consistency: FAIL")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
all_pass = cos_sim.min().item() > 0.99 and cos_sim_video.min().item() > 0.99
if all_pass:
    print("✅ All consistency tests PASSED")
else:
    print("❌ Some consistency tests FAILED")
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
